import json
import multiprocessing
from copy import deepcopy
import polars as pl
from ..formulaTools import formulaTools
from ..isotopeScoring import compute_isotope_pattern, isotope_ratio_columns, isotope_ratio_json_fields
from ..SGR import SGRGenerator
from ..utils import Bunch
from ..ParquetCache import ParquetCache
from ..utils import get_app_folder
import re

# sys.path.append("C:/PyMetExtract/PyMetExtract")  # Removed hardcoded path

exID = "Num"
exMZ = "MZ"
exRT = "RT"
exAccMass = "M"
exXCount = "Xn"
exIonMode = "Ionisation_Mode"
exCharge = "Charge"


local_folder = get_app_folder()
sfCache = ParquetCache(local_folder + "/sfcache.pqts")


def getCallStr(m, ppm, useAtoms, atomsRange, fixed, useSGR):
    return "sfg.findFormulas(%.5f, useAtoms=%s, atomsRange=%s, fixed='%s', useSevenGoldenRules=%s, useSecondRule=True, ppm=%.2f))" % (m, str(useAtoms), str(atomsRange), fixed, useSGR, ppm)


def genSFs(m, ppm, useAtoms, atomsRange, fixed, useSGR):
    sfg = SGRGenerator()
    sfs = sfg.findFormulas(
        m,
        useAtoms=useAtoms,
        atomsRange=atomsRange,
        useSevenGoldenRules=useSGR,
        useSecondRule=True,
        ppm=ppm,
    )
    return sfs


def _compute_formula(args):
    """
    Module-level pool worker: compute sum formulas and return the list directly.
    Must be at module level so it can be pickled by multiprocessing on Windows (spawn).
    Does not access the cache or any shared state.
    """
    return genSFs(args.m, args.ppm, args.useAtoms, args.atomsRange, args.fixed, args.useSGR)


_counter = None


def calcSumFormulas(args):
    global _counter
    m = args.m
    ppm = args.ppm
    useAtoms = args.useAtoms
    atomsRange = args.atomsRange
    fixed = args.fixed
    useSGR = args.useSGR
    lock = args.lock
    _counter = args.counter

    callStr = getCallStr(m, ppm, useAtoms, atomsRange, fixed, useSGR)

    if lock is not None:
        lock.acquire()
    sfs = sfCache.get(callStr)
    if lock is not None:
        lock.release()

    if sfs is None:
        sfs = genSFs(m, ppm, useAtoms, atomsRange, fixed, useSGR)

        if lock is not None:
            lock.acquire()
        sfCache.set(callStr, sfs)
        # print("\rNew SF calculated          ", counter.value, "done..", callStr,)
        _counter.value = _counter.value + 1
        if lock is not None:
            lock.release()
    else:
        if lock is not None:
            lock.acquire()
        # print("\rFetched result from cache  ", counter.value, "done..", callStr,)
        _counter.value = _counter.value + 1
        if lock is not None:
            lock.release()

    return sfs


def processPolarsTable(
    df,
    columns,
    adducts,
    ppm=5.0,
    ppmCorrectionPosMode=0,
    ppmCorrectionNegMode=0,
    useAtoms=[],
    atomsRange=[],
    smCol="sumFormula_",
    useSevenGoldenRules=True,
    useCn=True,
    pwMaxSet=None,
    pwValSet=None,
    nCores=1,
):
    """
    Process a Polars dataframe and add sum formula annotations.

    Args:
        df: Polars DataFrame with metabolite data
        columns: Dict mapping expected column names to actual column names
        adducts: List of adduct definitions
        ppm: Mass accuracy in ppm
        ppmCorrectionPosMode: PPM correction for positive mode
        ppmCorrectionNegMode: PPM correction for negative mode
        useAtoms: List of atoms to use
        atomsRange: List of [min, max] ranges for each atom
        smCol: Prefix for sum formula columns
        useSevenGoldenRules: Whether to use seven golden rules
        useCn: How to use carbon count constraint
        pwMaxSet: Progress callback for max value
        pwValSet: Progress callback for current value
        nCores: Number of CPU cores to use

    Returns:
        Polars DataFrame with sum formula columns added
    """
    if len(useAtoms) == 0:
        useAtoms = ["C", "H", "O", "N", "P"]
        atomsRange = [[-1, -1]]  # C
        atomsRange.append([0, 130])  # H
        atomsRange.append([0, 40])  # O
        atomsRange.append([0, 10])  # N
        atomsRange.append([0, 10])  # P

    # Add sum formula columns if they don't exist
    existing_cols = df.columns
    new_cols = {}

    for col_name in [
        smCol + "_CHO_count",
        smCol + "_CHON_count",
        smCol + "_CHOP_count",
        smCol + "_CHOS_count",
        smCol + "_CHONP_count",
        smCol + "_CHONS_count",
        smCol + "_CHOPS_count",
        smCol + "_CHONPS_count",
        smCol + "_all_count",
    ]:
        if col_name not in existing_cols:
            new_cols[col_name] = pl.lit(None, dtype=pl.Int64)

    for col_name in [
        smCol + "_CHO",
        smCol + "_CHON",
        smCol + "_CHOP",
        smCol + "_CHOS",
        smCol + "_CHONP",
        smCol + "_CHONS",
        smCol + "_CHOPS",
        smCol + "_CHONPS",
        smCol + "_all",
    ]:
        if col_name not in existing_cols:
            new_cols[col_name] = pl.lit("", dtype=pl.Utf8)

    if new_cols:
        df = df.with_columns(**new_cols)

    fT = formulaTools()
    useCn_lower = useCn.lower()
    rows_list = df.to_dicts()

    # --- Step 1: collect unique formula jobs keyed by callStr ---
    # row_jobs[i] = list of (callStr, ion_label, neutral_mass_m) for row i
    row_jobs = []
    unique_jobs = {}  # callStr -> Bunch of compute args (no lock, no counter needed)

    for row_dict in rows_list:
        ionMode = row_dict[columns[exIonMode]]
        mz = float(row_dict[columns[exMZ]])
        ppm_corr = ppmCorrectionPosMode if ionMode == "+" else ppmCorrectionNegMode
        mz = mz + mz * ppm_corr / 1e6

        accMass = row_dict[columns[exAccMass]] if columns[exAccMass] in row_dict else ""
        if accMass is None:
            accMass = ""

        xCount = int(row_dict[columns[exXCount]])
        charge = row_dict[columns[exCharge]]

        row_atomsRange = deepcopy(atomsRange)
        if useCn_lower == "exact":
            row_atomsRange[0] = xCount
        elif useCn_lower == "don't use":
            pass
        elif useCn_lower in ("min", "minimum"):
            row_atomsRange[0] = (xCount, row_atomsRange[0][1])
        elif useCn_lower.startswith("plusminus"):
            g = 2
            if len("plusminus_") < len(useCn):
                g = int(useCn[len("plusminus_") :])
            row_atomsRange[0] = (xCount - g, xCount + g)

        this_row_jobs = []
        if accMass == "":
            for adduct in adducts:
                if ionMode == adduct[2] and charge == adduct[3]:
                    m = (mz - adduct[1]) * adduct[3] / adduct[4]
                    callStr = getCallStr(m, ppm, useAtoms, row_atomsRange, "C", useSevenGoldenRules)
                    this_row_jobs.append((callStr, "[M" + adduct[0] + "]", m))
                    if callStr not in unique_jobs:
                        unique_jobs[callStr] = Bunch(
                            m=m,
                            ppm=ppm,
                            useAtoms=useAtoms,
                            atomsRange=deepcopy(row_atomsRange),
                            fixed="C",
                            useSGR=useSevenGoldenRules,
                        )
        else:
            for m_str in str(accMass).split(","):
                m_val = float(m_str.replace('"', "").strip())
                m_val = m_val + m_val * ppm_corr / 1e6
                callStr = getCallStr(m_val, ppm, useAtoms, row_atomsRange, "C", useSevenGoldenRules)
                this_row_jobs.append((callStr, "[M]", m_val))
                if callStr not in unique_jobs:
                    unique_jobs[callStr] = Bunch(
                        m=m_val,
                        ppm=ppm,
                        useAtoms=useAtoms,
                        atomsRange=deepcopy(row_atomsRange),
                        fixed="C",
                        useSGR=useSevenGoldenRules,
                    )
        row_jobs.append(this_row_jobs)

    # --- Step 2: bulk-load the cache once and split into hits/misses ---
    # sfCache.get_all() reads the parquet file a single time and returns a plain
    # dict, avoiding the O(n) DataFrame scan that sfCache.get() does per key.
    all_cached = sfCache.get_all()
    formula_dict = {}  # callStr -> list[str]
    missing_strs = []
    missing_jobs = []
    for callStr, job in unique_jobs.items():
        if callStr in all_cached:
            formula_dict[callStr] = all_cached[callStr]
        else:
            missing_strs.append(callStr)
            missing_jobs.append(job)

    # --- Step 3: compute missing formulas (parallel when nCores > 1) ---
    # _compute_formula is a module-level function so it can be pickled on Windows.
    # Only unique masses are computed once; no lock or Manager needed.
    if missing_jobs:
        if pwMaxSet:
            pwMaxSet(len(missing_jobs))
        if nCores > 1 and len(missing_jobs) > 1:
            chunksize = max(1, len(missing_jobs) // (nCores * 4))
            with multiprocessing.Pool(processes=nCores) as pool:
                computed = list(pool.imap(_compute_formula, missing_jobs, chunksize=chunksize))
        else:
            computed = []
            for i, job in enumerate(missing_jobs):
                if pwValSet and i % 10 == 0:
                    pwValSet(i)
                computed.append(genSFs(job.m, job.ppm, job.useAtoms, job.atomsRange, job.fixed, job.useSGR))

        new_entries = {}
        for callStr, sfs in zip(missing_strs, computed):
            formula_dict[callStr] = sfs
            new_entries[callStr] = sfs
        # Write all new results to the parquet cache in a single file write.
        if new_entries:
            sfCache.set_many(new_entries)

    # --- Step 4: classify formulas per row and collect column values ---
    if pwMaxSet:
        pwMaxSet(len(rows_list))

    suffix_keys = ["_CHO", "_CHOS", "_CHOP", "_CHON", "_CHONP", "_CHONS", "_CHOPS", "_CHONPS", "_all"]
    col_data = {smCol + k: [] for k in suffix_keys}
    col_data.update({smCol + k + "_count": [] for k in suffix_keys})
    compound_hits = []

    for idx, (row_dict, this_row_jobs) in enumerate(zip(rows_list, row_jobs)):
        if pwValSet and idx % 10 == 0:
            pwValSet(idx)

        dbe = {smCol + k: [] for k in suffix_keys}
        other_isotopologs = row_dict.get("Other_Isotopologs")
        for callStr, ion_label, m in this_row_jobs:
            for e in formula_dict.get(callStr, []):
                mT = fT.calcMolWeight(fT.parseFormula(e))
                isotope_abs_error, isotope_ratios = compute_isotope_pattern(e, other_isotopologs)
                ent = {
                    "formula": e,
                    "ion": ion_label,
                    "mass_error_ppm": (m - mT) * 1e6 / m,
                    "mass_error_mass": m - mT,
                    "isotope_abs_error": isotope_abs_error,
                    **isotope_ratio_json_fields(isotope_ratios),
                }

                has_n = "N" in ent["formula"]
                has_p = "P" in ent["formula"]
                has_s = "S" in ent["formula"]
                if has_p and has_n and has_s:
                    dbe[smCol + "_CHONPS"].append(ent)
                    element_class = "CHONPS"
                elif has_p and has_s:
                    dbe[smCol + "_CHOPS"].append(ent)
                    element_class = "CHOPS"
                elif has_s and has_n:
                    dbe[smCol + "_CHONS"].append(ent)
                    element_class = "CHONS"
                elif has_n and has_p:
                    dbe[smCol + "_CHONP"].append(ent)
                    element_class = "CHONP"
                elif has_p:
                    dbe[smCol + "_CHOP"].append(ent)
                    element_class = "CHOP"
                elif has_n:
                    dbe[smCol + "_CHON"].append(ent)
                    element_class = "CHON"
                elif has_s:
                    dbe[smCol + "_CHOS"].append(ent)
                    element_class = "CHOS"
                else:
                    dbe[smCol + "_CHO"].append(ent)
                    element_class = "CHO"
                dbe[smCol + "_all"].append(ent)
                compound_hits.append(
                    {
                        "SumFormula": e,
                        "Element_Class": element_class,
                        "Ion_Adduct": ion_label,
                        "MassErrorPPM": ent["mass_error_ppm"],
                        "MassErrorMass": ent["mass_error_mass"],
                        "Iso_AE": isotope_abs_error,
                        **isotope_ratio_columns(isotope_ratios),
                        "Feature_Num": row_dict.get(columns[exID]),
                        "Feature_RT": row_dict.get(columns[exRT]),
                        "Feature_MZ": row_dict.get(columns[exMZ]),
                        "Feature_Xn": row_dict.get(columns[exXCount]),
                        "Feature_Ionisation_Mode": row_dict.get(columns[exIonMode]),
                        "Feature_Charge": row_dict.get(columns[exCharge]),
                        "Feature_M": row_dict.get(columns[exAccMass]),
                        "Feature_OGroup": row_dict.get("OGroup"),
                        "Feature_Ion": row_dict.get("Ion"),
                        "Feature_Loss": row_dict.get("Loss"),
                        "Feature_Relative_peakarea_in_group": row_dict.get("Relative_peakarea_in_group"),
                        "Feature_Average_peakarea": row_dict.get("Average_peakarea"),
                    }
                )

        for k in suffix_keys:
            hits = dbe[smCol + k]
            if hits:
                col_data[smCol + k].append("Too many hits %d" % len(hits) if len(hits) > 250 else json.dumps(hits, ensure_ascii=True))
                col_data[smCol + k + "_count"].append(len(hits))
            else:
                col_data[smCol + k].append("")
                col_data[smCol + k + "_count"].append(None)

    # Update DataFrame in a single batched with_columns call instead of 18 separate ones.
    df = df.with_columns([pl.Series(col_name, values) for col_name, values in col_data.items()])
    return df, compound_hits


## adduct definition (name, m/z increment, polarity, charges, number of M
adductsP = [
    ("+H", 1.007276, "+", 1, 1),
    ("+NH4", 18.033823, "+", 1, 1),
    ("+Na", 22.989218, "+", 1, 1),
    ("+K", 38.963158, "+", 1, 1),
]
adductsN = [
    ("-H", -1.007276, "-", 1, 1),
    ("+Na-2H", 20.974666, "-", 1, 1),
    ("+Cl", 34.969402, "-", 1, 1),
    ("+FA-H", 44.998201, "-", 1, 1),
    ("+Hac-H", 59.013851, "-", 1, 1),
    ("+Br", 78.918885, "-", 1, 1),
]


# taken from http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
# use for a=["1", "2", "10", "11", "3"]
# natSort(a)
# for a=[("1", 1), ("2", 2), ("10", 3), ("11", 4), ("3", 5)]
# natSort(a, key=itemgetter(0))
def natSort(l, key=lambda ent: ent):
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(ent, key=key):
        return [convert(c) for c in re.split("([0-9]+)", str(key(ent)))]

    l.sort(key=alphanum_key)
    return l


def annotatePolarsTableWithSumFormulas(
    results_df,
    useAtoms,
    atomsRange,
    Xn,
    useExactXn,
    ppm=5.0,
    ppmCorrectionPosMode=0,
    ppmCorrectionNegMode=0,
    adducts=adductsN + adductsP,
    pwMaxSet=None,
    pwValSet=None,
    nCores=1,
):
    """
    Annotate a Polars dataframe with generated sum formulas.

    Args:
        results_df: Polars DataFrame with metabolite data
        useAtoms: List of atoms to use
        atomsRange: List of [min, max] ranges for each atom
        Xn: Element to check (e.g., "C")
        useExactXn: How to check Xn
        ppm: Mass accuracy in ppm
        ppmCorrectionPosMode: PPM correction for positive mode
        ppmCorrectionNegMode: PPM correction for negative mode
        adducts: List of adduct definitions
        pwMaxSet: Progress callback for max value
        pwValSet: Progress callback for current value
        nCores: Number of CPU cores to use

    Returns:
        Polars DataFrame with sum formula columns added
    """
    return processPolarsTable(
        results_df,
        columns={
            exID: exID,
            exMZ: exMZ,
            exRT: exRT,
            exAccMass: exAccMass,
            exXCount: exXCount,
            exIonMode: exIonMode,
            exCharge: exCharge,
        },
        adducts=adducts,
        ppm=ppm,
        ppmCorrectionPosMode=ppmCorrectionPosMode,
        ppmCorrectionNegMode=ppmCorrectionNegMode,
        smCol="SFs",
        useAtoms=deepcopy(useAtoms),
        atomsRange=deepcopy(atomsRange),
        useCn=useExactXn,
        useSevenGoldenRules=True,
        pwMaxSet=pwMaxSet,
        pwValSet=pwValSet,
        nCores=nCores,
    )


# Legacy function for backward compatibility (TSV file-based)
def annotateResultsWithSumFormulas(
    resultsFile,
    useAtoms,
    atomsRange,
    Xn,
    useExactXn,
    ppm=5.0,
    ppmCorrectionPosMode=0,
    ppmCorrectionNegMode=0,
    adducts=adductsN + adductsP,
    pwMaxSet=None,
    pwValSet=None,
    nCores=1,
    toFile=None,
):
    """
    Legacy function that reads from TSV file and writes results.
    For new code, use annotatePolarsTableWithSumFormulas instead.
    """
    if toFile is None:
        toFile = resultsFile

    # Read TSV file
    df = pl.read_csv(resultsFile, separator="\t")

    # Process with Polars
    df, _ = annotatePolarsTableWithSumFormulas(
        df,
        useAtoms=useAtoms,
        atomsRange=atomsRange,
        Xn=Xn,
        useExactXn=useExactXn,
        ppm=ppm,
        ppmCorrectionPosMode=ppmCorrectionPosMode,
        ppmCorrectionNegMode=ppmCorrectionNegMode,
        adducts=adducts,
        pwMaxSet=pwMaxSet,
        pwValSet=pwValSet,
        nCores=nCores,
    )

    # Write TSV file
    df.write_csv(toFile, separator="\t")
