import os
import sys
import csv
from copy import deepcopy
from ..formulaTools import formulaTools
import logging
from .. import LoggingSetup
import polars as pl

sys.path.append("C:/development/PyMetExtract")


exID = "Num"
exMZ = "MZ"
exRT = "RT"
exAccMass = "M"
exXCount = "Xn"
exIonMode = "Ionisation_Mode"
exCharge = "Charge"


LoggingSetup.LoggingSetup.Instance().initLogging()


def smilesToSumFormula(smiles):
    """Derive a neutral-element sum formula string from a SMILES code using RDKit.

    Returns the sum formula string (e.g. 'C6H12O6') or None if the SMILES could
    not be parsed.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception:
        logging.error("DB import error: RDKit is required to derive sum formulas from SMILES but could not be imported")
        return None

    if smiles is None or smiles == "":
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return rdMolDescriptors.CalcMolFormula(mol)


def formulasEqual(formulaA, formulaB, fT):
    """Compare two sum formulas by their elemental composition (ignoring formatting and charge).

    Returns True if both formulas describe the same set of elements with the same counts.
    """
    try:
        elemsA = fT.parseFormula(formulaA)
        elemsB = fT.parseFormula(formulaB)
    except Exception:
        return False

    elemsA = {k: v for k, v in elemsA.items() if v != 0}
    elemsB = {k: v for k, v in elemsB.items() if v != 0}

    return elemsA == elemsB


class DBEntry:
    def __init__(
        self,
        dbName,
        num,
        name,
        sumFormula,
        mass,
        rt_min,
        mz,
        polarity,
        additionalInfo,
        hitType="",
    ):
        self.dbName = dbName
        self.num = num
        self.name = name
        self.sumFormula = sumFormula
        self.mass = mass
        self.matchErrorPPM = -1
        self.matchErrorMass = -1
        self.rt_min = rt_min
        self.mz = mz
        self.polarity = polarity
        self.additionalInfo = additionalInfo

        self.hitType = hitType

    def __str__(self):
        return (
            "DB: "
            + str(self.dbName)
            + " Num: "
            + str(self.num)
            + " Name: "
            + str(self.name)
            + " SumFormula: "
            + str(self.sumFormula)
            + " Mass: "
            + str(self.mass)
            + " matchErrorPPM: "
            + str(self.matchErrorPPM)
            + " matchErrorMass: "
            + str(self.matchErrorMass)
            + " rtMin: "
            + str(self.rt_min)
            + " mz: "
            + str(self.mz)
            + " polarity: "
            + str(self.polarity)
            + " additionalInfo: "
            + str(self.additionalInfo)
        )


class DBSearch:
    def __init__(self):
        self.dbEntriesNeutral = []
        self.dbEntriesMZ = []

    def addEntriesFromFile(self, dbName, dbFile, callBackCheckFunction=None, error_collector=None, mismatch_collector=None):
        imported = 0
        notImported = 0
        mismatches = 0

        curEntriesCount = len(self.dbEntriesMZ) + len(self.dbEntriesNeutral)

        fT = formulaTools()

        # check if file exists

        if not os.path.exists(dbFile):
            logging.error("DB import error: File not found %s" % (dbFile))
            raise Exception("DB import error: File not found %s" % (dbFile))

        if dbFile.lower().endswith(".xlsx"):
            df = None
            try:
                df = pl.read_excel(dbFile, sheet_name="Template")
            except Exception:
                try:
                    df = pl.read_excel(dbFile, sheet_name="Sheet 1")
                except Exception:
                    try:
                        df = pl.read_excel(dbFile)
                    except Exception:
                        logging.error("DB import error: Could not read Excel file %s; tried sheets 'Template', 'Sheet 1' and default sheet" % (dbFile))
                        raise Exception("DB import error: Could not read Excel file %s; tried sheets 'Template', 'Sheet 1' and default sheet" % (dbFile))

            # Build a list-of-lists interface compatible with the TSV path
            header_row = list(df.columns)
            data_rows = [[str(v) if v is not None else "" for v in row] for row in df.iter_rows()]
            all_rows = [header_row] + data_rows
            row_source = all_rows
        else:
            row_source = None  # will use file-based iteration below

        def _iter_rows():
            if row_source is not None:
                yield from row_source
            else:
                with open(dbFile) as fIn:
                    yield from csv.reader(fIn, delimiter="\t")

        headers = {}
        for rowi, row in enumerate(_iter_rows()):
            if len(row) == 0:  # Skip empty lines
                continue
            if row[0].startswith("#"):
                continue
            if rowi == 0:
                for j in range(len(row)):
                    headers[row[j]] = j

            else:
                try:
                    num = row[headers["Num"]].strip().replace('"', "DOURBLEPRIME").replace("'", "PRIME").replace("\t", "TAB").replace("\n", "RETURN").replace("\r", "CarrRETURN").replace("#", "HASH")
                    name = row[headers["Name"]].strip().replace('"', "DOURBLEPRIME").replace("'", "PRIME").replace("\t", "TAB").replace("\n", "RETURN").replace("\r", "CarrRETURN").replace("#", "HASH")
                    sumFormula = row[headers["SumFormula"]].strip().replace('"', "DOURBLEPRIME").replace("'", "PRIME").replace("\t", "TAB").replace("\n", "RETURN").replace("\r", "CarrRETURN").replace("#", "HASH")
                    rt_min = None
                    try:
                        rt_min = float(row[headers["Rt_min"]]) if row[headers["Rt_min"]] != "" else None
                    except Exception:
                        _msg = "   - Error row %d: The Rt_min value '%s' of the entry %s '%s' could not be parsed as a float, not using RT for this compound" % (rowi, row[headers["Rt_min"]], num, name)
                        logging.error(_msg)
                        if error_collector is not None:
                            error_collector.append(_msg)
                    mz = float(row[headers["MZ"]]) if row[headers["MZ"]] != "" else None
                    polarity = row[headers["IonisationMode"]].strip().replace('"', "DOURBLEPRIME").replace("'", "PRIME").replace("\t", "TAB").replace("\n", "RETURN").replace("\r", "CarrRETURN").replace("#", "HASH")
                    additionalInfo = {}
                    for header in headers.keys():
                        if header not in [
                            "Num",
                            "Name",
                            "SumFormula",
                            "Rt_min",
                            "MZ",
                            "IonisationMode",
                        ]:
                            additionalInfo[header] = row[headers[header]].replace('"', "DOURBLEPRIME").replace("'", "PRIME").replace("\t", "TAB").replace("\n", "RETURN").replace("\r", "CarrRETURN").replace("#", "HASH")

                    # SMILES handling: derive the sum formula from the SMILES code (if provided)
                    smiles = ""
                    if "SMILES" in headers:
                        smiles = row[headers["SMILES"]].strip()
                    if smiles != "":
                        smiles_formula = smilesToSumFormula(smiles)
                        if smiles_formula is None:
                            _msg = "   - Error row %d: The SMILES code (%s) of the entry %s '%s' could not be parsed, ignoring SMILES for this compound" % (rowi, smiles, num, name)
                            logging.error(_msg)
                            if error_collector is not None:
                                error_collector.append(_msg)
                        elif sumFormula == "":
                            # Only a SMILES code is provided: derive and use the sum formula for annotation
                            sumFormula = smiles_formula
                        elif not formulasEqual(sumFormula, smiles_formula, fT):
                            # Provided sum formula and SMILES-derived sum formula disagree: report and skip
                            _msg = "   - SMILES/SumFormula mismatch in row %d: entry %s '%s' has SumFormula '%s' but SMILES '%s' yields '%s'; this entry is not used" % (
                                rowi,
                                num,
                                name,
                                sumFormula,
                                smiles,
                                smiles_formula,
                            )
                            logging.error(_msg)
                            if error_collector is not None:
                                error_collector.append(_msg)
                            if mismatch_collector is not None:
                                mismatch_collector.append(_msg)
                            mismatches += 1
                            continue

                    mass = 0
                    formula_charge = 0
                    entry_mz = mz
                    entry_polarity = polarity
                    is_charged_formula = False
                    if sumFormula != "":
                        try:
                            elems, formula_charge = fT.parseFormulaWithCharge(sumFormula)
                            mass = fT.calcMolWeight(elems)
                            if formula_charge != 0:
                                # Charged formula: compute ionic m/z directly from the formula
                                ionic_mass = fT.calcMolWeight(elems, charge=formula_charge)
                                entry_mz = ionic_mass / abs(formula_charge)
                                entry_polarity = "+" if formula_charge > 0 else "-"
                                is_charged_formula = True
                        except Exception:
                            _msg = "   - Error row %d: The sumformula (%s) of the entry %s '%s' could not be parsed" % (rowi, sumFormula, num, name)
                            logging.error(_msg)
                            if error_collector is not None:
                                error_collector.append(_msg)
                            notImported += 1

                    dbEntry = DBEntry(
                        dbName,
                        num,
                        name,
                        sumFormula,
                        mass,
                        rt_min,
                        entry_mz,
                        entry_polarity,
                        additionalInfo,
                    )

                    use = True
                    if callBackCheckFunction is not None:
                        use = callBackCheckFunction(dbEntry)

                    if use:
                        if is_charged_formula:
                            # Charged formula entries are stored by their ionic m/z for direct matching
                            self.dbEntriesMZ.append(dbEntry)
                        elif mass > 0:
                            self.dbEntriesNeutral.append(dbEntry)
                        else:
                            self.dbEntriesMZ.append(dbEntry)
                        imported += 1

                except Exception as ex:
                    _msg = "   - Error row %d: %s" % (dbName, rowi, ex)
                    logging.error(_msg)
                    if error_collector is not None:
                        error_collector.append(_msg)
                    notImported += 1

        if notImported > 0:
            _summary_msg = "Warning: Not imported %d entries (see above errors)" % (notImported)
            logging.error(_summary_msg)
            if error_collector is not None:
                error_collector.append(_summary_msg)

        if mismatches > 0:
            _summary_msg = "Warning: Skipped %d entries due to SMILES/SumFormula mismatches (see above)" % (mismatches)
            logging.error(_summary_msg)
            if error_collector is not None:
                error_collector.append(_summary_msg)

        _summary_msg = "   - Imported DB %s with %d entries" % (
            dbName,
            len(self.dbEntriesMZ) + len(self.dbEntriesNeutral) - curEntriesCount,
        )
        logging.info(_summary_msg)
        if error_collector is not None:
            error_collector.append(_summary_msg)

        return imported, notImported

    def optimizeDB(self):
        self.dbEntriesNeutral = sorted([e for e in self.dbEntriesNeutral if e.mass is not None], key=lambda x: x.mass)
        self.dbEntriesMZ = sorted([e for e in self.dbEntriesMZ if e.mz is not None], key=lambda x: x.mz)

    def _findGeneric(self, list, getValue, valueLeft, valueRight):
        if len(list) == 0:
            return []

        # implement binary search to find a value in the sorted list between valueLeft and valueRight
        left = 0
        right = len(list) - 1

        while left <= right:
            middle = (left + right) // 2
            middleValue = getValue(list[middle])

            if middleValue < valueLeft:
                left = middle + 1
            elif middleValue > valueRight:
                right = middle - 1
            else:
                # find the leftmost and rightmost index with values between valueLeft and valueRight
                leftIndex = middle
                while leftIndex >= 0 and valueLeft <= getValue(list[leftIndex]) <= valueRight:
                    leftIndex -= 1
                leftIndex += 1

                rightIndex = middle
                while rightIndex < len(list) and valueLeft <= getValue(list[rightIndex]) <= valueRight:
                    rightIndex += 1
                rightIndex -= 1

                return list[leftIndex : rightIndex + 1]

        return []

    def searchDBForMZ(
        self,
        mz,
        polarity,
        charges,
        ppm,
        rt_min=None,
        rt_error=0.1,
        checkXN="Exact",
        element="C",
        Xn=1,
        adducts=[
            ("+H", 1.007276, "+", 1, 1),
            ("+NH4", 18.033823, "+", 1, 1),
            ("+Na", 22.989218, "+", 1, 1),
            ("+K", 38.963158, "+", 1, 1),
            ("+CH3OH+H", 33.033489, "+", 1, 1),
            ("-H", -1.007276, "-", 1, 1),
            ("+Na-2H", 20.974666, "-", 1, 1),
            ("+Cl", 34.969402, "-", 1, 1),
            ("+Br", 78.918885, "-", 1, 1),
            ("-2H+", -2 * 1.007276, "-", 1, 1),
        ],
    ):
        possibleHits = []

        if checkXN not in ["Don't use", "Exact", "Minimum"] and not checkXN.startswith("PlusMinus_"):
            logging.error("Unknown option '%s' for parameter checkXN, must be either of 'Don't use', 'Exact', 'Minimum' or 'PlusMinux_X' where X is a positive integer" % (checkXN))
            raise Exception("Unknown option '%s' for parameter checkXN, must be either of 'Don't use', 'Exact', 'Minimum' or 'PlusMinux_X' where X is a positive integer" % (checkXN))

        ## search for non-charged DB entries by subtracting putative adducts from the provided mz value
        for adduct in adducts:
            if polarity == adduct[2] and charges == adduct[3]:
                mass = (mz - adduct[1]) * adduct[3] / adduct[4]

                entries = self._findGeneric(
                    self.dbEntriesNeutral,
                    lambda x: x.mass,
                    mass - mass * ppm / 1000000.0,
                    mass + mass * ppm / 1000000.0,
                )
                if entries:
                    for entry in entries:
                        if rt_min is None or entry.rt_min is None or (abs(rt_min - entry.rt_min) <= rt_error):
                            elems = None
                            if entry.sumFormula != "":
                                fT = formulaTools()
                                elems = fT.parseFormula(entry.sumFormula)
                            if (
                                checkXN == "Don't use"
                                or elems is None
                                or (checkXN == "Exact" and element in elems.keys() and elems[element] == Xn)
                                or (checkXN == "Minimum" and element in elems.keys() and elems[element] >= Xn)
                                or (checkXN.startswith("PlusMinus_") and element in elems.keys() and abs(elems[element] - Xn) <= int(checkXN[10 : len(checkXN)]))
                            ):
                                entry = deepcopy(entry)
                                entry.hitType = "(MZ as %s) matched to (DB-mass)" % (adduct[0])
                                entry.matchErrorPPM = (mass - entry.mass) * 1e6 / mass
                                entry.matchErrorMass = mass - entry.mass
                                possibleHits.append(entry)

        ## search for charged DB entries by the provided mz value

        entries = self._findGeneric(
            self.dbEntriesMZ,
            lambda x: x.mz,
            mz - mz * ppm / 1000000.0,
            mz + mz * ppm / 1000000.0,
        )
        if entries:
            for entry in entries:
                if entry.polarity == polarity and (rt_min is None or entry.rt_min is None or (abs(rt_min - entry.rt_min) <= rt_error)):
                    elems = None
                    if entry.sumFormula != "":
                        fT = formulaTools()
                        elems = fT.parseFormula(entry.sumFormula)
                    if (
                        checkXN == "Don't use"
                        or elems is None
                        or (checkXN == "Exact" and element in elems.keys() and elems[element] == Xn)
                        or (checkXN == "Minimum" and element in elems.keys() and elems[element] >= Xn)
                        or (checkXN.startswith("PlusMinus_") and element in elems.keys() and abs(elems[element] - Xn) <= int(checkXN[10 : len(checkXN)]))
                    ):
                        entry = deepcopy(entry)
                        entry.hitType = "(MZ) matched to (DB-MZ) match"
                        entry.matchErrorPPM = (mz - entry.mz) * 1e6 / mz
                        entry.matchErrorMass = mz - entry.mz
                        possibleHits.append(entry)

        return possibleHits

    def searchDBForMass(
        self,
        mass,
        polarity,
        charges,
        ppm,
        rt_min=None,
        rt_error=0.1,
        checkXN="Exact",
        element="C",
        Xn=1,
        adducts=[
            ("+H", 1.007276, "+", 1, 1),
            ("+NH4", 18.033823, "+", 1, 1),
            ("+Na", 22.989218, "+", 1, 1),
            ("+K", 38.963158, "+", 1, 1),
            ("+CH3OH+H", 33.033489, "+", 1, 1),
            ("-H", -1.007276, "-", 1, 1),
            ("+Na-2H", 20.974666, "-", 1, 1),
            ("+Cl", 34.969402, "-", 1, 1),
            ("+Br", 78.918885, "-", 1, 1),
            ("-2H+", -2 * 1.007276, "-", 1, 1),
        ],
        mz=None,
    ):
        possibleHits = []

        ## search for charged DB entries by adding adducts to the provided mass
        for adduct in adducts:
            if charges == adduct[3]:
                mz = mass * adduct[4] / adduct[3] + adduct[1]

                entries = self._findGeneric(
                    self.dbEntriesMZ,
                    lambda x: x.mz,
                    mz - mz * ppm / 1000000.0,
                    mz + mz * ppm / 1000000.0,
                )
                if entries:
                    for entry in entries:
                        if entry.polarity == adduct[2] and (rt_min is None or entry.rt_min is None or (abs(rt_min - entry.rt_min) <= rt_error)):
                            elems = None
                            if entry.sumFormula != "":
                                fT = formulaTools()
                                elems = fT.parseFormula(entry.sumFormula)
                            if (
                                checkXN == "Don't use"
                                or elems is None
                                or (checkXN == "Exact" and element in elems.keys() and elems[element] == Xn)
                                or (checkXN == "Minimum" and element in elems.keys() and elems[element] >= Xn)
                                or (checkXN.startswith("PlusMinus_") and element in elems.keys() and abs(elems[element] - Xn) <= int(checkXN[10 : len(checkXN)]))
                            ):
                                entry = deepcopy(entry)
                                entry.hitType = "(mass) matched to (DB-MZ) match with %s" % (adduct[0])
                                entry.matchErrorPPM = (mz - entry.mz) * 1e6 / mz
                                entry.matchErrorMass = mz - entry.mz
                                possibleHits.append(entry)

        ## search for non-charged DB entries by the provided mass
        entries = self._findGeneric(
            self.dbEntriesNeutral,
            lambda x: x.mass,
            mass - mass * ppm / 1000000.0,
            mass + mass * ppm / 1000000.0,
        )
        if entries:
            for entry in entries:
                if rt_min is None or entry.rt_min is None or (abs(rt_min - entry.rt_min) <= rt_error):
                    elems = None
                    if entry.sumFormula != "":
                        fT = formulaTools()
                        elems = fT.parseFormula(entry.sumFormula)
                    if (
                        checkXN == "Don't use"
                        or elems is None
                        or (checkXN == "Exact" and element in elems.keys() and elems[element] == Xn)
                        or (checkXN == "Minimum" and element in elems.keys() and elems[element] >= Xn)
                        or (checkXN.startswith("PlusMinus_") and element in elems.keys() and abs(elems[element] - Xn) <= int(checkXN[10 : len(checkXN)]))
                    ):
                        entry = deepcopy(entry)
                        entry.hitType = "(Mass) matched to (DB-Mass) match"
                        entry.matchErrorPPM = (mass - entry.mass) * 1e6 / mass
                        entry.matchErrorMass = mass - entry.mass
                        possibleHits.append(entry)

        ## search for charged formula DB entries by directly matching the feature m/z
        ## (only when polarity matches; these entries were imported from formulas with explicit charge)
        if mz is not None:
            entries = self._findGeneric(
                self.dbEntriesMZ,
                lambda x: x.mz,
                mz - mz * ppm / 1000000.0,
                mz + mz * ppm / 1000000.0,
            )
            if entries:
                for entry in entries:
                    if entry.polarity == polarity and (rt_min is None or entry.rt_min is None or (abs(rt_min - entry.rt_min) <= rt_error)):
                        elems = None
                        if entry.sumFormula != "":
                            fT = formulaTools()
                            elems = fT.parseFormula(entry.sumFormula)
                        if (
                            checkXN == "Don't use"
                            or elems is None
                            or (checkXN == "Exact" and element in elems.keys() and elems[element] == Xn)
                            or (checkXN == "Minimum" and element in elems.keys() and elems[element] >= Xn)
                            or (checkXN.startswith("PlusMinus_") and element in elems.keys() and abs(elems[element] - Xn) <= int(checkXN[10 : len(checkXN)]))
                        ):
                            entry = deepcopy(entry)
                            entry.hitType = "(MZ) matched to (DB charged formula m/z)"
                            entry.matchErrorPPM = (mz - entry.mz) * 1e6 / mz
                            entry.matchErrorMass = mz - entry.mz
                            possibleHits.append(entry)

        return possibleHits

    def searchDB(
        self,
        mass,
        mz,
        polarity,
        charges,
        rt_min,
        ppm=5.0,
        rt_error=0.1,
        checkXN="Exact",
        element="C",
        Xn=1,
        adducts=[
            ("+H", 1.007276, "+", 1, 1),
            ("+NH4", 18.033823, "+", 1, 1),
            ("+Na", 22.989218, "+", 1, 1),
            ("+K", 38.963158, "+", 1, 1),
            ("+CH3OH+H", 33.033489, "+", 1, 1),
            ("-H", -1.007276, "-", 1, 1),
            ("+Na-2H", 20.974666, "-", 1, 1),
            ("+Cl", 34.969402, "-", 1, 1),
            ("+Br", 78.918885, "-", 1, 1),
            ("-2H+", -2 * 1.007276, "-", 1, 1),
        ],
    ):
        if mass is not None:
            return self.searchDBForMass(
                mass,
                polarity,
                charges,
                ppm,
                rt_min,
                rt_error,
                checkXN,
                element,
                Xn,
                adducts,
                mz=mz,
            )
        else:
            return self.searchDBForMZ(
                mz,
                polarity,
                charges,
                ppm,
                rt_min,
                rt_error,
                checkXN,
                element,
                Xn,
                adducts,
            )


if False and __name__ == "__main__":
    db = DBSearch()
    db.addEntriesFromFile("SomeMets", "N:/iBAM/Christoph/Maria/DB_BiolPaper_Mets.txt")
    db.optimizeDB()

    print("hits from mass obtained from negative ion mode")
    for hit in db.searchDB(
        mass=290.1379484,
        mz=290.1379484 - 1.007276,
        polarity="-",
        charges=1,
        rt_min=None,
        ppm=5,
        rt_error=0.1,
        checkXN="Minimal",
        element="C",
        Xn=9,
    ):
        print(hit.sumFormula, hit.hitType, hit.name, str(hit.additionalInfo), str(hit))
    print("hits from mass obtained from positive ion mode")
    for hit in db.searchDB(
        mass=290.1379484,
        mz=290.1379484 - 1.007276,
        polarity="+",
        charges=1,
        rt_min=None,
        ppm=5,
        rt_error=0.1,
        checkXN="Minimal",
        element="C",
        Xn=9,
    ):
        print(hit.sumFormula, hit.hitType, hit.name, str(hit.additionalInfo), str(hit))

    print("mz hits obtained from neg mode ion")
    for hit in db.searchDB(
        mass=None,
        mz=290.1379484 - 1.007276,
        polarity="-",
        charges=1,
        rt_min=None,
        ppm=5,
        rt_error=0.1,
        checkXN="Minimal",
        element="C",
        Xn=9,
    ):
        print(hit.sumFormula, hit.hitType, hit.name, str(hit.additionalInfo), str(hit))

    print("mz hits obtained from pos mode ion")
    for hit in db.searchDB(
        mass=None,
        mz=290.1379484 + 1.007276,
        polarity="+",
        charges=1,
        rt_min=None,
        ppm=5,
        rt_error=0.1,
        checkXN="Minimal",
        element="C",
        Xn=9,
    ):
        print(hit.sumFormula, hit.hitType, hit.name, str(hit.additionalInfo), str(hit))


if True and __name__ == "__main__":
    db = DBSearch()
    # db.addEntriesFromFile("SOS", "C:/Users/cbueschl/Desktop/me2db_sos_db_250214.tsv")
    # db.addEntriesFromFile("primary_metabolites", "C:/Users/cbueschl/Desktop/me2db_primary_metabolites.tsv")
    # db.addEntriesFromFile("flaxCyc", "C:/Users/cbueschl/Desktop/me2db_flaxCyc.tsv")
    db.addEntriesFromFile("KEGG", "C:/development/MetaboliteDBs/DBsForMetExtractII/KEGG_compounds.tsv")
    db.optimizeDB()

    mass = None
    mz = 119.0503171

    polarity = "-"
    charges = 1

    hits = db.searchDB(
        mass=mass,
        mz=mz,
        polarity=polarity,
        charges=charges,
        rt_min=None,
        ppm=5,
        checkXN="Don't use",
    )

    import sys

    sys.stdout.flush()
    print("\n\n")
    print("DBName\tNum\tName\tChemicalFormula\tErrorPPM\tAdditionalInfo")
    for hit in hits:
        print(
            "%s\t%s\t%s\t%s\t%s\t%s"
            % (
                hit.dbName,
                hit.num,
                hit.name,
                hit.sumFormula,
                hit.matchErrorPPM,
                hit.additionalInfo,
            )
        )
