from __future__ import absolute_import, division, print_function
import json
import logging
import os
import pprint
import polars as pl
from .formulaTools import formulaTools
from .PolarsDB import PolarsDB
from .resultsPostProcessing import generateSumFormulas as sumFormulaGeneration
from .resultsPostProcessing import searchDatabases
from .utils import add_sheet_to_excel

pp = pprint.PrettyPrinter(indent=1)

_fT = formulaTools()


def _formula_to_mass(formula):
    """Compute the monoisotopic neutral mass of a sum-formula string (e.g. "C6H12O6"),
    or return None if it cannot be parsed."""
    if not formula:
        return None
    try:
        elems, charge = _fT.parseFormulaWithCharge(str(formula))
        return _fT.calcMolWeight(elems, charge=charge)
    except Exception:
        return None


# defines a group (input data) for which statistic columns (_Stat_) are calculated
def addGroup(to, groupName, minCount, cols):
    to[groupName] = {}

    to[groupName]["minCount"] = minCount
    to[groupName]["colsNames"] = cols
    to[groupName]["cols"] = []


# matches the input columns to the respective statistic column
def matchRows(groups, rowNames):
    for groupName, groupProps in groups.items():
        for rowName in groupProps["colsNames"]:
            for i in range(len(rowNames)):
                if rowNames[i] == rowName:
                    groupProps["cols"].append(i)


# re-arrange data and add statistic columns
def addStatsColumnToResults(metaFile, groups, outputOrder, sheet_name="Sheet1", new_sheet_name="Sheet2", commentStartingCharacter="#"):
    """
    Add statistics columns to results matrix.

    Args:
        metaFile: Input Excel file path
        groups: Dictionary of group definitions
        toFile: Output Excel file path
        outputOrder: List of group names in desired output order
        sheet_name: Name of the sheet to read from and write to
        commentStartingCharacter: Character that marks comment rows
    """
    # Read Excel file using polars
    df = pl.read_excel(metaFile, sheet_name=sheet_name)
    headers = df.columns

    # Match columns to groups
    matchRows(groups, headers)

    # Find MZ and RT column indices for sorting
    mzInd = headers.index("MZ") if "MZ" in headers else -1
    rtInd = headers.index("RT") if "RT" in headers else -1

    # Add group statistics columns
    for groupName in outputOrder:
        # Count non-empty values in group columns
        col_names = [headers[i] for i in groups[groupName]["cols"]]

        # Create a new column that counts non-null, non-empty values
        count_expr = pl.lit(0, pl.Int64)
        for col_name in col_names:
            # Check if column value is not null and not empty string
            count_expr = count_expr + pl.when((pl.col(col_name).is_not_null()) & (pl.col(col_name).cast(pl.Utf8).str.strip_chars() != "")).then(1).otherwise(0)

        # make couint_exp only posiitive values and remove 0 with None
        count_expr = pl.when(count_expr > 0).then(count_expr).otherwise(None)
        df = df.with_columns(count_expr.alias(groupName))

    # Sort by RT and MZ if both columns exist
    if rtInd != -1 and mzInd != -1:
        df = df.sort(["RT", "MZ"])

    # Write the dataframe to Excel
    add_sheet_to_excel(metaFile, df, new_sheet_name)


# omit those columns that have less results than required by the user input
def performGroupOmit(infile, groupStats, sheet_name="Sheet1", new_sheet_name="FilteredResults"):
    """
    Filter rows based on group statistics and create separate sheets for omitted and false positive features.

    Args:
        infile: Input Excel file path
        groupStats: List of tuples (group_name, min_count, omit_flag, false_positive_flag)
        outfile: Output Excel file path
        sheet_name: Name of the sheet to read from
    """
    # Read Excel file using polars
    df = pl.read_excel(infile, sheet_name=sheet_name)
    headers = df.columns

    # Initialize filter expressions
    all_gomit = True
    use_filter = pl.select(pl.repeat(False, df.height)).to_series()

    for gname, gmin, gomit, gremoveAsFalsePositive in groupStats:
        print("Processing group:", gname, "with min count:", gmin, "omit:", gomit, "remove as false positive:", gremoveAsFalsePositive)
        if gname in headers:
            if gomit:
                # Include rows where this group has >= min_count
                use_filter = use_filter | ((df[gname].is_not_null()) & (df[gname].cast(pl.Int64) >= gmin))
                print("   - Applying omit filter: keeping rows where", gname, ">=", gmin, "(", use_filter.sum(), "rows will be kept after this filter)")
                all_gomit = False

    # Create filtered dataframes
    if all_gomit:
        # No omit filters applied, keep all data
        data_df = df
        not_used_df = pl.DataFrame()
    else:
        data_df = df.filter(use_filter)
        not_used_df = df.filter(~use_filter)

    false_positive_filter = pl.select(pl.repeat(False, data_df.height)).to_series()
    for gname, gmin, gomit, gremoveAsFalsePositive in groupStats:
        print("Processing group:", gname, "with min count:", gmin, "omit:", gomit, "remove as false positive:", gremoveAsFalsePositive)
        if gname in headers:
            if gremoveAsFalsePositive:
                # Mark as false positive if any value > 0 in this group
                false_positive_filter = false_positive_filter | ((data_df[gname].is_not_null()) & (data_df[gname].cast(pl.Int64) > 0))
                print("   - Applying false positive filter: marking rows where", gname, "> 0 as false positives (", false_positive_filter.sum(), "rows will be marked as false positives after this filter)")

    if false_positive_filter.sum() > 0:
        false_positives_df = data_df.filter(false_positive_filter)
        data_df = data_df.filter(~false_positive_filter)
    else:
        false_positives_df = pl.DataFrame()

    print("Final counts after filtering: kept", len(data_df), "rows, omitted", len(not_used_df), "rows, false positives marked:", len(false_positives_df) if "false_positives_df" in locals() else 0)

    add_sheet_to_excel(infile, data_df, new_sheet_name, overwrite=True)
    if len(not_used_df) > 0:
        add_sheet_to_excel(infile, not_used_df, new_sheet_name + "_Omitted")
    if len(false_positives_df) > 0:
        add_sheet_to_excel(infile, false_positives_df, new_sheet_name + "_FalsePositives")


def annotateWithDatabases(
    file,
    sheet_name,
    new_sheet_name,
    dbFiles,
    useAdducts,
    ppm,
    correctppmPosMode,
    correctppmNegMode,
    rtError,
    useRt,
    checkXnInHits,
    processedElement,
    pwMaxSet=None,
    pwValSet=None,
    smiles_mismatches=None,
):
    """
    Annotate metabolites by searching in databases using PolarsDB.

    This function:
    1. Loads the results table from the specified sheet using PolarsDB
    2. Imports database entries from provided files
    3. Adds all necessary database annotation columns
    4. Searches for database hits for each metabolite
    5. Updates only rows with hits using the [] operator
    6. Saves the annotated results to a new sheet

    Args:
        file: Path to the results file (PolarsDB format)
        sheet_name: Name of the sheet to read from (e.g., "3_Reintegrated")
        new_sheet_name: Name of the sheet to write to (e.g., "6_Annotated")
        dbFiles: List of database file paths
        useAdducts: List of adduct definitions [[name, mzoffset, polarity, charge, mCount], ...]
        ppm: Mass accuracy in ppm
        correctppmPosMode: PPM correction for positive mode
        correctppmNegMode: PPM correction for negative mode
        rtError: Maximum RT error for matching
        useRt: Whether to use RT in matching
        checkXnInHits: How to check Xn ("Exact", "Minimum", "Don't use", or "PlusMinus_X")
        processedElement: Element to check in formulas (e.g., "C")
        pwMaxSet: Progress callback for max value
        pwValSet: Progress callback for current value

    Returns:
        List of annotation column names added
    """
    logging.info("Starting database search annotation using PolarsDB")
    logging.info(f"Reading file {file}")

    # Load the PolarsDB
    plDB = PolarsDB(file, format="xlsx", load_all_tables=True)
    db_info_messages = []

    # Load the results dataframe from the specified sheet
    try:
        results_df = plDB.get_table(sheet_name)
    except Exception as e:
        logging.error(f"Failed to load sheet '{sheet_name}': {e}")
        plDB.close()
        raise

    # Clear any stale database-annotation columns and the compound-focused sheet(s) left over from a
    # previous annotation run before adding the new results, so no old hits (e.g. from a database that
    # is no longer selected, or from a row that no longer matches) can survive into the new output.
    stale_cols = [c for c in results_df.columns if c.startswith("DBs_") or c.startswith("DBs_RT_")]
    if stale_cols:
        logging.info(f"Clearing {len(stale_cols)} stale database-annotation column(s) from sheet '{sheet_name}': {stale_cols}")
        results_df = results_df.drop(stale_cols)
    compound_sheet_name = f"{new_sheet_name}_Compounds"
    if plDB.has_table(compound_sheet_name):
        logging.info(f"Clearing stale sheet '{compound_sheet_name}' before database search annotation")
        plDB.remove_table(compound_sheet_name)

    # Initialize database search
    db = searchDatabases.DBSearch()
    dbNames = []

    # Import database files and collect database names
    logging.info(f"\n\n#########################################\nImporting {len(dbFiles)} database file(s)")
    for dbFile in dbFiles:
        logging.info(f"\n-------------------------\nImporting database file: {dbFile}")
        dbName = dbFile[dbFile.rfind("/") + 1 : dbFile.rfind(".")]
        dbNames.append(dbName)
        errors = []
        try:
            imported, not_imported = db.addEntriesFromFile(dbName, dbFile, error_collector=errors, mismatch_collector=smiles_mismatches)
            if db_info_messages is not None:
                db_info_messages.append(f"Database: {dbName} (file: {dbFile})")
                db_info_messages.append(f"  Imported: {imported} entries successfully")
                if not_imported > 0:
                    db_info_messages.append(f"  Not imported: {not_imported} entries due to errors")
                for err in errors:
                    db_info_messages.append(f"  {err}")
        except IOError as e:
            logging.error(f"   - Cannot process database file '{dbName}' at '{dbFile}': {e}")
            if db_info_messages is not None:
                db_info_messages.append(f"Database: {dbName} (file: {dbFile})")
                db_info_messages.append(f"  Fatal error: {e}")
            continue
    logging.info(f"\nFinished importing databases. Total imported entries: MZ: {len(db.dbEntriesMZ)}, Neutral: {len(db.dbEntriesNeutral)}")

    # Optimize database for searching
    logging.info("\nOptimizing database for searching")
    db.optimizeDB()

    # Add all necessary annotation columns BEFORE searching
    logging.info("Adding annotation columns to dataframe")
    annotationColumns = []
    for dbName in dbNames:
        cols_to_add = [
            (f"DBs_{dbName}_count", ""),
            (f"DBs_{dbName}", ""),
            (f"DBs_RT_{dbName}_count", ""),
            (f"DBs_RT_{dbName}", ""),
        ]

        for col_name, default_value in cols_to_add:
            annotationColumns.append(col_name)
            if col_name not in results_df.columns:
                if isinstance(default_value, int):
                    results_df = results_df.with_columns(pl.lit(default_value, dtype=pl.Int64).alias(col_name))
                else:
                    results_df = results_df.with_columns(pl.lit(default_value, dtype=pl.Utf8).alias(col_name))

    # Create database search helper class
    class DBSearchHelper:
        def __init__(
            self,
            dbs,
            ppm,
            correctppmPosMode,
            correctppmNegMode,
            rtError,
            useRt,
            checkXnInHits,
            processedElement,
        ):
            self.ppm = ppm
            self.correctppmPosMode = correctppmPosMode
            self.correctppmNegMode = correctppmNegMode
            self.rtError = rtError
            self.useRt = useRt
            self.checkXnInHits = checkXnInHits
            self.processedElement = processedElement
            self.dbs = dbs
            self.fT = formulaTools()

        def searchForRow(self, row):
            """
            Search databases for a single row and return dict of hits per database.

            Returns:
                tuple: (hits_per_db, hit_objects)
                    - hits_per_db: Dictionary with formatted hits strings
                      Format: {dbName: {'hits': [...], 'hitsRT': [...]}}
                    - hit_objects: List of (hit, row_info) tuples for compound sheet
            """
            # Extract values from row
            mass = row["M"] if row["M"] is not None and row["M"] != "" and "," not in str(row["M"]) else None
            if mass is not None:
                try:
                    mass = float(mass)
                except Exception:
                    mass = None

            mz = float(row["MZ"])
            polarity = row["Ionisation_Mode"]
            rt_min = float(row["RT"])
            charges = int(row["Charge"])

            xn = 0
            try:
                xn = int(row["Xn"])
            except Exception:
                pass

            # Apply mass correction
            if mass is not None:
                mass = mass + mass * 1.0 * (self.correctppmPosMode if polarity == "+" else self.correctppmNegMode) / 1e6
            mz = mz + mz * 1.0 * (self.correctppmPosMode if polarity == "+" else self.correctppmNegMode) / 1e6

            # Search database
            hits_per_db = {}
            hit_objects = []

            for hit in self.dbs.searchDB(
                mass=mass,
                mz=mz,
                polarity=polarity,
                charges=charges,
                rt_min=rt_min if self.useRt else None,
                ppm=self.ppm,
                rt_error=self.rtError,
                checkXN=self.checkXnInHits,
                element=self.processedElement,
                Xn=xn,
                adducts=useAdducts,
            ):
                if hit.dbName not in hits_per_db:
                    hits_per_db[hit.dbName] = {"hits": [], "hitsRT": []}

                hit_str = f"(Name: {hit.name}, Type: {hit.hitType}, Num: {hit.num}, Formula: {hit.sumFormula}, RT: {hit.rt_min}, MassErrorPPM: {hit.matchErrorPPM:.5f}, MassErrorMass: {hit.matchErrorMass:.5f}, Additional information: {hit.additionalInfo})"
                hits_per_db[hit.dbName]["hits"].append(hit_str)

                if hit.rt_min is not None and hit.rt_min != "":
                    try:
                        rtDelta = abs(float(hit.rt_min) - rt_min)
                        rt_hit_str = f"RT delta: {rtDelta:.2f} (Name: {hit.name} Type: {hit.hitType}, Num: {hit.num}, Formula: {hit.sumFormula}, RT: {hit.rt_min}, MassErrorPPM: {hit.matchErrorPPM:.5f}, MassErrorMass: {hit.matchErrorMass:.5f}, RTError: {float(hit.rt_min) - rt_min if hit.rt_min is not None and hit.rt_min != '' else ''}, Additional information: {hit.additionalInfo})"
                        hits_per_db[hit.dbName]["hitsRT"].append((rtDelta, rt_hit_str))
                    except Exception as e:
                        logging.error(f"Error processing RT for database hit: {e}")

                # Store hit object with row information for compound-focused sheet
                row_info = {
                    "Feature_Num": row.get("Num"),
                    "Feature_RT": row.get("RT"),
                    "Feature_MZ": row.get("MZ"),
                    "Feature_Xn": row.get("Xn"),
                    "Feature_Ionisation_Mode": row.get("Ionisation_Mode"),
                    "Feature_Charge": row.get("Charge"),
                    "Feature_M": row.get("M"),
                    "Feature_OGroup": row.get("OGroup"),
                    "Feature_Ion": row.get("Ion"),
                    "Feature_Loss": row.get("Loss"),
                    "Feature_Relative_peakarea_in_group": row.get("Relative_peakarea_in_group"),
                    "Feature_Average_peakarea": row.get("Average_peakarea"),
                }
                hit_objects.append((hit, row_info))

            return hits_per_db, hit_objects

    # Create search helper instance
    searcher = DBSearchHelper(
        db,
        ppm,
        correctppmPosMode,
        correctppmNegMode,
        rtError,
        useRt,
        checkXnInHits,
        processedElement,
    )

    # Process each row and update only those with hits
    total_rows = len(results_df)
    if pwMaxSet is not None:
        pwMaxSet(total_rows)

    logging.info(f"Searching database hits for {total_rows} metabolites, parameters are ppm: {ppm}, correctppmPosMode: {correctppmPosMode}, correctppmNegMode: {correctppmNegMode}, rtError: {rtError}, useRt: {useRt}, checkXnInHits: {checkXnInHits}, processedElement: {processedElement}")

    # Collect all hits for compound-focused sheet
    all_compound_hits = []

    for row_idx in range(total_rows):
        if pwValSet is not None and row_idx % 10 == 0:
            pwValSet(row_idx)

        # Get row as dict
        row = results_df.row(row_idx, named=True)

        # Search for database hits
        hits_per_db, hit_objects = searcher.searchForRow(row)

        print(f"Row {row_idx + 1}/{total_rows}, mz {row.get('MZ')}, rt {row.get('RT')}, charge {row.get('Charge')}, polarity {row.get('Ionisation_Mode')}, Xn {row.get('Xn')}\n   - Found hits in {len(hits_per_db)} databases")

        # Update only if there are hits
        if hits_per_db:
            for dbName, hit_data in hits_per_db.items():
                # Update count and hits columns
                results_df[row_idx, f"DBs_{dbName}_count"] = len(hit_data["hits"]) if hit_data["hits"] else ""
                # results_df[row_idx, f"DBs_{dbName}"] = '"%s"' % (";\n".join(hit_data["hits"])) if hit_data["hits"] else ""
                results_df[row_idx, f"DBs_{dbName}"] = json.dumps(hit_data["hits"], ensure_ascii=True) if hit_data["hits"] else ""

                # Update RT hits if available
                if len(hit_data["hitsRT"]) > 0:
                    sorted_hits = sorted(hit_data["hitsRT"], key=lambda x: x[0])
                    results_df[row_idx, f"DBs_RT_{dbName}_count"] = len(hit_data["hitsRT"]) if hit_data["hitsRT"] else ""
                    # results_df[row_idx, f"DBs_RT_{dbName}"] = '"%s"' % (";\n".join([b[1] for b in sorted_hits])) if hit_data["hitsRT"] else ""
                    results_df[row_idx, f"DBs_RT_{dbName}"] = json.dumps([b[1] for b in sorted_hits], ensure_ascii=True) if hit_data["hitsRT"] else ""

        # Collect hit objects for compound sheet
        all_compound_hits.extend(hit_objects)

    # Save annotated results to new sheet
    logging.info(f"Saving annotated results to sheet: {new_sheet_name}")
    plDB.set_table(new_sheet_name, results_df)

    # Create compound-focused sheet
    if all_compound_hits:
        logging.info(f"Creating compound-focused sheet with {len(all_compound_hits)} database hits")

        # Collect all unique additionalInfo keys across all hits so every row gets a column
        all_additional_keys = {}
        for hit, _row_info in all_compound_hits:
            for k in hit.additionalInfo.keys():
                ks = "".join(c.replace(".", "_") if c.isalnum() else "_" for c in k)  # Sanitize key to be a valid column name
                all_additional_keys[k] = ks

        def sanitize_str(s):
            if s is None:
                return ""
            # TODO
            # return ascii(s)[1:-1]
            return str(s)

        compound_rows = []
        for hit, row_info in all_compound_hits:
            compound_row = {
                # Database entry information
                "DB_Name": sanitize_str(hit.dbName),
                "DB_Num": sanitize_str(hit.num),
                "DB_CompoundName": sanitize_str(hit.name),
                "DB_SumFormula": sanitize_str(hit.sumFormula),
                "DB_Mass": float(hit.mass) if hit.mass is not None and hit.mass != "" else None,
                "DB_RT_min": float(hit.rt_min) if hit.rt_min is not None and hit.rt_min != "" else None,
                "DB_MZ": float(hit.mz) if hit.mz is not None and hit.mz != "" else None,
                "DB_Polarity": sanitize_str(hit.polarity),
                "HitType": sanitize_str(hit.hitType),
                "MatchErrorPPM": float(hit.matchErrorPPM) if hit.matchErrorPPM is not None else None,
                "MatchErrorMass": float(hit.matchErrorMass) if hit.matchErrorMass is not None else None,
                # Feature information where the hit was found
                "Feature_Num": row_info["Feature_Num"],
                "Feature_OGroup": row_info["Feature_OGroup"],
                "Feature_RT": row_info["Feature_RT"],
                "Feature_MZ": row_info["Feature_MZ"],
                "Feature_Xn": row_info["Feature_Xn"],
                "Feature_Ionisation_Mode": row_info["Feature_Ionisation_Mode"],
                "Feature_Charge": row_info["Feature_Charge"],
                "Feature_M": row_info["Feature_M"],
                "Feature_Ion": row_info["Feature_Ion"],
                "Feature_Loss": row_info["Feature_Loss"],
                "Feature_Relative_peakarea_in_group": row_info["Feature_Relative_peakarea_in_group"],
                "Feature_Average_peakarea": row_info["Feature_Average_peakarea"],
            }

            # Expand additionalInfo (e.g. SMILES, InChI, ...) into individual columns with a DB_Info_ prefix
            for k, ks in all_additional_keys.items():
                col = f"DB_Info_{ks}"
                if col not in compound_row:
                    compound_row[col] = sanitize_str(hit.additionalInfo.get(k, None))

            # add compound row and new columns
            compound_rows.append(compound_row)

        # Build schema overrides only for the known numeric columns; let polars infer the rest
        schema_overrides = {
            "DB_Mass": pl.Float64,
            "DB_RT_min": pl.Float64,
            "DB_MZ": pl.Float64,
            "MatchErrorPPM": pl.Float64,
            "MatchErrorMass": pl.Float64,
            "Feature_RT": pl.Float64,
            "Feature_MZ": pl.Float64,
            "Feature_M": pl.Utf8,
            "Feature_Relative_peakarea_in_group": pl.Float64,
            "Feature_Average_peakarea": pl.Float64,
        }

        # Create dataframe for compound-focused sheet
        compound_df = pl.DataFrame(compound_rows, schema_overrides=schema_overrides, infer_schema_length=len(compound_rows))

        # Sort by database name, compound name, then feature id
        sort_cols = [c for c in ["DB_Name", "DB_CompoundName", "Feature_Num"] if c in compound_df.columns]
        if sort_cols:
            compound_df = compound_df.sort(sort_cols)

        # Save to compound-focused sheet
        compound_sheet_name = f"{new_sheet_name}_Compounds"
        logging.info(f"Saving compound-focused results to sheet '{compound_sheet_name}' with {len(compound_df)} hits")
        plDB.set_table(compound_sheet_name, compound_df)
    else:
        logging.info("No database hits found, skipping compound-focused sheet")

    db_info_db = pl.DataFrame(db_info_messages)
    plDB.set_table("DB_info", db_info_db)
    plDB.close()

    logging.info(f"Database search annotation completed. Added {len(annotationColumns)} columns")
    return annotationColumns


def annotateWithMSMSLibrary(
    file,
    sheet_name,
    new_sheet_name,
    library_files,
    msms_by_feature,
    algorithm_name="ModifiedCosineHungarian",
    mz_tolerance=0.01,
    precursor_mz_tolerance=0.01,
    min_matched_peaks=4,
    score_cutoff=0.8,
    fragment_min_rel_abundance=0.0,
    require_same_precursor_mz=True,
    pwMaxSet=None,
    pwValSet=None,
):
    """
    Annotate metabolites by matching their experimental MS/MS spectra against one or more
    MGF/JSON spectral library files using matchms.

    For every feature (row of the results table) and every library file, a column named
    "MSMS_<library file name>" is added, containing a JSON list of match dicts:
        {"score", "matched_peaks", "compound_names", "exp_matched_abundances", "db_matched_abundances"}
    where "exp_matched_abundances"/"db_matched_abundances" are lists of {mz, relative_intensity}
    (relative to the respective spectrum, i.e. each spectrum's own max peak = 100%).

    A "<new_sheet_name>_MSMS" sheet is also created, with one row per feature-spectrum match
    (spectra as primary fields, plus score/matched-peaks/etc. and feature meta-information),
    analogous to "<new_sheet_name>_Compounds" for database hits.

    Args:
        file: Path to the results file (PolarsDB format)
        sheet_name: Name of the sheet to read from
        new_sheet_name: Name of the sheet to write to (e.g. "5_Annotated")
        library_files: List of library-file info dicts: {"path", "type" ("mgf"/"json"), "precursor_mz_key"}
        msms_by_feature: dict mapping feature "Num" -> list of experimental spectrum dicts:
            {"mz": array-like, "intensities": array-like, "polarity": "+"/"-"/None, "precursor_mz": float or None}
        algorithm_name: matchms similarity algorithm name (see mgfLibrary.MSMS_LIBRARY_ALGORITHMS)
        mz_tolerance: fragment m/z tolerance (Da) used by the similarity algorithm
        precursor_mz_tolerance: max allowed precursor m/z deviation (Da) between exp. and db. spectra
        min_matched_peaks: minimum number of matched fragments required to keep a match
        score_cutoff: minimum similarity score required to keep a match
        pwMaxSet: optional progress callback for maximum value
        pwValSet: optional progress callback for current value

    Returns:
        List of annotation column names added
    """
    from .MSMS import mgfLibrary

    logging.info("Starting MS/MS spectral library annotation using PolarsDB")

    plDB = PolarsDB(file, format="xlsx", load_all_tables=True)

    try:
        results_df = plDB.get_table(sheet_name)
    except Exception as e:
        logging.error(f"Failed to load sheet '{sheet_name}': {e}")
        plDB.close()
        raise

    # Clear any stale MSMS-annotation columns and the MSMS-focused sheet left over from a previous
    # annotation run (e.g. from a library that is no longer selected) before adding the new results.
    stale_cols = [c for c in results_df.columns if c.startswith("MSMS_")]
    if stale_cols:
        logging.info(f"Clearing {len(stale_cols)} stale MSMS-annotation column(s) from sheet '{sheet_name}': {stale_cols}")
        results_df = results_df.drop(stale_cols)
    msms_sheet_name_stale = f"{new_sheet_name}_MSMS"
    if plDB.has_table(msms_sheet_name_stale):
        logging.info(f"Clearing stale sheet '{msms_sheet_name_stale}' before MSMS library annotation")
        plDB.remove_table(msms_sheet_name_stale)

    # Load libraries (MGF and/or JSON). Store tuple (spectra_list, entry_dict) per library name
    libraries = {}
    for library_entry in library_files:
        lib_path = library_entry["path"]
        lib_name = os.path.basename(lib_path)
        lib_name = lib_name[: lib_name.rfind(".")] if "." in lib_name else lib_name
        try:
            specs = mgfLibrary.load_library_entry(library_entry)
            mgfLibrary.prepare_library_spectra(specs, fragment_min_rel_abundance=fragment_min_rel_abundance)
            libraries[lib_name] = (specs, library_entry)
        except Exception as e:
            logging.error(f"Failed to load spectral library file '{lib_path}': {e}")
            libraries[lib_name] = ([], library_entry)

    # Build MSMS_info summary: per-library counts and polarity breakdowns
    msms_info_rows = []
    for lib_name, (specs, lib_entry) in libraries.items():
        total = len(specs) if specs is not None else 0
        pos = sum(1 for s in specs if getattr(s, "polarity", None) == "+") if specs else 0
        neg = sum(1 for s in specs if getattr(s, "polarity", None) == "-") if specs else 0
        unspecified = total - pos - neg
        msms_info_rows.append(
            {
                "Library": lib_name,
                "Path": lib_entry.get("path") if lib_entry else None,
                "TotalSpectra": total,
                "PositiveSpectra": pos,
                "NegativeSpectra": neg,
                "UnspecifiedSpectra": unspecified,
            }
        )

    # Write MSMS_info summary into the output file so it's always present even if there are no matches
    try:
        if msms_info_rows:
            msms_info_df = pl.DataFrame(msms_info_rows)
            plDB.set_table("MSMS_info", msms_info_df)
    except Exception as e:
        logging.warning(f"Failed to write MSMS_info sheet: {e}")

    annotationColumns = []
    for mgf_name in libraries:
        col_name = f"MSMS_{mgf_name}"
        annotationColumns.append(col_name)
        if col_name not in results_df.columns:
            results_df = results_df.with_columns(pl.lit("", dtype=pl.Utf8).alias(col_name))

    spectra_rows = []
    num_col_idx = {row_num: idx for idx, row_num in enumerate(results_df["Num"].to_list())} if "Num" in results_df.columns else {}
    total_exp_spectra = sum(len(exp_spectra) for exp_spectra in msms_by_feature.values())
    if pwMaxSet is not None:
        pwMaxSet(total_exp_spectra)
    if pwValSet is not None:
        pwValSet(0)

    completed_exp_spectra = 0

    for feature_num, exp_spectra in msms_by_feature.items():
        row_idx = num_col_idx.get(feature_num)
        if row_idx is None:
            continue
        row = results_df.row(row_idx, named=True)

        for mgf_name, lib_info in libraries.items():
            library_spectra, lib_entry = lib_info
            col_name = f"MSMS_{mgf_name}"
            col_matches = []

            for exp_spec in exp_spectra:
                completed_exp_spectra += 1
                matches = mgfLibrary.match_spectrum_against_library(
                    exp_mz=exp_spec["mz"],
                    exp_intensities=exp_spec["intensities"],
                    exp_polarity=exp_spec.get("polarity"),
                    exp_precursor_mz=exp_spec.get("precursor_mz"),
                    library_spectra=library_spectra,
                    algorithm_name=algorithm_name,
                    mz_tolerance=mz_tolerance,
                    precursor_mz_tolerance=precursor_mz_tolerance,
                    require_same_precursor_mz=require_same_precursor_mz,
                    min_matched_peaks=min_matched_peaks,
                    score_cutoff=score_cutoff,
                    fragment_min_rel_abundance=fragment_min_rel_abundance,
                )
                if pwValSet is not None:
                    pwValSet(completed_exp_spectra)
                for m in matches:
                    col_matches.append(m)
                    # Build spectra row with requested retained metadata fields
                    base_row = {
                        "Library": mgf_name,
                        "Score": m["score"],
                        "Precursor_MZ_Diff": None,
                        "Matched_Fragments": m["matched_peaks"],
                        "Compound_Name": m.get("compound_name"),
                        "Compound_ID": m.get("db_spectrum_index"),
                        "Feature_Num": row.get("Num"),
                        "Feature_OGroup": row.get("OGroup"),
                        "Feature_RT": row.get("RT"),
                        "Feature_MZ": row.get("MZ"),
                        "Feature_Xn": row.get("Xn"),
                        "Feature_Ionisation_Mode": row.get("Ionisation_Mode"),
                        "Feature_Charge": row.get("Charge"),
                    }

                    # Add retained library metadata keys if present in lib_entry
                    retain_keys = lib_entry.get("retain_keys", []) if lib_entry else []
                    try:
                        db_idx = int(m.get("db_spectrum_index"))
                    except Exception:
                        db_idx = None
                    if db_idx is not None and db_idx is not False:
                        try:
                            lib_spec = library_spectra[db_idx] if 0 <= db_idx < len(library_spectra) else None
                        except Exception:
                            lib_spec = None
                    else:
                        lib_spec = None

                    if lib_spec is not None and retain_keys:
                        for k in retain_keys:
                            # store under the original key name
                            base_row[k] = lib_spec.metadata.get(k) if lib_spec.metadata is not None else None

                    # Always resolve structure-related fields (SMILES/InChI/InChIKey/Formula), regardless
                    # of the user's retain_keys selection, so structures can always be depicted for a match.
                    if lib_spec is not None and lib_spec.metadata is not None:
                        for k, v in mgfLibrary.resolve_structure_fields(lib_spec.metadata).items():
                            if v is not None:
                                base_row[k] = v

                    # Theoretical (monoisotopic) mass of the resolved sum formula, for display
                    # alongside the formula ("<formula> (<mass>)")
                    formula = base_row.get("Formula")
                    base_row["TheoreticalMass"] = _formula_to_mass(formula) if formula else None

                    # Always resolve acquisition metadata of the library (reference) spectrum
                    # (instrument, fragmentation mode, collision energy, RT), regardless of the
                    # user's retain_keys selection.
                    if lib_spec is not None and lib_spec.metadata is not None:
                        for k, v in mgfLibrary.resolve_spectrum_meta_fields(lib_spec.metadata).items():
                            base_row[f"Library_{k}"] = v
                    base_row["Library_Precursor_MZ"] = lib_spec.precursor_mz if lib_spec is not None else None

                    # Metadata of the matched EXPERIMENTAL scan (sample it came from, its own RT/
                    # precursor m/z/fragmentation mode/collision energy), so a match can be traced
                    # back to the exact scan it originated from.
                    if isinstance(exp_spec, dict):
                        base_row["Sample_File"] = exp_spec.get("sample_file")
                        base_row["Scan_RT"] = exp_spec.get("retention_time_min")
                        base_row["Scan_Precursor_MZ"] = exp_spec.get("precursor_mz")
                        base_row["Scan_Fragmentation_Mode"] = exp_spec.get("fragmentation_mode")
                        base_row["Scan_Collision_Energy"] = exp_spec.get("collision_energy")

                    # Add fragment counts for DB and experimental spectra
                    try:
                        base_row["DB_NumFragments"] = len(lib_spec.mz) if lib_spec is not None and hasattr(lib_spec, "mz") else None
                    except Exception:
                        base_row["DB_NumFragments"] = None
                    try:
                        base_row["Exp_NumFragments"] = len(exp_spec.get("mz") if isinstance(exp_spec, dict) else getattr(exp_spec, "mz", []))
                    except Exception:
                        base_row["Exp_NumFragments"] = None

                    # compute precursor mz diff (experimental - database) if available
                    try:
                        exp_prec = exp_spec.get("precursor_mz")
                        db_prec = lib_spec.precursor_mz if lib_spec is not None and hasattr(lib_spec, "precursor_mz") else None
                        if exp_prec is not None and db_prec is not None:
                            base_row["Precursor_MZ_Diff"] = float(exp_prec) - float(db_prec)
                    except Exception:
                        base_row["Precursor_MZ_Diff"] = None

                    spectra_rows.append(base_row)

            if col_matches:
                # Only include the minimal requested fields in the per-row JSON: score, matched_peaks, compound_name, compound_id and library
                match_summaries = [
                    {
                        "score": m["score"],
                        "matched_peaks": m["matched_peaks"],
                        "compound_name": m.get("compound_name"),
                        "compound_id": m.get("db_spectrum_index"),
                        "library": mgf_name,
                    }
                    for m in sorted(col_matches, key=lambda m: m["score"], reverse=True)
                ]
                results_df[row_idx, col_name] = json.dumps(match_summaries, ensure_ascii=True)

    logging.info(f"Saving MSMS-library-annotated results to sheet: {new_sheet_name}")
    plDB.set_table(new_sheet_name, results_df)

    if spectra_rows:
        msms_sheet_name = f"{new_sheet_name}_MSMS"
        # Infer schema from present fields; ensure numeric types for Score/Matched_Fragments/Feature_RT/Feature_MZ when possible
        schema_overrides = {}
        # attempt to set numeric types if keys exist
        if any("Score" in r for r in spectra_rows):
            schema_overrides["Score"] = pl.Float64
        if any("Matched_Fragments" in r for r in spectra_rows):
            schema_overrides["Matched_Fragments"] = pl.Int64
        if any("Feature_RT" in r for r in spectra_rows):
            schema_overrides["Feature_RT"] = pl.Float64
        if any("Feature_MZ" in r for r in spectra_rows):
            schema_overrides["Feature_MZ"] = pl.Float64
        if any("Precursor_MZ_Diff" in r for r in spectra_rows):
            schema_overrides["Precursor_MZ_Diff"] = pl.Float64
        if any("DB_NumFragments" in r for r in spectra_rows):
            schema_overrides["DB_NumFragments"] = pl.Int64
        if any("Exp_NumFragments" in r for r in spectra_rows):
            schema_overrides["Exp_NumFragments"] = pl.Int64
        if any("TheoreticalMass" in r for r in spectra_rows):
            schema_overrides["TheoreticalMass"] = pl.Float64
        if any("Library_RT" in r for r in spectra_rows):
            schema_overrides["Library_RT"] = pl.Float64
        if any("Library_Precursor_MZ" in r for r in spectra_rows):
            schema_overrides["Library_Precursor_MZ"] = pl.Float64
        if any("Scan_RT" in r for r in spectra_rows):
            schema_overrides["Scan_RT"] = pl.Float64
        if any("Scan_Precursor_MZ" in r for r in spectra_rows):
            schema_overrides["Scan_Precursor_MZ"] = pl.Float64
        if any("Scan_Collision_Energy" in r for r in spectra_rows):
            schema_overrides["Scan_Collision_Energy"] = pl.Float64

        # Ensure all user-selected retain_keys are present as columns (even if empty)
        all_retain_keys = set()
        for _lib_name, lib_info in libraries.items():
            # libraries values are (specs, entry_dict)
            lib_entry = lib_info[1] if isinstance(lib_info, (list, tuple)) and len(lib_info) > 1 else lib_info
            rk = lib_entry.get("retain_keys", []) if lib_entry and isinstance(lib_entry, dict) else []
            for k in rk:
                all_retain_keys.add(k)

        # Create DataFrame
        spectra_df = pl.DataFrame(spectra_rows, schema_overrides=schema_overrides, infer_schema_length=len(spectra_rows))
        # Add missing retain-key columns as nulls to ensure they appear in the sheet
        for rk in sorted(all_retain_keys):
            if rk not in spectra_df.columns:
                spectra_df = spectra_df.with_columns(pl.lit(None).alias(rk))
        # Reorder columns: ensure Precursor_MZ_Diff follows Score and DB_NumFragments/Exp_NumFragments follow Matched_Fragments
        try:
            if "Matched_Fragments" in spectra_rows[0]:
                # create DataFrame first then reorder if necessary
                pass
        except Exception:
            pass

        sort_cols = [c for c in ["Library", "Score"] if c in spectra_df.columns]
        # Ensure fragment count columns are placed right after Matched_Fragments
        cols = list(spectra_df.columns)
        # Ensure Precursor_MZ_Diff appears right after Score
        if "Score" in cols:
            new_cols = []
            added = set()
            for c in cols:
                if c == "Score":
                    new_cols.append("Score")
                    added.add("Score")
                    if "Precursor_MZ_Diff" in cols and "Precursor_MZ_Diff" not in added:
                        new_cols.append("Precursor_MZ_Diff")
                        added.add("Precursor_MZ_Diff")
                elif c not in added:
                    new_cols.append(c)
                    added.add(c)
            cols = new_cols
        if "Matched_Fragments" in cols:
            # Build a new column order: everything up to Matched_Fragments, then Matched_Fragments,
            # then DB_NumFragments and Exp_NumFragments (if present), then the remaining columns in original order.
            pre = []
            seen = set()
            for c in cols:
                if c == "Matched_Fragments":
                    break
                pre.append(c)
                seen.add(c)
            # collect remaining after Matched_Fragments, excluding the fragment-count
            # columns which are explicitly re-inserted right after Matched_Fragments
            remaining = [c for c in cols if c not in seen and c not in ("Matched_Fragments", "DB_NumFragments", "Exp_NumFragments")]
            new_cols = pre + ["Matched_Fragments"]
            for k in ("DB_NumFragments", "Exp_NumFragments"):
                if k in spectra_df.columns:
                    new_cols.append(k)
            new_cols.extend(remaining)
            # Select only columns that actually exist, without introducing duplicates
            seen_final = set()
            new_cols_dedup = []
            for c in new_cols:
                if c in spectra_df.columns and c not in seen_final:
                    new_cols_dedup.append(c)
                    seen_final.add(c)
            spectra_df = spectra_df.select(new_cols_dedup)
        if sort_cols:
            spectra_df = spectra_df.sort(sort_cols, descending=[False, True] if len(sort_cols) == 2 else True)
        logging.info(f"Saving MSMS spectra-focused results to sheet '{msms_sheet_name}' with {len(spectra_df)} matches")
        plDB.set_table(msms_sheet_name, spectra_df)
        # MSMS_info already written earlier
    else:
        logging.info("No MS/MS library matches found, skipping MSMS-focused sheet")

    plDB.close()

    logging.info(f"MS/MS spectral library annotation completed. Added {len(annotationColumns)} columns")
    return annotationColumns


def testDatabaseImports(dbFiles):
    """
    Test-import database files and return per-db import results without writing to any output file.

    Args:
        dbFiles: List of database file paths

    Returns:
        List of dicts: [{'db_name': str, 'db_file': str, 'imported': int, 'not_imported': int, 'errors': list[str]}, ...]
    """
    results = []
    for dbFile in dbFiles:
        dbName = dbFile[dbFile.rfind("/") + 1 : dbFile.rfind(".")]
        errors = []
        db = searchDatabases.DBSearch()
        try:
            imported, not_imported = db.addEntriesFromFile(dbName, dbFile, error_collector=errors)
        except Exception as e:
            errors.insert(0, f"Fatal error: {e}")
            imported = 0
            not_imported = 0
        results.append(
            {
                "db_name": dbName,
                "db_file": dbFile,
                "imported": imported,
                "not_imported": not_imported,
                "errors": errors,
            }
        )
    return results


def annotateWithSumFormulas(
    file,
    sheet_name,
    useAtoms,
    atomsRange,
    processedElement,
    useExactXn,
    ppm,
    ppmCorrectionPosMode,
    ppmCorrectionNegMode,
    useAdducts,
    pwMaxSet=None,
    pwValSet=None,
    nCores=1,
):
    """
    Annotate metabolites with generated sum formulas using PolarsDB.

    This function:
    1. Loads the results table from the specified sheet using PolarsDB
    2. Generates sum formulas for each metabolite based on element ranges
    3. Adds sum formula columns with different element combinations
    4. Overwrites the input sheet with annotated results

    Args:
        file: Path to the results file (PolarsDB format)
        sheet_name: Name of the sheet to read from and write to (e.g., "6_Annotated")
        useAtoms: List of atoms to use in sum formulas (e.g., ["C", "H", "O", "N", "S"])
        atomsRange: List of [min, max] ranges for each atom
        processedElement: Element to check in formulas (e.g., "C")
        useExactXn: How to check Xn ("Exact", "Minimum", "Don't use", or "PlusMinus_X")
        ppm: Mass accuracy in ppm
        ppmCorrectionPosMode: PPM correction for positive mode
        ppmCorrectionNegMode: PPM correction for negative mode
        useAdducts: List of adduct definitions [[name, mzoffset, polarity, charge, mCount], ...]
        pwMaxSet: Progress callback for max value
        pwValSet: Progress callback for current value
        nCores: Number of CPU cores to use

    Returns:
        List of sum formula column names added
    """
    logging.info("Starting sum formula generation using PolarsDB")
    logging.info(f"Reading file {file}")

    # Load the PolarsDB
    plDB = PolarsDB(file, format="xlsx", load_all_tables=True)

    # Load the results dataframe from the specified sheet
    try:
        results_df = plDB.get_table(sheet_name)
    except Exception as e:
        logging.error(f"Failed to load sheet '{sheet_name}': {e}")
        plDB.close()
        raise

    # Clear the sum-formula-focused sheet left over from a previous annotation run before adding the
    # new results. (The SFs_* columns themselves are always fully recomputed/overwritten below for
    # every row, so no separate column-clearing is needed for those.)
    sf_sheet_name_stale = f"{sheet_name}_SumFormulas"
    if plDB.has_table(sf_sheet_name_stale):
        logging.info(f"Clearing stale sheet '{sf_sheet_name_stale}' before sum formula annotation")
        plDB.remove_table(sf_sheet_name_stale)

    # Use the new Polars-based sum formula generation directly
    logging.info(f"Generating sum formulas with atoms: {useAtoms}")
    results_df, sf_compound_hits = sumFormulaGeneration.annotatePolarsTableWithSumFormulas(
        results_df,
        useAtoms,
        atomsRange,
        processedElement,
        useExactXn,
        ppm=ppm,
        ppmCorrectionPosMode=ppmCorrectionPosMode,
        ppmCorrectionNegMode=ppmCorrectionNegMode,
        adducts=useAdducts,
        pwMaxSet=pwMaxSet,
        pwValSet=pwValSet,
        nCores=nCores,
    )

    # Get list of added columns
    smCol = "SFs"
    annotationColumns = [
        f"{smCol}_CHO",
        f"{smCol}_CHOS",
        f"{smCol}_CHOP",
        f"{smCol}_CHON",
        f"{smCol}_CHONP",
        f"{smCol}_CHOPS",
        f"{smCol}_CHONS",
        f"{smCol}_CHONPS",
        f"{smCol}_all",
        f"{smCol}_CHO_count",
        f"{smCol}_CHOS_count",
        f"{smCol}_CHOP_count",
        f"{smCol}_CHON_count",
        f"{smCol}_CHONP_count",
        f"{smCol}_CHOPS_count",
        f"{smCol}_CHONS_count",
        f"{smCol}_CHONPS_count",
        f"{smCol}_all_count",
    ]

    # Overwrite the input sheet with annotated results
    logging.info(f"Overwriting sheet '{sheet_name}' with sum formula annotations")
    plDB.set_table(sheet_name, results_df)

    # Create formula-focused transposed sheet (one row per formula hit, mirroring the _Compounds sheet)
    sf_sheet_name = f"{sheet_name}_SumFormulas"
    if sf_compound_hits:
        logging.info(f"Creating sum formula compound sheet with {len(sf_compound_hits)} hits")
        schema_overrides = {
            "MassErrorPPM": pl.Float64,
            "MassErrorMass": pl.Float64,
            "Feature_RT": pl.Float64,
            "Feature_MZ": pl.Float64,
            # Feature_M can hold comma-separated neutral masses so keep it as a string
            "Feature_M": pl.Utf8,
            "Feature_Relative_peakarea_in_group": pl.Float64,
            "Feature_Average_peakarea": pl.Float64,
        }
        sf_hits_df = pl.DataFrame(
            sf_compound_hits,
            schema_overrides=schema_overrides,
            infer_schema_length=len(sf_compound_hits),
        )
        sort_cols = [c for c in ["Element_Class", "SumFormula", "Feature_Num"] if c in sf_hits_df.columns]
        if sort_cols:
            sf_hits_df = sf_hits_df.sort(sort_cols)
        logging.info(f"Saving sum formula compound sheet to: {sf_sheet_name}")
        plDB.set_table(sf_sheet_name, sf_hits_df)
    else:
        logging.info("No sum formula hits found, skipping compound-focused sheet")

    plDB.close()

    logging.info(f"Sum formula generation completed. Added {len(annotationColumns)} columns")
    return annotationColumns
