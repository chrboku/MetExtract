"""
Generates the "help" sheet that documents the structure of the step 2 processing results
(the "<experiment>_grouped.xlsx" file produced by bracketing / re-integration / convolution /
annotation).

The sheet is a simple 3-column table (name, description, detailed_description) that is meant
to be human-readable directly in Excel. It is organized into sections (sheets overview, common
columns, annotation columns, sample-specific columns, and a detailed section for the database
annotation columns added to the "5_Annotated" sheet).

To extend this documentation (e.g. because a new sheet or column was added elsewhere in the
code), simply add a new entry to the appropriate list below (SHEET_DESCRIPTIONS,
COMMON_COLUMNS, ANNOTATION_COLUMNS, SAMPLE_COLUMNS or ANNOTATED_DATABASES_DETAIL) or register an
entirely new section by adding a tuple to SECTIONS.
"""

import polars as pl

HELP_SHEET_NAME = "help"
HELP_COLUMNS = ["name", "description", "detailed_description"]


def _entry(name, description, detailed_description):
    return {"name": name, "description": description, "detailed_description": detailed_description}


# ---------------------------------------------------------------------------
# Section 1: sheets present in the results Excel file
# ---------------------------------------------------------------------------
SHEET_DESCRIPTIONS = [
    _entry("To better view this help, select the columns A, B, and C and resize their width to 400px. Then select columns B and C and click Home -> 'Wrap text'.", "", ""),
    _entry("-----------------", "-----------------", "-----------------"),
    _entry("", "", ""),
    _entry(
        "help",
        "This sheet.",
        "Documentation of all sheets and columns contained in this results file. Generated automatically whenever step 2 processing (bracketing / re-integration / convolution / annotation) is run.",
    ),
    _entry(
        "5_Annotated",
        "Final results: detected metabolic features relatively quantified and annotated with database hits and/or generated sum formulas.",
        "The last available feature table (from '4_Convoluted', '3_Reintegrated' or '2_StatColumns', "
        "whichever completed most recently) with annotation columns appended: 'DBs_*' database search "
        "hits and/or 'SFs_*' generated sum-formula candidates. See the dedicated section below for "
        "details on the database annotation columns. This is normally the primary results sheet used "
        "for further analysis.",
    ),
    _entry(
        "5_Annotated_Compounds",
        "One row per database hit (compound-centric view of '5_Annotated').",
        "Flattened view of the database search results: each row is a single compound hit (database "
        "name, compound name, formula, mass, RT, match error, ...) together with the feature it was "
        "found in ('Feature_*' columns). Makes it easy to filter/sort hits by compound rather than by "
        "feature. See the dedicated section below for column details.",
    ),
    _entry(
        "5_Annotated_SumFormulas",
        "One row per generated sum-formula hit (formula-centric view of '5_Annotated').",
        "Flattened view of the generated sum formula candidates: each row is a single formula hit for an element combination (CHO, CHON, CHOS, ...) together with the feature it belongs to ('Feature_*' columns).",
    ),
    _entry(
        "5_Annotated_MSMS",
        "One row per MS/MS spectrum hit (spectrum-centric view of '5_Annotated').",
        "Flattened view of the MS/MS spectra search results: each row is a single MS/MS spectrum hit "
        "with 'Library'/'Score'/'Matched_Fragments'/'Compound_Name'/'Compound_ID' describing the match "
        "itself; auto-resolved 'SMILES'/'InChI'/'InChIKey'/'Formula'/'TheoreticalMass' and "
        "'Library_Instrument'/'Library_Fragmentation_Mode'/'Library_Collision_Energy'/'Library_RT'/"
        "'Library_Precursor_MZ' describing the matched reference (library) spectrum; "
        "'Sample_File'/'Scan_RT'/'Scan_Precursor_MZ'/'Scan_Fragmentation_Mode'/'Scan_Collision_Energy' "
        "describing the experimental scan the match was found in; and 'Feature_*' columns identifying "
        "the feature it was found in.",
    ),
    _entry(
        "0_sampleStats",
        "Per-sample summary statistics collected during bracketing.",
        "One row per processed sample file with summary information (e.g. number of chromatographic peaks/feature pairs) collected while building '1_Bracketed'.",
    ),
    _entry(
        "1_Bracketed",
        "Feature pairs bracketed (grouped) across samples by m/z.",
        "Output of the bracketing step. Contains all detected native/labeled feature pairs (one row, each identified by a separate value in the column 'Num'), with accurate mass (within the configured ppm tolerance) and retention time across all processed files. Sample-specific columns hold the per-file peak information.",
    ),
    _entry(
        "2_StatColumns",
        "Bracketed features with per-sample-group detection statistics added.",
        "Same rows as '1_Bracketed', with one additional column per defined sample group "
        "('<GroupName>_Stat') counting in how many files of that group the feature was found. Features "
        "found in fewer files than the group's configured minimum are removed (see "
        "'2_StatColumns_Omitted' / '2_StatColumns_FalsePositives').",
    ),
    _entry(
        "2_StatColumns_Omitted",
        "Features removed because they were not found often enough.",
        "Rows from '2_StatColumns' that did not reach the minimum number of detections configured for any sample group and were therefore excluded from further processing.",
    ),
    _entry(
        "2_StatColumns_FalsePositives",
        "Features flagged and removed as likely false positives.",
        "Rows from '2_StatColumns' that were marked (via the sample group settings, 'removeAsFalsePositive') as false positives and were therefore excluded from further processing.",
    ),
    _entry(
        "3_Reintegrated",
        "Bracketed features with missed peaks re-integrated in all samples.",
        "Output of the re-integration step. For each feature, chromatographic peaks that were not detected in a given file (e.g. below the peak-picking threshold) are re-integrated directly from the raw data using the feature's expected m/z and RT, filling in missing per-file values.",
    ),
    _entry(
        "4_Convoluted",
        "Features grouped into metabolite groups (OGroups) by peak-shape/ratio correlation.",
        "Output of the convolution/grouping step. Feature pairs that co-elute and correlate in peak shape (and, optionally, native:labeled ratio) across samples are merged into metabolite groups ('OGroup'). Adds the 'Relative_peakarea_in_group' and 'Average_peakarea' columns.",
    ),
    _entry(
        "4_Convoluted_doublePeaks",
        "Features flagged as double/split peaks during convolution.",
        "Rows moved out of '4_Convoluted' because they were identified as duplicate (split) chromatographic peaks of an already grouped feature.",
    ),
    _entry(
        "Parameters",
        "Processing parameters used to generate this results file.",
        "Key/value log of the settings used for individual-file processing and bracketing (version, date, UUID, and all relevant parameter values), kept for traceability/reproducibility.",
    ),
    _entry(
        "DB_info",
        "Log messages from the database import/search step.",
        "Diagnostic messages produced while importing the compound database file(s) used for annotation (number of imported/skipped entries, parsing errors, SMILES/sum-formula mismatches, ...).",
    ),
    _entry(
        "MSMS_info",
        "Log messages from the MSMS matching step.",
        "Diagnostic messages produced while importing the MS/MS spectral library file(s) used for annotation (number of imported/skipped spectra, parsing errors, ...).",
    ),
    _entry(
        "MSMS_info",
        "Summary of loaded MS/MS spectral libraries.",
        "Lists the MS/MS spectral library files that were loaded for MS/MS matching, with the number of spectra found in each file and a breakdown by detected polarity/ion mode (e.g., Positive/Negative). Helpful for diagnosing missing or unexpectedly empty libraries.",
    ),
    _entry(
        "__dTypes__",
        "Internal: column data types for all sheets.",
        "Internal bookkeeping sheet used by MetExtract II (PolarsDB) to persist the polars column data types of every other sheet across save/load cycles. Not intended for manual use.",
    ),
]


# ---------------------------------------------------------------------------
# Section 2: columns common to 1_Bracketed / 2_StatColumns / 3_Reintegrated /
# 4_Convoluted / 5_Annotated
# ---------------------------------------------------------------------------
COMMON_COLUMNS = [
    _entry("", "", ""),
    _entry("-----------------", "-----------------", "-----------------"),
    _entry(
        "Num",
        "Unique feature (feature pair) number.",
        "Primary key identifying a bracketed native/labeled feature pair across the whole results file; used to cross-reference between sheets (e.g. 'Feature_Num' in the '5_Annotated_Compounds'/'5_Annotated_SumFormulas' sheets). Cannot be used to reference features in individual samples.",
    ),
    _entry(
        "OGroup",
        "Metabolite group ID.",
        "Identifier of the metabolite group ('OGroup') this feature was assigned to during convolution "
        "(peak-shape/ratio correlation across samples). Rows sharing the same OGroup are considered "
        "isotopologues/adducts/fragments of the same underlying metabolite. Empty before the "
        "convolution step has run.",
    ),
    _entry(
        "Relative_peakarea_in_group",
        "Relative abundance rank within the metabolite group.",
        "Ratio of this feature's average peak area to the most abundant feature in the same OGroup (1.0 = most abundant member of the group). Added by the convolution step.",
    ),
    _entry(
        "Average_peakarea",
        "Mean peak area across all sample files.",
        "Average of the native peak area for this feature across all files in which it was detected. Added by the convolution step.",
    ),
    _entry(
        "Identity / Comment",
        "User-editable identification/comment fields.",
        "Free-text fields intended for manual annotation/curation of the feature; not populated automatically.",
    ),
    _entry(
        "Ionisation_Mode",
        "Ionisation polarity ('+' or '-').",
        "Ionisation polarity in which the feature pair was detected.",
    ),
    _entry(
        "RT",
        "Retention time (minutes).",
        "Mean apex retention time of the native peak, used for bracketing/matching features across samples.",
    ),
    _entry(
        "MZ",
        "Native m/z.",
        "Mean accurate mass-to-charge ratio of the native (unlabeled) form of the feature pair.",
    ),
    _entry(
        "Charge",
        "Charge state.",
        "Detected/assumed charge state (loading) of the ion derived from the carbon isotopolog pattern.",
    ),
    _entry(
        "Xn",
        "Number of labeled atoms.",
        "Number of atoms of the labeling element (e.g. 13C) incorporated, i.e. the difference between the native and labeled isotopologue.",
    ),
    _entry(
        "N_found_Samples",
        "Number of files the feature was originally detected in.",
        "Count of individual sample files in which this feature pair was detected by peak picking (before re-integration of missed peaks).",
    ),
    _entry(
        "Ion",
        "Assigned adduct(s).",
        "Adduct annotation(s) assigned to the feature (e.g. [M+H]+, [M+Na]+), comma-separated if several were found compatible.",
    ),
    _entry(
        "Loss",
        "Assigned in-source neutral loss / fragment description(s).",
        "Neutral-loss / in-source fragment annotation(s), comma-separated, describing how this feature relates to a putative parent ion.",
    ),
    _entry(
        "M",
        "Neutral monoisotopic mass candidate(s).",
        "Calculated neutral mass(es) of the underlying metabolite based on the assigned adduct(s), comma-separated if several candidates exist.",
    ),
    _entry(
        "Other_Isotopologs",
        "Other isotopologue peaks associated with this feature.",
        "Additional isotopologue peaks (e.g. M+1, M+2, M'-1, M'+1) detected together with the native/labeled pair.",
    ),
    _entry(
        "L_MZ",
        "Labeled m/z.",
        "Accurate mass-to-charge ratio of the labeled form of the feature pair (native m/z + Xn * label mass shift).",
    ),
    _entry(
        "D_MZ",
        "Mass difference between the native and labeled peak.",
        "Mass shift between the native and labeled peak of this feature pair (L_MZ - MZ).",
    ),
    _entry(
        "RT_Range",
        "Retention time spread across samples/replicates.",
        "Minimum-maximum range of apex retention times observed for this feature across all files, used to assess RT consistency.",
    ),
    _entry(
        "MZ_Range",
        "m/z spread across samples/replicates.",
        "Minimum-maximum range of m/z values observed for this feature across all files, used to assess mass accuracy consistency.",
    ),
    _entry(
        "PeakScalesNL",
        "Deprecated: Wavelet peak scales used for native/labeled peak detection.",
        "Deprecated: Chromatographic peak-picking wavelet scale(s) at which the native and labeled peaks were detected.",
    ),
    _entry(
        "ScanEvent",
        "MS scan event / filter line used.",
        "Instrument scan event (filter string) associated with the polarity of this feature (from the configured positive/negative scan event settings).",
    ),
    _entry(
        "Tracer",
        "Configured isotope tracer.",
        "Name/description of the isotope tracer (labelling experiment) used to generate this results file.",
    ),
    _entry(
        "doublePeak",
        "Double/split-peak marker.",
        "Flag (0/1) set when the feature was identified as a duplicate/split chromatographic peak of another already-grouped feature; such rows are moved to the corresponding '..._doublePeaks' sheet.",
    ),
    _entry(
        "<GroupName>_Stat",
        "Number of detections within the sample group '<GroupName>'.",
        "One such column per user-defined sample group. Counts the number of files belonging to that sample group in which this feature was found. Added by the statistics step ('2_StatColumns').",
    ),
]


# ---------------------------------------------------------------------------
# Section 3: annotation columns added to 5_Annotated (overview - see the
# dedicated "5_Annotated - database annotation details" section further below
# for the DBs_* columns)
# ---------------------------------------------------------------------------
ANNOTATION_COLUMNS = [
    _entry("", "", ""),
    _entry("-----------------", "-----------------", "-----------------"),
    _entry(
        "DBs_<database>_count",
        "Number of mass-only database hits (Level 3 annotation) for this feature in '<database>'.",
        "See the '5_Annotated - Database annotation details' section below.",
    ),
    _entry(
        "DBs_<database>",
        "Mass-only database hits (Level 3 annotation) for this feature in '<database>' (JSON list).",
        "See the '5_Annotated - Database annotation details' section below.",
    ),
    _entry(
        "DBs_RT_<database>_count",
        "Number of mass+RT database hits (Level 3 annotation) for this feature in '<database>'.",
        "See the '5_Annotated - Database annotation details' section below.",
    ),
    _entry(
        "DBs_RT_<database>",
        "Mass+RT database hits (Level 3 annotation) for this feature in '<database>' (JSON list, sorted by RT error).",
        "See the '5_Annotated - Database annotation details' section below.",
    ),
    _entry(
        "SFs_<elements> (e.g. SFs_CHO, SFs_CHON, SFs_CHONPS, SFs_all)",
        "Generated sum-formula candidates restricted to the given element combination.",
        "Comma/semicolon-separated list of candidate sum formulas matching this feature's m/z within the configured ppm tolerance, using only the listed elements ('SFs_all' allows any of the configured elements). Added by the sum-formula generation step (Seven Golden Rules).",
    ),
    _entry(
        "SFs_<elements>_count (e.g. SFs_CHO_count, SFs_all_count)",
        "Number of generated sum-formula candidates for the given element combination.",
        "Count of entries in the corresponding 'SFs_<elements>' column.",
    ),
]


# ---------------------------------------------------------------------------
# Section 4: sample-specific columns (one set of columns per sample file,
# prefixed with the sample's file name)
# ---------------------------------------------------------------------------
SAMPLE_COLUMNS = [
    _entry("", "", ""),
    _entry("-----------------", "-----------------", "-----------------"),
    _entry(
        "<file>_Found",
        "Whether the feature was detected or re-integrated in this sample.",
        "Detection flag for the native/labeled peak pair in this specific sample file.",
    ),
    _entry(
        "<file>_Area_N / <file>_Area_L",
        "Peak area (native / labeled) in this sample.",
        "Integrated chromatographic peak area of the native (N) / labeled (L) peak in this sample file (Trapezoid integration from peak start and end using numpy trapezoid).",
    ),
    _entry(
        "<file>_Abundance_N / <file>_Abundance_L",
        "Peak apex abundance (native / labeled) in this sample.",
        "Apex intensity (normalized abundance) of the native (N) / labeled (L) peak in this sample file.",
    ),
    _entry(
        "<file>_peaksCorr",
        "Peak-shape correlation between native and labeled peak in this sample.",
        "Pearson correlation coefficient between the native and labeled extracted-ion chromatograms around the peak apex, used to validate that both peaks represent the same compound.",
    ),
    _entry(
        "<file>_SNR_N / <file>_SNR_L",
        "Signal-to-noise ratio (native / labeled) in this sample.",
        "Signal-to-noise ratio of the native (N) / labeled (L) chromatographic peak as determined by the peak picker.",
    ),
    _entry(
        "<file>_N_startRT / _N_apexRT / _N_endRT",
        "Native peak retention time boundaries in this sample.",
        "Start, apex and end retention time of the native chromatographic peak in this sample file.",
    ),
    _entry(
        "<file>_L_startRT / _L_apexRT / _L_endRT",
        "Labeled peak retention time boundaries in this sample.",
        "Start, apex and end retention time of the labeled chromatographic peak in this sample file.",
    ),
    _entry(
        "<file>_N_FWHM / <file>_L_FWHM",
        "Peak full width at half maximum (native / labeled) in this sample.",
        "Full width at half maximum of the native (N) / labeled (L) chromatographic peak, a measure of peak sharpness.",
    ),
    _entry(
        "<file>_N_PeakWidth / <file>_L_PeakWidth",
        "Peak width (native / labeled) in this sample.",
        "Total width (end - start RT) of the native (N) / labeled (L) chromatographic peak.",
    ),
    _entry(
        "<file>_N_ApexToFlankFactor / <file>_L_ApexToFlankFactor",
        "Apex-to-flank intensity ratio (native / labeled) in this sample.",
        "Ratio between the apex intensity and the flank (edge) intensity of the peak, used as a peak quality/shape filter criterion.",
    ),
    _entry(
        "<file>_N_ApexToFlankIncrease / <file>_L_ApexToFlankIncrease",
        "Apex-to-flank intensity increase (native / labeled) in this sample.",
        "Absolute increase in intensity from the flank to the apex of the peak, used as a peak quality/shape filter criterion.",
    ),
    _entry(
        "<file>_peaksRatio",
        "Native:labeled intensity ratio in this sample.",
        "Ratio of native to labeled peak intensity in this sample file, used to validate isotope labeling consistency (e.g. via 'useRatio'/'minRatio'/'maxRatio' settings).",
    ),
    _entry(
        "<file>_artificialEICLShift",
        "Artificial EIC shift applied to the labeled trace, if any.",
        "Retention-time shift artificially applied to the labeled chromatogram extraction window in this sample (used to compensate for known chromatographic isotope effects).",
    ),
    _entry(
        "<file>_FID",
        "Feature ID in the raw per-file results.",
        "Internal feature identifier referencing the corresponding chromatographic peak pair in the individual-file processing results ('.meii'/database) for this sample. This is used to cross-reference between the per-file results and the bracketed results file, but independent of the Num column.",
    ),
    _entry(
        "<file>_GroupID",
        "Feature group ID in the raw per-file results.",
        "Internal per-file group identifier of the chromatographic peak pair, as assigned during individual-file processing (before bracketing). This is used to cross-reference between the per-file results and the bracketed results file, but independent of the OGroup column.",
    ),
    _entry(
        "<file>_isotopologRatios",
        "Isotope pattern abundance ratios in this sample (JSON).",
        "JSON-encoded isotopologue abundance ratios recorded for this feature pair in this sample file, used for isotope-pattern based validation/annotation.",
    ),
    _entry(
        "<file>_isoArea",
        "Peak areas of the isotopologs between M and M' in this sample (JSON).",
        "JSON-encoded dictionary ('M+x' -> area) of the individual isotopolog peak areas quantified between M and M' (from M+1 up to M'-1) in this sample file. Only isotopologs with an area of at least 1% relative to the less abundant of M/M' are included.",
    ),
    _entry(
        "<file>_isoEnrichment",
        "Isotopic enrichment of this feature pair in this sample (JSON).",
        "JSON-encoded dictionary with keys 'M' (enrichment calculated from M and the highest detected M+x isotopolog) and 'M'' (enrichment calculated from M' and M'-1), following the same calculation as the 'Isotopic enrichment' tool.",
    ),
    _entry(
        "<file>_isoCount",
        "Number of isotopologs detected between M and M' in this sample.",
        "Count of isotopologs (M+1 up to M'-1) with an area of at least 1% relative to the less abundant of M/M' in this sample file.",
    ),
]


# ---------------------------------------------------------------------------
# Section 5: detailed documentation of the "5_Annotated" sheet's database
# annotation columns, plus the related "5_Annotated_Compounds" sheet.
# ---------------------------------------------------------------------------
ANNOTATED_DATABASES_DETAIL = [
    _entry("", "", ""),
    _entry("-----------------", "-----------------", "-----------------"),
    _entry(
        "DBs_<database>_count",
        "Number of mass-only hits found in database '<database>'.",
        "Count of database entries whose (adduct-corrected) mass or m/z matched this feature's MZ/M within the configured ppm tolerance (and, if 'checkXnInHits' is set, a compatible Xn/label count). One pair of 'DBs_<database>_count'/'DBs_<database>' columns is created per imported database file.",
    ),
    _entry(
        "DBs_<database>",
        "Mass-only hits found in database '<database>' (JSON list of strings).",
        "JSON-encoded list of formatted hit descriptions, each of the form: '(Name: <compound name>, Type: <hit type>, Num: <db entry id>, Formula: <sum formula>, RT: <db RT, if known>, MassErrorPPM: <ppm>, MassErrorMass: <Da>, Additional information: <db-specific extra fields>)'.",
    ),
    _entry(
        "DBs_RT_<database>_count",
        "Number of hits in '<database>' that also matched the expected retention time.",
        "Count of the subset of 'DBs_<database>' hits for which the database entry has a known RT within 'rtError' of the feature's RT (only populated when 'useRt' is enabled and the database entry provides an RT).",
    ),
    _entry(
        "DBs_RT_<database>",
        "RT-matched hits in '<database>', sorted by RT error (JSON list of strings).",
        "JSON-encoded list of formatted hit descriptions (same fields as 'DBs_<database>', plus 'RT delta'/'RTError'), sorted by ascending absolute RT error - the first entry is the closest RT match.",
    ),
    _entry(
        "5_Annotated_Compounds columns",
        "Columns of the compound-centric sheet ('5_Annotated_Compounds').",
        "One row per individual database hit across all features, with columns: 'DB_Name' (database the "
        "hit came from), 'DB_Num' (entry id in that database), 'DB_CompoundName', 'DB_SumFormula', "
        "'DB_Mass', 'DB_RT_min', 'DB_MZ', 'DB_Polarity', 'HitType' (mass-only vs. mass+RT), "
        "'MatchErrorPPM', 'MatchErrorMass' describing the database hit itself; 'DB_Info_<field>' columns "
        "(one per additional field present in the imported database file, e.g. 'DB_Info_SMILES', "
        "'DB_Info_InChI') holding that database entry's extra metadata; and 'Feature_Num', "
        "'Feature_OGroup', 'Feature_RT', 'Feature_MZ', 'Feature_Xn', 'Feature_Ionisation_Mode', "
        "'Feature_Charge', 'Feature_M', 'Feature_Ion', 'Feature_Loss', "
        "'Feature_Relative_peakarea_in_group', 'Feature_Average_peakarea' identifying the feature "
        "(row in '5_Annotated') the hit was found for. Sorted by DB_Name, DB_CompoundName, Feature_Num.",
    ),
]


# ---------------------------------------------------------------------------
# Registry of sections, in the order they should appear in the help sheet.
# To add a new section, append a (title, entries) tuple here.
# ---------------------------------------------------------------------------
SECTIONS = [
    ("Sheets in this results file", SHEET_DESCRIPTIONS),
    ("Common columns (present in 1_Bracketed / 2_StatColumns / 3_Reintegrated / 4_Convoluted / 5_Annotated)", COMMON_COLUMNS),
    ("Annotation columns (added to 5_Annotated)", ANNOTATION_COLUMNS),
    ("Sample-specific columns (repeated once per sample file, prefixed with the file name)", SAMPLE_COLUMNS),
    ("5_Annotated - Database annotation details", ANNOTATED_DATABASES_DETAIL),
]


def generate_help_sheet() -> pl.DataFrame:
    """
    Build the "help" sheet describing the structure of the step 2 processing results file.

    Returns:
        polars.DataFrame with columns ["name", "description", "detailed_description"], containing
        one section-title row (description/detailed_description left empty) followed by the
        section's entries, for every section registered in SECTIONS.
    """
    rows = []
    for title, entries in SECTIONS:
        rows.append(_entry(f"## {title}", "", ""))
        rows.extend(entries)
        rows.append(_entry("", "", ""))

    # Drop the trailing blank separator row after the last section
    if rows and rows[-1]["name"] == "":
        rows.pop()

    return pl.DataFrame(rows, schema=HELP_COLUMNS)
