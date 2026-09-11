from __future__ import absolute_import, division, print_function
import base64
import datetime
import hashlib
import json
import logging
import os
import statistics
import time
import uuid
import xml.etree.ElementTree as _ET
from collections import OrderedDict, defaultdict
from math import isnan, log
from multiprocessing import Manager, Pool
from operator import itemgetter
from pickle import loads as pickle_loads
import numpy as np
import polars as pl
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from . import HCA_general, exportAsFeatureML
from .Chromatogram import Chromatogram
from .MZHCA import HierarchicalClustering, cutTreeSized
from .PolarsDB import PolarsDB
from .metaboliteGrouping import split_group_by_relative_abundance
from .utils import ChromPeakPair
from .utils import (
    Bunch,
    CallBackMethod,
    corr,
    getDBSuffix,
    getSubGraphs,
    mean,
    natSort,
    sd,
)
from .XICAlignment import XICAlignment
from .utils import mapArrayToRefTimes, get_app_folder
from .resultsPostProcessing.helpSheet import generate_help_sheet


def log10(x):
    return log(x, 10)


# ---------------------------------------------------------------------------
# Per-file sample-stats cache helpers
# ---------------------------------------------------------------------------


def _get_stats_cache_dir():
    cache_dir = os.path.join(get_app_folder(), "sampleStats")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _load_cached_stats(filepath, scan_event_key):
    """Return cached stats dict for *filepath* or None if missing / stale."""
    try:
        fp = os.path.abspath(filepath)
        key = hashlib.md5(fp.encode("utf-8")).hexdigest()
        cache_path = os.path.join(_get_stats_cache_dir(), f"{key}.json")
        if not os.path.exists(cache_path):
            return None
        mtime = os.path.getmtime(fp)
        with open(cache_path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("_mtime") != mtime:
            return None
        if cached.get("_scan_event_key") != scan_event_key:
            return None
        return cached.get("data")
    except Exception:
        return None


def _save_cached_stats(filepath, data, scan_event_key):
    """Write stats dict to the per-file cache."""
    try:
        fp = os.path.abspath(filepath)
        key = hashlib.md5(fp.encode("utf-8")).hexdigest()
        cache_path = os.path.join(_get_stats_cache_dir(), f"{key}.json")
        mtime = os.path.getmtime(fp)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"_mtime": mtime, "_scan_event_key": scan_event_key, "data": data}, fh)
    except Exception as e:
        logging.debug(f"Could not save stats cache for {filepath}: {e}")


# HELPER METHOD for writing first page of PDF (unused)
def _writeFirstPage(pdf, groupSizePPM, maxTimeDeviation, align, nPolynom):
    curHeight = 780

    pdf.drawString(60, curHeight, "Max. Group Size")
    pdf.drawString(200, curHeight, "%.1f" % groupSizePPM)
    curHeight -= 20

    pdf.drawString(60, curHeight, "Max. Time Deviation")
    pdf.drawString(200, curHeight, "%.2f seconds" % maxTimeDeviation)
    curHeight -= 20

    pdf.drawString(60, curHeight, "Align")
    pdf.drawString(200, curHeight, "%s" % ("yes" if align else "no"))
    curHeight -= 20

    if align:
        pdf.drawString(60, curHeight, "Polynom Order")
        pdf.drawString(200, curHeight, "%d" % nPolynom)
        curHeight -= 20

    pdf.showPage()


def _aggregate_isotopolog_ratios(grouped_peaks):
    aggregated = defaultdict(list)

    for _, chrom_peak in grouped_peaks:
        for isotopolog, value in getattr(chrom_peak, "isotopologRatios", {}).items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            aggregated[isotopolog].append(numeric_value)

    if not aggregated:
        return None

    summary = OrderedDict()
    for isotopolog in sorted(aggregated):
        values = aggregated[isotopolog]
        mean_value = mean(values)
        summary[isotopolog] = {
            "mean": mean_value,
            "rsd": (sd(values) / mean_value) if len(values) > 1 and mean_value != 0 else None,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "n": len(values),
        }

    return json.dumps(summary)


# store used configuration to DB file
def writeConfigToDB(
    db,
    align,
    file,
    groupSizePPM,
    maxLoading,
    maxTimeDeviation,
    xCounts,
    nPolynom,
    negativeScanEvent,
    positiveScanEvent,
    meVersion,
):
    db.insert_row("config", {"key": "MEVersion", "value": str(meVersion)})
    db.insert_row("config", {"key": "FPBRACK_xCounts", "value": str(xCounts)})
    db.insert_row("config", {"key": "FPBRACK_groupSizePPM", "value": str(groupSizePPM)})
    db.insert_row("config", {"key": "FPBRACK_positiveScanEvent", "value": str(positiveScanEvent)})
    db.insert_row("config", {"key": "FPBRACK_negativeScanEvent", "value": str(negativeScanEvent)})
    db.insert_row("config", {"key": "FPBRACK_maxTimeDeviation", "value": str(maxTimeDeviation)})
    db.insert_row("config", {"key": "FPBRACK_maxLoading", "value": str(maxLoading)})
    db.insert_row("config", {"key": "FPBRACK_file", "value": str(file)})
    db.insert_row("config", {"key": "FPBRACK_align", "value": str(align)})
    db.insert_row("config", {"key": "FPBRACK_nPolynom", "value": str(nPolynom)})


def _get_mzml_metadata(filepath):
    """Extract startTimeStamp and specific cvParams from an mzML header via streaming XML parsing."""
    result = {"startTimeStamp": None, "MS:1000073": None, "MS:1000079": None}
    ns = {"mzml": "http://psi.hupo.org/ms/mzml"}
    try:
        for event, elem in _ET.iterparse(filepath, events=("start",)):
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "run":
                result["startTimeStamp"] = elem.get("startTimeStamp")
            elif tag == "cvParam":
                acc = elem.get("accession", "")
                if acc in ("MS:1000073", "MS:1000079"):
                    result[acc] = elem.get("name", acc)
            # Stop after the first spectrum to avoid reading the whole file
            elif tag == "spectrum":
                break
    except Exception as e:
        logging.debug(f"Could not parse mzML metadata from {filepath}: {e}")
    return result


def _percentile_stats(diffs):
    """Return a dict of percentile statistics for a list of float values."""
    if not diffs:
        return {k: None for k in ("min", "p10", "p25", "median", "mean", "p75", "p90", "max", "sd")}
    arr = np.array(diffs, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def _intensity_percentile_stats(intensities):
    """Return intensity distribution stats (min, p10, p25, median, p75, p90–p99, max)."""
    _keys = ("min", "p10", "p25", "median", "p75", "p90", "p91", "p92", "p93", "p94", "p95", "p96", "p97", "p98", "p99", "max")
    if not intensities:
        return {k: None for k in _keys}
    arr = np.array(intensities, dtype=float)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p91": float(np.percentile(arr, 91)),
        "p92": float(np.percentile(arr, 92)),
        "p93": float(np.percentile(arr, 93)),
        "p94": float(np.percentile(arr, 94)),
        "p95": float(np.percentile(arr, 95)),
        "p96": float(np.percentile(arr, 96)),
        "p97": float(np.percentile(arr, 97)),
        "p98": float(np.percentile(arr, 98)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_sample_stats(all_files, positiveScanEvent=None, negativeScanEvent=None):
    """Compute per-file sample statistics for a list of raw LC-MS/MS files.

    Returns a list of dicts, one per file, with keys:
        file, ms1_pos, ms1_neg, ms2_pos, ms2_neg, last_rt,
        ms1_timediff_*, ms2_timediff_*,
        startTimeStamp, MS:1000073, MS:1000079
    """
    rows = []
    for filepath in all_files:
        filepath = str(filepath).replace("\\", "/")
        # v3: compute MS1 intensity stats from actual signal data (intensity_list)
        scan_event_key = f"{positiveScanEvent}|{negativeScanEvent}|v6"

        # Try cache first
        cached_row = _load_cached_stats(filepath, scan_event_key)
        if cached_row is not None:
            rows.append(cached_row)
            continue

        try:
            chrom = Chromatogram()
            chrom.parse_file(filepath, ignoreCharacterData=False)
        except Exception as e:
            logging.warning(f"compute_sample_stats: could not parse {filepath}: {e}")
            rows.append({"file": os.path.basename(filepath)})
            continue

        pos_se = positiveScanEvent if positiveScanEvent and positiveScanEvent != "None" else None
        neg_se = negativeScanEvent if negativeScanEvent and negativeScanEvent != "None" else None

        # MS1 scan counts per polarity / selected scan event
        ms1_pos_scans = [s for s in chrom.MS1_list if s.polarity == "+" and (pos_se is None or s.filter_line == pos_se)]
        ms1_neg_scans = [s for s in chrom.MS1_list if s.polarity == "-" and (neg_se is None or s.filter_line == neg_se)]
        ms1_pos = len(ms1_pos_scans)
        ms1_neg = len(ms1_neg_scans)

        # MS2 scan counts per polarity
        ms2_pos = sum(1 for s in chrom.MS2_list if s.polarity == "+")
        ms2_neg = sum(1 for s in chrom.MS2_list if s.polarity == "-")

        # Adjacent MS1 time differences (selected scan events only)
        ms1_times = sorted(s.retention_time for s in chrom.MS1_list if (pos_se is None or s.filter_line == pos_se or s.polarity == "+") and (neg_se is None or s.filter_line == neg_se or s.polarity == "-"))
        if pos_se:
            ms1_times = sorted(s.retention_time for s in chrom.MS1_list if s.filter_line == pos_se or s.filter_line == neg_se)
        ms1_diffs = [ms1_times[i + 1] - ms1_times[i] for i in range(len(ms1_times) - 1)]

        # Adjacent MS2 time differences
        ms2_times = sorted(s.retention_time for s in chrom.MS2_list)
        ms2_diffs = [ms2_times[i + 1] - ms2_times[i] for i in range(len(ms2_times) - 1)]

        # Last scan RT (max of all scans)
        all_rts = [s.retention_time for s in chrom.MS1_list] + [s.retention_time for s in chrom.MS2_list]
        last_rt = max(all_rts) / 60.0 if all_rts else None

        ms1_stats = _percentile_stats(ms1_diffs)
        ms2_stats = _percentile_stats(ms2_diffs)

        # Collect ALL individual signal intensities from the actual peak lists
        ms1_signalInt_pos = _intensity_percentile_stats([log10(v) for s in ms1_pos_scans for v in s.intensity_list.tolist()])
        ms1_signalInt_neg = _intensity_percentile_stats([log10(v) for s in ms1_neg_scans for v in s.intensity_list.tolist()])

        # mzML-specific metadata
        mzml_meta = {"startTimeStamp": None, "MS:1000073": None, "MS:1000079": None}
        if filepath.lower().endswith(".mzml"):
            mzml_meta = _get_mzml_metadata(filepath)

        row = {
            "file": os.path.basename(filepath),
            "startTimeStamp": mzml_meta["startTimeStamp"],
            "ms1_pos": ms1_pos,
            "ms1_neg": ms1_neg,
            "ms2_pos": ms2_pos,
            "ms2_neg": ms2_neg,
            "last_rt_min": last_rt,
            "ms1_dt_min": ms1_stats["min"],
            "ms1_dt_p10": ms1_stats["p10"],
            "ms1_dt_p25": ms1_stats["p25"],
            "ms1_dt_median": ms1_stats["median"],
            "ms1_dt_mean": ms1_stats["mean"],
            "ms1_dt_p75": ms1_stats["p75"],
            "ms1_dt_p90": ms1_stats["p90"],
            "ms1_dt_max": ms1_stats["max"],
            "ms1_dt_sd": ms1_stats["sd"],
            "ms2_dt_min": ms2_stats["min"],
            "ms2_dt_p10": ms2_stats["p10"],
            "ms2_dt_p25": ms2_stats["p25"],
            "ms2_dt_median": ms2_stats["median"],
            "ms2_dt_mean": ms2_stats["mean"],
            "ms2_dt_p75": ms2_stats["p75"],
            "ms2_dt_p90": ms2_stats["p90"],
            "ms2_dt_max": ms2_stats["max"],
            "ms2_dt_sd": ms2_stats["sd"],
            "ms1_signalInt_pos_min": ms1_signalInt_pos["min"],
            "ms1_signalInt_pos_p10": ms1_signalInt_pos["p10"],
            "ms1_signalInt_pos_p25": ms1_signalInt_pos["p25"],
            "ms1_signalInt_pos_median": ms1_signalInt_pos["median"],
            "ms1_signalInt_pos_p75": ms1_signalInt_pos["p75"],
            "ms1_signalInt_pos_p90": ms1_signalInt_pos["p90"],
            "ms1_signalInt_pos_p91": ms1_signalInt_pos["p91"],
            "ms1_signalInt_pos_p92": ms1_signalInt_pos["p92"],
            "ms1_signalInt_pos_p93": ms1_signalInt_pos["p93"],
            "ms1_signalInt_pos_p94": ms1_signalInt_pos["p94"],
            "ms1_signalInt_pos_p95": ms1_signalInt_pos["p95"],
            "ms1_signalInt_pos_p96": ms1_signalInt_pos["p96"],
            "ms1_signalInt_pos_p97": ms1_signalInt_pos["p97"],
            "ms1_signalInt_pos_p98": ms1_signalInt_pos["p98"],
            "ms1_signalInt_pos_p99": ms1_signalInt_pos["p99"],
            "ms1_signalInt_pos_max": ms1_signalInt_pos["max"],
            "ms1_signalInt_neg_min": ms1_signalInt_neg["min"],
            "ms1_signalInt_neg_p10": ms1_signalInt_neg["p10"],
            "ms1_signalInt_neg_p25": ms1_signalInt_neg["p25"],
            "ms1_signalInt_neg_median": ms1_signalInt_neg["median"],
            "ms1_signalInt_neg_p75": ms1_signalInt_neg["p75"],
            "ms1_signalInt_neg_p90": ms1_signalInt_neg["p90"],
            "ms1_signalInt_neg_p91": ms1_signalInt_neg["p91"],
            "ms1_signalInt_neg_p92": ms1_signalInt_neg["p92"],
            "ms1_signalInt_neg_p93": ms1_signalInt_neg["p93"],
            "ms1_signalInt_neg_p94": ms1_signalInt_neg["p94"],
            "ms1_signalInt_neg_p95": ms1_signalInt_neg["p95"],
            "ms1_signalInt_neg_p96": ms1_signalInt_neg["p96"],
            "ms1_signalInt_neg_p97": ms1_signalInt_neg["p97"],
            "ms1_signalInt_neg_p98": ms1_signalInt_neg["p98"],
            "ms1_signalInt_neg_p99": ms1_signalInt_neg["p99"],
            "ms1_signalInt_neg_max": ms1_signalInt_neg["max"],
            "MS:1000073": mzml_meta["MS:1000073"],
            "MS:1000079": mzml_meta["MS:1000079"],
        }
        _save_cached_stats(filepath, row, scan_event_key)
        rows.append(row)
    return rows


# bracket results
def bracketResults(
    indGroups,
    xCounts,
    groupSizePPM,
    positiveScanEvent=None,
    negativeScanEvent=None,
    maxTimeDeviation=0.36 * 60,
    maxLoading=1,
    file="./results.tsv",
    align=True,
    nPolynom=1,
    pwMaxSet=None,
    pwValSet=None,
    pwTextSet=None,
    meVersion="",
    generalProcessingParams=Bunch(),
    start=0,
):
    excel_file = file.replace(".tsv", ".xlsx")
    if os.path.exists(excel_file):
        os.remove(excel_file)

    # create results ParquetDB tables
    try:
        writePDF = False
        # used for debug purposes
        colos = [colors.red]

        # XICAlignment groups peaks by RT proximity (PTW warping removed)
        xicAlign = XICAlignment()

        results = []

        # generate results for each input file
        i = 1
        for key in natSort(indGroups.keys()):
            files = indGroups[key]

            for ident in files:
                conn = None
                curs = None
                if os.path.isfile(ident + getDBSuffix()):
                    conn = PolarsDB(ident + getDBSuffix())
                    curs = conn.cursor()
                fname = ident
                if ".mzxml" in ident.lower():
                    fname = ident[max(ident.rfind("/") + 1, ident.rfind("\\") + 1) : ident.lower().rfind(".mzxml")]
                if ".mzml" in ident.lower():
                    fname = ident[max(ident.rfind("/") + 1, ident.rfind("\\") + 1) : ident.lower().rfind(".mzml")]
                b = Bunch(
                    filePath=ident,
                    fileName=fname,
                    featurePairs=[],
                    conn=conn,
                    curs=curs,
                )

                results.append(b)
            i += 1

        # Initialize data collection for Excel output
        excel_data = OrderedDict()
        excel_data["Num"] = []
        excel_data["OGroup"] = []
        excel_data["Relative_peakarea_in_group"] = []
        excel_data["Average_peakarea"] = []
        excel_data["Identity"] = []
        excel_data["Comment"] = []
        excel_data["Ionisation_Mode"] = []
        excel_data["RT"] = []
        excel_data["MZ"] = []
        excel_data["Charge"] = []
        excel_data["Xn"] = []
        excel_data["N_found_Samples"] = []
        excel_data["Ion"] = []
        excel_data["Loss"] = []
        excel_data["M"] = []
        excel_data["Other_Isotopologs"] = []
        excel_data["L_MZ"] = []
        excel_data["D_MZ"] = []
        excel_data["RT_Range"] = []
        excel_data["MZ_Range"] = []
        excel_data["PeakScalesNL"] = []
        excel_data["ScanEvent"] = []
        excel_data["Tracer"] = []

        excel_data_dTypes = OrderedDict()
        excel_data_dTypes["Num"] = pl.Utf8
        excel_data_dTypes["OGroup"] = pl.Utf8
        excel_data_dTypes["Relative_peakarea_in_group"] = pl.Float64
        excel_data_dTypes["Average_peakarea"] = pl.Float64
        excel_data_dTypes["Comment"] = pl.Utf8
        excel_data_dTypes["Identity"] = pl.Utf8
        excel_data_dTypes["MZ"] = pl.Float64
        excel_data_dTypes["L_MZ"] = pl.Float64
        excel_data_dTypes["D_MZ"] = pl.Float64
        excel_data_dTypes["MZ_Range"] = pl.Utf8
        excel_data_dTypes["RT"] = pl.Float64
        excel_data_dTypes["RT_Range"] = pl.Utf8
        excel_data_dTypes["PeakScalesNL"] = pl.Utf8
        excel_data_dTypes["Xn"] = pl.Utf8
        excel_data_dTypes["N_found_Samples"] = pl.Int64
        excel_data_dTypes["Charge"] = pl.Int64
        excel_data_dTypes["ScanEvent"] = pl.Utf8
        excel_data_dTypes["Ionisation_Mode"] = pl.Utf8
        excel_data_dTypes["Tracer"] = pl.Utf8
        excel_data_dTypes["Ion"] = pl.Utf8
        excel_data_dTypes["Loss"] = pl.Utf8
        excel_data_dTypes["M"] = pl.Utf8
        excel_data_dTypes["Other_Isotopologs"] = pl.Utf8

        # Add file-specific columns
        for res in results:
            fname = res.fileName
            if ".mzxml" in fname.lower():
                fname = fname[max(fname.rfind("/") + 1, fname.rfind("\\") + 1) : fname.lower().rfind(".mzxml")]
            if ".mzml" in fname.lower():
                fname = fname[max(fname.rfind("/") + 1, fname.rfind("\\") + 1) : fname.lower().rfind(".mzml")]

            excel_data[fname + "_Found"] = []
            excel_data[fname + "_Area_N"] = []
            excel_data[fname + "_Area_L"] = []
            excel_data[fname + "_Abundance_N"] = []
            excel_data[fname + "_Abundance_L"] = []
            excel_data[fname + "_peaksCorr"] = []
            excel_data[fname + "_SNR_N"] = []
            excel_data[fname + "_SNR_L"] = []
            excel_data[fname + "_N_startRT"] = []
            excel_data[fname + "_N_apexRT"] = []
            excel_data[fname + "_N_endRT"] = []
            excel_data[fname + "_L_startRT"] = []
            excel_data[fname + "_L_apexRT"] = []
            excel_data[fname + "_L_endRT"] = []
            excel_data[fname + "_N_FWHM"] = []
            excel_data[fname + "_L_FWHM"] = []
            excel_data[fname + "_N_PeakWidth"] = []
            excel_data[fname + "_L_PeakWidth"] = []
            excel_data[fname + "_N_ApexToFlankFactor"] = []
            excel_data[fname + "_L_ApexToFlankFactor"] = []
            excel_data[fname + "_N_ApexToFlankIncrease"] = []
            excel_data[fname + "_L_ApexToFlankIncrease"] = []
            excel_data[fname + "_peaksRatio"] = []
            excel_data[fname + "_artificialEICLShift"] = []
            excel_data[fname + "_FID"] = []
            excel_data[fname + "_GroupID"] = []
            excel_data[fname + "_isotopologRatios"] = []
            excel_data[fname + "_isoArea"] = []
            excel_data[fname + "_isoEnrichment"] = []
            excel_data[fname + "_isoCount"] = []

            excel_data_dTypes[fname + "_Found"] = pl.Utf8
            excel_data_dTypes[fname + "_Area_N"] = pl.Utf8
            excel_data_dTypes[fname + "_Area_L"] = pl.Utf8
            excel_data_dTypes[fname + "_Abundance_N"] = pl.Utf8
            excel_data_dTypes[fname + "_Abundance_L"] = pl.Utf8
            excel_data_dTypes[fname + "_peaksCorr"] = pl.Utf8
            excel_data_dTypes[fname + "_SNR_N"] = pl.Utf8
            excel_data_dTypes[fname + "_SNR_L"] = pl.Utf8
            excel_data_dTypes[fname + "_N_startRT"] = pl.Utf8
            excel_data_dTypes[fname + "_N_apexRT"] = pl.Utf8
            excel_data_dTypes[fname + "_N_endRT"] = pl.Utf8
            excel_data_dTypes[fname + "_L_startRT"] = pl.Utf8
            excel_data_dTypes[fname + "_L_apexRT"] = pl.Utf8
            excel_data_dTypes[fname + "_L_endRT"] = pl.Utf8
            excel_data_dTypes[fname + "_N_FWHM"] = pl.Utf8
            excel_data_dTypes[fname + "_L_FWHM"] = pl.Utf8
            excel_data_dTypes[fname + "_N_PeakWidth"] = pl.Utf8
            excel_data_dTypes[fname + "_L_PeakWidth"] = pl.Utf8
            excel_data_dTypes[fname + "_N_ApexToFlankFactor"] = pl.Utf8
            excel_data_dTypes[fname + "_L_ApexToFlankFactor"] = pl.Utf8
            excel_data_dTypes[fname + "_N_ApexToFlankIncrease"] = pl.Utf8
            excel_data_dTypes[fname + "_L_ApexToFlankIncrease"] = pl.Utf8
            excel_data_dTypes[fname + "_peaksRatio"] = pl.Utf8
            excel_data_dTypes[fname + "_artificialEICLShift"] = pl.Utf8
            excel_data_dTypes[fname + "_FID"] = pl.Utf8
            excel_data_dTypes[fname + "_GroupID"] = pl.Utf8
            excel_data_dTypes[fname + "_isotopologRatios"] = pl.Utf8
            excel_data_dTypes[fname + "_isoArea"] = pl.Utf8
            excel_data_dTypes[fname + "_isoEnrichment"] = pl.Utf8
            excel_data_dTypes[fname + "_isoCount"] = pl.Utf8

        excel_data["doublePeak"] = []
        excel_data_dTypes["doublePeak"] = pl.Int64

        # get ionisation modes scan events
        ionModes = {}
        if positiveScanEvent != "None":
            ionModes["+"] = positiveScanEvent
        if negativeScanEvent != "None":
            ionModes["-"] = negativeScanEvent
        assert len(ionModes) > 0

        totalChromPeaks = 0
        totalChromPeaksProc = 0
        tracersDeltaMZ = {}

        ## TODO this used the last open file. May be invalid. improve
        isMetabolisationExperiment = False
        for res in results:
            if res.conn is not None:
                config_table = res.conn.get_table("config")
                if not config_table.is_empty():
                    config_rows = config_table.filter(pl.col("key") == "metabolisationExperiment")
                    if len(config_rows) > 0:
                        isMetabolisationExperiment = str(config_rows[0, "value"]).lower() == "true"
                        break

        ## TODO this used the last open file. May be invalid. improve
        if isMetabolisationExperiment:
            for res in results:
                if res.conn is not None:
                    tracer_table = res.conn.get_table("tracerConfiguration")
                    if not tracer_table.is_empty():
                        for row in tracer_table.iter_rows(named=True):
                            tracerName = str(row["name"])
                            dmz = float(row["deltaMZ"])

                            if tracerName not in tracersDeltaMZ:
                                tracersDeltaMZ[tracerName] = dmz

                            if abs(tracersDeltaMZ[tracerName] - dmz) < 0.00001:
                                logging.warning("Warning: Tracers have not been configured identical in all measurement files")
        else:
            ## TODO this used the last open file. May be invalid. improve
            for res in results:
                if res.conn is not None:
                    config_table = res.conn.get_table("config")
                    if not config_table.is_empty():
                        config_rows = config_table.filter(pl.col("key") == "xOffset")
                        if len(config_rows) > 0:
                            dmz = float(config_rows[0, "value"])

                            if "FLE" not in tracersDeltaMZ:
                                tracersDeltaMZ["FLE"] = dmz

                            if abs(tracersDeltaMZ["FLE"] - dmz) < 0.00001:
                                logging.warning("Warning: Tracers have not been configured identical in all measurement files")

        grp = 1
        if writePDF:
            pdf = canvas.Canvas(file + "_0.pdf")
        if writePDF:
            _writeFirstPage(pdf, groupSizePPM, maxTimeDeviation, align, nPolynom)
        xy = 1

        curNum = 1

        xCounts = set()
        for res in results:
            if res.conn is not None:
                chrom_peaks = res.conn.get_table("chromPeaks")
                if not chrom_peaks.is_empty():
                    unique_xcounts = chrom_peaks.select(pl.col("xcount")).unique()
                    for row in unique_xcounts.iter_rows(named=True):
                        xCounts.add(row["xcount"])
        xCounts = sorted(list(xCounts))

        # xCountshelp = []
        # a=xCounts.replace(" ", "").replace(";", ",").split(",")
        # for j in a:
        #     if "-" in j:
        #         xCountshelp.extend(range(int(j[0:j.find("-")]), int(j[j.find("-")+1:len(j)])+1))
        #     else:
        #         xCountshelp.append(int(j))
        # xCounts=sorted(list(set(xCountshelp)))

        # prefetch number of bracketing steps for the progress bar
        totalThingsToDo = 0
        for ionMode in ionModes:
            for tracer in tracersDeltaMZ:
                for xCount in xCounts:
                    for cLoading in range(maxLoading, 0, -1):
                        totalThingsToDo += 1

        doneSoFar = 0
        if pwMaxSet is not None:
            pwMaxSet.put(Bunch(mes="max", val=totalThingsToDo))

        # bracket data
        # process ionModes, tracers, xCount and loading separately
        for ionMode in ionModes:
            scanEvent = ionModes[ionMode]
            for xCount in xCounts:
                for cLoading in range(maxLoading, 0, -1):
                    # check each processed LC-HRMS file if the used data processing parameters match
                    for res in results:
                        res.featurePairs = []

                        if pwTextSet is not None:
                            # Log time used for bracketing
                            elapsed = (time.time() - start) / 60.0
                            hours = ""
                            if elapsed >= 60.0:
                                hours = "%d hours " % (elapsed // 60)
                            mins = "%.2f mins" % (elapsed % 60.0)
                            pwTextSet.put(
                                Bunch(
                                    mes="text",
                                    val="<p align='right' >%s%s elapsed</p>\n\n\n\nProcessing \n  (Ionmode: %s, XCount: %s, Charge: %d) \n  File: %s"
                                    % (
                                        hours,
                                        mins,
                                        ionMode,
                                        xCount,
                                        cLoading,
                                        res.fileName,
                                    ),
                                )
                            )
                        if res.conn is not None:
                            chrom_peaks = res.conn.get_table("chromPeaks")
                            feature_groups = res.conn.get_table("featureGroupFeatures")
                            tracer_config = res.conn.get_table("tracerConfiguration")

                            if not chrom_peaks.is_empty():
                                # Filter chromPeaks
                                filtered_peaks = chrom_peaks.filter((pl.col("ionMode") == ionMode) & (pl.col("xcount") == xCount) & (pl.col("Loading") == cLoading))

                                # Join with feature groups if available
                                if not feature_groups.is_empty():
                                    filtered_peaks = filtered_peaks.join(feature_groups.select(["fID", "fGroupID"]), left_on="id", right_on="fID", how="left")
                                else:
                                    filtered_peaks = filtered_peaks.with_columns(pl.lit(None).alias("fGroupID"))

                                # Join with tracer configuration if available
                                if not tracer_config.is_empty():
                                    filtered_peaks = filtered_peaks.join(tracer_config.select(["id", "name"]), left_on="tracer", right_on="id", how="left").rename({"name": "tracerName"})
                                else:
                                    filtered_peaks = filtered_peaks.with_columns(pl.lit(None).alias("tracerName"))

                                for row in filtered_peaks.iter_rows(named=True):
                                    try:
                                        cp = ChromPeakPair(
                                            id=row["id"],
                                            tracer=row["tracer"],
                                            NPeakCenter=row["NPeakCenter"],
                                            NPeakCenterMin=row["NPeakCenterMin"],
                                            NPeakScale=row["NPeakScale"],
                                            NSNR=row["NSNR"],
                                            NPeakArea=row["NPeakArea"],
                                            mz=row["mz"],
                                            lmz=row["lmz"],
                                            tmz=row["tmz"],
                                            xCount=str(row["xcount"]),
                                            LPeakCenter=row["LPeakCenter"],
                                            LPeakCenterMin=row["LPeakCenterMin"],
                                            LPeakScale=row["LPeakScale"],
                                            LSNR=row["LSNR"],
                                            LPeakArea=row["LPeakArea"],
                                            loading=row["Loading"],
                                            fGroupID=row.get("fGroupID"),
                                            tracerName=row.get("tracerName"),
                                            ionMode=str(row["ionMode"]),
                                            NPeakAbundance=float(row["NPeakAbundance"]),
                                            LPeakAbundance=float(row["LPeakAbundance"]),
                                            NBorderLeft=float(row["NBorderLeft"]),
                                            NBorderRight=float(row["NBorderRight"]),
                                            LBorderLeft=float(row["LBorderLeft"]),
                                            LBorderRight=float(row["LBorderRight"]),
                                            N_startRT=float(row["N_startRT"]) if row.get("N_startRT") is not None else float(row["NPeakCenterMin"]),
                                            N_endRT=float(row["N_endRT"]) if row.get("N_endRT") is not None else float(row["NPeakCenterMin"]),
                                            L_startRT=float(row["L_startRT"]) if row.get("L_startRT") is not None else float(row["LPeakCenterMin"]),
                                            L_endRT=float(row["L_endRT"]) if row.get("L_endRT") is not None else float(row["LPeakCenterMin"]),
                                            N_FWHM=float(row["N_FWHM"]) if row.get("N_FWHM") is not None else 0.0,
                                            L_FWHM=float(row["L_FWHM"]) if row.get("L_FWHM") is not None else 0.0,
                                            N_Baseline=float(row["N_Baseline"]) if row.get("N_Baseline") is not None else 0.0,
                                            L_Baseline=float(row["L_Baseline"]) if row.get("L_Baseline") is not None else 0.0,
                                            artificialEICLShift=int(row["artificialEICLShift"]),
                                            peaksRatio=float(row["peaksRatio"]),
                                            peaksCorr=float(row["peaksCorr"]),
                                            isotopologRatios=pickle_loads(base64.b64decode(row["isotopologRatios"])) if row.get("isotopologRatios") else {},
                                            isoAreas=pickle_loads(base64.b64decode(row["isoAreas"])) if row.get("isoAreas") else {},
                                            isoEnrichment=pickle_loads(base64.b64decode(row["isoEnrichment"])) if row.get("isoEnrichment") else {},
                                            isoCount=row.get("isoCount") or 0,
                                        )

                                        assert cp.ionMode in ionModes.keys()
                                        res.featurePairs.append(cp)
                                    except TypeError as err:
                                        print(
                                            "  TypeError in file %s, id %s, (Ionmode: %s, XCount: %s, Charge: %d)"
                                            % (
                                                str(res.fileName),
                                                str(row["id"]),
                                                ionMode,
                                                xCount,
                                                cLoading,
                                            ),
                                            str(err),
                                        )
                                    except Exception:
                                        print(
                                            "  some general error in file %s, id %s, (Ionmode: %s, XCount: %s, Charge: %d)"
                                            % (
                                                str(res.fileName),
                                                str(row["id"]),
                                                ionMode,
                                                xCount,
                                                cLoading,
                                            )
                                        )

                        totalChromPeaks = totalChromPeaks + len(res.featurePairs)

                    if pwTextSet is not None:
                        # Log time used for bracketing
                        elapsed = (time.time() - start) / 60.0
                        hours = ""
                        if elapsed >= 60.0:
                            hours = "%d hours " % (elapsed // 60)
                        mins = "%.2f mins" % (elapsed % 60.0)
                        pwTextSet.put(
                            Bunch(
                                mes="text",
                                val="<p align='right' >%s%s elapsed</p>\n\n\n\nClustering \n  (Ionmode: %s, XCount: %s, Charge: %d)" % (hours, mins, ionMode, xCount, cLoading),
                            )
                        )

                    for tracer in tracersDeltaMZ:
                        ## TODO this does not work correctly. improve
                        if pwValSet is not None:
                            pwValSet.put(Bunch(mes="value", val=doneSoFar))
                        doneSoFar += 1

                        # get all results that match current bracketing criteria
                        chromPeaksAct = 0
                        for res in results:
                            chromPeaksAct = chromPeaksAct + len([chromPeak.mz for chromPeak in res.featurePairs])

                        allMZs = []
                        for res in results:
                            allMZs.extend([chromPeak.mz for chromPeak in res.featurePairs])

                        # cluster all current results with HCA
                        allMZs = sorted(allMZs)
                        lastMZ = None
                        u = 0
                        curAllMZs = []

                        ## pre-separate detected feature pairs to improve speed of HCA
                        while len(allMZs) > 0:
                            procCurSet = False
                            if u < len(allMZs) and (lastMZ is None or (allMZs[u] - lastMZ) < 3 * (groupSizePPM * allMZs[u] / 1e6)):
                                curAllMZs.append(allMZs[u])
                                lastMZ = allMZs[u]
                                u += 1
                            else:
                                lastMZ = None
                                allMZs = allMZs[u:]
                                u = 0
                                procCurSet = True

                            if procCurSet or len(allMZs) == u:
                                if len(curAllMZs) > 0:
                                    hc = HierarchicalClustering(
                                        curAllMZs,
                                        dist=lambda x, y: x.getValue() - y.getValue(),
                                        val=lambda x: x,
                                        mean=lambda x, y: x / y,
                                        add=lambda x, y: x + y,
                                    )

                                    # cut HCA tree in subclusters
                                    for n in cutTreeSized(hc.getTree(), groupSizePPM):
                                        try:
                                            lowMZ = min([kid.getValue() for kid in n.getKids()])
                                            minMZ = max([kid.getValue() for kid in n.getKids()])

                                            maxMZInGroup = lowMZ

                                            partChromPeaks = {}
                                            partXICs = {}
                                            toDel = {}
                                            allChromPeaks = []
                                            for res in results:
                                                chromPeaks = []
                                                toDel[res.filePath] = set()
                                                for i in range(len(res.featurePairs)):
                                                    chromPeak = res.featurePairs[i]
                                                    if chromPeak.mz <= minMZ:
                                                        toDel[res.filePath].add(i)
                                                        chromPeaks.append(chromPeak)
                                                        allChromPeaks.append(chromPeak)
                                                        maxMZInGroup = max(
                                                            maxMZInGroup,
                                                            chromPeak.mz,
                                                        )

                                                if align:
                                                    xics = []
                                                    # get EICs of current subCluster
                                                    for chromPeak in chromPeaks:
                                                        xics_table = res.conn.get_table("XICs")
                                                        chrom_peaks_table = res.conn.get_table("chromPeaks")

                                                        if not xics_table.is_empty() and not chrom_peaks_table.is_empty():
                                                            # Join XICs with chromPeaks
                                                            joined = chrom_peaks_table.filter(pl.col("id") == chromPeak.id).join(xics_table, left_on="eicID", right_on="id", how="inner")

                                                            if len(joined) > 0:
                                                                row = joined[0]
                                                                scanCount = float(row["scanCount"])
                                                                times = [float(r) for r in str(row["times"]).split(";")]
                                                                xicL = [float(r) for r in str(row["xicL"]).split(";")]
                                                                eicID = int(row["eicid"])
                                                                xics.append(
                                                                    [
                                                                        scanCount,
                                                                        times,
                                                                        xicL,
                                                                        eicID,
                                                                    ]
                                                                )
                                                partChromPeaks[res.filePath] = chromPeaks
                                                if align:
                                                    partXICs[res.filePath] = xics

                                            xy = xy + 1
                                            if (xy % 50) == 0:
                                                if writePDF:
                                                    pdf.save()
                                                    pdf = canvas.Canvas(file + "_%d.pdf" % xy)
                                                    _writeFirstPage(
                                                        pdf,
                                                        groupSizePPM,
                                                        maxTimeDeviation,
                                                        align,
                                                        nPolynom,
                                                    )
                                                    print(
                                                        "\r   new pdf %d metabolic features.." % xy,
                                                    )

                                            # debug purposes: create PDF for current subcluster
                                            if writePDF:
                                                drawing = Drawing(500, 350)

                                                lp = LinePlot()
                                                lp.x = 50
                                                lp.y = 30
                                                lp.height = 350
                                                lp.width = 500

                                                dd = []

                                                tr = 0
                                                pdf.drawString(
                                                    40,
                                                    810,
                                                    "Tracer: %s IonMode: %s" % (tracer, ionMode),
                                                )
                                                pdf.drawString(
                                                    40,
                                                    795,
                                                    "MZ: (min: %f - max: %f (tolerated: %f ppm: %.1f)) "
                                                    % (
                                                        lowMZ,
                                                        maxMZInGroup,
                                                        minMZ,
                                                        (maxMZInGroup - lowMZ) * 1000000 / lowMZ,
                                                    ),
                                                )
                                                pdf.drawString(
                                                    40,
                                                    780,
                                                    "xCount: %s Z: %d Feature pairs: %d"
                                                    % (
                                                        xCount,
                                                        cLoading,
                                                        len(partXICs),
                                                    ),
                                                )
                                                pdj = []
                                                pdjm = []
                                                for r in partXICs.keys():
                                                    xics = partXICs[r]
                                                    for xic in xics:
                                                        pdj.append(xic[2])
                                                        pdjm.append(xic[1])
                                                        dd.append(
                                                            [
                                                                (
                                                                    xic[1][i] / 60.0,
                                                                    xic[2][i],
                                                                )
                                                                for i in range(0, len(xic[1]))
                                                            ]
                                                        )
                                                        tr = tr + 1
                                                pdf.drawString(
                                                    40,
                                                    765,
                                                    "Aligning %d EICs" % (tr),
                                                )

                                                lp.data = dd

                                                i = 0
                                                for k in partXICs.keys():
                                                    lp.lines[i].strokeColor = colos[i % len(colos)]
                                                    lp.lines[i].strokeWidth = 0.01
                                                    i = i + 1

                                                lp.joinedLines = 1
                                                drawing.add(lp)
                                                renderPDF.draw(drawing, pdf, 10, 380)

                                                alignedEICs = xicAlign.getAligendXICs(
                                                    pdj,
                                                    pdjm,
                                                    align=align,
                                                    nPolynom=nPolynom,
                                                )[0]
                                                drawing = Drawing(500, 350)
                                                lp = LinePlot()
                                                lp.x = 50
                                                lp.y = 30
                                                lp.height = 350
                                                lp.width = 500

                                                dd = []

                                                tr = 0
                                                for xic in alignedEICs:
                                                    dd.append([(i / 60.0, xic[i]) for i in range(0, len(xic))])
                                                    tr = tr + 1

                                                lp.data = dd

                                                i = 0
                                                for k in range(len(alignedEICs)):
                                                    lp.lines[i].strokeColor = colos[i % len(colos)]
                                                    lp.lines[i].strokeWidth = 0.01
                                                    i = i + 1

                                                lp.joinedLines = 1
                                                drawing.add(lp)
                                                renderPDF.draw(drawing, pdf, 10, 0)

                                                pdf.showPage()

                                            eics = []
                                            peaks = []
                                            scantimes = []
                                            dd = []
                                            do = []
                                            for k in partChromPeaks.keys():
                                                for i in range(len(partChromPeaks[k])):
                                                    do.append(k)
                                                    dd.append(partChromPeaks[k][i].mz)
                                                    if align:
                                                        eics.append(partXICs[k][i][2])
                                                        scantimes.append(partXICs[k][i][1])
                                                    peaks.append([partChromPeaks[k][i]])

                                            # optional: align EICs; get bracketed chromatographic peaks
                                            aligned = xicAlign.alignXIC(
                                                eics,
                                                peaks,
                                                scantimes,
                                                align=align,
                                                maxTimeDiff=maxTimeDeviation,
                                                nPolynom=nPolynom,
                                            )
                                            aligned = [(x[0][0], int(x[0][1])) for x in aligned]

                                            if writePDF:
                                                uz = 800
                                                o = 0
                                                for peak in peaks:
                                                    for p in peak:
                                                        pdf.drawString(
                                                            40,
                                                            uz,
                                                            "%35s: %d %.2f -> %d %d"
                                                            % (
                                                                do[o][(1 + do[o].rfind("/")) :],
                                                                int(p.NPeakCenter),
                                                                p.NPeakCenterMin / 60.0,
                                                                int(aligned[o][0]),
                                                                aligned[o][1],
                                                            ),
                                                        )
                                                    uz -= 20
                                                    o += 1

                                            maxGroup = max(aligned, key=itemgetter(1))[1]
                                            minGroup = min(aligned, key=itemgetter(1))[1]

                                            groupedChromPeaks = []
                                            groupedChromPeaksAVGMz = []
                                            groupedChromPeaksAVGLMz = []
                                            groupedChromPeaksAVGTMz = []
                                            groupedChromPeaksAVGTimes = []
                                            groupedChromPeaksAVGNPeakScale = []
                                            groupedChromPeaksAVGLPeakScale = []

                                            for i in range(maxGroup + 1):
                                                groupedChromPeaks.append({})
                                                groupedChromPeaksAVGMz.append([])
                                                groupedChromPeaksAVGLMz.append([])
                                                groupedChromPeaksAVGTMz.append([])
                                                groupedChromPeaksAVGTimes.append([])
                                                groupedChromPeaksAVGNPeakScale.append([])
                                                groupedChromPeaksAVGLPeakScale.append([])

                                            # calculate average values (RT and mz) for feature pairs in the sub-subclusters
                                            j = 0
                                            for k in partChromPeaks.keys():
                                                for i in range(len(partChromPeaks[k])):
                                                    if k not in groupedChromPeaks[aligned[j][1]]:
                                                        groupedChromPeaks[aligned[j][1]][k] = []
                                                    groupedChromPeaks[aligned[j][1]][k].append(
                                                        (
                                                            aligned[j],
                                                            partChromPeaks[k][i],
                                                        )
                                                    )
                                                    groupedChromPeaksAVGMz[aligned[j][1]].append(partChromPeaks[k][i].mz)
                                                    groupedChromPeaksAVGLMz[aligned[j][1]].append(partChromPeaks[k][i].lmz)
                                                    groupedChromPeaksAVGTMz[aligned[j][1]].append(partChromPeaks[k][i].tmz)
                                                    groupedChromPeaksAVGTimes[aligned[j][1]].append(partChromPeaks[k][i].NPeakCenterMin)
                                                    groupedChromPeaksAVGNPeakScale[aligned[j][1]].append(partChromPeaks[k][i].NPeakScale)
                                                    groupedChromPeaksAVGLPeakScale[aligned[j][1]].append(partChromPeaks[k][i].LPeakScale)

                                                    j = j + 1

                                            assert j == len(aligned)
                                            assert len(groupedChromPeaks) == len(groupedChromPeaksAVGMz) == len(groupedChromPeaksAVGTimes)

                                            # write results to data matrix and SQLite DB
                                            for i in range(minGroup, maxGroup + 1):
                                                if len(groupedChromPeaks[i]) > 0:
                                                    avgmz = sum(groupedChromPeaksAVGMz[i]) / len(groupedChromPeaksAVGMz[i])
                                                    avglmz = sum(groupedChromPeaksAVGLMz[i]) / len(groupedChromPeaksAVGLMz[i])
                                                    avgtmz = sum(groupedChromPeaksAVGTMz[i]) / len(groupedChromPeaksAVGTMz[i])
                                                    avgtime = sum(groupedChromPeaksAVGTimes[i]) / len(groupedChromPeaksAVGTimes[i])
                                                    avgNPeakScale = sum(groupedChromPeaksAVGNPeakScale[i]) / len(groupedChromPeaksAVGNPeakScale[i])
                                                    avgLPeakScale = sum(groupedChromPeaksAVGLPeakScale[i]) / len(groupedChromPeaksAVGLPeakScale[i])

                                                    # Append data to excel_data dictionary
                                                    excel_data["Num"].append(f"Fp.{curNum}")
                                                    excel_data["OGroup"].append(None)
                                                    excel_data["Relative_peakarea_in_group"].append(None)
                                                    excel_data["Average_peakarea"].append(None)
                                                    excel_data["Identity"].append(None)
                                                    excel_data["Comment"].append("")
                                                    excel_data["RT"].append(avgtime / 60.0)
                                                    excel_data["RT_Range"].append(
                                                        "%.3f - %.3f"
                                                        % (
                                                            min(groupedChromPeaksAVGTimes[i]) / 60.0,
                                                            max(groupedChromPeaksAVGTimes[i]) / 60.0,
                                                        )
                                                    )
                                                    excel_data["PeakScalesNL"].append(
                                                        "%.1f:%.1f"
                                                        % (
                                                            avgNPeakScale,
                                                            avgLPeakScale,
                                                        )
                                                    )
                                                    excel_data["ScanEvent"].append(scanEvent)
                                                    excel_data["Ionisation_Mode"].append(ionMode)
                                                    excel_data["Tracer"].append(str(tracer))
                                                    excel_data["MZ"].append(avgmz)
                                                    excel_data["L_MZ"].append(avglmz)
                                                    excel_data["D_MZ"].append(avgtmz)
                                                    excel_data["MZ_Range"].append(
                                                        "%.6f - %.6f"
                                                        % (
                                                            min(groupedChromPeaksAVGMz[i]),
                                                            max(groupedChromPeaksAVGMz[i]),
                                                        )
                                                    )
                                                    excel_data["Xn"].append(str(xCount))
                                                    excel_data["Charge"].append(cLoading)

                                                    excel_data["Ion"].append(None)
                                                    excel_data["Loss"].append(None)
                                                    excel_data["M"].append(None)
                                                    aggregated_isotopologs = []
                                                    for sample_peaks in groupedChromPeaks[i].values():
                                                        aggregated_isotopologs.extend(sample_peaks)
                                                    excel_data["Other_Isotopologs"].append(_aggregate_isotopolog_ratios(aggregated_isotopologs))

                                                    doublePeak = 0
                                                    nFoundSamples = 0
                                                    for j in range(len(results)):
                                                        res = results[j].filePath
                                                        fname = results[j].fileName
                                                        if ".mzxml" in fname.lower():
                                                            fname = fname[max(fname.rfind("/") + 1, fname.rfind("\\") + 1) : fname.lower().rfind(".mzxml")]
                                                        if ".mzml" in fname.lower():
                                                            fname = fname[max(fname.rfind("/") + 1, fname.rfind("\\") + 1) : fname.lower().rfind(".mzml")]

                                                        if res in groupedChromPeaks[i] and len(groupedChromPeaks[i][res]) > 0:
                                                            doublePeak += 1 if len(groupedChromPeaks[i][res]) > 1 else 0
                                                            nFoundSamples += 1

                                                            excel_data[fname + "_Found"].append(";".join(["Direct" for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_Area_N"].append(";".join([str(peak[1].NPeakArea) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_Area_L"].append(";".join([str(peak[1].LPeakArea) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_Abundance_N"].append(";".join([str(peak[1].NPeakAbundance) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_Abundance_L"].append(";".join([str(peak[1].LPeakAbundance) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_peaksCorr"].append(";".join([str(peak[1].peaksCorr) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_SNR_N"].append(";".join([str(peak[1].NSNR) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_SNR_L"].append(";".join([str(peak[1].LSNR) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_startRT"].append(";".join([str(getattr(peak[1], "N_startRT", peak[1].NPeakCenterMin) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_apexRT"].append(";".join([str(peak[1].NPeakCenterMin / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_endRT"].append(";".join([str(getattr(peak[1], "N_endRT", peak[1].NPeakCenterMin) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_startRT"].append(";".join([str(getattr(peak[1], "L_startRT", peak[1].LPeakCenterMin) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_apexRT"].append(";".join([str(peak[1].LPeakCenterMin / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_endRT"].append(";".join([str(getattr(peak[1], "L_endRT", peak[1].LPeakCenterMin) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_FWHM"].append(";".join([str(getattr(peak[1], "N_FWHM", 0.0) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_FWHM"].append(";".join([str(getattr(peak[1], "L_FWHM", 0.0) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_PeakWidth"].append(";".join([str((getattr(peak[1], "N_endRT", peak[1].NPeakCenterMin) - getattr(peak[1], "N_startRT", peak[1].NPeakCenterMin)) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_PeakWidth"].append(";".join([str((getattr(peak[1], "L_endRT", peak[1].LPeakCenterMin) - getattr(peak[1], "L_startRT", peak[1].LPeakCenterMin)) / 60.0) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_ApexToFlankFactor"].append(";".join([str(float(peak[1].NPeakAbundance) / max(getattr(peak[1], "N_Baseline", 0.0), 1.0)) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_ApexToFlankFactor"].append(";".join([str(float(peak[1].LPeakAbundance) / max(getattr(peak[1], "L_Baseline", 0.0), 1.0)) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_N_ApexToFlankIncrease"].append(";".join([str(float(peak[1].NPeakAbundance) - getattr(peak[1], "N_Baseline", 0.0)) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_L_ApexToFlankIncrease"].append(";".join([str(float(peak[1].LPeakAbundance) - getattr(peak[1], "L_Baseline", 0.0)) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_peaksRatio"].append(";".join([str(peak[1].peaksRatio) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_artificialEICLShift"].append(";".join([str(peak[1].artificialEICLShift) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_FID"].append(";".join([str(peak[1].id) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_GroupID"].append(";".join([str(peak[1].fGroupID) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_isotopologRatios"].append(";".join([json.dumps(getattr(peak[1], "isotopologRatios", {})) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_isoArea"].append(";".join([json.dumps(getattr(peak[1], "isoAreas", {})) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_isoEnrichment"].append(";".join([json.dumps(getattr(peak[1], "isoEnrichment", {})) for peak in groupedChromPeaks[i][res]]))
                                                            excel_data[fname + "_isoCount"].append(";".join([str(getattr(peak[1], "isoCount", 0)) for peak in groupedChromPeaks[i][res]]))

                                                        else:
                                                            excel_data[fname + "_Found"].append(None)
                                                            excel_data[fname + "_Area_N"].append(None)
                                                            excel_data[fname + "_Area_L"].append(None)
                                                            excel_data[fname + "_Abundance_N"].append(None)
                                                            excel_data[fname + "_Abundance_L"].append(None)
                                                            excel_data[fname + "_peaksCorr"].append(None)
                                                            excel_data[fname + "_SNR_N"].append(None)
                                                            excel_data[fname + "_SNR_L"].append(None)
                                                            excel_data[fname + "_N_startRT"].append(None)
                                                            excel_data[fname + "_N_apexRT"].append(None)
                                                            excel_data[fname + "_N_endRT"].append(None)
                                                            excel_data[fname + "_L_startRT"].append(None)
                                                            excel_data[fname + "_L_apexRT"].append(None)
                                                            excel_data[fname + "_L_endRT"].append(None)
                                                            excel_data[fname + "_N_FWHM"].append(None)
                                                            excel_data[fname + "_L_FWHM"].append(None)
                                                            excel_data[fname + "_N_PeakWidth"].append(None)
                                                            excel_data[fname + "_L_PeakWidth"].append(None)
                                                            excel_data[fname + "_N_ApexToFlankFactor"].append(None)
                                                            excel_data[fname + "_L_ApexToFlankFactor"].append(None)
                                                            excel_data[fname + "_N_ApexToFlankIncrease"].append(None)
                                                            excel_data[fname + "_L_ApexToFlankIncrease"].append(None)
                                                            excel_data[fname + "_peaksRatio"].append(None)
                                                            excel_data[fname + "_artificialEICLShift"].append(None)
                                                            excel_data[fname + "_FID"].append(None)
                                                            excel_data[fname + "_GroupID"].append(None)
                                                            excel_data[fname + "_isotopologRatios"].append(None)
                                                            excel_data[fname + "_isoArea"].append(None)
                                                            excel_data[fname + "_isoEnrichment"].append(None)
                                                            excel_data[fname + "_isoCount"].append(None)

                                                    excel_data["N_found_Samples"].append(nFoundSamples)

                                                    if doublePeak > 0:
                                                        excel_data["doublePeak"].append(doublePeak)
                                                    else:
                                                        excel_data["doublePeak"].append(None)

                                                    curNum = curNum + 1

                                            if writePDF:
                                                pdf.showPage()

                                        except Exception as e:
                                            import traceback

                                            traceback.print_exc()
                                            logging.error(str(traceback))

                                            logging.error("Error: %s" % (str(e)))
                                        grp = grp + 1

                                        for res in results:
                                            if res.filePath in toDel:
                                                toDel[res.filePath] = [a for a in toDel[res.filePath]]
                                                toDel[res.filePath].sort()
                                                toDel[res.filePath].reverse()
                                                for i in toDel[res.filePath]:
                                                    res.featurePairs.pop(i)
                                                    chromPeaksAct = chromPeaksAct - 1
                                                    totalChromPeaksProc = totalChromPeaksProc + 1

                                        if pwTextSet is not None:
                                            # Log time used for bracketing
                                            elapsed = (time.time() - start) / 60.0
                                            hours = ""
                                            if elapsed >= 60.0:
                                                hours = "%d hours " % (elapsed // 60)
                                            mins = "%.2f mins" % (elapsed % 60.0)
                                            pwTextSet.put(
                                                Bunch(
                                                    mes="text",
                                                    val="<p align='right' >%s%s elapsed</p>\n\n\n\nBracketing results\n%d feature pairs (%d individual pairs processed).. (Ionmode: %s, XCount: %s, Charge: %d)"
                                                    % (
                                                        hours,
                                                        mins,
                                                        curNum,
                                                        totalChromPeaksProc,
                                                        ionMode,
                                                        xCount,
                                                        cLoading,
                                                    ),
                                                )
                                            )
                                curAllMZs = []
        if writePDF:
            pdf.save()

        # Prepare metadata strings
        parameters_df = {"Parameter": [], "Value": []}
        parameters_df["Parameter"].append("# MetExtract II")
        parameters_df["Value"].append("")
        parameters_df["Parameter"].append("version")
        parameters_df["Value"].append(f"{meVersion}")
        parameters_df["Parameter"].append("# Execution")
        parameters_df["Value"].append("")
        parameters_df["Parameter"].append("Date")
        parameters_df["Value"].append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        parameters_df["Parameter"].append("UUID")
        parameters_df["Value"].append(f"{uuid.uuid4()}")

        parameters_df["Parameter"].append("# Individual files processing parameters")
        parameters_df["Value"].append("")

        for k, v in generalProcessingParams.__dict__.items():
            parameters_df["Parameter"].append(str(k))
            parameters_df["Value"].append(str(v))

        xCountsFmt = []
        try:
            for xn in sorted(xCounts):
                if xn - 1 in xCounts and xn + 1 in xCounts:
                    xCountsFmt.append("-")
                else:
                    if xn - 1 not in xCounts:
                        xCountsFmt.append(", ")
                    xCountsFmt.append(xn)
            xCountsFmt.pop(0)
            seen = set()
            seen_add = seen.add
            xCountsFmt = [x for x in xCountsFmt if not (x in seen or seen_add(x)) or x == ", "]
        except Exception:
            xCountsFmt = xCounts

        parameters_df["Parameter"].append("# Bracketing data processing parameters")
        parameters_df["Value"].append("")
        parameters_df["Parameter"].append("xCounts")
        parameters_df["Value"].append("".join([str(t) for t in xCountsFmt]))
        parameters_df["Parameter"].append("Group size ppm")
        parameters_df["Value"].append(f"{groupSizePPM}")
        parameters_df["Parameter"].append("max. time deviation")
        parameters_df["Value"].append(f"{maxTimeDeviation} min")
        parameters_df["Parameter"].append("max. loading")
        parameters_df["Value"].append(f"{maxLoading}")
        parameters_df["Parameter"].append("PTW alignment")
        parameters_df["Value"].append(f"{align}")
        if align:
            parameters_df["Parameter"].append("n polynom for alignment")
            parameters_df["Value"].append(f"{nPolynom}")
        parameters_df["Parameter"].append("")
        parameters_df["Value"].append("")
        parameters_df["Parameter"].append("")
        parameters_df["Value"].append("")

        params_df = pl.DataFrame(parameters_df)
        df = pl.DataFrame(excel_data, schema=excel_data_dTypes).sort("RT")

        excel_file = file.replace(".tsv", ".xlsx")
        plDB = PolarsDB(excel_file, format="xlsx")
        plDB.insert_table("help", generate_help_sheet())

        # Collect all file paths for sample stats
        all_files = []
        for key in natSort(indGroups.keys()):
            for ident in indGroups[key]:
                all_files.append(ident)
        sample_stats_rows = compute_sample_stats(all_files, positiveScanEvent, negativeScanEvent)
        if sample_stats_rows:
            # Build a uniform schema across all rows
            all_keys = list(sample_stats_rows[0].keys())
            stats_dict = {k: [row.get(k) for row in sample_stats_rows] for k in all_keys}
            sample_stats_df = pl.DataFrame(stats_dict)
            plDB.insert_table("0_sampleStats", sample_stats_df)

        plDB.insert_table("Parameters", params_df)
        plDB.insert_table("1_Bracketed", df)
        plDB.commit()
        plDB.close()

        exportAsFeatureML.convertMEMatrixToFeatureMLSepPolarities(excel_file, sheet_name="1_Bracketed", postfix="_ab")

    except Exception:
        import traceback

        traceback.print_exc()
        print(str(traceback))

        logging.error("Error during bracketing of files")

    finally:
        logging.info("Bracketing done...")


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


# store used configuration to DB file
def writeMetaboliteGroupingConfigToDB(db, minConnectionsInFiles, minConnectionRate, groups):
    db.insert_row("Parameters", {"Parameter": "MEGroups", "Value": f"{groups}"})
    db.insert_row(
        "Parameters",
        {"Parameter": "MEGROUP_minConnectionsInFiles", "Value": f"{minConnectionsInFiles}"},
    )
    db.insert_row("Parameters", {"Parameter": "MEGROUP_minConnectionRate", "Value": f"{minConnectionRate}"})


def getBordersFor(db, fID, file):
    ffp_table = db.get_table("foundfeaturepairs")
    if not ffp_table.is_empty():
        rows = ffp_table.filter((pl.col("file") == file) & (pl.col("resID") == fID))
        if len(rows) > 0:
            row = rows[0]
            return (float(row["NBorderLeft"]), float(row["NBorderRight"]), float(row["LBorderLeft"]), float(row["LBorderRight"]))
    return None


def getPeak(times, rt, borderleft, borderright):
    ## find best matching rt
    ind, tim = min(enumerate(times), key=lambda x: abs(x[1] - rt))
    minind = int(max(0, ind - int(borderleft)))
    maxind = int(min(len(times) - 1, ind + borderright))

    return (minind, maxind)


def getPeakByRT(times, start_rt, end_rt):
    """Find scan indices corresponding to start and end retention times."""
    if len(times) == 0:
        return (0, 0)
    minind = min(range(len(times)), key=lambda x: abs(times[x] - start_rt))
    maxind = min(range(len(times)), key=lambda x: abs(times[x] - end_rt))
    minind = max(0, minind)
    maxind = min(len(times) - 1, maxind)
    return (minind, maxind)


def calculateMetaboliteGroups(
    file="./results.xlsx",
    groups=[],
    eicPPM=10.0,
    maxAnnotationTimeWindow=0.05,
    minConnectionsInFiles=1,
    minConnectionRate=0.4,
    minPeakCorrelation=0.85,
    useAbundanceSimilarity=True,
    abundanceSimilarityThreshold=None,
    useRatio=False,
    runIdentificationInstance=None,
    pwMaxSet=None,
    pwValSet=None,
    pwTextSet=None,
    cpus=1,
    toFile=None,
    sheet_name="1_Bracketed",
    new_sheet_name="2_StatColumns",
):
    if toFile is None:
        toFile = file

    logging.info("Starting convoluting feature pairs..")

    resDB = Bunch(conn=None, curs=None)
    resDB.conn = PolarsDB(file, format="xlsx", load_all_tables=True)
    resDB.curs = resDB.conn.cursor()

    try:
        # Select only those groups that shall be used for metabolite group grouping
        ## load mzxml files for improved metabolite convolution
        useGroups = []
        useGroupsForConfig = []
        numFilesForConvolution = 0
        for group in groups:
            if group.useForMetaboliteGrouping:
                useGroups.append(group)
                useGroupsForConfig.append(str(group.name) + ":" + str(group.files))

                for fi in group.files:
                    numFilesForConvolution += 1

        if pwMaxSet is not None:
            pwMaxSet.put(Bunch(mes="max", val=numFilesForConvolution))
        if pwValSet is not None:
            pwValSet.put(Bunch(mes="value", val=0))

        # connect to results db
        writeMetaboliteGroupingConfigToDB(
            resDB.conn,
            minConnectionsInFiles,
            minConnectionRate,
            str(useGroupsForConfig).replace("'", "").replace('"', ""),
        )

        # read results table from Excel file
        table_df = resDB.conn.get_table(sheet_name)

        cols = table_df.columns

        # check if necessary columns are present
        ## overall columns
        assert "Num" in cols
        assert "OGroup" in cols
        assert "Ion" in cols
        assert "Loss" in cols
        assert "M" in cols
        assert "doublePeak" in cols

        # find double peaks as non nan and greater than 0
        # cast to Int64 first to handle both Utf8 and Int64 column types
        res = table_df.with_columns(_z=pl.when(pl.col("doublePeak").is_not_null() & (pl.col("doublePeak").cast(pl.Int64, strict=False).fill_null(0) > 0)).then(pl.lit("b__doublePeak")).otherwise(pl.lit("a__normal"))).sort("_z").partition_by("_z", maintain_order=True, include_key=False, as_dict=True)

        table_df = None
        double_peaks_df = None

        print(f"found keys: {res.keys()}")

        if ("b__doublePeak",) in res:
            double_peaks_df = res[("b__doublePeak",)]
        if ("a__normal",) in res:
            table_df = res[("a__normal",)]

        if double_peaks_df is not None and len(double_peaks_df) > 0:
            logging.info(f"  found double peaks: {len(double_peaks_df)} of {len(table_df)} feature pairs ({(len(double_peaks_df) / len(table_df)) * 100:.2f}%)")

        required_columns_not_found = []
        for group in useGroups:
            for f in group.files:
                fShort = f[f.rfind("/") + 1 : f.rfind(".")]

                if f"{fShort}_FID" not in cols:
                    required_columns_not_found.append(f"{fShort}_FID")
                if f"{fShort}_GroupID" not in cols:
                    required_columns_not_found.append(f"{fShort}_GroupID")
        if len(required_columns_not_found) > 0:
            print(f"ERROR: the following required columns were not found: {', '.join(required_columns_not_found)}")
            assert False

        # fetch all feature pairs from the results file
        nodes = {}
        for row in table_df.iter_rows(named=True):
            nodes[row["Num"]] = Bunch(
                fpNum=row["Num"],
                mz=row["MZ"],
                lmz=row["L_MZ"],
                rt=row["RT"],
                xn=row["Xn"],
                charge=row["Charge"],
                scanEvent=row["ScanEvent"],
                ionMode=row["Ionisation_Mode"],
                oGroup=None,
                correlationToOtherFPs={},
            )

        ## generate all correlations and sil ratios
        fileCorrelations = {}
        fileSILRatios = {}
        borders = {}
        procObjects = []
        for group in useGroups:
            for fi in group.files:
                borders[fi] = {}
                fil = os.path.basename(fi).replace(".mzXML", "").replace(".mzML", "")

                for row in table_df.iter_rows(named=True):
                    fpNum = row["Num"]
                    fID = row[f"{fil}_FID"]

                    if fID is not None:
                        n_start_rt = row.get(f"{fil}_N_startRT")
                        n_apex_rt = row.get(f"{fil}_N_apexRT")
                        n_end_rt = row.get(f"{fil}_N_endRT")
                        l_start_rt = row.get(f"{fil}_L_startRT")
                        l_apex_rt = row.get(f"{fil}_L_apexRT")
                        l_end_rt = row.get(f"{fil}_L_endRT")
                        if n_start_rt is not None and n_end_rt is not None and l_start_rt is not None and l_end_rt is not None:
                            borders[fi][fpNum] = (float(n_start_rt), float(n_apex_rt), float(n_end_rt), float(l_start_rt), float(l_apex_rt), float(l_end_rt))

                if fi not in fileCorrelations.keys():
                    s = ConvoluteFPsInFile(eicPPM, fi, maxAnnotationTimeWindow, nodes, borders)
                    procObjects.append(s)

        p = Pool(processes=cpus, maxtasksperchild=1)  # only in python >=2.7; experimental
        manager = Manager()
        manager.Queue()

        start = time.time()
        # start the multiprocessing pool
        res = p.imap_unordered(processConvoluteFPsInFile, procObjects)

        # wait until all subprocesses have finished re-integrating their respective LC-HRMS data file
        loop = True
        range(min(len(procObjects), cpus))
        while loop:
            completed = res._index
            if completed == len(procObjects):
                loop = False
            else:
                if pwValSet is not None:
                    pwValSet.put(Bunch(mes="value", val=completed))

                elapsed = (time.time() - start) / 60.0
                hours = ""
                if elapsed >= 60.0:
                    hours = "%d hours " % (elapsed // 60)
                mins = "%.2f mins" % (elapsed % 60.0)

                if pwTextSet is not None:
                    pwTextSet.put(
                        Bunch(
                            mes="text",
                            val="<p align='right' >%s%s elapsed</p>\n\n%d / %d files done (%d parallel)" % (hours, mins, completed, len(procObjects), cpus),
                        )
                    )

                time.sleep(0.5)

        p.close()

        ## fetch multiprocessing results
        for rei, re in enumerate(res):
            fileCorrelations[procObjects[rei].fi] = re[0]
            fileSILRatios[procObjects[rei].fi] = re[1]

        logging.info("Testing SIL ratios")
        ## test all feature pair pairs for co-elution and similar SIL ratio
        connections = {}
        for fpNumA in nodes.keys():
            connections[fpNumA] = {}
        for fpNumA in nodes.keys():
            nodeA = nodes[fpNumA]

            for fpNumB in nodes.keys():
                nodeB = nodes[fpNumB]

                if nodeA.fpNum != nodeB.fpNum and nodeA.fpNum < nodeB.fpNum and abs(nodeB.rt - nodeA.rt) <= maxAnnotationTimeWindow:
                    ## test feature pair pair convolution in all samples
                    allCorrelations = []
                    allSILRatios = []
                    for group in useGroups:
                        for fi in group.files:
                            if fi in fileCorrelations.keys() and fpNumA in fileCorrelations[fi].keys() and fpNumB in fileCorrelations[fi][fpNumA].keys():
                                co = fileCorrelations[fi][fpNumA][fpNumB]
                                allCorrelations.append(co)
                            if fi in fileSILRatios.keys() and fpNumA in fileSILRatios[fi].keys() and fpNumB in fileSILRatios[fi][fpNumA].keys():
                                silRatio = fileSILRatios[fi][fpNumA][fpNumB]
                                allSILRatios.append(silRatio)

                    if len(allCorrelations) > 0 and len(allSILRatios) > 0:
                        if sum([co >= minPeakCorrelation for co in allCorrelations]) >= minConnectionRate * len(allCorrelations) and (not useRatio or sum([rat for rat in allSILRatios]) >= minConnectionRate * len(allSILRatios)):
                            nodes[nodeA.fpNum].correlationToOtherFPs[nodeB.fpNum] = True
                            nodes[nodeB.fpNum].correlationToOtherFPs[nodeA.fpNum] = True

                        m = mean(allCorrelations)
                        connections[fpNumA][fpNumB] = m
                        connections[fpNumB][fpNumA] = m
                    else:
                        connections[fpNumA][fpNumB] = 0
                        connections[fpNumB][fpNumA] = 0

        logging.info(" .. done")

        nodes2 = {}
        for fpNumA in nodes.keys():
            if fpNumA not in nodes2.keys():
                nodes2[fpNumA] = []
            for fpNumB in nodes[fpNumA].correlationToOtherFPs.keys():
                nodes2[fpNumA].append(fpNumB)

        for k in nodes2.keys():
            uniq = []
            for u in nodes2[k]:
                if u not in uniq:
                    uniq.append(u)
            nodes2[k] = uniq

        # get subgraphs from the feature pair graph. Each subgraph represents one convoluted
        # feature group
        tGroups = getSubGraphs(nodes2)

        def splitGroupWithHCA(tGroup, correlations):
            # if 1 or two feature pairs are in a group, automatically use them (no further splitting possible)
            if len(tGroup) <= 2:
                return [tGroup]

            # construct 2-dimensional data matrix
            data = []
            for tG1 in tGroup:
                t = []
                for tG2 in tGroup:
                    if tG1 in correlations.keys() and tG2 in correlations[tG1].keys():
                        t.append(correlations[tG1][tG2])
                    elif tG1 == tG2:
                        t.append(1.0)
                    else:
                        t.append(0.0)
                data.append(t)

            # calculate HCA tree using the correlations between all feature pairs
            hc = HCA_general.HCA_generic()
            tree = hc.generateTree(objs=data, ids=tGroup)

            # split the HCA tree in sub-clusters
            def checkSubCluster(tree, hca, corrThreshold, cutOffMinRatio):
                if isinstance(tree, HCA_general.HCALeaf):
                    return False
                elif isinstance(tree, HCA_general.HCAComposite):
                    corrs = hca.getLinkFor(tree)
                    inds = hca.getIndsFor(tree)

                    aboveThreshold = sum([corr > corrThreshold for i, corr in enumerate(corrs) if i in inds])
                    return not (aboveThreshold * 1.0 / len(inds)) >= cutOffMinRatio

                else:
                    raise RuntimeError("Unknown if-branch")

            subClusts = hc.splitTreeWithCallback(
                tree,
                CallBackMethod(
                    _target=checkSubCluster,
                    corrThreshold=minPeakCorrelation,
                    cutOffMinRatio=minConnectionRate,
                ).getRunMethod(),
                recursive=True,
            )

            # convert the subclusters into arrays of feature pairs belonging to the same metabolite
            return [[leaf.getID() for leaf in subClust.getLeaves()] for subClust in subClusts]

        groups = []
        done = 0
        for tGroup in tGroups:
            sGroups = splitGroupWithHCA(tGroup, connections)
            groups.extend(sGroups)

            done = done + 1

        if useAbundanceSimilarity:
            logging.info("Refining feature groups by abundance profile similarity")
            # use per-file native peak abundances to compare feature profile similarity across samples
            abundance_cols = [col for col in table_df.columns if col.endswith("_Abundance_N")]
            abundance_vectors = {}
            if len(abundance_cols) > 0:
                for row in table_df.select(["Num"] + abundance_cols).iter_rows(named=True):
                    abundance_vectors[row["Num"]] = [row.get(col) for col in abundance_cols]

            effective_abundance_threshold = minPeakCorrelation if abundanceSimilarityThreshold is None else abundanceSimilarityThreshold
            refined_groups = []
            for tGroup in groups:
                refined_groups.extend(
                    split_group_by_relative_abundance(
                        tGroup,
                        abundance_vectors,
                        min_peak_correlation=effective_abundance_threshold,
                        min_connection_rate=minConnectionRate,
                    )
                )
            groups = refined_groups

        # Separate feature pair clusters; softer
        curGroup = 1
        for tGroup in groups:
            for i in range(len(tGroup)):
                # Update in DataFrame
                table_df = table_df.with_columns(pl.when(pl.col("Num") == tGroup[i]).then(pl.lit(curGroup)).otherwise(pl.col("OGroup")).alias("OGroup"))

            curGroup += 1

        logging.info("Annotating ions in feature groups")
        # Annotate groups with common adducts and in-source fragments
        if runIdentificationInstance is not None:
            groups = defaultdict(list)

            for row in table_df.iter_rows(named=True):
                groups[row["OGroup"]].append(ChromPeakPair(id=row["Num"], fGroupID=row["OGroup"], mz=row["MZ"], ionMode=row["Ionisation_Mode"], loading=row["Charge"], adducts=[], heteroAtomsFeaturePairs=[], xCount=int(row["Xn"]), fDesc=[]))

            for ogrp in groups.keys():
                chromPeaks = {}
                for fp in groups[ogrp]:
                    chromPeaks[fp.id] = fp

                runIdentificationInstance.annotateChromPeaks(chromPeaks.keys(), chromPeaks)

            for ogrp in groups.keys():
                for fp in groups[ogrp]:
                    table_df = table_df.with_columns(pl.when(pl.col("Num") == fp.id).then(pl.lit(",".join([str(a) for a in fp.adducts]))).otherwise(pl.col("Ion")).alias("Ion"))
                    table_df = table_df.with_columns(pl.when(pl.col("Num") == fp.id).then(pl.lit(",".join([str(a) for a in fp.fDesc]))).otherwise(pl.col("Loss")).alias("Loss"))
                    table_df = table_df.with_columns(pl.when(pl.col("Num") == fp.id).then(pl.lit(",".join([str(a) for a in fp.Ms]))).otherwise(pl.col("M")).alias("M"))

        # Calculate column Relative_peakarea_in_group and Average_peakarea per group
        # use the average abundance from any column ending with _Abundance_N, ignore missing values
        if "Average_peakarea" not in table_df.columns:
            table_df = table_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("Average_peakarea"))
        for g in table_df.select(pl.col("OGroup")).unique().to_series():
            group_rows = table_df.filter(pl.col("OGroup") == g)
            abundance_cols = [col for col in table_df.columns if col.endswith("_Abundance_N")]
            avg_abundances = []
            for row in group_rows.iter_rows(named=True):
                abundances = []
                for col in abundance_cols:
                    val = row[col]
                    if val is not None:
                        try:
                            abundances.append(float(val))
                        except Exception:
                            pass
                if len(abundances) > 0:
                    avg_abundance = sum(abundances) / len(abundances)
                else:
                    avg_abundance = 0
                avg_abundances.append((row["Num"], avg_abundance))

            # Sort by average abundance descending
            avg_abundances.sort(key=lambda x: x[1], reverse=True)
            total_abundance, _max_abundance = sum([ab for _, ab in avg_abundances]), max([ab for _, ab in avg_abundances])

            # Assign Relative_peakarea_in_group and Average_peakarea
            for rank, (fpNum, avg_abundance) in enumerate(avg_abundances, start=1):
                abundance_ratio = avg_abundance / total_abundance if total_abundance > 0 else 0.0
                table_df = table_df.with_columns(pl.when(pl.col("Num") == fpNum).then(pl.lit(abundance_ratio)).otherwise(pl.col("Relative_peakarea_in_group")).alias("Relative_peakarea_in_group"))
                table_df = table_df.with_columns(pl.when(pl.col("Num") == fpNum).then(pl.lit(avg_abundance)).otherwise(pl.col("Average_peakarea")).alias("Average_peakarea"))

        # Convert columns OGroup, Relative_peakarea_in_group and Average_peakarea to the correct types
        table_df = table_df.with_columns(pl.col("OGroup").cast(pl.Int64))
        table_df = table_df.with_columns(pl.col("Relative_peakarea_in_group").cast(pl.Float64))
        table_df = table_df.with_columns(pl.col("Average_peakarea").cast(pl.Float64))

        # Sort table_df by OGroup asc and Relative_peakarea_in_group desc
        table_df = table_df.sort(["OGroup", "Relative_peakarea_in_group"], descending=[False, True])

        resDB.conn.insert_row("Parameters", {"Parameter": "# Grouping", "Value": ""})
        resDB.conn.insert_row("Parameters", {"Parameter": "MEConvoluting_groups", "Value": f"{useGroupsForConfig}".replace("'", "").replace('"', "")})
        resDB.conn.insert_row("Parameters", {"Parameter": "MEConvoluting_minConnectionsInFiles", "Value": f"{minConnectionsInFiles}"})
        resDB.conn.insert_row("Parameters", {"Parameter": "MEConvoluting_minConnectionRate", "Value": f"{minConnectionRate}"})
        resDB.conn.insert_row("Parameters", {"Parameter": "MEConvoluting_minPeakCorrelation", "Value": f"{minPeakCorrelation}"})
        resDB.conn.insert_row("Parameters", {"Parameter": "MEConvoluting_useAbundanceSimilarity", "Value": f"{useAbundanceSimilarity}"})
        resDB.conn.insert_row(
            "Parameters",
            {
                "Parameter": "MEConvoluting_abundanceSimilarityThreshold",
                "Value": f"{minPeakCorrelation if abundanceSimilarityThreshold is None else abundanceSimilarityThreshold}",
            },
        )
        resDB.conn.set_table(new_sheet_name, table_df)
        if double_peaks_df is not None and len(double_peaks_df) > 0:
            resDB.conn.insert_table(new_sheet_name + "_doublePeaks", double_peaks_df)
        resDB.conn.commit()

        exportAsFeatureML.convertMEMatrixToFeatureMLSepPolarities(toFile, sheet_name=new_sheet_name, postfix="_ac")

    except Exception as ex:
        import traceback

        traceback.print_exc()

        raise ex

    finally:
        resDB.conn.commit()
        resDB.curs.close()
        resDB.conn.close()


class ConvoluteFPsInFile:
    def __init__(self, eicPPM, fi, maxAnnotationTimeWindow, nodes, borders):
        self.eicPPM = eicPPM
        self.fi = fi
        self.maxAnnotationTimeWindow = maxAnnotationTimeWindow
        self.nodes = nodes
        self.borders = borders

        self.fileCorrelations = None
        self.fileSILRatios = None

    def getConvolutionInFile(self):
        startProc = time.time()

        eicPPM = self.eicPPM
        fi = self.fi
        maxAnnotationTimeWindow = self.maxAnnotationTimeWindow
        nodes = self.nodes
        borders = self.borders

        fileCorrelations = {}
        fileSILRatios = {}

        mzXML = Chromatogram()
        mzXML.parse_file(fi)

        filterLines = mzXML.getFilterLines(
            includeMS1=True,
            includeMS2=False,
            includePosPolarity=True,
            includeNegPolarity=True,
        )

        for fpNumA in nodes.keys():
            fileCorrelations[fpNumA] = {}
            fileSILRatios[fpNumA] = {}

        for fpNumA in nodes.keys():
            nodeA = nodes[fpNumA]

            for fpNumB in nodes.keys():
                nodeB = nodes[fpNumB]

                if nodeA.fpNum != nodeB.fpNum and nodeA.fpNum < nodeB.fpNum and abs(nodeB.rt - nodeA.rt) <= maxAnnotationTimeWindow:
                    ra = None
                    if fi in borders.keys() and nodeA.fpNum in borders[fi].keys():
                        ra = borders[fi][nodeA.fpNum]
                    rb = None
                    if fi in borders.keys() and nodeB.fpNum in borders[fi].keys():
                        rb = borders[fi][nodeB.fpNum]

                    if ra is not None and rb is not None:
                        na_start, na_apex, na_end, la_start, la_apex, la_end = ra
                        nb_start, nb_apex, nb_end, lb_start, lb_apex, lb_end = rb

                        na_hw_left = max(0.001, na_apex - na_start)
                        na_hw_right = max(0.001, na_end - na_apex)
                        la_hw_left = max(0.001, la_apex - la_start)
                        la_hw_right = max(0.001, la_end - la_apex)
                        nb_hw_left = max(0.001, nb_apex - nb_start)
                        nb_hw_right = max(0.001, nb_end - nb_apex)
                        max(0.001, lb_apex - lb_start)
                        max(0.001, lb_end - lb_apex)

                        if nodeA.scanEvent in filterLines and nodeB.scanEvent in filterLines:
                            meanRT = mean([nodeA.rt, nodeB.rt])

                            eicAL, timesA, scanIdsA, mzs = mzXML.getEIC(
                                nodeA.lmz,
                                ppm=eicPPM,
                                filterLine=nodeA.scanEvent,
                                removeSingles=True,
                                intThreshold=0,
                                useMS1=True,
                                useMS2=False,
                                startTime=meanRT * 60 - 120,
                                endTime=meanRT * 60 + 120,
                            )
                            eicBL, timesB, scanIdsB, mzs = mzXML.getEIC(
                                nodeB.lmz,
                                ppm=eicPPM,
                                filterLine=nodeB.scanEvent,
                                removeSingles=True,
                                intThreshold=0,
                                useMS1=True,
                                useMS2=False,
                                startTime=meanRT * 60 - 120,
                                endTime=meanRT * 60 + 120,
                            )

                            eicA, timesA, scanIdsA, mzs = mzXML.getEIC(
                                nodeA.mz,
                                ppm=eicPPM,
                                filterLine=nodeA.scanEvent,
                                removeSingles=True,
                                intThreshold=0,
                                useMS1=True,
                                useMS2=False,
                                startTime=meanRT * 60 - 120,
                                endTime=meanRT * 60 + 120,
                            )
                            eicB, timesB, scanIdsB, mzs = mzXML.getEIC(
                                nodeB.mz,
                                ppm=eicPPM,
                                filterLine=nodeB.scanEvent,
                                removeSingles=True,
                                intThreshold=0,
                                useMS1=True,
                                useMS2=False,
                                startTime=meanRT * 60 - 120,
                                endTime=meanRT * 60 + 120,
                            )

                            timesMin = [t for t in timesA]
                            timesMin.extend([t for t in timesB])
                            timesMin = sorted(timesMin)

                            eicAL = mapArrayToRefTimes(eicAL, timesA, timesMin)
                            eicBL = mapArrayToRefTimes(eicBL, timesB, timesMin)
                            eicA = mapArrayToRefTimes(eicA, timesA, timesMin)
                            eicB = mapArrayToRefTimes(eicB, timesB, timesMin)

                            timesMin = [t / 60.0 for t in timesMin]

                            ## A) Test correlation of different feature pairs
                            try:
                                corr_left = min(na_hw_left, nb_hw_left) * 2
                                corr_right = min(na_hw_right, nb_hw_right) * 2
                                lI, rI = getPeakByRT(
                                    timesMin,
                                    meanRT - corr_left,
                                    meanRT + corr_right,
                                )

                                eicAC = eicAL[lI:rI]
                                eicBC = eicBL[lI:rI]

                                co = corr(eicAC, eicBC)

                                if not (isnan(co)) and co is not None:
                                    fileCorrelations[fpNumA][fpNumB] = co
                            except Exception as err:
                                logging.error("  Error during convolution of feature pairs (Peak-correlation, Nums: %s and %s, message: %s).." % (fpNumA, fpNumB, str(err)))

                            ## B) Test similarity of native:labeled ratio
                            try:
                                sil_hw_a_n = min(na_hw_left, na_hw_right) * 0.8
                                sil_hw_a_l = min(la_hw_left, la_hw_right) * 0.8
                                lISIL, rISIL = getPeakByRT(
                                    timesMin,
                                    nodeA.rt - sil_hw_a_n,
                                    nodeA.rt + sil_hw_a_l,
                                )

                                eicANCSIL = eicA[lISIL:rISIL]
                                eicALCSIL = eicAL[lISIL:rISIL]

                                folds = [eicANCSIL[i] / eicALCSIL[i] for i in range(len(eicANCSIL)) if eicALCSIL[i] > 0]
                                ma = mean(folds)
                                sa = sd(folds)

                                lISIL, rISIL = getPeakByRT(
                                    timesMin,
                                    nodeB.rt - sil_hw_a_n,
                                    nodeB.rt + sil_hw_a_l,
                                )

                                eicBNCSIL = eicB[lISIL:rISIL]
                                eicBLCSIL = eicBL[lISIL:rISIL]

                                folds = [eicBNCSIL[i] / eicBLCSIL[i] for i in range(len(eicBNCSIL)) if eicBLCSIL[i] > 0]
                                mb = mean(folds)
                                sb = sd(folds)

                                if ma is not None and mb is not None and sa is not None and sb is not None:
                                    silRatioFold = 1
                                    try:
                                        silRatioFold = max([ma, mb]) / min([ma, mb])
                                    except RuntimeWarning:
                                        pass
                                    # print(ma, mb, sa, sb, max([ma, mb]), min([ma, mb]))
                                    fileSILRatios[fpNumA][fpNumB] = silRatioFold <= 1 + max(0.5, 3 * mean([sa, sb]))

                                    # print(fiName, nodeA.mz, nodeB.mz, meanRT, silRatioFold, co)
                            except Exception as err:
                                logging.error("  Error during convolution of feature pairs (SIL-ratio, Nums: %s and %s, message: %s).." % (fpNumA, fpNumB, str(err)))

        logging.info("  Finished convoluting feature pairs in file %s (%.1f minutes)" % (fi, (time.time() - startProc) / 60.0))
        return (fileCorrelations, fileSILRatios)


def processConvoluteFPsInFile(cif):
    se = cif.getConvolutionInFile()
    return se
