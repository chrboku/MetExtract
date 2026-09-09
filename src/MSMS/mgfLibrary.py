"""
MGF / JSON spectral library loading and matching against experimental MS/MS spectra.

This module:
    * parses .mgf spectral library files (raw KEY=VALUE metadata blocks, kept exactly as
      written in the file, so the user can pick which field holds the precursor m/z)
    * parses .json spectral library files (a JSON list of arbitrarily nested spectrum
      dicts; nested dict/list structures are flattened to "level0///level1///..." keys;
      fragments are read from the "PK$PEAK" field: a list of [mz, abs_intensity, rel_intensity])
    * resolves the polarity ("+"/"-") of each library spectrum (from an ION_MODE/CHARGE/
      adduct-like field, matched case-insensitively on the last "///"-separated segment)
    * uses ``matchms`` to compare an experimental MS2 spectrum against all (polarity- and
      precursor-mz-matching) library spectra with a configurable similarity algorithm

It intentionally has no Qt/GUI dependencies so it can be unit tested and reused by the
"MSMS spectra overview" dialog as well as the annotation pipeline.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os

import numpy as np

try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.similarity import CosineGreedy, CosineHungarian, ModifiedCosineGreedy, ModifiedCosineHungarian, NeutralLossesCosine

    MATCHMS_AVAILABLE = True
except Exception:
    MATCHMS_AVAILABLE = False

# Registry of selectable matchms similarity algorithms for MGF-library matching.
MSMS_LIBRARY_ALGORITHMS = {}
if MATCHMS_AVAILABLE:
    MSMS_LIBRARY_ALGORITHMS = {
        "ModifiedCosineHungarian": ModifiedCosineHungarian,
        "ModifiedCosineGreedy": ModifiedCosineGreedy,
        "CosineHungarian": CosineHungarian,
        "CosineGreedy": CosineGreedy,
        "NeutralLossesCosine": NeutralLossesCosine,
    }
DEFAULT_MSMS_LIBRARY_ALGORITHM = "ModifiedCosineHungarian"

# Key used in JSON spectral library entries to hold the peak list.
JSON_PEAKS_KEY = "PK$PEAK"
FLATTEN_SEPARATOR = "///"


class MGFLibrarySpectrum:
    """A single spectrum loaded from an MGF or JSON spectral library file."""

    __slots__ = (
        "mz",
        "intensities",
        "metadata",
        "compound_name",
        "precursor_mz",
        "polarity",
        "polarity_source",
        "source_file",
        "spectrum_index",
        "matchms_spectrum",
        "matchms_prepared_fragment_min_rel_abundance",
    )

    def __init__(
        self,
        mz,
        intensities,
        metadata,
        source_file,
        spectrum_index,
        precursor_mz=None,
        compound_name=None,
        polarity=None,
        polarity_source=None,
    ):
        self.mz = np.asarray(mz, dtype=np.float64)
        self.intensities = np.asarray(intensities, dtype=np.float64)
        self.metadata = metadata or {}
        self.source_file = source_file
        self.spectrum_index = spectrum_index
        self.compound_name = compound_name or f"Spectrum #{spectrum_index}"
        self.precursor_mz = precursor_mz
        self.polarity = polarity
        self.polarity_source = polarity_source
        self.matchms_spectrum = None
        self.matchms_prepared_fragment_min_rel_abundance = None


def _find_by_last_segment(metadata, candidates):
    """
    Search a (possibly flattened) metadata dict for a key whose last "///"-separated
    segment matches (case-insensitively) one of the given candidate names.
    """
    for k, v in metadata.items():
        last_segment = str(k).split(FLATTEN_SEPARATOR)[-1].strip().lower()
        if last_segment in candidates:
            return v
    return None


def _parse_float_field(value):
    """Parse a (possibly multi-token, e.g. MGF PEPMASS "mz intensity") field into a float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    token = text.replace(",", " ").split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def _parse_polarity_value(value):
    """Parse a polarity-like value ("Positive"/"pos"/"+"/"Negative"/"neg"/"-", case-insensitive)
    into "+"/"-", or None if it does not match any recognised form."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("positive", "pos", "+"):
        return "+"
    if text in ("negative", "neg", "-"):
        return "-"
    return None


def resolve_polarity(metadata, polarity_key=None):
    """
    Determine the polarity ("+"/"-") of a library spectrum from its (possibly flattened)
    metadata.

    Resolution order:
        1. ``polarity_key`` (a user-selected metadata field name, exact match), if given
        2. ION_MODE / ionmode / polarity field ("positive"/"negative"/"+"/"-")
        3. CHARGE field (e.g. "1+", "-1", "1-")
        4. Adduct name (e.g. "[M+H]+", "[M-H]-")

    Returns (polarity_or_None, source_str) where source_str is one of
    "user_field", "ion_mode", "charge", "adduct", or None if it could not be determined.
    """
    if not metadata:
        return None, None

    if polarity_key and polarity_key in metadata:
        parsed = _parse_polarity_value(metadata[polarity_key])
        if parsed:
            return parsed, "user_field"

    ion_mode = _find_by_last_segment(metadata, {"ion_mode", "ionmode", "polarity"})
    parsed = _parse_polarity_value(ion_mode)
    if parsed:
        return parsed, "ion_mode"

    charge = _find_by_last_segment(metadata, {"charge"})
    if charge is not None:
        charge_str = str(charge).strip()
        if charge_str:
            if charge_str.startswith("-") or charge_str.endswith("-"):
                return "-", "charge"
            if charge_str.startswith("+") or charge_str.endswith("+"):
                return "+", "charge"
            try:
                charge_val = float(charge_str)
                if charge_val > 0:
                    return "+", "charge"
                if charge_val < 0:
                    return "-", "charge"
            except ValueError:
                pass

    adduct = _find_by_last_segment(metadata, {"adduct", "adduct_type"})
    if adduct:
        adduct_str = str(adduct).strip()
        if adduct_str.endswith("]+") or adduct_str.endswith("+"):
            return "+", "adduct"
        if adduct_str.endswith("]-") or adduct_str.endswith("-"):
            return "-", "adduct"

    return None, None


def resolve_compound_name(metadata, spectrum_index):
    name = _find_by_last_segment(metadata, {"compound_name", "name", "title"})
    return str(name) if name else f"Spectrum #{spectrum_index}"


def resolve_precursor_mz(metadata, precursor_mz_key=None):
    """
    Determine the precursor m/z of a library spectrum.

    If ``precursor_mz_key`` (a user-selected metadata field name, exact match) is given and
    present, it is used. Otherwise a few common field names are tried as a fallback.
    """
    if precursor_mz_key and precursor_mz_key in metadata:
        return _parse_float_field(metadata[precursor_mz_key])
    value = _find_by_last_segment(metadata, {"precursor_mz", "pepmass", "precursor_m/z", "precursormz"})
    return _parse_float_field(value)


# ---------------------------------------------------------------------------
# Raw MGF parsing (kept independent from matchms so the exact field names as
# written in the file are preserved and selectable by the user).
# ---------------------------------------------------------------------------


def _parse_mgf_blocks(path):
    """Parse a .mgf file into a list of (metadata_dict, mz_list, intensity_list) tuples,
    one per BEGIN IONS/END IONS block, preserving metadata field names exactly as written."""
    blocks = []
    metadata = {}
    mz = []
    intensities = []
    in_block = False

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            upper_line = line.upper()
            if upper_line == "BEGIN IONS":
                in_block = True
                metadata = {}
                mz = []
                intensities = []
                continue
            if upper_line == "END IONS":
                in_block = False
                blocks.append((metadata, mz, intensities))
                continue
            if not in_block:
                continue

            if "=" in line and line.split("=", 1)[0].strip().replace("_", "").isalnum():
                key, _, value = line.partition("=")
                metadata[key.strip()] = value.strip()
            else:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        mz.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                    except ValueError:
                        pass

    return blocks


def discover_mgf_properties(path):
    """Return the sorted set of all metadata field names found across all spectra in an MGF file."""
    properties = set()
    for metadata, _mz, _intensities in _parse_mgf_blocks(path):
        properties.update(metadata.keys())
    return sorted(properties)


def load_mgf_file(path, precursor_mz_key=None, polarity_key=None):
    """
    Load all spectra from a single .mgf file.

    Args:
        path: path to the .mgf file
        precursor_mz_key: name of the metadata field (as written in the file, e.g. "PEPMASS")
            to use as the precursor m/z; falls back to common field names if not given/found.
        polarity_key: name of the metadata field (as written in the file, e.g. "IONMODE") to use
            as the polarity ("Positive"/"pos"/"+"/"Negative"/"neg"/"-", case-insensitive); falls
            back to ION_MODE/CHARGE/adduct heuristics if not given/found.

    Returns a list of MGFLibrarySpectrum objects.
    """
    spectra = []
    for idx, (metadata, mz, intensities) in enumerate(_parse_mgf_blocks(path)):
        polarity, polarity_source = resolve_polarity(metadata, polarity_key)
        spectra.append(
            MGFLibrarySpectrum(
                mz,
                intensities,
                metadata,
                source_file=path,
                spectrum_index=idx,
                precursor_mz=resolve_precursor_mz(metadata, precursor_mz_key),
                compound_name=resolve_compound_name(metadata, idx),
                polarity=polarity,
                polarity_source=polarity_source,
            )
        )
    logging.info(f"Loaded {len(spectra)} spectra from MGF library file '{path}'")
    return spectra


# ---------------------------------------------------------------------------
# MSP spectral library parsing (NIST/MassBank-style "Key: Value" metadata blocks,
# separated by blank lines, without MGF's "BEGIN IONS"/"END IONS" markers).
# ---------------------------------------------------------------------------

_MSP_NUM_PEAKS_KEYS = ("num peaks", "numpeaks")


def _parse_msp_blocks(path):
    """Parse a .msp file into a list of (metadata_dict, mz_list, intensity_list) tuples,
    one per entry, preserving metadata field names exactly as written (before the first
    ":"). Entries are separated by one or more blank lines; each entry starts with
    "Key: Value" metadata lines, followed by a "Num Peaks: N" line and then N lines of
    "mz intensity[ annotation]" peak data."""
    blocks = []
    metadata = {}
    mz = []
    intensities = []
    num_peaks_expected = None
    peaks_read = 0
    in_entry = False

    def _flush():
        if in_entry and (metadata or mz):
            blocks.append((metadata, mz, intensities))

    with open(path, encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                # Blank line: end of the current entry (if one is open) once all its peaks were read.
                if in_entry and (num_peaks_expected is None or peaks_read >= num_peaks_expected):
                    _flush()
                    metadata = {}
                    mz = []
                    intensities = []
                    num_peaks_expected = None
                    peaks_read = 0
                    in_entry = False
                continue

            if num_peaks_expected is not None and peaks_read < num_peaks_expected:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    try:
                        mz.append(float(parts[0]))
                        intensities.append(float(parts[1]))
                        peaks_read += 1
                        continue
                    except ValueError:
                        pass

            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if not in_entry:
                    in_entry = True
                    metadata = {}
                    mz = []
                    intensities = []
                    num_peaks_expected = None
                    peaks_read = 0
                if key.lower() in _MSP_NUM_PEAKS_KEYS:
                    try:
                        num_peaks_expected = int(value.split()[0])
                    except (ValueError, IndexError):
                        num_peaks_expected = None
                elif key in metadata:
                    # Repeated keys (e.g. multiple "Synon:" lines) are collected into a list.
                    if isinstance(metadata[key], list):
                        metadata[key].append(value)
                    else:
                        metadata[key] = [metadata[key], value]
                else:
                    metadata[key] = value

    if in_entry:
        _flush()

    return blocks


def discover_msp_properties(path):
    """Return the sorted set of all metadata field names found across all spectra in an MSP file."""
    properties = set()
    for metadata, _mz, _intensities in _parse_msp_blocks(path):
        properties.update(metadata.keys())
    return sorted(properties)


def load_msp_file(path, precursor_mz_key=None, polarity_key=None):
    """
    Load all spectra from a single .msp file.

    Args:
        path: path to the .msp file
        precursor_mz_key: name of the metadata field (as written in the file, e.g.
            "PrecursorMZ") to use as the precursor m/z; falls back to common field names
            if not given/found.
        polarity_key: name of the metadata field (as written in the file, e.g. "Ion_mode")
            to use as the polarity ("Positive"/"pos"/"+"/"Negative"/"neg"/"-",
            case-insensitive); falls back to ION_MODE/CHARGE/adduct heuristics if not
            given/found.

    Returns a list of MGFLibrarySpectrum objects.
    """
    spectra = []
    for idx, (metadata, mz, intensities) in enumerate(_parse_msp_blocks(path)):
        polarity, polarity_source = resolve_polarity(metadata, polarity_key)
        spectra.append(
            MGFLibrarySpectrum(
                mz,
                intensities,
                metadata,
                source_file=path,
                spectrum_index=idx,
                precursor_mz=resolve_precursor_mz(metadata, precursor_mz_key),
                compound_name=resolve_compound_name(metadata, idx),
                polarity=polarity,
                polarity_source=polarity_source,
            )
        )
    logging.info(f"Loaded {len(spectra)} spectra from MSP library file '{path}'")
    return spectra


# ---------------------------------------------------------------------------
# JSON spectral library parsing.
# ---------------------------------------------------------------------------


def _flatten_json(obj, prefix, out):
    """Recursively flatten a nested dict/list structure into ``out`` using FLATTEN_SEPARATOR."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{FLATTEN_SEPARATOR}{k}" if prefix else str(k)
            _flatten_json(v, key, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}{FLATTEN_SEPARATOR}{i}" if prefix else str(i)
            _flatten_json(v, key, out)
    else:
        out[prefix] = obj


def _parse_json_peaks(entry):
    """Parse the "PK$PEAK" field (list of [mz, abs_intensity, rel_intensity]) into (mz, intensities)."""
    mz = []
    intensities = []
    for fragment in entry.get(JSON_PEAKS_KEY, []) or []:
        if not isinstance(fragment, (list, tuple)) or len(fragment) < 3:
            continue
        try:
            mz.append(float(fragment[0]))
            intensities.append(float(fragment[2]))
        except (TypeError, ValueError):
            continue
    return mz, intensities


def _flatten_json_entry(entry):
    entry_without_peaks = {k: v for k, v in entry.items() if k != JSON_PEAKS_KEY} if isinstance(entry, dict) else entry
    flat = {}
    _flatten_json(entry_without_peaks, "", flat)
    return flat


def discover_json_properties(path):
    """Return the sorted set of all flattened field names found across all spectra in a JSON library file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON spectral library file '{path}' must contain a list of spectra.")

    properties = set()
    for entry in data:
        if isinstance(entry, dict):
            properties.update(_flatten_json_entry(entry).keys())
    return sorted(properties)


def load_json_file(path, precursor_mz_key=None, polarity_key=None):
    """
    Load all spectra from a single JSON spectral library file (a list of nested spectrum dicts).

    Args:
        path: path to the .json file
        precursor_mz_key: name of the flattened metadata field (e.g.
            "MS$FOCUSED_ION///PRECURSOR_M/Z") to use as the precursor m/z.
        polarity_key: name of the flattened metadata field (e.g.
            "AC$MASS_SPECTROMETRY_ION_MODE") to use as the polarity ("Positive"/"pos"/"+"/
            "Negative"/"neg"/"-", case-insensitive); falls back to ION_MODE/CHARGE/adduct
            heuristics if not given/found.

    Returns a list of MGFLibrarySpectrum objects.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"JSON spectral library file '{path}' must contain a list of spectra.")

    spectra = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        mz, intensities = _parse_json_peaks(entry)
        metadata = _flatten_json_entry(entry)
        polarity, polarity_source = resolve_polarity(metadata, polarity_key)
        spectra.append(
            MGFLibrarySpectrum(
                mz,
                intensities,
                metadata,
                source_file=path,
                spectrum_index=idx,
                precursor_mz=resolve_precursor_mz(metadata, precursor_mz_key),
                compound_name=resolve_compound_name(metadata, idx),
                polarity=polarity,
                polarity_source=polarity_source,
            )
        )
    logging.info(f"Loaded {len(spectra)} spectra from JSON library file '{path}'")
    return spectra


def discover_properties(path, file_type):
    if file_type == "json":
        return discover_json_properties(path)
    if file_type == "msp":
        return discover_msp_properties(path)
    return discover_mgf_properties(path)


def load_library_file(path, file_type, precursor_mz_key=None, polarity_key=None):
    """Dispatch to load_mgf_file, load_json_file or load_msp_file based on ``file_type``
    ("mgf"/"json"/"msp")."""
    if file_type == "json":
        return load_json_file(path, precursor_mz_key, polarity_key)
    if file_type == "msp":
        return load_msp_file(path, precursor_mz_key, polarity_key)
    return load_mgf_file(path, precursor_mz_key, polarity_key)


def load_library_entry(entry):
    """Load spectra for a library-file info dict {"path", "type", "precursor_mz_key", "polarity_key"}."""
    return load_library_file(
        entry["path"],
        entry.get("type", "mgf"),
        entry.get("precursor_mz_key"),
        entry.get("polarity_key"),
    )


def spectra_without_polarity(library_spectra):
    """Return the subset of library spectra for which polarity could not be resolved."""
    return [s for s in library_spectra if s.polarity is None]


def spectra_missing_precursor_mz(library_spectra):
    """Return the subset of library spectra missing a usable precursor m/z or fragment peaks.
    These spectra lack information required for matching and are always excluded (see
    ``prepare_library_spectra`` and ``match_spectrum_against_library``)."""
    return [s for s in library_spectra if s.precursor_mz is None or s.mz.size == 0]


def get_similarity_algorithm(name, mz_tolerance):
    """Instantiate a matchms similarity algorithm by name with the given m/z tolerance."""
    if not MATCHMS_AVAILABLE:
        raise RuntimeError("The 'matchms' package is not available.")
    cls = MSMS_LIBRARY_ALGORITHMS.get(name, MSMS_LIBRARY_ALGORITHMS.get(DEFAULT_MSMS_LIBRARY_ALGORITHM))
    return cls(tolerance=mz_tolerance)


def _to_matchms_spectrum(mz, intensities, metadata=None):
    mz = np.asarray(mz, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    order = np.argsort(mz)
    mz = mz[order]
    intensities = intensities[order]
    return MatchmsSpectrum(mz=mz, intensities=intensities, metadata=metadata or {})


def prepare_library_spectra(library_spectra, fragment_min_rel_abundance=0.0):
    """Prepare and cache matchms spectra for a library list exactly once per threshold."""
    threshold = float(fragment_min_rel_abundance or 0.0)
    for lib_spec in library_spectra:
        if lib_spec is None or lib_spec.mz.size == 0 or lib_spec.precursor_mz is None:
            continue
        if lib_spec.matchms_spectrum is not None and lib_spec.matchms_prepared_fragment_min_rel_abundance == threshold:
            continue

        lib_meta = dict(lib_spec.metadata) if lib_spec.metadata else {}
        if lib_spec.precursor_mz is not None:
            try:
                lib_meta["precursor_mz"] = float(lib_spec.precursor_mz)
            except Exception:
                lib_meta["precursor_mz"] = lib_spec.precursor_mz

        mz = lib_spec.mz
        intensities = lib_spec.intensities
        if threshold and intensities.size > 0:
            max_lib_int = float(np.max(intensities)) if np.max(intensities) > 0 else 1.0
            keep_mask = intensities >= (max_lib_int * (threshold / 100.0))
            if np.any(keep_mask):
                mz = mz[keep_mask]
                intensities = intensities[keep_mask]

        lib_spec.matchms_spectrum = _to_matchms_spectrum(mz, intensities, metadata=lib_meta)
        lib_spec.matchms_prepared_fragment_min_rel_abundance = threshold

    return library_spectra


def _relative_abundances(mz, intensities, matched_mask):
    """Return the relative abundance (%) of matched peaks, normalized so ALL peaks sum/scale to 100%."""
    if intensities.size == 0:
        return []
    max_intensity = float(np.max(intensities)) if np.max(intensities) > 0 else 1.0
    return [{"mz": float(m), "relative_intensity": float(i) / max_intensity * 100.0} for m, i, is_matched in zip(mz, intensities, matched_mask) if is_matched]


def match_spectrum_against_library(
    exp_mz,
    exp_intensities,
    exp_polarity,
    exp_precursor_mz,
    library_spectra,
    algorithm_name=DEFAULT_MSMS_LIBRARY_ALGORITHM,
    mz_tolerance=0.01,
    precursor_mz_tolerance=0.01,
    min_matched_peaks=4,
    score_cutoff=0.8,
    fragment_min_rel_abundance=0.0,
    require_same_precursor_mz=True,
):
    """
    Compare one experimental MS2 spectrum against all library spectra with matching polarity
    and matching precursor m/z (within precursor_mz_tolerance, if both precursor m/z values
    are known), using the given matchms algorithm.

    Returns a list of match dicts sorted by descending score, each with keys:
        score, matched_peaks, compound_name, db_spectrum_index, source_file,
        exp_matched_abundances (list of {mz, relative_intensity} relative to exp spectrum),
        db_matched_abundances (list of {mz, relative_intensity} relative to db spectrum)
    """
    if not MATCHMS_AVAILABLE:
        raise RuntimeError("The 'matchms' package is not available.")

    exp_mz = np.asarray(exp_mz, dtype=np.float64)
    exp_intensities = np.asarray(exp_intensities, dtype=np.float64)
    if exp_mz.size == 0:
        return []

    algorithm = get_similarity_algorithm(algorithm_name, mz_tolerance)
    # Provide precursor_mz in the metadata for matchms (some scorers expect this slot)
    # Optionally pre-filter experimental fragments below the given relative-abundance threshold
    if fragment_min_rel_abundance and exp_intensities.size > 0:
        max_int = float(np.max(exp_intensities)) if np.max(exp_intensities) > 0 else 1.0
        keep_mask = exp_intensities >= (max_int * (float(fragment_min_rel_abundance) / 100.0))
        # if filtering removes all peaks, keep original arrays to avoid empty spectrum
        if np.any(keep_mask):
            exp_mz = exp_mz[keep_mask]
            exp_intensities = exp_intensities[keep_mask]

    exp_meta = {}
    if exp_precursor_mz is not None:
        try:
            exp_meta["precursor_mz"] = float(exp_precursor_mz)
        except Exception:
            exp_meta["precursor_mz"] = exp_precursor_mz
    exp_spectrum = _to_matchms_spectrum(exp_mz, exp_intensities, metadata=exp_meta)

    results = []
    for lib_spec in library_spectra:
        # a library spectrum without a precursor m/z or fragment peaks lacks information required
        # for matching and is always excluded
        if lib_spec.precursor_mz is None or lib_spec.mz.size == 0:
            continue
        # optional polarity check
        if exp_polarity and lib_spec.polarity and lib_spec.polarity != exp_polarity:
            continue
        # optional precursor m/z matching check (only if both are known and requirement enabled)
        if require_same_precursor_mz and exp_precursor_mz is not None:
            if abs(exp_precursor_mz - lib_spec.precursor_mz) > precursor_mz_tolerance:
                continue

        lib_matchms_spectrum = lib_spec.matchms_spectrum
        if lib_matchms_spectrum is None:
            prepare_library_spectra([lib_spec], fragment_min_rel_abundance=fragment_min_rel_abundance)
            lib_matchms_spectrum = lib_spec.matchms_spectrum

        try:
            score_result = algorithm.pair(exp_spectrum, lib_matchms_spectrum)
        except Exception as e:
            logging.warning(f"MSMS library matching failed for '{lib_spec.compound_name}': {e}")
            continue

        score = float(score_result["score"])
        matches = int(score_result["matches"])

        if matches < min_matched_peaks or score < score_cutoff:
            continue

        # Determine which experimental / library peaks were "matched" within tolerance,
        # for relative-abundance reporting.
        exp_matched_mask = np.zeros(exp_mz.size, dtype=bool)
        db_matched_mask = np.zeros(lib_spec.mz.size, dtype=bool)
        for i, mz_a in enumerate(exp_mz):
            j = int(np.argmin(np.abs(lib_spec.mz - mz_a)))
            if abs(lib_spec.mz[j] - mz_a) <= mz_tolerance:
                exp_matched_mask[i] = True
                db_matched_mask[j] = True

        # include precursor information and precursor difference (exp - db) when available
        prec_diff = None
        try:
            if exp_precursor_mz is not None and lib_spec.precursor_mz is not None:
                prec_diff = float(exp_precursor_mz) - float(lib_spec.precursor_mz)
        except Exception:
            prec_diff = None

        results.append(
            {
                "score": score,
                "matched_peaks": matches,
                "compound_name": lib_spec.compound_name,
                "db_spectrum_index": lib_spec.spectrum_index,
                "source_file": os.path.basename(lib_spec.source_file),
                "exp_matched_abundances": _relative_abundances(exp_mz, exp_intensities, exp_matched_mask),
                "db_matched_abundances": _relative_abundances(lib_spec.mz, lib_spec.intensities, db_matched_mask),
                "db_precursor_mz": lib_spec.precursor_mz,
                "exp_precursor_mz": exp_precursor_mz,
                "precursor_mz_diff": prec_diff,
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results
