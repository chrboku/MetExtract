from __future__ import absolute_import, division, print_function
import json
import logging

from .formulaTools import formulaTools

# Isotopologs compared for the isotope-pattern score: (name in Other_Isotopologs, element,
# isotope key in formulaTools.elemDetails, number of labelled atoms)
ISOTOPOLOGS = [
    ("M+13C", "C", "13C", 1),
    ("M+13C2", "C", "13C", 2),
    ("M+15N", "N", "15N", 1),
    ("M+34S", "S", "34S", 1),
    ("M+37Cl", "Cl", "37Cl", 1),
    ("M-54Fe", "Fe", "54Fe", 1),
]
ISOTOPOLOG_NAMES = [name for name, _, _, _ in ISOTOPOLOGS]

# sentinel returned when the score cannot be computed (unparseable formula, missing data, ...)
UNDEFINED_SCORE = -1.0

_fT = formulaTools()


def parse_other_isotopologs(other_isotopologs_json):
    """Parse the "Other_Isotopologs" JSON column value into a dict, or None if missing/invalid."""
    if not other_isotopologs_json:
        return None
    try:
        parsed = json.loads(other_isotopologs_json)
        return parsed if parsed else None
    except (TypeError, ValueError) as e:
        logging.debug(f"Could not parse Other_Isotopologs value '{other_isotopologs_json}': {e}")
        return None


def parse_sum_formula(sum_formula):
    """Parse a sum-formula string into an element-count dict, or None if it can't be parsed."""
    if not sum_formula:
        return None
    try:
        return _fT.parseFormula(str(sum_formula))
    except Exception as e:
        logging.debug(f"Could not parse sum formula '{sum_formula}': {e}")
        return None


def theoretical_isotopolog_ratio(elems, element_symbol, isotope_key, count):
    """Theoretical abundance ratio (relative to the monoisotopic form, M=1) of the given
    isotopolog for a parsed sum formula, or 0.0 if the formula doesn't contain enough atoms
    of the required element."""
    if elems.get(element_symbol, 0) < count:
        return 0.0
    labeled = dict(elems)
    labeled[isotope_key] = count
    return _fT.getAbundanceToMonoisotopic(labeled)


def compute_isotope_pattern(sum_formula, other_isotopologs_json):
    """Compare the theoretical isotope pattern of `sum_formula` to the observed pattern (median
    values) in `other_isotopologs_json`, over the 6 isotopologs in ISOTOPOLOGS.

    Returns (score, ratios):
        score: sum of the absolute differences (measured - theoretical) over the 6 isotopologs,
            or UNDEFINED_SCORE if it cannot be computed.
        ratios: dict {isotopolog_name: {"measured": float, "theoretical": float}}, or {} if the
            score is undefined.
    """
    observed = parse_other_isotopologs(other_isotopologs_json)
    if observed is None or not sum_formula:
        return UNDEFINED_SCORE, {}

    elems = parse_sum_formula(sum_formula)
    if elems is None:
        return UNDEFINED_SCORE, {}

    try:
        ratios = {}
        abs_errors = []
        for name, element_symbol, isotope_key, count in ISOTOPOLOGS:
            stats = observed.get(name) or {}
            measured = stats.get("median") or 0.0
            theoretical = theoretical_isotopolog_ratio(elems, element_symbol, isotope_key, count)
            ratios[name] = {"measured": measured, "theoretical": theoretical}
            abs_errors.append(abs(measured - theoretical))
    except Exception as e:
        logging.debug(f"Could not compute isotope pattern score for formula '{sum_formula}': {e}")
        return UNDEFINED_SCORE, {}

    return sum(abs_errors), ratios


def isotope_ratio_json_fields(ratios):
    """Flatten a ratios dict (as returned by compute_isotope_pattern) into
    {"isotope_ratios_<name>": {"measured":..., "theoretical":...}, ...} for merging directly into
    a per-hit JSON entry."""
    return {f"isotope_ratios_{name}": ratios.get(name) for name in ISOTOPOLOG_NAMES}


def isotope_ratio_columns(ratios):
    """Flatten a ratios dict (as returned by compute_isotope_pattern) into flat
    "Iso_R_<name>_M"/"Iso_R_<name>_T" columns for merging into a
    per-hit sheet row. Zero ratios are reported as None so the cells stay empty."""
    cols = {}
    for name in ISOTOPOLOG_NAMES:
        entry = ratios.get(name)
        measured = entry["measured"] if entry else None
        theoretical = entry["theoretical"] if entry else None
        cols[f"Iso_R_{name}_M"] = measured if measured else None
        cols[f"Iso_R_{name}_T"] = theoretical if theoretical else None
    return cols


def format_isotope_ratios_for_text(ratios):
    """Render the per-isotopolog measured/theoretical dict as 'isotope_ratios_X: {...}, ...' for
    embedding into free-text hit descriptions (e.g. the database-search hit strings)."""
    return ", ".join(f"isotope_ratios_{name}: {json.dumps(ratios.get(name))}" for name in ISOTOPOLOG_NAMES)
