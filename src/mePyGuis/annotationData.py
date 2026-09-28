"""Read-only access layer for the "5_Annotated_MSMS" / "5_Annotated_Compounds" /
"5_Annotated_SumFormulas" sheets of a bracketed results file, shared by the
per-feature "Feature annotations" panel and the "Annotation browser" tree.
"""

from __future__ import annotations

import logging

from ..formulaTools import formulaTools

MSMS_SHEET = "5_Annotated_MSMS"
COMPOUNDS_SHEET = "5_Annotated_Compounds"
SUMFORMULAS_SHEET = "5_Annotated_SumFormulas"

_fT = formulaTools()


def formula_to_mass(formula) -> float | None:
    """Compute the monoisotopic neutral mass of a sum-formula string (e.g. "C6H12O6"),
    or return None if it cannot be parsed."""
    if not formula:
        return None
    try:
        elems, charge = _fT.parseFormulaWithCharge(str(formula))
        return _fT.calcMolWeight(elems, charge=charge)
    except Exception:
        return None


class AnnotationStore:
    """Lazily indexes the three annotation sheets of a `PolarsDB` results file by
    `Feature_Num`, so lookups for a single selected feature (or a full tree
    rebuild) are cheap after the first access. A missing sheet simply yields
    an empty result everywhere (the caller just skips that section)."""

    def __init__(self, db_con, main_table_name=None):
        self.db_con = db_con
        self.main_table_name = main_table_name
        self._msms_rows = None
        self._compound_rows = None
        self._sumformula_rows = None
        self._avg_abundance_by_num = None

    def _has(self, sheet_name):
        return self.db_con is not None and self.db_con.has_table(sheet_name)

    def _average_abundance(self, feature_num):
        if self._avg_abundance_by_num is None:
            self._avg_abundance_by_num = {}
            if self.main_table_name and self.db_con is not None and self.db_con.has_table(self.main_table_name):
                main_df = self.db_con.tables[self.main_table_name]
                if "Num" in main_df.columns and "Average_peakarea" in main_df.columns:
                    for row in main_df.select(["Num", "Average_peakarea"]).to_dicts():
                        self._avg_abundance_by_num[row["Num"]] = row["Average_peakarea"]
        return self._avg_abundance_by_num.get(feature_num)

    def _load_rows(self, sheet_name):
        if not self._has(sheet_name):
            return []
        try:
            return self.db_con.tables[sheet_name].to_dicts()
        except Exception:
            logging.exception("Failed to read sheet '%s' for annotation display", sheet_name)
            return []

    def has_msms(self):
        return self._has(MSMS_SHEET)

    def has_compounds(self):
        return self._has(COMPOUNDS_SHEET)

    def has_sumformulas(self):
        return self._has(SUMFORMULAS_SHEET)

    def all_msms_rows(self):
        if self._msms_rows is None:
            self._msms_rows = self._load_rows(MSMS_SHEET)
            for row in self._msms_rows:
                row["Feature_Average_peakarea"] = self._average_abundance(row.get("Feature_Num"))
        return self._msms_rows

    def all_compound_rows(self):
        if self._compound_rows is None:
            self._compound_rows = self._load_rows(COMPOUNDS_SHEET)
        return self._compound_rows

    def all_sumformula_rows(self):
        if self._sumformula_rows is None:
            self._sumformula_rows = self._load_rows(SUMFORMULAS_SHEET)
        return self._sumformula_rows

    def msms_for_feature(self, feature_num):
        return [r for r in self.all_msms_rows() if r.get("Feature_Num") == feature_num]

    def compounds_for_feature(self, feature_num):
        return [r for r in self.all_compound_rows() if r.get("Feature_Num") == feature_num]

    def sumformulas_for_feature(self, feature_num):
        return [r for r in self.all_sumformula_rows() if r.get("Feature_Num") == feature_num]

    def invalidate(self):
        """Force all cached indices to be rebuilt on next access (e.g. after re-annotation)."""
        self._msms_rows = None
        self._compound_rows = None
        self._sumformula_rows = None
        self._avg_abundance_by_num = None


def get_structure_smiles(row: dict) -> str | None:
    """Return the best-available SMILES string for a compound/MSMS annotation row, or None."""
    for key in ("SMILES", "DB_Info_SMILES", "smiles"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def format_formula_with_mass(formula, mass) -> str:
    """Format a sum formula together with its theoretical mass in brackets, e.g.
    "C6H12O6 (180.0634)". Falls back to just the formula, or "" if neither is available."""
    if not formula:
        return ""
    if mass is None or mass == "":
        return str(formula)
    try:
        return f"{formula} ({float(mass):.4f})"
    except (TypeError, ValueError):
        return str(formula)


def format_meta_info(fields: list[tuple[str, object]]) -> str:
    """Join non-empty (label, value) pairs into a single "label: value; label2: value2" string,
    used to display miscellaneous metadata compactly in a single tree/table column."""
    parts = [f"{label}: {value}" for label, value in fields if value is not None and value != ""]
    return "; ".join(parts)
