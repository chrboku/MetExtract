"""Top-level "Annotation browser" dock: a 4-level tree (Type -> Library/Database ->
Compound/Formula -> Feature) built from the "5_Annotated_MSMS" / "5_Annotated_Compounds" /
"5_Annotated_SumFormulas" sheets, letting the user browse annotation results across the
whole experiment (as opposed to the per-feature "Feature annotations" panel).
"""

from __future__ import annotations

from collections import defaultdict

from PySide6 import QtCore, QtGui, QtWidgets

from .annotationData import AnnotationStore, format_formula_with_mass, format_meta_info, formula_to_mass

# Column 0 is the entry name; columns 1-8 describe the feature the row belongs to and are only
# filled in for actual feature (leaf) rows. For compound/formula (aggregate) rows, these cells
# instead show additional "key: value" metadata (in grey) since no single feature applies there.
# Columns 9-15 are populated differently depending on the branch (MS/MS, Compound, Sum formula).
COLUMNS = [
    "Entry",
    "Feature Num",
    "OGroup",
    "Feature m/z",
    "Feature RT",
    "Xn",
    "Z",
    "Polarity",
    "Avg. abundance",
    "Sum formula",
    "Name",
    "RT",
    "Precursor/DB m/z",
    "Score",
    "Fragments/Hit type",
    "Meta info",
]

# Literal (case-insensitive) database/library metadata field names carrying acquisition info,
# mapped to a canonical label; entries sharing a label are de-duplicated (first match wins) so
# that e.g. "FRAGMENTATION_MODE" and "Fragmentation_Mode" don't produce separate columns.
_DB_META_FIELD_LABELS = {
    "instrument": "Instrument",
    "instrument_type": "Instrument type",
    "fragmentation_mode": "Fragmentation",
    "collision_energy": "Collision energy",
    "collisioin_energy": "Collision energy",
}

_FEATURE_HIT_BACKGROUND = QtGui.QColor(230, 230, 230)
_META_TEXT_COLOR = QtGui.QColor(140, 140, 140)


def _resolve_msms_meta_pairs(row):
    """Resolve instrument/fragmentation/collision-energy metadata of a MS/MS compound's
    database entry, from whichever (case-insensitively matching) column is present.
    Returns a list of (label, value) pairs."""
    found = {}
    for key, value in row.items():
        if value in (None, ""):
            continue
        label = _DB_META_FIELD_LABELS.get(key.lower())
        if label and label not in found:
            found[label] = value
    return list(found.items())


class AnnotationBrowserWidget(QtWidgets.QWidget):
    """Experiment-wide, annotation-centric browser. Emits `featureSelected(int)` with the
    feature's "Num" when the user clicks a feature (leaf) row."""

    featureSelected = QtCore.Signal(int)
    filterMetabolitesRequested = QtCore.Signal(list, list)  # (ogroups, nums)

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by compound name / formula / library / database ...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        layout.addLayout(filter_row)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree, 1)

        self._id_filter_visible_nums = None  # None = no active Experiment results ID filter

    def clear(self):
        self.tree.clear()

    def set_id_filter(self, visible_nums):
        """Called by MExtract whenever the Experiment results OGroup/Num ID filter changes.
        `visible_nums` is the set of feature Nums still visible in the Experiment results tree
        (None = no filter active). Hides feature rows not in the set, and any
        library/compound/type row left with no visible feature underneath it."""
        self._id_filter_visible_nums = visible_nums
        self._apply_filter(self.filter_edit.text())

    def _show_context_menu(self, position):
        """Right-click menu for a non-feature (aggregate) row: lets the user filter the
        Experiment results tree/volcano plots by all OGroups/Nums nested under this entry."""
        item = self.tree.itemAt(position)
        if item is None or item.data(0, QtCore.Qt.UserRole) is not None:
            return  # only offered for non-leaf (aggregate) rows, not individual feature entries

        ogroups = set()
        nums = set()

        def _collect(node):
            num = node.data(0, QtCore.Qt.UserRole)
            if num is not None:
                nums.add(num)
                ogroup_text = node.text(2)
                if ogroup_text:
                    ogroups.add(ogroup_text)
            for c in range(node.childCount()):
                _collect(node.child(c))

        _collect(item)
        if not ogroups and not nums:
            return

        menu = QtWidgets.QMenu(self.tree)
        action = menu.addAction("Filter these metabolites")
        chosen = menu.exec(self.tree.viewport().mapToGlobal(position))
        if chosen is action:
            self.filterMetabolitesRequested.emit(sorted(ogroups), sorted(nums))

    def load(self, store: AnnotationStore):
        """(Re)build the whole tree from the given `AnnotationStore`."""
        self.tree.clear()

        if store.has_msms():
            self._add_msms_branch(store.all_msms_rows())
        if store.has_compounds():
            self._add_compound_branch(store.all_compound_rows())
        if store.has_sumformulas():
            self._add_sumformula_branch(store.all_sumformula_rows())

        self.tree.expandToDepth(0)
        for col in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(col)
        self._apply_filter(self.filter_edit.text())

    # ------------------------------------------------------------------ MS/MS
    def _add_msms_branch(self, rows):
        if not rows:
            return
        type_item = QtWidgets.QTreeWidgetItem([f"MS/MS ({len(rows)})"])
        self.tree.addTopLevelItem(type_item)

        by_library = defaultdict(list)
        for row in rows:
            by_library[row.get("Library") or "(unknown)"].append(row)

        for library_name, library_rows in sorted(by_library.items(), key=lambda kv: str(kv[0])):
            library_item = QtWidgets.QTreeWidgetItem([f"{library_name} ({len(library_rows)})"])
            type_item.addChild(library_item)

            by_compound = defaultdict(list)
            for row in library_rows:
                by_compound[(row.get("Compound_Name") or "(unnamed)", row.get("Compound_ID"))].append(row)

            for (compound_name, compound_id), compound_rows in sorted(by_compound.items(), key=lambda kv: str(kv[0][0])):
                first = compound_rows[0]
                compound_item = QtWidgets.QTreeWidgetItem(
                    [f"{compound_name} (ID: {compound_id})" if compound_id is not None else str(compound_name)]
                    + _meta_cells(_resolve_msms_meta_pairs(first))
                    + [
                        format_formula_with_mass(first.get("Formula"), first.get("TheoreticalMass")),
                        str(first.get("NAME", "")),
                        _fmt(first.get("Library_RT")),
                        _fmt(first.get("Library_Precursor_MZ")),
                        "",
                        "",
                        "",
                    ]
                )
                _style_meta_row(compound_item)
                library_item.addChild(compound_item)

                for row in compound_rows:
                    score = row.get("Score")
                    feature_item = QtWidgets.QTreeWidgetItem(
                        [_fmt(score) if score is not None else "(match)"]
                        + _feature_columns(row)
                        + [
                            format_formula_with_mass(row.get("Formula"), row.get("TheoreticalMass")),
                            str(row.get("NAME", "")),
                            _fmt(row.get("Scan_RT")),
                            _fmt(row.get("Scan_Precursor_MZ")),
                            _fmt(score),
                            _fmt(row.get("Matched_Fragments")),
                            format_meta_info([("Sample", row.get("Sample_File"))]),
                        ]
                    )
                    feature_item.setData(0, QtCore.Qt.UserRole, row.get("Feature_Num"))
                    _mark_as_hit(feature_item)
                    compound_item.addChild(feature_item)

    # -------------------------------------------------------------- Compound
    def _add_compound_branch(self, rows):
        if not rows:
            return
        type_item = QtWidgets.QTreeWidgetItem([f"Compound ({len(rows)})"])
        self.tree.addTopLevelItem(type_item)

        by_db = defaultdict(list)
        for row in rows:
            by_db[row.get("DB_Name") or "(unknown)"].append(row)

        for db_name, db_rows in sorted(by_db.items(), key=lambda kv: str(kv[0])):
            db_item = QtWidgets.QTreeWidgetItem([f"{db_name} ({len(db_rows)})"])
            type_item.addChild(db_item)

            by_compound = defaultdict(list)
            for row in db_rows:
                by_compound[(row.get("DB_CompoundName") or "(unnamed)", row.get("DB_Num"))].append(row)

            for (compound_name, compound_num), compound_rows in sorted(by_compound.items(), key=lambda kv: str(kv[0][0])):
                first = compound_rows[0]
                info_fields = [(k[len("DB_Info_") :], v) for k, v in first.items() if k.startswith("DB_Info_") and k != "DB_Info_SMILES" and v]
                compound_item = QtWidgets.QTreeWidgetItem(
                    [f"{compound_name} (ID: {compound_num})" if compound_num is not None else str(compound_name)]
                    + _meta_cells(info_fields)
                    + [
                        format_formula_with_mass(first.get("DB_SumFormula"), first.get("DB_Mass")),
                        str(compound_name),
                        _fmt(first.get("DB_RT_min")),
                        _fmt(first.get("DB_MZ")),
                        "",
                        "",
                        "",
                    ]
                )
                _style_meta_row(compound_item)
                db_item.addChild(compound_item)

                for row in compound_rows:
                    feature_item = QtWidgets.QTreeWidgetItem(["Feature"] + _feature_columns(row) + ["", "", "", "", "", str(row.get("HitType", "")), ""])
                    feature_item.setData(0, QtCore.Qt.UserRole, row.get("Feature_Num"))
                    _mark_as_hit(feature_item)
                    compound_item.addChild(feature_item)

    # ----------------------------------------------------------- Sum formula
    def _add_sumformula_branch(self, rows):
        if not rows:
            return
        type_item = QtWidgets.QTreeWidgetItem([f"Sum formula ({len(rows)})"])
        self.tree.addTopLevelItem(type_item)

        by_class = defaultdict(list)
        for row in rows:
            by_class[row.get("Element_Class") or "(unknown)"].append(row)

        for element_class, class_rows in sorted(by_class.items(), key=lambda kv: str(kv[0])):
            class_item = QtWidgets.QTreeWidgetItem([f"{element_class} ({len(class_rows)})"])
            type_item.addChild(class_item)

            by_formula = defaultdict(list)
            for row in class_rows:
                by_formula[row.get("SumFormula") or "(unknown)"].append(row)

            for formula, formula_rows in sorted(by_formula.items(), key=lambda kv: str(kv[0])):
                formula_item = QtWidgets.QTreeWidgetItem(
                    [str(formula)]
                    + _meta_cells([])
                    + [
                        format_formula_with_mass(formula, formula_to_mass(formula)),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                _style_meta_row(formula_item)
                class_item.addChild(formula_item)

                for row in formula_rows:
                    feature_item = QtWidgets.QTreeWidgetItem(["Feature"] + _feature_columns(row) + ["", "", "", "", "", "", ""])
                    feature_item.setData(0, QtCore.Qt.UserRole, row.get("Feature_Num"))
                    _mark_as_hit(feature_item)
                    formula_item.addChild(feature_item)

    def _on_item_clicked(self, item, _column):
        feature_num = item.data(0, QtCore.Qt.UserRole)
        if feature_num is not None:
            self.featureSelected.emit(feature_num)

    def _apply_filter(self, text):
        text = text.strip().lower()
        visible_nums = self._id_filter_visible_nums
        for i in range(self.tree.topLevelItemCount()):
            type_item = self.tree.topLevelItem(i)
            type_visible = False
            for j in range(type_item.childCount()):
                library_item = type_item.child(j)
                library_visible = False
                for k in range(library_item.childCount()):
                    compound_item = library_item.child(k)
                    text_match = not text or text in compound_item.text(0).lower() or text in library_item.text(0).lower()

                    has_visible_feature = visible_nums is None
                    for f in range(compound_item.childCount()):
                        feature_item = compound_item.child(f)
                        num = feature_item.data(0, QtCore.Qt.UserRole)
                        id_match = visible_nums is None or num in visible_nums
                        feature_item.setHidden(not id_match)
                        has_visible_feature |= id_match

                    match = text_match and has_visible_feature
                    compound_item.setHidden(not match)
                    library_visible |= match
                library_item.setHidden(not library_visible)
                type_visible |= library_visible
            type_item.setHidden(not type_visible)


def _mark_as_hit(item):
    """Give a feature (leaf) row a light grey background to help visually distinguish it during navigation."""
    for col in range(item.columnCount()):
        item.setBackground(col, _FEATURE_HIT_BACKGROUND)


def _meta_cells(pairs, n=8):
    """Distribute up to `n` non-empty (label, value) pairs, one per column, as "label: value"
    strings. Used to fill the feature-info columns (otherwise unused) of compound/formula rows
    with auxiliary metadata."""
    cells = [""] * n
    filtered = [(label, value) for label, value in pairs if value not in (None, "")]
    for i, (label, value) in enumerate(filtered[:n]):
        cells[i] = f"{label}: {_fmt(value)}"
    return cells


def _style_meta_row(item):
    """Render a compound/formula-level row's feature-info columns (1-8) in grey, since they hold
    auxiliary "key: value" metadata rather than an actual feature's data."""
    brush = QtGui.QBrush(_META_TEXT_COLOR)
    for col in range(1, 9):
        item.setForeground(col, brush)


def _feature_columns(row):
    return [
        str(row.get("Feature_Num", "")),
        str(row.get("Feature_OGroup", "")),
        _fmt(row.get("Feature_MZ")),
        _fmt(row.get("Feature_RT")),
        str(row.get("Feature_Xn", "")),
        str(row.get("Feature_Charge", "")),
        str(row.get("Feature_Ionisation_Mode", "")),
        _fmt(row.get("Feature_Average_peakarea")),
    ]


def _fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
