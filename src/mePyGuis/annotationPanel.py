"""Vertical, collapsible-card list of the annotation results ("5_Annotated_MSMS" /
"5_Annotated_Compounds" / "5_Annotated_SumFormulas") for a single selected feature, shown
as a new panel inside the "Experiment results" dock area.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from .annotationData import AnnotationStore, get_structure_smiles
from .structureRenderer import render_structure_pixmap


class _AnnotationCard(QtWidgets.QWidget):
    """A single collapsible entry. By default (`lazy=True`) the body (metadata grid + structure
    + optional mirror plot) is only built the first time it is expanded; pass `lazy=False` to
    build it immediately (used for the type/database grouping levels, so filtering still works
    without the user having to expand every section first)."""

    def __init__(self, title, build_body_fn, parent=None, lazy=True, start_expanded=False):
        super().__init__(parent)
        self._build_body_fn = build_body_fn
        self._body_built = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle_btn = QtWidgets.QPushButton()
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet("text-align: left; padding: 4px 8px; font-weight: bold;")
        layout.addWidget(self.toggle_btn)

        self.body = QtWidgets.QWidget()
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(16, 4, 4, 8)
        layout.addWidget(self.body)

        self._title = title
        self.toggle_btn.toggled.connect(self._on_toggled)

        if not lazy:
            self._build_body_fn(self.body_layout)
            self._body_built = True
        self.toggle_btn.setChecked(start_expanded)
        self._on_toggled(start_expanded)

    def _on_toggled(self, checked):
        arrow = "\u25bc" if checked else "\u25b6"
        self.toggle_btn.setText(f"{arrow} {self._title}")
        if checked and not self._body_built:
            self._build_body_fn(self.body_layout)
            self._body_built = True
        self.body.setVisible(checked)


def _fmt_score(value):
    try:
        return f"{float(value):.2f}" if value not in (None, "") else value
    except (TypeError, ValueError):
        return value


def _fmt_mz(value):
    try:
        return f"{float(value):.4f}" if value not in (None, "") else value
    except (TypeError, ValueError):
        return value


def _fmt_rt(value):
    try:
        return f"{float(value):.2f} minutes" if value not in (None, "") else value
    except (TypeError, ValueError):
        return value


def _fmt_abundance(value):
    try:
        return f"{float(value):.2e}" if value not in (None, "") else value
    except (TypeError, ValueError):
        return value


def _add_metadata_grid(layout, fields):
    """Add a compact "label: value" grid, laid out as 3 side-by-side pairs (6 columns), for the
    given ordered [(label, value), ...] pairs, skipping empty/None values. Pairs are filled
    column-wise (top-to-bottom within a column, then continuing in the next column)."""
    non_empty = [(label, value) for label, value in fields if value is not None and value != ""]
    grid = QtWidgets.QGridLayout()
    grid.setHorizontalSpacing(12)
    grid.setVerticalSpacing(2)
    pairs_per_row = 3
    n_rows = math.ceil(len(non_empty) / pairs_per_row) if non_empty else 0
    for i, (label, value) in enumerate(non_empty):
        pair_idx, row = divmod(i, n_rows)
        label_col = pair_idx * 2
        grid.addWidget(QtWidgets.QLabel(f"<b>{label}:</b>"), row, label_col, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        value_label = QtWidgets.QLabel(str(value))
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        grid.addWidget(value_label, row, label_col + 1, QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
    for pair_idx in range(pairs_per_row):
        grid.setColumnStretch(pair_idx * 2 + 1, 1)
    container = QtWidgets.QWidget()
    container.setLayout(grid)
    layout.addWidget(container)


def _extra_fields(row, known_keys):
    """Return (label, value) pairs for any row columns not already explicitly shown, so extra
    library/database columns from the results sheet are never silently hidden."""
    return [(k, v) for k, v in row.items() if k not in known_keys and v not in (None, "")]


def _add_structure_image(layout, smiles):
    if not smiles:
        return
    pixmap = render_structure_pixmap(smiles)
    if pixmap is None:
        return
    label = QtWidgets.QLabel()
    label.setPixmap(pixmap)
    layout.addWidget(label)


def _add_mirror_plot(layout, exp_scan, lib_spectrum):
    """Build a head-to-tail mirror plot (experimental above, library below) if both spectra
    are available, and add it to `layout`; returns the created canvas, or None if skipped."""
    if exp_scan is None or lib_spectrum is None:
        return None
    exp_mz = np.asarray(getattr(exp_scan, "mz_list", []), dtype=float)
    exp_intens = np.asarray(getattr(exp_scan, "intensity_list", []), dtype=float)
    lib_mz = np.asarray(getattr(lib_spectrum, "mz", []), dtype=float)
    lib_intens = np.asarray(getattr(lib_spectrum, "intensities", []), dtype=float)
    if len(exp_mz) == 0 or len(lib_mz) == 0:
        return None

    fig = Figure(figsize=(4.5, 2.8), dpi=80, facecolor="white")
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    exp_norm = exp_intens / exp_intens.max() * 100.0 if exp_intens.max() > 0 else exp_intens
    lib_norm = lib_intens / lib_intens.max() * 100.0 if lib_intens.max() > 0 else lib_intens
    ax.vlines(exp_mz, 0, exp_norm, colors="dodgerblue", linewidth=1.2, label="Experimental")
    ax.vlines(lib_mz, 0, -lib_norm, colors="firebrick", linewidth=1.2, label="Library")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("m/z", fontsize=5)
    ax.set_ylabel("Rel. intensity (%)", fontsize=5)
    ax.tick_params(axis="both", labelsize=5)
    ax.legend(loc="upper right", fontsize=4)
    fig.tight_layout()
    canvas.setMinimumHeight(220)
    layout.addWidget(canvas)
    return canvas


def _add_structure_and_match_row(layout, smiles, mirror_plot_fn):
    """Add the structure image and the MSMS mirror plot ("MSMS Match") side by side in one
    row, with the structure given 30% and the mirror plot 70% of the available width."""
    structure_pixmap = render_structure_pixmap(smiles) if smiles else None

    row_widget = QtWidgets.QWidget()
    row_layout = QtWidgets.QHBoxLayout(row_widget)
    row_layout.setContentsMargins(0, 0, 0, 0)

    structure_container = QtWidgets.QWidget()
    structure_layout = QtWidgets.QVBoxLayout(structure_container)
    structure_layout.setContentsMargins(0, 0, 0, 0)
    if structure_pixmap is not None:
        structure_label = QtWidgets.QLabel()
        structure_label.setPixmap(structure_pixmap)
        structure_layout.addWidget(structure_label)
    row_layout.addWidget(structure_container, 3)

    match_container = QtWidgets.QWidget()
    match_layout = QtWidgets.QVBoxLayout(match_container)
    match_layout.setContentsMargins(0, 0, 0, 0)
    mirror_plot_fn(match_layout)
    row_layout.addWidget(match_container, 7)

    if structure_pixmap is not None or match_container.layout().count():
        layout.addWidget(row_widget)


class FeatureAnnotationsPanel(QtWidgets.QWidget):
    """Panel showing the MS/MS-, database- and sum-formula annotation results of whichever
    feature is currently selected in the "Experiment results" tree. Populated from the
    "5_Annotated_MSMS"/"5_Annotated_Compounds"/"5_Annotated_SumFormulas" sheets (a missing
    sheet simply hides that section)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store: AnnotationStore | None = None
        self._library_cache = None
        self._get_experimental_scan = None
        self._current_feature_num = None

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter:"))
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by compound name / formula / library ...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self.filter_edit, 1)
        outer_layout.addLayout(filter_row)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        outer_layout.addWidget(self.scroll_area, 1)

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(QtCore.Qt.AlignTop)
        self.scroll_area.setWidget(self.content_widget)

        self.no_feature_label = QtWidgets.QLabel("No feature selected.")
        self.content_layout.addWidget(self.no_feature_label)

        self._section_cards = []  # [(section_title, filter_text, card_widget), ...]

    def configure(self, get_experimental_scan=None):
        """*get_experimental_scan(feature_num)* should return a scan-like object with
        `mz_list`/`intensity_list` attributes for the mirror plot, or None if unavailable."""
        self._get_experimental_scan = get_experimental_scan

    def clear(self):
        self._current_feature_num = None
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._section_cards = []
        self.content_layout.addWidget(QtWidgets.QLabel("No feature selected."))

    def show_feature(self, feature_num, store: AnnotationStore, library_cache=None):
        self._store = store
        self._library_cache = library_cache
        self._current_feature_num = feature_num

        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._section_cards = []

        any_section = False
        if store.has_msms():
            any_section |= self._add_type_section("MS/MS", store.msms_for_feature(feature_num), "Library", self._build_msms_card)
        if store.has_compounds():
            any_section |= self._add_type_section("Compound", store.compounds_for_feature(feature_num), "DB_Name", self._build_compound_card)
        if store.has_sumformulas():
            any_section |= self._add_type_section("Sum formula", store.sumformulas_for_feature(feature_num), "Element_Class", self._build_sumformula_card)

        if not any_section:
            self.content_layout.addWidget(QtWidgets.QLabel("No annotation results available for this feature."))

    def _add_type_section(self, type_title, rows, group_key, build_card_fn):
        """Add a collapsible type-level section (e.g. "MS/MS"), containing one collapsible
        sub-section per distinct `group_key` value (e.g. per library/database), each holding
        the individual (lazily-built) annotation cards."""
        if not rows:
            return False

        groups = defaultdict(list)
        for row in rows:
            groups[row.get(group_key) or "(unknown)"].append(row)

        def build_type_body(layout):
            for group_name, group_rows in sorted(groups.items(), key=lambda kv: str(kv[0])):

                def build_group_body(inner_layout, group_rows=group_rows):
                    for row in group_rows:
                        card = build_card_fn(row)
                        inner_layout.addWidget(card)
                        self._section_cards.append((type_title, self._filter_text_for_row(type_title, row), card))

                group_card = _AnnotationCard(f"{group_name} ({len(group_rows)})", build_group_body, lazy=False, start_expanded=True)
                layout.addWidget(group_card)

        type_card = _AnnotationCard(f"{type_title} ({len(rows)})", build_type_body, lazy=False, start_expanded=True)
        self.content_layout.addWidget(type_card)
        return True

    @staticmethod
    def _filter_text_for_row(section_title, row):
        return " ".join(str(v) for v in row.values() if v is not None).lower()

    def _apply_filter(self, text):
        text = text.strip().lower()
        for _title, filter_text, card in self._section_cards:
            card.setVisible(text in filter_text if text else True)

    def _build_msms_card(self, row):
        compound_name = row.get("Compound_Name") or "(unnamed spectrum)"
        title = f"{compound_name} - {row.get('Library', '')} (score={row.get('Score'):.3f})" if row.get("Score") is not None else compound_name

        known_keys = {
            "Library",
            "Compound_Name",
            "Compound_ID",
            "Score",
            "Matched_Fragments",
            "Precursor_MZ_Diff",
            "Formula",
            "TheoreticalMass",
            "InChI",
            "InChIKey",
            "SMILES",
            "Library_Instrument",
            "Library_Fragmentation_Mode",
            "Library_Collision_Energy",
            "Library_RT",
            "Library_Precursor_MZ",
            "Sample_File",
            "Scan_RT",
            "Scan_Precursor_MZ",
            "Scan_Fragmentation_Mode",
            "Scan_Collision_Energy",
            "Feature_Num",
            "Feature_OGroup",
            "Feature_MZ",
            "Feature_RT",
            "Feature_Xn",
            "Feature_Charge",
            "Feature_Ionisation_Mode",
            "Feature_Average_peakarea",
        }

        def build_body(layout):
            _add_metadata_grid(
                layout,
                [
                    ("Library", row.get("Library")),
                    ("Compound name", row.get("Compound_Name")),
                    ("Compound ID", row.get("Compound_ID")),
                    ("Score", _fmt_score(row.get("Score"))),
                    ("Matched fragments", row.get("Matched_Fragments")),
                    ("Precursor m/z diff", _fmt_mz(row.get("Precursor_MZ_Diff"))),
                    ("Formula", row.get("Formula")),
                    ("Theoretical mass", row.get("TheoreticalMass")),
                    ("InChI", row.get("InChI")),
                    ("InChIKey", row.get("InChIKey")),
                    ("Library instrument", row.get("Library_Instrument")),
                    ("Library fragmentation mode", row.get("Library_Fragmentation_Mode")),
                    ("Library collision energy", row.get("Library_Collision_Energy")),
                    ("Library RT", _fmt_rt(row.get("Library_RT"))),
                    ("Library precursor m/z", _fmt_mz(row.get("Library_Precursor_MZ"))),
                    ("Sample", row.get("Sample_File")),
                    ("Scan RT", _fmt_rt(row.get("Scan_RT"))),
                    ("Scan precursor m/z", _fmt_mz(row.get("Scan_Precursor_MZ"))),
                    ("Scan fragmentation mode", row.get("Scan_Fragmentation_Mode")),
                    ("Scan collision energy", row.get("Scan_Collision_Energy")),
                    ("Feature Num", row.get("Feature_Num")),
                    ("OGroup", row.get("Feature_OGroup")),
                    ("m/z", _fmt_mz(row.get("Feature_MZ"))),
                    ("RT", _fmt_rt(row.get("Feature_RT"))),
                    ("Xn", row.get("Feature_Xn")),
                    ("Z", row.get("Feature_Charge")),
                    ("Polarity", row.get("Feature_Ionisation_Mode")),
                    ("Average abundance", _fmt_abundance(row.get("Feature_Average_peakarea"))),
                ]
                + _extra_fields(row, known_keys),
            )

            def build_match(match_layout):
                if self._library_cache is not None and self._get_experimental_scan is not None:
                    lib_spectrum = self._library_cache.get_spectrum(row.get("Library"), row.get("Compound_ID"))
                    exp_scan = self._get_experimental_scan(row.get("Feature_Num"))
                    _add_mirror_plot(match_layout, exp_scan, lib_spectrum)

            _add_structure_and_match_row(layout, get_structure_smiles(row), build_match)

        return _AnnotationCard(title, build_body)

    def _build_compound_card(self, row):
        title = f"{row.get('DB_CompoundName') or '(unnamed compound)'} - {row.get('DB_Name', '')}"

        known_keys = {
            "DB_Name",
            "DB_Num",
            "DB_CompoundName",
            "DB_SumFormula",
            "DB_Mass",
            "DB_RT_min",
            "DB_MZ",
            "DB_Polarity",
            "HitType",
            "MatchErrorPPM",
            "MatchErrorMass",
            "Feature_Num",
            "Feature_OGroup",
            "Feature_MZ",
            "Feature_RT",
            "Feature_Xn",
            "Feature_Charge",
            "Feature_Ionisation_Mode",
            "Feature_Average_peakarea",
        }
        known_keys |= {k for k in row if k.startswith("DB_Info_")}

        def build_body(layout):
            _add_metadata_grid(
                layout,
                [
                    ("Database", row.get("DB_Name")),
                    ("Database entry ID", row.get("DB_Num")),
                    ("Compound name", row.get("DB_CompoundName")),
                    ("Sum formula", row.get("DB_SumFormula")),
                    ("Mass", row.get("DB_Mass")),
                    ("RT (min)", _fmt_rt(row.get("DB_RT_min"))),
                    ("m/z", _fmt_mz(row.get("DB_MZ"))),
                    ("Polarity", row.get("DB_Polarity")),
                    ("Hit type", row.get("HitType")),
                    ("Match error (ppm)", row.get("MatchErrorPPM")),
                    ("Match error (Da)", row.get("MatchErrorMass")),
                    ("Feature Num", row.get("Feature_Num")),
                    ("OGroup", row.get("Feature_OGroup")),
                    ("Feature m/z", _fmt_mz(row.get("Feature_MZ"))),
                    ("Feature RT", _fmt_rt(row.get("Feature_RT"))),
                    ("Feature Xn", row.get("Feature_Xn")),
                    ("Z", row.get("Feature_Charge")),
                    ("Polarity (feature)", row.get("Feature_Ionisation_Mode")),
                    ("Average abundance", _fmt_abundance(row.get("Feature_Average_peakarea"))),
                ]
                + [(k[len("DB_Info_") :], v) for k, v in row.items() if k.startswith("DB_Info_") and k != "DB_Info_SMILES"]
                + _extra_fields(row, known_keys),
            )
            _add_structure_image(layout, get_structure_smiles(row))

        return _AnnotationCard(title, build_body)

    def _build_sumformula_card(self, row):
        title = f"{row.get('SumFormula') or '(unknown formula)'} ({row.get('Element_Class', '')})"

        def build_body(layout):
            _add_metadata_grid(
                layout,
                [
                    ("Sum formula", row.get("SumFormula")),
                    ("Element class", row.get("Element_Class")),
                    ("Mass error (ppm)", row.get("MassErrorPPM")),
                    ("Mass error (Da)", row.get("MassErrorMass")),
                    ("Feature Num", row.get("Feature_Num")),
                    ("OGroup", row.get("Feature_OGroup")),
                    ("m/z", _fmt_mz(row.get("Feature_MZ"))),
                    ("RT", _fmt_rt(row.get("Feature_RT"))),
                    ("Xn", row.get("Feature_Xn")),
                    ("Z", row.get("Feature_Charge")),
                    ("Polarity", row.get("Feature_Ionisation_Mode")),
                    ("Average abundance", _fmt_abundance(row.get("Feature_Average_peakarea"))),
                ],
            )

        return _AnnotationCard(title, build_body)
