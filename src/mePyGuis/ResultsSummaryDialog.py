"""
Results Summary Dialog for MetExtract II.

Replaces the scrollable text overview with two structured tables:
- Table 1: Per-file rows (one row per file × ionMode), coloured by group, sortable.
- Table 2: Bracketing/convoluted results summary.
"""

from PySide6 import QtCore, QtGui, QtWidgets


class ResultsSummaryDialog(QtWidgets.QDialog):
    """Dialog showing a tabular overview of MetExtract II processing results."""

    def __init__(self, parent=None, file_rows=None, summary_data=None):
        """
        Parameters
        ----------
        file_rows : list[dict]
            Each dict has keys:
                group_name, group_color, file, ion_mode,
                n_mzs, n_mz_bins, n_features, n_metabolites,
                mz_delta_mean, mz_delta_std,
                avg_ratio_signals, avg_ratio_signals_std,
                avg_ratio_area, avg_ratio_abundance,
                avg_enrichment, avg_enrichment_std,
                error (str or None)
        summary_data : dict or None
            Keys:
                n_features, n_metabolites,
                features_1, features_2, features_3, features_4, features_5,
                features_5_to_10, features_10_to_20, features_gt20,
                pos_only, neg_only, both_modes,
                omitted_features (int or None)
        """
        super().__init__(parent)
        self.setWindowTitle("Processing Results Overview")
        self.resize(1300, 700)

        self._file_rows = file_rows or []
        self._summary_data = summary_data or {}

        self._build_ui()
        self._populate_file_table()
        self._populate_summary_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # ── Note label ────────────────────────────────────────────────
        note = QtWidgets.QLabel("<b>Note:</b> All values are based on the last data processing run and are influenced by the chosen parameters. Use only as a quick overview.")
        note.setWordWrap(True)
        root.addWidget(note)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root.addWidget(splitter, stretch=1)

        # ── Table 1: per-file results ──────────────────────────────────
        top_group = QtWidgets.QGroupBox("Results of individual files")
        top_layout = QtWidgets.QVBoxLayout(top_group)
        top_layout.setContentsMargins(4, 4, 4, 4)

        self.fileTable = QtWidgets.QTableWidget(0, 12)
        self.fileTable.setHorizontalHeaderLabels(
            [
                "Group",
                "File",
                "Ion mode",
                "Signal pairs",
                "MZ bins",
                "Feature pairs",
                "Metabolites",
                "m/z Δ (ppm mean)",
                "m/z Δ (ppm std)",
                "M:M' ratio (signals)",
                "M:M' ratio (area / abund.)",
                "L-Enrichment (%)",
            ]
        )
        self.fileTable.setSortingEnabled(True)
        self.fileTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fileTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.fileTable.horizontalHeader().setStretchLastSection(True)
        self.fileTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.fileTable.verticalHeader().setVisible(False)
        top_layout.addWidget(self.fileTable)
        splitter.addWidget(top_group)

        # ── Table 2: convoluted / bracketing summary ───────────────────
        bottom_group = QtWidgets.QGroupBox("Bracketing / convoluted results summary")
        bottom_layout = QtWidgets.QVBoxLayout(bottom_group)
        bottom_layout.setContentsMargins(4, 4, 4, 4)

        self.summaryTable = QtWidgets.QTableWidget(0, 2)
        self.summaryTable.setHorizontalHeaderLabels(["Metric", "Value"])
        self.summaryTable.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.summaryTable.horizontalHeader().setStretchLastSection(True)
        self.summaryTable.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.summaryTable.verticalHeader().setVisible(False)
        bottom_layout.addWidget(self.summaryTable)
        splitter.addWidget(bottom_group)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        # ── Close button ───────────────────────────────────────────────
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _make_item(self, text, align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter):
        item = QtWidgets.QTableWidgetItem(str(text))
        item.setTextAlignment(align)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _make_num_item(self, value):
        """Create a table item that sorts numerically."""
        if value is None:
            item = QtWidgets.QTableWidgetItem("")
        else:
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.DisplayRole, value)
        item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _populate_file_table(self):
        self.fileTable.setSortingEnabled(False)
        self.fileTable.setRowCount(len(self._file_rows))

        # Track group → background colour so adjacent rows of the same group share the same tint
        _group_colors = {}

        for row_idx, row in enumerate(self._file_rows):
            group_name = row.get("group_name", "")
            group_color = row.get("group_color", "")

            # Resolve background colour for this group row
            bg_color = None
            if group_color:
                try:
                    qc = QtGui.QColor(group_color)
                    if qc.isValid():
                        # Use a lighter tint so text remains legible
                        qc.setAlpha(60)
                        bg_color = qc
                except Exception:
                    pass

            if bg_color is None and group_name:
                # Assign a deterministic pastel from a fixed palette
                if group_name not in _group_colors:
                    _pastel = [
                        QtGui.QColor(200, 220, 255, 80),
                        QtGui.QColor(200, 255, 220, 80),
                        QtGui.QColor(255, 220, 200, 80),
                        QtGui.QColor(240, 200, 255, 80),
                        QtGui.QColor(255, 255, 200, 80),
                        QtGui.QColor(200, 255, 255, 80),
                    ]
                    _group_colors[group_name] = _pastel[len(_group_colors) % len(_pastel)]
                bg_color = _group_colors[group_name]

            error = row.get("error")

            cells = [
                self._make_item(group_name),
                self._make_item(row.get("file", "")),
                self._make_item(row.get("ion_mode", "")),
            ]

            if error:
                cells.append(self._make_item(f"Error: {error}"))
                # Fill remaining columns with empty items
                while len(cells) < 12:
                    cells.append(self._make_item(""))
            else:
                n_mzs = row.get("n_mzs")
                n_mz_bins = row.get("n_mz_bins")
                n_features = row.get("n_features")
                n_metabolites = row.get("n_metabolites")
                mz_mean = row.get("mz_delta_mean")
                mz_std = row.get("mz_delta_std")
                ratio_sig = row.get("avg_ratio_signals")
                ratio_sig_std = row.get("avg_ratio_signals_std")
                ratio_area = row.get("avg_ratio_area")
                ratio_abund = row.get("avg_ratio_abundance")
                enr = row.get("avg_enrichment")
                enr_std = row.get("avg_enrichment_std")

                cells.append(self._make_num_item(n_mzs))
                cells.append(self._make_num_item(n_mz_bins))
                cells.append(self._make_num_item(n_features))
                cells.append(self._make_num_item(n_metabolites))
                cells.append(self._make_item("%.2f" % mz_mean if mz_mean is not None else ""))
                cells.append(self._make_item("%.2f" % mz_std if mz_std is not None else ""))
                cells.append(self._make_item("%.2f ± %.2f" % (ratio_sig, ratio_sig_std) if ratio_sig is not None else ""))
                cells.append(
                    self._make_item(
                        "%s / %s"
                        % (
                            "%.2f" % ratio_area if ratio_area is not None else "-",
                            "%.2f" % ratio_abund if ratio_abund is not None else "-",
                        )
                    )
                )
                cells.append(self._make_item("%.2f ± %.2f" % (100 * enr, 100 * enr_std) if enr is not None and enr_std is not None else ("%.2f" % (100 * enr) if enr is not None else "")))

            for col_idx, cell in enumerate(cells):
                if bg_color is not None:
                    cell.setBackground(QtGui.QBrush(bg_color))
                self.fileTable.setItem(row_idx, col_idx, cell)

        self.fileTable.setSortingEnabled(True)
        self.fileTable.resizeColumnsToContents()

    def _add_summary_row(self, label, value):
        row = self.summaryTable.rowCount()
        self.summaryTable.insertRow(row)
        self.summaryTable.setItem(row, 0, self._make_item(label))
        self.summaryTable.setItem(row, 1, self._make_num_item(value) if isinstance(value, (int, float)) else self._make_item(str(value) if value is not None else ""))

    def _add_summary_section(self, title):
        """Add a bold header row."""
        row = self.summaryTable.rowCount()
        self.summaryTable.insertRow(row)
        header_item = QtWidgets.QTableWidgetItem(title)
        header_item.setFlags(header_item.flags() & ~QtCore.Qt.ItemIsEditable)
        font = header_item.font()
        font.setBold(True)
        header_item.setFont(font)
        header_item.setBackground(QtGui.QBrush(QtGui.QColor(230, 230, 230)))
        self.summaryTable.setItem(row, 0, header_item)
        empty = QtWidgets.QTableWidgetItem("")
        empty.setBackground(QtGui.QBrush(QtGui.QColor(230, 230, 230)))
        empty.setFlags(empty.flags() & ~QtCore.Qt.ItemIsEditable)
        self.summaryTable.setItem(row, 1, empty)

    def _populate_summary_table(self):
        d = self._summary_data
        if not d:
            self._add_summary_row("No bracketing results available", "")
            return

        self._add_summary_section("Overall")
        self._add_summary_row("Feature pairs", d.get("n_features", ""))
        self._add_summary_row("Metabolites (OGroups)", d.get("n_metabolites", ""))
        if d.get("omitted_features") is not None:
            self._add_summary_row("Omitted feature pairs", d.get("omitted_features", ""))

        self._add_summary_section("Metabolites by number of feature pairs")
        self._add_summary_row("1 feature pair", d.get("features_1", ""))
        self._add_summary_row("2 feature pairs", d.get("features_2", ""))
        self._add_summary_row("3 feature pairs", d.get("features_3", ""))
        self._add_summary_row("4 feature pairs", d.get("features_4", ""))
        self._add_summary_row("5 feature pairs", d.get("features_5", ""))
        self._add_summary_row(">5 and ≤10 feature pairs", d.get("features_5_to_10", ""))
        self._add_summary_row(">10 and ≤20 feature pairs", d.get("features_10_to_20", ""))
        self._add_summary_row(">20 feature pairs", d.get("features_gt20", ""))

        self._add_summary_section("Metabolites by ionization mode")
        self._add_summary_row("Positive mode only", d.get("pos_only", ""))
        self._add_summary_row("Negative mode only", d.get("neg_only", ""))
        self._add_summary_row("Both modes", d.get("both_modes", ""))

        self.summaryTable.resizeColumnsToContents()
