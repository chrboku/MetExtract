"""
MSMS spectra overview dialog.

Accessible from Tools > "MSMS spectra overview". Shows all loaded MGF/JSON spectral
library files in a tree (file -> spectra), and displays the selected spectrum
(stem plot) plus a table of its parsed metadata.
"""

from __future__ import absolute_import, division, print_function

import os

from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..MSMS import mgfLibrary


class MSMSSpectraOverviewDialog(QtWidgets.QDialog):
    def __init__(self, library_files, parent=None):
        """
        Args:
            library_files: list of library-file info dicts {"path", "type" ("mgf"/"json"),
                "precursor_mz_key"} to display.
        """
        super().__init__(parent)
        self.setWindowTitle("MSMS spectra overview")
        self.resize(1000, 650)

        self._library_by_file = {}

        main_layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: tree of files -> spectra
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Library file / spectrum"])
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.tree)

        # Right: spectrum plot + metadata table
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)

        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        right_layout.addWidget(self.canvas, stretch=2)

        self.metadata_table = QtWidgets.QTableWidget()
        self.metadata_table.setColumnCount(2)
        self.metadata_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self.metadata_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right_layout.addWidget(self.metadata_table, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([300, 700])

        self._load_files(library_files or [])

    def _load_files(self, library_files):
        for entry in library_files:
            path = entry["path"]
            file_item = QtWidgets.QTreeWidgetItem([f"{os.path.basename(path)} ({entry.get('type', 'mgf').upper()})"])
            file_item.setToolTip(0, path)
            self.tree.addTopLevelItem(file_item)
            try:
                spectra = mgfLibrary.load_library_entry(entry)
            except Exception as e:
                error_item = QtWidgets.QTreeWidgetItem([f"<error loading file: {e}>"])
                file_item.addChild(error_item)
                continue

            self._library_by_file[path] = spectra
            for spec in spectra:
                label = spec.compound_name
                if spec.polarity:
                    label += f" ({spec.polarity})"
                spec_item = QtWidgets.QTreeWidgetItem([label])
                spec_item.setData(0, QtCore.Qt.UserRole, (path, spec.spectrum_index))
                file_item.addChild(spec_item)
        self.tree.expandAll()

    def _on_selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        data = item.data(0, QtCore.Qt.UserRole)
        if data is None:
            self._show_spectrum(None)
            return
        path, spectrum_index = data
        spectra = self._library_by_file.get(path, [])
        spec = next((s for s in spectra if s.spectrum_index == spectrum_index), None)
        self._show_spectrum(spec)

    def _show_spectrum(self, spec):
        self.ax.clear()
        self.metadata_table.setRowCount(0)
        if spec is None:
            self.canvas.draw_idle()
            return

        self.ax.vlines(spec.mz, 0, spec.intensities, color="dodgerblue", linewidth=1)
        self.ax.set_xlabel("m/z")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title(spec.compound_name)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        rows = [
            ("Compound name", spec.compound_name),
            ("Source file", os.path.basename(spec.source_file)),
            ("Precursor m/z", spec.precursor_mz),
            ("Polarity", spec.polarity),
            ("Polarity source", spec.polarity_source),
            ("Number of peaks", len(spec.mz)),
        ]
        for key, value in spec.metadata.items():
            rows.append((key, value))

        self.metadata_table.setRowCount(len(rows))
        for row_idx, (key, value) in enumerate(rows):
            self.metadata_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.metadata_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(str(value)))
