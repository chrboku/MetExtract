"""
Isotope pattern explorer dialog.

Accessible from Tools > "Isotope pattern explorer". Lets the user enter a sum formula and the
measured (relative) ratios for the 6 isotopologs used for the isotope-pattern score, then shows
the per-isotopolog absolute-error contributions, their sum, and bar plots comparing the
theoretical and observed isotope patterns.
"""

from __future__ import absolute_import, division, print_function

from PySide6 import QtCore, QtWidgets
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ..isotopeScoring import ISOTOPOLOGS, parse_sum_formula, theoretical_isotopolog_ratio

# plot text is rendered at half the default matplotlib font sizes
_TITLE_FONTSIZE = rcParams["axes.titlesize"] if isinstance(rcParams["axes.titlesize"], (int, float)) else 10
_LABEL_FONTSIZE = rcParams["axes.labelsize"] if isinstance(rcParams["axes.labelsize"], (int, float)) else 10
_TICK_FONTSIZE = rcParams["xtick.labelsize"] if isinstance(rcParams["xtick.labelsize"], (int, float)) else 10


class IsotopePatternExplorerDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Isotope pattern explorer")
        self.resize(1000, 650)

        main_layout = QtWidgets.QVBoxLayout(self)

        # Sum formula input
        formula_layout = QtWidgets.QHBoxLayout()
        formula_layout.addWidget(QtWidgets.QLabel("Sum formula:"))
        self.formula_edit = QtWidgets.QLineEdit()
        self.formula_edit.setPlaceholderText("e.g. C6H12O6")
        self.formula_edit.textChanged.connect(self._recalculate)
        formula_layout.addWidget(self.formula_edit)
        self.formula_error_label = QtWidgets.QLabel()
        self.formula_error_label.setStyleSheet("color: red;")
        formula_layout.addWidget(self.formula_error_label)
        main_layout.addLayout(formula_layout)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        # Left: measured-ratio inputs + results table
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)

        left_layout.addWidget(QtWidgets.QLabel("Measured (relative) isotopolog ratios:"))
        self.measured_spins = {}
        input_grid = QtWidgets.QGridLayout()
        for row, (name, _element_symbol, _isotope_key, _count) in enumerate(ISOTOPOLOGS):
            input_grid.addWidget(QtWidgets.QLabel(name), row, 0)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(6)
            spin.setRange(0.0, 10.0)
            spin.setSingleStep(0.001)
            spin.valueChanged.connect(self._recalculate)
            input_grid.addWidget(spin, row, 1)
            self.measured_spins[name] = spin
        left_layout.addLayout(input_grid)

        self.total_score_label = QtWidgets.QLabel("Total absolute error: n/a")
        font = self.total_score_label.font()
        font.setBold(True)
        self.total_score_label.setFont(font)
        left_layout.addWidget(self.total_score_label)

        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Isotopolog", "Measured", "Theoretical", "Abs. error"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left_layout.addWidget(self.results_table, stretch=1)

        splitter.addWidget(left_widget)

        # Right: theoretical / observed pattern plots
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        self.figure = Figure(figsize=(5, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax_theoretical = self.figure.add_subplot(211)
        self.ax_measured = self.figure.add_subplot(212)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right_widget)

        splitter.setSizes([450, 550])

        self._recalculate()

    def _recalculate(self):
        formula = self.formula_edit.text().strip()
        elems = parse_sum_formula(formula) if formula else None

        if formula and elems is None:
            self.formula_error_label.setText("Could not parse sum formula")
        else:
            self.formula_error_label.setText("")

        names = []
        measured_values = []
        theoretical_values = []
        abs_errors = []
        for name, element_symbol, isotope_key, count in ISOTOPOLOGS:
            measured = self.measured_spins[name].value()
            theoretical = theoretical_isotopolog_ratio(elems, element_symbol, isotope_key, count) if elems is not None else 0.0
            names.append(name)
            measured_values.append(measured)
            theoretical_values.append(theoretical)
            abs_errors.append(abs(measured - theoretical))

        self.results_table.setRowCount(len(names))
        for row, name in enumerate(names):
            self.results_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.results_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{measured_values[row]:.6g}"))
            self.results_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{theoretical_values[row]:.6g}"))
            self.results_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{abs_errors[row]:.6g}"))

        total_score = sum(abs_errors) if elems is not None else None
        self.total_score_label.setText(f"Total absolute error: {total_score:.6g}" if total_score is not None else "Total absolute error: n/a")

        self.ax_theoretical.clear()
        self.ax_theoretical.bar(names, theoretical_values, color="dodgerblue")
        self.ax_theoretical.set_title("Theoretical isotope pattern", fontsize=_TITLE_FONTSIZE / 2)
        self.ax_theoretical.set_ylabel("Relative ratio", fontsize=_LABEL_FONTSIZE / 2)
        self.ax_theoretical.tick_params(axis="x", rotation=45, labelsize=_TICK_FONTSIZE / 2)
        self.ax_theoretical.tick_params(axis="y", labelsize=_TICK_FONTSIZE / 2)

        self.ax_measured.clear()
        self.ax_measured.bar(names, measured_values, color="darkorange")
        self.ax_measured.set_title("Observed (measured) isotope pattern", fontsize=_TITLE_FONTSIZE / 2)
        self.ax_measured.set_ylabel("Relative ratio", fontsize=_LABEL_FONTSIZE / 2)
        self.ax_measured.tick_params(axis="x", rotation=45, labelsize=_TICK_FONTSIZE / 2)
        self.ax_measured.tick_params(axis="y", labelsize=_TICK_FONTSIZE / 2)

        self.figure.tight_layout()
        self.canvas.draw_idle()
