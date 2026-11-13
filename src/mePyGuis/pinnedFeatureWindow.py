"""Pinned feature window.

A stand-alone window holding a copy of the "Experiment results" plots (raw XICs, abundance profiles,
separated peaks, MS1 isotope patterns and isotopic pattern) of one feature pair. It has no feature
list, so it stays on its feature when the selection in the Experiment results changes.
"""

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

from .annotationPanel import FeatureAnnotationsPanel

POPOUT_COLOR = "#FF8C00"


def make_popout_icon(size=32, color="white"):
    """A 'pop out' icon: a window frame with an arrow leaving it towards the top right corner."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), size * 0.09)
    pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    s = float(size)
    frame = QtGui.QPainterPath()
    frame.moveTo(s * 0.42, s * 0.18)
    frame.lineTo(s * 0.18, s * 0.18)
    frame.lineTo(s * 0.18, s * 0.82)
    frame.lineTo(s * 0.82, s * 0.82)
    frame.lineTo(s * 0.82, s * 0.58)
    painter.drawPath(frame)
    painter.drawLine(QtCore.QPointF(s * 0.46, s * 0.54), QtCore.QPointF(s * 0.84, s * 0.16))
    head = QtGui.QPainterPath()
    head.moveTo(s * 0.58, s * 0.16)
    head.lineTo(s * 0.84, s * 0.16)
    head.lineTo(s * 0.84, s * 0.42)
    painter.drawPath(head)
    painter.end()
    return QtGui.QIcon(pixmap)


def _new_plot(parent, adjust):
    """Plot holder with the same attributes (fig, canvas, axes, twinxs, mpl_toolbar) as the plots of the main window."""
    plot = QtCore.QObject(parent)
    plot.dpi = 50
    plot.fig = Figure((5.0, 4.0), dpi=plot.dpi, facecolor="white")
    plot.fig.subplots_adjust(**adjust)
    plot.canvas = FigureCanvas(plot.fig)
    plot.axes = plot.fig.add_subplot(111)
    plot.axes.spines["top"].set_visible(False)
    plot.axes.spines["right"].set_visible(False)
    plot.twinxs = [plot.axes]
    plot.mpl_toolbar = NavigationToolbar(plot.canvas, parent)
    return plot


class PinnedUiProxy:
    """Stands in for the main window's `ui` while a pinned window is drawn: the widgets of the pinned window
    take the place of the Experiment results plots, all other widgets (e.g. the plot options) are the main window's."""

    def __init__(self, base, overrides):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        return overrides[name] if name in overrides else getattr(object.__getattribute__(self, "_base"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_base"), name, value)


class PinnedFeatureWindow(QtWidgets.QWidget):
    refreshRequested = Signal(object)

    def __init__(self, plot_items):
        # no parent: a window owned by the main window would always stay on top of it
        super().__init__(None, QtCore.Qt.WindowType.Window)
        self.plot_items = list(plot_items)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Pinned feature - {self._title()}")
        self.setWindowIcon(make_popout_icon(64, POPOUT_COLOR))
        self.resize(1500, 950)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(f"<b>{self._title()}</b><br>{self._details()}")
        title.setTextFormat(QtCore.Qt.TextFormat.RichText)
        header.addWidget(title, 1)
        refresh_btn = QtWidgets.QPushButton("Redraw")
        refresh_btn.setToolTip("Redraw the plots of this feature with the current plot options of the Experiment results.")
        refresh_btn.clicked.connect(lambda: self.refreshRequested.emit(self))
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        self.eic_plot = _new_plot(self, dict(left=0.05, bottom=0.05, right=0.99, top=0.95))
        self.abundance_plot = _new_plot(self, dict(left=0.08, bottom=0.15, right=0.99, top=0.95))
        self.separated_plot = _new_plot(self, dict(left=0.05, bottom=0.05, right=0.99, top=0.95))
        self.ms_scan_plot = _new_plot(self, dict(left=0.05, bottom=0.05, right=0.99, top=0.95))
        self.isotopic_plot = _new_plot(self, dict(left=0.08, bottom=0.15, right=0.99, top=0.9))
        self.isotopic_table = QtWidgets.QTableWidget()

        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        for label, plot in (("Raw XICs", self.eic_plot), ("Abundance profiles", self.abundance_plot), ("Separated peaks", self.separated_plot)):
            top.addWidget(self._plot_box(label, plot))

        bottom = QtWidgets.QTabWidget()
        self.bottom_tabs = bottom
        bottom.addTab(self._build_msms_page(), "MS/MS")
        self.annotations_panel = FeatureAnnotationsPanel()
        bottom.addTab(self.annotations_panel, "Feature annotations")
        bottom.addTab(self._plot_box(None, self.ms_scan_plot), "MS1 isotope patterns")
        isotopic = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        isotopic.addWidget(self._plot_box(None, self.isotopic_plot))
        isotopic.addWidget(self.isotopic_table)
        isotopic.setStretchFactor(0, 7)
        isotopic.setStretchFactor(1, 3)
        bottom.addTab(isotopic, "Isotopic pattern")

        main = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(bottom)
        main.setStretchFactor(0, 1)
        main.setStretchFactor(1, 1)
        layout.addWidget(main, 1)

    def _build_msms_page(self):
        """Counterpart of the Experiment results "MS/MS" panel: table of the feature's MS/MS spectra, similarity
        summaries and the spectra plot."""
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.msms_typestats_label = QtWidgets.QLabel()
        self.msms_similarity_label = QtWidgets.QLabel()
        for label in (self.msms_typestats_label, self.msms_similarity_label):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            left_layout.addWidget(label)
        self.msms_sqrt_check = QtWidgets.QCheckBox("Square-root intensity axis")
        left_layout.addWidget(self.msms_sqrt_check)
        table = QtWidgets.QTableWidget(0, 7)
        table.setMinimumWidth(200)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setHorizontalHeaderLabels(["N/L", "Prec. Abundance", "Prec. m/z", "RT (min)", "Group", "File", "Filter string"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setSortingEnabled(True)
        table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.msms_table = table
        left_layout.addWidget(table, 1)

        self.msms_plot = _new_plot(self, dict(left=0.08, bottom=0.08, right=0.98, top=0.95))
        self.msms_plot.fig.clear()
        self.msms_plot.axes = []
        splitter.addWidget(left)
        splitter.addWidget(self._plot_box(None, self.msms_plot))
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 6)
        return splitter

    @staticmethod
    def _plot_box(label, plot):
        box = QtWidgets.QGroupBox(label or "")
        box_layout = QtWidgets.QVBoxLayout(box)
        box_layout.setContentsMargins(2, 2, 2, 2)
        box_layout.addWidget(plot.mpl_toolbar)
        box_layout.addWidget(plot.canvas, 1)
        return box

    def _title(self):
        ids = [str(pi.id) for pi in self.plot_items if getattr(pi, "id", None) is not None]
        if not ids:
            return "feature"
        shown = ", ".join(ids[:4]) + (f" (+{len(ids) - 4})" if len(ids) > 4 else "")
        return f"FP {shown}"

    def _details(self):
        parts = []
        for pi in self.plot_items[:3]:
            mz, lmz, rt = getattr(pi, "mz", None), getattr(pi, "lmz", None), getattr(pi, "rt", None)
            text = f"FP {getattr(pi, 'id', '?')}"
            if mz is not None:
                text += f": m/z {mz:.5f}" + (f" / {lmz:.5f}" if lmz else "")
            if rt is not None:
                text += f", RT {rt / 60.0:.2f} min"
            parts.append(text)
        if len(self.plot_items) > 3:
            parts.append(f"+{len(self.plot_items) - 3} more")
        return "<br>".join(parts)

    def ui_overrides(self):
        """The plot/table widgets of this window by the names of the main window's Experiment results widgets."""
        return {
            "resultsExperiment_plot": self.eic_plot,
            "resultsExperimentAbundance_plot": self.abundance_plot,
            "resultsExperimentSeparatedPeaks_plot": self.separated_plot,
            "resultsExperimentMSScanPeaks_plot": self.ms_scan_plot,
            "resultsExperimentIsotopicPattern_plot": self.isotopic_plot,
            "tableWidget_isotopicEnrichment": self.isotopic_table,
            "plMSMS_exp": self.msms_plot,
            "msms_SpectraList_exp": self.msms_table,
            "checkBox_msms_sqrt_exp": self.msms_sqrt_check,
            "label_msms_exp_typestats": self.msms_typestats_label,
            "label_msms_exp_selected_similarity": self.msms_similarity_label,
            "featureAnnotationsPanel": self.annotations_panel,
        }
