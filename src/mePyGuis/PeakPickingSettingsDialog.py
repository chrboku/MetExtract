"""
Peak Picking Settings Dialog for MetExtract II.

Provides a tabbed dialog where each tab corresponds to one of the five
peak picking algorithms. The active (selected) tab determines which
algorithm will be used for chromatographic peak picking.

Features:
- EIC specification area (file, polarity, mz, ppm) with visualization
- Live peak picking results overlay when parameters change
- Algorithm info button (?) on each tab with references
"""

import os
import numpy as np
from PySide6 import QtCore, QtWidgets

# Algorithm descriptions and references
_ALGO_INFO = {
    "wavelettransform": (
        "Wavelet Transform Peak Picker",
        "Uses the Continuous Wavelet Transform (CWT) with Ricker (Mexican-hat) wavelets to detect "
        "chromatographic peaks across multiple scales. Ridge lines in the CWT coefficient matrix "
        "are traced from coarse to fine scales; ridges that exceed the SNR threshold and span "
        "enough scales are reported as peaks.\n\n"
        "Parameters:\n"
        "• Min/Max scale — range of wavelet widths (in scans)\n"
        "• Number of scales — resolution between min and max\n"
        "• SNR threshold — signal-to-noise cutoff\n"
        "• Min ridge length — minimum scales a ridge must span\n"
        "• Gap threshold — allowed gaps in a ridge\n\n"
        "References:\n"
        "Du P., Kibbe W.A., Lin S.M. (2006). Improved peak detection in mass spectrum by "
        "incorporating continuous wavelet transform-based pattern matching. Bioinformatics, "
        "22(17), 2059-2065.\n\n"
        "Based on the R MassSpecWavelet package approach, reimplemented in Python with scipy.",
    ),
    "gradientdescent": (
        "Gradient Descent Peak Picker",
        "Smooths the EIC, finds local maxima, then walks downhill from each apex until the signal "
        "stalls or drops below a threshold. Stall detection uses a configurable number of "
        "consecutive scans with insufficient intensity increase.\n\n"
        "Parameters:\n"
        "• Smoothing method/window/iterations — noise reduction before detection\n"
        "• Min apex intensity — minimum peak height\n"
        "• Min flank intensity — boundary cutoff\n"
        "• Min increase ratio — stall detection sensitivity\n"
        "• Consecutive scans — stall window\n"
        "• Min/Max peak width — width filter in scans\n\n"
        "This is a custom algorithm developed for MetExtract II.",
    ),
    "gmm": (
        "GMM (Curve Fitting) Peak Picker",
        "Segments the EIC into regions of interest (chunks) based on a baseline fraction, "
        "then fits overlapping Gaussian or Exponentially Modified Gaussian (EMG) curves using "
        "non-linear least squares (scipy.optimize.curve_fit).\n\n"
        "Parameters:\n"
        "• Distribution type — gaussian or EMG for tailing peaks\n"
        "• Baseline fraction — threshold for chunking\n"
        "• Min chunk size — minimum region length\n"
        "• Max peaks per chunk — deconvolution complexity limit\n"
        "• Fit max iterations — optimizer iteration cap\n\n"
        "References:\n"
        "Yu T., Bhatt D.P., et al. (2009). Practical guidelines for estimation of peak areas "
        "in LC-MS bioanalysis. Bioanalysis, 1(1), 87-95.",
    ),
    "secondderivative": (
        "Second Derivative Peak Picker",
        "Applies a Savitzky-Golay second derivative filter to the EIC. Peaks are identified "
        "where the second derivative crosses zero (negative to positive = peak start, positive "
        "to negative = peak end) with a negative minimum between crossings (= apex).\n\n"
        "Parameters:\n"
        "• SG window/polyorder — Savitzky-Golay filter settings\n"
        "• Pre-smoothing — optional smoothing before differentiation\n"
        "• Min apex intensity — height threshold\n\n"
        "References:\n"
        "Savitzky A., Golay M.J.E. (1964). Smoothing and differentiation of data by simplified "
        "least squares procedures. Analytical Chemistry, 36(8), 1627-1639.",
    ),
    "matchedfilter": (
        "Matched Filter Peak Picker",
        "Generates a Gaussian template of the expected peak width and cross-correlates it with "
        "the EIC. Local maxima in the correlation trace above a threshold are reported as peaks.\n\n"
        "Parameters:\n"
        "• Expected peak width — template width in scans\n"
        "• Correlation threshold — minimum match score (0–1)\n"
        "• Min distance — merge distance between peaks\n\n"
        "References:\n"
        "Smith C.A., Want E.J., et al. (2006). XCMS: Processing mass spectrometry data for "
        "metabolite profiling using nonlinear peak alignment, matching, and identification. "
        "Analytical Chemistry, 78(3), 779-787.",
    ),
}


class PeakPickingSettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring chromatographic peak picking parameters.

    Each algorithm has its own tab. The currently selected tab is the
    algorithm that will be used for processing.
    """

    # Signal emitted when the user accepts the dialog
    settingsAccepted = QtCore.Signal(dict)

    def __init__(self, parent=None, current_settings: dict | None = None, sample_files=None, scan_events=None, initial_eic_rows=None):
        super().__init__(parent)
        self.setWindowTitle("Chromatographic Peak Picking Settings")
        self.setMinimumSize(900, 650)
        self.resize(1100, 750)

        if current_settings is None:
            current_settings = {}

        self._settings = dict(current_settings)
        self._eic_entries = []  # list of (file, polarity, mz, ppm) tuples from UI
        # sample_files: list of (group_name, file_path) from loaded experimental groups
        self._sample_files = sample_files or []
        # scan_events: {"+ ": [filter_line, ...], "-": [filter_line, ...]}
        self._scan_events = scan_events or {}
        # Saved EIC rows to restore on re-open: list of (file_idx, filter_idx, mz, ppm)
        self._initial_eic_rows = initial_eic_rows or []

        self._build_ui()
        self._load_settings(current_settings)
        if self._initial_eic_rows:
            self._restore_eic_rows(self._initial_eic_rows)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root_layout = QtWidgets.QVBoxLayout(self)

        # ── Two-column horizontal splitter ──────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # ── LEFT PANEL: algorithm tabs + post-processing filter ─────────
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)

        self.tabWidget = QtWidgets.QTabWidget()
        left_layout.addWidget(self.tabWidget, stretch=1)

        self._build_wavelet_tab()
        self._build_gradient_tab()
        self._build_gmm_tab()
        self._build_second_derivative_tab()
        self._build_matched_filter_tab()

        left_layout.addWidget(self._build_peak_filter_group(), stretch=0)

        splitter.addWidget(left_widget)

        # ── RIGHT PANEL: vertical splitter with EIC definitions + plot ──
        right_widget = QtWidgets.QWidget()
        right_outer = QtWidgets.QVBoxLayout(right_widget)
        right_outer.setContentsMargins(4, 0, 0, 0)
        right_outer.setSpacing(0)

        right_vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        right_outer.addWidget(right_vsplit, stretch=1)

        # ── Top half of splitter: EIC definitions ──────────────────────
        eic_group = QtWidgets.QGroupBox("EIC Definitions (for preview)")
        eic_layout = QtWidgets.QVBoxLayout(eic_group)
        eic_layout.setSpacing(4)

        self.eicTable = QtWidgets.QTableWidget(0, 4)
        self.eicTable.setHorizontalHeaderLabels(["File", "Filter Line", "m/z", "ppm"])
        self.eicTable.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.eicTable.setToolTip("Define EICs to preview peak picking. Each row specifies a loaded file, the MS1 filter line (scan event), m/z value, and ppm tolerance.")
        hdr = self.eicTable.horizontalHeader()
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        hdr.setStretchLastSection(False)
        self.eicTable.resizeEvent = self._resize_eic_columns
        eic_layout.addWidget(self.eicTable)

        eic_btn_layout = QtWidgets.QHBoxLayout()
        self.btnAddEic = QtWidgets.QPushButton("Add EIC")
        self.btnAddEic.setToolTip("Add a new EIC definition row")
        self.btnRemoveEic = QtWidgets.QPushButton("Remove Selected")
        self.btnRemoveEic.setToolTip("Remove selected EIC definition(s)")
        self.btnDuplicateEic = QtWidgets.QPushButton("Duplicate Selected")
        self.btnDuplicateEic.setToolTip("Duplicate the selected EIC definition row")
        self.btnDuplicateEic1Da = QtWidgets.QPushButton("Duplicate +1 Da")
        self.btnDuplicateEic1Da.setToolTip("Duplicate the selected EIC and shift m/z by +1.00335484 Da (one \u00b9\u00b3C isotope step)")
        self.btnDuplicateEicHalfDa = QtWidgets.QPushButton("Duplicate +\u00bd Da")
        self.btnDuplicateEicHalfDa.setToolTip("Duplicate the selected EIC and shift m/z by +0.50167742 Da (half \u00b9\u00b3C isotope step for doubly-charged ions)")
        self.btnPickEic = QtWidgets.QPushButton("Pick Peaks")
        self.btnPickEic.setToolTip("Run peak picking on defined EICs and show results below")
        eic_btn_layout.addWidget(self.btnAddEic)
        eic_btn_layout.addWidget(self.btnRemoveEic)
        eic_btn_layout.addWidget(self.btnDuplicateEic)
        eic_btn_layout.addWidget(self.btnDuplicateEic1Da)
        eic_btn_layout.addWidget(self.btnDuplicateEicHalfDa)
        eic_btn_layout.addStretch()
        eic_btn_layout.addWidget(self.btnPickEic)
        eic_layout.addLayout(eic_btn_layout)

        right_vsplit.addWidget(eic_group)

        # ── Bottom half of splitter: progress bar + plot ───────────────
        plot_outer = QtWidgets.QWidget()
        plot_outer_layout = QtWidgets.QVBoxLayout(plot_outer)
        plot_outer_layout.setContentsMargins(0, 0, 0, 0)
        plot_outer_layout.setSpacing(2)

        # Progress bar (hidden until peak picking runs)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.hide()
        plot_outer_layout.addWidget(self._progress_bar, stretch=0)

        self._setup_plot_area(plot_outer_layout)

        right_vsplit.addWidget(plot_outer)

        # EIC list ~25 %, plot ~75 % of right panel height
        right_vsplit.setStretchFactor(0, 1)
        right_vsplit.setStretchFactor(1, 3)

        splitter.addWidget(right_widget)

        # Give left ~40% and right ~60% of initial width
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # ── Bottom buttons ───────────────────────────────────────────────
        btn_layout = QtWidgets.QHBoxLayout()
        self.btnDefaults = QtWidgets.QPushButton("Restore Defaults")
        self.btnOk = QtWidgets.QPushButton("OK")
        self.btnCancel = QtWidgets.QPushButton("Cancel")
        btn_layout.addWidget(self.btnDefaults)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btnOk)
        btn_layout.addWidget(self.btnCancel)
        root_layout.addLayout(btn_layout)

        # Connections
        self.btnOk.clicked.connect(self._accept)
        self.btnCancel.clicked.connect(self.reject)
        self.btnDefaults.clicked.connect(self._restore_defaults)
        self.btnAddEic.clicked.connect(self._add_eic_row)
        self.btnRemoveEic.clicked.connect(self._remove_eic_rows)
        self.btnDuplicateEic.clicked.connect(self._duplicate_eic_row)
        self.btnDuplicateEic1Da.clicked.connect(lambda: self._duplicate_eic_row_with_mz_shift(1.00335484))
        self.btnDuplicateEicHalfDa.clicked.connect(lambda: self._duplicate_eic_row_with_mz_shift(0.50167742))
        self.btnPickEic.clicked.connect(self._run_preview)

    def _resize_eic_columns(self, event):
        """Keep EIC table column widths at 45/25/15/15 ratio (File column 25% narrower)."""
        total = self.eicTable.viewport().width()
        ratios = [0.45, 0.25, 0.15, 0.15]
        hdr = self.eicTable.horizontalHeader()
        for col, ratio in enumerate(ratios):
            hdr.resizeSection(col, max(1, int(total * ratio)))
        if event is not None:
            QtWidgets.QTableWidget.resizeEvent(self.eicTable, event)

    # -- Plot area helpers ---------------------------------------------

    def _setup_plot_area(self, layout):
        """Create the EIC visualization area: matplotlib canvas or QLabel fallback."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self._has_matplotlib = True
            self._eic_figure = Figure()
            self._eic_canvas = FigureCanvasQTAgg(self._eic_figure)
            self._eic_canvas.setMinimumHeight(240)
            self._eic_canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self._draw_placeholder()
            self._eic_hover_annotation = None  # active annotation widget
            self._eic_peak_patches = []  # list of (axes, patch, peak) for hover
            self._eic_canvas.mpl_connect("motion_notify_event", self._on_canvas_hover)

            # Toolbar: pan (left-drag) and zoom (right-drag / scroll box)
            self._eic_toolbar = NavigationToolbar2QT(self._eic_canvas, None)
            _keep = {"Pan", "Zoom", "Back", "Forward", "Home"}
            for action in self._eic_toolbar.actions():
                if action.text() not in _keep and not action.isSeparator():
                    action.setVisible(False)

            plot_container = QtWidgets.QWidget()
            plot_vbox = QtWidgets.QVBoxLayout(plot_container)
            plot_vbox.setContentsMargins(0, 0, 0, 0)
            plot_vbox.setSpacing(0)
            plot_vbox.addWidget(self._eic_toolbar)

            self._eic_scroll = QtWidgets.QScrollArea()
            self._eic_scroll.setWidgetResizable(True)
            self._eic_scroll.setWidget(self._eic_canvas)
            self._eic_scroll.setMinimumHeight(220)
            plot_vbox.addWidget(self._eic_scroll, stretch=1)

            layout.addWidget(plot_container, stretch=3)
        except ImportError:
            self._has_matplotlib = False
            self.eicPlotArea = QtWidgets.QLabel("EIC visualization will appear here when EICs are defined and 'Pick Peaks' is clicked.\n(Requires mzXML/mzML files accessible from this machine.)")
            self.eicPlotArea.setAlignment(QtCore.Qt.AlignCenter)
            self.eicPlotArea.setMinimumHeight(150)
            self.eicPlotArea.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ccc; color: #888; font-style: italic;")
            layout.addWidget(self.eicPlotArea, stretch=3)

    def _draw_placeholder(self):
        """Render a placeholder message in the matplotlib canvas."""
        self._eic_figure.clear()
        ax = self._eic_figure.add_subplot(111)
        ax.set_axis_off()
        self._eic_figure.text(
            0.5,
            0.5,
            "Click 'Pick Peaks' to visualize EICs and detected peaks.",
            ha="center",
            va="center",
            color="gray",
            style="italic",
            fontsize=11,
        )
        self._eic_figure.tight_layout()
        self._eic_canvas.draw()

    def _show_status_text(self, message: str, color: str = "gray"):
        """Show a plain message in the results area (matplotlib canvas or text label)."""
        if self._has_matplotlib:
            self._eic_figure.clear()
            ax = self._eic_figure.add_subplot(111)
            ax.set_axis_off()
            self._eic_figure.text(
                0.5,
                0.5,
                message,
                ha="center",
                va="center",
                color=color,
                style="italic",
                fontsize=10,
            )
            self._eic_figure.tight_layout()
            self._eic_canvas.draw()
        else:
            self.eicPlotArea.setText(message)

    # Alternating peak fill/marker colors (chronological: 1st peak → color 0, etc.)
    _PEAK_FILL_COLORS = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896"]
    _PEAK_MARKER_COLORS = ["#1f77b4", "#d65f00", "#2ca02c", "#d62728"]

    def _plot_eic_results(self, results):
        """Render EIC traces with detected peaks in the embedded matplotlib canvas."""
        import os

        import matplotlib.ticker as mticker

        n = len(results)
        if n == 0:
            self._draw_placeholder()
            return

        fig = self._eic_figure
        fig.clear()
        self._eic_peak_patches = []  # reset hover data

        subplot_h = 2.8
        fig.set_size_inches(7, max(subplot_h, subplot_h * n))

        _EIC_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Create all subplots at once with a fresh GridSpec to avoid matplotlib
        # accumulating axes from previous calls with a different row count.
        all_axes = fig.subplots(n, 1, squeeze=False)[:, 0]

        for i, (ax, res) in enumerate(zip(all_axes, results)):
            fpath, filter_line, mz, ppm = res[:4]
            rt_array, int_array, peaks, load_error, peak_error = res[4], res[5], res[6], res[7], res[8]
            fname = os.path.basename(fpath)
            fl_label = filter_line if filter_line else "(auto)"

            if load_error or rt_array is None or len(rt_array) == 0:
                ax.set_axis_off()
                msg = "Load error: %s" % load_error if load_error else "No data returned for this EIC."
                ax.text(0.5, 0.5, msg, ha="center", va="center", color="red", transform=ax.transAxes, fontsize=9)
                ax.set_title("%s  \u2014  m/z=%.5f \u00b1%.1f ppm  [%s]" % (fname, mz, ppm, fl_label), fontsize=9)
                continue

            color = _EIC_COLORS[i % len(_EIC_COLORS)]
            rt_min = rt_array / 60.0

            # Always plot the EIC trace
            ax.plot(rt_min, int_array, color=color, linewidth=1.2)
            ax.fill_between(rt_min, int_array, alpha=0.12, color=color)

            # Sort peaks chronologically by apex RT to assign colors in order
            sorted_peaks = sorted(enumerate(peaks), key=lambda x: x[1].apex_rt)

            for rank, (j, pk) in enumerate(sorted_peaks):
                pk_start = pk.start_rt / 60.0
                pk_end = pk.end_rt / 60.0
                pk_apex = pk.apex_rt / 60.0
                fill_c = self._PEAK_FILL_COLORS[rank % 4]
                mark_c = self._PEAK_MARKER_COLORS[rank % 4]

                mask = (rt_min >= pk_start) & (rt_min <= pk_end)
                if mask.any():
                    ax.fill_between(rt_min, int_array, where=mask, alpha=0.45, color=fill_c)

                apex_int = pk.apex_intensity if pk.apex_intensity > 0 else float(np.interp(pk_apex, rt_min, int_array))
                ax.plot(pk_apex, apex_int, marker="v", color=mark_c, markersize=7, zorder=5, linestyle="none")

                ax.annotate(
                    str(j + 1),
                    xy=(pk_apex, apex_int),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=mark_c,
                )

                # Build hover text and register the peak span for mouse events
                area_str = "%.3e" % pk.area if pk.area > 0 else "n/a"
                snr_str = "%.1f" % pk.snr if pk.snr > 0 else "n/a"
                pw_sec = pk.end_rt - pk.start_rt
                fwhm_sec = pk.fwhm if pk.fwhm > 0 else 0.0
                flank = max(pk.baseline, 1e-12)
                atf_factor = pk.apex_intensity / flank
                atf_increase = pk.apex_intensity - pk.baseline
                hover_text = (
                    f"Peak #{j + 1}\n"
                    f"Start:         {pk_start:.3f} min\n"
                    f"End:           {pk_end:.3f} min\n"
                    f"Apex:          {pk_apex:.3f} min\n"
                    f"Width:         {pw_sec:.2f} s\n"
                    f"FWHM:          {fwhm_sec:.2f} s\n"
                    f"Area:          {area_str}\n"
                    f"SNR:           {snr_str}\n"
                    f"Apex/Flank \u00d7:  {atf_factor:.2f}\n"
                    f"Apex\u2212Flank:    {atf_increase:.3e}"
                )
                self._eic_peak_patches.append((ax, pk_start, pk_end, hover_text))

            # Title: peak count or error note
            if peak_error:
                title_suffix = "\u2014 peak picking error"
                ax.text(0.01, 0.97, "\u26a0 %s" % peak_error, transform=ax.transAxes, fontsize=7, color="darkred", va="top", ha="left", style="italic")
            elif peaks:
                title_suffix = "\u2014 %d peak(s) detected" % len(peaks)
            else:
                title_suffix = "\u2014 no peaks detected"

            ax.set_title(
                "%s  \u2014  m/z=%.5f \u00b1%.1f ppm  [%s]  %s" % (fname, mz, ppm, fl_label, title_suffix),
                fontsize=9,
            )
            ax.set_xlabel("Retention time (min)", fontsize=8)
            ax.set_ylabel("Intensity", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%.2e" % v if v >= 10000 else "%.0f" % v))
            ax.margins(x=0.01)

        fig.tight_layout(pad=0.5)
        dpi = fig.dpi
        needed_h = max(240, int(fig.get_size_inches()[1] * dpi))
        # Resize the canvas widget to match figure height without disturbing layout.
        # Only call setMinimumHeight once; after that just resize the internal pixmap.
        if needed_h != self._eic_canvas.minimumHeight():
            self._eic_canvas.setMinimumHeight(needed_h)
        self._eic_canvas.draw_idle()

    def _on_canvas_hover(self, event):
        """Show a Qt tooltip with peak info when hovering over a peak region."""
        if event.inaxes is None or event.xdata is None:
            QtWidgets.QToolTip.hideText()
            return
        for ax, pk_start, pk_end, hover_text in self._eic_peak_patches:
            if ax is event.inaxes and pk_start <= event.xdata <= pk_end:
                global_pos = self._eic_canvas.mapToGlobal(QtCore.QPoint(int(event.x), int(self._eic_canvas.height() - event.y)))
                QtWidgets.QToolTip.showText(global_pos, hover_text, self._eic_canvas)
                return
        QtWidgets.QToolTip.hideText()

    # -- Post-processing filter group ----------------------------------

    def _build_peak_filter_group(self):
        """Build the post-processing peak filter group box."""
        grp = QtWidgets.QGroupBox("Post-processing peak filters (applied to all algorithms)")
        grp.setCheckable(True)
        grp.setChecked(True)
        grp.setToolTip("When enabled, peaks outside the given criteria are discarded after peak picking (applied to all algorithms).")
        self.grp_peak_filter = grp
        form = QtWidgets.QFormLayout(grp)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.pf_min_peak_width = QtWidgets.QDoubleSpinBox()
        self.pf_min_peak_width.setRange(0.0, 999999.0)
        self.pf_min_peak_width.setValue(3.0)
        self.pf_min_peak_width.setDecimals(2)
        self.pf_min_peak_width.setSuffix(" sec")
        self.pf_min_peak_width.setToolTip("Minimum chromatographic peak width (seconds). Peaks narrower than this are discarded.")
        lbl = QtWidgets.QLabel("Min peak width:")
        lbl.setToolTip(self.pf_min_peak_width.toolTip())
        form.addRow(lbl, self.pf_min_peak_width)

        self.pf_max_peak_width = QtWidgets.QDoubleSpinBox()
        self.pf_max_peak_width.setRange(0.0, 999999.0)
        self.pf_max_peak_width.setValue(50.0)
        self.pf_max_peak_width.setDecimals(2)
        self.pf_max_peak_width.setSuffix(" sec")
        self.pf_max_peak_width.setToolTip("Maximum chromatographic peak width (seconds). Peaks wider than this are discarded.")
        lbl = QtWidgets.QLabel("Max peak width:")
        lbl.setToolTip(self.pf_max_peak_width.toolTip())
        form.addRow(lbl, self.pf_max_peak_width)

        self.pf_min_fwhm = QtWidgets.QDoubleSpinBox()
        self.pf_min_fwhm.setRange(0.0, 999999.0)
        self.pf_min_fwhm.setValue(2.0)
        self.pf_min_fwhm.setDecimals(2)
        self.pf_min_fwhm.setSuffix(" sec")
        self.pf_min_fwhm.setToolTip("Minimum Full Width at Half Maximum (seconds). Peaks with FWHM below this are discarded.")
        lbl = QtWidgets.QLabel("Min FWHM:")
        lbl.setToolTip(self.pf_min_fwhm.toolTip())
        form.addRow(lbl, self.pf_min_fwhm)

        self.pf_max_fwhm = QtWidgets.QDoubleSpinBox()
        self.pf_max_fwhm.setRange(0.0, 999999.0)
        self.pf_max_fwhm.setValue(10.0)
        self.pf_max_fwhm.setDecimals(2)
        self.pf_max_fwhm.setSuffix(" sec")
        self.pf_max_fwhm.setToolTip("Maximum Full Width at Half Maximum (seconds). Peaks with FWHM above this are discarded.")
        lbl = QtWidgets.QLabel("Max FWHM:")
        lbl.setToolTip(self.pf_max_fwhm.toolTip())
        form.addRow(lbl, self.pf_max_fwhm)

        self.pf_min_apex_to_flank_factor = QtWidgets.QDoubleSpinBox()
        self.pf_min_apex_to_flank_factor.setRange(0.0, 10000.0)
        self.pf_min_apex_to_flank_factor.setValue(3.0)
        self.pf_min_apex_to_flank_factor.setDecimals(2)
        self.pf_min_apex_to_flank_factor.setSingleStep(0.5)
        self.pf_min_apex_to_flank_factor.setToolTip("Minimum ratio of apex intensity to the lowest flank intensity. Peaks where apex / min(left boundary, right boundary) is below this factor are discarded. Set to 0 to disable.")
        lbl = QtWidgets.QLabel("Min. apex to flank factor:")
        lbl.setToolTip(self.pf_min_apex_to_flank_factor.toolTip())
        form.addRow(lbl, self.pf_min_apex_to_flank_factor)

        self.pf_min_apex_to_flank_increase = QtWidgets.QDoubleSpinBox()
        self.pf_min_apex_to_flank_increase.setRange(0.0, 1e12)
        self.pf_min_apex_to_flank_increase.setValue(100000.0)
        self.pf_min_apex_to_flank_increase.setDecimals(1)
        self.pf_min_apex_to_flank_increase.setSingleStep(100.0)
        self.pf_min_apex_to_flank_increase.setToolTip("Minimum absolute intensity difference between the apex and the lowest flank. Peaks where apex \u2212 min(left boundary, right boundary) is below this value are discarded. Set to 0 to disable.")
        lbl = QtWidgets.QLabel("Min. apex to flank increase:")
        lbl.setToolTip(self.pf_min_apex_to_flank_increase.toolTip())
        form.addRow(lbl, self.pf_min_apex_to_flank_increase)

        self.pf_min_snr = QtWidgets.QDoubleSpinBox()
        self.pf_min_snr.setRange(0.0, 10000.0)
        self.pf_min_snr.setValue(3.0)
        self.pf_min_snr.setDecimals(2)
        self.pf_min_snr.setSingleStep(0.5)
        self.pf_min_snr.setToolTip("Minimum signal-to-noise ratio for accepting a peak. Peaks with SNR below this threshold are discarded. Set to 0 to disable.")
        lbl = QtWidgets.QLabel("Min. SNR:")
        lbl.setToolTip(self.pf_min_snr.toolTip())
        form.addRow(lbl, self.pf_min_snr)

        return grp

    # -- Tab builders ---------------------------------------------------

    def _add_info_button(self, form, algo_key):
        """Add a '?' button at the bottom of a tab that shows algorithm info."""
        btn = QtWidgets.QPushButton("?")
        btn.setFixedSize(28, 28)
        btn.setToolTip("Show algorithm description and references")
        btn.setStyleSheet("QPushButton { background-color: orange; color: black; font-weight: bold; border-radius: 4px; }QPushButton:hover { background-color: #ff8c00; }QPushButton:pressed { background-color: #e07b00; }")
        btn.clicked.connect(lambda checked, k=algo_key: self._show_algo_info(k))
        info_layout = QtWidgets.QHBoxLayout()
        info_layout.addWidget(btn)
        info_layout.addStretch()
        form.addRow(info_layout)

    def _build_wavelet_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.wt_min_scale = QtWidgets.QSpinBox()
        self.wt_min_scale.setRange(1, 500)
        self.wt_min_scale.setValue(3)
        self.wt_min_scale.setToolTip("Minimum wavelet scale in scans. Defines the narrowest peak width the algorithm will detect. Smaller values find narrower peaks.")
        lbl = QtWidgets.QLabel("Min wavelet scale (scans):")
        lbl.setToolTip(self.wt_min_scale.toolTip())
        form.addRow(lbl, self.wt_min_scale)

        self.wt_max_scale = QtWidgets.QSpinBox()
        self.wt_max_scale.setRange(2, 1000)
        self.wt_max_scale.setValue(11)
        self.wt_max_scale.setToolTip("Maximum wavelet scale in scans. Defines the widest peak width the algorithm will detect. Larger values find broader peaks.")
        lbl = QtWidgets.QLabel("Max wavelet scale (scans):")
        lbl.setToolTip(self.wt_max_scale.toolTip())
        form.addRow(lbl, self.wt_max_scale)

        self.wt_num_scales = QtWidgets.QSpinBox()
        self.wt_num_scales.setRange(4, 256)
        self.wt_num_scales.setValue(8)
        self.wt_num_scales.setToolTip("Number of wavelet scales evaluated between min and max. More scales give finer resolution but increase computation time.")
        lbl = QtWidgets.QLabel("Number of scales:")
        lbl.setToolTip(self.wt_num_scales.toolTip())
        form.addRow(lbl, self.wt_num_scales)

        self.wt_min_ridge_length = QtWidgets.QSpinBox()
        self.wt_min_ridge_length.setRange(1, 100)
        self.wt_min_ridge_length.setValue(4)
        self.wt_min_ridge_length.setToolTip("Minimum number of consecutive wavelet scales a ridge must span to be considered a real peak. Filters out noise ridges.")
        lbl = QtWidgets.QLabel("Min ridge length:")
        lbl.setToolTip(self.wt_min_ridge_length.toolTip())
        form.addRow(lbl, self.wt_min_ridge_length)

        self.wt_gap_threshold = QtWidgets.QSpinBox()
        self.wt_gap_threshold.setRange(0, 50)
        self.wt_gap_threshold.setValue(2)
        self.wt_gap_threshold.setToolTip("Maximum number of consecutive scale gaps allowed in a ridge before it is terminated. Small values enforce stricter continuity.")
        lbl = QtWidgets.QLabel("Gap threshold:")
        lbl.setToolTip(self.wt_gap_threshold.toolTip())
        form.addRow(lbl, self.wt_gap_threshold)

        self._add_info_button(form, "wavelettransform")
        self.tabWidget.addTab(tab, "Wavelet Transform")

    def _build_gradient_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.gd_smoothing_method = QtWidgets.QComboBox()
        self.gd_smoothing_method.addItems(["gaussian", "moving_average", "savitzky_golay"])
        self.gd_smoothing_method.setToolTip("Smoothing filter applied to the EIC before peak detection. Gaussian is recommended for most LC-HRMS data.")
        lbl = QtWidgets.QLabel("Smoothing method:")
        lbl.setToolTip(self.gd_smoothing_method.toolTip())
        form.addRow(lbl, self.gd_smoothing_method)

        self.gd_smoothing_window = QtWidgets.QSpinBox()
        self.gd_smoothing_window.setRange(1, 201)
        self.gd_smoothing_window.setValue(5)
        self.gd_smoothing_window.setToolTip("Width of the smoothing window in scans. Larger values produce stronger smoothing but may merge close peaks.")
        lbl = QtWidgets.QLabel("Smoothing window size:")
        lbl.setToolTip(self.gd_smoothing_window.toolTip())
        form.addRow(lbl, self.gd_smoothing_window)

        self.gd_smoothing_iterations = QtWidgets.QSpinBox()
        self.gd_smoothing_iterations.setRange(1, 20)
        self.gd_smoothing_iterations.setValue(1)
        self.gd_smoothing_iterations.setToolTip("Number of times the smoothing filter is applied. Multiple passes give stronger noise reduction.")
        lbl = QtWidgets.QLabel("Smoothing iterations:")
        lbl.setToolTip(self.gd_smoothing_iterations.toolTip())
        form.addRow(lbl, self.gd_smoothing_iterations)

        self.gd_min_increase_ratio = QtWidgets.QDoubleSpinBox()
        self.gd_min_increase_ratio.setRange(0.0, 1.0)
        self.gd_min_increase_ratio.setValue(0.05)
        self.gd_min_increase_ratio.setDecimals(3)
        self.gd_min_increase_ratio.setSingleStep(0.01)
        self.gd_min_increase_ratio.setToolTip("Minimum relative intensity increase required per scan during slope descent. Used to detect when the signal stalls.")
        lbl = QtWidgets.QLabel("Min increase ratio:")
        lbl.setToolTip(self.gd_min_increase_ratio.toolTip())
        form.addRow(lbl, self.gd_min_increase_ratio)

        self.gd_consecutive_scans = QtWidgets.QSpinBox()
        self.gd_consecutive_scans.setRange(1, 50)
        self.gd_consecutive_scans.setValue(3)
        self.gd_consecutive_scans.setToolTip("Number of consecutive scans with insufficient increase to trigger a stall. Descent stops when this many flat scans are observed.")
        lbl = QtWidgets.QLabel("Consecutive scans (stall check):")
        lbl.setToolTip(self.gd_consecutive_scans.toolTip())
        form.addRow(lbl, self.gd_consecutive_scans)

        self._add_info_button(form, "gradientdescent")
        self.tabWidget.addTab(tab, "Gradient Descent")

    def _build_gmm_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.gmm_distribution = QtWidgets.QComboBox()
        self.gmm_distribution.addItems(["gaussian", "emg"])
        self.gmm_distribution.setToolTip("Distribution model used for curve fitting. 'gaussian' fits symmetric peaks; 'emg' (Exponentially Modified Gaussian) fits asymmetric/tailing peaks.")
        lbl = QtWidgets.QLabel("Distribution type:")
        lbl.setToolTip(self.gmm_distribution.toolTip())
        form.addRow(lbl, self.gmm_distribution)

        self.gmm_baseline_fraction = QtWidgets.QDoubleSpinBox()
        self.gmm_baseline_fraction.setRange(0.0, 10.0)
        self.gmm_baseline_fraction.setValue(0.1)
        self.gmm_baseline_fraction.setDecimals(3)
        self.gmm_baseline_fraction.setSingleStep(0.01)
        self.gmm_baseline_fraction.setToolTip("Fraction of the maximum intensity used to define the baseline. Regions below this threshold are used to split the EIC into chunks.")
        lbl = QtWidgets.QLabel("Baseline fraction:")
        lbl.setToolTip(self.gmm_baseline_fraction.toolTip())
        form.addRow(lbl, self.gmm_baseline_fraction)

        self.gmm_min_chunk_size = QtWidgets.QSpinBox()
        self.gmm_min_chunk_size.setRange(2, 500)
        self.gmm_min_chunk_size.setValue(5)
        self.gmm_min_chunk_size.setToolTip("Minimum number of scans in a chunk for it to be processed. Chunks shorter than this are skipped.")
        lbl = QtWidgets.QLabel("Min chunk size (scans):")
        lbl.setToolTip(self.gmm_min_chunk_size.toolTip())
        form.addRow(lbl, self.gmm_min_chunk_size)

        self.gmm_max_peaks_per_chunk = QtWidgets.QSpinBox()
        self.gmm_max_peaks_per_chunk.setRange(1, 50)
        self.gmm_max_peaks_per_chunk.setValue(10)
        self.gmm_max_peaks_per_chunk.setToolTip("Maximum number of overlapping peaks to fit in a single chunk. Higher values allow more complex deconvolutions but increase computation time.")
        lbl = QtWidgets.QLabel("Max peaks per chunk:")
        lbl.setToolTip(self.gmm_max_peaks_per_chunk.toolTip())
        form.addRow(lbl, self.gmm_max_peaks_per_chunk)

        self.gmm_fit_max_iterations = QtWidgets.QSpinBox()
        self.gmm_fit_max_iterations.setRange(100, 100000)
        self.gmm_fit_max_iterations.setValue(5000)
        self.gmm_fit_max_iterations.setSingleStep(500)
        self.gmm_fit_max_iterations.setToolTip("Maximum iterations for the curve fitting optimizer. Increase if convergence warnings appear for complex peaks.")
        lbl = QtWidgets.QLabel("Fit max iterations:")
        lbl.setToolTip(self.gmm_fit_max_iterations.toolTip())
        form.addRow(lbl, self.gmm_fit_max_iterations)

        self._add_info_button(form, "gmm")
        self.tabWidget.addTab(tab, "GMM (Curve Fitting)")

    def _build_second_derivative_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.sd_sg_window = QtWidgets.QSpinBox()
        self.sd_sg_window.setRange(5, 201)
        self.sd_sg_window.setValue(11)
        self.sd_sg_window.setSingleStep(2)
        self.sd_sg_window.setToolTip("Window length for the Savitzky-Golay second derivative filter. Must be odd. Larger windows smooth more aggressively.")
        lbl = QtWidgets.QLabel("Savitzky-Golay window:")
        lbl.setToolTip(self.sd_sg_window.toolTip())
        form.addRow(lbl, self.sd_sg_window)

        self.sd_sg_polyorder = QtWidgets.QSpinBox()
        self.sd_sg_polyorder.setRange(2, 10)
        self.sd_sg_polyorder.setValue(3)
        self.sd_sg_polyorder.setToolTip("Polynomial order for the Savitzky-Golay filter. Must be less than the window length. Higher orders preserve sharper peak shapes.")
        lbl = QtWidgets.QLabel("SG polynomial order:")
        lbl.setToolTip(self.sd_sg_polyorder.toolTip())
        form.addRow(lbl, self.sd_sg_polyorder)

        self.sd_pre_smoothing = QtWidgets.QComboBox()
        self.sd_pre_smoothing.addItems(["none", "gaussian", "moving_average", "savitzky_golay"])
        self.sd_pre_smoothing.setToolTip("Optional smoothing applied to the EIC before computing the second derivative. Useful for noisy data.")
        lbl = QtWidgets.QLabel("Pre-smoothing method:")
        lbl.setToolTip(self.sd_pre_smoothing.toolTip())
        form.addRow(lbl, self.sd_pre_smoothing)

        self.sd_pre_smoothing_window = QtWidgets.QSpinBox()
        self.sd_pre_smoothing_window.setRange(1, 201)
        self.sd_pre_smoothing_window.setValue(5)
        self.sd_pre_smoothing_window.setToolTip("Window size for the optional pre-smoothing filter in scans.")
        lbl = QtWidgets.QLabel("Pre-smoothing window:")
        lbl.setToolTip(self.sd_pre_smoothing_window.toolTip())
        form.addRow(lbl, self.sd_pre_smoothing_window)

        self.sd_pre_smoothing_iterations = QtWidgets.QSpinBox()
        self.sd_pre_smoothing_iterations.setRange(1, 20)
        self.sd_pre_smoothing_iterations.setValue(1)
        self.sd_pre_smoothing_iterations.setToolTip("Number of times pre-smoothing is applied. More iterations give stronger smoothing.")
        lbl = QtWidgets.QLabel("Pre-smoothing iterations:")
        lbl.setToolTip(self.sd_pre_smoothing_iterations.toolTip())
        form.addRow(lbl, self.sd_pre_smoothing_iterations)

        self.sd_min_apex_intensity = QtWidgets.QDoubleSpinBox()
        self.sd_min_apex_intensity.setRange(0, 1e12)
        self.sd_min_apex_intensity.setValue(0.0)
        self.sd_min_apex_intensity.setDecimals(1)
        self.sd_min_apex_intensity.setToolTip("Minimum intensity at the peak apex. Peaks with an apex below this value are discarded.")
        lbl = QtWidgets.QLabel("Min apex intensity:")
        lbl.setToolTip(self.sd_min_apex_intensity.toolTip())
        form.addRow(lbl, self.sd_min_apex_intensity)

        self._add_info_button(form, "secondderivative")
        self.tabWidget.addTab(tab, "Second Derivative")

    def _build_matched_filter_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

        self.mf_expected_peak_width = QtWidgets.QSpinBox()
        self.mf_expected_peak_width.setRange(3, 500)
        self.mf_expected_peak_width.setValue(15)
        self.mf_expected_peak_width.setToolTip("Expected peak width in scans, used to generate the Gaussian template for cross-correlation.")
        lbl = QtWidgets.QLabel("Expected peak width (scans):")
        lbl.setToolTip(self.mf_expected_peak_width.toolTip())
        form.addRow(lbl, self.mf_expected_peak_width)

        self.mf_correlation_threshold = QtWidgets.QDoubleSpinBox()
        self.mf_correlation_threshold.setRange(0.0, 1.0)
        self.mf_correlation_threshold.setValue(0.5)
        self.mf_correlation_threshold.setDecimals(3)
        self.mf_correlation_threshold.setSingleStep(0.05)
        self.mf_correlation_threshold.setToolTip("Minimum normalized cross-correlation score for accepting a peak. Higher values require a closer match to the template shape.")
        lbl = QtWidgets.QLabel("Correlation threshold:")
        lbl.setToolTip(self.mf_correlation_threshold.toolTip())
        form.addRow(lbl, self.mf_correlation_threshold)

        self.mf_template_type = QtWidgets.QComboBox()
        self.mf_template_type.addItems(["gaussian"])
        self.mf_template_type.setToolTip("Shape of the template used for cross-correlation. Currently only Gaussian is supported.")
        lbl = QtWidgets.QLabel("Template type:")
        lbl.setToolTip(self.mf_template_type.toolTip())
        form.addRow(lbl, self.mf_template_type)

        self.mf_min_distance = QtWidgets.QSpinBox()
        self.mf_min_distance.setRange(1, 500)
        self.mf_min_distance.setValue(5)
        self.mf_min_distance.setToolTip("Minimum distance in scans between two adjacent peaks. Peaks closer than this are merged, keeping the stronger one.")
        lbl = QtWidgets.QLabel("Min distance between peaks (scans):")
        lbl.setToolTip(self.mf_min_distance.toolTip())
        form.addRow(lbl, self.mf_min_distance)

        self._add_info_button(form, "matchedfilter")
        self.tabWidget.addTab(tab, "Matched Filter")

    # ------------------------------------------------------------------
    # Settings management
    # ------------------------------------------------------------------

    _ALGORITHM_TAB_MAP = {
        "wavelettransform": 0,
        "gradientdescent": 1,
        "gmm": 2,
        "secondderivative": 3,
        "matchedfilter": 4,
    }

    _TAB_ALGORITHM_MAP = {v: k for k, v in _ALGORITHM_TAB_MAP.items()}

    def _load_settings(self, s: dict):
        """Populate controls from a settings dictionary."""
        algo = s.get("algorithm", "wavelettransform").lower()
        tab_idx = self._ALGORITHM_TAB_MAP.get(algo, 0)
        self.tabWidget.setCurrentIndex(tab_idx)

        # Wavelet Transform
        self.wt_min_scale.setValue(s.get("wt_min_scale", 3))
        self.wt_max_scale.setValue(s.get("wt_max_scale", 11))
        self.wt_num_scales.setValue(s.get("wt_num_scales", 8))
        self.wt_min_ridge_length.setValue(s.get("wt_min_ridge_length", 4))
        self.wt_gap_threshold.setValue(s.get("wt_gap_threshold", 2))

        # Gradient Descent
        sm = s.get("gd_smoothing_method", "gaussian")
        idx = self.gd_smoothing_method.findText(sm)
        if idx >= 0:
            self.gd_smoothing_method.setCurrentIndex(idx)
        self.gd_min_apex_to_flank_factor_stub = None  # removed; now in post-processing filters
        self.gd_smoothing_window.setValue(s.get("gd_smoothing_window", 5))
        self.gd_smoothing_iterations.setValue(s.get("gd_smoothing_iterations", 1))
        self.gd_min_increase_ratio.setValue(s.get("gd_min_increase_ratio", 0.05))
        self.gd_consecutive_scans.setValue(s.get("gd_consecutive_scans", 3))

        # GMM
        dist = s.get("gmm_distribution", "gaussian")
        idx = self.gmm_distribution.findText(dist)
        if idx >= 0:
            self.gmm_distribution.setCurrentIndex(idx)
        self.gmm_baseline_fraction.setValue(s.get("gmm_baseline_fraction", 0.1))
        self.gmm_min_chunk_size.setValue(s.get("gmm_min_chunk_size", 5))
        self.gmm_max_peaks_per_chunk.setValue(s.get("gmm_max_peaks_per_chunk", 10))
        self.gmm_fit_max_iterations.setValue(s.get("gmm_fit_max_iterations", 5000))

        # Second Derivative
        self.sd_sg_window.setValue(s.get("sd_sg_window", 11))
        self.sd_sg_polyorder.setValue(s.get("sd_sg_polyorder", 3))
        ps = s.get("sd_pre_smoothing", "none")
        idx = self.sd_pre_smoothing.findText(ps)
        if idx >= 0:
            self.sd_pre_smoothing.setCurrentIndex(idx)
        self.sd_pre_smoothing_window.setValue(s.get("sd_pre_smoothing_window", 5))
        self.sd_pre_smoothing_iterations.setValue(s.get("sd_pre_smoothing_iterations", 1))
        self.sd_min_apex_intensity.setValue(s.get("sd_min_apex_intensity", 0.0))

        # Matched Filter
        self.mf_expected_peak_width.setValue(s.get("mf_expected_peak_width", 15))
        self.mf_correlation_threshold.setValue(s.get("mf_correlation_threshold", 0.5))
        tt = s.get("mf_template_type", "gaussian")
        idx = self.mf_template_type.findText(tt)
        if idx >= 0:
            self.mf_template_type.setCurrentIndex(idx)
        self.mf_min_distance.setValue(s.get("mf_min_distance", 5))

        # Peak width / FWHM filter
        self.grp_peak_filter.setChecked(s.get("pf_enabled", True))
        self.pf_min_peak_width.setValue(s.get("pf_min_peak_width", 3.0))
        self.pf_max_peak_width.setValue(s.get("pf_max_peak_width", 50.0))
        self.pf_min_fwhm.setValue(s.get("pf_min_fwhm", 2.0))
        self.pf_max_fwhm.setValue(s.get("pf_max_fwhm", 10.0))
        self.pf_min_apex_to_flank_factor.setValue(s.get("pf_min_apex_to_flank_factor", 3.0))
        self.pf_min_apex_to_flank_increase.setValue(s.get("pf_min_apex_to_flank_increase", 100000.0))
        self.pf_min_snr.setValue(s.get("pf_min_snr", 3.0))

    def _collect_settings(self) -> dict:
        """Read all controls and return a flat settings dict."""
        tab_idx = self.tabWidget.currentIndex()
        algo = self._TAB_ALGORITHM_MAP.get(tab_idx, "wavelettransform")
        s = {"algorithm": algo}

        # Wavelet Transform
        s["wt_min_scale"] = self.wt_min_scale.value()
        s["wt_max_scale"] = self.wt_max_scale.value()
        s["wt_num_scales"] = self.wt_num_scales.value()
        s["wt_min_ridge_length"] = self.wt_min_ridge_length.value()
        s["wt_gap_threshold"] = self.wt_gap_threshold.value()

        # Gradient Descent
        s["gd_smoothing_method"] = self.gd_smoothing_method.currentText()
        s["gd_smoothing_window"] = self.gd_smoothing_window.value()
        s["gd_smoothing_iterations"] = self.gd_smoothing_iterations.value()
        s["gd_min_increase_ratio"] = self.gd_min_increase_ratio.value()
        s["gd_consecutive_scans"] = self.gd_consecutive_scans.value()

        # GMM
        s["gmm_distribution"] = self.gmm_distribution.currentText()
        s["gmm_baseline_fraction"] = self.gmm_baseline_fraction.value()
        s["gmm_min_chunk_size"] = self.gmm_min_chunk_size.value()
        s["gmm_max_peaks_per_chunk"] = self.gmm_max_peaks_per_chunk.value()
        s["gmm_fit_max_iterations"] = self.gmm_fit_max_iterations.value()

        # Second Derivative
        s["sd_sg_window"] = self.sd_sg_window.value()
        s["sd_sg_polyorder"] = self.sd_sg_polyorder.value()
        s["sd_pre_smoothing"] = self.sd_pre_smoothing.currentText()
        s["sd_pre_smoothing_window"] = self.sd_pre_smoothing_window.value()
        s["sd_pre_smoothing_iterations"] = self.sd_pre_smoothing_iterations.value()
        s["sd_min_apex_intensity"] = self.sd_min_apex_intensity.value()

        # Matched Filter
        s["mf_expected_peak_width"] = self.mf_expected_peak_width.value()
        s["mf_correlation_threshold"] = self.mf_correlation_threshold.value()
        s["mf_template_type"] = self.mf_template_type.currentText()
        s["mf_min_distance"] = self.mf_min_distance.value()

        # Peak width / FWHM + apex/flank filter
        s["pf_enabled"] = self.grp_peak_filter.isChecked()
        s["pf_min_peak_width"] = self.pf_min_peak_width.value()
        s["pf_max_peak_width"] = self.pf_max_peak_width.value()
        s["pf_min_fwhm"] = self.pf_min_fwhm.value()
        s["pf_max_fwhm"] = self.pf_max_fwhm.value()
        s["pf_min_apex_to_flank_factor"] = self.pf_min_apex_to_flank_factor.value()
        s["pf_min_apex_to_flank_increase"] = self.pf_min_apex_to_flank_increase.value()
        s["pf_min_snr"] = self.pf_min_snr.value()

        return s

    def _accept(self):
        self._settings = self._collect_settings()
        self.settingsAccepted.emit(self._settings)
        self.accept()

    def _restore_defaults(self):
        self._load_settings(self.get_default_settings())

    def get_settings(self) -> dict:
        """Return the last accepted settings dict."""
        return dict(self._settings)

    @staticmethod
    def get_default_settings() -> dict:
        """Return default settings."""
        return {
            "algorithm": "gradientdescent",
            "gd_smoothing_method": "gaussian",
            "gd_smoothing_window": 5,
            "gd_smoothing_iterations": 1,
            "gd_min_increase_ratio": 0.05,
            "gd_consecutive_scans": 3,
            "wt_min_scale": 11,
            "wt_max_scale": 19,
            "wt_num_scales": 8,
            "wt_min_ridge_length": 4,
            "wt_gap_threshold": 2,
            "pf_enabled": True,
            "pf_min_peak_width": 3.0,
            "pf_max_peak_width": 50.0,
            "pf_min_fwhm": 2.0,
            "pf_max_fwhm": 10.0,
            "pf_min_apex_to_flank_factor": 3.0,
            "pf_min_apex_to_flank_increase": 100000.0,
            "pf_min_snr": 3.0,
        }

    # ------------------------------------------------------------------
    # EIC management
    # ------------------------------------------------------------------

    def get_eic_rows(self):
        """Return the current EIC table contents as a list of (file_path, filter_text, mz, ppm).

        Stores the actual file path (from combobox userData) and filter line text
        so that the rows can be stably persisted and restored across dialog openings
        and group file save/load cycles.
        """
        rows = []
        for row in range(self.eicTable.rowCount()):
            fileW = self.eicTable.cellWidget(row, 0)
            filterW = self.eicTable.cellWidget(row, 1)
            mzW = self.eicTable.cellWidget(row, 2)
            ppmW = self.eicTable.cellWidget(row, 3)
            file_path = fileW.currentData() if fileW else None
            filter_text = filterW.currentText() if filterW else ""
            mz = mzW.value() if mzW else 0.0
            ppm = ppmW.value() if ppmW else 5.0
            rows.append((file_path, filter_text, mz, ppm))
        return rows

    def _restore_eic_rows(self, eic_rows):
        """Populate the EIC table from a list of (file_path, filter_text, mz, ppm) tuples.

        Matches the file path against combobox userData and the filter text against
        combobox item texts.  Falls back gracefully when an entry is not found.
        """
        for entry in eic_rows:
            file_path, filter_text, mz, ppm = entry
            self._add_eic_row()
            row = self.eicTable.rowCount() - 1
            fileW = self.eicTable.cellWidget(row, 0)
            if fileW and file_path is not None:
                for i in range(fileW.count()):
                    if fileW.itemData(i) == file_path:
                        fileW.setCurrentIndex(i)
                        break
            filterW = self.eicTable.cellWidget(row, 1)
            if filterW and filter_text:
                idx = filterW.findText(filter_text)
                if idx >= 0:
                    filterW.setCurrentIndex(idx)
            mzW = self.eicTable.cellWidget(row, 2)
            if mzW:
                mzW.setValue(mz)
            ppmW = self.eicTable.cellWidget(row, 3)
            if ppmW:
                ppmW.setValue(ppm)

    def _add_eic_row(self):
        """Add a row to the EIC definition table using loaded files and filter lines."""
        row = self.eicTable.rowCount()
        self.eicTable.insertRow(row)

        # File – QComboBox populated from loaded experimental groups
        fileCombo = QtWidgets.QComboBox()
        if self._sample_files:
            for group_name, fpath in self._sample_files:
                fname = os.path.basename(fpath)
                fileCombo.addItem(f"{group_name}  \u2014  {fname}", userData=fpath)
        else:
            fileCombo.addItem("(no files loaded)", userData=None)
        self.eicTable.setCellWidget(row, 0, fileCombo)

        # Filter line – QComboBox with positive and negative scan events
        filterCombo = QtWidgets.QComboBox()
        all_filter_lines = []
        for pol_key in ("+", "-"):
            for fl in self._scan_events.get(pol_key, []):
                if fl and fl != "None":
                    all_filter_lines.append(fl)
        if all_filter_lines:
            for fl in all_filter_lines:
                filterCombo.addItem(fl)
        else:
            filterCombo.addItem("(no filter lines)")
        self.eicTable.setCellWidget(row, 1, filterCombo)

        # m/z
        mzSpin = QtWidgets.QDoubleSpinBox()
        mzSpin.setRange(0, 99999.0)
        mzSpin.setDecimals(5)
        mzSpin.setValue(100.0)
        self.eicTable.setCellWidget(row, 2, mzSpin)

        # ppm
        ppmSpin = QtWidgets.QDoubleSpinBox()
        ppmSpin.setRange(0.1, 1000.0)
        ppmSpin.setDecimals(1)
        ppmSpin.setValue(5.0)
        self.eicTable.setCellWidget(row, 3, ppmSpin)

    def _remove_eic_rows(self):
        """Remove selected rows from the EIC table."""
        rows = sorted({idx.row() for idx in self.eicTable.selectedIndexes()}, reverse=True)
        for r in rows:
            self.eicTable.removeRow(r)

    def _duplicate_eic_row(self):
        """Duplicate the selected EIC row, inserting a copy directly below it."""
        selected_rows = sorted({idx.row() for idx in self.eicTable.selectedIndexes()})
        if not selected_rows:
            return
        # Duplicate the last selected row (insert after it)
        src_row = selected_rows[-1]
        insert_at = src_row + 1

        # Read current widget values from the source row
        fileW = self.eicTable.cellWidget(src_row, 0)
        filterW = self.eicTable.cellWidget(src_row, 1)
        mzW = self.eicTable.cellWidget(src_row, 2)
        ppmW = self.eicTable.cellWidget(src_row, 3)

        src_file_idx = fileW.currentIndex() if fileW else 0
        src_filter_idx = filterW.currentIndex() if filterW else 0
        src_mz = mzW.value() if mzW else 100.0
        src_ppm = ppmW.value() if ppmW else 5.0

        # Insert new row at position
        self.eicTable.insertRow(insert_at)

        # File combo
        newFileCombo = QtWidgets.QComboBox()
        if self._sample_files:
            for group_name, fpath in self._sample_files:
                fname = os.path.basename(fpath)
                newFileCombo.addItem(f"{group_name}  \u2014  {fname}", userData=fpath)
        else:
            newFileCombo.addItem("(no files loaded)", userData=None)
        newFileCombo.setCurrentIndex(src_file_idx)
        self.eicTable.setCellWidget(insert_at, 0, newFileCombo)

        # Filter combo
        newFilterCombo = QtWidgets.QComboBox()
        all_filter_lines = []
        for pol_key in ("+", "-"):
            for fl in self._scan_events.get(pol_key, []):
                if fl and fl != "None":
                    all_filter_lines.append(fl)
        if all_filter_lines:
            for fl in all_filter_lines:
                newFilterCombo.addItem(fl)
        else:
            newFilterCombo.addItem("(no filter lines)")
        newFilterCombo.setCurrentIndex(src_filter_idx)
        self.eicTable.setCellWidget(insert_at, 1, newFilterCombo)

        # m/z
        newMzSpin = QtWidgets.QDoubleSpinBox()
        newMzSpin.setRange(0, 99999.0)
        newMzSpin.setDecimals(5)
        newMzSpin.setValue(src_mz)
        self.eicTable.setCellWidget(insert_at, 2, newMzSpin)

        # ppm
        newPpmSpin = QtWidgets.QDoubleSpinBox()
        newPpmSpin.setRange(0.1, 1000.0)
        newPpmSpin.setDecimals(1)
        newPpmSpin.setValue(src_ppm)
        self.eicTable.setCellWidget(insert_at, 3, newPpmSpin)

        # Select the duplicated row
        self.eicTable.selectRow(insert_at)

    def _duplicate_eic_row_with_mz_shift(self, mz_shift: float):
        """Duplicate the selected EIC row and offset the m/z by *mz_shift*."""
        self._duplicate_eic_row()
        # _duplicate_eic_row selects the new row right after insertion
        new_row = self.eicTable.currentRow()
        if new_row < 0:
            return
        mzW = self.eicTable.cellWidget(new_row, 2)
        if mzW is not None:
            mzW.setValue(mzW.value() + mz_shift)

    def _run_preview(self):
        """Attempt to load EICs and run peak picking for preview."""
        # Gather EIC definitions
        eics = []
        for row in range(self.eicTable.rowCount()):
            fileW = self.eicTable.cellWidget(row, 0)
            fpath = fileW.currentData() if fileW else None
            if fpath is None and fileW is not None:
                fpath = fileW.currentText()  # fallback for standalone mode
            fpath = (fpath or "").strip()
            filterW = self.eicTable.cellWidget(row, 1)
            filter_line = filterW.currentText() if filterW else None
            if filter_line in (None, "", "(no filter lines)"):
                filter_line = None
            mzW = self.eicTable.cellWidget(row, 2)
            mz = mzW.value() if mzW else 0.0
            ppmW = self.eicTable.cellWidget(row, 3)
            ppm = ppmW.value() if ppmW else 5.0
            if fpath and fpath not in ("(no files loaded)",):
                eics.append((fpath, filter_line, mz, ppm))

        if not eics:
            self._show_status_text("No EICs defined. Add at least one EIC row and click 'Pick Peaks'.")
            return

        # Show progress bar
        self._progress_bar.setRange(0, len(eics))
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Loading EICs…  0 / %d" % len(eics))
        self._progress_bar.show()
        QtWidgets.QApplication.processEvents()

        # Try to load and pick peaks
        # NOTE: imports are lazy here because this dialog can be opened outside the
        # full MetExtract environment (e.g. standalone testing) where Chromatogram
        # and peakpickers may not be available.
        try:
            from ..Chromatogram import Chromatogram
            from ..chromPeakPicking.peakpickers import (
                GMMPeakPicker,
                GradientDescentPeakPicker,
                MatchedFilterPeakPicker,
                SecondDerivativePeakPicker,
                WaveletTransformPeakPicker,
                filter_peaks,
            )

            settings = self._collect_settings()

            # ── Build picker once (same settings for all EICs) ─────────
            algo = settings.get("algorithm", "wavelettransform")
            if algo == "wavelettransform":
                picker = WaveletTransformPeakPicker(
                    min_scale=settings.get("wt_min_scale", 3),
                    max_scale=settings.get("wt_max_scale", 11),
                    num_scales=settings.get("wt_num_scales", 8),
                    min_ridge_length=settings.get("wt_min_ridge_length", 4),
                    gap_threshold=settings.get("wt_gap_threshold", 2),
                )
            elif algo == "gradientdescent":
                picker = GradientDescentPeakPicker(
                    smoothing_method=settings.get("gd_smoothing_method", "gaussian"),
                    smoothing_window=settings.get("gd_smoothing_window", 5),
                    smoothing_iterations=settings.get("gd_smoothing_iterations", 1),
                    min_increase_ratio=settings.get("gd_min_increase_ratio", 0.05),
                    consecutive_scans=settings.get("gd_consecutive_scans", 3),
                )
            elif algo == "gmm":
                picker = GMMPeakPicker(
                    distribution=settings.get("gmm_distribution", "gaussian"),
                )
            elif algo == "secondderivative":
                picker = SecondDerivativePeakPicker(
                    sg_window=settings.get("sd_sg_window", 11),
                )
            elif algo == "matchedfilter":
                picker = MatchedFilterPeakPicker(
                    expected_peak_width=settings.get("mf_expected_peak_width", 15),
                )
            else:
                picker = WaveletTransformPeakPicker()

            # ── Group EIC rows by file path to load each file only once ─
            # eic_by_file: {fpath: [(original_idx, filter_line, mz, ppm), ...]}
            from collections import OrderedDict

            eic_by_file: dict = OrderedDict()
            for orig_idx, (fpath, filter_line, mz, ppm) in enumerate(eics):
                eic_by_file.setdefault(fpath, []).append((orig_idx, filter_line, mz, ppm))

            # Pre-allocate results list (filled by original index)
            results = [None] * len(eics)

            processed = 0
            for fpath, file_eics in eic_by_file.items():
                # Load the file once
                chromatogram = None
                file_load_error = None
                self._progress_bar.setFormat(f"Loading {os.path.basename(fpath)}…")
                QtWidgets.QApplication.processEvents()
                try:
                    chromatogram = Chromatogram()
                    chromatogram.parse_file(fpath)
                except Exception as e:
                    file_load_error = str(e)

                for orig_idx, filter_line, mz, ppm in file_eics:
                    processed += 1
                    self._progress_bar.setFormat(f"Processing {processed} / {len(eics)}: {os.path.basename(fpath)}")
                    self._progress_bar.setValue(processed - 1)
                    QtWidgets.QApplication.processEvents()

                    rt_array = None
                    int_array = None
                    peaks = []
                    load_error = file_load_error
                    peak_error = None

                    if chromatogram is not None:
                        try:
                            eic_data, times, scan_ids, mzs = chromatogram.getEIC(mz, ppm, filterLine=filter_line)
                            rt_array = np.array(times, dtype=float)
                            int_array = np.array(eic_data, dtype=float)
                        except Exception as e:
                            load_error = str(e)

                    if rt_array is not None:
                        try:
                            peaks = picker.pick_peaks(rt_array, int_array) or []

                            if settings.get("pf_enabled", False):
                                peaks = filter_peaks(
                                    peaks,
                                    min_peak_width=settings.get("pf_min_peak_width", 0.0),
                                    max_peak_width=settings.get("pf_max_peak_width", 9999.0),
                                    min_fwhm=settings.get("pf_min_fwhm", 0.0),
                                    max_fwhm=settings.get("pf_max_fwhm", 9999.0),
                                    min_apex_to_flank_factor=settings.get("pf_min_apex_to_flank_factor", 0.0),
                                    min_apex_to_flank_increase=settings.get("pf_min_apex_to_flank_increase", 0.0),
                                    min_snr=settings.get("pf_min_snr", 0.0),
                                )
                        except Exception as e:
                            peak_error = str(e)
                            peaks = []

                    # results tuple: (fpath, filter_line, mz, ppm, rt_array, int_array, peaks, load_error, peak_error)
                    results[orig_idx] = (fpath, filter_line, mz, ppm, rt_array, int_array, peaks, load_error, peak_error)

            self._progress_bar.setValue(len(eics))
            self._progress_bar.setFormat("Done")
            QtWidgets.QApplication.processEvents()

            if self._has_matplotlib:
                self._plot_eic_results(results)
            else:
                # Text summary fallback
                lines = []
                for res in results:
                    fpath, filter_line, mz, ppm = res[:4]
                    rt_array, int_array, peaks, load_error, peak_error = res[4], res[5], res[6], res[7], res[8]
                    fl_label = filter_line if filter_line else "(auto)"
                    lines.append("EIC: %s  m/z=%.5f \u00b1%.1f ppm  [%s]" % (fpath.split("/")[-1], mz, ppm, fl_label))
                    if load_error:
                        lines.append("  Load error: %s" % load_error)
                    else:
                        if peak_error:
                            lines.append("  Peak picking error: %s" % peak_error)
                        lines.append("  Found %d peak(s):" % len(peaks))
                        for i, pk in enumerate(peaks):
                            lines.append("    Peak %d: RT=%.3f min, FWHM=%.3f, SNR=%.1f, Area=%.0f" % (i + 1, pk.apex_rt / 60.0, pk.fwhm, pk.snr, pk.area))
                    lines.append("")
                self.eicPlotArea.setText("\n".join(lines))
        except ImportError:
            self._progress_bar.hide()
            self._show_status_text("Preview not available: required modules (Chromatogram, peakpickers) not found.\nThis feature requires the full MetExtract II environment.")
        except Exception as e:
            self._progress_bar.hide()
            self._show_status_text("Error during preview: %s" % str(e), color="red")
        else:
            # Keep the bar visible briefly then hide it after plot is shown
            QtCore.QTimer.singleShot(1500, self._progress_bar.hide)

    # ------------------------------------------------------------------
    # Algorithm info
    # ------------------------------------------------------------------

    def _show_algo_info(self, algo_key):
        """Show an info dialog with algorithm description and references."""
        title, text = _ALGO_INFO.get(algo_key, ("Unknown", "No information available."))
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    dlg = PeakPickingSettingsDialog()
    if dlg.exec() == QtWidgets.QDialog.Accepted:
        print("Settings:", dlg.get_settings())
    sys.exit(0)
