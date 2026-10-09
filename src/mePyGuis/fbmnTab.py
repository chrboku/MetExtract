"""Feature-Based Molecular Networking (FBMN) panel.

Nothias, LF., Petras, D., Schmid, R. et al. Feature-based molecular networking in the GNPS
analysis environment. Nat Methods 17, 905-908 (2020). https://doi.org/10.1038/s41592-020-0933-6
"""

import inspect
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.optimize import linear_sum_assignment
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

try:
    from matchms import Spectrum as MatchmsSpectrum
    from matchms.filtering import normalize_intensities as matchms_normalize_intensities

    MATCHMS_AVAILABLE = True
except Exception:
    MATCHMS_AVAILABLE = False


FORM_COLORS = {"native": "#1E90FF", "labeled": "#B22222"}
HIGHLIGHT_COLOR = "#32CD32"
FORM_OPTIONS = [
    ("Native spectra", ["native"]),
    ("Labeled spectra", ["labeled"]),
    ("Native and labeled spectra (separate nodes)", ["native", "labeled"]),
]
SELECTION_OPTIONS = [("Most abundant spectrum", "most_abundant")]
NETWORK_SHEET = "FBMN_network"
# stay below Excel's limit of 32767 characters per cell; longer JSON texts are split over several rows
NETWORK_CELL_LIMIT = 32000


def _plain(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _node_id(node, num_counts):
    return str(node["num"]) if num_counts[node["num"]] == 1 else f"{node['num']}_{node['form']}"


def _form_label(form):
    return "M\u2032" if form == "labeled" else "M"


def _feature_title(data):
    xn = data.get("xn")
    return f"FP {data['num']} ({_form_label(data['form'])}) | Met {data.get('ogroup') or '-'} | Ion mode {data.get('polarity') or '-'} | MZ {data['mz']:.4f} | RT {data['rt']:.2f} min | Xn {xn if xn is not None else '-'}"


def _scan_info_parts(info):
    """(sample, RT text, precursor intensity text) of a spectrum entry {sample, rt, prec_intensity}."""
    info = info or {}
    rt, pi = info.get("rt"), info.get("prec_intensity")
    return (info.get("sample") or "-", f"{rt:.2f} min" if rt is not None else "-", f"{pi:.3g}" if pi is not None else "-")


def _scan_info_text(info):
    sample, rt, pi = _scan_info_parts(info)
    return f"sample {sample} | MS/MS RT {rt} | prec. intensity {pi}"


def _spectrum_arrays(spec):
    mz = np.asarray(spec.peaks.mz, dtype=float)
    intens = np.asarray(spec.peaks.intensities, dtype=float)
    max_int = intens.max() if intens.size else 0.0
    return mz, intens, (intens / max_int * 100.0 if max_int > 0 else intens)


def _matched_peak_pairs(algorithm, spec_a, spec_b):
    """Return the (index_a, index_b) fragment pairs matchms uses for the score of `algorithm`
    (same candidate shifts, peak weights and greedy/Hungarian assignment as matchms)."""
    name = type(algorithm).__name__
    tolerance = getattr(algorithm, "tolerance", 0.1)
    mz_power = getattr(algorithm, "mz_power", 0.0)
    intensity_power = getattr(algorithm, "intensity_power", 1.0)
    mz_a, it_a, _ = _spectrum_arrays(spec_a)
    mz_b, it_b, _ = _spectrum_arrays(spec_b)
    if mz_a.size == 0 or mz_b.size == 0:
        return []

    shift = float(spec_a.get("precursor_mz")) - float(spec_b.get("precursor_mz"))
    if name.startswith("NeutralLosses"):
        shifts = [shift]
    elif name.startswith("ModifiedCosine") and abs(shift) > tolerance:
        shifts = [0.0, shift]
    else:
        shifts = [0.0]

    w_a = mz_a**mz_power * it_a**intensity_power
    w_b = mz_b**mz_power * it_b**intensity_power
    weights = {}
    for s in shifts:
        for i, j in zip(*np.nonzero(np.abs(mz_a[:, None] - (mz_b[None, :] + s)) <= tolerance)):
            w = float(w_a[i] * w_b[j])
            if w > weights.get((int(i), int(j)), -1.0):
                weights[(int(i), int(j))] = w
    if not weights:
        return []

    if "Hungarian" in name:
        rows = sorted({i for i, _ in weights})
        cols = sorted({j for _, j in weights})
        row_idx = {r: k for k, r in enumerate(rows)}
        col_idx = {c: k for k, c in enumerate(cols)}
        matrix = np.zeros((len(rows), len(cols)))
        for (i, j), w in weights.items():
            matrix[row_idx[i], col_idx[j]] = w
        r_ind, c_ind = linear_sum_assignment(matrix, maximize=True)
        return [(rows[r], cols[c]) for r, c in zip(r_ind, c_ind) if matrix[r, c] > 0]

    used_a, used_b, pairs = set(), set(), []
    for (i, j), _ in sorted(weights.items(), key=lambda kv: -kv[1]):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


SQRT_TICKS_PCT = [0, 5, 10, 25, 50, 75, 100]


def _y_values(rel, sqrt_axis):
    """Map relative intensities (%) to plotted heights (square root of % if `sqrt_axis`)."""
    return np.sqrt(np.clip(rel, 0.0, None)) if sqrt_axis else rel


def _y_top(sqrt_axis):
    return 11.0 if sqrt_axis else 110.0


def _format_intensity_axis(ax, sqrt_axis, mirror, fontsize):
    if sqrt_axis:
        ticks = [math.sqrt(p) for p in SQRT_TICKS_PCT]
        labels = [f"{p:g}" for p in SQRT_TICKS_PCT]
        if mirror:
            ticks = [-t for t in reversed(ticks[1:])] + ticks
            labels = list(reversed(labels[1:])) + labels
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_ylabel("Rel. intensity (%, square-root scale)", fontsize=fontsize)
    else:
        ax.set_ylabel("Rel. intensity (%)", fontsize=fontsize)
    top = _y_top(sqrt_axis)
    ax.set_ylim(-top if mirror else 0, top)


def _draw_spectrum(fig, data, spec, fontsize=8, sqrt_axis=False, info=None):
    """Returns (ax, peaks) with peaks = [(mz, rel, plotted_y)]."""
    fig.clear()
    ax = fig.add_subplot(111)
    mz, _, rel = _spectrum_arrays(spec)
    y = _y_values(rel, sqrt_axis)
    ax.vlines(mz, 0, y, colors=FORM_COLORS.get(data["form"], "dodgerblue"), linewidth=1.2)
    precursor = float(spec.get("precursor_mz"))
    ax.vlines(precursor, 0, 1e9, colors="gray", linestyles="--", linewidth=1.0)
    _format_intensity_axis(ax, sqrt_axis, False, fontsize)
    ax.set_title(f"{_feature_title(data)}\nprec. m/z {precursor:.4f} (dashed) | {len(mz)} fragments | {_scan_info_text(info)}", fontsize=fontsize)
    ax.set_xlabel("m/z", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    fig.tight_layout()
    return ax, [(mz, rel, y)]


def _draw_mirror(fig, data_a, spec_a, data_b, spec_b, score, pairs, fontsize=8, sqrt_axis=False, info_a=None, info_b=None):
    """Mirror plot (A up, B down); fragments matched by the matchms algorithm are drawn thicker.
    Returns (ax, peaks) with peaks = [(mz, rel, plotted_y)] per spectrum (y negative for B)."""
    fig.clear()
    ax = fig.add_subplot(111)
    stats = []
    peaks = []
    for label, sign, spec, matched, color, info in (("A", 1, spec_a, {i for i, _ in pairs}, "dodgerblue", info_a), ("B", -1, spec_b, {j for _, j in pairs}, "firebrick", info_b)):
        mz, intens, rel = _spectrum_arrays(spec)
        y = sign * _y_values(rel, sqrt_axis)
        mask = np.zeros(mz.size, dtype=bool)
        mask[list(matched)] = True
        ax.vlines(mz[~mask], 0, y[~mask], colors=color, linewidth=1.0, alpha=0.6)
        ax.vlines(mz[mask], 0, y[mask], colors=color, linewidth=3.0)
        ax.vlines(float(spec.get("precursor_mz")), 0, sign * 1e9, colors=color, linestyles="--", linewidth=1.0)
        peaks.append((mz, rel, y))
        explained = intens[mask].sum() / intens.sum() * 100.0 if intens.sum() > 0 else 0.0
        stats.append(f"{label}: {_scan_info_text(info)}; {mz.size} fragments; the {int(mask.sum())} matched fragments explain {explained:.1f}% of the summed intensity; precursor m/z {float(spec.get('precursor_mz')):.4f} (dashed)")
    stats.append(f"Score calculated from all {len(spec_a.peaks.mz)} + {len(spec_b.peaks.mz)} fragments; {len(pairs)} matched fragment pairs contribute (thick lines)")

    ax.axhline(0, color="black", linewidth=0.8)
    _format_intensity_axis(ax, sqrt_axis, True, fontsize)
    ax.set_title(f"A: {_feature_title(data_a)}\nB: {_feature_title(data_b)}\nScore {score:.3f}", fontsize=fontsize)
    ax.set_xlabel("m/z", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    fig.text(0.01, 0.01, "\n".join(stats), fontsize=fontsize - 1, va="bottom", ha="left")
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    return ax, peaks


class _MirrorDialog(QtWidgets.QDialog):
    """Non-modal mirror plot window: hover/click fragments for annotations, mouse-wheel zoom,
    right-button drag to pan, optional square-root intensity axis."""

    def __init__(self, parent, draw_func, title, sqrt_axis, controls=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        self._draw_func = draw_func
        self._pinned = []
        self._pinned_artists = []
        self._hover_artist = None
        self._pan_start = None

        layout = QtWidgets.QVBoxLayout(self)
        self.fig = Figure((10.0, 6.5), dpi=90, facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(self.toolbar)
        self.sqrt_check = QtWidgets.QCheckBox("Square-root intensity axis")
        self.sqrt_check.setChecked(sqrt_axis)
        self.sqrt_check.toggled.connect(self._redraw)
        top_row.addWidget(self.sqrt_check)
        top_row.addStretch(1)
        layout.addLayout(top_row)
        if controls is not None:
            layout.addWidget(controls)
        hint = QtWidgets.QLabel("Hover a fragment for its m/z and intensity; click to pin/unpin the annotation. Mouse wheel: zoom m/z (Shift+wheel: intensity); right-button drag: pan.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._redraw()

    def _redraw(self, *args):
        self._sqrt = self.sqrt_check.isChecked()
        self._ax, self._peaks = self._draw_func(self.fig, self._sqrt)
        self._hover_artist = None
        self._pinned_artists = [self._annotate(p, pinned=True) for p in self._pinned]
        self.canvas.draw_idle()
        self.toolbar.update()

    def refresh(self):
        """Redraw after the plotted spectra changed (pinned annotations are dropped)."""
        self._pinned = []
        self._redraw()

    def _find_peak(self, event):
        """Nearest fragment within a few pixels in m/z whose plotted height is within +/-5 %
        (+/-0.3 sqrt(%) units on the square-root axis) of the cursor."""
        if event.inaxes is not self._ax or event.xdata is None or event.ydata is None:
            return None
        y_tol = 0.3 if self._sqrt else 5.0
        best, best_dx = None, 6.0
        for side_idx, (mz, rel, y) in enumerate(self._peaks):
            if mz.size == 0:
                continue
            dx = np.abs(self._ax.transData.transform(np.column_stack([mz, y]))[:, 0] - event.x)
            for k in np.nonzero((dx <= best_dx) & (np.abs(y - event.ydata) <= y_tol))[0]:
                if dx[k] <= best_dx:
                    best_dx = dx[k]
                    best = (float(mz[k]), float(rel[k]), 1 if side_idx == 0 else -1)
        return best

    def _annotate(self, peak, pinned):
        mz, rel, side = peak
        y = side * float(_y_values(np.array([rel]), self._sqrt)[0])
        return self._ax.annotate(
            f"m/z {mz:.4f}\n{rel:.1f} %",
            xy=(mz, y),
            xytext=(8, 8 if side > 0 else -8),
            textcoords="offset points",
            va="bottom" if side > 0 else "top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow" if pinned else "white", edgecolor="#888888", alpha=0.95),
            zorder=10,
        )

    def _on_move(self, event):
        if self._pan_start is not None and event.x is not None:
            x0, y0, xlim, ylim = self._pan_start
            inv = self._ax.transData.inverted()
            (dx0, dy0), (dx1, dy1) = inv.transform([(x0, y0), (event.x, event.y)])
            self._ax.set_xlim(xlim[0] - (dx1 - dx0), xlim[1] - (dx1 - dx0))
            self._ax.set_ylim(ylim[0] - (dy1 - dy0), ylim[1] - (dy1 - dy0))
            self.canvas.draw_idle()
            return
        if self._hover_artist is not None:
            self._hover_artist.remove()
            self._hover_artist = None
        peak = self._find_peak(event)
        if peak is not None and peak not in self._pinned:
            self._hover_artist = self._annotate(peak, pinned=False)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes is not self._ax or self.toolbar.mode != "":
            return
        if event.button == 3:
            self._pan_start = (event.x, event.y, self._ax.get_xlim(), self._ax.get_ylim())
            return
        if event.button != 1:
            return
        peak = self._find_peak(event)
        if peak is None:
            return
        if peak in self._pinned:
            idx = self._pinned.index(peak)
            self._pinned.pop(idx)
            self._pinned_artists.pop(idx).remove()
        else:
            if self._hover_artist is not None:
                self._hover_artist.remove()
                self._hover_artist = None
            self._pinned.append(peak)
            self._pinned_artists.append(self._annotate(peak, pinned=True))
        self.canvas.draw_idle()

    def _on_release(self, event):
        if event.button == 3:
            self._pan_start = None

    def _on_scroll(self, event):
        if event.inaxes is not self._ax or event.xdata is None:
            return
        factor = 1 / 1.2 if event.button == "up" else 1.2
        if event.key == "shift":
            lo, hi = self._ax.get_ylim()
            c = event.ydata
            self._ax.set_ylim(c - (c - lo) * factor, c + (hi - c) * factor)
        else:
            lo, hi = self._ax.get_xlim()
            c = event.xdata
            self._ax.set_xlim(c - (c - lo) * factor, c + (hi - c) * factor)
        self.canvas.draw_idle()


def _distinct_colors(n):
    """n visually distinct hex colors (tab10, then the light tab20 shades, then evenly spaced hues)."""
    palette = [mcolors.to_hex(c) for c in matplotlib.colormaps["tab10"].colors] + [mcolors.to_hex(c) for c in matplotlib.colormaps["tab20"].colors[1::2]]
    if n <= len(palette):
        return palette[:n]
    return [mcolors.to_hex(mcolors.hsv_to_rgb(((k * 0.618034) % 1.0, 0.65, 0.9))) for k in range(n)]


class _CategoryColorDialog(QtWidgets.QDialog):
    """Table of the unique values of a text column with a color-picker button per value."""

    def __init__(self, parent, column, labels, colors):
        super().__init__(parent)
        self.setWindowTitle(f"Node colors for '{column}'")
        self.resize(420, 480)
        self._colors = dict(colors)
        self._labels = labels
        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(len(labels), 2)
        self.table.setHorizontalHeaderLabels(["Value", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, label in enumerate(labels):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(label))
            btn = QtWidgets.QPushButton()
            btn.clicked.connect(lambda _checked=False, lab=label, b=btn: self._pick(lab, b))
            self._style_button(btn, self._colors[label])
            self.table.setCellWidget(row, 1, btn)
        layout.addWidget(self.table, 1)
        btn_row = QtWidgets.QHBoxLayout()
        reset_btn = QtWidgets.QPushButton("Reset to defaults")
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch(1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        layout.addLayout(btn_row)

    @staticmethod
    def _style_button(btn, color):
        qc = QtGui.QColor(color)
        fg = "black" if 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue() > 128 else "white"
        btn.setText(qc.name())
        btn.setStyleSheet(f"background-color: {qc.name()}; color: {fg}; border: 1px solid gray;")

    def _pick(self, label, btn):
        chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._colors[label]), self, f"Color for '{label}'")
        if chosen.isValid():
            self._colors[label] = chosen.name()
            self._style_button(btn, chosen.name())

    def _reset(self):
        for row, (label, color) in enumerate(zip(self._labels, _distinct_colors(len(self._labels)))):
            self._colors[label] = color
            self._style_button(self.table.cellWidget(row, 1), color)

    def colors(self):
        return dict(self._colors)


GLYPH_W, GLYPH_H = 64.0, 42.0
SPIE_RADIUS = 28.0
NODE_SHAPES = [
    ("Circles", None),
    ("Abundance boxplots per group", "boxplot"),
    ("Abundance spie charts per sample", "spie"),
    ("Abundance pie chart per group (mean)", "pie_mean"),
    ("Abundance pie chart per group (median)", "pie_median"),
]
ROUND_GLYPHS = ("spie", "pie_mean", "pie_median")


class _NodeItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, index, data, radius, label_text):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.index = index
        self.feature = data
        self.radius = radius
        self._circle_radius = radius
        self._glyph = None
        self._glyph_kind = None
        self._highlighted = False
        self.edges = []
        self._base_brush = QtGui.QBrush(QtGui.QColor(FORM_COLORS.get(data["form"], "#1E90FF")))
        self.setBrush(self._base_brush)
        self.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1))
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setZValue(2)

        self.label = QtWidgets.QGraphicsSimpleTextItem(label_text, self)
        font = self.label.font()
        font.setPointSizeF(7)
        self.label.setFont(font)
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2.0, radius + 2)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def set_highlighted(self, highlighted):
        self._highlighted = highlighted
        if highlighted:
            self.setBrush(QtGui.QBrush(QtGui.QColor(HIGHLIGHT_COLOR)))
            self.setPen(QtGui.QPen(QtGui.QColor("#006400"), 3))
            self.setZValue(3)
        else:
            self.setBrush(self._base_brush)
            self.setPen(QtGui.QPen(QtGui.QColor("#333333"), 1))
            self.setZValue(2)

    def set_base_color(self, color):
        self._base_brush = QtGui.QBrush(QtGui.QColor(color))
        if not self._highlighted:
            self.setBrush(self._base_brush)

    def base_color(self):
        return self._base_brush.color().name()

    def set_glyph(self, groups, kind="boxplot"):
        """Draw the node as a small abundance plot (`kind` 'boxplot' or 'spie') of groups =
        [(name, color, per-sample values)], or, if groups is None, as a circle."""
        self.prepareGeometryChange()
        self._glyph = groups
        self._glyph_kind = kind if groups is not None else None
        if groups is None:
            r = self._circle_radius
            self.setRect(-r, -r, 2 * r, 2 * r)
            self.radius = r
        elif kind in ROUND_GLYPHS:
            self.setRect(-SPIE_RADIUS, -SPIE_RADIUS, 2 * SPIE_RADIUS, 2 * SPIE_RADIUS)
            self.radius = SPIE_RADIUS
        else:
            self.setRect(-GLYPH_W / 2.0, -GLYPH_H / 2.0, GLYPH_W, GLYPH_H)
            self.radius = max(GLYPH_W, GLYPH_H) / 2.0
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2.0, self.rect().bottom() + 2)
        self.update()

    def shape(self):
        if self._glyph_kind is None or self._glyph_kind in ROUND_GLYPHS:
            return super().shape()
        path = QtGui.QPainterPath()
        path.addRect(self.rect())
        return path

    def _wedges(self):
        """[(color, value or None)] for the round glyphs: samples (ascending within each group, missing last)
        for 'spie', or the group mean/median for the pie charts."""
        if self._glyph_kind == "spie":
            wedges = []
            for _name, color, values in self._glyph:
                present = sorted(v for v in values if v is not None)
                wedges += [(color, v) for v in present] + [(color, None)] * (len(values) - len(present))
            return wedges
        average = np.mean if self._glyph_kind == "pie_mean" else np.median
        wedges = []
        for _name, color, values in self._glyph:
            present = [v for v in values if v is not None]
            wedges.append((color, float(average(present)) if present else None))
        return wedges

    def _paint_round_glyph(self, painter):
        """Equal-angle wedges starting at 12 o'clock; the wedge area is proportional to the abundance
        (radius ~ sqrt(value / max value of this node)). No outlines or background."""
        wedges = self._wedges()
        positive = [v for _, v in wedges if v is not None and v > 0]
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        if self._highlighted:
            painter.setPen(self.pen())
            painter.drawEllipse(self.rect())
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
        if not wedges or not positive:
            painter.setPen(QtGui.QColor("#555555"))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "no data")
            return
        v_max = max(positive)
        span = 360.0 / len(wedges)
        for k, (color, v) in enumerate(wedges):
            if v is None or v <= 0:
                continue
            r = SPIE_RADIUS * math.sqrt(v / v_max)
            painter.setBrush(QtGui.QColor(color))
            painter.drawPie(QtCore.QRectF(-r, -r, 2 * r, 2 * r), int(round((90.0 - k * span) * 16)), -int(round(span * 16)))

    def paint(self, painter, option, widget=None):
        if self._glyph is None:
            super().paint(painter, option, widget)
            return
        if self._glyph_kind in ROUND_GLYPHS:
            self._paint_round_glyph(painter)
            return
        rect = self.rect()
        background = QtGui.QColor(self.brush().color())
        background.setAlpha(70)
        painter.setPen(self.pen())
        painter.setBrush(QtGui.QColor("white"))
        painter.drawRect(rect)
        painter.setBrush(background)
        painter.drawRect(rect)

        groups = [(name, color, [math.log10(v) for v in values if v and v > 0]) for name, color, values in self._glyph]
        all_values = [v for _, _, vs in groups for v in vs]
        if not groups or not all_values:
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "no data")
            return
        lo, hi = min(all_values), max(all_values)
        if hi - lo < 1e-9:
            lo, hi = lo - 0.5, hi + 0.5
        inner = rect.adjusted(4, 4, -4, -4)

        def ymap(v):
            return inner.bottom() - (v - lo) / (hi - lo) * inner.height()

        slot = inner.width() / len(groups)
        dark_pen = QtGui.QPen(QtGui.QColor("#333333"), 0.8)
        for k, (_name, color, values) in enumerate(groups):
            if not values:
                continue
            cx = inner.left() + (k + 0.5) * slot
            half_w = slot * 0.3
            q1, med, q3 = np.percentile(values, [25, 50, 75])
            painter.setPen(dark_pen)
            painter.drawLine(QtCore.QLineF(cx, ymap(min(values)), cx, ymap(max(values))))
            box_color = QtGui.QColor(color)
            box_color.setAlpha(210)
            painter.setBrush(box_color)
            painter.drawRect(QtCore.QRectF(cx - half_w, ymap(q3), 2 * half_w, max(ymap(q1) - ymap(q3), 1.0)))
            painter.setPen(QtGui.QPen(QtGui.QColor("black"), 1.4))
            painter.drawLine(QtCore.QLineF(cx - half_w, ymap(med), cx + half_w, ymap(med)))

    def full_scene_rect(self):
        return self.mapRectToScene(self.boundingRect() | self.childrenBoundingRect())


class _EdgeItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, node_a, node_b, score, width):
        super().__init__()
        self.node_a = node_a
        self.node_b = node_b
        self.score = score
        self.matched_pairs = None
        self.setPen(QtGui.QPen(QtGui.QColor(110, 110, 110, 200), width))
        self.setZValue(1)
        self.label = QtWidgets.QGraphicsSimpleTextItem(f"{score:.2f}", self)
        font = self.label.font()
        font.setPointSizeF(6)
        self.label.setFont(font)
        self.label.setBrush(QtGui.QBrush(QtGui.QColor("#505050")))
        node_a.edges.append(self)
        node_b.edges.append(self)
        self.update_position()

    def update_position(self):
        p1 = self.node_a.pos()
        p2 = self.node_b.pos()
        self.setLine(QtCore.QLineF(p1, p2))
        br = self.label.boundingRect()
        self.label.setPos((p1.x() + p2.x()) / 2.0 - br.width() / 2.0, (p1.y() + p2.y()) / 2.0 - br.height() / 2.0)

    def shape(self):
        path = QtGui.QPainterPath()
        path.moveTo(self.line().p1())
        path.lineTo(self.line().p2())
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidth(max(6.0, self.pen().widthF() + 4.0))
        return stroker.createStroke(path)


class _NetworkView(QtWidgets.QGraphicsView):
    hoverChanged = Signal(object, QtCore.QPoint)
    nodeClicked = Signal(object)
    edgeClicked = Signal(object)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setMouseTracking(True)
        self._press_pos = None

    def wheelEvent(self, event):
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        self.scale(factor, factor)

    def _item_at(self, pos):
        item = self.itemAt(pos)
        while item is not None and not isinstance(item, (_NodeItem, _EdgeItem)):
            item = item.parentItem()
        return item

    def mousePressEvent(self, event):
        self._press_pos = event.position().toPoint()
        self.hoverChanged.emit(None, QtCore.QPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if event.buttons() == QtCore.Qt.MouseButton.NoButton:
            self.hoverChanged.emit(self._item_at(event.position().toPoint()), event.globalPosition().toPoint())

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._press_pos is not None and (pos - self._press_pos).manhattanLength() < 4:
            item = self._item_at(pos)
            if isinstance(item, _NodeItem):
                self.nodeClicked.emit(item)
            elif isinstance(item, _EdgeItem):
                self.edgeClicked.emit(item)
        self._press_pos = None

    def leaveEvent(self, event):
        self.hoverChanged.emit(None, QtCore.QPoint())
        super().leaveEvent(event)


class _SpectrumPopup(QtWidgets.QFrame):
    """Floating tooltip-like window showing an MS/MS spectrum or a mirror plot."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("QFrame { background: white; border: 1px solid #888888; }")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        self.fig = Figure((7.0, 4.6), dpi=80, facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.resize(580, 380)

    def show_at(self, global_pos):
        x = global_pos.x() + 18
        y = global_pos.y() + 18
        screen = QtGui.QGuiApplication.screenAt(global_pos)
        if screen is not None:
            geo = screen.availableGeometry()
            if x + self.width() > geo.right():
                x = global_pos.x() - self.width() - 18
            if y + self.height() > geo.bottom():
                y = global_pos.y() - self.height() - 18
        self.move(x, y)
        self.show()


class FBMNWidget(QtWidgets.QWidget):
    """Interactive feature-based molecular network built from the most abundant MS/MS
    spectrum of each feature pair."""

    featureClicked = Signal(str)
    saveClustersRequested = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._get_nodes = None
        self._algorithms = {}
        self._nodes = []
        self._node_items = []
        self._edge_items = []
        self._graph = None
        self._components = []
        self._cluster_of = {}
        self._hover_item = None
        self._specs = []
        self._gen_params = {}
        self._mirror_dialogs = []
        self._get_table = None
        self._get_group_abundances = None
        self._get_saved_network = None
        self._get_node_info = None
        self._category_colors = {}
        self._popup = _SpectrumPopup(self)
        self._setup_ui()

    def configure(self, get_nodes, algorithms, get_table=None, get_group_abundances=None, get_saved_network=None, get_node_info=None):
        """get_nodes(forms, selection) -> list of node dicts (or None if unavailable);
        get_table() -> loaded results sheet (polars DataFrame) or None;
        get_group_abundances([(num, form), ...]) -> {(num, form): [(group, color, values), ...]};
        get_saved_network() -> rows (dicts) of the saved network sheet or None;
        get_node_info([(num, form), ...]) -> {(num, form): node dict without spectrum} from the results sheet."""
        self._get_nodes = get_nodes
        self._get_table = get_table
        self._get_group_abundances = get_group_abundances
        self._get_saved_network = get_saved_network
        self._get_node_info = get_node_info
        self._algorithms = dict(algorithms)
        self.algorithm_combo.clear()
        self.algorithm_combo.addItems(list(self._algorithms.keys()))
        idx = self.algorithm_combo.findText("ModifiedCosineHungarian")
        self.algorithm_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        main_layout.addWidget(splitter)

        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        params_box = QtWidgets.QGroupBox("Parameters")
        form = QtWidgets.QFormLayout(params_box)

        tol_row = QtWidgets.QHBoxLayout()
        self.tolerance_spin = QtWidgets.QDoubleSpinBox()
        self.tolerance_spin.setDecimals(4)
        self.tolerance_spin.setRange(0.0001, 1000.0)
        self.tolerance_spin.setSingleStep(0.005)
        self.tolerance_spin.setValue(0.01)
        self.tolerance_unit_combo = QtWidgets.QComboBox()
        self.tolerance_unit_combo.addItems(["m/z (Da)", "ppm"])
        self.tolerance_unit_combo.currentIndexChanged.connect(lambda i: self.tolerance_spin.setValue(10.0 if i == 1 else 0.01))
        tol_row.addWidget(self.tolerance_spin, 1)
        tol_row.addWidget(self.tolerance_unit_combo)
        tol_widget = QtWidgets.QWidget()
        tol_widget.setLayout(tol_row)
        tol_row.setContentsMargins(0, 0, 0, 0)
        tol_widget.setToolTip("Fragment bin width / matching tolerance passed to matchms. ppm values are converted to Da at the higher precursor m/z of the two compared spectra.")
        form.addRow("Fragment bin width:", tol_widget)

        self.algorithm_combo = QtWidgets.QComboBox()
        form.addRow("Similarity algorithm:", self.algorithm_combo)

        self.form_combo = QtWidgets.QComboBox()
        for label, _ in FORM_OPTIONS:
            self.form_combo.addItem(label)
        form.addRow("Spectra:", self.form_combo)

        self.selection_combo = QtWidgets.QComboBox()
        for label, _ in SELECTION_OPTIONS:
            self.selection_combo.addItem(label)
        form.addRow("Spectrum per feature pair:", self.selection_combo)

        frag_row = QtWidgets.QHBoxLayout()
        frag_row.setContentsMargins(0, 0, 0, 0)
        self.min_fragments_spin = QtWidgets.QSpinBox()
        self.min_fragments_spin.setRange(1, 1000)
        self.min_fragments_spin.setValue(4)
        self.min_fragments_pct_spin = QtWidgets.QDoubleSpinBox()
        self.min_fragments_pct_spin.setRange(0.0, 100.0)
        self.min_fragments_pct_spin.setDecimals(1)
        self.min_fragments_pct_spin.setValue(5.0)
        self.min_fragments_pct_spin.setSuffix(" %")
        frag_row.addWidget(self.min_fragments_spin)
        frag_row.addWidget(QtWidgets.QLabel("above"))
        frag_row.addWidget(self.min_fragments_pct_spin)
        frag_widget = QtWidgets.QWidget()
        frag_widget.setLayout(frag_row)
        frag_widget.setToolTip("A spectrum is only used if it has at least this many fragments with an intensity above the given percentage of its most abundant fragment.")
        form.addRow("Min. fragments:", frag_widget)

        self.min_score_spin = QtWidgets.QDoubleSpinBox()
        self.min_score_spin.setRange(0.0, 1.0)
        self.min_score_spin.setDecimals(3)
        self.min_score_spin.setSingleStep(0.05)
        self.min_score_spin.setValue(0.7)
        form.addRow("Min. edge score:", self.min_score_spin)

        left_layout.addWidget(params_box)

        self.generate_btn = QtWidgets.QPushButton("Generate network")
        self.generate_btn.clicked.connect(self._generate)
        left_layout.addWidget(self.generate_btn)

        self.layout_btn = QtWidgets.QPushButton("Force-directed layout")
        self.layout_btn.clicked.connect(lambda: self._apply_layout(force=True))
        left_layout.addWidget(self.layout_btn)

        self.fit_btn = QtWidgets.QPushButton("Fit view")
        self.fit_btn.clicked.connect(self._fit_view)
        left_layout.addWidget(self.fit_btn)

        self.show_scores_check = QtWidgets.QCheckBox("Show edge scores")
        self.show_scores_check.toggled.connect(self._update_score_visibility)
        left_layout.addWidget(self.show_scores_check)

        self.sqrt_check = QtWidgets.QCheckBox("Square-root intensity axis in spectrum plots")
        self.sqrt_check.toggled.connect(lambda _checked: setattr(self, "_hover_item", None))
        left_layout.addWidget(self.sqrt_check)

        display_box = QtWidgets.QGroupBox("Node display")
        display_layout = QtWidgets.QVBoxLayout(display_box)
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("Color by:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItem("Form (native/labeled)", None)
        self.color_combo.currentIndexChanged.connect(lambda _i: self._apply_node_colors(ask=True))
        color_row.addWidget(self.color_combo, 1)
        self.edit_colors_btn = QtWidgets.QPushButton("Colors...")
        self.edit_colors_btn.setToolTip("Edit the colors of the values of a text column.")
        self.edit_colors_btn.clicked.connect(lambda: self._apply_node_colors(ask=True))
        color_row.addWidget(self.edit_colors_btn)
        display_layout.addLayout(color_row)
        self.color_legend = QtWidgets.QLabel("")
        self.color_legend.setWordWrap(True)
        self.color_legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        display_layout.addWidget(self.color_legend)
        shape_row = QtWidgets.QHBoxLayout()
        shape_row.addWidget(QtWidgets.QLabel("Node shape:"))
        self.node_shape_combo = QtWidgets.QComboBox()
        for label, kind in NODE_SHAPES:
            self.node_shape_combo.addItem(label, kind)
        self.node_shape_combo.setToolTip(
            "Boxplots: log10 peak areas per experimental group.\n"
            "Spie charts: one equal-angle wedge per sample, samples grouped (ascending within a group) and drawn in their group color.\n"
            "Pie charts: one equal-angle wedge per group showing the group's mean or median peak area.\n"
            "Wedge area is proportional to the peak area (relative to the largest wedge of the feature).\n"
            "Native nodes use the _Area_N, labeled nodes the _Area_L columns."
        )
        self.node_shape_combo.currentIndexChanged.connect(lambda _i: self._apply_node_glyphs())
        shape_row.addWidget(self.node_shape_combo, 1)
        display_layout.addLayout(shape_row)
        left_layout.addWidget(display_box)

        self.save_btn = QtWidgets.QPushButton("Save network to results file")
        self.save_btn.setToolTip(f"Writes the cluster id of each feature pair into the column 'FBMN_cluster' of the loaded results sheet and the network (generation parameters, nodes with their positions and edges with their scores; one JSON per cluster) into the sheet '{NETWORK_SHEET}'.")
        self.save_btn.clicked.connect(self._request_save)
        left_layout.addWidget(self.save_btn)

        self.restore_btn = QtWidgets.QPushButton("Restore network from results file")
        self.restore_btn.setToolTip(f"Restores the network and its parameters saved in the sheet '{NETWORK_SHEET}' of the results file. The MS/MS spectra are taken from the loaded raw data.")
        self.restore_btn.clicked.connect(self._restore_network)
        left_layout.addWidget(self.restore_btn)

        self.export_btn = QtWidgets.QPushButton("Export network for Cytoscape...")
        self.export_btn.setToolTip("Writes the network (nodes identified by Num, with all node/edge attributes and the current layout and colors) to a single Cytoscape JSON (.cyjs) or GraphML file.")
        self.export_btn.clicked.connect(self._export_cytoscape)
        left_layout.addWidget(self.export_btn)

        self.status_label = QtWidgets.QLabel("Click 'Generate network' to build the molecular network.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.status_label)

        legend = QtWidgets.QLabel(
            f"<span style='color:{FORM_COLORS['native']}'>&#9679;</span> native (M) &nbsp; "
            f"<span style='color:{FORM_COLORS['labeled']}'>&#9679;</span> labeled (M\u2032) &nbsp; "
            f"<span style='color:{HIGHLIGHT_COLOR}'>&#9679;</span> selected<br>"
            "Node size: log10 abundance in the most abundant sample.<br>"
            "Mouse wheel: zoom; drag background: pan; drag node: move; click node: show in Experiment results."
        )
        legend.setWordWrap(True)
        left_layout.addWidget(legend)
        left_layout.addStretch(1)

        self.scene = QtWidgets.QGraphicsScene(self)
        self.view = _NetworkView(self.scene)
        self.view.hoverChanged.connect(self._on_hover)
        self.view.nodeClicked.connect(lambda item: self.featureClicked.emit(str(item.feature["num"])))
        self.view.edgeClicked.connect(self._open_mirror_dialog)

        self._params_panel = left
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        top_row = QtWidgets.QHBoxLayout()
        self.toggle_params_btn = QtWidgets.QPushButton("\u25c0 Hide parameters")
        self.toggle_params_btn.clicked.connect(self._toggle_params_panel)
        top_row.addWidget(self.toggle_params_btn)
        top_row.addStretch(1)
        right_layout.addLayout(top_row)
        right_layout.addWidget(self.view, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        self._set_network_buttons_enabled(False)

    def _toggle_params_panel(self):
        visible = self._params_panel.isHidden()
        self._params_panel.setVisible(visible)
        self.toggle_params_btn.setText("\u25c0 Hide parameters" if visible else "\u25b6 Show parameters")

    def _set_network_buttons_enabled(self, enabled):
        self.layout_btn.setEnabled(enabled)
        self.fit_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)
        self.export_btn.setEnabled(enabled)

    def clear(self):
        self._popup.hide()
        self._hover_item = None
        self.scene.clear()
        self._nodes = []
        self._specs = []
        self._node_items = []
        self._edge_items = []
        self._graph = None
        self._components = []
        self._cluster_of = {}
        self._set_network_buttons_enabled(False)

    # -- network generation --

    def _generate(self):
        if not MATCHMS_AVAILABLE:
            QtWidgets.QMessageBox.warning(self, "FBMN", "matchms is not available in this environment.")
            return
        if self._get_nodes is None:
            return
        forms = FORM_OPTIONS[self.form_combo.currentIndex()][1]
        selection = SELECTION_OPTIONS[self.selection_combo.currentIndex()][1]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            nodes = self._get_nodes(forms, selection)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if nodes is None:
            return

        self.clear()
        min_fragments = self.min_fragments_spin.value()
        min_pct = self.min_fragments_pct_spin.value()
        kept = []
        for n in nodes:
            if self._select_spectrum(n, min_fragments, min_pct):
                kept.append(n)
        nodes = kept
        if not nodes:
            self.status_label.setText("No feature pairs with a usable MS/MS spectrum (check the MS/MS filter options and the min. fragments setting).")
            return

        for n in nodes:
            n["precursor_mz"] = float(n["scan"].precursor_mz)
        self._gen_params = self._current_gen_params()
        specs = [self._to_spectrum(n["scan"]) for n in nodes]
        edges = self._compute_edges(nodes, specs)
        if edges is None:
            self._nodes = []
            self.status_label.setText("Network generation cancelled.")
            return

        self._specs = specs
        self._build_network(nodes, edges)

    def _current_gen_params(self):
        return {
            "tolerance": self.tolerance_spin.value(),
            "use_ppm": self.tolerance_unit_combo.currentIndex() == 1,
            "algorithm": self.algorithm_combo.currentText(),
            "forms": list(FORM_OPTIONS[self.form_combo.currentIndex()][1]),
            "selection": SELECTION_OPTIONS[self.selection_combo.currentIndex()][1],
            "min_fragments": self.min_fragments_spin.value(),
            "min_fragments_pct": self.min_fragments_pct_spin.value(),
            "min_score": self.min_score_spin.value(),
        }

    def _refresh_color_columns(self):
        table = self._get_table() if self._get_table else None
        current = self.color_combo.currentData()
        self.color_combo.blockSignals(True)
        self.color_combo.clear()
        self.color_combo.addItem("Form (native/labeled)", None)
        if table is not None:
            for column in table.columns:
                self.color_combo.addItem(column, column)
        idx = self.color_combo.findData(current)
        self.color_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.color_combo.blockSignals(False)

    @staticmethod
    def _has_enough_fragments(scan, min_fragments, min_pct):
        intens = np.asarray(scan.intensity_list, dtype=float)
        if intens.size == 0 or intens.max() <= 0:
            return False
        return int(np.count_nonzero(intens >= intens.max() * min_pct / 100.0)) >= min_fragments

    @classmethod
    def _select_spectrum(cls, node, min_fragments, min_pct):
        """Set the node's spectrum to its most abundant one passing the fragment filter; False if none passes."""
        entry = next((e for e in node.get("spectra") or [] if cls._has_enough_fragments(e["scan"], min_fragments, min_pct)), None)
        if entry is None:
            return False
        node["scan"] = entry["scan"]
        node["scan_info"] = entry
        return True

    @staticmethod
    def _to_spectrum(scan):
        mz = np.asarray(scan.mz_list, dtype=float)
        intens = np.asarray(scan.intensity_list, dtype=float)
        order = np.argsort(mz)
        spec = MatchmsSpectrum(mz=mz[order], intensities=intens[order], metadata={"precursor_mz": float(scan.precursor_mz)})
        return matchms_normalize_intensities(spec)

    def _make_algorithm(self, tolerance):
        cls = self._algorithms.get(self._gen_params["algorithm"])
        kwargs = {"tolerance": tolerance}
        # score on the entire spectra, also for NeutralLossesCosine
        if "ignore_peaks_above_precursor" in inspect.signature(cls.__init__).parameters:
            kwargs["ignore_peaks_above_precursor"] = False
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()

    def _tolerance_for(self, ia, ib):
        tolerance = self._gen_params["tolerance"]
        if not self._gen_params["use_ppm"]:
            return tolerance
        return round(tolerance * 1e-6 * max(self._nodes[ia]["precursor_mz"], self._nodes[ib]["precursor_mz"]), 6)

    @staticmethod
    def _pair_score(algorithm, spec_a, spec_b):
        try:
            result = algorithm.pair(spec_a, spec_b)
        except Exception:
            return 0.0
        try:
            return float(result["score"])
        except (TypeError, IndexError, ValueError, KeyError):
            return float(result)

    def _score_matrix(self, algorithm, specs):
        try:
            try:
                result = algorithm.matrix(specs, specs, is_symmetric=True, progress_bar=False)
            except TypeError:
                result = algorithm.matrix(specs, specs, is_symmetric=True)
            return np.asarray(result["score"] if result.dtype.names else result, dtype=float)
        except Exception:
            n = len(specs)
            scores = np.zeros((n, n))
            for a in range(n):
                for b in range(a + 1, n):
                    scores[a, b] = scores[b, a] = self._pair_score(algorithm, specs[a], specs[b])
            return scores

    def _compute_edges(self, nodes, specs):
        """Return a list of (i, j, score) for node pairs above the score threshold, or None if cancelled.
        Only spectra of the same form (native/labeled) and polarity are compared."""
        threshold = self.min_score_spin.value()
        tolerance = self._gen_params["tolerance"]
        use_ppm = self._gen_params["use_ppm"]
        self._nodes = nodes

        groups = defaultdict(list)
        for i, n in enumerate(nodes):
            groups[(n["form"], n.get("polarity"))].append(i)
        total = sum(len(g) * (len(g) - 1) // 2 for g in groups.values())

        progress = QtWidgets.QProgressDialog("Calculating spectral similarities...", "Cancel", 0, max(total, 1), self)
        progress.setWindowTitle("FBMN")
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()

        edges = []
        done = 0
        try:
            for idxs in groups.values():
                if len(idxs) < 2:
                    continue
                if not use_ppm:
                    scores = self._score_matrix(self._make_algorithm(tolerance), [specs[i] for i in idxs])
                    for a in range(len(idxs)):
                        for b in range(a + 1, len(idxs)):
                            if scores[a, b] >= threshold:
                                edges.append((idxs[a], idxs[b], float(scores[a, b])))
                    done += len(idxs) * (len(idxs) - 1) // 2
                    progress.setValue(done)
                    QtWidgets.QApplication.processEvents()
                    if progress.wasCanceled():
                        return None
                else:
                    algorithm_cache = {}
                    for a in range(len(idxs)):
                        for b in range(a + 1, len(idxs)):
                            ia, ib = idxs[a], idxs[b]
                            tol_da = self._tolerance_for(ia, ib)
                            if tol_da not in algorithm_cache:
                                algorithm_cache[tol_da] = self._make_algorithm(tol_da)
                            score = self._pair_score(algorithm_cache[tol_da], specs[ia], specs[ib])
                            if score >= threshold:
                                edges.append((ia, ib, score))
                            done += 1
                            if done % 200 == 0:
                                progress.setValue(done)
                                QtWidgets.QApplication.processEvents()
                                if progress.wasCanceled():
                                    return None
        finally:
            progress.close()
        return edges

    def _build_network(self, nodes, edges, positions=None):
        self._nodes = nodes
        graph = nx.Graph()
        graph.add_nodes_from(range(len(nodes)))
        for i, j, score in edges:
            graph.add_edge(i, j, score=score)
        self._graph = graph
        self._components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: (-len(c), c[0]))
        self._cluster_of = {i: ci for ci, comp in enumerate(self._components) for i in comp}

        log_abund = [math.log10(n["abundance"]) if n.get("abundance") and n["abundance"] > 1.0 else None for n in nodes]
        valid = [v for v in log_abund if v is not None]
        lo, hi = (min(valid), max(valid)) if valid else (0.0, 1.0)
        span = max(hi - lo, 1e-9)

        for i, n in enumerate(nodes):
            radius = 8.0 if log_abund[i] is None else 8.0 + 22.0 * (log_abund[i] - lo) / span
            label = f"FP {n['num']}\nMet {n.get('ogroup') or '-'}\nIon mode {n.get('polarity') or '-'}\nmz {n['mz']:.4f}\nRt {n['rt']:.2f}min \nXn {n.get('xn', '')}"
            if len(self._gen_params["forms"]) > 1:
                label += f" ({_form_label(n['form'])})"
            item = _NodeItem(i, n, radius, label)
            self.scene.addItem(item)
            self._node_items.append(item)

        threshold = self._gen_params["min_score"]
        for i, j, score in edges:
            width = 1.0 + 4.0 * (score - threshold) / max(1.0 - threshold, 1e-9)
            edge = _EdgeItem(self._node_items[i], self._node_items[j], score, width)
            self.scene.addItem(edge)
            self._edge_items.append(edge)

        if positions is None:
            self._apply_layout(force=False)
        else:
            for item, (x, y) in zip(self._node_items, positions):
                item.setPos(x, y)
            self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-100, -100, 100, 100))
        self._update_score_visibility(self.show_scores_check.isChecked())
        self._set_network_buttons_enabled(True)
        self._refresh_color_columns()
        self._apply_node_colors(ask=False)
        self._apply_node_glyphs()
        n_singletons = sum(1 for c in self._components if len(c) == 1)
        self.status_label.setText(f"{len(nodes)} nodes, {len(edges)} edges, {len(self._components)} clusters ({n_singletons} singletons).")
        self._fit_view()

    def _apply_layout(self, force):
        """Lay out each cluster (circular, or force-directed if `force`) and pack the clusters
        into rows, largest first."""
        if self._graph is None:
            return
        placed = []
        for comp in self._components:
            n = len(comp)
            if n == 1:
                local = {comp[0]: (0.0, 0.0)}
                extent = 0.0
            else:
                sub = self._graph.subgraph(comp)
                pos = nx.spring_layout(sub, weight="score", iterations=100) if force else nx.circular_layout(sub)
                local = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
                extent = 70.0 * math.sqrt(n)
            half = extent + max(self._node_items[i].radius for i in comp) + 70.0
            placed.append((comp, local, extent, half))

        row_width = max(math.sqrt(sum((2 * p[3]) ** 2 for p in placed)) * 1.3, max(2 * p[3] for p in placed))
        x = y = row_height = 0.0
        for comp, local, extent, half in placed:
            size = 2 * half
            if x > 0 and x + size > row_width:
                x = 0.0
                y += row_height
                row_height = 0.0
            for i in comp:
                lx, ly = local[i]
                self._node_items[i].setPos(x + half + lx * extent, y + half + ly * extent)
            x += size
            row_height = max(row_height, size)
        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-100, -100, 100, 100))

    def _fit_view(self):
        if self._node_items:
            self.view.fitInView(self.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def _update_score_visibility(self, visible):
        for edge in self._edge_items:
            edge.label.setVisible(visible)

    # -- node display --

    def _apply_node_colors(self, ask=False):
        """Color the nodes by the selected results-sheet column: numeric -> viridis, text -> user-defined
        color per unique value (dialog shown if `ask`), or by form if no column is selected."""
        column = self.color_combo.currentData()
        self.edit_colors_btn.setEnabled(False)
        if not self._node_items:
            self.color_legend.setText("")
            return
        if column is None:
            for item in self._node_items:
                item.set_base_color(FORM_COLORS.get(item.feature["form"], "#1E90FF"))
            self.color_legend.setText("")
            return
        table = self._get_table() if self._get_table else None
        if table is None or column not in table.columns or "Num" not in table.columns:
            return
        values_by_num = dict(zip(table["Num"].to_list(), table[column].to_list()))
        values = [values_by_num.get(item.feature["num"]) for item in self._node_items]
        missing_color = "#BBBBBB"

        if table.schema[column].is_numeric():
            numeric = [float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None for v in values]
            valid = [v for v in numeric if v is not None]
            if not valid:
                for item in self._node_items:
                    item.set_base_color(missing_color)
                self.color_legend.setText(f"<b>{column}</b>: no values")
                return
            lo, hi = min(valid), max(valid)
            span = hi - lo if hi > lo else 1.0
            cmap = matplotlib.colormaps["viridis"]
            for item, v in zip(self._node_items, numeric):
                item.set_base_color(missing_color if v is None else mcolors.to_hex(cmap((v - lo) / span)))
            stops = "".join(f"<span style='color:{mcolors.to_hex(cmap(t))}'>&#9632;</span>" for t in np.linspace(0, 1, 12))
            self.color_legend.setText(f"<b>{column}</b>: {lo:.4g} {stops} {hi:.4g}<br><span style='color:{missing_color}'>&#9679;</span> missing")
            return

        self.edit_colors_btn.setEnabled(True)
        labels = sorted({str(v) for v in values if v is not None and str(v) != ""})
        mapping = dict(self._category_colors.get(column, {}))
        new_labels = [label for label in labels if label not in mapping]
        if new_labels:
            unused = [c for c in _distinct_colors(len(labels) + len(mapping)) if c not in mapping.values()]
            for label, color in zip(new_labels, unused):
                mapping[label] = color
        if ask and labels:
            dlg = _CategoryColorDialog(self, column, labels, {label: mapping[label] for label in labels})
            if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                mapping.update(dlg.colors())
        self._category_colors[column] = mapping
        for item, v in zip(self._node_items, values):
            item.set_base_color(mapping.get(str(v), missing_color) if v is not None and str(v) != "" else missing_color)
        shown = labels[:30]
        entries = " &nbsp; ".join(f"<span style='color:{mapping[label]}'>&#9679;</span> {label}" for label in shown)
        more = f" &nbsp; (+{len(labels) - len(shown)} more)" if len(labels) > len(shown) else ""
        self.color_legend.setText(f"<b>{column}</b>: {entries}{more} &nbsp; <span style='color:{missing_color}'>&#9679;</span> missing")

    def _apply_node_glyphs(self):
        if not self._node_items:
            return
        kind = self.node_shape_combo.currentData()
        if kind is None or self._get_group_abundances is None:
            for item in self._node_items:
                item.set_glyph(None)
            return
        missing = [(item.feature["num"], item.feature["form"]) for item in self._node_items if "group_abundances" not in item.feature]
        if missing:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                fetched = self._get_group_abundances(missing) or {}
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            for item in self._node_items:
                key = (item.feature["num"], item.feature["form"])
                if key in fetched:
                    item.feature["group_abundances"] = fetched[key]
        for item in self._node_items:
            item.set_glyph(item.feature.get("group_abundances", []), kind)

    # -- export --

    def _export_cytoscape(self):
        """Write the network to a single Cytoscape JSON (.cyjs) or GraphML file; nodes are identified by Num."""
        if not self._node_items:
            return
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(self, "Export network for Cytoscape", "fbmn_network.cyjs", "Cytoscape JSON (*.cyjs);;GraphML (*.graphml)")
        if not path:
            return
        use_graphml = path.lower().endswith(".graphml") or ("GraphML" in selected_filter and not path.lower().endswith(".cyjs"))

        num_counts = Counter(n["num"] for n in self._nodes)
        color_column = self.color_combo.currentData()
        table = self._get_table() if (self._get_table and color_column) else None
        color_values = dict(zip(table["Num"].to_list(), table[color_column].to_list())) if table is not None and color_column in table.columns else {}

        node_ids = []
        node_records = []
        for i, (n, item) in enumerate(zip(self._nodes, self._node_items)):
            node_id = _node_id(n, num_counts)
            node_ids.append(node_id)
            attrs = {
                "id": node_id,
                "name": node_id,
                "Num": _plain(n["num"]),
                "OGroup": n.get("ogroup"),
                "form": n["form"],
                "polarity": n.get("polarity"),
                "mz": _plain(n["mz"]),
                "precursor_mz": _plain(n.get("precursor_mz")),
                "RT": _plain(n["rt"]),
                "Xn": _plain(n.get("xn")),
                "abundance": _plain(n.get("abundance")),
                "n_fragments": int(len(self._specs[i].peaks.mz)) if self._specs[i] is not None else None,
                "FBMN_cluster": f"FBMN_cluster_{self._cluster_of[i] + 1}",
                "color": item.base_color(),
            }
            if color_column and color_column not in attrs:
                attrs[color_column] = _plain(color_values.get(n["num"]))
            node_records.append((attrs, item.pos()))

        edge_records = []
        for k, edge in enumerate(self._edge_items):
            edge_records.append(
                {
                    "id": f"e{k}",
                    "source": node_ids[edge.node_a.index],
                    "target": node_ids[edge.node_b.index],
                    "interaction": self._gen_params.get("algorithm", "similarity"),
                    "score": float(edge.score),
                    "matched_fragments": len(self._ensure_matched_pairs(edge)) if self._edge_has_spectra(edge) else None,
                }
            )

        try:
            if use_graphml:
                graph = nx.Graph()
                for attrs, pos in node_records:
                    graph.add_node(attrs["id"], **{k: ("" if v is None else v) for k, v in attrs.items() if k != "id"}, x=pos.x(), y=pos.y())
                for e in edge_records:
                    graph.add_edge(e["source"], e["target"], **{k: v for k, v in e.items() if k not in ("source", "target")})
                nx.write_graphml(graph, path)
            else:
                cyjs = {
                    "format_version": "1.0",
                    "generated_by": "MetExtract II",
                    "target_cytoscapejs_version": "~2.1",
                    "data": {"name": "MetExtract II FBMN", "algorithm": self._gen_params.get("algorithm"), "min_score": self._gen_params.get("min_score")},
                    "elements": {
                        "nodes": [{"data": attrs, "position": {"x": pos.x(), "y": pos.y()}} for attrs, pos in node_records],
                        "edges": [{"data": e} for e in edge_records],
                    },
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cyjs, f, indent=1, default=str)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "FBMN", f"Could not export the network: {ex}")
            return
        QtWidgets.QMessageBox.information(self, "FBMN", f"Exported {len(node_records)} nodes and {len(edge_records)} edges to\n{path}\n\nIn Cytoscape use File > Import > Network from File; node attribute 'color' holds the current node colors.")

    # -- interaction --

    def _on_hover(self, item, global_pos):
        if item is None:
            self._hover_item = None
            self._popup.hide()
            return
        if (isinstance(item, _NodeItem) and self._specs[item.index] is None) or (isinstance(item, _EdgeItem) and not self._edge_has_spectra(item)):
            self._hover_item = item
            self._popup.hide()
            return
        if item is not self._hover_item:
            self._hover_item = item
            if isinstance(item, _NodeItem):
                _draw_spectrum(self._popup.fig, item.feature, self._specs[item.index], sqrt_axis=self.sqrt_check.isChecked(), info=item.feature.get("scan_info"))
            else:
                self._draw_edge_mirror(self._popup.fig, item, sqrt_axis=self.sqrt_check.isChecked())
            self._popup.canvas.draw()
        self._popup.show_at(global_pos)

    def _edge_has_spectra(self, edge):
        return self._specs[edge.node_a.index] is not None and self._specs[edge.node_b.index] is not None

    def _ensure_matched_pairs(self, edge):
        if edge.matched_pairs is None:
            ia, ib = edge.node_a.index, edge.node_b.index
            edge.matched_pairs = _matched_peak_pairs(self._make_algorithm(self._tolerance_for(ia, ib)), self._specs[ia], self._specs[ib])
        return edge.matched_pairs

    def _draw_edge_mirror(self, fig, edge, fontsize=8, sqrt_axis=False):
        ia, ib = edge.node_a.index, edge.node_b.index
        return _draw_mirror(
            fig,
            edge.node_a.feature,
            self._specs[ia],
            edge.node_b.feature,
            self._specs[ib],
            edge.score,
            self._ensure_matched_pairs(edge),
            fontsize=fontsize,
            sqrt_axis=sqrt_axis,
            info_a=edge.node_a.feature.get("scan_info"),
            info_b=edge.node_b.feature.get("scan_info"),
        )

    def _open_mirror_dialog(self, edge):
        """Show the mirror plot of an edge in an interactive, non-modal window; the compared spectra
        can be chosen from all MS/MS spectra of both features."""
        self._popup.hide()
        if not self._edge_has_spectra(edge):
            self.status_label.setText("The MS/MS spectra of this edge are not available in the loaded raw data.")
            return
        ia, ib = edge.node_a.index, edge.node_b.index
        # capture the data so the window keeps working after the network is regenerated
        algorithm = self._make_algorithm(self._tolerance_for(ia, ib))
        default_pairs = self._ensure_matched_pairs(edge)
        sides = []
        for data, spec in ((edge.node_a.feature, self._specs[ia]), (edge.node_b.feature, self._specs[ib])):
            entries = sorted(data.get("spectra") or [dict(data.get("scan_info") or {}, scan=data.get("scan"))], key=lambda e: -(e.get("prec_intensity") or 0.0))
            default = next((k for k, e in enumerate(entries) if e.get("scan") is data.get("scan")), 0)
            sides.append({"data": data, "entries": entries, "default": default, "specs": {default: spec}})
        selected = [sides[0]["default"], sides[1]["default"]]
        results = {tuple(selected): (edge.score, default_pairs)}

        def spectrum(side, k):
            if k not in side["specs"]:
                side["specs"][k] = self._to_spectrum(side["entries"][k]["scan"])
            return side["specs"][k]

        def draw(fig, sqrt_axis):
            spec_a, spec_b = spectrum(sides[0], selected[0]), spectrum(sides[1], selected[1])
            key = tuple(selected)
            if key not in results:
                results[key] = (self._pair_score(algorithm, spec_a, spec_b), _matched_peak_pairs(algorithm, spec_a, spec_b))
            score, pairs = results[key]
            return _draw_mirror(
                fig,
                sides[0]["data"],
                spec_a,
                sides[1]["data"],
                spec_b,
                score,
                pairs,
                fontsize=10,
                sqrt_axis=sqrt_axis,
                info_a=sides[0]["entries"][selected[0]],
                info_b=sides[1]["entries"][selected[1]],
            )

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QFormLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        combos = []
        for label, side in (("A", sides[0]), ("B", sides[1])):
            combo = QtWidgets.QComboBox()
            combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
            for k, entry in enumerate(side["entries"]):
                sample, rt, pi = _scan_info_parts(entry)
                combo.addItem(f"prec. intensity {pi} | RT {rt} | {sample}{'  [network]' if k == side['default'] else ''}")
            combo.setCurrentIndex(side["default"])
            combo.setToolTip("MS/MS spectra of the feature, sorted by precursor intensity; [network] marks the spectrum used for the network.")
            controls_layout.addRow(f"Spectrum {label} (FP {side['data']['num']}, {len(side['entries'])} spectra):", combo)
            combos.append(combo)

        dlg = _MirrorDialog(
            self,
            draw,
            f"FBMN mirror plot: FP {sides[0]['data']['num']} vs. FP {sides[1]['data']['num']}",
            self.sqrt_check.isChecked(),
            controls=controls,
        )

        def on_selected(side_idx, k):
            if k < 0:
                return
            selected[side_idx] = k
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                dlg.refresh()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        for side_idx, combo in enumerate(combos):
            combo.currentIndexChanged.connect(lambda k, s=side_idx: on_selected(s, k))
        self._mirror_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _obj=None, d=dlg: self._mirror_dialogs.remove(d) if d in self._mirror_dialogs else None)
        dlg.show()

    def highlight_features(self, feature_ids):
        """Color the nodes of the given feature pairs (Num) green and zoom to the cluster of the first one."""
        if not self._node_items:
            return
        keys = {str(f) for f in feature_ids}
        targets = []
        for item in self._node_items:
            on = str(item.feature["num"]) in keys
            item.set_highlighted(on)
            if on:
                targets.append(item.index)
        if targets:
            rect = QtCore.QRectF()
            for i in self._components[self._cluster_of[targets[0]]]:
                rect = rect.united(self._node_items[i].full_scene_rect())
            self.view.fitInView(rect.adjusted(-60, -60, 60, 60), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def cluster_assignments(self):
        """Return {Num: 'FBMN_cluster_<n>'} (multiple clusters joined by ';' if a feature has several nodes)."""
        clusters = defaultdict(set)
        for i, n in enumerate(self._nodes):
            clusters[n["num"]].add(self._cluster_of[i] + 1)
        return {num: ";".join(f"FBMN_cluster_{c}" for c in sorted(cs)) for num, cs in clusters.items()}

    def _request_save(self):
        if self._nodes:
            self.saveClustersRequested.emit(self.cluster_assignments())

    # -- network sheet (save / restore) --

    def network_rows(self, extra_params=None):
        """Return the rows of the network sheet: first the generation parameters, then one row per cluster
        with its nodes (Num and position) and edges (Num pairs and scores) as JSON. Feature metadata and
        MS/MS spectra are not included (they are taken from the results sheet / raw data on restore)."""
        edges_by_cluster = defaultdict(list)
        for edge in self._edge_items:
            ia, ib = edge.node_a.index, edge.node_b.index
            edges_by_cluster[self._cluster_of[ia]].append({"source": _plain(self._nodes[ia]["num"]), "target": _plain(self._nodes[ib]["num"]), "score": round(float(edge.score), 6)})

        gp = self._gen_params
        params = {
            "generated_by": "MetExtract II",
            "saved": datetime.now().isoformat(timespec="seconds"),
            **(extra_params or {}),
            "fragment_bin_width": gp["tolerance"],
            "fragment_bin_width_unit": "ppm" if gp["use_ppm"] else "Da",
            "similarity_algorithm": gp["algorithm"],
            "forms": gp["forms"],
            "spectrum_selection": gp["selection"],
            "min_fragments": gp["min_fragments"],
            "min_fragments_above_pct": gp["min_fragments_pct"],
            "min_edge_score": gp["min_score"],
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edge_items),
            "n_clusters": len(self._components),
        }
        records = [("parameters", len(self._nodes), len(self._edge_items), params)]
        for ci, comp in enumerate(self._components):
            nodes = []
            for i in comp:
                pos = self._node_items[i].pos()
                nodes.append({"num": _plain(self._nodes[i]["num"]), "x": round(pos.x(), 2), "y": round(pos.y(), 2)})
            label = f"FBMN_cluster_{ci + 1}"
            # only spectra of the same form are compared, so all nodes of a cluster share it
            payload = {"cluster": label, "form": self._nodes[comp[0]]["form"], "nodes": nodes, "edges": edges_by_cluster[ci]}
            records.append((label, len(comp), len(edges_by_cluster[ci]), payload))

        rows = []
        for label, n_nodes, n_edges, payload in records:
            text = json.dumps(payload, default=str)
            for part, start in enumerate(range(0, max(len(text), 1), NETWORK_CELL_LIMIT), start=1):
                rows.append({"Cluster": label, "Part": part, "Nodes": n_nodes, "Edges": n_edges, "JSON": text[start : start + NETWORK_CELL_LIMIT]})
        return rows

    @staticmethod
    def _parse_network_rows(rows):
        """Return (parameters, [cluster dicts]) from the rows of the network sheet."""
        parts = defaultdict(list)
        for r in rows:
            parts[str(r["Cluster"])].append((int(r["Part"]), r["JSON"] or ""))
        payloads = {label: json.loads("".join(text for _, text in sorted(p))) for label, p in parts.items()}
        params = payloads.pop("parameters", None)
        if params is None:
            raise ValueError("the 'parameters' row is missing")
        return params, list(payloads.values())

    def _set_params_controls(self, params):
        self.tolerance_unit_combo.setCurrentIndex(1 if params.get("fragment_bin_width_unit") == "ppm" else 0)
        self.tolerance_spin.setValue(float(params["fragment_bin_width"]))
        algorithm = params.get("similarity_algorithm")
        if algorithm in self._algorithms:
            self.algorithm_combo.setCurrentText(algorithm)
        forms = list(params.get("forms") or ["native"])
        self.form_combo.setCurrentIndex(next((k for k, (_, f) in enumerate(FORM_OPTIONS) if f == forms), 0))
        self.selection_combo.setCurrentIndex(next((k for k, (_, s) in enumerate(SELECTION_OPTIONS) if s == params.get("spectrum_selection")), 0))
        self.min_fragments_spin.setValue(int(params.get("min_fragments", self.min_fragments_spin.value())))
        self.min_fragments_pct_spin.setValue(float(params.get("min_fragments_above_pct", self.min_fragments_pct_spin.value())))
        self.min_score_spin.setValue(float(params.get("min_edge_score", self.min_score_spin.value())))

    def _restore_network(self):
        """Rebuild the network (parameters, nodes, positions, edges and scores) from the network sheet of the
        results file; the MS/MS spectra are re-collected from the loaded raw data (no similarity recalculation)."""
        if self._get_saved_network is None:
            return
        rows = self._get_saved_network()
        if not rows:
            return
        try:
            params, clusters = self._parse_network_rows(rows)
            self._set_params_controls(params)
        except (ValueError, KeyError, TypeError) as ex:
            QtWidgets.QMessageBox.warning(self, "FBMN", f"Could not read the saved network: {ex}")
            return
        gen_params = self._current_gen_params()
        if params.get("similarity_algorithm") not in self._algorithms:
            QtWidgets.QMessageBox.warning(self, "FBMN", f"The saved similarity algorithm '{params.get('similarity_algorithm')}' is not available; '{gen_params['algorithm']}' is used for the mirror plots.")

        fetched = {}
        if self._get_nodes is not None and MATCHMS_AVAILABLE:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                available = self._get_nodes(gen_params["forms"], gen_params["selection"]) or []
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            for n in available:
                self._select_spectrum(n, gen_params["min_fragments"], gen_params["min_fragments_pct"])
            fetched = {(str(n["num"]), n["form"]): n for n in available}
        table = self._get_table() if self._get_table else None
        table_nums = {str(v): v for v in table["Num"].to_list()} if table is not None and "Num" in table.columns else {}
        try:
            keys = [(table_nums.get(str(sn["num"]), sn["num"]), cluster["form"]) for cluster in clusters for sn in cluster["nodes"]]
        except (KeyError, TypeError) as ex:
            QtWidgets.QMessageBox.warning(self, "FBMN", f"Could not read the saved network: {ex}")
            return
        missing_keys = [k for k in keys if (str(k[0]), k[1]) not in fetched]
        node_info = (self._get_node_info(missing_keys) or {}) if (missing_keys and self._get_node_info) else {}

        nodes, specs, positions, edges = [], [], [], []
        n_unknown = 0
        try:
            for cluster in clusters:
                form = cluster["form"]
                local = {}
                for sn in cluster["nodes"]:
                    num = table_nums.get(str(sn["num"]), sn["num"])
                    found = fetched.get((str(num), form))
                    if found is not None:
                        node = dict(found)
                        node["precursor_mz"] = float(found["scan"].precursor_mz)
                        specs.append(self._to_spectrum(found["scan"]))
                    elif (num, form) in node_info:
                        node = dict(node_info[(num, form)], num=num, form=form, scan=None)
                        specs.append(None)
                    else:
                        n_unknown += 1
                        continue
                    local[str(num)] = len(nodes)
                    nodes.append(node)
                    positions.append((float(sn["x"]), float(sn["y"])))
                for e in cluster["edges"]:
                    a, b = local.get(str(e["source"])), local.get(str(e["target"]))
                    if a is not None and b is not None:
                        edges.append((a, b, float(e["score"])))
        except (ValueError, KeyError, TypeError) as ex:
            QtWidgets.QMessageBox.warning(self, "FBMN", f"Could not read the saved network: {ex}")
            return
        if not nodes:
            self.status_label.setText("The saved network has no nodes.")
            return

        self.clear()
        self._gen_params = gen_params
        self._specs = specs
        self._build_network(nodes, edges, positions)
        n_missing = sum(1 for s in specs if s is None)
        status = f"Restored network (saved {params.get('saved', '?')}): {self.status_label.text()}"
        if n_missing:
            status += f" The MS/MS spectra of {n_missing} nodes are not available in the loaded raw data."
        if n_unknown:
            status += f" {n_unknown} saved nodes are not present in the results sheet and were skipped."
        self.status_label.setText(status)
