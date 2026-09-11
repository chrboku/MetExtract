# -*- coding: utf-8 -*-
"""
Statistics Tab GUI for MetExtract II

This module provides the GUI components for the Statistics tab, including:
- Tree view for selecting analysis methods
- Data Quality Overview visualizations
- PCA, HCA, and Heat Map plots
- Interactive Volcano Plots with rectangular selection

Copyright (C) 2015 MetExtract Team
License: GNU General Public License v2 (GPLv2)
"""

import matplotlib
from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from scipy.cluster.hierarchy import dendrogram
from .wheel_safe_widgets import CtrlWheelComboBox
from .statisticsModule import DataQualityAnalysis, MultivariateAnalysis, SelectionManager, StatisticsData, UnivariateAnalysis

matplotlib.use("Qt5Agg")

QComboBox = CtrlWheelComboBox

# Reference sizes matching matplotlib's own defaults, used as the 100% baseline for _scale_plot_fonts()
_BASE_TITLE_FONTSIZE = 18.0
_BASE_LABEL_FONTSIZE = 18.0
_BASE_TICK_FONTSIZE = 18.0


def _scale_plot_fonts(fig, scale: float):
    """Scale every axes' title/axis-label/tick/legend font size in `fig` to `scale` times matplotlib's defaults."""
    title_size = _BASE_TITLE_FONTSIZE * scale
    label_size = _BASE_LABEL_FONTSIZE * scale
    tick_size = _BASE_TICK_FONTSIZE * scale
    for ax in fig.get_axes():
        if ax.get_title():
            ax.title.set_fontsize(title_size)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.tick_params(axis="both", labelsize=tick_size)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(tick_size)
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(title_size)


def _ids_for_positions(volcano_data: Dict[str, Any], positions: List[int]) -> Tuple[List[Any], List[Any]]:
    """Return (ogroups, nums) for the given positional indices into `volcano_data`."""
    feature_names = volcano_data.get("feature_names", [])
    feature_group_ids = volcano_data.get("featureGroupIDs", [])
    nums = [feature_names[i] for i in positions if i < len(feature_names)]
    ogroups = [feature_group_ids[i] for i in positions if i < len(feature_group_ids)]
    return ogroups, nums


class AddComparisonDialog(QDialog):
    """Dialog for adding a new volcano plot comparison."""

    def __init__(self, available_groups: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Volcano Plot Comparison")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        # Group 1 selection
        layout.addWidget(QLabel("Select first group (Control/Reference):"))
        self.group1_combo = QComboBox()
        self.group1_combo.addItems(available_groups)
        layout.addWidget(self.group1_combo)

        # Group 2 selection
        layout.addWidget(QLabel("Select second group (Treatment/Condition):"))
        self.group2_combo = QComboBox()
        self.group2_combo.addItems(available_groups)
        layout.addWidget(self.group2_combo)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_groups(self) -> Tuple[str, str]:
        """Return selected groups."""
        return (self.group1_combo.currentText(), self.group2_combo.currentText())


class GroupSelectionDialog(QDialog):
    """Dialog for selecting which groups to include in multivariate analyses."""

    last_selected_groups: List[str] = []  # Class variable to remember last selection

    def __init__(self, available_groups: List[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Groups for Analysis")
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select groups to include in the analysis:"))

        # Create checkboxes for each group
        self.checkboxes = {}
        for group in available_groups:
            checkbox = QtWidgets.QCheckBox(group)
            # Use last selection if available, otherwise select all
            if GroupSelectionDialog.last_selected_groups:
                checkbox.setChecked(group in GroupSelectionDialog.last_selected_groups)
            else:
                checkbox.setChecked(True)  # All selected by default
            self.checkboxes[group] = checkbox
            layout.addWidget(checkbox)

        # Select All / Deselect All buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(self._select_all)
        deselect_all_btn.clicked.connect(self._deselect_all)
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        layout.addLayout(button_layout)

        # OK/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _select_all(self):
        """Select all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def _deselect_all(self):
        """Deselect all checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def get_selected_groups(self) -> List[str]:
        """Return list of selected group names."""
        selected = [group for group, checkbox in self.checkboxes.items() if checkbox.isChecked()]
        # Remember this selection for next time
        GroupSelectionDialog.last_selected_groups = selected
        return selected


class InteractiveVolcanoCanvas(FigureCanvas):
    """Canvas for volcano plot with interactive rectangular selection."""

    selectionChanged = Signal(list, bool)  # Signal emitted when selection changes
    idsFilterRequested = Signal(list, list)  # (ogroups, nums) emitted on Ctrl+drag rectangle

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.volcano_data = None
        self.scatter = None
        self.highlighted_indices = []
        self.rect_selector = None
        self._press_pos = None
        self.title = "Volcano Plot"
        self.filtered_visible_nums = None  # None = no Experiment results filter active

        # Set up matplotlib event handling
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_mouse_release)

        # Initialize rectangle selector
        self._setup_rect_selector()

    def _setup_rect_selector(self):
        """Set up the rectangle selector for feature selection."""
        self.rect_selector = RectangleSelector(
            self.axes,
            self._on_select,
            useblit=True,
            button=[1],  # Left mouse button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

    def _on_select(self, eclick, erelease):
        """Handle rectangle selection. Ctrl+drag adds the OGroup/Num of the enclosed dots to
        the Experiment results ID filter instead of performing a normal (additive) selection."""
        if self.volcano_data is None:
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        # Get data bounds
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Find points within selection rectangle
        log2_fc = self.volcano_data["log2_fc"]
        neg_log10_pval = self.volcano_data["neg_log10_pval"]

        selected_mask = (log2_fc >= x_min) & (log2_fc <= x_max) & (neg_log10_pval >= y_min) & (neg_log10_pval <= y_max)

        selected_indices = np.where(selected_mask)[0].tolist()

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            ogroups, nums = _ids_for_positions(self.volcano_data, selected_indices)
            self.idsFilterRequested.emit(ogroups, nums)
            return

        self.selectionChanged.emit(selected_indices, False)

    def _on_mouse_press(self, event):
        """Record mouse-press position for drag detection."""
        if event.button == 1:
            self._press_pos = (event.x, event.y)

    def _on_mouse_release(self, event):
        """Emit selectionChanged for a single-point click (no drag)."""
        if event.button != 1 or event.inaxes != self.axes or self.volcano_data is None:
            return
        if self._press_pos is None:
            return
        dx = abs(event.x - self._press_pos[0])
        dy = abs(event.y - self._press_pos[1])
        self._press_pos = None
        if dx > 5 or dy > 5:
            return  # Was a drag; handled by RectangleSelector
        log2_fc = self.volcano_data["log2_fc"]
        neg_log10_pval = self.volcano_data["neg_log10_pval"]
        try:
            pts_display = self.axes.transData.transform(np.column_stack([log2_fc, neg_log10_pval]))
            click_display = np.array([event.x, event.y])
            dists = np.sqrt(np.sum((pts_display - click_display) ** 2, axis=1))
            closest = int(np.argmin(dists))
            if dists[closest] <= 10:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                additive = modifiers == Qt.ControlModifier
                self.selectionChanged.emit([closest], additive)
        except Exception:
            pass

    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == "escape":
            self.highlighted_indices = []
            self.update_highlighting()

    def set_volcano_data(self, data: Dict[str, Any]):
        """Set the volcano plot data and redraw."""
        self.volcano_data = data
        self.draw_volcano(preserve_view=False)

    def set_id_filter(self, visible_nums: Optional[set]):
        """Dim (10% alpha) every dot whose Num is not in `visible_nums`; pass None to disable dimming."""
        self.filtered_visible_nums = visible_nums
        self.draw_volcano(preserve_view=True)

    def draw_volcano(self, preserve_view: bool = False):
        """Draw the volcano plot."""
        if self.volcano_data is None:
            return

        old_xlim = None
        old_ylim = None
        if preserve_view and self.axes.has_data():
            old_xlim = self.axes.get_xlim()
            old_ylim = self.axes.get_ylim()

        if not self.volcano_data.get("success", False):
            # Show error message
            self.axes.clear()
            error_msg = self.volcano_data.get("error", "Unknown error")
            self.axes.text(0.5, 0.5, f"Error:\n{error_msg}", ha="center", va="center", transform=self.axes.transAxes, color="red", wrap=True)
            self.axes.set_xlim(0, 1)
            self.axes.set_ylim(0, 1)
            self.fig.tight_layout()
            self.draw()
            return

        self.axes.clear()

        log2_fc = self.volcano_data["log2_fc"]
        neg_log10_pval = self.volcano_data["neg_log10_pval"]
        significant = self.volcano_data["significant"]
        fc_threshold = self.volcano_data.get("fc_threshold", 1.0)
        pvalue_threshold = self.volcano_data.get("pvalue_threshold", 0.05)

        # Check if we have data to plot
        if len(log2_fc) == 0:
            self.axes.text(0.5, 0.5, "No features to plot", ha="center", va="center", transform=self.axes.transAxes, color="gray")
            self.fig.tight_layout()
            self.draw()
            return

        # Color points based on significance (highlighted points drawn separately on top)
        highlighted_set = set(self.highlighted_indices)
        colors = []
        for i, sig in enumerate(significant):
            if i in highlighted_set:
                colors.append("gray")  # placeholder; will be hidden by top layer
            elif sig:
                if log2_fc[i] > 0:
                    colors.append("red")
                else:
                    colors.append("blue")
            else:
                colors.append("gray")

        # Dim dots for features hidden by the active Experiment results filter(s)
        if self.filtered_visible_nums is not None:
            feature_names = self.volcano_data.get("feature_names", [])
            alphas = [0.7 if (i < len(feature_names) and feature_names[i] in self.filtered_visible_nums) else 0.1 for i in range(len(log2_fc))]
        else:
            alphas = 0.7

        self.scatter = self.axes.scatter(log2_fc, neg_log10_pval, c=colors, alpha=alphas, s=30, edgecolors="none")

        # Draw highlighted points on top: green, 2× size
        if highlighted_set:
            h_idx = np.array(sorted(highlighted_set))
            valid = h_idx[h_idx < len(log2_fc)]
            if len(valid):
                self.axes.scatter(
                    log2_fc[valid],
                    neg_log10_pval[valid],
                    c="green",
                    alpha=1.0,
                    s=60,
                    edgecolors="darkgreen",
                    linewidths=0.8,
                    zorder=5,
                )

        # Add threshold lines
        self.axes.axhline(y=-np.log10(pvalue_threshold), color="gray", linestyle="--", alpha=0.5)
        self.axes.axvline(x=-fc_threshold, color="gray", linestyle="--", alpha=0.5)
        self.axes.axvline(x=fc_threshold, color="gray", linestyle="--", alpha=0.5)

        self.axes.set_xlabel("log₂(Fold Change)")
        self.axes.set_ylabel("-log₁₀(p-value)")
        self.axes.set_title(self.title)

        if old_xlim is not None and old_ylim is not None:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)

        _scale_plot_fonts(self.fig, 0.5)
        self.fig.tight_layout()
        self.draw()

    def update_highlighting(self, indices: Optional[List[int]] = None):
        """Update highlighted points."""
        if indices is not None:
            self.highlighted_indices = indices
        self.draw_volcano(preserve_view=True)

    def clear_highlighting(self):
        """Clear all highlighting."""
        self.highlighted_indices = []
        self.draw_volcano(preserve_view=True)


class MultiVolcanoWidget(QWidget):
    """Displays all volcano plot comparisons as subplots of a single figure.

    A single `NavigationToolbar` controls zoom/pan; each subplot's axes are
    independent (matplotlib toolbar zoom/pan acts on whichever axes the mouse
    is over), so panning/zooming one comparison doesn't affect the others.
    Selecting a feature (click or rectangle-drag) in any subplot highlights it
    in every subplot, since all comparisons share the same underlying feature
    ordering (`data.index` of the active feature table).
    """

    featureSelected = Signal(list)  # Signal when features are selected (position indices)
    idsFilterRequested = Signal(list, list)  # (ogroups, nums) emitted on Ctrl+drag rectangle

    def __init__(self, parent=None):
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        outer_layout.addWidget(self.toolbar)
        outer_layout.addWidget(self.canvas)

        self.subplots: List[Dict[str, Any]] = []  # {"ax", "title", "volcano_data", "rect_selector"}
        self.highlighted_indices: List[int] = []
        self.filtered_visible_nums = None  # None = no Experiment results filter active
        self.selection_manager = SelectionManager()
        self.selection_manager.register_callback(self._on_selection_changed)

        self._press_pos = None
        self._press_ax = None
        self.canvas.mpl_connect("button_press_event", self._on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._on_mouse_release)
        self.canvas.mpl_connect("key_press_event", self._on_key_press)

    def set_comparisons(self, comparisons: List[Tuple[str, str]], data: Dict[str, Any]):
        """
        Set up volcano subplots for all comparisons.

        Args:
            comparisons: List of (group1, group2) tuples
            data: Dictionary containing feature data and group info
        """
        self.fig.clear()
        self.subplots.clear()
        self.highlighted_indices = []

        if not comparisons:
            self.canvas.draw()
            return

        # 2 or 3 comparisons stack vertically (1 column); 4+ switch to a rows x columns grid.
        n_plots = len(comparisons)
        if n_plots <= 3:
            cols = 1
            rows = n_plots
        else:
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)

        for i, (group1, group2) in enumerate(comparisons):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            entry = {"ax": ax, "title": f"{group1} vs {group2}", "volcano_data": None, "rect_selector": None}

            # Calculate volcano data for this comparison
            if "feature_data" in data and "group_info" in data:
                feature_data = data["feature_data"]
                group_info = data["group_info"]

                if group1 in group_info and group2 in group_info:
                    metadata = data.get("metadata") if "metadata" in data else None
                    entry["volcano_data"] = UnivariateAnalysis.calculate_volcano_data(feature_data, group_info[group1], group_info[group2], metadata=metadata)

            self.subplots.append(entry)
            self._draw_subplot(entry)

            entry["rect_selector"] = RectangleSelector(
                ax,
                lambda eclick, erelease, entry=entry: self._on_rect_select(entry, eclick, erelease),
                useblit=True,
                button=[1],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
            )

        _scale_plot_fonts(self.fig, 0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_subplot(self, entry: Dict[str, Any], preserve_view: bool = False):
        """(Re)draw a single subplot's volcano data, honoring the current highlight set."""
        ax = entry["ax"]
        vd = entry["volcano_data"]

        old_xlim = old_ylim = None
        if preserve_view and ax.has_data():
            old_xlim, old_ylim = ax.get_xlim(), ax.get_ylim()

        ax.clear()
        ax.set_title(entry["title"])

        if vd is None or not vd.get("success", False):
            error_msg = "No data" if vd is None else vd.get("error", "Unknown error")
            ax.text(0.5, 0.5, f"Error:\n{error_msg}", ha="center", va="center", transform=ax.transAxes, color="red", wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            return

        log2_fc = vd["log2_fc"]
        neg_log10_pval = vd["neg_log10_pval"]
        significant = vd["significant"]
        fc_threshold = vd.get("fc_threshold", 1.0)
        pvalue_threshold = vd.get("pvalue_threshold", 0.05)

        if len(log2_fc) == 0:
            ax.text(0.5, 0.5, "No features to plot", ha="center", va="center", transform=ax.transAxes, color="gray")
            return

        highlighted_set = set(self.highlighted_indices)
        colors = []
        for i, sig in enumerate(significant):
            if i in highlighted_set:
                colors.append("gray")  # hidden by highlighted overlay below
            elif sig:
                colors.append("red" if log2_fc[i] > 0 else "blue")
            else:
                colors.append("gray")

        if self.filtered_visible_nums is not None:
            feature_names = vd.get("feature_names", [])
            alphas = [0.7 if (i < len(feature_names) and feature_names[i] in self.filtered_visible_nums) else 0.1 for i in range(len(log2_fc))]
        else:
            alphas = 0.7

        ax.scatter(log2_fc, neg_log10_pval, c=colors, alpha=alphas, s=25, edgecolors="none")

        if highlighted_set:
            h_idx = np.array(sorted(highlighted_set))
            valid = h_idx[h_idx < len(log2_fc)]
            if len(valid):
                ax.scatter(log2_fc[valid], neg_log10_pval[valid], c="green", alpha=1.0, s=50, edgecolors="darkgreen", linewidths=0.8, zorder=5)

        ax.axhline(y=-np.log10(pvalue_threshold), color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=-fc_threshold, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=fc_threshold, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("log₂(Fold Change)")
        ax.set_ylabel("-log₁₀(p-value)")

        if old_xlim is not None:
            ax.set_xlim(old_xlim)
            ax.set_ylim(old_ylim)

    def _on_rect_select(self, entry: Dict[str, Any], eclick, erelease):
        """Handle rectangle selection within one subplot's axes. Ctrl+drag adds the
        OGroup/Num of the enclosed dots to the Experiment results ID filter instead of
        performing a normal selection."""
        vd = entry["volcano_data"]
        if vd is None or not vd.get("success", False):
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        log2_fc = vd["log2_fc"]
        neg_log10_pval = vd["neg_log10_pval"]
        selected_mask = (log2_fc >= x_min) & (log2_fc <= x_max) & (neg_log10_pval >= y_min) & (neg_log10_pval <= y_max)
        selected_indices = np.where(selected_mask)[0].tolist()

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            ogroups, nums = _ids_for_positions(vd, selected_indices)
            self.idsFilterRequested.emit(ogroups, nums)
            return

        self._handle_selection(selected_indices, False)

    def _on_mouse_press(self, event):
        """Record mouse-press position/axes for drag detection."""
        if event.button == 1:
            self._press_pos = (event.x, event.y)
            self._press_ax = event.inaxes

    def _on_mouse_release(self, event):
        """Emit a selection for a single-point click (no drag) on whichever subplot was clicked."""
        if event.button != 1 or event.inaxes is None or self._press_pos is None:
            return
        dx = abs(event.x - self._press_pos[0])
        dy = abs(event.y - self._press_pos[1])
        press_ax = self._press_ax
        self._press_pos = None
        self._press_ax = None
        if dx > 5 or dy > 5 or event.inaxes != press_ax:
            return  # Was a drag; handled by the subplot's RectangleSelector

        entry = next((e for e in self.subplots if e["ax"] == event.inaxes), None)
        if entry is None:
            return
        vd = entry["volcano_data"]
        if vd is None or not vd.get("success", False):
            return

        log2_fc = vd["log2_fc"]
        neg_log10_pval = vd["neg_log10_pval"]
        try:
            pts_display = entry["ax"].transData.transform(np.column_stack([log2_fc, neg_log10_pval]))
            click_display = np.array([event.x, event.y])
            dists = np.sqrt(np.sum((pts_display - click_display) ** 2, axis=1))
            closest = int(np.argmin(dists))
            if dists[closest] <= 10:
                modifiers = QtWidgets.QApplication.keyboardModifiers()
                additive = modifiers == Qt.ControlModifier
                self._handle_selection([closest], additive)
        except Exception:
            pass

    def _on_key_press(self, event):
        if event.key == "escape":
            self.update_highlighting([])

    def _handle_selection(self, indices: List[int], additive: bool):
        """Handle a user-driven selection from any subplot."""
        self.selection_manager.add_selection(indices, additive)
        self.featureSelected.emit(self.selection_manager.get_selected_indices())

    def _on_selection_changed(self, indices: List[int]):
        """Update all subplots when the internal selection manager's selection changes."""
        self.update_highlighting(indices)

    def update_highlighting(self, indices: List[int]):
        """Programmatically highlight feature positions in every subplot (no selection signal emitted)."""
        self.highlighted_indices = list(indices)
        for entry in self.subplots:
            self._draw_subplot(entry, preserve_view=True)
        _scale_plot_fonts(self.fig, 0.5)
        self.fig.tight_layout()
        self.canvas.draw()

    def set_id_filter(self, visible_nums: Optional[set]):
        """Dim (10% alpha) every dot whose Num is not in `visible_nums`; pass None to disable dimming."""
        self.filtered_visible_nums = visible_nums
        for entry in self.subplots:
            self._draw_subplot(entry, preserve_view=True)
        _scale_plot_fonts(self.fig, 0.5)
        self.fig.tight_layout()
        self.canvas.draw()


class NumericTableWidgetItem(QTableWidgetItem):
    """Custom QTableWidgetItem for numeric sorting."""

    def __init__(self, value):
        super().__init__(str(value))
        self.numeric_value = value

    def __lt__(self, other):
        """Compare items for sorting."""
        if isinstance(other, NumericTableWidgetItem):
            # Handle N/A values - put them at the end
            if self.numeric_value == "N/A":
                return False
            if other.numeric_value == "N/A":
                return True
            # Both are numeric
            try:
                return float(self.numeric_value) < float(other.numeric_value)
            except (ValueError, TypeError):
                return str(self.numeric_value) < str(other.numeric_value)
        return super().__lt__(other)


class _BoldSelectedDelegate(QStyledItemDelegate):
    """Renders selected rows bold without changing their background color."""

    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        if opt.state & QStyle.State_Selected:
            opt.state &= ~QStyle.State_Selected
            font = opt.font
            font.setBold(True)
            opt.font = font
        super().paint(painter, opt, index)


class SelectedFeaturesTable(QTreeWidget):
    """Tree-like view for selected features, grouped by Group ID."""

    viewFeatureRequested = Signal(int, str)  # Signal to view feature in experiment results

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(13)
        self.setHeaderLabels(["Feature ID", "Group ID", "m/z (Total)", "RT (Sig ↑)", "Log2 FC (Sig ↓)", "p-value", "G1 Mean", "G1 Median", "G1 SD", "G2 Mean", "G2 Median", "G2 SD", "Index"])
        self.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setStyleSheet("")
        self.setItemDelegate(_BoldSelectedDelegate(self))
        self.setSortingEnabled(False)
        self.feature_data = []
        self._feature_item_by_id = {}

    @staticmethod
    def _format_numeric(value, fmt):
        if value == "N/A":
            return "N/A"
        try:
            return fmt.format(float(value))
        except (ValueError, TypeError):
            return str(value)

    @staticmethod
    def _group_sort_key(group_id):
        try:
            return (0, int(float(group_id)))
        except (ValueError, TypeError):
            return (1, str(group_id))

    def _make_feature_item(self, idx, feature_metadata, group_stats):
        feature_pair_id = "N/A"
        feature_group_id = "N/A"
        mz = "N/A"
        rt = "N/A"
        fc = "N/A"
        pval = "N/A"

        if feature_metadata:
            feature_pair_id = feature_metadata.get("featurePairID", {}).get(idx, "N/A")
            feature_group_id = feature_metadata.get("featureGroupID", {}).get(idx, "N/A")
            mz = feature_metadata.get("mz", {}).get(idx, "N/A")
            rt = feature_metadata.get("rt", {}).get(idx, "N/A")
            fc = feature_metadata.get("log2_fc", {}).get(idx, "N/A")
            pval = feature_metadata.get("pvalue", {}).get(idx, "N/A")

        if group_stats and idx in group_stats.get("g1_mean", {}):
            g1_mean = group_stats["g1_mean"][idx]
            g1_median = group_stats["g1_median"][idx]
            g1_sd = group_stats["g1_sd"][idx]
        else:
            g1_mean, g1_median, g1_sd = "N/A", "N/A", "N/A"

        if group_stats and idx in group_stats.get("g2_mean", {}):
            g2_mean = group_stats["g2_mean"][idx]
            g2_median = group_stats["g2_median"][idx]
            g2_sd = group_stats["g2_sd"][idx]
        else:
            g2_mean, g2_median, g2_sd = "N/A", "N/A", "N/A"

        texts = [
            str(feature_pair_id),
            str(feature_group_id),
            self._format_numeric(mz, "{:.4f}"),
            self._format_numeric(rt, "{:.2f}"),
            self._format_numeric(fc, "{:.3f}"),
            self._format_numeric(pval, "{:.2e}"),
            self._format_numeric(g1_mean, "{:.2e}"),
            self._format_numeric(g1_median, "{:.2e}"),
            self._format_numeric(g1_sd, "{:.2e}"),
            self._format_numeric(g2_mean, "{:.2e}"),
            self._format_numeric(g2_median, "{:.2e}"),
            self._format_numeric(g2_sd, "{:.2e}"),
            str(idx),
        ]

        item = QTreeWidgetItem(texts)
        item.setData(0, Qt.UserRole, idx)
        item.setData(1, Qt.UserRole, feature_group_id)
        item.setData(0, Qt.UserRole + 1, feature_pair_id)
        return item

    def update_features(self, indices: List[int], feature_metadata: Optional[Dict[str, Any]] = None, group_stats: Optional[Dict[str, Any]] = None, row_colors: Optional[Dict[int, Any]] = None):
        """Update grouped tree with selected features."""
        self.clear()
        self.feature_data = []
        self._feature_item_by_id = {}

        grouped = {}
        for idx in indices:
            group_id = "N/A"
            if feature_metadata:
                group_id = feature_metadata.get("featureGroupID", {}).get(idx, "N/A")
            grouped.setdefault(group_id, []).append(idx)

        for group_id in sorted(grouped.keys(), key=self._group_sort_key):
            group_label = f"Group {group_id}"
            group_indices = grouped[group_id]
            total = len(group_indices)
            sig_higher = 0
            sig_lower = 0
            if row_colors:
                for idx in group_indices:
                    c = row_colors.get(idx)
                    if c is not None:
                        if c.red() > c.blue():
                            sig_higher += 1
                        elif c.blue() > c.red():
                            sig_lower += 1

            group_item = QTreeWidgetItem([group_label, str(group_id), str(total), str(sig_higher), str(sig_lower), "", "", "", "", "", "", "", ""])
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsSelectable)
            self.addTopLevelItem(group_item)

            for idx in group_indices:
                feature_item = self._make_feature_item(idx, feature_metadata, group_stats)
                group_item.addChild(feature_item)
                self._feature_item_by_id[idx] = feature_item

                if row_colors and idx in row_colors:
                    brush = QtGui.QBrush(row_colors[idx])
                    for c in range(self.columnCount()):
                        feature_item.setBackground(c, brush)

                self.feature_data.append({"index": idx, "mz": feature_item.text(2), "rt": feature_item.text(3)})

        self.expandAll()

    def select_feature_ids(self, feature_ids: List[int]):
        self.clearSelection()
        first_item = None
        for fid in feature_ids:
            item = self._feature_item_by_id.get(fid)
            if item is None:
                continue
            item.setSelected(True)
            if first_item is None:
                first_item = item

        if first_item is not None:
            self.scrollToItem(first_item)
            self.setCurrentItem(first_item)

    def get_selected_feature_ids(self) -> List[int]:
        feature_ids = []
        for item in self.selectedItems():
            if item.childCount() > 0:
                continue
            fid = item.data(0, Qt.UserRole)
            if fid is None:
                continue
            try:
                feature_ids.append(int(fid))
            except (ValueError, TypeError):
                continue
        return feature_ids

    def get_current_feature_pair_id(self) -> Optional[int]:
        item = self.currentItem()
        if item is None or item.childCount() > 0:
            selected = self.selectedItems()
            item = selected[0] if selected else None

        if item is None or item.childCount() > 0:
            return None

        pair_id = item.data(0, Qt.UserRole + 1)
        try:
            if pair_id == "N/A":
                return None
            return int(float(pair_id))
        except (ValueError, TypeError):
            return None


class StatisticsCanvas(FigureCanvas):
    """Generic canvas for statistical visualizations."""

    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class StatisticsTabWidget(QWidget):
    """Main widget for the Statistics tab."""

    # Signal to switch to experiment results and show a specific feature
    showFeatureInExperiment = Signal(int)
    idsFilterRequested = Signal(list, list)  # (ogroups, nums) forwarded from volcano Ctrl+drag rectangles

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats_data = StatisticsData()
        self.selection_manager = SelectionManager()
        self._updating_from_volcano = False  # Guard against circular table↔volcano sync
        self._heatmap_state: Optional[Dict[str, Any]] = None
        self._active_id_filter_visible_nums: Optional[set] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Horizontal splitter so the user can resize the tree-view vs. the plot area
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel: Tree view for method selection
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)

        left_layout.addWidget(QLabel("<b>Analysis Methods</b>"))

        self.methods_tree = QTreeWidget()
        self.methods_tree.setHeaderHidden(True)
        self._populate_methods_tree()
        left_layout.addWidget(self.methods_tree)

        # Add comparison button
        self.add_comparison_btn = QPushButton("Add Volcano Comparison")
        self.add_comparison_btn.setEnabled(False)
        left_layout.addWidget(self.add_comparison_btn)

        # Remove comparison button
        self.remove_comparison_btn = QPushButton("Remove Selected Comparison")
        self.remove_comparison_btn.setEnabled(False)
        left_layout.addWidget(self.remove_comparison_btn)

        # Add abundance type selector
        left_layout.addWidget(QLabel("<b>Abundance Type</b>"))
        self.abundance_combo = QComboBox()
        self.abundance_combo.addItems(["Total (N + L)", "N (Natural)", "L (Labeled)"])
        self.abundance_combo.setCurrentIndex(1)  # Default to N (Natural)
        self.abundance_combo.currentIndexChanged.connect(self._on_abundance_type_changed)
        left_layout.addWidget(self.abundance_combo)

        # Add imputation method selector
        left_layout.addWidget(QLabel("<b>Imputation Method</b>"))
        self.imputation_combo = QComboBox()
        self.imputation_combo.addItems(["0-imputation", "LOD/2 imputation"])
        self.imputation_combo.currentIndexChanged.connect(self._on_imputation_changed)
        left_layout.addWidget(self.imputation_combo)

        # Add feature filtering selector
        left_layout.addWidget(QLabel("<b>Feature Selection</b>"))
        self.feature_filter_combo = QComboBox()
        self.feature_filter_combo.addItems(["All features", "Most abundant per OGroup"])
        self.feature_filter_combo.currentIndexChanged.connect(self._on_feature_filter_changed)
        left_layout.addWidget(self.feature_filter_combo)

        main_splitter.addWidget(left_panel)

        # Right panel: Content area with splitter
        right_splitter = QSplitter(Qt.Vertical)

        # Visualization area
        self.viz_container = QScrollArea()
        self.viz_container.setWidgetResizable(True)
        self.viz_widget = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_widget)
        self.viz_container.setWidget(self.viz_widget)

        # Heatmap pagination bar (only visible while the heatmap is shown); kept outside
        # viz_layout so it is not removed by _clear_visualization()
        self.heatmap_pagination_bar = QWidget()
        pagination_layout = QHBoxLayout(self.heatmap_pagination_bar)
        pagination_layout.setContentsMargins(4, 4, 4, 4)
        self.heatmap_prev_btn = QPushButton("\u25c0 Previous 100")
        self.heatmap_prev_btn.clicked.connect(self._heatmap_prev_page)
        self.heatmap_next_btn = QPushButton("Next 100 \u25b6")
        self.heatmap_next_btn.clicked.connect(self._heatmap_next_page)
        self.heatmap_page_label = QLabel("")
        pagination_layout.addWidget(self.heatmap_prev_btn)
        pagination_layout.addWidget(self.heatmap_page_label)
        pagination_layout.addWidget(self.heatmap_next_btn)
        pagination_layout.addStretch()
        self.heatmap_pagination_bar.setVisible(False)

        viz_outer = QWidget()
        viz_outer_layout = QVBoxLayout(viz_outer)
        viz_outer_layout.setContentsMargins(0, 0, 0, 0)
        viz_outer_layout.addWidget(self.heatmap_pagination_bar)
        viz_outer_layout.addWidget(self.viz_container)
        right_splitter.addWidget(viz_outer)

        # Selected features table
        table_container = QFrame()
        table_layout = QVBoxLayout(table_container)
        table_layout.addWidget(QLabel("<b>Selected Features</b>"))
        self.features_table = SelectedFeaturesTable()
        table_layout.addWidget(self.features_table)

        right_splitter.addWidget(table_container)
        right_splitter.setSizes([600, 200])

        main_splitter.addWidget(right_splitter)

        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([300, 900])

        main_layout.addWidget(main_splitter)

        # Store current visualization widgets
        self.current_canvas = None
        self.current_toolbar = None
        self.multi_volcano_widget = None
        self.current_volcano_data = None  # Store volcano data for feature selection

    def _populate_methods_tree(self):
        """Populate the methods tree with analysis categories."""
        # Data Quality Overview
        quality_item = QTreeWidgetItem(self.methods_tree, ["Data Quality Overview"])
        QTreeWidgetItem(quality_item, ["Feature Detection Counts"])
        QTreeWidgetItem(quality_item, ["RSD (Relative Standard Deviation)"])
        QTreeWidgetItem(quality_item, ["Abundance Histograms"])

        # Multivariate Analysis
        multivariate_item = QTreeWidgetItem(self.methods_tree, ["Multivariate Analysis"])
        QTreeWidgetItem(multivariate_item, ["PCA (Principal Component Analysis)"])
        QTreeWidgetItem(multivariate_item, ["HCA (Hierarchical Cluster Analysis)"])
        QTreeWidgetItem(multivariate_item, ["Heat Map"])

        # Univariate Analysis
        univariate_item = QTreeWidgetItem(self.methods_tree, ["Univariate Analysis"])
        self.volcano_parent_item = QTreeWidgetItem(univariate_item, ["Volcano Plots"])
        self.all_volcano_item = QTreeWidgetItem(self.volcano_parent_item, ["All Comparisons"])

        self.methods_tree.expandAll()

    def _connect_signals(self):
        """Connect signals and slots."""
        self.methods_tree.itemClicked.connect(self._on_method_selected)
        self.add_comparison_btn.clicked.connect(self._add_volcano_comparison)
        self.remove_comparison_btn.clicked.connect(self._remove_volcano_comparison)
        self.features_table.viewFeatureRequested.connect(self._on_view_feature_requested)
        self.features_table.itemSelectionChanged.connect(self._on_table_row_selected)
        self.selection_manager.register_callback(self._on_selection_changed)

    def _get_group_colors(self, groups: List[str] = None) -> Dict[str, Any]:
        """Get consistent colors for groups, synced with the colors defined for each group in the Input tab.

        Args:
            groups: List of group names to get colors for. If None, uses all groups from stats_data.

        Returns:
            Dictionary mapping group names to colors
        """
        # Always base colors on the full original group list to ensure consistency
        all_groups = self.stats_data.get_group_names()
        synced_colors = self.stats_data.group_colors
        fallback_colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_groups), 3)))  # min 3 to avoid edge cases

        # Create mapping for all groups: prefer the Input tab's group color, fall back to a generated one
        all_group_colors = {}
        for idx, group_name in enumerate(all_groups):
            all_group_colors[group_name] = synced_colors.get(group_name, fallback_colors[idx])

        # Return colors only for requested groups (or all if None)
        if groups is None:
            return all_group_colors
        else:
            return {g: all_group_colors[g] for g in groups if g in all_group_colors}

    def load_experiment_data(self, experiment_data: Dict[str, Any]):
        """Load experiment data for statistical analysis."""
        if self.stats_data.load_from_experiment(experiment_data):
            self.add_comparison_btn.setEnabled(len(self.stats_data.get_group_names()) >= 2)
            logging.info("Statistics tab: Experiment data loaded successfully")

            # Enable/disable abundance selector based on available data
            has_separate_abundances = self.stats_data.feature_data_N is not None and self.stats_data.feature_data_L is not None
            self.abundance_combo.setEnabled(has_separate_abundances)
            if not has_separate_abundances:
                self.abundance_combo.setCurrentIndex(0)  # Default to Total
        else:
            logging.warning("Statistics tab: Failed to load experiment data")

    def _on_abundance_type_changed(self, index: int):
        """Handle abundance type selection change."""
        abundance_types = ["Total", "N", "L"]
        self.stats_data.set_abundance_type(abundance_types[index])
        # Refresh current visualization if one is shown
        current_item = self.methods_tree.currentItem()
        if current_item:
            self._on_method_selected(current_item, 0)

    def _on_imputation_changed(self, index: int):
        """Handle imputation method selection change."""
        imputation_methods = ["zero", "lod_half"]
        self.stats_data.imputation_method = imputation_methods[index]
        logging.info(f"Imputation method changed to: {imputation_methods[index]}")
        # Refresh current visualization if one is shown
        current_item = self.methods_tree.currentItem()
        if current_item:
            self._on_method_selected(current_item, 0)

    def _on_feature_filter_changed(self, index: int):
        """Handle feature filter selection change."""
        filter_methods = ["all", "most_abundant"]
        self.stats_data.feature_filter = filter_methods[index]
        logging.info(f"Feature filter changed to: {filter_methods[index]}")
        # Refresh current visualization if one is shown
        current_item = self.methods_tree.currentItem()
        if current_item:
            self._on_method_selected(current_item, 0)

    def _on_method_selected(self, item: QTreeWidgetItem, column: int):
        """Handle method selection from tree."""
        method_name = item.text(0)
        parent = item.parent()

        # Enable/disable buttons based on selection
        is_volcano_comparison = parent and parent.text(0) == "Volcano Plots" and method_name != "All Comparisons"
        self.remove_comparison_btn.setEnabled(is_volcano_comparison)

        # Show appropriate visualization
        if method_name == "Feature Detection Counts":
            self._show_detection_counts()
        elif method_name == "RSD (Relative Standard Deviation)":
            self._show_rsd_plot()
        elif method_name == "Abundance Histograms":
            self._show_abundance_histogram()
        elif method_name == "PCA (Principal Component Analysis)":
            self._show_pca()
        elif method_name == "HCA (Hierarchical Cluster Analysis)":
            self._show_hca()
        elif method_name == "Heat Map":
            self._show_heatmap()
        elif method_name == "All Comparisons":
            self._show_all_volcano_plots()
        elif is_volcano_comparison:
            self._show_single_volcano_plot(item)

    def _clear_visualization(self):
        """Clear the current visualization."""
        while self.viz_layout.count():
            item = self.viz_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.current_canvas = None
        self.current_toolbar = None
        self.multi_volcano_widget = None
        self.heatmap_pagination_bar.setVisible(False)

    def _show_detection_counts(self):
        """Show feature detection counts visualization."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        detection_counts = DataQualityAnalysis.calculate_detection_counts(active_data, self.stats_data.group_info)

        if not detection_counts:
            self._show_no_data_message("No detection count data available")
            return

        groups = list(detection_counts.keys())
        n_groups = len(groups)
        group_colors = self._get_group_colors(groups)

        # Create subplots - one per group
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols

        canvas = StatisticsCanvas(self, width=12, height=4 * n_rows)
        toolbar = NavigationToolbar(canvas, self)

        canvas.fig.clear()
        axes = canvas.fig.subplots(n_rows, n_cols, squeeze=False)

        for idx, (group_name, counts) in enumerate(detection_counts.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            max_count = max(counts) if len(counts) > 0 else 1
            ax.hist(counts, bins=np.arange(1, max_count + 2) - 0.5, alpha=0.7, color=group_colors[group_name])
            ax.set_xlabel("Number of replicates with detection")
            ax.set_ylabel("Number of features")
            ax.set_title(f"{group_name}")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Feature Detection Counts by Group", fontweight="bold")
        _scale_plot_fonts(canvas.fig, 0.5)
        canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    def _add_percentile_lines(self, ax, values: np.ndarray, percent_unit: bool = False):
        """Add 10/25/50/75/90 percentile reference lines to a histogram axis."""
        if values is None or len(values) == 0:
            return

        line_specs = [
            (10, "P10", "#4C78A8", "--"),
            (25, "P25", "#59A14F", "-."),
            (50, "Median", "#E15759", "-"),
            (75, "P75", "#F28E2B", "-."),
            (90, "P90", "#B07AA1", "--"),
        ]

        for percentile, label, color, linestyle in line_specs:
            val = float(np.percentile(values, percentile))
            if percent_unit:
                legend_label = f"{label}: {val:.1f}%"
            else:
                legend_label = f"{label}: {val:.2f}"
            ax.axvline(val, color=color, linestyle=linestyle, linewidth=1.5, label=legend_label)

    def _show_rsd_plot(self):
        """Show RSD (Relative Standard Deviation) plot."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        rsd_values = DataQualityAnalysis.calculate_rsd(active_data, self.stats_data.group_info)

        if not rsd_values:
            self._show_no_data_message("No RSD data available")
            return

        groups = list(rsd_values.keys())
        n_groups = len(groups)
        group_colors = self._get_group_colors(groups)

        # Create subplots - one per group
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols

        canvas = StatisticsCanvas(self, width=12, height=4 * n_rows)
        toolbar = NavigationToolbar(canvas, self)

        canvas.fig.clear()
        axes = canvas.fig.subplots(n_rows, n_cols, squeeze=False)

        for idx, (group_name, rsd) in enumerate(rsd_values.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Filter out NaN values
            rsd_clean = rsd[~np.isnan(rsd)]

            if len(rsd_clean) > 0:
                n_bins = min(30, max(1, len(np.unique(rsd_clean))))
                try:
                    ax.hist(rsd_clean, bins=n_bins, alpha=0.7, color=group_colors[group_name], edgecolor="black")
                except ValueError:
                    ax.hist(rsd_clean, bins=1, alpha=0.7, color=group_colors[group_name], edgecolor="black")

                ax.set_xlabel("RSD (%)")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{group_name}")
                ax.grid(True, alpha=0.3, axis="y")

                self._add_percentile_lines(ax, rsd_clean, percent_unit=True)
                ax.legend(labelspacing=0.2, handlelength=1.0, handletextpad=0.4, borderpad=0.3, borderaxespad=0.3)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Relative Standard Deviation (RSD) by Group", fontweight="bold")
        _scale_plot_fonts(canvas.fig, 0.5)
        canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    def _show_abundance_histogram(self):
        """Show abundance histogram separated by group."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        groups = list(self.stats_data.group_info.keys())
        n_groups = len(groups)
        group_colors = self._get_group_colors(groups)

        # Create subplots - one per group
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols

        canvas = StatisticsCanvas(self, width=12, height=4 * n_rows)
        toolbar = NavigationToolbar(canvas, self)

        canvas.fig.clear()
        axes = canvas.fig.subplots(n_rows, n_cols, squeeze=False)

        for idx, group_name in enumerate(groups):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Get samples for this group
            group_samples = self.stats_data.group_info[group_name]
            available_cols = active_data.columns.tolist()
            matched_cols = [c for c in available_cols if c in group_samples]

            if matched_cols:
                group_data = active_data[matched_cols]
                counts, edges = DataQualityAnalysis.calculate_abundance_distribution(group_data)

                if len(counts) > 0:
                    ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", alpha=0.7, color=group_colors[group_name])
                    ax.set_xlabel("log₁₀(Intensity)")
                    ax.set_ylabel("Frequency")
                    ax.set_title(f"{group_name}")
                    ax.grid(True, alpha=0.3, axis="y")

                    all_values = group_data.values.flatten()
                    positive_values = all_values[all_values > 0]
                    if len(positive_values) > 0:
                        log_values = np.log10(positive_values)
                        self._add_percentile_lines(ax, log_values, percent_unit=False)
                        ax.legend(labelspacing=0.2, handlelength=1.0, handletextpad=0.4, borderpad=0.3, borderaxespad=0.3)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Feature Abundance Distribution by Group", fontweight="bold")
        _scale_plot_fonts(canvas.fig, 0.5)
        canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    def _show_pca(self):
        """Show PCA visualization."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        # Show group selection dialog
        groups = self.stats_data.get_group_names()
        if len(groups) > 1:
            dialog = GroupSelectionDialog(groups, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            selected_groups = dialog.get_selected_groups()
        else:
            selected_groups = groups

        if not selected_groups:
            self._show_no_data_message("No groups selected")
            return

        # Filter data to include only selected groups
        selected_samples = []
        for group in selected_groups:
            if group in self.stats_data.group_info:
                selected_samples.extend(self.stats_data.group_info[group])

        # Filter columns to include only selected samples
        available_cols = active_data.columns.tolist()
        matched_cols = [col for col in available_cols if col in selected_samples]

        logging.info(f"PCA: Selected {len(selected_samples)} samples from {len(selected_groups)} groups")
        logging.info(f"PCA: Available columns: {available_cols}")
        logging.info(f"PCA: Selected samples: {selected_samples}")
        logging.info(f"PCA: Matched columns: {matched_cols}")

        if len(matched_cols) < 2:
            self._show_no_data_message(f"Need at least 2 samples for PCA\nFound {len(matched_cols)} matching samples\nAvailable: {available_cols[:3]}\nExpected: {selected_samples[:3]}")
            return

        filtered_data = active_data[matched_cols]

        canvas = StatisticsCanvas(self, width=8, height=6)
        toolbar = NavigationToolbar(canvas, self)

        pca_result = MultivariateAnalysis.perform_pca(filtered_data)

        if pca_result.get("success", False):
            ax = canvas.axes
            scores = pca_result["scores"]
            sample_names = pca_result["sample_names"]
            var_ratio = pca_result["explained_variance_ratio"]

            # Color by group - only for selected groups
            group_colors = self._get_group_colors(selected_groups)

            # Map each sample to its group (None if it doesn't belong to any selected group)
            sample_group = {}
            for i, sample in enumerate(sample_names):
                sample_group[i] = None
                for group_name in selected_groups:
                    if group_name in self.stats_data.group_info and sample in self.stats_data.group_info[group_name]:
                        sample_group[i] = group_name
                        break

            scatter_artists: Dict[Optional[str], Any] = {}
            text_artists: Dict[int, Any] = {}
            ellipse_artists: Dict[str, Any] = {}
            original_alphas: Dict[int, float] = {}

            for group_name in list(selected_groups) + [None]:
                idxs = [i for i, g in sample_group.items() if g == group_name]
                if not idxs:
                    continue
                color = group_colors.get(group_name, "gray") if group_name else "gray"
                xs = scores[idxs, 0]
                ys = scores[idxs, 1]
                scatter_artists[group_name] = ax.scatter(xs, ys, c=[color] * len(idxs), s=100, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=3)

                for pos, i in enumerate(idxs):
                    text_artists[i] = ax.annotate(
                        sample_names[i],
                        (xs[pos], ys[pos]),
                        color="0.3",
                        fontsize=_BASE_LABEL_FONTSIZE * 0.3,
                        alpha=0.7,
                        xytext=(5, 5),
                        textcoords="offset points",
                    )

                # 95% confidence ellipse, only for groups with >= 3 samples
                if group_name is not None and len(idxs) >= 3:
                    ellipse_artists[group_name] = self._draw_confidence_ellipse(ax, xs, ys, color)

            ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")
            ax.set_title(f"PCA Score Plot ({self.stats_data.num_features_used} features)")
            ax.grid(True, alpha=0.3)

            # Remember original alphas so hovering can dim/restore them
            for artist in list(scatter_artists.values()) + list(text_artists.values()) + list(ellipse_artists.values()):
                original_alphas[id(artist)] = artist.get_alpha()

            self._pca_hover_data = {
                "ax": ax,
                "canvas": canvas,
                "scores": scores,
                "sample_group": sample_group,
                "scatter_artists": scatter_artists,
                "text_artists": text_artists,
                "ellipse_artists": ellipse_artists,
                "original_alphas": original_alphas,
                "current_hover": "__unset__",
            }
            canvas.mpl_connect("motion_notify_event", self._on_pca_hover)
            canvas.mpl_connect("axes_leave_event", lambda event: self._pca_apply_hover(None))

            _scale_plot_fonts(canvas.fig, 0.5)
            canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    @staticmethod
    def _draw_confidence_ellipse(ax, xs: np.ndarray, ys: np.ndarray, color) -> Any:
        """Draw a 95% confidence ellipse for a group's PCA scores and return the patch."""
        from matplotlib.patches import Ellipse
        from scipy.stats import chi2

        cov = np.cov(xs, ys)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        chi2_val = chi2.ppf(0.95, df=2)
        width, height = 2 * np.sqrt(np.maximum(eigenvalues, 0) * chi2_val)
        ellipse = Ellipse((np.mean(xs), np.mean(ys)), width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.15, linewidth=1.5, zorder=1)
        ax.add_patch(ellipse)
        return ellipse

    def _on_pca_hover(self, event):
        """Dim samples/ellipses/labels of every group other than the one being hovered over."""
        data = getattr(self, "_pca_hover_data", None)
        if data is None or event.inaxes != data["ax"] or event.xdata is None or event.ydata is None:
            self._pca_apply_hover(None)
            return

        scores = data["scores"]
        ax = data["ax"]
        # Transform both the pointer and the sample points through the same ax.transData so the
        # comparison stays correct regardless of any HiDPI mismatch between event.x/y and transData pixels.
        click_display = ax.transData.transform((event.xdata, event.ydata))
        pts_display = ax.transData.transform(scores[:, :2])
        dists = np.sqrt(np.sum((pts_display - click_display) ** 2, axis=1))
        closest = int(np.argmin(dists))
        hovered_group = data["sample_group"].get(closest) if dists[closest] <= 15 else None
        self._pca_apply_hover(hovered_group)

    def _pca_apply_hover(self, hovered_group: Optional[str]):
        """Set alpha to 30% for every artist not belonging to `hovered_group` (None = show all normally)."""
        data = getattr(self, "_pca_hover_data", None)
        if data is None or data["current_hover"] == hovered_group:
            return
        data["current_hover"] = hovered_group

        for group_name, artist in data["scatter_artists"].items():
            dim = hovered_group is not None and group_name != hovered_group
            artist.set_alpha(0.3 if dim else data["original_alphas"][id(artist)])
        for i, artist in data["text_artists"].items():
            dim = hovered_group is not None and data["sample_group"].get(i) != hovered_group
            artist.set_alpha(0.3 if dim else data["original_alphas"][id(artist)])
        # Ellipses are intentionally left untouched by hover; dimming them didn't work reliably.

        data["canvas"].draw_idle()

    def _show_hca(self):
        """Show HCA dendrogram."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        # Show group selection dialog
        groups = self.stats_data.get_group_names()
        if len(groups) > 1:
            dialog = GroupSelectionDialog(groups, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            selected_groups = dialog.get_selected_groups()
        else:
            selected_groups = groups

        if not selected_groups:
            self._show_no_data_message("No groups selected")
            return

        # Filter data to include only selected groups
        selected_samples = []
        for group in selected_groups:
            if group in self.stats_data.group_info:
                selected_samples.extend(self.stats_data.group_info[group])

        available_cols = active_data.columns.tolist()
        matched_cols = [col for col in available_cols if col in selected_samples]

        if len(matched_cols) < 2:
            self._show_no_data_message(f"Need at least 2 samples for HCA\nFound {len(matched_cols)} matching samples")
            return

        filtered_data = active_data[matched_cols]
        canvas = StatisticsCanvas(self, width=10, height=6)
        toolbar = NavigationToolbar(canvas, self)

        hca_result = MultivariateAnalysis.perform_hca(filtered_data)

        if hca_result.get("success", False):
            ax = canvas.axes
            dendrogram(hca_result["linkage_matrix"], labels=hca_result["sample_names"], ax=ax, leaf_rotation=90)
            ax.set_ylabel("Distance")
            ax.set_title(f"Hierarchical Cluster Analysis ({self.stats_data.num_features_used} features)")
            _scale_plot_fonts(canvas.fig, 0.5)
            canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    def _show_heatmap(self):
        """Show heat map visualization."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        # Show group selection dialog
        groups = self.stats_data.get_group_names()
        if len(groups) > 1:
            dialog = GroupSelectionDialog(groups, self)
            if dialog.exec_() != QDialog.Accepted:
                return
            selected_groups = dialog.get_selected_groups()
        else:
            selected_groups = groups

        if not selected_groups:
            self._show_no_data_message("No groups selected")
            return

        # Filter data to include only selected groups
        selected_samples = []
        for group in selected_groups:
            if group in self.stats_data.group_info:
                selected_samples.extend(self.stats_data.group_info[group])

        available_cols = active_data.columns.tolist()
        matched_cols = [col for col in available_cols if col in selected_samples]

        if len(matched_cols) < 2:
            self._show_no_data_message(f"Need at least 2 samples for heatmap\nFound {len(matched_cols)} matching samples")
            return

        # Reorder columns so samples are grouped by experimental group, for the heatmap layout
        ordered_cols, group_segments = self._group_heatmap_columns(matched_cols, selected_groups)
        filtered_data = active_data[ordered_cols]

        heatmap_result = MultivariateAnalysis.prepare_heatmap_data(filtered_data)

        if not heatmap_result.get("success", False):
            self._show_no_data_message(heatmap_result.get("error", "Could not compute heatmap"))
            return

        data = heatmap_result["data"]
        # Order features by variance (descending) so pagination shows the most informative features first
        feature_order = np.argsort(np.var(data, axis=1))[::-1]

        self._heatmap_state = {
            "data": data,
            "feature_order": feature_order,
            "col_names": heatmap_result["col_names"],
            "group_segments": group_segments,
            "group_colors": self._get_group_colors(selected_groups),
            "page": 0,
            "page_size": 100,
        }
        self._render_heatmap_page()

    def _group_heatmap_columns(self, matched_cols: List[str], selected_groups: List[str]) -> Tuple[List[str], List[Tuple[Optional[str], int, int]]]:
        """Reorder sample columns so each group's samples are contiguous.

        Returns the reordered column list and a list of (group_name, start, end) segments
        (end exclusive) describing which columns belong to which group in the new order.
        """
        ordered_cols: List[str] = []
        segments: List[Tuple[Optional[str], int, int]] = []
        for group_name in selected_groups:
            group_samples = self.stats_data.group_info.get(group_name, [])
            cols_in_group = [c for c in matched_cols if c in group_samples]
            if not cols_in_group:
                continue
            start = len(ordered_cols)
            ordered_cols.extend(cols_in_group)
            segments.append((group_name, start, len(ordered_cols)))

        # Samples that don't belong to any selected group (shouldn't normally happen)
        remaining = [c for c in matched_cols if c not in ordered_cols]
        if remaining:
            start = len(ordered_cols)
            ordered_cols.extend(remaining)
            segments.append((None, start, len(ordered_cols)))

        return ordered_cols, segments

    @staticmethod
    def _insert_heatmap_spacers(data: np.ndarray, segments: List[Tuple[Optional[str], int, int]], spacer_width: int = 1) -> Tuple[np.ndarray, List[Tuple[Optional[str], int, int]]]:
        """Insert NaN spacer columns between group segments for visual separation.

        Returns the new data array and segments adjusted to the new column positions.
        """
        pieces = []
        new_segments = []
        col_cursor = 0
        for i, (group_name, start, end) in enumerate(segments):
            seg_data = data[:, start:end]
            pieces.append(seg_data)
            new_start = col_cursor
            col_cursor += seg_data.shape[1]
            new_segments.append((group_name, new_start, col_cursor))
            if i < len(segments) - 1:
                pieces.append(np.full((data.shape[0], spacer_width), np.nan))
                col_cursor += spacer_width
        new_data = np.concatenate(pieces, axis=1) if pieces else data
        return new_data, new_segments

    def _render_heatmap_page(self):
        """Render the current page (up to 100 features) of the heatmap."""
        state = self._heatmap_state
        if state is None:
            return

        self._clear_visualization()

        data = state["data"]
        feature_order = state["feature_order"]
        page = state["page"]
        page_size = state["page_size"]
        total_features = len(feature_order)

        start = page * page_size
        end = min(start + page_size, total_features)
        page_feature_indices = feature_order[start:end]
        page_data = data[page_feature_indices, :]

        plot_data, segments = self._insert_heatmap_spacers(page_data, state["group_segments"])

        canvas = StatisticsCanvas(self, width=10, height=8)
        toolbar = NavigationToolbar(canvas, self)
        ax = canvas.axes

        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad(color="white")
        im = ax.imshow(plot_data, aspect="auto", cmap=cmap, interpolation="nearest")
        canvas.fig.colorbar(im, ax=ax, label="Z-score")

        ax.set_xlabel("Samples")
        ax.set_ylabel("Features")
        ax.set_title(f"Feature Heat Map (features {start + 1}-{end} of {total_features})")
        ax.set_xticks([])

        # Label each group's section once, in the group's color, above the plot
        group_colors = state["group_colors"]
        for group_name, seg_start, seg_end in segments:
            if group_name is None:
                continue
            mid = (seg_start + seg_end - 1) / 2
            ax.text(mid, 1.02, group_name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=group_colors.get(group_name, "black"), rotation=90)

        _scale_plot_fonts(canvas.fig, 0.5)
        canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

        # Pagination controls
        self.heatmap_page_label.setText(f"Features {start + 1}-{end} of {total_features}")
        self.heatmap_prev_btn.setEnabled(page > 0)
        self.heatmap_next_btn.setEnabled(end < total_features)
        self.heatmap_pagination_bar.setVisible(True)

    def _heatmap_prev_page(self):
        if self._heatmap_state is None or self._heatmap_state["page"] <= 0:
            return
        self._heatmap_state["page"] -= 1
        self._render_heatmap_page()

    def _heatmap_next_page(self):
        if self._heatmap_state is None:
            return
        state = self._heatmap_state
        max_page = (len(state["feature_order"]) - 1) // state["page_size"]
        if state["page"] >= max_page:
            return
        state["page"] += 1
        self._render_heatmap_page()

    def _show_all_volcano_plots(self):
        """Show all volcano plots simultaneously."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None or not self.stats_data.volcano_comparisons:
            self._show_no_data_message("No data or no comparisons defined")
            return

        self.multi_volcano_widget = MultiVolcanoWidget(self)
        self.multi_volcano_widget.idsFilterRequested.connect(self.idsFilterRequested)
        self.multi_volcano_widget.featureSelected.connect(self._on_features_selected)

        data = {"feature_data": active_data, "group_info": self.stats_data.group_info}

        self.multi_volcano_widget.set_comparisons(self.stats_data.volcano_comparisons, data)
        self.multi_volcano_widget.set_id_filter(self._active_id_filter_visible_nums)
        self.viz_layout.addWidget(self.multi_volcano_widget)

        # Store volcano data for the first comparison (for feature table)
        if self.stats_data.volcano_comparisons:
            group1, group2 = self.stats_data.volcano_comparisons[0]
            if group1 in self.stats_data.group_info and group2 in self.stats_data.group_info:
                self.current_volcano_data = UnivariateAnalysis.calculate_volcano_data(active_data, self.stats_data.group_info[group1], self.stats_data.group_info[group2], metadata=self.stats_data.feature_metadata)

                # Populate the features table with all features
                self._populate_features_table()

    def _show_single_volcano_plot(self, item: QTreeWidgetItem):
        """Show a single volcano plot for the selected comparison."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None:
            self._show_no_data_message()
            return

        # Parse comparison from item text
        text = item.text(0)
        parts = text.split(" vs ")
        if len(parts) != 2:
            return

        group1, group2 = parts[0], parts[1]

        if group1 not in self.stats_data.group_info or group2 not in self.stats_data.group_info:
            return

        canvas = InteractiveVolcanoCanvas(self, width=8, height=6)
        canvas.selectionChanged.connect(lambda indices, additive: self.selection_manager.add_selection(indices, additive))
        canvas.idsFilterRequested.connect(self.idsFilterRequested)
        toolbar = NavigationToolbar(canvas, self)

        volcano_data = UnivariateAnalysis.calculate_volcano_data(active_data, self.stats_data.group_info[group1], self.stats_data.group_info[group2], metadata=self.stats_data.feature_metadata)
        self.current_volcano_data = volcano_data  # Store for feature table

        canvas.set_volcano_data(volcano_data)
        canvas.set_id_filter(self._active_id_filter_visible_nums)

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

        # Populate the features table with all features
        self._populate_features_table()

    def _show_no_data_message(self, message: str = "No experiment data loaded"):
        """Show a message when no data is available."""
        label = QLabel(message)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: gray; font-size: 14px;")
        self.viz_layout.addWidget(label)

    def _add_volcano_comparison(self):
        """Add a new volcano plot comparison."""
        groups = self.stats_data.get_group_names()
        if len(groups) < 2:
            QtWidgets.QMessageBox.warning(self, "Cannot Add Comparison", "At least 2 groups are required for comparison.")
            return

        dialog = AddComparisonDialog(groups, self)
        if dialog.exec_() == QDialog.Accepted:
            group1, group2 = dialog.get_groups()
            if group1 == group2:
                QtWidgets.QMessageBox.warning(self, "Invalid Comparison", "Cannot compare a group with itself.")
                return

            if self.stats_data.add_volcano_comparison(group1, group2):
                # Add to tree
                QTreeWidgetItem(self.volcano_parent_item, [f"{group1} vs {group2}"])
                self.methods_tree.expandItem(self.volcano_parent_item)

    def _remove_volcano_comparison(self):
        """Remove the selected volcano comparison."""
        current_item = self.methods_tree.currentItem()
        if current_item and current_item.parent() == self.volcano_parent_item:
            text = current_item.text(0)
            if text != "All Comparisons":
                # Find and remove from data
                parts = text.split(" vs ")
                if len(parts) == 2:
                    for i, (g1, g2) in enumerate(self.stats_data.volcano_comparisons):
                        if g1 == parts[0] and g2 == parts[1]:
                            self.stats_data.remove_volcano_comparison(i)
                            break

                # Remove from tree
                self.volcano_parent_item.removeChild(current_item)

    def _on_features_selected(self, indices: List[int]):
        """Handle feature selection from volcano plots."""
        self.selection_manager.add_selection(indices, False)

    def _populate_features_table(self):
        """Populate the features table with all features from the current volcano data."""
        if self.current_volcano_data is None or not self.current_volcano_data.get("success", False):
            return
        vd = self.current_volcano_data

        def _to_list(v):
            return v.tolist() if hasattr(v, "tolist") else list(v)

        feature_names = _to_list(vd.get("feature_names", []))
        log2_fc_array = _to_list(vd.get("log2_fc", []))
        pvalues_array = _to_list(vd.get("pvalues", []))
        significant = _to_list(vd.get("significant", []))
        feature_pair_ids = _to_list(vd.get("featurePairIDs", []))
        feature_group_ids = _to_list(vd.get("featureGroupIDs", []))
        g1_means = _to_list(vd.get("g1_means", []))
        g1_medians = _to_list(vd.get("g1_medians", []))
        g1_sds = _to_list(vd.get("g1_sds", []))
        g2_means = _to_list(vd.get("g2_means", []))
        g2_medians = _to_list(vd.get("g2_medians", []))
        g2_sds = _to_list(vd.get("g2_sds", []))

        metadata = {}
        if self.stats_data.feature_metadata is not None:
            metadata = self.stats_data.feature_metadata.to_dict()
        metadata["log2_fc"] = {}
        metadata["pvalue"] = {}
        metadata["featurePairID"] = {}
        metadata["featureGroupID"] = {}
        group_stats = {"g1_mean": {}, "g1_median": {}, "g1_sd": {}, "g2_mean": {}, "g2_median": {}, "g2_sd": {}}

        COLOR_GRAY = QtGui.QColor(190, 190, 190, 60)
        COLOR_RED = QtGui.QColor(220, 80, 80, 70)
        COLOR_BLUE = QtGui.QColor(80, 80, 220, 70)
        row_colors = {}

        for pos_idx, feature_id in enumerate(feature_names):
            fc = log2_fc_array[pos_idx] if pos_idx < len(log2_fc_array) else 0.0
            pval = pvalues_array[pos_idx] if pos_idx < len(pvalues_array) else 1.0
            sig = significant[pos_idx] if pos_idx < len(significant) else False
            metadata["log2_fc"][feature_id] = fc
            metadata["pvalue"][feature_id] = pval
            metadata["featurePairID"][feature_id] = feature_pair_ids[pos_idx] if pos_idx < len(feature_pair_ids) else 0
            metadata["featureGroupID"][feature_id] = feature_group_ids[pos_idx] if pos_idx < len(feature_group_ids) else 0
            group_stats["g1_mean"][feature_id] = g1_means[pos_idx] if pos_idx < len(g1_means) else np.nan
            group_stats["g1_median"][feature_id] = g1_medians[pos_idx] if pos_idx < len(g1_medians) else np.nan
            group_stats["g1_sd"][feature_id] = g1_sds[pos_idx] if pos_idx < len(g1_sds) else np.nan
            group_stats["g2_mean"][feature_id] = g2_means[pos_idx] if pos_idx < len(g2_means) else np.nan
            group_stats["g2_median"][feature_id] = g2_medians[pos_idx] if pos_idx < len(g2_medians) else np.nan
            group_stats["g2_sd"][feature_id] = g2_sds[pos_idx] if pos_idx < len(g2_sds) else np.nan
            if sig:
                row_colors[feature_id] = COLOR_RED if fc > 0 else COLOR_BLUE
            else:
                row_colors[feature_id] = COLOR_GRAY

        self._updating_from_volcano = True
        try:
            self.features_table.update_features(feature_names, metadata, group_stats, row_colors=row_colors)
        finally:
            self._updating_from_volcano = False

    def _select_table_rows(self, feature_ids: List[int]):
        """Select tree rows whose feature index matches one of feature_ids."""
        self._updating_from_volcano = True
        try:
            self.features_table.select_feature_ids(feature_ids)
        finally:
            self._updating_from_volcano = False

    def _on_selection_changed(self, indices: List[int]):
        """Handle selection manager updates: select matching table rows, highlight volcano dots,
        and automatically show the (first) selected feature in the Experiment results pane."""
        feature_ids = indices
        if self.current_volcano_data and self.current_volcano_data.get("success", False):
            feature_names = self.current_volcano_data.get("feature_names", [])
            if feature_names:
                feature_ids = [feature_names[i] for i in indices if i < len(feature_names)]

        self._select_table_rows(feature_ids)

        if feature_ids:
            self.showFeatureInExperiment.emit(feature_ids[0])

        # Highlight dots in the volcano plot(s)
        if self.current_canvas is not None:
            self.current_canvas.update_highlighting(indices)
        if self.multi_volcano_widget is not None:
            self.multi_volcano_widget.update_highlighting(indices)

    def _on_table_row_selected(self):
        """Highlight volcano dots corresponding to table rows selected by the user, and
        show the (first) selected feature in the Experiment results pane."""
        if self._updating_from_volcano:
            return
        if self.current_volcano_data is None or not self.current_volcano_data.get("success", False):
            return
        feature_names = self.current_volcano_data.get("feature_names", [])
        if not feature_names:
            return
        # Build a reverse map: feature_id -> positional index
        feature_id_to_pos = {fid: pos for pos, fid in enumerate(feature_names)}
        selected_feature_ids = self.features_table.get_selected_feature_ids()
        pos_indices = []
        for feature_id in selected_feature_ids:
            pos = feature_id_to_pos.get(feature_id)
            if pos is not None:
                pos_indices.append(pos)
        canvas = self.current_canvas
        if canvas is not None:
            canvas.update_highlighting(pos_indices)
        if self.multi_volcano_widget is not None:
            self.multi_volcano_widget.update_highlighting(pos_indices)

        if selected_feature_ids:
            self.showFeatureInExperiment.emit(selected_feature_ids[0])

    def highlight_features_by_id(self, feature_ids: List[int]):
        """Highlight features selected in the Experiment results pane in the currently shown volcano plot(s)
        and select the matching row(s) in the Selected Features table underneath.

        Uses the `_updating_from_volcano` guard when touching the table so this one-way sync from
        Experiment results can never trigger a call back into Experiment results.
        """
        if self.current_volcano_data is None or not self.current_volcano_data.get("success", False):
            return
        feature_names = self.current_volcano_data.get("feature_names", [])
        if not feature_names:
            return
        feature_id_to_pos = {fid: pos for pos, fid in enumerate(feature_names)}
        pos_indices = [feature_id_to_pos[fid] for fid in feature_ids if fid in feature_id_to_pos]

        if self.current_canvas is not None and hasattr(self.current_canvas, "update_highlighting"):
            self.current_canvas.update_highlighting(pos_indices)
        if self.multi_volcano_widget is not None:
            self.multi_volcano_widget.update_highlighting(pos_indices)

        self._select_table_rows(feature_ids)

    def set_id_filter_state(self, visible_nums: Optional[set]):
        """Called by MExtract whenever the Experiment results filters change. `visible_nums` is
        the set of feature Nums still visible in the tree (None = no filter active, show all
        dots normally); dims every other dot to 10% alpha in whichever volcano plot(s) are shown."""
        self._active_id_filter_visible_nums = visible_nums
        if self.current_canvas is not None and hasattr(self.current_canvas, "set_id_filter"):
            self.current_canvas.set_id_filter(visible_nums)
        if self.multi_volcano_widget is not None:
            self.multi_volcano_widget.set_id_filter(visible_nums)

    def _on_view_feature_requested(self, feature_index: int, target: str):
        """Handle request to view feature in experiment results."""
        self.showFeatureInExperiment.emit(feature_index)

    def _calculate_group_statistics(self, feature_ids: List[int], feature_names: List[int]) -> Dict[str, Dict[int, float]]:
        """Calculate mean, median, and SD for both groups in the current volcano comparison."""
        if not self.stats_data.volcano_comparisons:
            return None

        # Get the first volcano comparison groups (or use the active one)
        group1, group2 = self.stats_data.volcano_comparisons[0]

        # Get active data
        active_data = self.stats_data.get_active_data()
        if active_data is None:
            return None

        # Get group samples
        group1_samples = self.stats_data.group_info.get(group1, [])
        group2_samples = self.stats_data.group_info.get(group2, [])

        # Filter to available columns
        g1_cols = [col for col in active_data.columns if col in group1_samples]
        g2_cols = [col for col in active_data.columns if col in group2_samples]

        # Initialize result dictionaries
        group_stats = {"g1_mean": {}, "g1_median": {}, "g1_sd": {}, "g2_mean": {}, "g2_median": {}, "g2_sd": {}}

        # Calculate statistics for each selected feature
        for feature_id in feature_ids:
            # Match feature_id with the row in active_data
            # feature_id should be in active_data.index
            if feature_id in active_data.index:
                feature_row = active_data.loc[feature_id]

                # Group 1 statistics
                if g1_cols:
                    g1_values = feature_row[g1_cols].values
                    g1_values_valid = g1_values[g1_values > 0]  # Filter out zeros
                    if len(g1_values_valid) > 0:
                        group_stats["g1_mean"][feature_id] = np.mean(g1_values_valid)
                        group_stats["g1_median"][feature_id] = np.median(g1_values_valid)
                        group_stats["g1_sd"][feature_id] = np.std(g1_values_valid, ddof=1) if len(g1_values_valid) > 1 else 0.0
                    else:
                        group_stats["g1_mean"][feature_id] = "N/A"
                        group_stats["g1_median"][feature_id] = "N/A"
                        group_stats["g1_sd"][feature_id] = "N/A"

                # Group 2 statistics
                if g2_cols:
                    g2_values = feature_row[g2_cols].values
                    g2_values_valid = g2_values[g2_values > 0]  # Filter out zeros
                    if len(g2_values_valid) > 0:
                        group_stats["g2_mean"][feature_id] = np.mean(g2_values_valid)
                        group_stats["g2_median"][feature_id] = np.median(g2_values_valid)
                        group_stats["g2_sd"][feature_id] = np.std(g2_values_valid, ddof=1) if len(g2_values_valid) > 1 else 0.0
                    else:
                        group_stats["g2_mean"][feature_id] = "N/A"
                        group_stats["g2_median"][feature_id] = "N/A"
                        group_stats["g2_sd"][feature_id] = "N/A"

        return group_stats
