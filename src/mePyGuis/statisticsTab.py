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
    QGridLayout,
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
        """Handle rectangle selection."""
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

        # Check if Ctrl is pressed for additive selection
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        additive = modifiers == Qt.ControlModifier

        self.selectionChanged.emit(selected_indices, additive)

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
            self.axes.text(0.5, 0.5, f"Error:\n{error_msg}", ha="center", va="center", transform=self.axes.transAxes, fontsize=10, color="red", wrap=True)
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
            self.axes.text(0.5, 0.5, "No features to plot", ha="center", va="center", transform=self.axes.transAxes, fontsize=12, color="gray")
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

        self.scatter = self.axes.scatter(log2_fc, neg_log10_pval, c=colors, alpha=0.7, s=30, edgecolors="none")

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
        self.axes.set_title("Volcano Plot")

        if old_xlim is not None and old_ylim is not None:
            self.axes.set_xlim(old_xlim)
            self.axes.set_ylim(old_ylim)

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
    """Widget for displaying multiple volcano plots simultaneously."""

    featureSelected = Signal(list)  # Signal when features are selected

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.volcano_canvases: List[InteractiveVolcanoCanvas] = []
        self.selection_manager = SelectionManager()
        self.selection_manager.register_callback(self._on_selection_changed)

    def set_comparisons(self, comparisons: List[Tuple[str, str]], data: Dict[str, Any]):
        """
        Set up volcano plots for all comparisons.

        Args:
            comparisons: List of (group1, group2) tuples
            data: Dictionary containing feature data and group info
        """
        # Clear layout - this deletes container widgets and their children (including canvases)
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset the canvases tracking list
        self.volcano_canvases.clear()

        if not comparisons:
            return

        # Calculate grid layout
        n_plots = len(comparisons)
        cols = min(3, n_plots)
        (n_plots + cols - 1) // cols

        for i, (group1, group2) in enumerate(comparisons):
            canvas = InteractiveVolcanoCanvas(self)
            canvas.selectionChanged.connect(self._handle_selection)

            row = i // cols
            col = i % cols

            # Create container with label
            container = QFrame()
            container_layout = QVBoxLayout(container)
            label = QLabel(f"{group1} vs {group2}")
            label.setAlignment(Qt.AlignCenter)
            container_layout.addWidget(label)
            container_layout.addWidget(canvas)

            self.layout.addWidget(container, row, col)
            self.volcano_canvases.append(canvas)

            # Calculate volcano data for this comparison
            if "feature_data" in data and "group_info" in data:
                feature_data = data["feature_data"]
                group_info = data["group_info"]

                if group1 in group_info and group2 in group_info:
                    metadata = data.get("metadata") if "metadata" in data else None
                    volcano_data = UnivariateAnalysis.calculate_volcano_data(feature_data, group_info[group1], group_info[group2], metadata=metadata)
                    canvas.set_volcano_data(volcano_data)

    def _handle_selection(self, indices: List[int], additive: bool):
        """Handle selection from any volcano plot."""
        self.selection_manager.add_selection(indices, additive)
        self.featureSelected.emit(self.selection_manager.get_selected_indices())

    def _on_selection_changed(self, indices: List[int]):
        """Update all volcano plots when selection changes."""
        for canvas in self.volcano_canvases:
            canvas.update_highlighting(indices)


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stats_data = StatisticsData()
        self.selection_manager = SelectionManager()
        self._updating_from_volcano = False  # Guard against circular table↔volcano sync

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Left panel: Tree view for method selection
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

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

        main_layout.addWidget(left_panel)

        # Right panel: Content area with splitter
        right_splitter = QSplitter(Qt.Vertical)

        # Visualization area
        self.viz_container = QScrollArea()
        self.viz_container.setWidgetResizable(True)
        self.viz_widget = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_widget)
        self.viz_container.setWidget(self.viz_widget)
        right_splitter.addWidget(self.viz_container)

        # Selected features table
        table_container = QFrame()
        table_layout = QVBoxLayout(table_container)
        table_layout.addWidget(QLabel("<b>Selected Features</b>"))
        self.features_table = SelectedFeaturesTable()
        table_layout.addWidget(self.features_table)

        view_feature_btn = QPushButton("View Selected in Experiment Results")
        view_feature_btn.clicked.connect(self._view_selected_feature)
        table_layout.addWidget(view_feature_btn)

        right_splitter.addWidget(table_container)
        right_splitter.setSizes([600, 200])

        main_layout.addWidget(right_splitter)

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
        """Get consistent colors for groups based on original group order.

        Args:
            groups: List of group names to get colors for. If None, uses all groups from stats_data.

        Returns:
            Dictionary mapping group names to colors
        """
        # Always base colors on the full original group list to ensure consistency
        all_groups = self.stats_data.get_group_names()
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_groups), 3)))  # min 3 to avoid edge cases

        # Create mapping for all groups
        all_group_colors = {}
        for idx, group_name in enumerate(all_groups):
            all_group_colors[group_name] = colors[idx]

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
            ax.set_xlabel("Number of replicates with detection", fontsize=8)
            ax.set_ylabel("Number of features", fontsize=8)
            ax.set_title(f"{group_name}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Feature Detection Counts by Group", fontsize=11, fontweight="bold")
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

                ax.set_xlabel("RSD (%)", fontsize=8)
                ax.set_ylabel("Frequency", fontsize=8)
                ax.set_title(f"{group_name}", fontsize=9)
                ax.tick_params(labelsize=7)
                ax.grid(True, alpha=0.3, axis="y")

                self._add_percentile_lines(ax, rsd_clean, percent_unit=True)
                ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Relative Standard Deviation (RSD) by Group", fontsize=11, fontweight="bold")
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
                    ax.set_xlabel("log₁₀(Intensity)", fontsize=8)
                    ax.set_ylabel("Frequency", fontsize=8)
                    ax.set_title(f"{group_name}", fontsize=9)
                    ax.tick_params(labelsize=7)
                    ax.grid(True, alpha=0.3, axis="y")

                    all_values = group_data.values.flatten()
                    positive_values = all_values[all_values > 0]
                    if len(positive_values) > 0:
                        log_values = np.log10(positive_values)
                        self._add_percentile_lines(ax, log_values, percent_unit=False)
                        ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(n_groups, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        canvas.fig.suptitle("Feature Abundance Distribution by Group", fontsize=11, fontweight="bold")
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

            for i, sample in enumerate(sample_names):
                sample_color = "gray"
                for group_name in selected_groups:
                    if group_name in self.stats_data.group_info:
                        samples = self.stats_data.group_info[group_name]
                        if sample in samples:
                            sample_color = group_colors[group_name]
                            break

                ax.scatter(scores[i, 0], scores[i, 1], c=sample_color, s=100, alpha=0.8, edgecolors="black", linewidth=0.5)
                ax.annotate(sample, (scores[i, 0], scores[i, 1]), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords="offset points")

            ax.set_xlabel(f"PC1 ({var_ratio[0] * 100:.1f}%)")
            ax.set_ylabel(f"PC2 ({var_ratio[1] * 100:.1f}%)")
            ax.set_title(f"PCA Score Plot ({self.stats_data.num_features_used} features)")
            ax.grid(True, alpha=0.3)

            # Add legend outside the plot area on the right side
            legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=group) for group, color in group_colors.items()]
            ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
            canvas.fig.tight_layout(rect=[0, 0, 0.82, 1])

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

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

        filtered_data = active_data[matched_cols]
        canvas = StatisticsCanvas(self, width=10, height=8)
        toolbar = NavigationToolbar(canvas, self)

        heatmap_result = MultivariateAnalysis.prepare_heatmap_data(filtered_data)

        if heatmap_result.get("success", False):
            ax = canvas.axes
            data = heatmap_result["data"]

            # Limit number of features for visualization
            max_features = 100
            if data.shape[0] > max_features:
                # Select top variable features
                var = np.var(data, axis=1)
                top_indices = np.argsort(var)[-max_features:]
                data = data[top_indices, :]

            im = ax.imshow(data, aspect="auto", cmap="RdBu_r", interpolation="nearest")
            canvas.fig.colorbar(im, ax=ax, label="Z-score")

            ax.set_xlabel("Samples")
            ax.set_ylabel("Features")
            n_features_displayed = min(data.shape[0], 100)
            ax.set_title(f"Feature Heat Map ({n_features_displayed}/{self.stats_data.num_features_used} features)")

            # Set sample labels if not too many
            if len(heatmap_result["col_names"]) <= 20:
                ax.set_xticks(range(len(heatmap_result["col_names"])))
                ax.set_xticklabels(heatmap_result["col_names"], rotation=90, fontsize=8)

            canvas.fig.tight_layout()

        self.viz_layout.addWidget(toolbar)
        self.viz_layout.addWidget(canvas)
        self.current_canvas = canvas
        self.current_toolbar = toolbar

    def _show_all_volcano_plots(self):
        """Show all volcano plots simultaneously."""
        self._clear_visualization()

        active_data = self.stats_data.get_active_data()
        if active_data is None or not self.stats_data.volcano_comparisons:
            self._show_no_data_message("No data or no comparisons defined")
            return

        self.multi_volcano_widget = MultiVolcanoWidget(self)
        self.multi_volcano_widget.featureSelected.connect(self._on_features_selected)

        data = {"feature_data": active_data, "group_info": self.stats_data.group_info}

        self.multi_volcano_widget.set_comparisons(self.stats_data.volcano_comparisons, data)
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
        toolbar = NavigationToolbar(canvas, self)

        volcano_data = UnivariateAnalysis.calculate_volcano_data(active_data, self.stats_data.group_info[group1], self.stats_data.group_info[group2], metadata=self.stats_data.feature_metadata)
        self.current_volcano_data = volcano_data  # Store for feature table

        canvas.set_volcano_data(volcano_data)

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
        """Handle selection manager updates: select matching table rows and highlight volcano dots."""
        feature_ids = indices
        if self.current_volcano_data and self.current_volcano_data.get("success", False):
            feature_names = self.current_volcano_data.get("feature_names", [])
            if feature_names:
                feature_ids = [feature_names[i] for i in indices if i < len(feature_names)]

        self._select_table_rows(feature_ids)

        # Highlight dots in the volcano plot
        if self.current_canvas is not None:
            self.current_canvas.update_highlighting(indices)

    def _on_table_row_selected(self):
        """Highlight volcano dots corresponding to table rows selected by the user."""
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
        if hasattr(self, "multi_volcano_widget") and self.multi_volcano_widget is not None:
            for vc in self.multi_volcano_widget.volcano_canvases:
                vc.update_highlighting(pos_indices)

    def _on_view_feature_requested(self, feature_index: int, target: str):
        """Handle request to view feature in experiment results."""
        self.showFeatureInExperiment.emit(feature_index)

    def _view_selected_feature(self):
        """View the currently selected feature in experiment results."""
        feature_id = self.features_table.get_current_feature_pair_id()
        if feature_id is None:
            QtWidgets.QMessageBox.information(self, "No Selection", "Please select a feature in the table first.")
            return
        self.showFeatureInExperiment.emit(feature_id)

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
