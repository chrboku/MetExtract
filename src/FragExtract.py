#    MetExtract II
#    Copyright (C) 2015
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from __future__ import absolute_import, division, print_function
import logging
from . import LoggingSetup
import base64
import os
import os.path
import pickle
import re
import sqlite3
import time
from copy import deepcopy
from multiprocessing import Manager, Pool, cpu_count, freeze_support
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets
from . import formulaTools, pyperclip
from .utilities.AboutDialog import AboutDialog
from .Chromatogram import Chromatogram
from .FragExtract_processTarget import ProcessTarget
from .mePyGuis.adductsEdit import ConfiguredAdduct, ConfiguredElement, adductsEdit
from .mePyGuis.FE_mainWindow import Ui_MainWindow
from .mePyGuis.ProgressWrapper import ProgressWrapper
from .MExtract import MetExtractVersion
from .utils import Bunch, CallBackMethod, SQLSelectAsObject, get_main_dir, is_float, is_int

LoggingSetup.LoggingSetup.Instance().initLogging()


# <editor-fold desc="### MatPlotLib imports and setup">


matplotlib.rcParams["savefig.dpi"] = 300
font = {"size": 16}
matplotlib.rc("font", **font)
globAlpha = 0.05
predefinedColors = [
    "DarkSlateGrey",
    "FireBrick",
    "YellowGreen",
    "SteelBlue",
    "DarkOrange",
    "Teal",
    "CornflowerBlue",
    "DarkOliveGreen",
    "SlateGrey",
    "CadetBlue",
    "DarkCyan",
    "Black",
    "DarkSeaGreen",
    "DimGray",
    "GoldenRod",
    "LightBlue",
    "MediumTurquoise",
    "RoyalBlue",
]

_ft = formulaTools.formulaTools()


# required for Plotting table
def convertXaToX(x, a):
    return ((1 - a) * 1) + (a * x)


# taken from http://www.scipy.org/Cookbook/Matplotlib/Transformations
# New enough versions have offset_copy (by Eric Firing):
if "offset_copy" in dir(matplotlib.transforms):
    from matplotlib.transforms import offset_copy

    def offset(ax, x, y):
        return offset_copy(ax.transData, x=x, y=y, units="dots")
else:  # Without offset_copy we have to do some black transform magic
    from matplotlib.transforms import blend_xy_sep_transform, identity_transform

    def offset(ax, x, y):
        # This trick makes a shallow copy of ax.transData (but fails for polar plots):
        trans = blend_xy_sep_transform(ax.transData, ax.transData)
        # Now we set the offset in pixels
        trans.set_offset((x, y), identity_transform())
        return trans


# show only bottom and left axes in a matplotlib graph
def simpleaxis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# remove all axes in a matplotlib graph
def noaxis(ax):
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axis("off")


# from mpltools import style
# style.use('ggplot')
# </editor-fold>


def clearPlot(plt, setXtoZero=False):
    for ax in plt.twinxs:
        plt.fig.delaxes(ax)
    plt.axes = plt.fig.add_subplot(111)

    simpleaxis(plt.axes)
    if setXtoZero:
        plt.axes.spines["bottom"].set_position("zero")

    plt.axes.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.twinxs = [plt.axes]


def drawCanvas(plt, ylim=None, xlim=None):
    for ax in plt.twinxs:
        if ylim is not None:
            if len(ylim) == 1:
                ax.set_ylim(ylim[0])
            elif len(ylim) == 2:
                ax.set_ylim(ylim[0], ylim[1])
        if xlim is not None:
            if len(xlim) == 1:
                ax.set_xlim(xlim[0])
            else:
                ax.set_xlim(xlim[0], xlim[1])

    plt.fig.tight_layout()
    plt.canvas.draw()


class MSMSTargetModel(QtCore.QAbstractTableModel):
    def __init__(self, data=[]):
        QtCore.QAbstractTableModel.__init__(self)
        self._data = data
        self.MScansDelegate = None
        self.MpScansDelegate = None

    def setMScansDelectate(self, delegate):
        self.MScansDelegate = delegate

    def setMpScansDelegate(self, delegate):
        self.MpScansDelegate = delegate

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 13

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.BackgroundColorRole and index.column() in range(8):
            cols = ["#A4D3EE", "white"]
            # if index.row() in range(8):
            #    return QtGui.QBrush(QtGui.QColor(cols[1]))

            prevNames = list(set([self._data[i].lcmsmsNativeFileName for i in range(index.row() + 1)]))
            return QtGui.QBrush(QtGui.QColor(cols[len(prevNames) % len(cols)]))

        if not index.isValid():
            return None
        if not (role == QtCore.Qt.DisplayRole or role == QtCore.Qt.EditRole):
            return None

        if index.column() == 0:
            a = str(self._data[index.row()].targetName)
            return a
        if index.column() == 1:
            a = str(self._data[index.row()].parentSumFormula)
            return a
        if index.column() == 2:
            a = str(self._data[index.row()].lcmsmsNativeFileName)
            a = a.replace("\\", "/")
            return a[a.rfind("/") + 1 :]
        if index.column() == 3:
            a = str(self._data[index.row()].lcmsmsLabeledFileName)
            a = a.replace("\\", "/")
            return a[a.rfind("/") + 1 :]
        if index.column() == 4:
            return str(self._data[index.row()].nativeMZ)
        if index.column() == 5:
            return self._data[index.row()].usedAdduct
        if index.column() == 6:
            return self._data[index.row()].cNMetabolite
        if index.column() == 7:
            return self._data[index.row()].rtMin
        if index.column() == 8:
            return self._data[index.row()].rtMax
        if index.column() == 9:
            return self._data[index.row()].scanEventIndexM
        if index.column() == 10:
            return self._data[index.row()].scanEventIndexMp
        if index.column() == 11:
            return self._data[index.row()].scanEventIndexFullScanM
        if index.column() == 12:
            return self._data[index.row()].scanEventIndexFullScanMp

    def setData(self, index, value, role=QtCore.Qt.DisplayRole):
        if index.column() == 0:
            s = str(value.toString())
            self._data[index.row()].targetName = s
        elif index.column() == 1:
            s = str(value.toString())
            self._data[index.row()].parentSumFormula = s
        elif index.column() == 4:
            s = str(value.toString())
            if is_float(s):
                mz = float(s)
                lmz = self._data[index.row()].cNMetabolite * 1.00335484 + mz
                self._data[index.row()].nativeMZ = mz

                update = False
                if self._data[index.row()].scanEventIndexM <= 0:
                    for sei, se in enumerate(self._data[index.row()].MS2ScanEventsNative):
                        try:
                            precurs = float(re.search("PreMZ: ([0-9]*\.[0-9]*) ", se).group(1))
                            ppm = abs(precurs - mz) * 1e6 / mz
                            if ppm <= 10:
                                self._data[index.row()].scanEventIndexM = sei
                                update = True
                        except Exception:
                            pass

                if self._data[index.row()].scanEventIndexMp <= 0:
                    for sei, se in enumerate(self._data[index.row()].MS2ScanEventsLabeled):
                        try:
                            precurs = float(re.search("PreMZ: ([0-9]*\.[0-9]*) ", se).group(1))
                            ppm = abs(precurs - lmz) * 1e6 / lmz
                            if ppm <= 10:
                                self._data[index.row()].scanEventIndexMp = sei
                                update = True
                        except Exception:
                            pass

                if update:
                    self.dataChanged.emit(
                        self.createIndex(0, 0),
                        self.createIndex(self.rowCount(0), self.columnCount(0)),
                    )
                return True
            else:
                return False
        elif index.column() == 5:
            value = str(value.toString())
            if value in self.adducts.keys():
                self._data[index.row()].usedAdduct = value
                self._data[index.row()].charge = self.adducts[value].charge
                return False
            else:
                return False
        elif index.column() == 6:
            s = str(value.toString())
            if is_int(s):
                self._data[index.row()].cNMetabolite = int(s)
                return True
            else:
                return False
        elif index.column() == 7:
            s = str(value.toString())
            if is_float(s):
                self._data[index.row()].rtMin = float(s)
                return True
            else:
                return False
        elif index.column() == 8:
            s = str(value.toString())
            if is_float(s):
                self._data[index.row()].rtMax = float(s)
                return True
            else:
                return False
        elif index.column() == 9:
            self._data[index.row()].scanEventIndexM = value
            return True
        elif index.column() == 10:
            self._data[index.row()].scanEventIndexMp = value
            return True
        elif index.column() == 11:
            self._data[index.row()].scanEventIndexFullScanNative = value
            return True
        elif index.column() == 12:
            self._data[index.row()].scanEventIndexFullScanLabeled = value
            return True

        return False

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return [
                "Target name",
                "Sum formula",
                "File native",
                "File labeled",
                "MZ",
                "Adduct",
                "Cn metabolite",
                "Min. Rt [min]",
                "Max. Rt [min]",
                "Target M",
                "Target M'",
                "Full scan native",
                "Full scan labeled",
            ][col]
        return None

    def insertRows(self, row, count, parent=QtCore.QModelIndex(), object=None, atPos=None):
        self.beginInsertRows(parent, row, row + count - 1)

        x = Bunch()
        if object is not None:
            x = object
        if atPos is None:
            atPos = len(self._data)
        self._data.insert(atPos, x)

        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent=QtCore.QModelIndex()):
        assert len(self._data) > row
        self.beginRemoveRows(parent, row, row + count - 1)
        for i in range(count):
            self._data.pop(row)
        self.endRemoveRows()
        return True

    def flags(self, index):
        if index.column() in [2, 3]:
            return QtCore.Qt.ItemIsEnabled
        else:
            return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEditable


class MSMSFileComboDelegate(QtWidgets.QItemDelegate):
    """
    A delegate that places a fully functioning QComboBox in every
    cell of the column to which it's applied
    """

    def __init__(self, MSMSTargetModel, parent):
        QtWidgets.QItemDelegate.__init__(self, parent)
        self.MSMSTargetModel = MSMSTargetModel

    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)

        if index.column() == 9:
            combo.addItems(self.MSMSTargetModel._data[index.row()].MS2ScanEventsNative)
        if index.column() == 10:
            combo.addItems(self.MSMSTargetModel._data[index.row()].MS2ScanEventsLabeled)
        elif index.column() == 11:
            combo.addItems(self.MSMSTargetModel._data[index.row()].MS1ScanEventsNative)
        elif index.column() == 12:
            combo.addItems(self.MSMSTargetModel._data[index.row()].MS1ScanEventsLabeled)

        self.connect(
            combo,
            QtCore.Signal("currentIndexChanged(int)"),
            self,
            QtCore.Slot("currentIndexChanged()"),
        )
        return combo

    def setEditorData(self, editor, index):
        editor.blockSignals(True)
        editor.setCurrentIndex(int(index.model().data(index)))
        editor.blockSignals(False)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentIndex())

    @QtCore.Slot()
    def currentIndexChanged(self):
        self.commitData.emit(self.sender())


def runFile(rI):
    rI.process()


def interruptIndividualFilesProcessing(selfObj, pool):
    if (
        QtWidgets.QMessageBox.question(
            selfObj,
            "FragExtract",
            "Are you sure you want to cancel?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        == QtWidgets.QMessageBox.Yes
    ):
        pool.close()
        pool.terminate()
        pool.join()

        selfObj.terminateJobs = True
        logging.info("Processing stopped by user")

        return True
    else:
        return False  # don't close progresswrapper and continue processing files


class FEMainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, initDir=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle("MetExtract II - FragExtract")

        logging.info("MetExtract II (module %s)" % "FragExtract")

        ## setup additional ui things
        self.MSMSTargetModel = MSMSTargetModel(data=[])

        self.elements = [
            ConfiguredElement(name="C", weight=12.0, numberValenzElectrons=4, minCount=0, maxCount=39),
            ConfiguredElement(
                name="H",
                weight=1.00783,
                numberValenzElectrons=1,
                minCount=0,
                maxCount=120,
            ),
            ConfiguredElement(
                name="O",
                weight=15.99491,
                numberValenzElectrons=6,
                minCount=0,
                maxCount=40,
            ),
            ConfiguredElement(
                name="N",
                weight=14.00307,
                numberValenzElectrons=5,
                minCount=0,
                maxCount=6,
            ),
            ConfiguredElement(
                name="S",
                weight=31.97207,
                numberValenzElectrons=6,
                minCount=0,
                maxCount=4,
            ),
        ]
        self.MSMSTargetModel.adducts = {
            "[M]+": ConfiguredAdduct(name="[M]+", mzoffset=-0.00055, charge=1, polarity="+", entryType="user"),
            "[M+H]+": ConfiguredAdduct(
                name="[M+H]+",
                mzoffset=1.007276,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+NH4]+": ConfiguredAdduct(
                name="[M+NH4]+",
                mzoffset=18.033823,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+Na]+": ConfiguredAdduct(
                name="[M+Na]+",
                mzoffset=22.989218,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+CH3OH+H]+": ConfiguredAdduct(
                name="[M+CH3OH+H]+",
                mzoffset=33.033489,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+K]+": ConfiguredAdduct(
                name="[M+K]+",
                mzoffset=38.963158,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+ACN+H]+": ConfiguredAdduct(
                name="[M+ACN+H]+",
                mzoffset=42.033823,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+2Na-H]+": ConfiguredAdduct(
                name="[M+2Na-H]+",
                mzoffset=44.971160,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+2K-H]+": ConfiguredAdduct(
                name="[M+2K-H]+",
                mzoffset=76.919040,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+CH3FeN]+": ConfiguredAdduct(
                name="[M+CH3FeN]+",
                mzoffset=84.96094,
                charge=1,
                polarity="+",
                entryType="user",
            ),
            "[M+2H]++": ConfiguredAdduct(
                name="[M+2H]++",
                mzoffset=1.007276,
                charge=2,
                polarity="+",
                entryType="user",
            ),
            "[M+H+NH4]++": ConfiguredAdduct(
                name="[M+H+NH4]++",
                mzoffset=9.520550,
                charge=2,
                polarity="+",
                entryType="user",
            ),
            "[M+H+Na]++": ConfiguredAdduct(
                name="[M+H+Na]++",
                mzoffset=11.998247,
                charge=2,
                polarity="+",
                entryType="user",
            ),
            "[M+H+K]++": ConfiguredAdduct(
                name="[M+H+K]++",
                mzoffset=19.985217,
                charge=2,
                polarity="+",
                entryType="user",
            ),
            "[M+2Na]++": ConfiguredAdduct(
                name="[M+2Na]++",
                mzoffset=22.989218,
                charge=2,
                polarity="+",
                entryType="user",
            ),
            "[M-H2O-H]-": ConfiguredAdduct(
                name="[M-H2O-H]-",
                mzoffset=-19.01839,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M]-": ConfiguredAdduct(name="[M]-", mzoffset=0.00055, charge=1, polarity="-", entryType="user"),
            "[M-H]-": ConfiguredAdduct(
                name="[M-H]-",
                mzoffset=-1.007276,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M+Na-2H]-": ConfiguredAdduct(
                name="[M+Na-2H]-",
                mzoffset=20.974666,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M+Cl]-": ConfiguredAdduct(
                name="[M+Cl]-",
                mzoffset=34.969402,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M+K-2H]-": ConfiguredAdduct(
                name="[M+K-2H]-",
                mzoffset=36.948606,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M+Br]-": ConfiguredAdduct(
                name="[M+Br]-",
                mzoffset=78.918885,
                charge=1,
                polarity="-",
                entryType="user",
            ),
            "[M+Fa-H]-": ConfiguredAdduct(
                name="[M+Fa-H]-",
                mzoffset=44.998201,
                charge=1,
                polarity="-",
                entryType="user",
            ),
        }

        self.MSMSFileComboDelegateM = MSMSFileComboDelegate(self.MSMSTargetModel, parent=self)
        self.MSMSFileComboDelegateMp = MSMSFileComboDelegate(self.MSMSTargetModel, parent=self)
        self.MSFileComboDelegateM = MSMSFileComboDelegate(self.MSMSTargetModel, parent=self)
        self.MSFileComboDelegateMp = MSMSFileComboDelegate(self.MSMSTargetModel, parent=self)

        self.processFilesTable.setItemDelegateForColumn(9, self.MSMSFileComboDelegateM)
        self.processFilesTable.setItemDelegateForColumn(10, self.MSMSFileComboDelegateMp)
        self.processFilesTable.setItemDelegateForColumn(11, self.MSFileComboDelegateM)
        self.processFilesTable.setItemDelegateForColumn(12, self.MSFileComboDelegateMp)

        self.processFilesTable.setModel(self.MSMSTargetModel)
        for row in range(0, self.MSMSTargetModel.rowCount()):
            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(row, 0))

        sectionSizes = [250, 100, 190, 190, 60, 120, 90, 80, 80, 420, 420, 200, 200]
        for i in range(len(sectionSizes)):
            self.processFilesTable.horizontalHeader().resizeSection(i, sectionSizes[i])

        self.processFilesTable.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.processFilesTable.customContextMenuRequested.connect(self.showTargetPopup)

        # Setup first plot
        # http://eli.thegreenplace.net/2009/01/20/matplotlib-with-pyqt-guis/
        self.pl1 = QtCore.QObject()
        self.pl1.dpi = 50
        self.pl1.fig = Figure((5.0, 4.0), dpi=self.pl1.dpi, facecolor="white")
        self.pl1.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.95)
        self.pl1.canvas = FigureCanvas(self.pl1.fig)
        self.pl1.canvas.setParent(self.visualizationWidget)
        self.pl1.axes = self.pl1.fig.add_subplot(111)
        simpleaxis(self.pl1.axes)
        self.pl1.twinxs = [self.pl1.axes]
        self.pl1.mpl_toolbar = NavigationToolbar(self.pl1.canvas, self.visualizationWidget)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.pl1.canvas)
        vbox.addWidget(self.pl1.mpl_toolbar)
        self.visualizationWidget.setLayout(vbox)

        # Setup EIC plot
        # http://eli.thegreenplace.net/2009/01/20/matplotlib-with-pyqt-guis/
        self.pl2 = QtCore.QObject()
        self.pl2.dpi = 50
        self.pl2.fig = Figure((5.0, 4.0), dpi=self.pl2.dpi, facecolor="white")
        self.pl2.fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=0.95)
        self.pl2.canvas = FigureCanvas(self.pl2.fig)
        self.pl2.canvas.setParent(self.eicWidget)
        self.pl2.axes = self.pl2.fig.add_subplot(111)
        simpleaxis(self.pl2.axes)
        self.pl2.twinxs = [self.pl2.axes]
        self.pl2.mpl_toolbar = NavigationToolbar(self.pl2.canvas, self.eicWidget)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.pl2.canvas)
        vbox.addWidget(self.pl2.mpl_toolbar)
        self.eicWidget.setLayout(vbox)

        self.tabWidget.setCurrentIndex(0)
        if cpu_count() == 1:
            self.keepCPUCoreUnusedCheckbox.setCheckState(QtCore.Qt.Unchecked)
            self.keepCPUCoreUnusedCheckbox.setVisible(False)
            self.useCPUCoresSpinner.setMaximum(cpu_count())
        else:
            self.keepCPUCoreUnusedCheckbox.setCheckState(QtCore.Qt.Unchecked)
            self.keepCPUCoreUnusedCheckbox.setCheckState(QtCore.Qt.Checked)
        self.updateCores()
        self.useCPUCoresSpinner.setValue(self.useCPUCoresSpinner.maximum())

        ## add event handlers
        self.addMSMSTarget.clicked.connect(self.addMSMSTargetHandler)
        self.deleteMSMSTarget.clicked.connect(self.deleteMSMSTargetHandler)

        self.saveCompilation.clicked.connect(self.saveCompilationHandler)
        self.loadCompilation.clicked.connect(self.loadCompilationHandler)

        self.keepCPUCoreUnusedCheckbox.toggled.connect(self.updateCores)

        ## add menu bar event handlers
        self.actionLoad_Settings.triggered.connect(self.loadSettings)
        self.actionSave_Settings.triggered.connect(self.saveSettings)
        self.actionExit.triggered.connect(self.exit)

        self.actionHelp.triggered.connect(self.showHelp)
        self.actionAbout.triggered.connect(self.showAbout)

        # add button events
        self.startProcessingButton.clicked.connect(self.processTargets)
        self.fragmentAnnotationButton.clicked.connect(self.showAnnotationConfiguration)
        self.configAdductsButton.clicked.connect(self.showConfigureAdductsDialog)

        # add processing tab events
        self.mappedResultsToFile = {}
        self.curFileSQLiteConnection = None
        self.processedFilesComboBox.currentIndexChanged.connect(self.setActiveResultsFile)
        self.openExternallyButton.clicked.connect(self.openFileExternally)
        self.resultsTreeWidget.itemSelectionChanged.connect(self.updateResultsIllustration)

        self.resultsTreeWidget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.resultsTreeWidget.customContextMenuRequested.connect(self.showResultsPopup)

    def showTargetPopup(self, position):
        menu = QtWidgets.QMenu()
        quitAction = menu.addAction("Delete target")
        copyAction = menu.addAction("Duplicate target")
        menu.addSeparator()
        pasteAction = menu.addAction("Paste in row")

        action = menu.exec_(self.processFilesTable.mapToGlobal(position))

        if action == quitAction:
            index = self.processFilesTable.indexAt(position)
            self.MSMSTargetModel.removeRows(index.row(), 1)

        if action == copyAction:
            indices = set()
            for index in self.processFilesTable.selectedIndexes():
                indices.add(index.row())
            for rowInd in sorted(indices, reverse=True):
                newObj = deepcopy(self.MSMSTargetModel._data[rowInd])
                newObj.targetName = "Copy of %s" % newObj.targetName
                self.MSMSTargetModel.insertRows(rowInd, 1, object=newObj, atPos=rowInd)

        if action == pasteAction:
            selIndex = self.processFilesTable.indexAt(position)

            text = pyperclip.paste().replace("\r", "").strip()

            for ti, t in enumerate(text.split("\n")):
                self.MSMSTargetModel.setData(
                    selIndex.sibling(selIndex.row() + ti, selIndex.column()),
                    t,
                )

    def showAnnotationConfiguration(self):
        t = adductsEdit(nls=self.elements, showAdductsConfiguration=False)
        if t.executeDialog() == QtWidgets.QDialog.Accepted:
            self.elements = t.getNeutralLosses()

            logging.info("Configured neutral loss elements:")
            for elem in self.elements:
                logging.info("  *", elem.name, elem)

    def updateCores(self):
        rem = 0
        if self.keepCPUCoreUnusedCheckbox.checkState() == QtCore.Qt.Checked:
            rem = 1
        self.useCPUCoresSpinner.setMaximum(max(1, cpu_count() - rem))
        if self.useCPUCoresSpinner.value() == self.useCPUCoresSpinner.maximum() - 1:
            self.useCPUCoresSpinner.setValue(self.useCPUCoresSpinner.maximum())

    def addMSMSTargetHandler(self):
        mzXMLFiles = QtWidgets.QFileDialog.getOpenFileNames(
            caption="Select chromatograms",
            filter="mzXML file (*.mzXML);;mzML file (*.mzML)",
        )

        if len(mzXMLFiles) == 0:
            return
        else:
            self.addMSMSTargetsFromFileNames(mzXMLFiles)

    def addMSMSTargetsFromFileNames(self, mzXMLFiles):
        pw = ProgressWrapper(1, parent=self, showIndProgress=False)
        pw.getCallingFunction()("max")(len(mzXMLFiles))
        curFi = 0
        pw.getCallingFunction()("value")(0)
        pw.getCallingFunction()("text")("Importing targets")
        pw.show()
        pw.getCallingFunction()("value")(0)

        tryMatchTargets = False
        if (
            QtWidgets.QMessageBox.question(
                self,
                "FragExtract",
                "Do you want to try to automatically match corresponding LC-MS/MS scans based on retention time?\n\nThe match will be performed using the start of two scanEvents and their pre-cursor MZs.\nIf a pair of two scanEvents is used for two targets at different retention times, it will not be separated and treated as one. \nPlease carefully review the list",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            == QtWidgets.QMessageBox.Yes
        ):
            tryMatchTargets = True

        opened = {}

        for mzXMLFile in mzXMLFiles:
            mzXMLFile = mzXMLFile.replace("\\", "/")
            mzXMLFile = str(mzXMLFile)

            matchFromDifferentFiles = False
            if "12C" in mzXMLFile and os.path.isfile(mzXMLFile.replace("12C", "13C")):
                if (
                    QtWidgets.QMessageBox.question(
                        self,
                        "FragExtract",
                        "A file with the same name but 13C instead of 12C was found. Shall this file be used for parsing the 13C spectra?\n\nFiles:\n%s\n%s" % (mzXMLFile, mzXMLFile.replace("12C", "13C")),
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    )
                    == QtWidgets.QMessageBox.Yes
                ):
                    matchFromDifferentFiles = True

            a = mzXMLFile.replace("\\", "/")
            a = a[a.rfind("/") + 1 :]

            if mzXMLFile not in opened.keys():
                mzxml = Chromatogram()
                mzxml.parse_file(mzXMLFile, ignoreCharacterData=True)
                opened[mzXMLFile] = mzxml

            if matchFromDifferentFiles:
                a = mzXMLFile.replace("\\", "/").replace("12C", "13C")
                a = a[a.rfind("/") + 1 :]

                if mzXMLFile.replace("12C", "13C") not in opened.keys():
                    mzxml = Chromatogram()
                    mzxml.parse_file(mzXMLFile.replace("12C", "13C"), ignoreCharacterData=True)
                    opened[mzXMLFile.replace("12C", "13C")] = mzxml

            mzxmlNative = opened[mzXMLFile]

            scanEventsMS2Native = mzxmlNative.getFilterLinesExtended(
                includeMS1=False,
                includeMS2=True,
                includePosPolarity=True,
                includeNegPolarity=True,
            )
            sortedScanEventsMS2Native = sorted(scanEventsMS2Native.keys())
            scanEventsMS1Native = mzxmlNative.getFilterLines(
                includeMS1=True,
                includeMS2=False,
                includePosPolarity=True,
                includeNegPolarity=True,
            )
            scanEventsMS1PerPolarityNative = mzxmlNative.getFilterLinesPerPolarity(includeMS1=True, includeMS2=False)

            mzxmlLabeled = opened[mzXMLFile]
            if matchFromDifferentFiles:
                mzxmlLabeled = opened[mzXMLFile.replace("12C", "13C")]

            scanEventsMS2Labeled = mzxmlLabeled.getFilterLinesExtended(
                includeMS1=False,
                includeMS2=True,
                includePosPolarity=True,
                includeNegPolarity=True,
            )
            sortedScanEventsMS2Labeled = sorted(scanEventsMS2Labeled.keys())
            scanEventsMS1Labeled = mzxmlLabeled.getFilterLines(
                includeMS1=True,
                includeMS2=False,
                includePosPolarity=True,
                includeNegPolarity=True,
            )
            scanEventsMS1PerPolarityLabeled = mzxmlLabeled.getFilterLinesPerPolarity(includeMS1=True, includeMS2=False)

            t = [
                (k, v)
                for k, v in sorted(
                    list(scanEventsMS2Native.items()),
                    key=lambda x: x[1].targetStartTime,
                )
            ]
            z = [
                (k, v)
                for k, v in sorted(
                    list(scanEventsMS2Labeled.items()),
                    key=lambda x: x[1].targetStartTime,
                )
            ]

            matches = []

            matchedTargets = []
            if tryMatchTargets:
                pass

                if len(t) > 2:
                    for i in range(len(t)):
                        for j in range(len(z)):
                            km, vm = t[i]
                            kmp, vmp = z[j]
                            if vm.preCursorMz < vmp.preCursorMz:
                                cn = round((vmp.preCursorMz - vm.preCursorMz) / 1.00335484, 0)

                                # print(vm.preCursorMz, vm.polarity, vm.scanTimes)
                                if vm.polarity == vmp.polarity and cn > 0 and (vmp.preCursorMz - vm.preCursorMz) > 1 and (abs(vmp.preCursorMz - vm.preCursorMz - cn * 1.00335) * 1000000.0 / vmp.preCursorMz) <= 10:
                                    # print("   ", vmp.preCursorMz, vm.polarity, cn)
                                    for irt in vm.scanTimes:
                                        bestMatch = None
                                        bestMatchDiff = 1000000
                                        for jrt in vmp.scanTimes:
                                            if jrt >= irt and abs(jrt - irt) <= 12:
                                                if abs(jrt - irt) < bestMatchDiff:
                                                    bestMatch = jrt
                                                    bestMatchDiff = abs(jrt - irt)

                                        if bestMatch is not None:
                                            jrt = bestMatch

                                            b = Bunch(
                                                M=km,
                                                Mp=kmp,
                                                cn=cn,
                                                nativeMz=vm.preCursorMz,
                                                labelledMz=vmp.preCursorMz,
                                                startRt=irt,
                                                endRt=jrt,
                                                polarity=vm.polarity,
                                            )
                                            matches.append(b)

                for i in range(len(matches)):
                    obj = Bunch(
                        targetName="Unknown %d" % (len(self.MSMSTargetModel._data) + i),
                        parentSumFormula="",
                        lcmsmsNativeFileName=mzXMLFile,
                        lcmsmsLabeledFileName=mzXMLFile.replace("12C", "13C") if matchFromDifferentFiles else mzXMLFile,
                        scanEventIndexM=0,
                        scanEventIndexMp=0,
                        scanEventIndexFullScanM=0,
                        scanEventIndexFullScanMp=0,
                        nativeMZ=0,
                        charge=1,
                        MS2ScanEventsNative=sortedScanEventsMS2Native,
                        MS2ScanEventsLabeled=sortedScanEventsMS2Labeled,
                        MS1ScanEventsNative=sorted(list(scanEventsMS1Native)),
                        MS1ScanEventsLabeled=sorted(list(scanEventsMS1Labeled)),
                        cNMetabolite=-1,
                        rtMin=0,
                        rtMax=0,
                        usedAdduct="[M+H]+",
                    )

                    if tryMatchTargets and i < len(matches):
                        obj.scanEventIndexM = sortedScanEventsMS2Native.index(matches[i].M)
                        obj.scanEventIndexMp = sortedScanEventsMS2Labeled.index(matches[i].Mp)
                        obj.scanEventIndexFullScanM = obj.MS1ScanEventsNative.index(list(scanEventsMS1PerPolarityNative[matches[i].polarity])[0])
                        obj.scanEventIndexFullScanMp = obj.MS1ScanEventsLabeled.index(list(scanEventsMS1PerPolarityLabeled[matches[i].polarity])[0])
                        obj.nativeMZ = matches[i].nativeMz
                        obj.cNMetabolite = matches[i].cn
                        obj.rtMin = matches[i].startRt / 60.0 - 0.3
                        obj.rtMax = matches[i].endRt / 60.0 + 0.3
                        obj.usedAdduct = "[M+H]+" if matches[i].polarity == "+" else "[M-H]-"

                    matchedTargets.append(obj)

                curFi += 1
                pw.getCallingFunction()("value")(curFi)

                if (
                    QtWidgets.QMessageBox.question(
                        self,
                        "FragExtract",
                        "%d MSMS scans were found to be potential targets. Do you want to add them?" % (len(matchedTargets)),
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    )
                    == QtWidgets.QMessageBox.Yes
                ):
                    for obj in matchedTargets:
                        self.MSMSTargetModel.insertRows(len(self.MSMSTargetModel._data), 1, object=obj)

                        self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 9))
                        self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 10))
                        self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 11))
                        self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 12))
            else:
                if len(scanEventsMS2Native) > 2:
                    val, okClicked = QtWidgets.QInputDialog.getInteger(
                        self,
                        "FragExtract",
                        "The file (%s) contains more than 2 MS/MS scan events.\nDo you want to define multipletargets for this file?\nClick 'Cancel' to add file only once" % a,
                        1,
                        1,
                        len(scanEventsMS2Native),
                    )
                    if okClicked:
                        addMultipleTimes = val
                        for i in range(addMultipleTimes):
                            obj = Bunch(
                                targetName="Unknown %d" % (len(self.MSMSTargetModel._data) + i),
                                parentSumFormula="",
                                lcmsmsNativeFileName=mzXMLFile,
                                lcmsmsLabeledFileName=mzXMLFile.replace("12C", "13C") if matchFromDifferentFiles else mzXMLFile,
                                scanEventIndexM=0,
                                scanEventIndexMp=0,
                                scanEventIndexFullScanM=0,
                                scanEventIndexFullScanMp=0,
                                nativeMZ=-1.1,
                                charge=1,
                                MS2ScanEventsNative=sortedScanEventsMS2Native,
                                MS2ScanEventsLabeled=sortedScanEventsMS2Labeled,
                                MS1ScanEventsNative=sorted(list(scanEventsMS1Native)),
                                MS1ScanEventsLabeled=sorted(list(scanEventsMS1Labeled)),
                                cNMetabolite=-1,
                                rtMin=-1.1,
                                rtMax=-1.1,
                                usedAdduct="",
                            )

                            self.MSMSTargetModel.insertRows(len(self.MSMSTargetModel._data), 1, object=obj)

                            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 9))
                            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 10))
                            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 11))
                            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 12))
        pw.hide()

        self.updateResultsView()

        if tryMatchTargets:
            QtWidgets.QMessageBox.information(
                self,
                "FragExtract",
                "Targets have been matched automatically.\nPlease review carefully and correct if necessary.",
            )

    def deleteMSMSTargetHandler(self):
        todel = set()
        for index in self.processFilesTable.selectedIndexes():
            todel.add(index.row())
        for rowInd in sorted(todel, reverse=True):
            self.MSMSTargetModel.removeRows(rowInd, 1)

    def showConfigureAdductsDialog(self):
        t = adductsEdit(
            adds=self.MSMSTargetModel.adducts.values(),
            showRelationshipConfiguratio=False,
        )
        if t.executeDialog() == QtWidgets.QDialog.Accepted:
            self.MSMSTargetModel.adducts = {}

            logging.info("Configured adducts:")
            for adduct in t.getAdducts():
                logging.info("  *", adduct.name)
                self.MSMSTargetModel.adducts[adduct.name] = adduct

    def loadSettings(self):
        settingsFile = QtWidgets.QFileDialog.getOpenFileName(
            caption="Specify settings file",
            filter="Settings (*.ini);; Group compilations (*.grp)",
        )

        if len(settingsFile) == 0:
            return
        else:
            self._loadSettingsFromSettingsFile(settingsFile)

    def loadSettingsFromSettingsFile(self, settingsFile):
        sett = QtCore.QSettings(settingsFile, QtCore.QSettings.IniFormat)

        sett.beginGroup("Settings")

        if sett.contains("EICPPM"):
            self.eicPPMSpinner.setValue(sett.value("EICPPM").toInt()[0])
        if sett.contains("IntensityThreshold"):
            self.intensityThresoldSpinner.setValue(sett.value("IntensityThreshold").toDouble()[0])
        if sett.contains("MinScaledIntensity"):
            self.minScaledPeakIntensity_Spinner.setValue(sett.value("MinScaledIntensity").toDouble()[0])
        if sett.contains("scalePrecursorMZ"):
            self.scalePrecursorMZ.setChecked(sett.value("scalePrecursorMZ").toBool())
        if sett.contains("maxPPMMatchingError"):
            self.maxPPMErrorMatching_Spinner.setValue(sett.value("maxPPMMatchingError").toDouble()[0])
        if sett.contains("useZeroLabelingAtoms"):
            self.useZeroLabelingAtoms.setChecked(sett.value("useZeroLabelingAtoms").toBool())
        if sett.contains("minXn"):
            self.minXn.setValue(sett.value("minXn").toInt()[0])
        if sett.contains("useTracExtractAnnotation"):
            self.useTracExtractAnnotation.setChecked(sett.value("useTracExtractAnnotation").toBool())
        if sett.contains("maxRelativeIntensityError"):
            self.maxRelIntensityError_spinner.setValue(sett.value("maxRelativeIntensityError").toDouble()[0])

        if sett.contains("configuredElements"):
            self.elements = pickle.loads(base64.b64decode(str(sett.value("configuredElements").toString())))
        if sett.contains("annotationPPM"):
            self.annotationPPMErrorSpinner.setValue(sett.value("annotationPPM").toDouble()[0])

        if sett.contains("applyParentFragmentConsistencyRule"):
            self.applyParentFragmentConsistencyRuleCheckBox.setChecked(sett.value("applyParentFragmentConsistencyRule").toBool())

        if sett.contains("doubleSpinBox_massBankPPMError"):
            self.doubleSpinBox_massBankPPMError.setValue(sett.value("doubleSpinBox_massBankPPMError").toDouble()[0])
        if sett.contains("doubleSpinBox_massBankMinRelAbundance"):
            self.doubleSpinBox_massBankMinRelAbundance.setValue(sett.value("doubleSpinBox_massBankMinRelAbundance").toDouble()[0])
        if sett.contains("doubleSpinBox_massBankMinimalScore"):
            self.doubleSpinBox_massBankMinimalScore.setValue(sett.value("doubleSpinBox_massBankMinimalScore").toDouble()[0])
        if sett.contains("spinBox_massBankHitsToLoad"):
            self.spinBox_massBankHitsToLoad.setValue(sett.value("spinBox_massBankHitsToLoad").toInt()[0])

        if sett.contains("saveAsTSV"):
            self.saveResultsAsTSVCheckBox.setChecked(sett.value("saveAsTSV").toBool())
        if sett.contains("saveAsPDF"):
            self.saveResultsAsPDFCheckBox.setChecked(sett.value("saveAsPDF").toBool())
        if sett.contains("exportAsSirius"):
            self.exportAsSiriuscheckBox.setChecked(sett.value("exportAsSirius").toBool())

        sett.endGroup()

    def saveSettings(self):
        settingsFile = QtWidgets.QFileDialog.getSaveFileName(caption="Specify settings file", filter="Settings (*.ini)")

        if len(settingsFile) == 0:
            return
        else:
            self._saveSettingsToSettingsFile(settingsFile)

    def saveSettingsToSettingsFile(self, settingsFile):
        sett = QtCore.QSettings(settingsFile, QtCore.QSettings.IniFormat)

        sett.beginGroup("Settings")

        sett.setValue("EICPPM", self.eicPPMSpinner.value())
        sett.setValue("IntensityThreshold", self.intensityThresoldSpinner.value())
        sett.setValue("MinScaledIntensity", self.minScaledPeakIntensity_Spinner.value())
        sett.setValue("scalePrecursorMZ", self.scalePrecursorMZ.checkState() == QtCore.Qt.Checked)
        sett.setValue("maxPPMMatchingError", self.maxPPMErrorMatching_Spinner.value())
        sett.setValue(
            "useZeroLabelingAtoms",
            self.useZeroLabelingAtoms.checkState() == QtCore.Qt.Checked,
        )
        sett.setValue("minXn", self.minXn.value())
        sett.setValue(
            "useTracExtractAnnotation",
            self.useTracExtractAnnotation.checkState() == QtCore.Qt.Checked,
        )
        sett.setValue("maxRelativeIntensityError", self.maxRelIntensityError_spinner.value())

        sett.setValue("configuredElements", base64.b64encode(pickle.dumps(self.elements)))
        sett.setValue("annotationPPM", self.annotationPPMErrorSpinner.value())

        sett.setValue(
            "applyParentFragmentConsistencyRule",
            self.applyParentFragmentConsistencyRuleCheckBox.checkState() == QtCore.Qt.Checked,
        )

        sett.setValue(
            "doubleSpinBox_massBankPPMError",
            self.doubleSpinBox_massBankPPMError.value(),
        )
        sett.setValue(
            "doubleSpinBox_massBankMinRelAbundance",
            self.doubleSpinBox_massBankMinRelAbundance.value(),
        )
        sett.setValue(
            "doubleSpinBox_massBankMinimalScore",
            self.doubleSpinBox_massBankMinimalScore.value(),
        )
        sett.setValue("spinBox_massBankHitsToLoad", self.spinBox_massBankHitsToLoad.value())

        sett.setValue("saveAsTSV", self.saveResultsAsTSVCheckBox.checkState() == QtCore.Qt.Checked)
        sett.setValue("saveAsPDF", self.saveResultsAsPDFCheckBox.checkState() == QtCore.Qt.Checked)
        sett.setValue(
            "exportAsSirius",
            self.exportAsSiriuscheckBox.checkState() == QtCore.Qt.Checked,
        )
        self.exportAsSiriuscheckBox.checkState()
        sett.endGroup()

    def exit(self):
        import sys

        sys.exit(0)

    def showHelp(self):
        import subprocess
        import sys
        import webbrowser

        url = get_main_dir() + "/documentation/index.html"
        if sys.platform == "darwin":  # in case of OS X
            subprocess.Popen(["open", url])
        else:
            webbrowser.open_new_tab(url)

    def showAbout(self):
        dialog = AboutDialog(
            parent=self,
            app_name="FragExtract",
            version=MetExtractVersion,
            institute_name="CAC",
        )
        dialog.exec()

    def saveCompilationHandler(self):
        groupFile = QtWidgets.QFileDialog.getSaveFileName(caption="Specify group file", filter="Group file (*.grp)")

        if len(groupFile) == 0:
            return
        else:
            saveSettings = (
                QtWidgets.QMessageBox.question(
                    self,
                    "FragExtract",
                    "Do you want to include the used settings in this compilation?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                == QtWidgets.QMessageBox.Yes
            )

            self.saveCompilationToFileName(groupFile, saveSettings=saveSettings)

    def saveCompilationToFileName(self, groupFile, saveSettings=True):
        grps = QtCore.QSettings(groupFile, QtCore.QSettings.IniFormat)

        grps.beginGroup("ConfiguredLCMSMSFiles")
        numOfConfigurations = len(self.MSMSTargetModel._data)
        grps.setValue("ConfiguredLCMSMSTargets", numOfConfigurations)

        for i, configured in zip(range(numOfConfigurations), self.MSMSTargetModel._data):
            grps.setValue("%d__TargetName" % i, configured.targetName)
            grps.setValue("%d__Sumformula" % i, configured.parentSumFormula)
            grps.setValue("%d__TargetM" % i, configured.scanEventIndexM)
            grps.setValue("%d__TargetMprime" % i, configured.scanEventIndexMp)
            grps.setValue("%d__CnMetabolite" % i, configured.cNMetabolite)
            grps.setValue("%d__rtMin" % i, configured.rtMin)
            grps.setValue("%d__rtMax" % i, configured.rtMax)
            grps.setValue("%d__fullScanEventM" % i, configured.scanEventIndexFullScanM)
            grps.setValue("%d__fullScanEventMp" % i, configured.scanEventIndexFullScanMp)
            grps.setValue("%d__nativeMZ" % i, configured.nativeMZ)
            grps.setValue("%d__charge" % i, configured.charge)
            grps.setValue("%d__usedAdduct" % i, configured.usedAdduct)

            try:
                relFilePath = "./" + str(
                    os.path.relpath(
                        configured.lcmsmsNativeFileName,
                        os.path.split(str(groupFile))[0],
                    ).replace("\\", "/")
                )
            except ValueError:
                # Files are on different drives, use absolute path
                relFilePath = str(configured.lcmsmsNativeFileName).replace("\\", "/")
            grps.setValue("%d__FileNative" % i, relFilePath)

            try:
                relFilePath = "./" + str(
                    os.path.relpath(
                        configured.lcmsmsLabeledFileName,
                        os.path.split(str(groupFile))[0],
                    ).replace("\\", "/")
                )
            except ValueError:
                # Files are on different drives, use absolute path
                relFilePath = str(configured.lcmsmsLabeledFileName).replace("\\", "/")
            grps.setValue("%d__FileLabeled" % i, relFilePath)

        grps.endGroup()

        if saveSettings:
            self.saveSettingsToSettingsFile(groupFile)

    def loadCompilationHandler(self):
        groupFile = QtWidgets.QFileDialog.getOpenFileName(caption="Specify group file", filter="Group file (*.grp)")

        if len(groupFile) == 0:
            return
        else:
            self.loadCompilationFromGroupFile(groupFile)

    def loadCompilationFromGroupFile(self, groupFile):
        grps = QtCore.QSettings(groupFile, QtCore.QSettings.IniFormat)

        grps.beginGroup("ConfiguredLCMSMSFiles")
        numOfConfigurations = grps.value("ConfiguredLCMSMSTargets").toInt()[0]

        pw = ProgressWrapper(1, parent=self, showIndProgress=False)
        pw.getCallingFunction()("max")(numOfConfigurations)
        pw.getCallingFunction()("value")(0)
        pw.getCallingFunction()("text")("Importing targets")
        pw.show()
        pw.getCallingFunction()("value")(0)

        opened = {}

        for i in range(numOfConfigurations):
            targetName = str(grps.value("%d__TargetName" % i).toString())
            sumFormula = str(grps.value("%d__Sumformula" % i).toString())
            scanEventIndexM = grps.value("%d__TargetM" % i).toInt()[0]
            scanEventIndexMp = grps.value("%d__TargetMprime" % i).toInt()[0]
            cnMetabolite = grps.value("%d__CnMetabolite" % i).toInt()[0]
            rtMin = grps.value("%d__rtMin" % i).toDouble()[0]
            rtMax = grps.value("%d__rtMax" % i).toDouble()[0]
            scanEventIndexFSM = grps.value("%d__fullScanEventM" % i).toInt()[0]
            scanEventIndexFSMp = grps.value("%d__fullScanEventMp" % i).toInt()[0]
            nativeMZ = grps.value("%d__nativeMZ" % i).toDouble()[0]
            charge = grps.value("%d__charge" % i).toDouble()[0]
            usedAdduct = str(grps.value("%d__usedAdduct" % i).toString())

            if os.path.isabs(str(grps.value("%d__FileNative" % i).toString()).replace("\\", "/")):
                fileNameNative = str(grps.value("%d__FileNative" % i).toString())
            else:
                fileNameNative = os.path.split(str(groupFile))[0].replace("\\", "/") + "/" + str(grps.value("%d__FileNative" % i).toString()).replace("\\", "/")

            mzXMLFileNative = str(fileNameNative)

            if mzXMLFileNative not in opened.keys():
                mzxml = Chromatogram()
                mzxml.parse_file(mzXMLFileNative, ignoreCharacterData=True)
                opened[mzXMLFileNative] = mzxml

            mzxml = opened[mzXMLFileNative]
            scanEventsMS2Native = sorted(
                list(
                    mzxml.getFilterLines(
                        includeMS1=False,
                        includeMS2=True,
                        includePosPolarity=True,
                        includeNegPolarity=True,
                    )
                )
            )
            scanEventsMS1Native = sorted(
                list(
                    mzxml.getFilterLines(
                        includeMS1=True,
                        includeMS2=False,
                        includePosPolarity=True,
                        includeNegPolarity=True,
                    )
                )
            )

            if os.path.isabs(str(grps.value("%d__FileLabeled" % i).toString()).replace("\\", "/")):
                fileNameLabeled = str(grps.value("%d__FileLabeled" % i).toString())
            else:
                fileNameLabeled = os.path.split(str(groupFile))[0].replace("\\", "/") + "/" + str(grps.value("%d__FileLabeled" % i).toString()).replace("\\", "/")

            mzXMLFileLabeled = str(fileNameLabeled)

            if mzXMLFileLabeled not in opened.keys():
                mzxml = Chromatogram()
                mzxml.parse_file(mzXMLFileLabeled, ignoreCharacterData=True)
                opened[mzXMLFileLabeled] = mzxml

            mzxml = opened[mzXMLFileLabeled]
            scanEventsMS2Labeled = sorted(
                list(
                    mzxml.getFilterLines(
                        includeMS1=False,
                        includeMS2=True,
                        includePosPolarity=True,
                        includeNegPolarity=True,
                    )
                )
            )
            scanEventsMS1Labeled = sorted(
                list(
                    mzxml.getFilterLines(
                        includeMS1=True,
                        includeMS2=False,
                        includePosPolarity=True,
                        includeNegPolarity=True,
                    )
                )
            )

            self.MSMSTargetModel.insertRows(
                len(self.MSMSTargetModel._data),
                1,
                object=Bunch(
                    targetName=targetName,
                    parentSumFormula=sumFormula,
                    lcmsmsNativeFileName=fileNameNative,
                    lcmsmsLabeledFileName=fileNameLabeled,
                    scanEventIndexM=scanEventIndexM,
                    scanEventIndexMp=scanEventIndexMp,
                    scanEventIndexFullScanM=scanEventIndexFSM,
                    scanEventIndexFullScanMp=scanEventIndexFSMp,
                    nativeMZ=nativeMZ,
                    charge=charge,
                    MS2ScanEventsNative=scanEventsMS2Native,
                    MS2ScanEventsLabeled=scanEventsMS2Labeled,
                    MS1ScanEventsNative=scanEventsMS1Native,
                    MS1ScanEventsLabeled=scanEventsMS1Labeled,
                    cNMetabolite=cnMetabolite,
                    rtMin=rtMin,
                    rtMax=rtMax,
                    usedAdduct=usedAdduct,
                ),
            )

            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 9))
            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 10))
            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 11))
            self.processFilesTable.openPersistentEditor(self.MSMSTargetModel.index(self.MSMSTargetModel.rowCount() - 1, 12))

            pw.getCallingFunction()("value")(i + 1)
        pw.hide()

        grps.endGroup()

        if "Settings" in grps.childGroups():
            if (
                QtWidgets.QMessageBox.question(
                    self,
                    "FragExtract",
                    "Settings detected. Do you want to load the associated settings from the compilation?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                )
                == QtWidgets.QMessageBox.Yes
            ):
                self.loadSettingsFromSettingsFile(groupFile)

        self.updateResultsView()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()

            links = []
            mzxmls = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))

            for link in links:
                if link.lower().endswith("mzxml") or link.lower().endswith("mzml"):
                    mzxmls.append(link)

                if link.lower().endswith(".grp"):
                    self.loadCompilationFromGroupFile(link)
                elif link.lower().endswith(".ini"):
                    self.loadSettingsFromSettingsFile(link)

            if len(mzxmls) > 0:
                self.addMSMSTargetsFromFileNames(sorted(mzxmls))

        else:
            event.ignore()

    def processTargets(self):
        self.terminateJobs = False

        if (
            QtWidgets.QMessageBox.question(
                self,
                "FragExtract",
                "Do you want to save the compilation?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            == QtWidgets.QMessageBox.Yes
        ):
            self.saveCompilationHandler()

        if (
            QtWidgets.QMessageBox.question(
                self,
                "FragExtract",
                "Do you want to start the processing?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            == QtWidgets.QMessageBox.No
        ):
            return

        self.closeActiveResultsFile()

        self.processedFilesComboBox.clear()
        self.processedFilesComboBox.addItem("--")

        for index, target in zip(range(len(self.MSMSTargetModel._data)), self.MSMSTargetModel._data):
            if os.path.exists(str(target.lcmsmsNativeFileName) + ".identified.sqlite") and os.path.isfile(str(target.lcmsmsNativeFileName) + ".identified.sqlite"):
                os.remove(str(target.lcmsmsNativeFileName) + ".identified.sqlite")

        filesTargets = {}
        pIds = {}
        indGroups = {}
        for index, target in enumerate(self.MSMSTargetModel._data):
            if target.targetName == "" or target.targetName is None:
                target.targetName = "Unknown target %d" % (index + 1)

            pIds[index + 1] = "%s//%s (%d)" % (
                str(target.lcmsmsNativeFileName),
                str(target.targetName),
                index + 1,
            )
            if str(target.lcmsmsNativeFileName)[(str(target.lcmsmsNativeFileName).rfind("/") + 1) :] not in indGroups.keys():
                indGroups[str(target.lcmsmsNativeFileName)[(str(target.lcmsmsNativeFileName).rfind("/") + 1) :]] = []
            indGroups[str(target.lcmsmsNativeFileName)[(str(target.lcmsmsNativeFileName).rfind("/") + 1) :]].append("%s//%s (%d)" % (str(target.lcmsmsNativeFileName), str(target.targetName), index + 1))

            processingTarget = Bunch(
                id=0,
                targetName=str(target.targetName),
                lcmsmsNativeFile=str(target.lcmsmsNativeFileName),
                lcmsmsLabeledFile=str(target.lcmsmsLabeledFileName),
                parentSumFormula=str(target.parentSumFormula),
                precursorMZ=target.nativeMZ,
                adductObj=self.MSMSTargetModel.adducts[target.usedAdduct],
                adduct=self.MSMSTargetModel.adducts[target.usedAdduct].name,
                adductMZOffset=self.MSMSTargetModel.adducts[target.usedAdduct].mzoffset,
                Cn=int(target.cNMetabolite),
                chargeCount=target.charge,
                startRT=target.rtMin,
                stopRT=target.rtMax,
                scanEventMS1M="",
                scanEventMS1Mp="",
                scanIDNativeRaw="",
                scanIDLabelledRaw="",
                scanEventMS2Native="",
                scanEventMS2Labeled="",
                nativeScanEventNum=target.scanEventIndexM,
                labelledScanEventNum=target.scanEventIndexMp,
                nativefullScanEventNum=target.scanEventIndexFullScanM,
                labeledfullScanEventNum=target.scanEventIndexFullScanMp,
                pID=index + 1,
            )

            if str(target.lcmsmsNativeFileName) not in filesTargets.keys():
                filesTargets[str(target.lcmsmsNativeFileName)] = []
            filesTargets[str(target.lcmsmsNativeFileName)].append(processingTarget)

        cpus = min(cpu_count(), self.useCPUCoresSpinner.value())
        p = Pool(processes=min(len(filesTargets.keys()), cpus))  # , maxtasksperchild=1) # only in python >=2.7
        manager = Manager()
        lock = manager.Lock()
        queue = manager.Queue()

        targetsToProc = []
        index = 0
        for lcmsmsNativeFileName, targets in filesTargets.items():
            f = ProcessTarget(
                targets=targets,
                chromatogramFileNative=targets[0].lcmsmsNativeFile,
                chromatogramFileLabeled=targets[0].lcmsmsLabeledFile,
                fullScanEICppm=self.eicPPMSpinner.value(),
                fullScanThreshold=self.intensityThresoldSpinner.value(),
                minMSMSPeakIntensityScaled=self.minScaledPeakIntensity_Spinner.value(),
                scalePrecursorMZ=self.scalePrecursorMZ.checkState() == QtCore.Qt.Checked,
                matchingPPM=self.maxPPMErrorMatching_Spinner.value(),
                maxRelError=self.maxRelIntensityError_spinner.value(),
                minXn=self.minXn.value(),
                useZeroLabelingAtoms=self.useZeroLabelingAtoms.checkState() == QtCore.Qt.Checked,
                useTracExtractAnnotation=self.useTracExtractAnnotation.checkState() == QtCore.Qt.Checked,
                labellingOffset=1.00335484,
                annotationElements=self.elements,
                annotationPPM=self.annotationPPMErrorSpinner.value(),
                useParentFragmentConsistencyRule=self.applyParentFragmentConsistencyRuleCheckBox.checkState() == QtCore.Qt.Checked,
                massBankPPMError=self.doubleSpinBox_massBankPPMError.value(),
                massBankMinRelAbundance=int(self.doubleSpinBox_massBankMinRelAbundance.value()),
                massBankMinimalScore=self.doubleSpinBox_massBankMinimalScore.value(),
                massBankHitsToLoad=self.spinBox_massBankHitsToLoad.value(),
                saveAsTSV=self.saveResultsAsTSVCheckBox.checkState() == QtCore.Qt.Checked,
                saveAsPDF=self.saveResultsAsPDFCheckBox.checkState() == QtCore.Qt.Checked,
                exportAsSirius=self.exportAsSiriuscheckBox.checkState() == QtCore.Qt.Checked,
                lock=lock,
                queue=queue,
                pID=index + 1,
                feVersion=MetExtractVersion,
            )
            targetsToProc.append(f)

            index += 1

        # start the multiprocessing
        p.imap_unordered(runFile, targetsToProc)

        pw = ProgressWrapper(
            pwCount=min(len(filesTargets.keys()), cpus) + 1,
            showIndProgress=True,
            indGroups=indGroups,
            parent=self,
            closeCallback=CallBackMethod(_target=interruptIndividualFilesProcessing, selfObj=self, pool=p).getRunMethod(),
        )

        for w in range(1, min(len(filesTargets.keys()), cpus) + 1):
            pw.setMax(1.0, w)

        pw.show()
        pwMain = pw.getCallingFunction(0)
        pwMain("max")(len(self.MSMSTargetModel._data))
        pwMain("value")(0)

        start = time.time()

        # monitor processing of individual LC-HRMS files and report to the user
        loop = True
        freeSlots = range(min(len(targetsToProc), cpus))
        assignedThreads = {}
        completed = 0
        history = {}
        while loop and not self.terminateJobs:
            if completed == len(self.MSMSTargetModel._data):
                loop = False
            else:
                pwMain("value")(completed)

                while not (queue.empty()):
                    mes = queue.get(block=False, timeout=1)
                    if mes.pid not in history.keys():
                        history[mes.pid] = []
                    history[mes.pid].append(mes)

                    if "start" == mes.mes:
                        if len(freeSlots) > 0:
                            w = freeSlots.pop()
                            assignedThreads[mes.pid] = w

                            pw.getCallingFunction()("statuscolor")(pIds[mes.pid], "orange")
                            pw.getCallingFunction()("statustext")(
                                pIds[mes.pid],
                                text="File: %s\nStatus: %s\nProcess ID: %d" % (pIds[mes.pid], "processing", mes.pid),
                            )
                        else:
                            logging.error("Something went wrong..")
                            logging.error('Progress bars do not work correctly, but files will be processed and "finished.." will be printed..')

                    if mes.mes in ["log", "text", "max", "value"]:
                        if mes.pid in assignedThreads:
                            pw.getCallingFunction(assignedThreads[mes.pid] + 1)(mes.mes)(mes.val)
                        else:
                            logging.error("Error %d" % mes.pid)

                    if "end" == mes.mes or "failed" == mes.mes:
                        completed += 1
                        freeS = assignedThreads[mes.pid]
                        pw.getCallingFunction(assignedThreads[mes.pid] + 1)("text")("")
                        pw.getCallingFunction(assignedThreads[mes.pid] + 1)("value")(0)

                        pw.getCallingFunction()("statuscolor")(
                            pIds[mes.pid],
                            "olivedrab" if mes.mes == "end" else "firebrick",
                        )
                        pw.getCallingFunction()("statustext")(
                            pIds[mes.pid],
                            text="File: %s\nStatus: %s"
                            % (
                                pIds[mes.pid],
                                "finished" if mes.mes == "end" else "failed",
                            ),
                        )

                        if freeS == -1:
                            logging.error("Something went wrong..")
                            logging.error('Progress bars do not work correctly, but files will be processed and "finished.." will be printed..')
                        else:
                            assignedThreads[mes.pid] = -1
                            freeSlots.append(freeS)

                elapsed = (time.time() - start) / 60.0
                hours = ""
                if elapsed >= 60.0:
                    if elapsed < 120.0:
                        hours = "1 hour "
                    else:
                        hours = "%d hours " % (elapsed // 60)

                pwMain("text")(
                    "%s %.2f min elapsed %d / %d targets done (%d parallel)"
                    % (
                        hours,
                        elapsed % 60,
                        completed,
                        len(self.MSMSTargetModel._data),
                        min(cpus, len(targetsToProc)),
                    )
                )
                time.sleep(0.5)

        pw.setSkipCallBack(True)
        pw.hide()

        # Log time used for processing of individual files
        elapsed = (time.time() - start) / 60.0
        hours = ""
        if elapsed >= 60.0:
            if elapsed < 120.0:
                hours = "1 hour "
            else:
                hours = "%d hours " % (elapsed // 60)
        mins = "%.2f min(s)" % (elapsed % 60.0)

        if not self.terminateJobs:
            logging.info("Individual files processed (%s%s).." % (hours, mins))

        p.close()
        p.terminate()
        p.join()

        self.updateResultsView()

    def openFileExternally(self):
        curFile = str(self.processedFilesComboBox.currentText())
        if curFile in self.mappedResultsToFile.keys():
            os.startfile(self.mappedResultsToFile[curFile])

    def updateResultsView(self):
        files = sorted(list(set([d.lcmsmsNativeFileName for d in self.MSMSTargetModel._data])))

        self.processedFilesComboBox.clear()
        self.processedFilesComboBox.addItem("--")

        self.mappedResultsToFile = {}
        for file in files:
            file = str(file)
            if os.path.isfile(file + ".identified.sqlite"):
                a = file.replace("\\", "/")
                a = a[a.rfind("/") + 1 :]

                self.processedFilesComboBox.addItem(a)
                self.mappedResultsToFile[a] = file

    def closeActiveResultsFile(self):
        if self.curFileSQLiteConnection is not None:
            self.curFileSQLiteConnection.curs.close()
            self.curFileSQLiteConnection.conn.close()
            self.curFileSQLiteConnection = None

    def setActiveResultsFile(self):
        self.closeActiveResultsFile()

        if str(self.processedFilesComboBox.currentText()) == "--":
            return

        curFile = str(self.processedFilesComboBox.currentText())
        if curFile in self.mappedResultsToFile.keys():
            curFile = self.mappedResultsToFile[curFile]

            self.curFileSQLiteConnection = Bunch(conn=sqlite3.connect(curFile + ".identified.sqlite"))
            self.curFileSQLiteConnection.curs = self.curFileSQLiteConnection.conn.cursor()

            self.resultsTreeWidget.clear()

            targetsItem = QtWidgets.QTreeWidgetItem(["MSMS targets"])
            targetsItem.myType = "MSMSTargets_TL"
            targetsItem.myID = 1
            targetsItem.myData = []

            targets = [
                t
                for t in SQLSelectAsObject(
                    self.curFileSQLiteConnection.curs,
                    selectStatement="select id, targetName, precursorMZ, chargeCount, Cn, startRT, stopRT, scanEventMS1M, scanEventMS1Mp, scanEventMS2Native, scanEventMS2Labeled, scanIDNativeRaw, scanIDLabelledRaw, scanRTNativeRaw, scanRTLabeledRaw, parentSumFormula from Targets",
                )
            ]
            for target in targets:
                targetItem = QtWidgets.QTreeWidgetItem(
                    [
                        target.targetName,
                        "%s" % str(target.precursorMZ),
                        "%d" % target.Cn,
                        str(target.parentSumFormula),
                        "%d" % target.chargeCount,
                        "%.2f (%s)" % (target.scanRTNativeRaw / 60.0, target.scanIDNativeRaw),
                        "%.2f (%s)" % (target.scanRTLabeledRaw / 60.0, target.scanIDLabelledRaw),
                        target.scanEventMS2Native,
                        target.scanEventMS2Labeled,
                        target.scanEventMS1M,
                        target.scanEventMS1Mp,
                    ]
                )
                targetItem.myType = "MSMSTarget"
                targetItem.myID = target.id
                targetItem.myData = target
                targetsItem.addChild(targetItem)
                targetsItem.myData.append(target.id)

                annotatedSpectrum = [
                    p
                    for p in SQLSelectAsObject(
                        self.curFileSQLiteConnection.curs,
                        selectStatement="select mzs, ints, annos as annos from MSSpectra where forTarget=%d and type='native_cleaned'" % target.id,
                    )
                ][0]

                if len(annotatedSpectrum.mzs) > 0:
                    annotatedSpectrum.mzs = [float(mz) for mz in annotatedSpectrum.mzs.split(",")]
                else:
                    annotatedSpectrum.mzs = []

                if len(annotatedSpectrum.ints) > 0:
                    annotatedSpectrum.ints = [float(i) for i in annotatedSpectrum.ints.split(",")]
                else:
                    annotatedSpectrum.ints = []

                if len(annotatedSpectrum.annos) > 0:
                    annotatedSpectrum.annos = pickle.loads(base64.b64decode(annotatedSpectrum.annos))
                else:
                    annotatedSpectrum.annos = []

                for index in range(len(annotatedSpectrum.mzs)):
                    b = Bunch(
                        index=index,
                        mz=annotatedSpectrum.mzs[index],
                        relInt=annotatedSpectrum.ints[index],
                        Cn=annotatedSpectrum.annos[index].Cn,
                        sumFormulas=annotatedSpectrum.annos[index].generatedSumFormulas,
                        dppmSpectra=annotatedSpectrum.annos[index].dppmSpectra,
                        ratioError=annotatedSpectrum.annos[index].ratioError,
                        ratioSimilarity=annotatedSpectrum.annos[index].ratioSimilarity,
                    )
                    peakItem = QtWidgets.QTreeWidgetItem(
                        [
                            "Peak %d" % b.index,
                            "%.4f" % b.mz,
                            "%d (mz %.4f; dppm %.2f)" % (b.Cn, b.mz + 1.00335 * b.Cn, b.dppmSpectra),
                            "",
                            "",
                            "",
                            "Rel.int: %.1f%%" % (b.relInt),
                            "Ratio: %.5f" % (b.ratioSimilarity),
                        ]
                    )
                    peakItem.myType = "PeakAnnotation"
                    peakItem.myID = target.id
                    peakItem.myData = b
                    for io in range(8):
                        peakItem.setBackground(io, QtGui.QColor("lightgrey"))
                    targetItem.addChild(peakItem)

                    totSumForms = 0
                    for key, val in b.sumFormulas.items():
                        totSumForms += len(val)
                        if len(val) == 0:
                            del b.sumFormulas[key]

                    if totSumForms == 0:
                        pass
                    elif totSumForms == 1:
                        peakItem.setData(
                            3,
                            QtCore.Qt.DisplayRole,
                            str(list(b.sumFormulas.values())[0][0].sumFormula),
                        )
                        peakItem.setData(4, QtCore.Qt.DisplayRole, str(list(b.sumFormulas.keys())[0]))
                        peakItem.setData(
                            5,
                            QtCore.Qt.DisplayRole,
                            str(list(b.sumFormulas.values())[0][0].neutralLossToParent),
                        )
                    else:
                        peakItem.setData(3, QtCore.Qt.DisplayRole, "*")
                        for adduct in b.sumFormulas.keys():
                            if len(b.sumFormulas[adduct]) > 0:
                                for sumForm in b.sumFormulas[adduct]:
                                    sumFormItem = QtWidgets.QTreeWidgetItem(
                                        [
                                            "",
                                            "",
                                            "",
                                            sumForm.sumFormula,
                                            adduct,
                                            str(sumForm.neutralLossToParent),
                                        ]
                                    )
                                    sumFormItem.myType = "PeakAnnotation_Adduct_SumFormula"
                                    sumFormItem.myID = target.id
                                    sumFormItem.myData = b
                                    for io in range(8):
                                        sumFormItem.setBackground(io, QtGui.QColor("lightgrey"))
                                    peakItem.addChild(sumFormItem)

                mbMatches = [
                    p
                    for p in SQLSelectAsObject(
                        self.curFileSQLiteConnection.curs,
                        selectStatement="SELECT id, title, exactMass, formula, score FROM MassBankHits WHERE forTarget=%d" % target.id,
                    )
                ]
                if len(mbMatches) > 0:
                    annos = QtWidgets.QTreeWidgetItem(["MassBank hits (%d)" % len(mbMatches)])
                    annos.myType = "MassBankHits"
                    annos.myID = -1
                    annos.myData = Bunch()
                    for p in mbMatches:
                        anno = QtWidgets.QTreeWidgetItem(
                            [
                                str(d)
                                for d in [
                                    p.title,
                                    p.id,
                                    p.formula,
                                    p.exactMass,
                                    p.score,
                                ]
                            ]
                        )
                        anno.targetID = target.id
                        anno.myType = "MassBankHit"
                        anno.myID = p.id
                        anno.myData = p
                        annos.addChild(anno)
                        targetItem.addChild(annos)

            self.resultsTreeWidget.addTopLevelItem(targetsItem)
            self.resultsTreeWidget.expandItem(targetsItem)

    def fetchAndPlotMSMSTarget(self, id, highlightCleandPeak=None, showMassBankSpectra=None):
        target = [
            p
            for p in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT targetName, parentSumFormula, scanIDNativeRaw, scanIDLabelledRaw, scanRTNativeRaw, scanRTLabeledRaw, Cn, scanEventMS2Native, id, scanEventMS2Labeled, precursorMZ, chargeCount, eicFS, eicLFS, timesFS, timesLFS FROM Targets WHERE id=%d" % (id),
            )
        ]

        assert len(target) == 1
        target = target[0]

        target.eicFS = pickle.loads(base64.b64decode(target.eicFS))
        target.eicLFS = pickle.loads(base64.b64decode(target.eicLFS))
        target.timesFS = pickle.loads(base64.b64decode(target.timesFS))
        target.timesLFS = pickle.loads(base64.b64decode(target.timesLFS))

        nativeSpectra = [
            spec
            for spec in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT scanID, mzs, ints, type FROM MSSpectra WHERE forTarget=%d AND type='native_raw'" % id,
            )
        ][0]
        nativeSpectra.mzs = [float(mz) for mz in nativeSpectra.mzs.split(",")]
        nativeSpectra.ints = [float(i) for i in nativeSpectra.ints.split(",")]
        labelledSpectra = [
            spec
            for spec in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT scanID, mzs, ints, type FROM MSSpectra WHERE forTarget=%d AND type='labelled_raw'" % id,
            )
        ][0]
        labelledSpectra.mzs = [float(mz) for mz in labelledSpectra.mzs.split(",")]
        labelledSpectra.ints = [float(i) for i in labelledSpectra.ints.split(",")]

        nativeSpectraCleaned = [
            spec
            for spec in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT mzs, ints, annos, type FROM MSSpectra WHERE forTarget=%d AND type='native_cleaned'" % id,
            )
        ][0]
        labelledSpectraCleaned = [
            spec
            for spec in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT mzs, ints, annos, type FROM MSSpectra WHERE forTarget=%d AND type='labelled_cleaned'" % id,
            )
        ][0]

        if len(nativeSpectraCleaned.mzs) > 0:
            nativeSpectraCleaned.mzs = [float(mz) for mz in nativeSpectraCleaned.mzs.split(",")]
        else:
            nativeSpectraCleaned.mzs = []

        if len(nativeSpectraCleaned.ints) > 0:
            nativeSpectraCleaned.ints = [float(i) for i in nativeSpectraCleaned.ints.split(",")]
        else:
            nativeSpectraCleaned.ints = []

        if len(nativeSpectraCleaned.annos) > 0:
            nativeSpectraCleaned.annos = pickle.loads(base64.b64decode(nativeSpectraCleaned.annos))
        else:
            nativeSpectraCleaned.annos = []

        if len(labelledSpectraCleaned.mzs) > 0:
            labelledSpectraCleaned.mzs = [float(mz) for mz in labelledSpectraCleaned.mzs.split(",")]
        else:
            labelledSpectraCleaned.mzs = []

        if len(labelledSpectraCleaned.ints) > 0:
            labelledSpectraCleaned.ints = [float(i) for i in labelledSpectraCleaned.ints.split(",")]
        else:
            labelledSpectraCleaned.ints = []

        if len(labelledSpectraCleaned.annos) > 0:
            labelledSpectraCleaned.annos = pickle.loads(base64.b64decode(labelledSpectraCleaned.annos))
        else:
            labelledSpectraCleaned.annos = []

        if self.plotNativeScanCheckbox.checkState() == QtCore.Qt.Checked:
            self.pl1.twinxs[0].vlines(
                x=[target.precursorMZ],
                ymin=[0],
                ymax=[125],
                color="Firebrick",
                linestyle="--",
            )
            self.pl1.twinxs[0].text(x=target.precursorMZ - 1, y=128, s="PC", color="Firebrick")
            self.pl1.twinxs[0].vlines(
                x=nativeSpectra.mzs,
                ymin=[0 for mz in nativeSpectra.mzs],
                ymax=nativeSpectra.ints,
                color=(101 / 255.0, 101 / 255.0, 101 / 255.0),
            )

        if self.plotLabelledScanCheckBox.checkState() == QtCore.Qt.Checked:
            self.pl1.twinxs[0].vlines(
                x=[target.precursorMZ + target.Cn * 1.00335 / target.chargeCount],
                ymin=[0],
                ymax=[-125],
                color="Firebrick",
                linestyle="--",
            )
            self.pl1.twinxs[0].text(
                x=target.precursorMZ + target.Cn * 1.00335 / target.chargeCount - 1,
                y=-136,
                s="PC'",
                color="Firebrick",
            )
            self.pl1.twinxs[0].vlines(
                x=labelledSpectra.mzs,
                ymin=[0 for mz in labelledSpectra.mzs],
                ymax=[-i for i in labelledSpectra.ints],
                color=(249 / 255.0, 164 / 255.0, 78 / 255.0),
            )

        if self.plotCleanedScanCheckBox.checkState() == QtCore.Qt.Checked:
            self.pl1.twinxs[0].vlines(
                x=nativeSpectraCleaned.mzs,
                ymin=[0 for mz in nativeSpectraCleaned.mzs],
                ymax=nativeSpectraCleaned.ints,
                color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                linewidth=2,
                alpha=1 if highlightCleandPeak is None else 0.5,
            )
            self.pl1.twinxs[0].vlines(
                x=[nativeSpectraCleaned.mzs[i] + nativeSpectraCleaned.annos[i].Cn * 1.00335 for i in range(len(nativeSpectraCleaned.mzs))],
                ymin=[-i + 0.5 for i in nativeSpectraCleaned.ints],
                ymax=[-i - 0.5 for i in nativeSpectraCleaned.ints],
                color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                linewidth=4,
                alpha=1 if highlightCleandPeak is None else 0.5,
            )

            if highlightCleandPeak is not None:
                self.pl1.twinxs[0].vlines(
                    x=nativeSpectraCleaned.mzs[highlightCleandPeak],
                    ymin=0,
                    ymax=nativeSpectraCleaned.ints[highlightCleandPeak],
                    color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                    linewidth=2,
                )
                self.pl1.twinxs[0].vlines(
                    x=nativeSpectraCleaned.mzs[highlightCleandPeak] + nativeSpectraCleaned.annos[highlightCleandPeak].Cn * 1.00335,
                    ymin=-nativeSpectraCleaned.ints[highlightCleandPeak] - 0.5,
                    ymax=-nativeSpectraCleaned.ints[highlightCleandPeak] + 0.5,
                    color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                    linewidth=2,
                )

        if self.plotCleanedScanCheckBox.checkState() == QtCore.Qt.Checked:
            self.pl1.twinxs[0].vlines(
                x=labelledSpectraCleaned.mzs,
                ymin=[0 for mz in labelledSpectraCleaned.mzs],
                ymax=[-i for i in labelledSpectraCleaned.ints],
                color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                linewidth=2,
                alpha=1 if highlightCleandPeak is None else 0.5,
            )
            self.pl1.twinxs[0].vlines(
                x=[labelledSpectraCleaned.mzs[i] - labelledSpectraCleaned.annos[i].Cn * 1.00335 for i in range(len(labelledSpectraCleaned.mzs))],
                ymin=[i + 0.5 for i in labelledSpectraCleaned.ints],
                ymax=[i - 0.5 for i in labelledSpectraCleaned.ints],
                color=(148 / 255.0, 32 / 255.0, 146 / 255.0),
                linewidth=4,
                alpha=1 if highlightCleandPeak is None else 0.5,
            )

        for index in range(len(nativeSpectraCleaned.annos)):
            if index == highlightCleandPeak:
                self.pl1.twinxs[0].text(
                    x=nativeSpectraCleaned.mzs[index],
                    y=nativeSpectraCleaned.ints[index] + 1,
                    s="%d" % index,
                    color="black",
                    horizontalalignment="center",
                )
                self.pl1.twinxs[0].text(
                    x=labelledSpectraCleaned.mzs[index],
                    y=-labelledSpectraCleaned.ints[index] - 1,
                    s="%d" % index,
                    color="black",
                    horizontalalignment="center",
                )
            else:
                self.pl1.twinxs[0].text(
                    x=nativeSpectraCleaned.mzs[index],
                    y=nativeSpectraCleaned.ints[index] + 1,
                    s="%d" % index,
                    color="lightgrey",
                    horizontalalignment="center",
                )
                self.pl1.twinxs[0].text(
                    x=labelledSpectraCleaned.mzs[index],
                    y=-labelledSpectraCleaned.ints[index] - 1,
                    s="%d" % index,
                    color="lightgrey",
                    horizontalalignment="center",
                )

        eics = [
            eic
            for eic in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT intensityList, timesList, forMZ, type FROM EICs WHERE forTarget=%d" % id,
            )
        ]

        try:
            for eic in eics:
                times = [float(f) for f in eic.timesList.split(",")]
                intensities = [float(f) for f in eic.intensityList.split(",")]

                times = [t / 60.0 for t in times]

                m = max(intensities)
                if m == 0:
                    m = 1
                intensities = [i / m for i in intensities]

                if eic.type == "Labeled":
                    intensities = [-i for i in intensities]

                col = "lightgrey"
                linewidth = 1

                self.pl2.twinxs[0].plot(times, intensities, color=col, linewidth=linewidth)
        except Exception:
            print(eic.timesList, eic.intensityList)

        try:
            for eic in eics:
                times = [float(f) for f in eic.timesList.split(",")]
                intensities = [float(f) for f in eic.intensityList.split(",")]

                times = [t / 60.0 for t in times]

                m = max(intensities)
                if m == 0:
                    m = 1
                intensities = [i / m for i in intensities]

                if eic.type == "Labeled":
                    intensities = [-i for i in intensities]

                if highlightCleandPeak is not None and abs(nativeSpectraCleaned.mzs[highlightCleandPeak] - eic.forMZ) <= 0.0001:
                    col = (148 / 255.0, 32 / 255.0, 146 / 255.0)
                    linewidth = 2
                    self.pl2.twinxs[0].plot(times, intensities, color=col, linewidth=linewidth)
        except Exception:
            print(eic.timesList, eic.intensityList)

        ma = max(target.eicFS)
        ma = ma if ma > 0 else 1
        self.pl2.twinxs[0].plot(
            [t / 60.0 for t in target.timesFS],
            [i / ma for i in target.eicFS],
            color="black",
        )
        ma = max(target.eicLFS)
        ma = ma if ma > 0 else 1
        self.pl2.twinxs[0].plot(
            [t / 60.0 for t in target.timesLFS],
            [-i / ma for i in target.eicLFS],
            color="black",
        )

        self.pl2.twinxs[0].plot(
            [target.scanRTNativeRaw / 60.0, target.scanRTNativeRaw / 60.0],
            [0, 1],
            color="firebrick",
        )
        self.pl2.twinxs[0].plot(
            [target.scanRTLabeledRaw / 60.0, target.scanRTLabeledRaw / 60.0],
            [0, -1],
            color="firebrick",
        )

        if showMassBankSpectra is not None:
            p = [
                p
                for p in SQLSelectAsObject(
                    self.curFileSQLiteConnection.curs,
                    selectStatement="SELECT mzs, relInts FROM MassBankHits WHERE id='%s'" % showMassBankSpectra,
                )
            ][0]
            mzs = [float(d) for d in p.mzs.split(";")]
            relInts = [float(d) for d in p.relInts.split(";")]
            maxVal = max(relInts)
            relInts = [100 * d / maxVal for d in relInts]

            self.pl1.twinxs[0].vlines(
                x=mzs,
                ymin=[0 for i in relInts],
                ymax=[-i for i in relInts],
                color="green",
            )

    def updateResultsIllustration(self):
        clearPlot(self.pl1)
        clearPlot(self.pl2)
        for item in self.resultsTreeWidget.selectedItems():
            if hasattr(item, "myType") and item.myType == "MSMSTarget":
                self.fetchAndPlotMSMSTarget(id=item.myID)
            if hasattr(item, "myType") and item.myType in [
                "PeakAnnotation",
                "PeakAnnotation_Adduct",
                "PeakAnnotation_Adduct_SumFormula",
            ]:
                self.fetchAndPlotMSMSTarget(id=item.myID, highlightCleandPeak=item.myData.index)
            if hasattr(item, "myType") and item.myType in ["MassBankHit"]:
                self.fetchAndPlotMSMSTarget(id=item.targetID, showMassBankSpectra=item.myData.id)

        drawCanvas(self.pl1)
        drawCanvas(self.pl2)

    def getClipboardRepresentationOfTarget(self, copyTargetID):
        dat = []

        target = [
            p
            for p in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT targetName, parentSumFormula, scanIDNativeRaw, scanIDLabelledRaw, Cn, scanEventMS2Native, id, scanEventMS2Labeled, precursorMZ, chargeCount FROM Targets WHERE id=%d" % (copyTargetID),
            )
        ]
        assert len(target) == 1
        target = target[0]
        dat.append(
            [
                "Num",
                "TargetName",
                "ParentSumFormula",
                "Cn",
                "PrecursorMZ",
                "NativeScanNum",
                "LabelledScanNum",
                "NativeMSMSScanEvent",
                "LabelledMSMSScanEvent",
                "chargeCount",
            ]
        )
        dat.append(
            [
                "%d" % target.id,
                str(target.targetName),
                str(target.parentSumFormula),
                "%d" % target.Cn,
                "%.4f" % target.precursorMZ,
                str(target.scanIDNativeRaw),
                str(target.scanIDLabelledRaw),
                str(target.scanEventMS2Native),
                str(target.scanEventMS2Labeled),
                "%d" % target.chargeCount,
            ]
        )
        dat.append("|")

        dat.append(
            [
                "|",
                "Num",
                "MZ",
                "RelativeIntensity [%]",
                "Cn",
                "Adduct",
                "SumFormula",
                "NeutralLossToParent",
                "RelPPMError",
            ]
        )
        annotatedSpectrum = [
            p
            for p in SQLSelectAsObject(
                self.curFileSQLiteConnection.curs,
                selectStatement="SELECT mzs, ints, annos AS annos FROM MSSpectra WHERE forTarget=%d AND type='native_cleaned'" % copyTargetID,
            )
        ][0]
        if len(annotatedSpectrum.mzs) > 0:
            annotatedSpectrum.mzs = [float(mz) for mz in annotatedSpectrum.mzs.split(",")]
        else:
            annotatedSpectrum.mzs = []
        if len(annotatedSpectrum.ints) > 0:
            annotatedSpectrum.ints = [float(i) for i in annotatedSpectrum.ints.split(",")]
        else:
            annotatedSpectrum.ints = []
        if len(annotatedSpectrum.annos) > 0:
            annotatedSpectrum.annos = pickle.loads(base64.b64decode(annotatedSpectrum.annos))
        else:
            annotatedSpectrum.annos = []
        for index in range(len(annotatedSpectrum.mzs)):
            b = Bunch(
                index=index,
                mz=annotatedSpectrum.mzs[index],
                relInt=annotatedSpectrum.ints[index],
                Cn=annotatedSpectrum.annos[index].Cn,
                sumFormulas=annotatedSpectrum.annos[index].generatedSumFormulas,
            )

            genSumForms = []
            for adduct in b.sumFormulas.keys():
                for sumForm in b.sumFormulas[adduct]:
                    genSumForms.append(
                        [
                            "|",
                            "",
                            "*",
                            "",
                            "",
                            adduct,
                            sumForm.sumFormula,
                            sumForm.neutralLossToParent,
                            "%.2f" % sumForm.deltaPPM,
                        ]
                    )

            if len(genSumForms) == 1:
                genSumForm = genSumForms[0]
                genSumForm[0:5] = [
                    "|",
                    "%d" % b.index,
                    "%.4f" % b.mz,
                    "%.1f" % (b.relInt),
                    "%d" % b.Cn,
                ]
                dat.append(genSumForm)
            elif len(genSumForms) > 1:
                dat.append(
                    [
                        "|",
                        "%d" % b.index,
                        "%.4f" % b.mz,
                        "%.1f" % (b.relInt),
                        "%d" % b.Cn,
                        "*",
                        "",
                        "",
                    ]
                )
                dat.extend(genSumForms)
            else:
                dat.append(
                    [
                        "|",
                        "%d" % b.index,
                        "%.4f" % b.mz,
                        "%.1f" % (b.relInt),
                        "%d" % b.Cn,
                        "",
                        "",
                        "",
                    ]
                )

        dat = ["\t".join(d) for d in dat]

        return dat

    def showResultsPopup(self, position):
        selectedItems = self.resultsTreeWidget.selectedItems()

        dat = []

        if len(selectedItems) == 1 and selectedItems[0].myType == "MSMSTargets_TL":
            menu = QtWidgets.QMenu()
            clipboardAction = menu.addAction("Copy")

            action = menu.exec_(self.resultsTreeWidget.mapToGlobal(position))
            if action == clipboardAction:
                for targetid in selectedItems[0].myData:
                    d = self.getClipboardRepresentationOfTarget(targetid)
                    if len(dat) > 0:
                        del d[0]
                        dat.extend(["", ""])
                    dat.extend(d)

        for selectedItem in selectedItems:
            if selectedItem.myType == "MSMSTarget":
                menu = QtWidgets.QMenu()
                clipboardAction = menu.addAction("Copy")

                action = menu.exec_(self.resultsTreeWidget.mapToGlobal(position))
                if action == clipboardAction:
                    copyTargetID = selectedItem.myID
                    dat.extend(self.getClipboardRepresentationOfTarget(copyTargetID))

        if len(dat) > 0:
            pyperclip.copy("\r\n".join(dat))

        # else:
        #    menu = QtWidgets.QMenu()
        #    menu.addAction("No action avaialble")
        #    menu.exec_(self.resultsTreeWidget.mapToGlobal(position))


def main():
    """Entry point for fragextract command"""
    # add freeze support (required for multiprocessing)
    freeze_support()

    import sys

    app = QtWidgets.QApplication(sys.argv)
    _style_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    with open(_style_path, "r") as _f:
        app.setStyleSheet(_f.read())
    Dialog = FEMainWindow()

    Dialog.show()
    x = app.exec()

    sys.exit(x)


if __name__ == "__main__":
    main()
