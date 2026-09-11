from __future__ import absolute_import, division, print_function
import os
from PySide6 import QtCore, QtGui, QtWidgets
from ..rawImport import UnsupportedFileError, prepare_import_files
from ..utils import natSort
from .groupEditor import Ui_GroupEditor


class groupEdit(QtWidgets.QDialog, Ui_GroupEditor):
    def __init__(self, parent=None, initDir=None, colors=["Red", "Blue", "Green"], activeColor=0):
        self.groupfiles = []
        if initDir is not None:
            self.initDir = initDir
        else:
            self.initDir = "."
            if "USERPROFILE" in os.environ:
                self.initDir = os.getenv("USERPROFILE")
            elif "HOME" in os.environ:
                self.initDir = os.getenv("HOME")

        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle("Group editor")

        self.dialogFinished.clicked.connect(self.dialogFin)
        self.dialogCanceled.clicked.connect(self.dialogCan)
        self.addFiles.clicked.connect(self.selectFiles)

        self.addFolder.setVisible(False)
        self.addFolder.clicked.connect(self.selectFolder)
        self.removeSelected.clicked.connect(self.removeSel)
        self.useAsMSMSTarget.setVisible(False)

        self.colors = list(colors)

        # Replace color combo box with a color picker button
        self.colorsComboBox.setVisible(False)

        self._colorButton = QtWidgets.QPushButton()
        self._colorButton.setToolTip("Click to choose a color")
        self._colorButton.clicked.connect(self._pickColor)
        self.gridLayout_3.addWidget(self._colorButton, 0, 2, 1, 2)

        # Set initial color
        if isinstance(activeColor, int) and 0 <= activeColor < len(self.colors):
            self._selectedColor = self.colors[activeColor]
        elif isinstance(activeColor, str):
            self._selectedColor = activeColor
        else:
            self._selectedColor = self.colors[0] if self.colors else "Red"
        self._styleColorButton()

        self.dialogFinished.setFocus()

        self.groupName.setValidator(QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9a-zA-Z_]*")))
        self.setAcceptDrops(True)

    def removeSel(self):
        todel = []
        for selectedIndex in self.groupFiles.selectedIndexes():
            todel.append(selectedIndex.row())
        todel.sort()
        todel.reverse()

        for ind in todel:
            self.groupFiles.takeItem(ind)
            self.groupfiles.pop(ind)

    def dialogCan(self):
        self.reject()

    def dialogFin(self):
        if len(self.getGroupName()) < 1:
            QtWidgets.QMessageBox.information(
                None,
                "Group name",
                "Please specify a group name",
                QtWidgets.QMessageBox.Ok,
            )
            return
        if len(self.getGroupFiles()) < 1:
            QtWidgets.QMessageBox.information(
                None,
                "Group files",
                "Please load one or more files into this group",
                QtWidgets.QMessageBox.Ok,
            )
            return

        self.accept()

    def getGroupName(self):
        return str(self.groupName.text())

    def getGroupFiles(self):
        return [str(t) for t in self.groupfiles]

    def getMinimumGroupFound(self):
        return int(self.groupMinimumFound.value())

    def getOmitFeatures(self):
        return self.omitFeatures.isChecked()

    def getUseForMetaboliteGrouping(self):
        return bool(self.useForMetaboliteGrouping.checkState() == QtCore.Qt.Checked)

    def getRemoveAsFalsePositive(self):
        return bool(self.removeAsFalsePositive.checkState() == QtCore.Qt.Checked)

    def getOpenDir(self):
        return self.initDir

    def getGroupColor(self):
        return self._selectedColor

    def _pickColor(self):
        """Open a QColorDialog and update the button."""
        current = QtGui.QColor(self._selectedColor)
        chosen = QtWidgets.QColorDialog.getColor(current, self, "Choose group color")
        if chosen.isValid():
            self._selectedColor = chosen.name()
            self._styleColorButton()

    def _styleColorButton(self):
        """Update button appearance to show the current colour."""
        qc = QtGui.QColor(self._selectedColor)
        if not qc.isValid():
            qc = QtGui.QColor("gray")
        luminance = 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()
        fg = "black" if luminance > 128 else "white"
        self._colorButton.setText(self._selectedColor)
        self._colorButton.setStyleSheet("background-color: %s; color: %s; border: 1px solid gray; padding: 2px 8px;" % (qc.name(), fg))

    def getUseAsMSMSTarget(self):
        return False

    def selectFiles(self):
        filenames, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            caption="Select group file(s)",
            dir=self.initDir,
            filter="mzXML (*.mzxml);;mzML (*.mzml);;group file (*.grp);;All files (*.*)",
        )
        filenames = natSort(filenames)
        raw_filenames = [filename for filename in filenames if filename.lower().endswith(".raw")]
        try:
            converted_raw_filenames = prepare_import_files(raw_filenames) if raw_filenames else []
        except UnsupportedFileError:
            return
        converted_raw_by_source = dict(zip(raw_filenames, converted_raw_filenames))

        for filename in filenames:
            if filename.lower().endswith(".raw"):
                filename = converted_raw_by_source[filename]
            if filename.lower().endswith(".mzxml") or filename.lower().endswith(".mzml"):
                self.groupFiles.addItem(filename.replace("\\", "/"))
                self.groupfiles.append(filename)
        if len(filenames) > 0:
            self.initDir = str(filenames[0]).replace("\\", "/")
            self.initDir = self.initDir[: self.initDir.rfind("/")]

    def selectFolder(self):
        foldername = str(
            QtWidgets.QFileDialog.getExistingDirectory(self, caption="Select folder containing mzxml files"),
            directory=self.initDir,
        )
        if len(foldername) > 0:
            self.initDir = foldername.replace("\\", "/")
            for root, dirs, files in os.walk(foldername):
                for file in files:
                    if file.lower().endswith(".mzxml") or file.lower().endswith(".mzml"):
                        self.groupFiles.addItem(root + "/" + file)
                        self.groupfiles.append(root + "/" + file)

    def executeDialog(
        self,
        groupName="",
        groupfiles=[],
        minimumGroupFound=1,
        omitFeatures=True,
        useForMetaboliteGrouping=True,
        removeAsFalsePositive=False,
        activeColor="Red",
        useAsMSMSTarget=False,
    ):
        for file in groupfiles:
            self.groupFiles.addItem(file)
            self.groupfiles.append(file)

            self.initDir = str(file).replace("\\", "/")
            self.initDir = self.initDir[: self.initDir.rfind("/")]

        # Set color via the picker button
        self._selectedColor = activeColor
        self._styleColorButton()

        self.groupName.setText(groupName)
        self.groupMinimumFound.setValue(minimumGroupFound)
        if omitFeatures:
            self.omitFeatures.setChecked(True)
        else:
            self.omitFeatures.setChecked(False)
        if useForMetaboliteGrouping:
            self.useForMetaboliteGrouping.setCheckState(QtCore.Qt.Checked)
        else:
            self.useForMetaboliteGrouping.setCheckState(QtCore.Qt.Unchecked)
        if removeAsFalsePositive:
            self.removeAsFalsePositive.setCheckState(QtCore.Qt.Checked)
        else:
            self.removeAsFalsePositive.setCheckState(QtCore.Qt.Unchecked)

        self.useAsMSMSTarget.setCheckState(QtCore.Qt.Unchecked)

        self.dialogFinished.setFocus()
        return self.exec()

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
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()).replace("\\", "/"))

            links = natSort(links)
            raw_links = [link for link in links if link.lower().endswith(".raw")]
            try:
                converted_raw_links = prepare_import_files(raw_links) if raw_links else []
            except UnsupportedFileError:
                return
            converted_raw_by_source = dict(zip(raw_links, converted_raw_links))
            for link in links:
                if link.lower().endswith(".raw"):
                    link = converted_raw_by_source[link]
                if link.lower().endswith(".mzxml") or link.lower().endswith(".mzml"):
                    self.groupFiles.addItem(link)
                    self.groupfiles.append(link)

        else:
            event.ignore()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = groupEdit()

    Dialog.executeDialog()
    x = app.exec()
    sys.exit(x)
