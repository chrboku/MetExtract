from PySide6 import QtWidgets
from .calcIsotopeEnrichmentDialog import Ui_Dialog
from ..formulaTools import calcIsoEnrichment


class calcIsoEnrichmentDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, parent=None, initDir=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)
        self.setWindowTitle("Calculate isotopic enrichment")

        self.tableWidget.setColumnWidth(0, 120)
        self.tableWidget.setColumnWidth(1, 120)
        self.tableWidget.setColumnWidth(2, 120)
        self.tableWidget.setColumnWidth(3, 120)
        self.tableWidget.setColumnWidth(4, 60)
        self.tableWidget.setColumnWidth(5, 150)
        self.tableWidget.setColumnWidth(6, 150)
        self.tableWidget.cellChanged.connect(self.updateTable)

    def updateTable(self, x, y):
        if y < 5:
            rowi = x
            try:
                m = float(self.tableWidget.item(rowi, 0).text())
                mpo = float(self.tableWidget.item(rowi, 1).text())
                cn = int(self.tableWidget.item(rowi, 4).text())

                self.tableWidget.setItem(
                    rowi,
                    5,
                    QtWidgets.QTableWidgetItem("%.4f%%" % (100.0 * calcIsoEnrichment(cn, 1, mpo / m))),
                )
            except Exception:
                pass
            try:
                mpmo = float(self.tableWidget.item(rowi, 2).text())
                mp = float(self.tableWidget.item(rowi, 3).text())
                cn = int(self.tableWidget.item(rowi, 4).text())

                self.tableWidget.setItem(
                    rowi,
                    6,
                    QtWidgets.QTableWidgetItem("%.4f%%" % (100.0 * calcIsoEnrichment(cn, 1, mpmo / mp))),
                )
            except Exception:
                pass

    def executeDialog(self):
        x = self.exec()

        return x


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Dialog = calcIsoEnrichmentDialog()

    Dialog.show()
    app.exec()

    sys.exit()
