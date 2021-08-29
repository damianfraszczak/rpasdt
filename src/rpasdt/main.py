import sys

from PyQt5 import QtWidgets

from rpasdt.common.exceptions import log_error
from rpasdt.controller.app import AppController
from rpasdt.gui.main_window import MainWindow

sys.excepthook = log_error
app = QtWidgets.QApplication([])
controller = AppController()
window = MainWindow(controller=controller)
window.show()
app.exec_()
