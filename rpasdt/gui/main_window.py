from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMdiArea, QMdiSubWindow, qApp

from rpasdt.controller.app import AppController
from rpasdt.controller.diffusion import DiffusionGraphController
from rpasdt.controller.source_detection import SourceDetectionGraphController
from rpasdt.gui.analysis.analysis import MainNetworkGraphPanel
from rpasdt.gui.analysis.diffusion import DiffusionGraphPanel
from rpasdt.gui.analysis.source_detection import SourceDetectionDialog
from rpasdt.gui.utils import create_action
from rpasdt.model.constants import APP_ICON_PATH, APP_NAME
from rpasdt.model.experiment import DiffusionExperiment, Experiment


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: AppController, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.controller = controller
        self.controller.window = self
        self.configure_gui()

    def add_subwindow_with_widget(self, widget, title, show=True):
        sub_window = QMdiSubWindow()
        sub_window.setWidget(widget)
        sub_window.setWindowTitle(title)
        sub_window.setWindowIcon(QIcon(APP_ICON_PATH))
        self.add_subwindow(sub_window, show)

    def add_subwindow(self, sub_window, show=True):
        self.mdi.addSubWindow(sub_window)

        if show:
            sub_window.show()
            self.layout_windows_tile()

    def layout_windows_tile(self):
        self.mdi.tileSubWindows()

    def layout_windows_cascade(self):
        self.mdi.cascadeSubWindows()

    def show_experiment_window(
        self, experiment: Experiment, controller: "ExperimentGraphController"
    ):
        network_widget = MainNetworkGraphPanel(
            controller=controller, title="Initial structure"
        )
        network_widget.node_clicked.connect(controller.handler_graph_node_clicked)
        self.add_subwindow_with_widget(
            title=f"Network of {experiment.name}", widget=network_widget
        )

    def show_diffusion_window(
        self, experiment: DiffusionExperiment, controller: DiffusionGraphController
    ):
        diffusion_widget = DiffusionGraphPanel(controller=controller)
        self.add_subwindow_with_widget(
            title=f"Diffusion simulation with {experiment.diffusion_type}",
            widget=diffusion_widget,
        )
        controller.handler_edit_diffusion()

    def show_source_detection_window(self, controller: SourceDetectionGraphController):
        self.add_subwindow(
            SourceDetectionDialog(
                title=f"Source detection with {controller.source_detector}",
                controller=controller,
            )
        )

    def configure_gui(self):
        self.createMenuBar()
        self.mdi = QMdiArea()
        self.setCentralWidget(self.mdi)
        self.showMaximized()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(APP_ICON_PATH))

        self.diffusion_widget = None

    def createMenuBar(self):
        self.statusBar()
        self.create_experiment_menu()
        menubar = self.menuBar()

        menubar.addAction(
            create_action(
                parent=self,
                title="&About",
                shortcut="Ctrl+A",
                tooltip="About the author",
                handler=self.controller.handler_about_dialog,
            )
        )
        menubar.addAction(
            create_action(
                parent=self,
                title="&Exit",
                shortcut="Ctrl+Q",
                tooltip="Exit application",
                handler=qApp.quit,
            )
        )

    def create_experiment_menu(self):
        menubar = self.menuBar()
        experimentMenu = menubar.addMenu("&Experiment")

        # exit action
        experimentMenu.addAction(
            create_action(
                parent=self,
                title="&Create",
                shortcut="Ctrl+N",
                tooltip="Create",
                handler=self.controller.handler_new_experiment,
            )
        )
        experimentMenu.addAction(
            create_action(
                parent=self,
                title="&Import",
                shortcut="Ctrl+I",
                tooltip="Import",
                handler=self.controller.handler_import_experiment,
            )
        )
