from typing import Callable, List, Tuple

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMdiSubWindow, QTabWidget

from rpasdt.gui.mathplotlib_components import (
    NetworkxGraphPanel,
    NetworkxGraphPanelWithToolbar,
    SimplePlotPanel,
)
from rpasdt.gui.table.models import ListTableModel
from rpasdt.gui.toolbar.toolbars import AnalysisNetworkGraphToolbar


class MainNetworkGraphPanel(NetworkxGraphPanelWithToolbar):
    pass


class AnalysisNetworkGraphPanel(NetworkxGraphPanel):
    def create_toolbar(self):
        return AnalysisNetworkGraphToolbar(self.canvas, self, self.controller)


class AppDialog(QMdiSubWindow):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.title = title
        self.configure_gui()

    def configure_gui(self):
        """Configure UI"""
        self.setWindowTitle(self.title)


class SimplePlotAppDialog(AppDialog):
    def __init__(self, title: str, plot_renderer: Callable) -> None:
        self.plot_renderer = plot_renderer
        super().__init__(title)

    def configure_gui(self):
        """Configure UI"""
        super().configure_gui()
        self.setWidget(
            SimplePlotPanel(title=self.title, plot_renderer=self.plot_renderer)
        )


class BaseAnalysisDialog(AppDialog):
    def add_tab(self, widget, title):
        self.tabPanel.addTab(widget, title)

    def add_table(
        self, headers: List[str], data: List[List], title: str
    ) -> Tuple[QtWidgets.QTableView, ListTableModel]:
        table = QtWidgets.QTableView()
        model = ListTableModel(data, headers)
        table.setModel(model)
        self.add_tab(table, title)
        return table, model

    def configure_gui(self):
        """Configure UI"""
        self.tabPanel = QTabWidget()
        self.setWidget(self.tabPanel)
        self.setWindowTitle(self.title)


class AnalysisDialog(BaseAnalysisDialog):
    """Default dialog with graph analysis visualisation."""

    def __init__(
        self, controller: "AnalysisGraphController", title: str, graph_panel: type
    ):
        self.controller = controller
        self.graph_panel = graph_panel
        self.graph = controller.graph
        self.graph_config = controller.graph_config
        self.data = controller.data
        super(AnalysisDialog, self).__init__(title)
        self.configure_gui()

    def get_table_data(self):
        """Return data displayed in table."""
        return [[node, measure] for node, measure in self.data.table_data]

    def get_table_header(self):
        """Return table header names."""
        return self.data.table_headers

    def get_graph_panel(self):
        """Return the graph panel responsible for graph visualisation."""
        return self.graph_panel(controller=self.controller, data=self.data)

    def configure_gui(self):
        """Configure UI"""
        super().configure_gui()
        self.network_panel = self.get_graph_panel()
        self.network_panel.title = self.title
        self.add_tab(widget=self.network_panel, title="Graph")
        self.add_table(
            headers=self.get_table_header(), data=self.get_table_data(), title="Data"
        )
