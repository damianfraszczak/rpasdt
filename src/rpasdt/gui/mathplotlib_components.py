import typing

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from rpasdt.algorithm.graph_drawing import compute_graph_draw_position
from rpasdt.gui.toolbar.toolbars import MainNetworkGraphToolbar
from rpasdt.network.networkx_utils import (
    create_node_network_array,
    create_node_network_dict,
)
from rpasdt.network.taxonomies import NodeAttributeEnum

matplotlib.use("Qt5Agg")

CLICK_EPSILON = 0.001
NODE_LABEL_OFFSET = 0.08


class MatplotlibPanel(QWidget):
    node_clicked = pyqtSignal(int)

    def __init__(
        self,
        title: typing.Optional[str] = None,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.configure_gui()

    def create_toolbar(self):
        return NavigationToolbar(self.canvas, self)

    def configure_gui(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        toolbar = self.create_toolbar()

        layout = QVBoxLayout()
        if toolbar:
            layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self.onclick)
        self.setLayout(layout)
        self.redraw()

    def draw_content(self) -> None:
        pass

    def draw_legend(self) -> None:
        pass

    def redraw(self) -> None:
        # set current figure to be active, to redraw self figure not other
        plt.figure(self.figure.number)
        self.figure.clf()
        if self.title:
            plt.title(self.title)
        self.draw_legend()
        self.draw_content()
        self.canvas.draw()

    def onclick(self, event):
        pass


class SimplePlotPanel(MatplotlibPanel):
    def __init__(
        self,
        plot_renderer: typing.Callable,
        title: typing.Optional[str] = None,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        self.plot_renderer = plot_renderer
        super().__init__(title=title, parent=parent)

    def draw_content(self) -> None:
        self.plot_renderer()


class NetworkxGraphPanel(MatplotlibPanel):
    def __init__(
        self,
        controller: "GraphController",
        parent: typing.Optional["QWidget"] = None,
        title: typing.Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.graph = controller.graph
        self.graph_config = controller.graph_config
        if controller:
            controller.graph_panel = self
        super().__init__(title=title, parent=parent)

    def find_node_index_for_event(self, event) -> typing.Optional[int]:
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if event.inaxes in self.figure.axes:
                distances = [
                    pow(x - clickX, 2) + pow(y - clickY, 2)
                    for node, (x, y) in self.graph_position.items()
                ]
                min_value = min(distances)
                if min_value < CLICK_EPSILON:
                    return distances.index(min_value)

    def draw_extra_labels(self):
        graph_position_attrs = {}
        for node, (x, y) in self.graph_position.items():
            graph_position_attrs[node] = (x, y + NODE_LABEL_OFFSET)
        nx.draw_networkx_labels(
            self.graph,
            graph_position_attrs,
            labels=create_node_network_dict(
                self.graph, NodeAttributeEnum.EXTRA_LABEL, skip_empty=True
            ),
        )

    @property
    def graph_position(self):
        if not self.graph_config.graph_position:
            self.graph_config.graph_position = compute_graph_draw_position(
                self.graph, self.graph_config.graph_layout
            )
        return self.graph_config.graph_position

    def onclick(self, event):
        node = self.find_node_index_for_event(event)
        if node is not None:
            self.node_clicked.emit(node)

    def draw_nodes(self):
        node_color = create_node_network_array(
            self.graph, NodeAttributeEnum.COLOR, self.graph_config.node_color
        )
        return nx.draw_networkx_nodes(
            self.graph,
            self.graph_position,
            node_size=create_node_network_array(
                self.graph, NodeAttributeEnum.SIZE, self.graph_config.node_size
            ),
            node_color=node_color,
        )

    def draw_edges(self):
        return nx.draw_networkx_edges(self.graph, self.graph_position)

    def draw_labels(self):
        return nx.draw_networkx_labels(
            self.graph,
            self.graph_position,
            labels=create_node_network_dict(
                self.graph, NodeAttributeEnum.EXTRA_LABEL, ""
            ),
            font_color=self.graph_config.node_label_font_color,
        )

    def draw_content(self) -> None:
        if not self.graph:
            return
        self.draw_nodes()
        self.draw_edges()
        if self.graph_config.display_node_labels:
            self.draw_labels()
        if self.graph_config.display_node_extra_labels:
            self.draw_extra_labels()


class NetworkxGraphPanelWithToolbar(NetworkxGraphPanel):
    def create_toolbar(self):
        return MainNetworkGraphToolbar(self.canvas, self, self.controller)
