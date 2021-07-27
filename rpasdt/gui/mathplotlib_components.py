import typing

import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import networkx as nx

from rpasdt.algorithm.graph_drawing import compute_graph_draw_position
from rpasdt.gui.toolbar.toolbars import MainNetworkGraphToolbar
from rpasdt.network.networkx_utils import create_node_network_dict, create_node_network_array
from rpasdt.network.taxonomies import NodeAttributeEnum

matplotlib.use('Qt5Agg')

CLICK_EPSILON = 0.001
NODE_LABEL_OFFSET = 0.08


class NetworkxGraphPanel(QWidget):
    node_clicked = pyqtSignal(int)

    def __init__(self,
                 controller: 'GraphController',
                 parent: typing.Optional['QWidget'] = None,
                 title: typing.Optional[str] = None
                 ) -> None:
        super().__init__(parent)
        self.controller = controller
        self.title = title
        self.graph = controller.graph
        self.graph_config = controller.graph_config
        if controller:
            controller.graph_panel = self
        self.configure_gui()

    def find_node_index_for_event(self, event) -> typing.Optional[int]:
        if event.inaxes:
            clickX = event.xdata
            clickY = event.ydata
            if event.inaxes in self.figure.axes:
                distances = [pow(x - clickX, 2) + pow(y - clickY, 2) for node, (x, y) in self.graph_position.items()]
                min_value = min(distances)
                if min_value < CLICK_EPSILON:
                    return distances.index(min_value)

    def draw_extra_labels(self):
        graph_position_attrs = {}
        for node, (x, y) in self.graph_position.items():
            graph_position_attrs[node] = (x, y + NODE_LABEL_OFFSET)
        nx.draw_networkx_labels(self.graph, graph_position_attrs,
                                labels=create_node_network_dict(self.graph, NodeAttributeEnum.EXTRA_LABEL,
                                                                skip_empty=True))

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
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.setLayout(layout)
        self.redraw()

    @property
    def graph_position(self):
        if not self.graph_config.graph_position:
            self.graph_config.graph_position = compute_graph_draw_position(self.graph, self.graph_config.graph_layout)
        return self.graph_config.graph_position

    def onclick(self, event):
        node = self.find_node_index_for_event(event)
        if node is not None:
            self.node_clicked.emit(node)

    def draw_nodes(self):
        node_color = create_node_network_array(self.graph, NodeAttributeEnum.COLOR,
                                               self.graph_config.node_color)
        return nx.draw_networkx_nodes(self.graph, self.graph_position,
                                      node_size=create_node_network_array(self.graph, NodeAttributeEnum.SIZE,
                                                                          self.graph_config.node_size),
                                      node_color=node_color,
                                      )

    def draw_edges(self):
        return nx.draw_networkx_edges(self.graph, self.graph_position)

    def draw_labels(self):
        return nx.draw_networkx_labels(self.graph, self.graph_position,
                                       labels=create_node_network_dict(self.graph, NodeAttributeEnum.EXTRA_LABEL, ''),
                                       font_color=self.graph_config.node_label_font_color
                                       )

    def draw_legend(self):
        pass

    def redraw(self):
        if not self.graph:
            return
        # set current figure to be active, to redraw self figure not other
        plt.figure(self.figure.number)
        self.figure.clf()
        if self.title:
            plt.title(self.title)
        nodes = self.draw_nodes()
        edges = self.draw_edges()
        if self.graph_config.display_node_labels:
            self.draw_labels()
        if self.graph_config.display_node_extra_labels:
            self.draw_extra_labels()
        self.draw_legend()
        self.canvas.draw()
        return (nodes, edges)


class NetworkxGraphPanelWithToolbar(NetworkxGraphPanel):
    def create_toolbar(self):
        return MainNetworkGraphToolbar(
            self.canvas, self, self.controller
        )
