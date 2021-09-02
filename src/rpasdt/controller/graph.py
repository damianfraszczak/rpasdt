from copy import copy

from networkx import Graph

from rpasdt.gui.analysis.analysis import AnalysisDialog
from rpasdt.gui.form_utils import get_node_edit_config
from rpasdt.gui.utils import show_dynamic_dialog
from rpasdt.model.experiment import GraphConfig


class GraphController:
    graph_panel: "NetworkxGraphPanel"

    def __init__(self, window: "MainWindow", graph: Graph, graph_config: GraphConfig):
        super().__init__()
        self.window = window
        self.graph = graph.copy()
        self.graph_config = copy(graph_config)

    def show_dialog(self, dialog: AnalysisDialog):
        self.window.add_subwindow(dialog)

    def update_graph(self, graph: Graph):
        self.graph = graph
        self.graph_panel.graph = graph
        self.redraw_graph()

    def get_node_edit_form_config(self, node_index, node):
        return get_node_edit_config(node_index)

    def redraw_graph(self):
        self.graph_panel.redraw()

    def handler_edit_graph_config(self):
        graph_config = show_dynamic_dialog(
            object=self.graph_config, title="Edit graph config"
        )
        if graph_config:
            graph_config.clear()
            self.graph_config_changed()

    def graph_config_changed(self):
        self.redraw_graph()

    def handler_graph_node_clicked(self, node_index):
        node = self.graph.nodes[node_index]
        if show_dynamic_dialog(
            object=node,
            config=self.get_node_edit_form_config(node_index=node_index, node=node),
        ):
            self.handler_graph_node_edited(node)

    def handler_graph_node_edited(self, node):
        self.redraw_graph()

    def handler_export_graph(self):
        pass
