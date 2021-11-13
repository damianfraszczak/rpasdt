import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rpasdt.algorithm.plots import plot_confusion_matrix
from rpasdt.gui.analysis.analysis import (
    AnalysisNetworkGraphPanel,
    BaseAnalysisDialog,
)
from rpasdt.gui.mathplotlib_components import SimplePlotPanel
from rpasdt.gui.toolbar.toolbars import SourceDetectionGraphToolbar

matplotlib.use("Qt5Agg")


class SourceDetectionGraphPanel(AnalysisNetworkGraphPanel):
    def create_toolbar(self):
        return SourceDetectionGraphToolbar(self.canvas, self, self.controller)

    def draw_nodes(self):
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Estimated source",
                markerfacecolor=self.graph_config.estimated_source_node_color,
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Source",
                markerfacecolor=self.graph_config.source_node_color,
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Infected",
                markerfacecolor=self.graph_config.infected_node_color,
                markersize=15,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Susceptible",
                markerfacecolor=self.graph_config.node_color,
                markersize=15,
            ),
        ]
        plt.legend(loc="lower right", handles=legend_elements)
        return super().draw_nodes()


class SourceDetectionDialog(BaseAnalysisDialog):
    """Dialog displaying source detection results."""

    def __init__(
        self, title: str, controller: "SourceDetectionGraphController"
    ) -> None:
        self.controller = controller
        super().__init__(title)

    def _create_source_detection_graph_tab(self):
        self.add_tab(
            widget=SourceDetectionGraphPanel(controller=self.controller), title="Graph"
        )

    def _create_confusion_matrix_tab(self):
        cm = self.controller.data.evaluation
        self.add_tab(
            title="Confusion matrix",
            widget=SimplePlotPanel(
                title="Confusion matrix",
                plot_renderer=lambda: plot_confusion_matrix(cm=cm, ax=plt.gca()),
            ),
        )

    def _create_metrics_table_tab(self):
        data = self.controller.data
        headers = ["Metric", "Value"]
        table_data = [
            ["Real sources", f"{data.real_sources}"],
            ["Detected sources", f"{data.detected_sources}"],
            ["Error distance", data.evaluation.error_distance],
        ]
        table_data.extend(list(data.evaluation.get_classification_report().items()))
        self.add_table(
            headers=headers,
            data=table_data,
            title="Evaluation",
        )

    def configure_gui(self):
        super().configure_gui()
        self._create_source_detection_graph_tab()
        self._create_metrics_table_tab()
        self._create_confusion_matrix_tab()
