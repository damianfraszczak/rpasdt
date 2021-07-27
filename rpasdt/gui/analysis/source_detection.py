import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rpasdt.gui.analysis.analysis import AnalysisNetworkGraphPanel, AnalysisDialog, BaseAnalysisDialog
from rpasdt.gui.toolbar.toolbars import SourceDetectionGraphToolbar

matplotlib.use('Qt5Agg')


class SourceDetectionGraphPanel(AnalysisNetworkGraphPanel):
    def create_toolbar(self):
        return SourceDetectionGraphToolbar(
            self.canvas, self, self.controller
        )

    def draw_nodes(self):
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Estimated source',
                   markerfacecolor=self.graph_config.estimated_source_node_color, markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Source',
                   markerfacecolor=self.graph_config.source_node_color, markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Infected',
                   markerfacecolor=self.graph_config.infected_node_color, markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Susceptible',
                   markerfacecolor=self.graph_config.node_color, markersize=15),
        ]
        plt.legend(loc='lower right', handles=legend_elements)
        return super().draw_nodes()


class SourceDetectionDialog(BaseAnalysisDialog):
    """Dialog displaying source detection results."""

    def __init__(self,
                 title: str,
                 controller: 'SourceDetectionGraphController') -> None:
        self.controller = controller
        super().__init__(title)

    def configure_gui(self):
        super().configure_gui()
        data = self.controller.data
        self.add_tab(widget=SourceDetectionGraphPanel(controller=self.controller), title='Graph')
        self.add_table(headers=['Detected sources', 'Real sources', 'Error distance', 'TP', 'FP', 'FN'],
                       data=[
                           [
                               f'{data.detected_sources}',
                               f'{data.real_sources}',
                               data.evaluation.error_distance,
                               data.evaluation.TP,
                               data.evaluation.FP,
                               data.evaluation.FN
                           ]
                       ],
                       title='Evaluation'
                       )
