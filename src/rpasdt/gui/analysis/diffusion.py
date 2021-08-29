import typing

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from rpasdt.gui.mathplotlib_components import NetworkxGraphPanel
from rpasdt.gui.toolbar.toolbars import DiffusionNetworkGraphToolbar

matplotlib.use("Qt5Agg")


class DiffusionGraphPanel(NetworkxGraphPanel):
    def __init__(
        self,
        controller,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        super().__init__(controller=controller, parent=parent)

    def create_toolbar(self):
        return DiffusionNetworkGraphToolbar(self.canvas, self, self.controller)

    def draw_nodes(self):
        legend_elements = [
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
