# https://stackoverflow.com/questions/43541376/how-to-draw-measures-with-networkx/43541777
# https://programmersought.com/article/56294025584/
import typing

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D

from rpasdt.gui.analysis.models import CommunityAnalysisData
from rpasdt.gui.mathplotlib_components import NetworkxGraphPanel

# needed to not display mathplotlib window
from rpasdt.network.networkx_utils import get_community_index, get_nodes_color

matplotlib.use("Qt5Agg")


class CommunityGraphPanel(NetworkxGraphPanel):
    def __init__(
        self,
        controller: "GraphController",
        data: CommunityAnalysisData,
        parent: typing.Optional["QWidget"] = None,
    ) -> None:
        self.communities_color = get_nodes_color(data.communities_list)
        self.data = data
        super().__init__(controller=controller, parent=parent)

    def draw_nodes(self):
        return nx.draw_networkx_nodes(
            self.graph,
            self.graph_position,
            node_size=250,
            node_color=[
                self.communities_color[get_community_index(community)]
                for community in self.data.node_values
            ],
            nodelist=self.data.node_list,
        )

    def draw_legend(self):
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"Community {community}",
                markerfacecolor=self.communities_color[get_community_index(community)],
                markersize=15,
            )
            for community in self.data.communities_list
        ]
        plt.legend(loc="lower right", handles=legend_elements)
