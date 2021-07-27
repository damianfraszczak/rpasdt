import typing

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
from PyQt5.QtWidgets import QWidget

from rpasdt.gui.analysis.models import AnalysisData
from rpasdt.gui.mathplotlib_components import NetworkxGraphPanel

matplotlib.use("Qt5Agg")


class CentralityGraphPanel(NetworkxGraphPanel):
    """Panel displaying centrality analysis results.

    It contains of tab panel with two tabs:
    - graph visualisation panel
    - table data of the selected centrality analysis
    """

    def __init__(
        self,
        controller: "GraphController",
        data: AnalysisData,
        parent: typing.Optional[QWidget] = None,
    ) -> None:
        self.data = data
        super().__init__(controller=controller, parent=parent)

    def draw_nodes(self):
        nodes = nx.draw_networkx_nodes(
            self.graph,
            self.graph_position,
            node_size=250,
            cmap=plt.cm.plasma,
            node_color=list(self.data.node_values),
            nodelist=self.data.node_list,
        )
        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
        plt.gcf().colorbar(nodes)
        return nodes
