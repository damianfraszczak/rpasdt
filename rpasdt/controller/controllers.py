from typing import Any, Dict

from networkx import Graph

from rpasdt.algorithm.centralities import CENTRALITY_OPERATION_MAP
from rpasdt.algorithm.communities import COMMUNITY_OPERATION_MAP
from rpasdt.common.utils import format_label
from rpasdt.controller.graph import GraphController
from rpasdt.gui.analysis.analysis import AnalysisDialog
from rpasdt.gui.analysis.centrality import CentralityGraphPanel
from rpasdt.gui.analysis.communities import CommunityGraphPanel
from rpasdt.gui.analysis.models import AnalysisData, CommunityAnalysisData
from rpasdt.gui.toolbar.toolbar_items import (
    CENTRALITY_OPTIONS,
    COMMUNITY_OPTIONS,
)
from rpasdt.gui.utils import run_long_task
from rpasdt.model.experiment import GraphConfig

CENTRALITY_HANDLER_PREFIX = "handler_analysis_centrality_"
COMMUNITY_HANDLER_PREFIX = "handler_analysis_community_"


class AnalysisGraphController(GraphController):
    def __init__(
        self, window: "MainWindow", graph: Graph, graph_config: GraphConfig, data: Dict
    ):
        super().__init__(window, graph, graph_config)
        self.data = data


class CommunityAnalysisControllerMixin:
    def __getattribute__(self, name: str) -> Any:

        if COMMUNITY_HANDLER_PREFIX in name:
            centrality_code = name.replace(COMMUNITY_HANDLER_PREFIX, "")
            func = COMMUNITY_OPERATION_MAP.get(centrality_code)
            title, tooltip = COMMUNITY_OPTIONS[centrality_code]
            label = format_label(title, "community")
            if func:
                return lambda: run_long_task(
                    title=f"Computing {label}",
                    function=func,
                    function_args=[self.graph],
                    callback=lambda degree: self.show_community_result(
                        data=degree, title=f"{label} visualisation"
                    ),
                )
        return super().__getattribute__(name)

    def show_community_result(self, data, title):
        self.show_dialog(
            AnalysisDialog(
                controller=AnalysisGraphController(
                    window=self.window,
                    graph=self.graph,
                    graph_config=self.graph_config,
                    data=CommunityAnalysisData(graph=self.graph, communities_data=data),
                ),
                title=title,
                graph_panel=CommunityGraphPanel,
            )
        )


class CentralityAnalysisControllerMixin:
    def __getattribute__(self, name: str) -> Any:

        if CENTRALITY_HANDLER_PREFIX in name:
            centrality_code = name.replace(CENTRALITY_HANDLER_PREFIX, "")
            func = CENTRALITY_OPERATION_MAP.get(centrality_code)
            title, tooltip = CENTRALITY_OPTIONS[centrality_code]
            label = format_label(title, "centrality")
            if func:
                return lambda: run_long_task(
                    title=f"Computing {label}",
                    function=func,
                    function_args=[self.graph],
                    callback=lambda data: self.show_analysis_result(
                        data=AnalysisData(graph=self.graph, nodes_data=data),
                        title=f"{label} visualisation",
                    ),
                )
        return super().__getattribute__(name)

    def show_analysis_result(self, data, title):
        self.show_dialog(
            AnalysisDialog(
                controller=AnalysisGraphController(
                    window=self.window,
                    graph=self.graph,
                    graph_config=self.graph_config,
                    data=data,
                ),
                title=title,
                graph_panel=CentralityGraphPanel,
            )
        )
