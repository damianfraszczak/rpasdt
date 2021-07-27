from dataclasses import dataclass, field
from typing import Any, Dict, List

from networkx import Graph

from rpasdt.algorithm.models import SingleSourceDetectionEvaluation
from rpasdt.network.networkx_utils import get_grouped_nodes


@dataclass
class AnalysisData:
    graph: Graph
    nodes_data: Dict[int, float] = field(default_factory=dict)
    table_headers = ["Node", "Value"]

    @property
    def node_list(self) -> List[int]:
        return list(self.nodes_data.keys())

    @property
    def node_values(self) -> List[float]:
        return list(self.nodes_data.values())

    @property
    def table_data(self) -> Any:
        return self.nodes_data.items()


@dataclass
class CommunityAnalysisData(AnalysisData):
    table_headers = ["Cluster", "Nodes"]
    communities_data: Dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        self.nodes_data = get_grouped_nodes(self.communities_data)

    @property
    def communities_list(self) -> List[int]:
        return list(self.communities_data.keys())

    @property
    def table_data(self):
        return [
            [cluster, ",".join([str(node) for node in nodes])]
            for cluster, nodes in self.communities_data.items()
        ]


@dataclass
class SourceDetectionAnalysisData:
    graph: Graph
    real_sources: List[int]
    detected_sources: List[int]
    evaluation: SingleSourceDetectionEvaluation
