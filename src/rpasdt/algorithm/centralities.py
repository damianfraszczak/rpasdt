"""Graph centrality measures."""
from typing import Dict, Union

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.taxonomies import CentralityOptionEnum
from rpasdt.common.utils import get_enum

CENTRALITY_OPERATION_MAP = {
    CentralityOptionEnum.DEGREE: nx.degree_centrality,
    CentralityOptionEnum.EIGENVECTOR: nx.eigenvector_centrality,
    CentralityOptionEnum.KATZ: nx.katz_centrality,
    CentralityOptionEnum.CLOSENESS: nx.closeness_centrality,
    CentralityOptionEnum.BETWEENNESS: nx.betweenness_centrality,
    CentralityOptionEnum.EDGE_BETWEENNESS: nx.edge_betweenness_centrality,
    CentralityOptionEnum.HARMONIC: nx.harmonic_centrality,
    CentralityOptionEnum.VOTE_RANK: nx.voterank,
    CentralityOptionEnum.PAGE_RANK: nx.pagerank,
}


def compute_centrality(
    type: Union[str, CentralityOptionEnum], graph: Graph, *args, **kwargs
) -> Dict[int, float]:
    return CENTRALITY_OPERATION_MAP.get(get_enum(type, CentralityOptionEnum))(graph)


def compute_unbiased_centrality(
    type: Union[str, CentralityOptionEnum], r: float, graph: Graph, *args, **kwargs
) -> Dict[int, float]:
    centrality_measure = compute_centrality(
        type=get_enum(type, CentralityOptionEnum), graph=graph
    )
    return {
        node: centrality / max(graph.degree(node) ** r, 0.0001)
        for node, centrality in centrality_measure.items()
    }
