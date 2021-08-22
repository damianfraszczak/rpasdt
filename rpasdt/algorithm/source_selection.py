import random

from networkx import Graph

from rpasdt.algorithm.centralities import compute_centrality
from rpasdt.algorithm.models import NetworkSourceSelectionConfig
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    SourceSelectionOptionEnum,
)


def __source_selection_random_nodes(graph: Graph, number_of_sources: int = 1):
    return random.choices(list(graph.nodes), k=number_of_sources)


def __source_selection_centrality(
    graph: Graph, centrality: CentralityOptionEnum, number_of_sources: int = 1
):
    return [
        node
        for node, _ in sorted(
            compute_centrality(type=centrality, graph=graph).items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ][:number_of_sources]


SOURCE_SELECTION_OPERATION_MAP = {
    SourceSelectionOptionEnum.RANDOM: __source_selection_random_nodes,
    SourceSelectionOptionEnum.DEGREE: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        type=CentralityOptionEnum.DEGREE,
        number_of_sources=number_of_sources,
    ),
    SourceSelectionOptionEnum.BETWEENNESS: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        type=CentralityOptionEnum.BETWEENNESS,
        number_of_sources=number_of_sources,
    ),
    SourceSelectionOptionEnum.CLOSENESS: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        type=CentralityOptionEnum.CLOSENESS,
        number_of_sources=number_of_sources,
    ),
}


def select_sources(config: NetworkSourceSelectionConfig, graph: Graph, *args, **kwargs):
    return SOURCE_SELECTION_OPERATION_MAP.get(config.algorithm)(
        graph=graph, number_of_sources=config.number_of_sources
    )
