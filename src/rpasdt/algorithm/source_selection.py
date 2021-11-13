"""Source selection methods."""
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
        centrality=CentralityOptionEnum.DEGREE,
        number_of_sources=number_of_sources,
    ),
    SourceSelectionOptionEnum.BETWEENNESS: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        centrality=CentralityOptionEnum.BETWEENNESS,
        number_of_sources=number_of_sources,
    ),
    SourceSelectionOptionEnum.CLOSENESS: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        centrality=CentralityOptionEnum.CLOSENESS,
        number_of_sources=number_of_sources,
    ),
    SourceSelectionOptionEnum.PAGE_RANK: lambda graph, number_of_sources: __source_selection_centrality(
        graph=graph,
        centrality=CentralityOptionEnum.PAGE_RANK,
        number_of_sources=number_of_sources,
    ),
}


def select_sources(config: NetworkSourceSelectionConfig, graph: Graph):
    return SOURCE_SELECTION_OPERATION_MAP.get(config.algorithm)(
        graph=graph, number_of_sources=config.number_of_sources
    )


def select_sources_with_params(
    graph: Graph,
    number_of_sources: int = 1,
    algorithm: SourceSelectionOptionEnum = SourceSelectionOptionEnum.RANDOM,
    *args,
    **kwargs
):
    return select_sources(
        config=NetworkSourceSelectionConfig(
            algorithm=algorithm, number_of_sources=number_of_sources
        ),
        graph=graph,
    )
