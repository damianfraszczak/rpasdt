"""Graph drawing utilities."""
from typing import List

import networkx as nx

from rpasdt.algorithm.taxonomies import (
    DiffusionGraphNodeRenderTypeEnum,
    GraphLayout,
)

GRAPH_LAYOUT_DRAW_ALGORITHM = {
    GraphLayout.CIRCULAR: lambda graph, *args, **kwargs: nx.circular_layout(graph),
    GraphLayout.KAMADA_KAWAI: lambda graph, *args, **kwargs: nx.kamada_kawai_layout(
        graph
    ),
    GraphLayout.PLANAR: lambda graph, *args, **kwargs: nx.planar_layout(graph),
    GraphLayout.RANDOM: lambda graph, seed=100, *args, **kwargs: nx.random_layout(
        graph, seed=seed
    ),
    GraphLayout.SPECTRAL: lambda graph, *args, **kwargs: nx.spectral_layout(graph),
    GraphLayout.SPRING: lambda graph, seed=100, *args, **kwargs: nx.spring_layout(
        graph, seed=seed
    ),
    GraphLayout.SHELL: lambda graph, *args, **kwargs: nx.shell_layout(graph),
}


def compute_graph_draw_position(
    graph: nx.Graph, layout: GraphLayout = None, *args, **kwargs
):
    layout = layout or GraphLayout.SPRING
    return GRAPH_LAYOUT_DRAW_ALGORITHM[layout](graph)


def get_diffusion_graph(
    source_graph: nx.Graph,
    infected_nodes: List[int],
    graph_node_rendering_type: DiffusionGraphNodeRenderTypeEnum = DiffusionGraphNodeRenderTypeEnum.ONLY_INFECTED,
) -> nx.Graph:
    if DiffusionGraphNodeRenderTypeEnum.FULL == graph_node_rendering_type:
        return source_graph.subgraph(source_graph.nodes())
    else:
        return source_graph.subgraph(infected_nodes)
