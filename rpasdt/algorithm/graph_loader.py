from typing import Dict

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.graph_export_import import GRAPH_IMPORTER
from rpasdt.algorithm.taxonomies import GraphTypeEnum


def load_custom_graph(graph_type_properties):
    graph_data_format = graph_type_properties["graph_data_format"]
    file_path = graph_type_properties["file_path"]
    with open(file_path, "r") as file:
        return GRAPH_IMPORTER[graph_data_format](file.read())


GRAPH_TYPE_LOADER = {
    GraphTypeEnum.WATTS_STROGATZ: lambda graph_type_properties: nx.watts_strogatz_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.KARATE_CLUB: lambda graph_type_properties: nx.karate_club_graph(),
    GraphTypeEnum.BALANCED_TREE: lambda graph_type_properties: nx.balanced_tree(
        **graph_type_properties
    ),
    GraphTypeEnum.COMPLETE: lambda graph_type_properties: nx.complete_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.ERDOS_RENYI: lambda graph_type_properties: nx.erdos_renyi_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.DAVIS_SOUTHERN: lambda graph_type_properties: nx.davis_southern_women_graph(),
    GraphTypeEnum.FLORENTINE_FAMILIES: lambda graph_type_properties: nx.florentine_families_graph(),
    GraphTypeEnum.LES_MISERABLES: lambda graph_type_properties: nx.les_miserables_graph(),
    GraphTypeEnum.CAVEMAN_GRAPH: lambda graph_type_properties: nx.caveman_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.CONNECTED_CAVEMAN_GRAPH: lambda graph_type_properties: nx.connected_caveman_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.CUSTOM: load_custom_graph,
}


def load_graph(
    graph_type: GraphTypeEnum, graph_type_properties: Dict = None, *args, **kwargs
) -> Graph:
    graph_type_properties = graph_type_properties or {}
    return GRAPH_TYPE_LOADER[graph_type](graph_type_properties=graph_type_properties)
