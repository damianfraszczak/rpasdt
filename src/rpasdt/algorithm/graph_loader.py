"""Graph loading utilities."""
from typing import Dict, Union

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.graph_export_import import GRAPH_IMPORTER
from rpasdt.algorithm.taxonomies import GraphDataFormatEnum, GraphTypeEnum
from rpasdt.common.utils import get_enum


def _load_graph(input_graph_path: str, graph_data_format: GraphDataFormatEnum) -> Graph:
    with open(input_graph_path, "r") as file:
        return GRAPH_IMPORTER[graph_data_format](file.read())


def _load_custom_graph(graph_type_properties: Dict) -> Graph:
    graph_data_format = graph_type_properties["graph_data_format"]
    file_path = graph_type_properties["file_path"]
    return _load_graph(input_graph_path=file_path, graph_data_format=graph_data_format)


GRAPH_TYPE_LOADER = {
    GraphTypeEnum.WATTS_STROGATZ: lambda graph_type_properties: nx.watts_strogatz_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.BARABASI_ALBERT: lambda graph_type_properties: nx.barabasi_albert_graph(
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
    GraphTypeEnum.STAR: lambda graph_type_properties: nx.star_graph(
        **graph_type_properties
    ),
    GraphTypeEnum.CUSTOM: _load_custom_graph,
}


def load_graph(
    graph_type: Union[GraphTypeEnum, str],
    graph_type_properties: Dict = None,
    *args,
    **kwargs
) -> Graph:
    graph_type_properties = graph_type_properties or {}
    graph_type = get_enum(graph_type, GraphTypeEnum)
    loader = GRAPH_TYPE_LOADER.get(graph_type)
    return loader(graph_type_properties=graph_type_properties)
