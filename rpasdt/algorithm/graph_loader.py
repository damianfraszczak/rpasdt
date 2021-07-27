from typing import Dict

from networkx import Graph
import networkx as nx

from rpasdt.algorithm.taxonomies import GraphTypeEnum

GRAPH_TYPE_LOADER = {
    GraphTypeEnum.WATTS_STROGATZ: lambda
        graph_type_properties: nx.watts_strogatz_graph(
        **graph_type_properties),
    GraphTypeEnum.KARATE_CLUB: lambda
        graph_type_properties: nx.karate_club_graph(),
    GraphTypeEnum.BALANCED_TREE: lambda
        graph_type_properties: nx.balanced_tree(**graph_type_properties),
    GraphTypeEnum.COMPLETE: lambda graph_type_properties: nx.complete_graph(
        **graph_type_properties),
    GraphTypeEnum.ERDOS_RENYI: lambda
        graph_type_properties: nx.erdos_renyi_graph(**graph_type_properties),
    GraphTypeEnum.DAVIS_SOUTHERN: lambda
        graph_type_properties: nx.davis_southern_women_graph(),
    GraphTypeEnum.FLORENTINE_FAMILIES: lambda
        graph_type_properties: nx.florentine_families_graph(),
    GraphTypeEnum.LES_MISERABLES: lambda
        graph_type_properties: nx.les_miserables_graph(),
    GraphTypeEnum.CAVEMAN_GRAPH: lambda
        graph_type_properties: nx.caveman_graph(**graph_type_properties),
    GraphTypeEnum.CONNECTED_CAVEMAN_GRAPH: lambda
        graph_type_properties: nx.connected_caveman_graph(
        **graph_type_properties),
}


def load_graph(
    graph_type: GraphTypeEnum,
    graph_type_properties: Dict = None, *args, **kwargs) -> Graph:
    graph_type_properties = graph_type_properties or {}
    return GRAPH_TYPE_LOADER[graph_type](
        graph_type_properties=graph_type_properties)
