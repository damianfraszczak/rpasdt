"""Network analysis utilities."""
import networkx as nx
from networkx import Graph

from rpasdt.algorithm.taxonomies import NetworkAnalysisOptionEnum

NETWORK_ANALYSIS_OPERATION_MAP = {
    NetworkAnalysisOptionEnum.DENSITY: nx.density,
    NetworkAnalysisOptionEnum.AVERAGE_CLUSTERING: nx.average_clustering,
    NetworkAnalysisOptionEnum.SUMMARY: nx.info,
    NetworkAnalysisOptionEnum.AVERAGE_SHORTEST_PATH: nx.average_shortest_path_length,
    NetworkAnalysisOptionEnum.ASSORTAVITY: nx.degree_assortativity_coefficient,
}


def compute_network_analysis(type: NetworkAnalysisOptionEnum, graph: Graph):
    return NETWORK_ANALYSIS_OPERATION_MAP.get(type)(graph)
