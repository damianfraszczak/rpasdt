"""Dynamic age source detection method."""
import copy

import networkx as nx
import numpy as np
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class DynamicAgeSourceDetector(CommunityBasedSourceDetector):
    def find_sources_in_community(self, graph: Graph):
        A = nx.adjacency_matrix(graph).todense().A
        dynamicAges = {node: 0 for node in graph.nodes}
        node_position = {node: index for index, node in enumerate(graph.nodes)}
        lamda_max = max(np.linalg.eigvals(A)).real

        for node in graph.nodes:
            A_new = copy.deepcopy(A)
            A_new = np.delete(A_new, node_position[node], axis=0)
            A_new = np.delete(A_new, node_position[node], axis=1)
            lamda_new = max(np.linalg.eigvals(A_new)).real
            dynamicAges[node] = (lamda_max - lamda_new) / lamda_max
        return dynamicAges
