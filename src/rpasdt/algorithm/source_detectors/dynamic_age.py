"""Dynamic age source detection method."""
import copy

import networkx as nx
import numpy as np
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class DynamicAgeSourceDetector(CommunityBasedSourceDetector):
    def estimate_sources(self, G: Graph, IG: Graph):
        A = nx.adjacency_matrix(IG).todense().A
        dynamicAges = {node: 0 for node in IG.nodes}
        node_position = {node: index for index, node in enumerate(IG.nodes)}
        lamda_max = max(np.linalg.eigvals(A)).real

        for node in IG.nodes:
            A_new = copy.deepcopy(A)
            A_new = np.delete(A_new, node_position[node], axis=0)
            A_new = np.delete(A_new, node_position[node], axis=1)
            lamda_new = max(np.linalg.eigvals(A_new)).real
            dynamicAges[node] = (lamda_max - lamda_new) / lamda_max
        return dynamicAges
