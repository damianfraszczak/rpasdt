"""Dynamic age source detection method."""
import copy
from typing import Dict, Union

import networkx as nx
import numpy as np

from rpasdt.algorithm.source_detectors.common import SourceDetector


class DynamicAgeSourceDetector(SourceDetector):
    def estimate_sources(self) -> Dict[int, Union[int, Dict[int, float]]]:
        graph = self.IG
        A = nx.adjacency_matrix(graph).todense().A
        dynamicAges = {node: 0 for node in graph.nodes}
        lamda_max = max(np.linalg.eigvals(A)).real

        for node in graph.nodes:
            A_new = copy.deepcopy(A)
            A_new = np.delete(A_new, node, axis=0)
            A_new = np.delete(A_new, node, axis=1)
            lamda_new = max(np.linalg.eigvals(A_new)).real
            dynamicAges[node] = (lamda_max - lamda_new) / lamda_max

        return dynamicAges
