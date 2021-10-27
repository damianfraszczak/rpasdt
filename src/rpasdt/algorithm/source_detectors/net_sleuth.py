"""NetSleuth source detection method."""
import networkx as nx
import numpy as np
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class NetSleuthCommunityBasedSourceDetector(CommunityBasedSourceDetector):
    def find_sources_in_community(self, graph: Graph):
        nodes = np.array(graph.nodes())
        L = nx.laplacian_matrix(graph).todense().A
        w, v = np.linalg.eig(L)
        v1 = v[np.where(w == np.min(w))][0]
        max_val = np.max(v1)
        sources = nodes[np.where(v1 == np.max(v1))]
        return {source: max_val for source in sources}
