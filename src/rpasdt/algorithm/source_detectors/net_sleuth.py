"""NetSleuth source detection method."""
import networkx as nx
import numpy as np
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class NetSleuthCommunityBasedSourceDetector(CommunityBasedSourceDetector):
    def estimate_sources(self, G: Graph, IG: Graph):
        nodes = np.array(IG.nodes())
        node_to_index = {k: v for v, k in enumerate(nodes)}
        L = nx.laplacian_matrix(IG).todense().A
        w, v = np.linalg.eig(L)
        v1 = v[np.where(w == np.min(w))][0]
        # max_val = np.max(v1)
        # sources = nodes[np.where(v1 == max_val)]
        # return {source: max_val for source in sources}
        return {node: v1[node_to_index[node]].real for node in nodes}
