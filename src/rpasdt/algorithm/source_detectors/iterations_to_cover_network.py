import math

from networkx import Graph

from rpasdt.algorithm.models import IterationsToCoverNetworkConfig
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)
from rpasdt.algorithm.source_selection import select_sources_with_params
from rpasdt.algorithm.utils import neighbors_of_k_hops


class IterationsToCoverNetworkSourceDetector(CommunityBasedSourceDetector):
    CONFIG_CLASS = IterationsToCoverNetworkConfig

    def find_sources_in_community(self, graph: Graph):
        IG = self.IG
        number_of_sources = math.ceil(len(IG) / 10)
        # find candidates based on some metric
        candidate_sources = select_sources_with_params(
            graph=graph,
            algorithm=self.config.node_selection_algorithm,
            number_of_sources=number_of_sources,
        )
        # extend candidates by neighbors
        for candidate in candidate_sources:
            candidate_sources | neighbors_of_k_hops(
                G=IG, node=candidate, k=self.config.neighbors_hops
            )
        # simulate propagation from all sources to nodes in the whole Graph until
        # infecting real number of nodes
