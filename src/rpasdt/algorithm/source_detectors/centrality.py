"""Centrality based source detection methods."""
from collections import Counter
from typing import Dict, Optional, Union

from networkx import Graph

from rpasdt.algorithm.centralities import (
    compute_centrality,
    compute_unbiased_centrality,
)
from rpasdt.algorithm.models import (
    CentralityBasedSourceDetectionConfig,
    CentralityCommunityBasedSourceDetectionConfig,
    MultipleCentralityBasedSourceDetectionConfig,
    UnbiasedCentralityBasedSourceDetectionConfig,
    UnbiasedCentralityCommunityBasedSourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
    SourceDetector,
)


class CentralityBasedSourceDetector(SourceDetector):
    CONFIG_CLASS = CentralityBasedSourceDetectionConfig

    def estimate_sources(self) -> Dict[int, Union[float, Dict[int, float]]]:
        return compute_centrality(type=self.config.centrality_algorithm, graph=self.IG)


class MultipleCentralityBasedSourceDetector(SourceDetector):
    CONFIG_CLASS = MultipleCentralityBasedSourceDetectionConfig

    def estimate_sources(self) -> Dict[int, Union[float, Dict[int, float]]]:
        sums = Counter()
        for alg in self.config.centrality_algorithms:
            sums.update(compute_centrality(type=alg, graph=self.IG))
        return {
            node: value / len(self.config.centrality_algorithms)
            for node, value in sums.items()
        }


class UnbiasedCentralityBasedSourceDetector(SourceDetector):
    CONFIG_CLASS = UnbiasedCentralityBasedSourceDetectionConfig

    def estimate_sources(self) -> Dict[int, Union[float, Dict[int, float]]]:
        return compute_unbiased_centrality(
            type=self.config.centrality_algorithm, r=self.config.r, graph=self.IG
        )


class CentralityCommunityBasedSourceDetector(CommunityBasedSourceDetector):
    CONFIG_CLASS = CentralityCommunityBasedSourceDetectionConfig

    def __init__(
        self,
        G: Graph,
        IG: Graph,
        config: Optional[CentralityCommunityBasedSourceDetectionConfig] = None,
    ):
        super().__init__(G, IG, config)

    def find_sources_in_community(self, graph: Graph):
        return compute_centrality(type=self.config.centrality_algorithm, graph=graph)

    def __str__(self) -> str:
        return f"Centrality based source detector: {self.config.centrality_algorithm}"


class UnbiasedCentralityCommunityBasedSourceDetector(
    CentralityCommunityBasedSourceDetector
):
    CONFIG_CLASS = UnbiasedCentralityCommunityBasedSourceDetectionConfig

    def find_sources_in_community(self, graph: Graph):
        return compute_unbiased_centrality(
            type=self.config.centrality_algorithm, graph=graph, r=self.config.r
        )
