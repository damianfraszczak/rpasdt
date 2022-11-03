from collections import defaultdict
from typing import List

from networkx import Graph

from rpasdt.algorithm.models import (
    EnsembleCommunitiesBasedSourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
    SourceDetector,
)


class EnsembleLearnerSourceDetector(CommunityBasedSourceDetector):
    CONFIG_CLASS = EnsembleCommunitiesBasedSourceDetectionConfig
    source_detectors: List[SourceDetector]

    def find_sources_in_community(self, graph: Graph):
        results = defaultdict(int)
        for sd in self.source_detectors:
            result = sd.f(graph)
            for key, value in result.items():
                results[key] += value
        results = {
            key: value / len(self.config.source_detectors)
            for key, value in results.items()
        }
        return results
