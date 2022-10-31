from collections import defaultdict

from networkx import Graph

from rpasdt.algorithm.models import (
    EnsembleCommunitiesBasedSourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class EnsembleLearnerSourceDetector(CommunityBasedSourceDetector):
    CONFIG_CLASS = EnsembleCommunitiesBasedSourceDetectionConfig

    def find_sources_in_community(self, graph: Graph):
        results = defaultdict(int)
        for sd in self.config.source_detectors:
            result = sd.find_sources_in_community(graph)
            for key, value in result.items():
                results[key] += value
        results = {
            key: value / len(self.config.source_detectors)
            for key, value in results.items()
        }
        return results
