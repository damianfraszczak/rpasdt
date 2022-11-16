from collections import defaultdict
from dataclasses import field
from typing import Dict, List, Union

from networkx import Graph

from rpasdt.algorithm.models import (
    EnsembleCommunitiesBasedSourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
    SourceDetector,
)
from rpasdt.common.utils import normalize_dict_values


class EnsembleLearnerSourceDetector(SourceDetector):
    CONFIG_CLASS = EnsembleCommunitiesBasedSourceDetectionConfig
    source_detectors: List[SourceDetector]

    def estimate_sources(
        self, G: Graph, IG: Graph
    ) -> Dict[int, Union[int, Dict[int, float]]]:
        """Compute each node score to be source."""
        results = defaultdict(int)
        for sd in self.source_detectors:
            result = sd.estimate_sources(G=G, IG=IG)
            result = normalize_dict_values(result)
            for key, value in result.items():
                results[key] += value
        results = {
            key: value / len(self.config.source_detectors)
            for key, value in results.items()
        }
        return results


class CommunityEnsembleLearnerSourceDetector(CommunityBasedSourceDetector):
    CONFIG_CLASS = EnsembleCommunitiesBasedSourceDetectionConfig
    source_detectors: List[SourceDetector] = field(default_factory=list)

    def __init__(
        self, G: Graph, IG: Graph, config: EnsembleCommunitiesBasedSourceDetectionConfig
    ):
        from rpasdt.algorithm.source_detectors.source_detection import (
            get_source_detector,
        )

        super().__init__(G, IG, config)
        self.source_detectors = []
        for detector_name, alg_details in config.source_detectors_config.items():
            self.source_detectors.append(
                get_source_detector(
                    algorithm=alg_details[0], G=G, IG=IG, config=alg_details[1]
                )
            )

    def find_sources_in_community(self, graph: Graph):
        results = defaultdict(float)
        for sd in self.source_detectors:
            result = sd.fin(graph)
            for key, value in result.items():
                results[key] += value

        results = {
            key: value / len(self.source_detectors) for key, value in results.items()
        }
        return results
