from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Union

from networkx import Graph

from rpasdt.algorithm.models import (
    EnsembleBasedSourceDetectionConfig,
    EnsembleCommunityBasedSourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
    SourceDetector,
)
from rpasdt.common.utils import normalize_dict_values, sort_dict_by_value


class EnsembleLearnerSourceDetector(SourceDetector):
    CONFIG_CLASS = EnsembleBasedSourceDetectionConfig
    source_detectors: List[SourceDetector]

    def __init__(self, G: Graph, IG: Graph, config: EnsembleBasedSourceDetectionConfig):
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
            key: value / len(self.source_detectors) for key, value in results.items()
        }
        self._node_estimations.update(results)
        return results

    def get_additional_data_for_source_evaluation(self) -> Dict[str, Any]:
        return {
            "node_estimations": sort_dict_by_value(self._node_estimations),
            **{
                str(sd): sd.get_additional_data_for_source_evaluation()
                for sd in self.source_detectors
            },
        }


class CommunityEnsembleLearnerSourceDetector(
    EnsembleLearnerSourceDetector, CommunityBasedSourceDetector
):
    CONFIG_CLASS = EnsembleCommunityBasedSourceDetectionConfig

    def __init__(
        self, G: Graph, IG: Graph, config: EnsembleCommunityBasedSourceDetectionConfig
    ):
        super().__init__(G, IG, config)

    @cached_property
    def detected_sources_estimation(self) -> Dict[int, float]:
        nodes_estimation = {}
        for cluster, nodes in self.communities.items():
            estimation = self.estimate_sources(G=self.G, IG=self.IG.subgraph(nodes))
            nodes_estimation.update(self.process_estimation(estimation))
        return sort_dict_by_value(nodes_estimation)
