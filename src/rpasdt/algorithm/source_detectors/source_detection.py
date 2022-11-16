"""Source detection utility methods."""
from typing import Optional

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.models import (
    CentralityCommunityBasedSourceDetectionConfig,
    SourceDetectionConfig,
)
from rpasdt.algorithm.source_detectors.centrality import (
    CentralityBasedSourceDetector,
    CentralityCommunityBasedSourceDetector,
    MultipleCentralityBasedSourceDetector,
    UnbiasedCentralityBasedSourceDetector,
    UnbiasedCentralityCommunityBasedSourceDetector,
)
from rpasdt.algorithm.source_detectors.common import SourceDetector
from rpasdt.algorithm.source_detectors.dynamic_age import (
    DynamicAgeSourceDetector,
)
from rpasdt.algorithm.source_detectors.ensemble_learner import (
    CommunityEnsembleLearnerSourceDetector,
)
from rpasdt.algorithm.source_detectors.jordan_center import (
    JordanCenterCommunityBasedSourceDetector,
)
from rpasdt.algorithm.source_detectors.net_sleuth import (
    NetSleuthCommunityBasedSourceDetector,
)
from rpasdt.algorithm.source_detectors.rumour_center import (
    RumorCenterCommunityBasedSourceDetector,
)
from rpasdt.algorithm.taxonomies import (
    CommunityOptionEnum,
    SourceDetectionAlgorithm,
)

SOURCE_DETECTORS = {
    SourceDetectionAlgorithm.CENTRALITY_BASED: CentralityBasedSourceDetector,
    SourceDetectionAlgorithm.MULTIPLE_CENTRALITY_BASED: MultipleCentralityBasedSourceDetector,
    SourceDetectionAlgorithm.UNBIASED_CENTRALITY_BASED: UnbiasedCentralityBasedSourceDetector,
    SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED: CentralityCommunityBasedSourceDetector,
    SourceDetectionAlgorithm.COMMUNITY_UNBIASED_CENTRALITY_BASED: UnbiasedCentralityCommunityBasedSourceDetector,
    SourceDetectionAlgorithm.DYNAMIC_AGE: DynamicAgeSourceDetector,
    SourceDetectionAlgorithm.NET_SLEUTH: NetSleuthCommunityBasedSourceDetector,
    SourceDetectionAlgorithm.RUMOR_CENTER: RumorCenterCommunityBasedSourceDetector,
    SourceDetectionAlgorithm.JORDAN_CENTER: JordanCenterCommunityBasedSourceDetector,
    SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER: CommunityEnsembleLearnerSourceDetector,
}


def get_source_detector(
    algorithm: SourceDetectionAlgorithm,
    G: Graph,
    IG: Graph,
    number_of_sources=1,
    config: Optional[SourceDetectionConfig] = None,
    *args,
    **kwargs
) -> SourceDetector:
    detector = SOURCE_DETECTORS.get(algorithm)(G=G, IG=IG, config=config)
    if number_of_sources > 1:
        detector.config.number_of_sources = number_of_sources
    for key, value in kwargs.items():
        setattr(detector.config, key, value)
    return detector


def detectors_test():
    G = nx.karate_club_graph()
    detectors = [
        # get_source_detector(
        #     SourceDetectionAlgorithm.CENTRALITY_BASED,
        #     G=G,
        #     IG=G,
        #     config=CentralityBasedSourceDetectionConfig(
        #         source_threshold=0.1,
        #     ),
        # ),
        get_source_detector(
            SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
            G=G,
            IG=G,
            config=CentralityCommunityBasedSourceDetectionConfig(
                source_threshold=0.0, communities_algorithm=CommunityOptionEnum.LOUVAIN
            ),
        ),
    ]
    for detector in detectors:
        print("########")
        print(str(detector))
        print(detector.detected_sources_estimation)
        print(detector.detected_sources)
        for key, value in detector.get_additional_data_for_source_evaluation().items():
            print("$$$$$$$$")
            print(key)
            print(value)


if __name__ == "__main__":
    detectors_test()
