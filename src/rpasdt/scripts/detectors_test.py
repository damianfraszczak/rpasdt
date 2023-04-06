import networkx as nx

from rpasdt.algorithm.models import (
    CentralityCommunityBasedSourceDetectionConfig,
    EnsembleCommunityBasedSourceDetectionConfig,
    SourceDetectorSimulationConfig,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    SourceDetectionAlgorithm,
)
from rpasdt.scripts.taxonomies import communities

source_detectors = {}
# source_detectors.update(
#     {
#         f"ensemble-centralities:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER,
#             config=EnsembleCommunityBasedSourceDetectionConfig(
#                 number_of_sources=x,
#                 communities_algorithm=cm,
#                 source_detectors_config={
#                     "DEGREE": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.DEGREE,
#                             communities_algorithm=cm,
#                             normalize_results=False,
#                         ),
#                     ),
#                     "BETWEENNESS": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.BETWEENNESS,
#                             communities_algorithm=cm,
#                             normalize_results=False,
#                         ),
#                     ),
#                 },
#             ),
#         )
#         for cm in communities
#     }
# )
source_detectors.update(
    {
        f"centrality-cm:{centrality}:{cm}": lambda x, centrality=centrality, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
            config=CentralityCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                centrality_algorithm=centrality,
                communities_algorithm=cm,
                source_threshold=1,
            ),
        )
        for centrality in [CentralityOptionEnum.DEGREE]
        for cm in communities
    }
)


def detectors_test():
    G = nx.karate_club_graph()
    for name, cc in source_detectors.items():
        print("########")
        source_detector_config = cc(None)
        source_detector = get_source_detector(
            algorithm=source_detector_config.alg,
            G=G,
            IG=G,
            config=source_detector_config.config,
            number_of_sources=None,
        )

        print(name)

        print(source_detector.detected_sources_estimation)
        print(source_detector.detected_sources)
        print(
            source_detector.get_additional_data_for_source_evaluation()["communities"]
        )
        # for key, value in source_detector.get_additional_data_for_source_evaluation().items():
        #     print("$$$$$$$$")
        #     print(key)
        #     print(value)


if __name__ == "__main__":
    detectors_test()
