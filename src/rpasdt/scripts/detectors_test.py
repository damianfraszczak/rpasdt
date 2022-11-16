import networkx as nx

from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.scripts.taxonomies import source_detectors


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
        # print(source_detector.detected_sources_estimation)
        print(source_detector.detected_sources)
        # for key, value in source_detector.get_additional_data_for_source_evaluation().items():
        #     print("$$$$$$$$")
        #     print(key)
        #     print(value)


if __name__ == "__main__":
    detectors_test()
