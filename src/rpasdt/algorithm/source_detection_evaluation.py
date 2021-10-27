"""Source detection evaluation metrics."""
from typing import List, Set, Union

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.models import (
    ExperimentSourceDetectionEvaluation,
    SingleSourceDetectionEvaluation,
)
from rpasdt.common.utils import multi_sum


def compute_error_distance(
    G: Graph, not_detected_sources: Set[int], invalid_detected_sources: Set[int]
):
    if not_detected_sources and invalid_detected_sources:
        return sum(
            [
                min(
                    [
                        nx.shortest_path_length(G, source=source, target=invalid_source)
                        for invalid_source in invalid_detected_sources
                    ]
                )
                for source in not_detected_sources
            ]
        )
    else:
        return 0


def compute_source_detection_evaluation(
    G: Graph, real_sources: List[int], detected_sources: Union[int, List[int]]
) -> SingleSourceDetectionEvaluation:
    detected_sources = (
        detected_sources if isinstance(detected_sources, list) else [detected_sources]
    )
    correctly_detected_sources = set(real_sources).intersection(detected_sources)
    invalid_detected_sources = set(detected_sources).difference(
        correctly_detected_sources
    )
    not_detected_sources = set(real_sources).difference(correctly_detected_sources)
    TP = len(correctly_detected_sources)
    FP = len(invalid_detected_sources)
    FN = len(real_sources) - TP
    error_distance = compute_error_distance(
        G=G,
        not_detected_sources=not_detected_sources,
        invalid_detected_sources=invalid_detected_sources,
    )

    return SingleSourceDetectionEvaluation(
        G=G,
        real_sources=real_sources,
        detected_sources=detected_sources,
        error_distance=error_distance,
        TP=TP,
        FP=FP,
        FN=FN,
    )


def compute_source_detection_experiment_evaluation(
    evaluations: List[SingleSourceDetectionEvaluation],
) -> ExperimentSourceDetectionEvaluation:
    aggregated = multi_sum(evaluations, "FP", "TP", "FN", "error_distance")
    FP, TP, FN, error_distance = (
        aggregated["FP"],
        aggregated["TP"],
        aggregated["FN"],
        aggregated["error_distance"],
    )
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = (
        0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    avg_error_distance = error_distance / len(evaluations)
    return ExperimentSourceDetectionEvaluation(
        f1score=f1score,
        precision=precision,
        recall=recall,
        avg_error_distance=avg_error_distance,
    )
