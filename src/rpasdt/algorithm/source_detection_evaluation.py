"""Source detection evaluation metrics."""
from typing import Any, Dict, List, Set, Union

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.models import (
    ClassificationMetrics,
    ExperimentSourceDetectionEvaluation,
    SingleSourceDetectionEvaluation,
)
from rpasdt.algorithm.utils import shortest_path_length
from rpasdt.common.utils import multi_sum


def compute_error_distance(
    G: Graph, not_detected_sources: Set[int], invalid_detected_sources: Set[int]
):
    bigger, smaller = invalid_detected_sources, not_detected_sources
    if len(invalid_detected_sources) >= len(not_detected_sources):
        bigger, smaller = not_detected_sources, invalid_detected_sources
    if not smaller:
        smaller = [-1 for node in bigger]
    return sum(
        [
            min(
                [
                    shortest_path_length(G, source=source, target=smaller)
                    if source in G and smaller in G
                    else len(G)
                    for smaller in invalid_detected_sources
                ]
            )
            for source in bigger
        ]
    )


def compute_confusion_matrix(
    real_sources: List[int],
    detected_sources: Union[int, List[int]],
    all_nodes_count: int,
) -> ClassificationMetrics:
    detected_sources = (
        detected_sources if isinstance(detected_sources, list) else [detected_sources]
    )
    correctly_detected_sources = set(real_sources).intersection(detected_sources)
    invalid_detected_sources = set(detected_sources).difference(
        correctly_detected_sources
    )
    not_detected_sources = set(real_sources).difference(correctly_detected_sources)
    P = len(real_sources)
    N = all_nodes_count - P
    FP = len(invalid_detected_sources)
    TP = len(correctly_detected_sources)
    FN = len(real_sources) - TP
    TN = N - FN

    return ClassificationMetrics(TP=TP, FP=FP, TN=TN, FN=FN, P=P, N=N)


def compute_source_detection_evaluation(
    G: Graph,
    real_sources: List[int],
    detected_sources: Union[int, List[int]],
    additional_data: Dict[str, Any],
) -> SingleSourceDetectionEvaluation:
    detected_sources = (
        detected_sources if isinstance(detected_sources, list) else [detected_sources]
    )
    correctly_detected_sources = set(real_sources).intersection(detected_sources)
    invalid_detected_sources = set(detected_sources).difference(
        correctly_detected_sources
    )
    not_detected_sources = set(real_sources).difference(correctly_detected_sources)
    P = len(real_sources)
    N = len(G.nodes) - P
    FP = len(invalid_detected_sources)
    TP = len(correctly_detected_sources)
    FN = len(real_sources) - TP
    TN = N - FN

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
        TN=TN,
        FN=FN,
        P=P,
        N=N,
        additional_data=additional_data,
    )


def compute_source_detection_experiment_evaluation(
    evaluations: List[SingleSourceDetectionEvaluation],
) -> ExperimentSourceDetectionEvaluation:
    aggregated = multi_sum(
        evaluations, "TP", "TN", "FP", "FN", "P", "N", "error_distance"
    )
    TP, TN, FP, FN, P, N, error_distance = (
        aggregated["TP"],
        aggregated["TN"],
        aggregated["FP"],
        aggregated["FN"],
        aggregated["P"],
        aggregated["N"],
        aggregated["error_distance"],
    )
    avg_error_distance = error_distance / len(evaluations)
    real_sources = []
    detected_sources = []
    additional_data = []
    time = 0
    for result in evaluations:
        real_sources.append(result.real_sources)
        detected_sources.append(result.detected_sources)
        additional_data.append(result.additional_data)
        time += result.additional_data.get("time", 0)

    return ExperimentSourceDetectionEvaluation(
        avg_error_distance=avg_error_distance,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN,
        P=P,
        N=N,
        real_sources=real_sources,
        detected_sources=detected_sources,
        additional_data=additional_data,
        avg_execution_time=time / len(evaluations),
    )
