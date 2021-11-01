"""Models used to describe algorithms."""
import statistics
from dataclasses import dataclass, field
from functools import cached_property
from math import floor
from typing import Any, Dict, List, Optional

from ndlib.models.DiffusionModel import DiffusionModel
from networkx import Graph

from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    DiffusionTypeEnum,
    GraphTypeEnum,
    NodeStatusEnum,
    NodeStatusToValueMapping,
    SourceDetectionAlgorithm,
    SourceSelectionOptionEnum,
)

DIFFUSION_NOT_COVERED = -1


class NetworkInformation:
    summary: str
    density: float
    avg_degree: float
    diameter: int
    average_clustering: float
    degree_histogram: List[int]
    number_of_edges: int
    bridges: List[int]
    number_connected_components: int


@dataclass
class NetworkSourceSelectionConfig:
    number_of_sources: int = 1
    algorithm: SourceSelectionOptionEnum = SourceSelectionOptionEnum.RANDOM


@dataclass
class SourceDetectionConfig:
    number_of_sources: Optional[int] = 1


@dataclass
class CentralityBasedSourceDetectionConfig(SourceDetectionConfig):
    centrality_algorithm: CentralityOptionEnum = CentralityOptionEnum.DEGREE


@dataclass
class MultipleCentralityBasedSourceDetectionConfig(SourceDetectionConfig):
    centrality_algorithms: List[CentralityOptionEnum] = field(default_factory=list)


@dataclass
class UnbiasedCentralityBasedSourceDetectionConfig(
    CentralityBasedSourceDetectionConfig
):
    r: float = 0.85


@dataclass
class CommunitiesBasedSourceDetectionConfig(SourceDetectionConfig):
    communities_algorithm: CommunityOptionEnum = CommunityOptionEnum.GIRVAN_NEWMAN


@dataclass
class CentralityCommunityBasedSourceDetectionConfig(
    CommunitiesBasedSourceDetectionConfig
):
    centrality_algorithm: CentralityOptionEnum = CentralityOptionEnum.DEGREE


@dataclass
class UnbiasedCentralityCommunityBasedSourceDetectionConfig(
    CentralityCommunityBasedSourceDetectionConfig
):
    r: float = 0.85


@dataclass
class SingleSourceDetectionEvaluation:
    G: Graph
    real_sources: List[int]
    detected_sources: List[int]
    error_distance: int
    TP: int
    FP: int
    FN: int


@dataclass
class ExperimentSourceDetectionEvaluation:
    avg_error_distance: int
    recall: float
    precision: float
    f1score: float


@dataclass
class SimulationConfig:
    number_of_experiments: int = 10

    graph: Optional[Graph] = None
    graph_type: Optional[GraphTypeEnum] = GraphTypeEnum.KARATE_CLUB
    graph_type_properties: Dict[str, Any] = field(default_factory=dict)

    source_nodes: Optional[List[int]] = None
    source_selection_config: NetworkSourceSelectionConfig = (
        NetworkSourceSelectionConfig()
    )


@dataclass
class DiffusionModelSimulationConfig:
    diffusion_model_type: DiffusionTypeEnum = DiffusionTypeEnum.SI
    diffusion_model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffusionSimulationConfig(SimulationConfig):
    iteration_bunch: int = 200
    diffusion_models: List[DiffusionModelSimulationConfig] = field(default_factory=list)


@dataclass
class DiffusionTrend:
    node_count: Dict = field(default=dict)  # [int, List]
    status_delta: Dict = field(default=dict)

    @staticmethod
    def from_raw_trends(raw_trends: List[Dict]):
        parsed_trends = raw_trends[0]["trends"]
        return DiffusionTrend(
            node_count=parsed_trends["node_count"],
            status_delta=parsed_trends["status_delta"],
        )


@dataclass
class DiffusionIteration:
    iteration: int
    status: Dict[int, int] = field(default=dict)
    node_count: Dict[int, Any] = field(default=dict)  # Dict[int, List[int]]
    status_delta: Dict[int, int] = field(default=dict)

    @staticmethod
    def from_iterations(raw_iterations: List[Dict]):
        def create_iteration(raw_iteration: Dict):
            return DiffusionIteration(
                iteration=raw_iteration["iteration"],
                status=raw_iteration["status"],
                node_count=raw_iteration["node_count"],
                status_delta=raw_iteration["status_delta"],
            )

        return [create_iteration(iteration) for iteration in raw_iterations]


@dataclass
class DiffusionSimulationModelResult:
    nodes_count: int
    edges_count: int
    trend: DiffusionTrend

    iterations: List[DiffusionIteration] = field(default_factory=list)

    def iteration_to_status_in_population(
        self,
        node_status: NodeStatusEnum = NodeStatusEnum.INFECTED,
        percentage: float = 100,
    ) -> int:
        nodes_count_lookup = floor(self.nodes_count * percentage / 100)
        node_status_int = NodeStatusToValueMapping[node_status]
        node_counts = (
            self.trend.node_count.get(node_status_int)
            or self.trend.node_count.get(str(node_status_int))
            or []
        )
        for index, nodes_count in enumerate(node_counts):
            if nodes_count >= nodes_count_lookup:
                return index
        return DIFFUSION_NOT_COVERED


@dataclass
class DiffusionSimulationResult:
    simulation_config: DiffusionSimulationConfig
    results: Dict[str, List[DiffusionSimulationModelResult]] = field(
        default_factory=dict
    )

    def add_result(
        self,
        diffusion_model: DiffusionModel,
        iterations: List[Dict],
        trends: List[Dict],
    ) -> None:
        model_name = diffusion_model.name
        nodes_count = len(diffusion_model.graph.nodes)
        edges_count = len(diffusion_model.graph.edges)
        model_results = self.results.get(model_name) or []

        model_results.append(
            DiffusionSimulationModelResult(
                nodes_count=nodes_count,
                edges_count=edges_count,
                trend=DiffusionTrend.from_raw_trends(trends),
                iterations=DiffusionIteration.from_iterations(iterations),
            )
        )
        self.results[model_name] = model_results

    def avg_iteration_to_status_in_population(
        self,
        node_status: NodeStatusEnum = NodeStatusEnum.INFECTED,
        percentage: float = 100,
    ) -> Dict[str, float]:
        return {
            model_name: statistics.mean(
                map(
                    lambda x: x.iteration_to_status_in_population(
                        node_status=node_status, percentage=percentage
                    ),
                    data,
                )
            )
            for model_name, data in self.results.items()
        }


@dataclass(frozen=True)
class SourceDetectorSimulationConfig:
    alg: SourceDetectionAlgorithm
    config: SourceDetectionConfig


@dataclass
class SourceDetectionSimulationConfig(DiffusionSimulationConfig):
    source_detectors: Dict[str, SourceDetectorSimulationConfig] = field(
        default_factory=dict
    )


@dataclass
class SourceDetectionSimulationResult:
    source_detection_config: SourceDetectionSimulationConfig
    raw_results: Dict[str, List[SingleSourceDetectionEvaluation]] = field(
        default_factory=dict
    )

    def add_result(
        self,
        name: str,
        source_detector_config: SourceDetectorSimulationConfig,
        evaluation: SingleSourceDetectionEvaluation,
    ) -> None:
        detector_results = self.raw_results.get(name) or []
        detector_results.append(evaluation)
        self.raw_results[name] = detector_results

    @cached_property
    def aggregated_results(
        self,
    ) -> Dict[SourceDetectorSimulationConfig, ExperimentSourceDetectionEvaluation]:
        from rpasdt.algorithm.source_detection_evaluation import (
            compute_source_detection_experiment_evaluation,
        )

        return {
            config: compute_source_detection_experiment_evaluation(results)
            for config, results in self.raw_results.items()
        }
