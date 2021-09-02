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
    node_count: List[int]
    status_delta: Any

    def __init__(self, raw_trends: List[Dict]):
        parsed_trends = raw_trends[0]["trends"]
        self.node_count = parsed_trends["node_count"]
        self.status_delta = parsed_trends["status_delta"]


@dataclass
class DiffusionIteration:
    iteration: int
    status: Dict[int, int]
    node_count: Dict[int, List[int]]
    status_delta: Dict[int, int]

    @staticmethod
    def from_iterations(raw_iterations: List[Dict]):
        return [DiffusionIteration(iteration) for iteration in raw_iterations]

    def __init__(self, raw_iteration: Dict):
        self.iteration: int = raw_iteration["iteration"]
        self.status: Dict[int, int] = raw_iteration["status"]
        self.node_count: Dict[int, List[int]] = raw_iteration["node_count"]
        self.status_delta: Dict[int, int] = raw_iteration["status_delta"]


@dataclass
class DiffusionSimulationModelResult:
    diffusion_model: DiffusionModel
    trend: DiffusionTrend
    iterations: List[DiffusionIteration] = field(default_factory=list)

    def iteration_to_status_in_population(
        self, node_status: NodeStatusEnum, percentage: float = 100
    ) -> int:
        nodes_count = len(self.diffusion_model.graph.nodes)
        nodes_count_lookup = floor(nodes_count * percentage / 100)
        node_status_int = NodeStatusToValueMapping[node_status]

        for index, nodes_count in enumerate(
            self.trend.node_count.get(node_status_int, [])
        ):
            if nodes_count >= nodes_count_lookup:
                return index
        return len(self.trend.node_count)


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
        model_results = self.results.get(model_name) or []
        model_results.append(
            DiffusionSimulationModelResult(
                diffusion_model=diffusion_model,
                trend=DiffusionTrend(trends),
                iterations=DiffusionIteration.from_iterations(iterations),
            )
        )
        self.results[model_name] = model_results

    def avg_iteration_to_status_in_population(
        self, node_status: NodeStatusEnum, percentage: float = 100
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
