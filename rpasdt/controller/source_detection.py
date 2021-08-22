from rpasdt.algorithm.source_detection_evaluation import (
    compute_source_detection_evaluation,
)
from rpasdt.algorithm.source_detectors.source_detection import SourceDetector
from rpasdt.controller.controllers import (
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
)
from rpasdt.controller.graph import GraphController
from rpasdt.gui.analysis.models import SourceDetectionAnalysisData
from rpasdt.model.experiment import DiffusionExperiment
from rpasdt.network.taxonomies import NodeAttributeEnum


class SourceDetectionGraphController(
    GraphController,
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
):
    def __init__(
        self,
        window: "MainWindow",
        experiment: DiffusionExperiment,
        source_detector: SourceDetector,
    ):
        super().__init__(window, experiment.diffusion_graph, experiment.graph_config)
        self.experiment = experiment
        self.source_detector = source_detector
        self.data = SourceDetectionAnalysisData(
            graph=self.graph,
            detected_sources=self.source_detector.detected_sources,
            real_sources=self.experiment.source_nodes,
            evaluation=compute_source_detection_evaluation(
                G=self.experiment.diffusion_graph,
                real_sources=self.experiment.source_nodes,
                detected_sources=self.source_detector.detected_sources,
            ),
        )
        self.mark_nodes()

    def mark_nodes(self):
        detected_sources = (
            self.source_detector.detected_sources
            if isinstance(self.source_detector.detected_sources, list)
            else [self.source_detector.detected_sources]
        )
        for estimated_node in detected_sources:
            self.graph.nodes[estimated_node][
                NodeAttributeEnum.COLOR
            ] = self.experiment.graph_config.estimated_source_node_color
