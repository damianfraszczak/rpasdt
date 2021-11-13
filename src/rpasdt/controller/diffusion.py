from typing import Dict, List

import matplotlib
from ndlib.models import DiffusionModel
from networkx import Graph

from rpasdt.algorithm.diffusion import (
    get_and_init_diffusion_model,
    get_nodes_by_diffusion_status,
)
from rpasdt.algorithm.graph_drawing import get_diffusion_graph
from rpasdt.algorithm.plots import (
    plot_diffusion_prevalence,
    plot_diffusion_trends,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    SourceDetector,
    get_source_detector,
)
from rpasdt.algorithm.taxonomies import (
    DiffusionGraphNodeRenderTypeEnum,
    NodeStatusEnum,
    SourceDetectionAlgorithm,
)
from rpasdt.controller.controllers import (
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
)
from rpasdt.controller.graph import GraphController
from rpasdt.controller.source_detection import SourceDetectionGraphController
from rpasdt.gui.analysis.analysis import SimplePlotAppDialog
from rpasdt.gui.form_utils import get_diffusion_model_form_config
from rpasdt.gui.utils import run_long_task, show_dynamic_dialog
from rpasdt.model.experiment import DiffusionExperiment
from rpasdt.network.taxonomies import NodeAttributeEnum

matplotlib.use("Qt5Agg")


class DiffusionGraphController(
    GraphController,
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
):
    def __init__(self, window: "MainWindow", experiment: DiffusionExperiment):
        super().__init__(window, experiment.diffusion_graph, experiment.graph_config)
        self.experiment = experiment
        self.diffusion_model: DiffusionModel = None
        self.raw_iterations: List[Dict] = list()

    def update_graph(self, graph: Graph):
        self.experiment.diffusion_graph = graph
        super().update_graph(graph)

    def clean_diffusion(self):
        self.diffusion_model = None
        self.update_graph(Graph())

    def init_diffusion(self):
        if self.diffusion_model:
            return
        self.raw_iterations = list()
        source_graph = self.experiment.source_graph
        source_nodes = self.experiment.source_nodes
        (
            self.diffusion_model,
            self.experiment.diffusion_model_properties,
        ) = get_and_init_diffusion_model(
            graph=source_graph,
            diffusion_type=self.experiment.diffusion_type,
            model_params=self.experiment.diffusion_model_properties,
            source_nodes=source_nodes,
        )

    def handler_edit_diffusion(self):
        self.init_diffusion()
        diffusion_model_properties = show_dynamic_dialog(
            object=self.experiment.diffusion_model_properties,
            config=get_diffusion_model_form_config(self.diffusion_model),
        )
        if diffusion_model_properties:
            self.experiment.diffusion_model_properties = diffusion_model_properties
            self.clean_diffusion()
            self.init_diffusion()

    def diffusion_clear_handler(self):
        self.clean_diffusion()
        self.init_diffusion()

    def diffusion_execute_iteration_handler(self):
        self.init_diffusion()
        iteration = self.diffusion_model.iteration()
        self.update_status_after_iterations([iteration])
        return iteration

    def diffusion_execute_iteration_bunch(self):
        self.init_diffusion()
        iterations = self.diffusion_model.iteration_bunch(
            self.experiment.diffusion_iteration_bunch
        )
        self.update_status_after_iterations(iterations)
        return iterations

    def diffusion_execute_iteration_bunch_handler(self):
        run_long_task(
            title="Computing degree centrality",
            function=self.diffusion_execute_iteration_bunch,
            callback=lambda iterations: self.redraw_graph(),
        )

    def update_status_after_iterations(self, iterations: List[Dict]):
        self.graph_panel.title = f"Iteration {self.diffusion_model.actual_iteration}"
        self.raw_iterations.extend(iterations)
        self.update_diffusion_graph()
        self.redraw_graph()

    @property
    def infected_nodes(self) -> List[int]:
        return get_nodes_by_diffusion_status(
            diffusion_model=self.diffusion_model, node_status=NodeStatusEnum.INFECTED
        )

    def graph_config_changed(self):
        self.update_diffusion_graph()
        super().graph_config_changed()

    def update_diffusion_graph(self):
        self.update_graph(
            get_diffusion_graph(
                source_graph=self.experiment.source_graph,
                infected_nodes=self.infected_nodes,
                graph_node_rendering_type=self.graph_config.graph_node_rendering_type,
            )
        )

    def redraw_graph(self):
        for infected_node in list(
            set(self.infected_nodes) - set(self.experiment.source_nodes)
        ):
            self.graph.nodes[infected_node][
                NodeAttributeEnum.COLOR
            ] = self.experiment.graph_config.infected_node_color
        super().redraw_graph()

    def handler_configure_source_detection(self, algorithm: SourceDetectionAlgorithm):
        IG = get_diffusion_graph(
            source_graph=self.experiment.source_graph,
            infected_nodes=self.infected_nodes,
            graph_node_rendering_type=DiffusionGraphNodeRenderTypeEnum.ONLY_INFECTED,
        )
        source_detector = get_source_detector(
            algorithm=algorithm,
            G=self.experiment.source_graph,
            IG=IG,
            number_of_sources=len(self.experiment.source_nodes),
        )
        config = show_dynamic_dialog(
            source_detector.config, "Configure source detector"
        )
        if config:
            run_long_task(
                function=source_detector.estimate_sources,
                title="Source estimation",
                callback=lambda sources: self.process_source_detection(
                    source_detector=source_detector
                ),
            )

    def process_source_detection(self, source_detector: SourceDetector):
        self.window.show_source_detection_window(
            controller=SourceDetectionGraphController(
                window=self.window,
                experiment=self.experiment,
                source_detector=source_detector,
            )
        )

    def handler_plot_diffusion_trend(self):
        self.window.add_subwindow(
            SimplePlotAppDialog(
                title=f"Trends for {self.diffusion_model.name}",
                plot_renderer=lambda: plot_diffusion_trends(
                    self.diffusion_model, self.raw_iterations
                ),
            )
        )

    def handler_plot_diffusion_prevalence(self):
        self.window.add_subwindow(
            SimplePlotAppDialog(
                title=f"Prevalence for {self.diffusion_model.name}",
                plot_renderer=lambda: plot_diffusion_prevalence(
                    self.diffusion_model, self.raw_iterations
                ),
            )
        )
