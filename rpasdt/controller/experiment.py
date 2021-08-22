from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.algorithm.models import NetworkSourceSelectionConfig
from rpasdt.algorithm.source_selection import select_sources
from rpasdt.controller.controllers import (
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
)
from rpasdt.controller.diffusion import DiffusionGraphController
from rpasdt.controller.graph import GraphController
from rpasdt.gui.dynamic_form.models import DynamicFormConfig
from rpasdt.gui.utils import (
    run_long_task,
    show_dynamic_dialog,
    show_save_file_dialog,
)
from rpasdt.model.experiment import DiffusionExperiment, Experiment
from rpasdt.model.utils import save_experiment
from rpasdt.network.models import NodeAttribute
from rpasdt.network.networkx_utils import (
    set_node_attributes,
    show_graph_config_dialog,
)
from rpasdt.network.taxonomies import NodeAttributeEnum


class ExperimentGraphController(
    GraphController,
    CentralityAnalysisControllerMixin,
    CommunityAnalysisControllerMixin,
    NetworkAnalysisControllerMixin,
):
    def __init__(self, window: "MainWindow", experiment: Experiment):
        super().__init__(window, experiment.graph, experiment.graph_config)
        self.experiment = experiment

    def reload_experiment(self, experiment: Experiment):
        if experiment:
            run_long_task(
                function=load_graph,
                function_kwargs={
                    "graph_type": experiment.graph_type,
                    "graph_type_properties": experiment.graph_type_properties,
                },
                title="Graph loading",
                callback=lambda graph: self.update_graph(graph) and self.redraw_graph(),
            )

    def handler_edit_experiment(self):
        self.reload_experiment(
            show_dynamic_dialog(
                object=self.experiment,
                title=f'Edit experiment {self.experiment.name or ""}',
                config=DynamicFormConfig(),
            )
        )

    def handler_edit_graph_type_properties(self):
        graph_type_properties = show_graph_config_dialog(
            graph_type=self.experiment.graph_typee,
            graph_type_properties=self.experiment.graph_type_properties,
        )
        if (
            graph_type_properties
            and graph_type_properties != self.experiment.graph_type_properties
        ):
            self.experiment.graph_type_properties = graph_type_properties
            self.reload_experiment(self.experiment)

    def handler_create_diffusion(self):
        experiment = show_dynamic_dialog(
            object=DiffusionExperiment(
                source_graph=self.graph, graph_config=self.graph_config
            ),
            title="Create a new diffusion experiment",
        )
        if experiment:
            self.window.show_diffusion_window(
                experiment=experiment,
                controller=DiffusionGraphController(
                    window=self.window, experiment=experiment
                ),
            )

    def handler_clear_sources(self):
        set_node_attributes(
            self.graph,
            [
                NodeAttribute(NodeAttributeEnum.SOURCE, False),
                NodeAttribute(
                    NodeAttributeEnum.COLOR, self.graph_config.susceptible_node_color
                ),
            ],
        )
        self.redraw_graph()

    def handler_select_sources(self):
        sources_config = show_dynamic_dialog(
            object=NetworkSourceSelectionConfig(), title="Select sources automatically"
        )
        if sources_config:
            self.handler_clear_sources()
            sources = select_sources(config=sources_config, graph=self.graph)
            set_node_attributes(
                self.graph,
                [
                    NodeAttribute(NodeAttributeEnum.SOURCE, True),
                    NodeAttribute(
                        NodeAttributeEnum.COLOR, self.graph_config.source_node_color
                    ),
                ],
                nodes_list=sources,
            )
            self.redraw_graph()

    def handler_graph_node_edited(self, node):
        if node.get(NodeAttributeEnum.SOURCE):
            node[NodeAttributeEnum.COLOR] = self.graph_config.source_node_color
        elif node.get(NodeAttributeEnum.COLOR) == self.graph_config.source_node_color:
            node[NodeAttributeEnum.COLOR] = self.graph_config.susceptible_node_color
        super().handler_graph_node_edited(node)

    def handler_export_experiment(self):
        file_path = show_save_file_dialog()
        if file_path:
            # TODO run in async
            save_experiment(experiment=self.experiment, file_path=file_path)
