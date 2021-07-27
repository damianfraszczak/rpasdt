from networkx import Graph

from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.controller.experiment import ExperimentGraphController
from rpasdt.gui.utils import (
    run_long_task,
    show_alert_dialog,
    show_dynamic_dialog,
)
from rpasdt.model.experiment import Experiment
from rpasdt.network.networkx_utils import (
    get_graph_default_properties,
    show_graph_config_dialog,
)


class AppController:
    window: "MainWindow"

    def load_experiment(self, experiment: Experiment, graph: Graph):
        experiment.graph = graph
        self.window.show_experiment_window(
            experiment=experiment,
            controller=ExperimentGraphController(
                window=self.window, experiment=experiment
            ),
        )

    def handler_about_dialog(self):
        show_alert_dialog(
            title="Author information",
            text="Damian Frąszczak Military University of Technology",
        )

    def handler_new_experiment(self):
        experiment = show_dynamic_dialog(
            object=Experiment(), title="Create new experiment"
        )
        if experiment:
            experiment.graph_type_properties = show_graph_config_dialog(
                graph_type=experiment.graph_type,
                graph_type_properties=get_graph_default_properties(
                    experiment.graph_type
                ),
            )
            if experiment.graph_type_properties is not None:
                run_long_task(
                    function=load_graph,
                    function_kwargs={
                        "graph_type": experiment.graph_type,
                        "graph_type_properties": experiment.graph_type_properties,
                    },
                    title="Graph loading",
                    callback=lambda graph: self.load_experiment(
                        experiment=experiment, graph=graph
                    ),
                )
