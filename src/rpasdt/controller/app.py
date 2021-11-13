"""Main controller."""
from networkx import Graph

from rpasdt import version
from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.controller.experiment import ExperimentGraphController
from rpasdt.gui.utils import (
    run_long_task,
    show_alert_dialog,
    show_dynamic_dialog,
    show_open_file_dialog,
)
from rpasdt.model.experiment import Experiment
from rpasdt.model.utils import import_experiment
from rpasdt.network.networkx_utils import (
    get_graph_default_properties,
    show_graph_config_dialog,
)


class AppController:
    """Main controller responsible for common interactions."""

    window: "MainWindow"

    def load_experiment(self, experiment: Experiment, graph: Graph):
        """Load experiment into toolkit handler."""
        experiment.graph = graph
        self.window.show_experiment_window(
            experiment=experiment,
            controller=ExperimentGraphController(
                window=self.window, experiment=experiment
            ),
        )

    def handler_about_dialog(self):
        """Display about dialog handler."""
        show_alert_dialog(
            title="Author information",
            text=f"Damian Frąszczak Military University of Technology.\n"
            f"RP&SDT {version.__version__} version",
        )

    def handler_new_experiment(self):
        """Show a new experiment dialog handler."""
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

    def handler_import_experiment(self):
        """Import experiment form file into toolkit handler."""
        file_path = show_open_file_dialog()
        if file_path:
            # TODO run in async
            experiment = import_experiment(file_path)
            self.load_experiment(experiment=experiment, graph=experiment.graph)
