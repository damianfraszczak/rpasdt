from rpasdt.algorithm.graph_export_import import GRAPH_EXPORTER, GRAPH_IMPORTER
from rpasdt.algorithm.taxonomies import GraphDataFormatEnum
from rpasdt.common.utils import (
    export_dataclass_as_json,
    import_dataclass_from_json,
)
from rpasdt.model.experiment import Experiment, ExperimentExportModel


def export_experiment(
    experiment: Experiment,
    graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
) -> ExperimentExportModel:
    return ExperimentExportModel(
        name=experiment.name,
        graph_type=experiment.graph_type,
        graph_type_properties=experiment.graph_type_properties,
        graph_config=experiment.graph_config,
        graph_data=GRAPH_EXPORTER[graph_data_format](experiment.graph),
        graph_data_format=graph_data_format,
    )


def save_experiment(
    experiment: Experiment,
    file_path: str,
    graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
) -> None:
    export_model: ExperimentExportModel = export_experiment(
        experiment=experiment, graph_data_format=graph_data_format
    )
    export_dataclass_as_json(object=export_model, file_path=file_path)


def import_experiment(file_path: str) -> Experiment:
    export_model: ExperimentExportModel = import_dataclass_from_json(
        type=ExperimentExportModel, file_path=file_path
    )

    return Experiment(
        name=export_model.name,
        graph_config=export_model.graph_config,
        graph_type=export_model.graph_type,
        graph_type_properties=export_model.graph_type_properties,
        graph=GRAPH_IMPORTER[export_model.graph_data_format](export_model.graph_data),
    )


# save_experiment(Experiment(graph=nx.karate_club_graph()), "file.json")
# experiment = import_experiment("file.json")

#
# with open("graph.adajcency", "w") as file:
#     file.write("\n".join(GRAPH_EXPORTER[GraphDataFormatEnum.MULTILINE_ADJLIST](nx.karate_club_graph())))

# print(load_custom_graph(
#     {"graph_data_format": GraphDataFormatEnum.MULTILINE_ADJLIST,
#      "file_path": "graph.adajcency"}))
