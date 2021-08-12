from rpasdt.algorithm.graph_export_import import GRAPH_EXPORTER, GRAPH_IMPORTER
from rpasdt.algorithm.taxonomies import GraphDataFormatEnum
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
    json_data = export_model.to_json()
    with open(file_path, "w") as file:
        file.write(json_data)


def import_experiment(file_path: str) -> Experiment:
    with open(file_path, "r") as file:
        json_data = file.read()
    export_model: ExperimentExportModel = ExperimentExportModel.from_json(json_data)
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
