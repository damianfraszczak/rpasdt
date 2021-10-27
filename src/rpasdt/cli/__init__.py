"""CLI interface.."""
from typing import Any, Dict, List, Optional

from networkx import Graph

from rpasdt.algorithm.centralities import compute_centrality
from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.diffusion import (
    get_and_init_diffusion_model,
    get_nodes_by_diffusion_status,
)
from rpasdt.algorithm.graph_drawing import get_diffusion_graph
from rpasdt.algorithm.graph_export_import import GRAPH_EXPORTER
from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.algorithm.models import (
    DiffusionSimulationConfig,
    SourceDetectionSimulationConfig,
)
from rpasdt.algorithm.simulation import (
    perform_diffusion_simulation,
    perform_source_detection_simulation,
)
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    DiffusionTypeEnum,
    GraphDataFormatEnum,
    GraphTypeEnum,
    NodeStatusEnum,
)
from rpasdt.common.utils import import_dataclass_from_json


def print_result(result: Any):
    print(result)


def transform_result_to_lines(result: Any):
    if isinstance(result, dict):
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, str):
        return [result]


def save_result(result: Any, output_file_path: str) -> None:
    with open(output_file_path, "w") as file:
        for line in transform_result_to_lines(result):
            file.write(f"{line}\n")


def _load_graph(input_graph_path: str, graph_data_format: GraphDataFormatEnum) -> Graph:
    return load_graph(
        graph_type=GraphTypeEnum.CUSTOM,
        graph_type_properties={
            "file_path": input_graph_path,
            "graph_data_format": graph_data_format,
        },
    )


class CLIInterface:
    """RP&SDT command line interface."""

    def generate_graph(
        self,
        graph_type: GraphTypeEnum,
        output_file_path: str,
        graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
        graph_type_properties: Optional[Dict] = None,
    ):
        """Generate graph.
        :param graph_type: graph_type to be generated
        :param graph_data_format: format of the graph to be saved
        :param output_file_path: file path to store a generated graph
        :param graph_type_properties: key=value graph params if not provided the default
        values will be used (default: None)
        will be printed to the console (default: None)
        """
        graph = load_graph(
            graph_type=graph_type, graph_type_properties=graph_type_properties
        )
        result = GRAPH_EXPORTER[graph_data_format](graph)
        save_result(result, output_file_path)

    def compute_centrality(
        self,
        input_graph_path: str,
        centrality: CentralityOptionEnum,
        graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
        output_file_path: Optional[str] = None,
    ) -> None:
        """Compute centrality for the given graph.
        :param input_graph_path: file path to the input graph
        :param graph_data_format: format of the input graph
        :param centrality: centrality algorithm
        :param output_file_path: file path to store result if not provided the result
        will be printed to the console (default: None)
        """
        graph = _load_graph(
            input_graph_path=input_graph_path, graph_data_format=graph_data_format
        )
        result = compute_centrality(graph=graph, type=centrality)
        if output_file_path:
            save_result(result, output_file_path)
        else:
            print_result(result)

    def compute_communities(
        self,
        input_graph_path: str,
        community: CommunityOptionEnum,
        graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
        community_properties: Optional[Dict] = None,
        output_file_path: Optional[str] = None,
    ) -> None:
        """Compute communities for the given graph.
        :param input_graph_path: file path to the input graph
        :param graph_data_format: format of the input graph
        :param community: community algorithm
        :param community_properties: properties of the community alg
        :param output_file_path: file path to store result if not provided the result
        will be printed to the console (default: None)
        """
        community_properties = community_properties or {}
        graph = _load_graph(
            input_graph_path=input_graph_path, graph_data_format=graph_data_format
        )
        result = find_communities(graph=graph, type=community, **community_properties)
        if output_file_path:
            save_result(result, output_file_path)
        else:
            print_result(result)

    def simulate_diffusion(
        self,
        input_graph_path: str,
        diffusion_type: DiffusionTypeEnum,
        source_nodes: List[int],
        graph_data_format: GraphDataFormatEnum = GraphDataFormatEnum.MULTILINE_ADJLIST,
        iterations: int = 200,
        model_params: Optional[Dict] = None,
        output_file_path: Optional[str] = None,
    ) -> None:
        """Simulate diffusion with selected model under given network.
        :param input_graph_path: file path to the input graph
        :param diffusion_type: diffusion type
        :param source_nodes: list of source nodes
        :param graph_data_format: format of the input graph
        :param iterations: number of iterations to simulate
        :param model_params: model parameters
        :param output_file_path: file path to store result if not provided the result
        will be printed to the console (default: None)
        """
        graph = _load_graph(
            input_graph_path=input_graph_path, graph_data_format=graph_data_format
        )

        diffusion_model, params = get_and_init_diffusion_model(
            graph=graph,
            diffusion_type=diffusion_type,
            source_nodes=source_nodes,
            model_params=model_params,
        )
        diffusion_model.iteration_bunch(iterations)
        diffusion_graph = get_diffusion_graph(
            source_graph=graph,
            infected_nodes=get_nodes_by_diffusion_status(
                diffusion_model=diffusion_model, node_status=NodeStatusEnum.INFECTED
            ),
        )
        result = GRAPH_EXPORTER[graph_data_format](diffusion_graph)
        if output_file_path:
            save_result(result, output_file_path)
        else:
            print_result(result)

    def diffusion_simulation_experiment(
        self, config_file_path: str, output_file_path: Optional[str] = None
    ) -> None:
        """Perform diffusion simulation experiment.
        :param config_file_path: file path to the DiffusionSimulationConfig json.
        :param output_file_path: file path to store result if not provided the result
        will be printed to the console (default: None)
        """
        simulation_config: DiffusionSimulationConfig = import_dataclass_from_json(
            file_path=config_file_path, type=DiffusionSimulationConfig
        )
        result = perform_diffusion_simulation(simulation_config=simulation_config)
        if output_file_path:
            save_result(result, output_file_path)
        else:
            print_result(result)

    def source_detection_experiment(
        self, config_file_path: str, output_file_path: Optional[str] = None
    ) -> None:
        """Perform diffusion simulation experiment.
        :param config_file_path: file path to the SourceDetectionSimulationConfig json.
        :param output_file_path: file path to store result if not provided the result
        will be printed to the console (default: None)
        """
        source_detection_config: SourceDetectionSimulationConfig = (
            import_dataclass_from_json(
                file_path=config_file_path, type=SourceDetectionSimulationConfig
            )
        )
        result = perform_source_detection_simulation(
            source_detection_config=source_detection_config
        )
        if output_file_path:
            save_result(result, output_file_path)
        else:
            print_result(result)
