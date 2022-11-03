import csv
import math

import networkx as nx

from rpasdt.algorithm.diffusion import get_nodes_by_diffusion_status
from rpasdt.algorithm.models import (
    DiffusionModelSimulationConfig,
    DiffusionSimulationConfig,
)
from rpasdt.algorithm.simulation import _simulate_diffusion
from rpasdt.algorithm.source_selection import select_sources_with_params
from rpasdt.algorithm.taxonomies import (
    DiffusionTypeEnum,
    NodeStatusEnum,
    SourceSelectionOptionEnum,
)
from rpasdt.scripts.taxonomies import (
    fallback_sources_number,
    graphs,
    sources_number,
)


def generate_network_propagation():
    header = ["network", "sources", "infected_nodes", "iteration"]

    for graph_function in graphs:
        graph = graph_function()
        print(f"Processing {graph_function.__name__}")
        processed_source_numbers = set()
        filename = f"results/propagations/{graph_function.__name__}.csv"
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()

        for source_number_ratio in sources_number:
            if source_number_ratio > 1:
                source_number = source_number_ratio
            else:
                source_number = max(
                    2, math.ceil(len(graph.nodes) * source_number_ratio)
                )
                if source_number in processed_source_numbers:
                    # max - min / 2
                    source_number = max(
                        2, math.ceil(len(graph.nodes) * fallback_sources_number)
                    )
                processed_source_numbers.add(source_number)

            sources = select_sources_with_params(
                graph=graph,
                number_of_sources=source_number,
                algorithm=SourceSelectionOptionEnum.BETWEENNESS,
            )
            diffusion_config = DiffusionSimulationConfig(
                graph=graph,
                source_nodes=sources,
                iteration_bunch=50,
                number_of_experiments=3,
                diffusion_models=[
                    DiffusionModelSimulationConfig(
                        diffusion_model_type=DiffusionTypeEnum.SIR,
                        diffusion_model_params={"beta": 0.01, "gamma": 0.005},
                    )
                ],
            )

            IG = None
            for simulation in _simulate_diffusion(diffusion_config):
                graph_connected = False

                while not graph_connected:
                    it_data = simulation.diffusion_model.iteration_bunch(5)[-1]

                    infected_nodes = set(
                        get_nodes_by_diffusion_status(
                            simulation.diffusion_model, NodeStatusEnum.INFECTED
                        )
                    )
                    infected_nodes.update(
                        get_nodes_by_diffusion_status(
                            simulation.diffusion_model, NodeStatusEnum.RECOVERED
                        )
                    )
                    IG = simulation.graph.subgraph(infected_nodes).copy()
                    graph_connected = nx.is_connected(IG)
                    coverage = len(IG.nodes) / len(graph.nodes)
                    print(f"{it_data}-{coverage}")
                print("PROPAGATION FINISHED")

                row = [
                    graph_function.__name__,
                    "|".join(
                        map(
                            str,
                            simulation.source_nodes,
                        )
                    ),
                    "|".join(map(str, IG.nodes)),
                    simulation.experiment_number,
                ]
                file = open(filename, "a")
                csvwriter = csv.writer(file)
                csvwriter.writerow(row)
                file.close()


generate_network_propagation()
