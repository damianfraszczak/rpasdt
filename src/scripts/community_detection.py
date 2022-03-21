import csv
import math
import os
import time

import networkx as nx
import numpy as np
import stopit
from scipy.io import mmread

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.diffusion import get_nodes_by_diffusion_status
from rpasdt.algorithm.models import (
    DiffusionModelSimulationConfig,
    DiffusionSimulationConfig,
)
from rpasdt.algorithm.simulation import _simulate_diffusion
from rpasdt.algorithm.source_selection import select_sources_with_params
from rpasdt.algorithm.taxonomies import (
    CommunityOptionEnum,
    DiffusionTypeEnum,
    NodeStatusEnum,
    SourceSelectionOptionEnum,
)
from rpasdt.common.exceptions import log_error


def get_project_root():
    return "../../"


def karate_graph():
    return nx.karate_club_graph()


def footbal():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "football.txt")
    )


def dolphin():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "dolphin.txt")
    )


def club():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "club.txt")
    )


def soc_epinions():
    # return nx.from_scipy_sparse_matrix(sp.io.mmread(fh))
    path = os.path.join(get_project_root(), "data", "community", "socfb-Berkeley13.mtx")
    return nx.Graph(np.matrix(mmread(path).todense()))


def soc_anybeat():
    # return nx.from_scipy_sparse_matrix(sp.io.mmread(fh))
    path = os.path.join(get_project_root(), "data", "community", "soc-anybeat.edges")
    return nx.read_edgelist(path)


def soc_wiki_elec():
    # return nx.from_scipy_sparse_matrix(sp.io.mmread(fh))
    path = os.path.join(get_project_root(), "data", "community", "soc-anybeat.edges")
    return nx.read_edgelist(path)


def facebook():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "facebook_combined.txt")
    )


def emailucore():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "emailucore.txt")
    )


def barabasi_1():
    return nx.barabasi_albert_graph(500, 5)


def barabasi_2():
    return nx.barabasi_albert_graph(1000, 5)


def watts_strogatz_graph_1():
    return nx.watts_strogatz_graph(n=500, k=10, p=0.4)


def watts_strogatz_graph_2():
    return nx.watts_strogatz_graph(n=1000, k=10, p=0.4)


graphs = {
    karate_graph,
    # dolphin,
    # footbal,
    # facebook,
    # barabasi_1,
    # barabasi_2,
    # watts_strogatz_graph_1,
    # watts_strogatz_graph_2,
    # soc_anybeat
}

communities = [
    # CommunityOptionEnum.LOUVAIN,
    # CommunityOptionEnum.BELIEF,
    # CommunityOptionEnum.LOUVAIN,
    # CommunityOptionEnum.LEIDEN,
    # CommunityOptionEnum.LABEL_PROPAGATION,
    # CommunityOptionEnum.GREEDY_MODULARITY,
    # CommunityOptionEnum.EIGENVECTOR,
    # CommunityOptionEnum.GA,
    # CommunityOptionEnum.GEMSEC,
    # CommunityOptionEnum.INFOMAP,
    # CommunityOptionEnum.KCUT,
    # CommunityOptionEnum.MARKOV_CLUSTERING,
    # CommunityOptionEnum.PARIS,
    # CommunityOptionEnum.SPINGLASS,
    # CommunityOptionEnum.SURPRISE_COMMUNITIES,
    # CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.SPECTRAL,
    # CommunityOptionEnum.SBM_DL,
]

results = []
sources_number = [0.001, 0.01, 0.1]
sources_number = [2, 3, 4]
coverages = [0.5, 0.75]
TIMEOUT = 120


def network_stats():
    for graph_function in graphs:
        graph = graph_function()
        data = []
        degree = [value for node, value in graph.degree()]
        data.append(graph_function.__name__)
        data.append(f"N:{len(graph.nodes)}")
        data.append(f"E:{len(graph.edges)}")
        data.append(f"D:{round(nx.density(graph), 4)}")
        data.append(f"A:{round(nx.degree_assortativity_coefficient(graph), 4)}")
        # data.append(f"K-core:{max(nx.core_number(graph).values())}")
        # data.append(f"trianges: {sum(nx.triangles(graph).values())}")
        data.append(f"AC:{round(nx.average_clustering(graph), 4)}")
        data.append(
            f"degree:{min(degree)}/{round(sum(degree) / len(degree), 2)}/{max(degree)}"
        )
        print(f"{':'.join(data)}")


def community_evaluation():
    header = [
        "graph",
        "community",
        "sources",
        "coverage",
        "iterations",
        "detected_sources",
        "sources_ratio",
        "time",
    ]

    for graph_function in graphs:
        graph = graph_function()
        for req_coverage in coverages:
            filename = f"{graph_function.__name__}_{req_coverage}_ce.csv"
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

                sources = select_sources_with_params(
                    graph=graph,
                    number_of_sources=source_number,
                    algorithm=SourceSelectionOptionEnum.BETWEENNESS,
                )
                diffusion_config = DiffusionSimulationConfig(
                    graph=graph,
                    source_nodes=sources,
                    iteration_bunch=1,
                    number_of_experiments=1,
                    diffusion_models=[
                        DiffusionModelSimulationConfig(
                            diffusion_model_type=DiffusionTypeEnum.SIR,
                            diffusion_model_params={"beta": 0.01, "gamma": 0.005},
                        )
                    ],
                )
                graph_connected = False
                IG = None
                simulation = next(_simulate_diffusion(diffusion_config))
                coverage = 0
                iterations = 0
                while not graph_connected or coverage < req_coverage:
                    it_data = simulation.diffusion_model.iteration()
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
                    IG = simulation.graph.subgraph(infected_nodes)
                    graph_connected = nx.is_connected(IG)
                    iterations = it_data["iteration"]
                    coverage = len(IG.nodes) / len(graph.nodes)

                for key in communities:
                    try:
                        with stopit.ThreadingTimeout(TIMEOUT) as context_manager:

                            start = time.time()
                            result = find_communities(type=key, graph=IG)
                            end = time.time()
                            total_time = end - start
                            if context_manager.state == context_manager.TIMED_OUT:
                                continue

                            sources_communities = set()
                            for source_node in sources:
                                for community, nodes in result.items():
                                    if source_node in nodes:
                                        sources_communities.add(community)
                                        break
                            row = [
                                graph_function.__name__,
                                key,
                                len(sources),
                                coverage,
                                iterations,
                                len(result),
                                len(sources_communities),
                                total_time,
                            ]
                            file = open(filename, "a")
                            csvwriter = csv.writer(file)
                            csvwriter.writerow(row)
                            file.close()
                    except Exception as e:
                        log_error(exc=e, show_error_dialog=False)


community_evaluation()
# print(results)
