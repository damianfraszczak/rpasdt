import csv
import math
import time

import matplotlib
import networkx as nx
import networkx.algorithms.community as nx_comm
import stopit
from networkx import convert_node_labels_to_integers

from rpasdt.algorithm.communities import find_communities
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
from rpasdt.algorithm.utils import nmi
from rpasdt.common.exceptions import log_error
from rpasdt.scripts.taxonomies import (
    communities,
    fallback_sources_number,
    graphs,
    sources_number,
)

matplotlib.use("Qt5Agg")


def cmodularity(G, communities, weight="weight", resolution=1):
    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m ** 2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum ** 2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm) if directed else out_degree_sum

        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, communities))


results = []

coverages = [0.5, 0.75]
TIMEOUT = 60 * 2


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


def get_real_communities(IG, sources) -> dict:
    bfs_map = {}
    communities = {}
    for source in sources:
        bfs_map[source] = list(nx.bfs_tree(IG, source).nodes)
        communities[source] = [source]

    for node in IG.nodes:
        if node in sources:
            continue
        source, mix_distance = -1, 100000
        for key, distances in bfs_map.items():
            distance = distances.index(node)
            if distance < mix_distance:
                mix_distance = distance
                source = key
        communities[source].append(node)

    return {index: communities[source] for index, source in enumerate(communities)}


def sd_communities():
    header = [
        "graph",
        "community",
        "sources",
        "nodes",
        "detected",
        "difference",
        "sources_ratio",
        "nmi",
        "modularity",
        "pcov",
        "pper",
    ]

    for graph_function in graphs:
        graph = graph_function()

        processed_source_numbers = set()
        filename = f"results/{graph_function.__name__}_sdc.csv"
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
            while not graph_connected:
                it_data = simulation.diffusion_model.iteration_bunch(2)[-1]

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
                # iterations = it_data["iteration"]
                coverage = len(IG.nodes) / len(graph.nodes)
                print(f"{it_data}-{coverage}")
            print("PROPAGATION FINISHED")
            real_communities = get_real_communities(IG, sources)
            for key in communities:
                try:
                    print(f"COM DETECTION {key}")
                    with stopit.ThreadingTimeout(TIMEOUT):
                        # zmienic labelki na id
                        N = IG.number_of_nodes()
                        mapping = dict(zip(IG.nodes(), range(0, N)))
                        sources = [mapping[source] for source in sources]

                        IG = convert_node_labels_to_integers(IG)
                        found_communities = find_communities(type=key, graph=IG)

                        nmin = nmi(real_communities, found_communities)
                        modularity = cmodularity(IG, found_communities.values())
                        pcov, pper = nx_comm.partition_quality(
                            IG, found_communities.values()
                        )
                        sources_communities = set()
                        for source_node in sources:
                            for community, nodes in found_communities.items():
                                if source_node in nodes:
                                    sources_communities.add(community)
                                    break

                        row = [
                            graph_function.__name__,
                            key,
                            len(sources),
                            len(IG.nodes),
                            len(found_communities.keys()),
                            len(sources) - len(found_communities.keys()),
                            len(sources_communities),
                            nmin,
                            modularity,
                            pcov,
                            pper,
                        ]
                        file = open(filename, "a")
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                        file.close()
                except Exception as e:
                    log_error(exc=e, show_error_dialog=False)
                    file = open(filename, "a")
                    csvwriter = csv.writer(file)
                    row = [
                        graph_function.__name__,
                        key,
                    ]
                    csvwriter.writerow(row)
                    file.close()


def community_evaluation2():
    header = [
        "graph",
        "community",
        "sources",
        "nodes",
        "coverage",
        "iterations",
        "detected_sources",
        "sources_ratio",
        "sizes",
        "avg_sizes",
        "time",
    ]

    for graph_function in graphs:
        graph = graph_function()

        processed_source_numbers = set()
        filename = f"results/{graph_function.__name__}_ce2.csv"
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
            while not graph_connected:
                it_data = simulation.diffusion_model.iteration_bunch(2)[-1]

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
                iterations = it_data["iteration"]
                coverage = len(IG.nodes) / len(graph.nodes)
                print(f"{it_data}-{coverage}")
            print("PROPAGATION FINISHED")
            for key in communities:
                try:
                    print(f"COM DETECTION {key}")
                    with stopit.ThreadingTimeout(TIMEOUT) as context_manager:

                        start = time.time()
                        result = find_communities(type=key, graph=IG)
                        end = time.time()
                        total_time = end - start
                        if context_manager.state == context_manager.TIMED_OUT:
                            continue

                        sources_communities = set()
                        if result:
                            for source_node in sources:
                                for community, nodes in result.items():
                                    if source_node in nodes:
                                        sources_communities.add(community)
                                        break
                        sizes = [len(c) for c in result.values()]
                        avg_size = sum(sizes) / len(sizes)

                        row = [
                            graph_function.__name__,
                            key,
                            len(sources),
                            len(IG.nodes),
                            coverage,
                            iterations,
                            len(result) if result else None,
                            len(sources_communities),
                            "|".join([str(c) for c in sizes]),
                            str(avg_size),
                            total_time,
                        ]
                        file = open(filename, "a")
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                        file.close()
                except Exception as e:
                    log_error(exc=e, show_error_dialog=False)


def community_evaluation():
    header = [
        "graph",
        "community",
        "sources",
        "coverage",
        "iterations",
        "detected_sources",
        "sources_ratio",
        "sizes",
        "avg_sizes",
        "time",
    ]

    for graph_function in graphs:
        graph = graph_function()
        for req_coverage in coverages:
            processed_source_numbers = set()
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
                    iteration_bunch=5,
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
                    it_data = simulation.diffusion_model.iteration_bunch(2)[-1]

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
                    iterations = it_data["iteration"]
                    coverage = len(IG.nodes) / len(graph.nodes)
                    print(f"{it_data}-{coverage}")
                print("PROPAGATION FINISHED")
                for key in communities:
                    try:
                        print(f"COM DETECTION {key}")
                        with stopit.ThreadingTimeout(TIMEOUT) as context_manager:

                            start = time.time()
                            result = find_communities(type=key, graph=IG)
                            end = time.time()
                            total_time = end - start
                            if context_manager.state == context_manager.TIMED_OUT:
                                continue

                            sources_communities = set()
                            if result:
                                for source_node in sources:
                                    for community, nodes in result.items():
                                        if source_node in nodes:
                                            sources_communities.add(community)
                                            break
                            sizes = [len(c) for c in result.values()]
                            avg_size = sum(sizes) / len(sizes)

                            row = [
                                graph_function.__name__,
                                key,
                                len(sources),
                                coverage,
                                iterations,
                                len(result) if result else None,
                                len(sources_communities),
                                "|".join([str(c) for c in sizes]),
                                str(avg_size),
                                total_time,
                            ]
                            file = open(filename, "a")
                            csvwriter = csv.writer(file)
                            csvwriter.writerow(row)
                            file.close()
                    except Exception as e:
                        log_error(exc=e, show_error_dialog=False)


# community_evaluation2()
# G = karate_graph()
# com = get_real_communities(G, [0, 33])
# found_communities = find_communities(type=CommunityOptionEnum.LOUVAIN,
#                                      graph=G)
# print(nx_comm.modularity(G, com.values()))
# print(nx_comm.modularity(G, found_communities.values()))
#
# print(nmi(found_communities, com))
#
# per_cov = nx_comm.partition_quality(G, found_communities.values())
# print(per_cov)

# sd_communities()
