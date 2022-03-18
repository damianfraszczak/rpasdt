import os
import time

import networkx as nx
import numpy as np
from scipy.io import mmread

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.taxonomies import CommunityOptionEnum


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


def barabasi():
    return nx.barabasi_albert_graph(30, 4)


def watts_strogatz_graph():
    return nx.watts_strogatz_graph(n=1000, k=8, p=0.04)


graphs = {
    # karate_graph,
    # dolphin,
    # footbal,
    # facebook,
    # barabasi,
    # watts_strogatz_graph,
    # soc_anybeat
}

communities = [
    CommunityOptionEnum.LOUVAIN,
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
    # CommunityOptionEnum.SPECTRAL,
    # CommunityOptionEnum.SBM_DL,
]

results = []
sources = [2, 4, 6, 8, 10]


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
    for graph_function in graphs:
        graph = graph_function()
        for source_number in sources:
            pass
        for key in communities:
            try:
                start = time.time()
                result = find_communities(type=key, graph=graph)
                end = time.time()
                total_time = end - start
                results.append(f"{key}:{len(result)}:{total_time}")
            except Exception as e:
                print(e)


network_stats()
# print(results)
