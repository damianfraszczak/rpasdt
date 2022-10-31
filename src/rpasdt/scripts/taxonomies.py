import os

import networkx as nx
import numpy as np
from scipy.io import mmread

from rpasdt.algorithm.taxonomies import CommunityOptionEnum


def get_project_root():
    return "../../../"


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


def foursqaure():
    path = os.path.join(get_project_root(), "data", "community", "soc-FourSquare.edges")
    return nx.read_edgelist(path)


def buzznet():
    path = os.path.join(get_project_root(), "data", "community", "soc-buzznet.edges")
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
    # karate_graph,
    # dolphin,
    # footbal,
    # facebook,
    # barabasi_1,
    # barabasi_2,
    # watts_strogatz_graph_1,
    # watts_strogatz_graph_2,
    soc_anybeat,
}

communities = [
    CommunityOptionEnum.LOUVAIN,
    # CommunityOptionEnum.BELIEF,
    CommunityOptionEnum.LEIDEN,
    CommunityOptionEnum.LABEL_PROPAGATION,
    CommunityOptionEnum.GREEDY_MODULARITY,
    CommunityOptionEnum.EIGENVECTOR,
    # CommunityOptionEnum.GA,
    CommunityOptionEnum.INFOMAP,
    # CommunityOptionEnum.KCUT,
    CommunityOptionEnum.MARKOV_CLUSTERING,
    CommunityOptionEnum.PARIS,
    # CommunityOptionEnum.SPINGLASS,
    CommunityOptionEnum.SURPRISE_COMMUNITIES,
    CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.SPECTRAL,
    # CommunityOptionEnum.SBM_DL,
]
communities = [
    CommunityOptionEnum.NODE_SIMILARITY,
]

sources_number = [0.001, 0.01, 0.1]
# sources_number = [0.1]
fallback_sources_number = 0.05
# sources_number = [2, 3, 4]
