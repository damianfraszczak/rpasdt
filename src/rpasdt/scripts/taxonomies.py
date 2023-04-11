import os

import networkx as nx
import numpy as np
from scipy.io import mmread

from rpasdt.algorithm.models import (
    CentralityCommunityBasedSourceDetectionConfig,
    CommunitiesBasedSourceDetectionConfig,
    EnsembleCommunityBasedSourceDetectionConfig,
    SourceDetectorSimulationConfig,
)
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    SourceDetectionAlgorithm,
)

NETWORK_NAMES = {
    "facebook": "Facebook",
    "barabasi_1": "SF-1",
    "barabasi_2": "SF-2",
    "watts_strogatz_graph_1": "SM-1",
    "watts_strogatz_graph_2": "SM-2",
    "soc_anybeat": "Social",
    # "football": "Football",
    "footbal": "Football",
    "karate_graph": "Karate club",
    "dolphin": "Dolphin",
}


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
    filename = os.path.join(get_project_root(), "data", "community", "barabasi_1.txt")
    if not os.path.exists(filename):
        G = nx.barabasi_albert_graph(500, 5)
        nx.write_adjlist(G, filename)
    return nx.read_adjlist(filename)


def barabasi_2():
    filename = os.path.join(get_project_root(), "data", "community", "barabasi_2.txt")
    if not os.path.exists(filename):
        G = nx.barabasi_albert_graph(1000, 5)
        nx.write_adjlist(G, filename)
    return nx.read_adjlist(filename)


def watts_strogatz_graph_1():
    filename = os.path.join(
        get_project_root(), "data", "community", "watts_strogatz_graph_1.txt"
    )
    if not os.path.exists(filename):
        G = nx.watts_strogatz_graph(n=500, k=10, p=0.4)
        nx.write_adjlist(G, filename)
    return nx.read_adjlist(filename)


def watts_strogatz_graph_2():
    filename = os.path.join(
        get_project_root(), "data", "community", "watts_strogatz_graph_2.txt"
    )
    if not os.path.exists(filename):
        G = nx.watts_strogatz_graph(n=1000, k=10, p=0.4)
        nx.write_adjlist(G, filename)
    return nx.read_adjlist(filename)


graphs = [
    # karate_graph,
    # dolphin,
    # footbal,
    # barabasi_1,
    # barabasi_2,
    # watts_strogatz_graph_1,
    # watts_strogatz_graph_2,
    # facebook,
    soc_anybeat,
]

SOURCE_THRESHOLD = None
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
    CommunityOptionEnum.NODE_SIMILARITY,
    # CommunityOptionEnum.SBM_DL,
]
communities = [CommunityOptionEnum.NODE_SIMILARITY]
# LN
# GN
# LV
# CNM
# LP
# IP
# SRC
# WP
METHOD_NAMES = {
    "louvain": "LV",
    "belief": "BF",
    "leiden": "LN",
    "label_propagation": "LP",
    "greedy_modularity": "CNM",
    "eigenvector": "GN",
    "ga": "GA",
    "infomap": "IP",
    "kcut": "Kcut",
    "markov_clustering": "MCL",
    "paris": "PS",
    "spinglass": "SPS",
    "surprise_communities": "SRC",
    "walktrap": "WP",
    "spectral": "SPL",
    "sbm_dl": "SBM",
}

# communites ktore maja sukcesy xd

communities = [
    # CommunityOptionEnum.LEIDEN,
    # CommunityOptionEnum.EIGENVECTOR,
    # CommunityOptionEnum.LOUVAIN,
    # CommunityOptionEnum.GREEDY_MODULARITY,
    # CommunityOptionEnum.LABEL_PROPAGATION,
    # CommunityOptionEnum.INFOMAP,
    # CommunityOptionEnum.SURPRISE_COMMUNITIES,
    # CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.NODE_SIMILARITY
]
sources_number = [0.001, 0.01, 0.1]
# sources_number = [0.1]
fallback_sources_number = 0.05
# sources_number = [2, 3, 4]

sd_centralities = [
    # CentralityOptionEnum.DEGREE,
    CentralityOptionEnum.BETWEENNESS,
    # CentralityOptionEnum.CLOSENESS,
    # CentralityOptionEnum.EDGE_BETWEENNESS,
    # CentralityOptionEnum.EIGENVECTOR
]

source_detectors = {}

# source_detectors.update(
#     {
#         f"centrality:{centrality}": lambda x,
#                                            centrality=centrality: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.CENTRALITY_BASED,
#             config=CentralityBasedSourceDetectionConfig(
#                 number_of_sources=x, centrality_algorithm=centrality
#             ),
#         )
#         for centrality in sd_centralities
#     }
# )
# #
# source_detectors.update(
#     {
#         f"unbiased:{centrality}": lambda x,
#                                          centrality=centrality: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.UNBIASED_CENTRALITY_BASED,
#             config=UnbiasedCentralityBasedSourceDetectionConfig(
#                 number_of_sources=x, centrality_algorithm=centrality
#             ),
#         )
#         for centrality in sd_centralities
#     }
# )
#
source_detectors.update(
    {
        f"centrality-cm:{centrality}:{cm}": lambda x, centrality=centrality, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
            config=CentralityCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                centrality_algorithm=centrality,
                communities_algorithm=cm,
                source_threshold=SOURCE_THRESHOLD,
            ),
        )
        for centrality in sd_centralities
        for cm in communities
    }
)
#
# source_detectors.update(
#     {
#         f"unbiased-cm:{centrality}:{cm}": lambda x, centrality=centrality,
#                                                  cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.COMMUNITY_UNBIASED_CENTRALITY_BASED,
#             config=UnbiasedCentralityCommunityBasedSourceDetectionConfig(
#                 number_of_sources=x,
#                 centrality_algorithm=centrality,
#                 communities_algorithm=cm,
#             ),
#         )
#         for centrality in sd_centralities
#         for cm in communities
#     }
# )
# source_detectors.update(
#     {
#         f"rumor:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.RUMOR_CENTER,
#             config=CommunitiesBasedSourceDetectionConfig(
#                 number_of_sources=x, communities_algorithm=cm
#             ),
#         )
#         for cm in communities
#     }
# )
source_detectors.update(
    {
        f"jordan:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.JORDAN_CENTER,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
)

source_detectors.update(
    {
        f"netsleuth-cm:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.NET_SLEUTH,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
)
# source_detectors.update(
#     {
#         f"dynamicage-cm:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.DYNAMIC_AGE,
#             config=CommunitiesBasedSourceDetectionConfig(
#                 number_of_sources=x, communities_algorithm=cm
#             ),
#         )
#         for cm in communities
#     }
# )


source_detectors.update(
    {
        f"ensemble:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER,
            config=EnsembleCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                communities_algorithm=cm,
                source_detectors_config={
                    "JORDAN": (
                        SourceDetectionAlgorithm.JORDAN_CENTER,
                        CommunitiesBasedSourceDetectionConfig(
                            number_of_sources=x, communities_algorithm=cm
                        ),
                    ),
                    "NETSLEUTH": (
                        SourceDetectionAlgorithm.NET_SLEUTH,
                        CommunitiesBasedSourceDetectionConfig(
                            number_of_sources=x, communities_algorithm=cm
                        ),
                    ),
                },
            ),
        )
        for cm in communities
    }
)
source_detectors.update(
    {
        f"ensemble-centralities:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER,
            config=EnsembleCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                communities_algorithm=cm,
                source_detectors_config={
                    "DEGREE": (
                        SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
                        CentralityCommunityBasedSourceDetectionConfig(
                            number_of_sources=x,
                            centrality_algorithm=CentralityOptionEnum.DEGREE,
                            communities_algorithm=cm,
                            source_threshold=SOURCE_THRESHOLD,
                        ),
                    ),
                    "BETWEENNESS": (
                        SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
                        CentralityCommunityBasedSourceDetectionConfig(
                            number_of_sources=x,
                            centrality_algorithm=CentralityOptionEnum.BETWEENNESS,
                            communities_algorithm=cm,
                            source_threshold=SOURCE_THRESHOLD,
                        ),
                    ),
                },
            ),
        )
        for cm in communities
    }
)
