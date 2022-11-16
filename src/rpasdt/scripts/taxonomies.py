import os

import networkx as nx
import numpy as np
from scipy.io import mmread

from rpasdt.algorithm.models import (
    CentralityCommunityBasedSourceDetectionConfig,
    CommunitiesBasedSourceDetectionConfig,
    SourceDetectorSimulationConfig,
)
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    SourceDetectionAlgorithm,
)


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


graphs = [
    karate_graph,
    dolphin,
    footbal,
    # barabasi_1,
    # barabasi_2,
    # watts_strogatz_graph_1,
    # watts_strogatz_graph_2,
    # facebook,
    # soc_anybeat,
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
    # CommunityOptionEnum.SBM_DL,
]
communities = [CommunityOptionEnum.LEIDEN, CommunityOptionEnum.NODE_SIMILARITY]
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
source_detectors.update(
    {
        f"rumor:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.RUMOR_CENTER,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
)
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
#         f"ensemble:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER,
#             config=EnsembleCommunitiesBasedSourceDetectionConfig(
#                 number_of_sources=x,
#                 communities_algorithm=cm,
#                 source_detectors_config={
#                     "BETWEENNESS": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.BETWEENNESS,
#                             communities_algorithm=cm,
#                             source_threshold=SOURCE_THRESHOLD,
#                         ),
#                     ),
#                     # "JORDAN": (SourceDetectionAlgorithm.JORDAN_CENTER,
#                     #            CommunitiesBasedSourceDetectionConfig(
#                     #                number_of_sources=x,
#                     #                communities_algorithm=cm
#                     #            )),
#                     "RUMOR": (
#                         SourceDetectionAlgorithm.RUMOR_CENTER,
#                         CommunitiesBasedSourceDetectionConfig(
#                             number_of_sources=x, communities_algorithm=cm
#                         ),
#                     ),
#                     "NETSLEUTH": (
#                         SourceDetectionAlgorithm.NET_SLEUTH,
#                         CommunitiesBasedSourceDetectionConfig(
#                             number_of_sources=x, communities_algorithm=cm
#                         ),
#                     ),
#                 },
#             ),
#         )
#         for cm in communities
#     }
# )
# source_detectors.update(
#     {
#         f"ensemble-centralities:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
#             alg=SourceDetectionAlgorithm.COMMUNITY_ENSEMBLE_LEARNER,
#             config=EnsembleCommunitiesBasedSourceDetectionConfig(
#                 number_of_sources=x,
#                 communities_algorithm=cm,
#                 source_detectors_config={
#                     "BETWEENNESS": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.BETWEENNESS,
#                             communities_algorithm=cm,
#                             source_threshold=SOURCE_THRESHOLD,
#                         ),
#                     ),
#                     "DEGREE": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.DEGREE,
#                             communities_algorithm=cm,
#                             source_threshold=SOURCE_THRESHOLD,
#                         ),
#                     ),
#                     "CLOSENESS": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.CLOSENESS,
#                             communities_algorithm=cm,
#                             source_threshold=SOURCE_THRESHOLD,
#                         ),
#                     ),
#                     "PAGE_RANK": (
#                         SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
#                         CentralityCommunityBasedSourceDetectionConfig(
#                             number_of_sources=x,
#                             centrality_algorithm=CentralityOptionEnum.PAGE_RANK,
#                             communities_algorithm=cm,
#                             source_threshold=SOURCE_THRESHOLD,
#                         ),
#                     ),
#                 },
#             ),
#         )
#         for cm in communities
#     }
# )
