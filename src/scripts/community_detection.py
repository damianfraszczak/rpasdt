import time

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.graph_loader import load_graph
from rpasdt.algorithm.taxonomies import CommunityOptionEnum, GraphTypeEnum

graphs = {GraphTypeEnum.KARATE_CLUB: {}}

communities = [
    CommunityOptionEnum.BELIEF,
    CommunityOptionEnum.LOUVAIN,
    CommunityOptionEnum.LEIDEN,
    CommunityOptionEnum.LABEL_PROPAGATION,
    CommunityOptionEnum.GREEDY_MODULARITY,
    CommunityOptionEnum.EIGENVECTOR,
    CommunityOptionEnum.GA,
    CommunityOptionEnum.GEMSEC,
    CommunityOptionEnum.INFOMAP,
    CommunityOptionEnum.KCUT,
    CommunityOptionEnum.MARKOV_CLUSTERING,
    CommunityOptionEnum.PARIS,
    CommunityOptionEnum.SPINGLASS,
    CommunityOptionEnum.SURPRISE_COMMUNITIES,
    CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.SPECTRAL,
    CommunityOptionEnum.SBM_DL,
]


results = []


def community_evaluation():
    for graph_type, properties in graphs.items():

        graph = load_graph(graph_type=graph_type, graph_type_properties=properties)
        for key in communities:
            try:
                start = time.time()
                result = find_communities(type=key, graph=graph)
                end = time.time()
                total_time = end - start
                results.append(f"{key}:{len(result)}:{total_time}")
            except Exception as e:
                print(e)


community_evaluation()
print(results)
