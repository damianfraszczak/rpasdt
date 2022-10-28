# ground true
# https://github.com/vlivashkin/community-graphs
# a
import os
from collections import defaultdict

# 'karate_club', 'dblp', 'amazon', 'youtube'
from cdlib import datasets
from networkx import Graph

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import nmi
from rpasdt.common.utils import get_object_value, get_project_root
from rpasdt.tests import df_similarity, karate_graph

graph_name_list = datasets.available_ground_truths()


def read_clusters_from_file(path):
    communities = defaultdict(list)
    with open(path) as file:
        for line in file:
            node, cluster = line.split("	")
            communities[int(cluster.strip())].append(node.strip())
    return communities


def map_communities(result):
    return {
        index: community
        for index, community in enumerate(get_object_value(result, "communities"))
    }


def karate_gt() -> dict:
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="karate_club",
        )
    )


def dolphin_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community", "dolphins.clusters.txt")
    )


def football_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community", "football.clusters.txt")
    )


def eu_core_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community", "eu-core.clusters.txt")
    )


def polblogs_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community", "polblogs.clusters.txt")
    )


def amazon_gt() -> dict:
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="amazon_club",
        )
    )


def amazon() -> Graph:
    return datasets.fetch_network_data(net_name="amazon_club", net_type="networkx")


def youtube_gt() -> dict:
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="youtube",
        )
    )


def youtube() -> Graph:
    return datasets.fetch_network_data(net_name="youtube", net_type="networkx")


def dblp_gt():
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="dblp",
        )
    )


def dblp() -> Graph:
    return datasets.fetch_network_data(net_name="dblp", net_type="networkx")


NETWORKS = [karate_graph]
GT_CLUSTERS = [karate_gt]
METHODS = [
    CommunityOptionEnum.LOUVAIN,
    CommunityOptionEnum.BELIEF,
    CommunityOptionEnum.LEIDEN,
    CommunityOptionEnum.LABEL_PROPAGATION,
    CommunityOptionEnum.GREEDY_MODULARITY,
    CommunityOptionEnum.EIGENVECTOR,
    CommunityOptionEnum.GA,
    CommunityOptionEnum.INFOMAP,
    CommunityOptionEnum.KCUT,
    CommunityOptionEnum.MARKOV_CLUSTERING,
    CommunityOptionEnum.PARIS,
    CommunityOptionEnum.SPINGLASS,
    CommunityOptionEnum.SURPRISE_COMMUNITIES,
    CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.SPECTRAL,
    CommunityOptionEnum.SBM_DL,
    CommunityOptionEnum.NODE_SIMILARITY,
]
for index, gf in enumerate(NETWORKS):
    g = gf()
    gt = GT_CLUSTERS[index]()
    n = len(g)
    e = len(g.edges)
    c = len(gt.keys())
    for m in METHODS:
        communities = find_communities(type=m, graph=g)
        nmi = nmi(communities, gt)
        print(f"{m} - {nmi}")
