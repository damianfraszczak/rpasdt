# ground true
# https://github.com/vlivashkin/community-graphs
# a
import csv
import os
from collections import defaultdict
from time import process_time

import networkx as nx
import networkx.algorithms.community as nx_comm

# 'karate_club', 'dblp', 'amazon', 'youtube'
from cdlib import datasets
from networkx import Graph

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import nmi
from rpasdt.common.utils import get_object_value, get_project_root
from rpasdt.scripts.community_detection import cmodularity


def read_clusters_from_file(path):
    communities = defaultdict(list)
    with open(path) as file:
        for line in file:
            node, cluster = line.split("	")
            communities[int(cluster.strip()) - 1].append(node.strip())
    return communities


def map_communities(result):
    return {
        index: community
        for index, community in
        enumerate(get_object_value(result, "communities"))
    }


def karate_gt() -> dict:
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="karate_club",
        )
    )


def dolphin_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community",
                     "dolphins.clusters.txt")
    )


def football_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community",
                     "football.clusters.txt")
    )


def eu_core():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community",
                     "eu-core.edges.txt")
    )


def eu_core_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community",
                     "eu-core.clusters.txt")
    )


def polbooks():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community",
                     "polbooks.edges.txt")
    )


def polbooks_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community",
                     "polbooks.clusters.txt")
    )


def polblogs():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community",
                     "polblogs.edges.txt")
    )


def polblogs_gt():
    return read_clusters_from_file(
        os.path.join(get_project_root(), "data", "community",
                     "polblogs.clusters.txt")
    )


def sp_data():
    return nx.read_gml(
        os.path.join(get_project_root(), "data", "community",
                     "sp_school_day_1.gml")
    )


def sp_data_gt():
    G = sp_data()
    return {node[0]: node[1]["gt"] for node in G.nodes(data=True)}


def amazon_gt() -> dict:
    return map_communities(
        datasets.fetch_ground_truth_data(
            net_name="amazon_club",
        )
    )


def amazon() -> Graph:
    return datasets.fetch_network_data(net_name="amazon_club",
                                       net_type="networkx")


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


NETWORKS = [
    # karate_graph,
    # footbal,
    # dolphin,
    # amazon,
    # youtube,
    # dblp,
    # eu_core,
    # polblogs,
    # polbooks,
    sp_data
]
GT_CLUSTERS = [
    # karate_gt,
    # football_gt,
    # dolphin_gt,
    # amazon_gt,
    # youtube_gt,
    # dblp_gt,
    # eu_core_gt,
    # polblogs_gt,
    # polbooks_gt,
    sp_data_gt
]
METHODS = [
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
    # CommunityOptionEnum.PARIS,
    # CommunityOptionEnum.SPINGLASS,
    CommunityOptionEnum.SURPRISE_COMMUNITIES,
    CommunityOptionEnum.WALKTRAP,
    # CommunityOptionEnum.SPECTRAL,
    # CommunityOptionEnum.SBM_DL,
    CommunityOptionEnum.NODE_SIMILARITY,
]

# METHODS = [CommunityOptionEnum.NODE_SIMILARITY]
header = [
    "graph",
    "method",
    "n",
    "e",
    "real communities",
    "found communities",
    "time",
    "m_nmi" "modularity",
    "pcov",
    "pper",
]


def analysis():
    for index, gf in enumerate(NETWORKS):
        g = gf()
        gt = GT_CLUSTERS[index]()
        n = len(g)
        e = len(g.edges)
        c = len(gt.keys())

        filename = f"results/communities/{gf.__name__}_communities_gt.csv"
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()

        for m in METHODS:
            print(m)
            start = int(round(process_time() * 1000))
            communities = find_communities(type=m, graph=g)
            end_ = int(round(process_time() * 1000)) - start

            m_nmi = -1
            try:
                m_nmi = nmi(communities, gt)
            except Exception as ex:
                print(f"NMI ERROR {ex}")
            modularity = cmodularity(g, communities.values())
            pcov, pper = nx_comm.partition_quality(g, communities.values())
            row = [
                gf.__name__,
                m,
                n,
                e,
                c,
                len(communities.keys()),
                end_,
                m_nmi,
                modularity,
                pcov,
                pper,
            ]
            file = open(filename, "a")
            csvwriter = csv.writer(file)
            csvwriter.writerow(row)
            file.close()


analysis()
