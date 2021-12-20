# do sprawdzenia
# https://orbifold.net/default/community-detection-using-networkx/

import math
from collections import defaultdict

import matplotlib

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import method_time

matplotlib.use("Qt5Agg")
import networkx as nx


def community_similarity_(G, c1, c2):
    c1_n, c2_n = set(), set()
    for node_c1 in c1:
        c1_n.add(node_c1)
        c1_n |= set(G.neighbors(node_c1))
    for node_c2 in c2:
        c1_n.add(node_c2)
        c2_n |= set(G.neighbors(node_c2))
    c1_c2_n_product = c1_n.intersection(c2_n)
    c1_c2_sum = c1_n.union(c2_n)
    return len(c1_c2_n_product) / len(c1_c2_sum)


def community_similarity(G, c1, c2):
    jaccard_coefficients = [node_similarity(G, a, b) for a in c1 for b in c2]
    return sum(jaccard_coefficients) / len(jaccard_coefficients)


def node_similarity(G, a, b):
    union_size = len(set(G[a]) | set(G[b]))
    if union_size == 0:
        return 0
    return len(list(nx.common_neighbors(G, a, b))) / union_size


def karate_graph():
    # 2 communities
    return nx.karate_club_graph()


def divided_by_edge_community():
    # 2 communities
    return nx.from_edgelist(
        [
            (1, 2),
            (1, 3),
            (2, 3),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (8, 12),
            (9, 10),
            (9, 11),
            (10, 11),
            (12, 13),
            (12, 14),
            (13, 14),
        ]
    )


def z2014():
    return nx.connected_caveman_graph(4, 3)


def cg():
    return nx.caveman_graph(50, 3)


def windmil():
    return nx.windmill_graph(50, 4)


def merge_communities(G, communities):
    community_avg_size = math.ceil(
        sum([len(nodes) for community, nodes in communities.items()]) / len(communities)
    )

    small_communities = dict(
        filter(lambda elem: len(elem[1]) < community_avg_size, communities.items())
    )

    for to_delete in small_communities.keys():
        communities.pop(to_delete, None)

    for small_c_number, small_c_nodes in sorted(
        small_communities.items(), key=lambda k: len(k[1]), reverse=True
    ):
        best_community, best_rank = -1, -1
        for c_number, c_nodes in communities.items():
            cc_sim = community_similarity(G, small_c_nodes, c_nodes)
            if cc_sim > best_rank:
                best_rank = cc_sim
                best_community = c_number
        communities[best_community].extend(small_c_nodes)


@method_time
def compute_communities(G):
    G = G.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )

    nodes_to_process = [node for node, centrality in sorted_by_degree]

    average_degree = sum(centrality for node, centrality in sorted_by_degree) / len(
        sorted_by_degree
    )

    communities = defaultdict(list)
    similarity_threshold = average_degree

    for node in nodes_to_process:
        if G.nodes[node]["community"]:
            current_community = G.nodes[node]["community"]
        else:
            current_community = max(communities.keys() or [0]) + 1
            G.nodes[node]["community"] = current_community
            communities[current_community].append(node)

        for node_n in G.neighbors(node):
            if G.nodes[node_n]["community"]:
                continue
            similarity = node_similarity(G, node, node_n)
            if similarity > similarity_threshold:
                G.nodes[node_n]["community"] = current_community
                communities[current_community].append(node_n)
    merge_communities(G, communities)

    # for small_community, nodes in small_communities.items():
    #     for community, c_nodes in communities.items():
    #

    # for community in sorted(small_communities,
    #                         key=lambda k: len(small_communities[k]),
    #                         reverse=True):
    #     pass

    # print(communities)
    # print(small_communities)
    # print(communities)
    # # blackmodel graph
    # # print(communities)
    # M = nx.quotient_graph(G, communities.values())
    # # nx.draw(G, with_labels=True)
    # # nx.draw(M, with_labels=True)
    # from matplotlib import pyplot as plt
    # plt.show()
    # return find_communities(
    #     graph=M,
    #     type=CommunityOptionEnum.LOUVAIN,
    #
    # ),

    return communities


@method_time
def louvain(G):
    return find_communities(graph=G, type=CommunityOptionEnum.LOUVAIN)


G = divided_by_edge_community()

# G = nx.from_edgelist([
#     (1, 2),
#     (2, 3),
#     (4, 5),
#     (5, 6),
#     (2, 5)
# ])
print(len(compute_communities(G)))
print(len(louvain(G).keys()))

# nx.draw(G, with_labels=True)
# from matplotlib import pyplot as plt

# plt.show()
