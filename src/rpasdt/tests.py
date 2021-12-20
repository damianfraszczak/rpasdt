# do sprawdzenia
# https://orbifold.net/default/community-detection-using-networkx/

import matplotlib
from matplotlib import cm

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import method_time
from rpasdt.network.networkx_utils import get_grouped_nodes

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
    return nx.windmill_graph(4, 4)


@method_time
def df_similarity(G):
    return find_communities(graph=G, type=CommunityOptionEnum.NODE_SIMILARITY)


@method_time
def louvain(G):
    return find_communities(graph=G, type=CommunityOptionEnum.LOUVAIN)


def draw_communities(G, partition):
    from matplotlib import pyplot as plt

    # draw the graph
    pos = nx.spring_layout(G, seed=50)
    grouped_nodes = get_grouped_nodes(partition)
    # color the nodes according to their partition
    cmap = cm.get_cmap("viridis", len(grouped_nodes.keys()))

    nx.draw_networkx_nodes(
        G,
        pos,
        grouped_nodes.keys(),
        node_size=40,
        cmap=cmap,
        node_color=list(grouped_nodes.values()),
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()


G = karate_graph()

# G = nx.from_edgelist([
#     (1, 2),
#     (2, 3),
#     (4, 5),
#     (5, 6),
#     (2, 5)
# ])
# print(df_similarity(G))
comm = df_similarity(G)
print(len(comm.keys()))

print(len(louvain(G).keys()))
# draw_communities(G, df_similarity(G))
# nx.draw(G, with_labels=True)
# from matplotlib import pyplot as plt

# plt.show()
