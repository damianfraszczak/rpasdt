# do sprawdzenia
# https://orbifold.net/default/community-detection-using-networkx/
import os

import matplotlib
import networkx as nx
from matplotlib import cm

from rpasdt.algorithm.communities import df_node_similarity, find_communities
from rpasdt.algorithm.similarity import jaccard_node_similarity
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import get_object_value, get_project_root, method_time
from rpasdt.network.networkx_utils import get_grouped_nodes

matplotlib.use("Qt5Agg")


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


def facebook():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "facebook_combined.txt")
    )


def emailucore():
    return nx.read_adjlist(
        os.path.join(get_project_root(), "data", "community", "emailucore.txt")
    )


def z2014():
    return nx.connected_caveman_graph(4, 3)


def cg():
    return nx.caveman_graph(1000, 10)


def windmil():
    return nx.windmill_graph(4, 4)


def barabasi():
    return nx.barabasi_albert_graph(100, 5)


def random_partition():
    # create a modular graph
    partition_sizes = [10, 20, 30, 40]
    return nx.random_partition_graph(partition_sizes, 0.5, 0.1)


def watts_strogatz_graph():
    return nx.watts_strogatz_graph(n=50, k=8, p=0.5)


@method_time
def df_similarity(G, **kwargs):
    return find_communities(graph=G, type=CommunityOptionEnum.NODE_SIMILARITY, **kwargs)


@method_time
def louvain(G, resolution=1.0):
    return find_communities(
        graph=G, type=CommunityOptionEnum.LOUVAIN, resolution=resolution
    )


def draw_communities(G, partition):
    from matplotlib import pyplot as plt

    grouped_nodes = get_grouped_nodes(partition)
    if len(G) > 500:
        pos = nx.spring_layout(G, iterations=15, seed=1721)
    else:
        # pos = community_layout(G, grouped_nodes)
        pos = nx.kamada_kawai_layout(G)

    # fig, ax = plt.subplots(figsize=(15, 9))
    # ax.axis("off")
    # nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)
    # draw the graph
    #

    # color the nodes according to their partition
    cmap = cm.get_cmap("tab20c", len(grouped_nodes.keys()))

    nx.draw_networkx_nodes(
        G,
        pos,
        grouped_nodes.keys(),
        node_size=200,
        cmap=cmap,
        node_color=list(grouped_nodes.values()),
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.show()


GRAPHS = {
    "divided": divided_by_edge_community,
    # "karate": karate_graph,
    # "windmil": windmil,
    # "football": footbal,
    # "dolphin": dolphin,
    # # "strogats": watts_strogatz_graph,
    # # "barabasi": barabasi,
    # "cg": cg,
    # "radnom_partition": random_partition,
    # "facebook": facebook,
}
similarity_functions = [
    jaccard_node_similarity,
    # sorensen_node_similarity,
    # academic_adar_node_similarity,
    # hub_promoted_index_node_similarity,
    # hub_depressed_index_node_similarity,
    # leicht_holme_node_similarity,
    # resource_allocation_index_node_similarity
]
for G_name in GRAPHS:
    G = GRAPHS[G_name]()
    for sim_f in similarity_functions:
        # comm = df_similarity(G, node_similarity_function=sim_f)
        comm = df_node_similarity(G, node_similarity_function=sim_f)
        # resultat jak z METODY
        comm = {
            index: community
            for index, community in enumerate(get_object_value(comm, "communities"))
        }
        # comm = louvain(G, resolution=2)
        print(
            f"{G_name}-{len(comm.keys())}-{[len(nodes) for nodes in comm.values()]}: {comm}"
        )
        # print(comm)
        draw_communities(G, comm)

# L = louvain(G)
# print(len(L))
# print(L)
# draw_communities(G, L)
# for sim_f in similarity_functions:
#     comm = df_similarity(G, node_similarity_function=sim_f)
#     print(
#         f"{sim_f.__name__}-{len(comm.keys())}-{[len(nodes) for nodes in comm.values()]}: {comm}"
#     )
# print(comm)
# draw_communities(G, comm)

# draw_communities(G, L)
#
# from matplotlib import pyplot as plt
#
# plt.show()
#
# nx.draw(G, with_labels=True)
