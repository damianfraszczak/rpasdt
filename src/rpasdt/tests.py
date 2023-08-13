# do sprawdzenia
# https://orbifold.net/default/community-detection-using-networkx/
import os
from datetime import datetime

import matplotlib
import networkx as nx
from matplotlib.lines import Line2D

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.df_community import df_node_similarity
from rpasdt.algorithm.similarity import jaccard_node_similarity
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import get_communities_size
from rpasdt.common.utils import get_object_value, get_project_root, method_time
from rpasdt.network.networkx_utils import (
    get_community_index,
    get_grouped_nodes,
    get_nodes_color,
)
from rpasdt.scripts.utils import configure_plot, draw_communities

matplotlib.use("Qt5Agg")


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


def florentine_graph():
    G = nx.Graph()
    G.add_edge(1, 3)
    G.add_edge(2, 5)
    G.add_edge(2, 7)
    G.add_edge(2, 13)
    G.add_edge(3, 13)
    G.add_edge(3, 8)
    G.add_edge(3, 9)
    G.add_edge(3, 10)
    G.add_edge(3, 4)
    G.add_edge(4, 14)
    G.add_edge(5, 7)
    G.add_edge(5, 11)
    G.add_edge(6, 8)
    G.add_edge(7, 11)
    G.add_edge(8, 9)
    # G.add_edge(9,12)
    G.add_edge(10, 15)
    G.add_edge(10, 12)
    G.add_edge(11, 12)
    G.add_edge(12, 16)
    return G


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


@method_time
def leiden(G, resolution=1.0):
    return find_communities(
        graph=G, type=CommunityOptionEnum.LEIDEN, initial_membership=None
    )


def data_ego_553587013409325058():
    return nx.read_adjlist(
        os.path.join(
            get_project_root(),
            "data",
            "twitter",
            "ego-graph-553587013409325058.adjlist",
        )
    )


GRAPHS = {
    # "divided": divided_by_edge_community,
    # "florentine_graph": florentine_graph,
    "karate": karate_graph,
    # "windmil": windmil,
    # "football": footbal,
    # "dolphin": dolphin,
    # # "strogats": watts_strogatz_graph,
    # # "barabasi": barabasi,
    # "cg": cg,
    # "radnom_partition": random_partition,
    # "facebook": facebook,
    # "twitter": data_ego_553587013409325058
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


def draw_community_results():
    for G_name in GRAPHS:
        G = GRAPHS[G_name]()
        for sim_f in similarity_functions:

            comm = df_node_similarity(G, node_similarity_function=sim_f)
            # resultat jak z METODY
            comm = {
                index: community
                for index, community in enumerate(get_object_value(comm, "communities"))
            }
            draw_communities(G, comm, name=f"df_{G_name}")
            comm = leiden(G)
            print(
                f"{G_name}-{len(comm.keys())}-{[len(nodes) for nodes in comm.values()]}: {comm}"
            )
            # print(comm)
            draw_communities(G, comm, name=f"leiden_{G_name}")


if __name__ == "__main__":
    draw_community_results()

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
# G = facebook()
# print(datetime.now())
# print(find_communities(type=CommunityOptionEnum.NODE_SIMILARITY, graph=G))
# print(datetime.now())
# print(find_communities(type=CommunityOptionEnum.NODE_SIMILARITY, graph=G))
# print(datetime.now())
