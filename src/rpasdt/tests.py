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
from rpasdt.scripts.utils import configure_plot

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


def draw_communities(G, partition, name=""):
    from matplotlib import pyplot as plt

    plt.clf()
    grouped_nodes = get_grouped_nodes(partition)
    if len(G) > 500:
        pos = nx.spring_layout(G, iterations=15, seed=1721)
    else:
        # pos = community_layout(G, grouped_nodes)
        pos = nx.kamada_kawai_layout(G)
    pos = pos or nx.spring_layout(G, seed=100)
    cf = plt.gcf()
    cf.set_facecolor("w")
    ax = cf.gca()
    ax.set_axis_off()
    plt.draw_if_interactive()
    # fig, ax = plt.subplots(figsize=(15, 9))
    # ax.axis("off")
    # nx.draw_networkx(G, pos=pos, ax=ax, **plot_options)
    # draw the graph
    #

    # color the nodes according to their partition
    # cmap = cm.get_cmap("tab20c", len(grouped_nodes.keys()))
    ccolors = get_nodes_color(partition)

    configure_plot()
    nx.draw_networkx_nodes(
        G,
        pos,
        grouped_nodes.keys(),
        node_size=200,
        node_color=[
            ccolors[get_community_index(community)]
            for community in grouped_nodes.values()
        ],
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    com_sizes = get_communities_size(partition)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=f"C{community + 1}, |C{community+ 1}| = {com_sizes[community]}",
            markerfacecolor=ccolors[get_community_index(community)],
            markersize=8,
        )
        for community in range(len(com_sizes))
    ]
    plt.legend(loc="lower right", handles=legend_elements)
    plt.tight_layout(pad=0)
    plt.show()
    # plt.savefig(
    #     f"/home/qtuser/{name}.png", bbox_inches="tight", transparent=True, pad_inches=0
    # )


GRAPHS = {
    # "divided": divided_by_edge_community,
    "florentine_graph": florentine_graph,
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
    pass
    # draw_community_results()

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
G = facebook()
print(datetime.now())
print(find_communities(type=CommunityOptionEnum.NODE_SIMILARITY, graph=G))
print(datetime.now())
print(find_communities(type=CommunityOptionEnum.NODE_SIMILARITY, graph=G))
print(datetime.now())
