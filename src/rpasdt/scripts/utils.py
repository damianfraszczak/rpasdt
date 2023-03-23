from typing import Optional

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from rpasdt.algorithm.df_community import (
    get_communities_size,
    get_grouped_nodes,
)
from rpasdt.network.networkx_utils import get_community_index, get_nodes_color

NODE_SIZE = 3000
NODE_COLOR = "#f0f8ff"
NODE_COLOR = "lightgrey"
NODE_LABEL_COLOR = "#000000"
FONT_SIZE = 20
NODE_LABEL_SIZE = FONT_SIZE
AXIS_FONT_SIZE = int(FONT_SIZE * 0.6)
LEGEND_FONT_SIZE = int(FONT_SIZE * 0.4)
FIG_SIZE = (8, 8)


def configure_plot(
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
):
    title = title or plt.gca().get_title()
    ylabel = ylabel or plt.gca().get_ylabel()
    xlabel = xlabel or plt.gca().get_xlabel()

    plt.title(title, fontsize=AXIS_FONT_SIZE)
    plt.xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    plt.ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
    # plt.legend(loc="best", fontsize=LEGEND_FONT_SIZE)

    plt.box(False)
    plt.margins(0.15, 0.15)
    # plt.axis("off")
    fig = plt.gcf()
    ax = fig.axes[0]
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    fig.set_frameon(False)


def get_real_communities(IG, sources) -> dict:
    bfs_map = {}
    communities = {}
    for source in sources:
        bfs_map[source] = list(nx.bfs_tree(IG, source).nodes)
        communities[source] = [source]

    for node in IG.nodes:
        if node in sources:
            continue
        source, mix_distance = -1, 100000
        for key, distances in bfs_map.items():
            distance = distances.index(node)
            if distance < mix_distance:
                mix_distance = distance
                source = key
        communities[source].append(node)

    return {index: communities[source] for index, source in enumerate(communities)}


def get_IG(G, infected_nodes, sources):
    infected_nodes = infected_nodes.split("|")
    sources = sources.split("|")

    IG = G.subgraph(infected_nodes)
    if len(IG.nodes) == 0:
        infected_nodes = [int(x) for x in infected_nodes]
        sources = [int(x) for x in sources]
        IG = G.subgraph(infected_nodes)
    return IG, infected_nodes, sources


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
