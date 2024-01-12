import math
import random

import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from networkx import star_graph

from rpasdt.algorithm.models import PropagationReconstructionConfig
from rpasdt.algorithm.propagation_reconstruction import (
    create_snapshot_IG,
    reconstruct_propagation,
)

matplotlib.use("Qt5Agg")

NODE_INFECTION_PROBABILITY_ATTR = "INFECTION_PROBABILITY"
WEIGHT_ATTR = "weight"


def plt_with_weight(G):
    pos = nx.spring_layout(G, seed=500)  # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def draw_results(G, RealIG, IG, EG, removed, extended):
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].set_title("Propagacja w sieci")
    pos = nx.spring_layout(RealIG, seed=100)
    nx.draw_networkx(
        RealIG,
        pos,
        node_color=["red" if node in [0, 33] else "#1f78b4" for node in G],
        ax=axes[0],
    )
    axes[1].set_title("Zaobserwowana propagacja")
    nx.draw_networkx(
        RealIG,
        pos,
        node_color=["yellow" if node in removed else "#1f78b4" for node in RealIG],
        ax=axes[1],
    )
    axes[2].set_title("Rekonstrukcja propagacji")
    nx.draw_networkx(
        RealIG,
        pos,
        node_color=[
            "green"
            if node in extended
            else ("yellow" if node in removed else "#1f78b4")
            for node in RealIG
        ],
        ax=axes[2],
    )

    plt.tight_layout()
    plt.show()


def main():
    G = nx.karate_club_graph()
    RealIG = G
    IG, removed = create_snapshot_IG(RealIG, delete_ratio=10)
    EG = reconstruct_propagation(
        PropagationReconstructionConfig(G=G, IG=IG, real_IG=RealIG)
    )
    extended = EG.nodes - IG.nodes
    print(f"Removed- {set(removed)}")
    print(f"Extended - {set(extended)}")
    draw_results(G, RealIG, IG, EG, removed, extended)


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
            (6, 3),
            (3, 8),
            (8, 9),
            (7, 5),
            (7, 9),
        ]
    )


def hub_example():
    network = divided_by_edge_community()
    color_map = ["grey" for _ in network]

    color_map[0] = "red"
    color_map[3] = "red"
    color_map[8] = "red"
    color_map[5] = "red"
    color_map[4] = "red"

    color_map[2] = "yellow"
    nx.draw_networkx(network, node_color=color_map, with_labels=True)  # node lables
    plt.title("Wizualizacja wykorzystania informacji dot. bycia mostem")
    plt.tight_layout()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Zainfekowany - I",
            markerfacecolor="red",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Zrekonstruowany - R",
            markerfacecolor="yellow",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Podejrzany - S",
            markerfacecolor="grey",
            markersize=15,
        ),
    ]
    plt.legend(loc="best", handles=legend_elements)
    plt.show()


def star_graph_example():
    network = star_graph(10)
    color_map = ["red" if node == 0 else "grey" for node in network]

    color_map[1] = "yellow"
    color_map[2] = "yellow"
    color_map[3] = "yellow"
    nx.draw_networkx(network, node_color=color_map, with_labels=True)  # node lables
    plt.title("Wizualizacja wykorzystania informacji dot. bycia mostem")
    plt.tight_layout()
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Zainfekowany - I",
            markerfacecolor="red",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Zrekonstruowany - R",
            markerfacecolor="yellow",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Podejrzany - S",
            markerfacecolor="grey",
            markersize=15,
        ),
    ]
    plt.legend(loc="best", handles=legend_elements)
    plt.show()


if __name__ == "__main__":
    main()
