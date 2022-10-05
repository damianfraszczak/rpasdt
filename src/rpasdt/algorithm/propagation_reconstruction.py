import math
import operator
import random
from functools import lru_cache, reduce
from typing import List, Set

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from networkx import Graph, star_graph

from rpasdt.algorithm.models import PropagationReconstructionConfig

matplotlib.use("Qt5Agg")

NODE_INFECTION_PROBABILITY_ATTR = "INFECTION_PROBABILITY"
WEIGHT_ATTR = "weight"


def _init_extended_network(config: PropagationReconstructionConfig) -> Graph:
    G, IG = config.G, config.IG
    EG = G.copy()
    nx.set_node_attributes(
        EG,
        {
            node: {NODE_INFECTION_PROBABILITY_ATTR: 1.0 if node in IG else 0.0}
            for node in G
        },
    )
    return EG


def _compute_edges_weights(EG: Graph):
    for edge in EG.edges(data=False):
        node_1, node_2 = edge[0], edge[1]
        new_weight = (
            EG.nodes[node_1][NODE_INFECTION_PROBABILITY_ATTR]
            * EG.nodes[node_2][NODE_INFECTION_PROBABILITY_ATTR]
        )
        EG[node_1][node_2][WEIGHT_ATTR] = new_weight


def _get_nodes_to_process(EG: Graph, threshold: float) -> List[int]:
    nodes = [
        node
        for node, data in EG.nodes(data=True)
        if data[NODE_INFECTION_PROBABILITY_ATTR] < threshold
    ]

    return list(
        sorted(
            nodes,
            key=lambda node: len(
                [
                    nn
                    for nn in nx.neighbors(EG, node)
                    if EG.nodes[node][NODE_INFECTION_PROBABILITY_ATTR] > threshold
                ]
            ),
            reverse=True,
        )
    )


def _check_node_in_external_network(node: int, infected_nodes: Set[int]) -> bool:
    """
    Return True if node is detected to send a rumor in other site
    False otherwise.

    """
    if random.random():
        return node in infected_nodes
    return False


def _compute_neighbors_probability(node: int, G: Graph) -> float:
    neighbors_probability = [
        G.nodes[node][NODE_INFECTION_PROBABILITY_ATTR] for node in nx.neighbors(G, node)
    ]
    return reduce(
        operator.mul,
        neighbors_probability,
        1,
    )


@lru_cache
def _get_shortest_path(G: Graph, source: int, target: int) -> List[int]:
    return nx.shortest_path(G, source=source, target=target)


def _check_if_node_is_on_path_between_infected_nodes(node: int, G: Graph) -> bool:
    neighbors = nx.neighbors(G, node)
    for n1 in neighbors:
        for n2 in neighbors:
            if n1 == n2:
                continue
            sp = _get_shortest_path(G, source=n1, target=n2)
            if node in sp:
                return True
    return False


def _compute_node_recovery(
    EG: Graph, node: int, config: PropagationReconstructionConfig
) -> float:
    neighbors_probability = _compute_neighbors_probability(node=node, G=EG)
    node_on_path = int(
        _check_if_node_is_on_path_between_infected_nodes(node=node, G=EG)
    )
    node_in_external_network = int(
        _check_node_in_external_network(
            node=node, infected_nodes=config.real_infected_nodes
        )
    )

    m1 = config.m1 * neighbors_probability
    m2 = config.m2 * node_on_path
    m3 = config.m3 * node_in_external_network
    return m1 + m2 + m3


def _remove_invalid_edges_and_nodes(EG, threshold):
    edges_to_remove = []
    for edge in EG.edges(data=True):
        data = edge[2]
        weight = data[WEIGHT_ATTR]
        if weight < threshold:
            edges_to_remove.append(edge)
    EG.remove_edges_from(edges_to_remove)
    EG.remove_nodes_from(list(nx.isolates(EG)))


def reconstruct_propagation(config: PropagationReconstructionConfig) -> Graph:
    EG = _init_extended_network(config)
    iter = 1
    nodes = [1]
    while iter < config.max_iterations and nodes:
        iter += 1
        nodes = _get_nodes_to_process(EG, config.threshold)
        for node in nodes:
            EG.nodes[node][NODE_INFECTION_PROBABILITY_ATTR] = _compute_node_recovery(
                EG=EG, node=node, config=config
            )
    _compute_edges_weights(EG)
    _remove_invalid_edges_and_nodes(EG, config.threshold)
    return EG


def plt_with_weight(G):
    pos = nx.spring_layout(G, seed=500)  # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def create_sample_IG(G, number_to_remove=None):
    IG = G.copy()
    if not number_to_remove:
        number_to_remove = math.ceil(len(G.nodes) / 10)
    to_remove = random.choices(list(G.nodes), k=number_to_remove)
    IG.remove_nodes_from(to_remove)
    return IG, to_remove


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
    IG, removed = create_sample_IG(RealIG, number_to_remove=10)
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
