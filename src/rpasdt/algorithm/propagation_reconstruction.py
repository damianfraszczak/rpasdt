import operator
import random
from functools import reduce
from typing import List

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.models import PropagationReconstructionConfig

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
    return [
        node
        for node, data in EG.nodes(data=True)
        if data[NODE_INFECTION_PROBABILITY_ATTR] < threshold
    ]


def _check_node_in_external_network(node: int, infected_nodes: List[int]) -> bool:
    """
    Return True if node is detected to send a rumor in other site
    False otherwise.

    """
    if random.random():
        return node in infected_nodes
    return False


def _compute_node_recovery(
    EG: Graph, node: int, config: PropagationReconstructionConfig
) -> float:
    m1 = config.m1 * reduce(
        operator.mul,
        [
            EG.nodes[node][NODE_INFECTION_PROBABILITY_ATTR]
            for node in nx.neighbors(EG, node)
        ],
        1,
    )
    m2 = config.m2
    m3 = config.m3 * _check_node_in_external_network(
        node=node, infected_nodes=config.real_infected_nodes
    )
    return m1 + m2 + m3


def reconstruct_propagation(config: PropagationReconstructionConfig) -> Graph:
    EG = _init_extended_network(config)

    nodes = _get_nodes_to_process(EG, config.threshold)
    current_iteration = 1
    changed = True
    while nodes and current_iteration < config.max_iterations and changed:
        current_iteration += 1

    _compute_edges_weights(EG)
    return EG


def plt_with_weight(G):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Qt5Agg")
    pos = nx.spring_layout(G, seed=500)  # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def create_sample_IG(G, number_to_remove=10):
    IG = G.copy()
    IG.remove_nodes_from(random.choices(list(G.nodes), k=number_to_remove))
    return IG


def main():
    G = nx.karate_club_graph()
    RealIG = G
    IG = create_sample_IG(RealIG)
    EG = reconstruct_propagation(
        PropagationReconstructionConfig(G=G, IG=IG, real_IG=RealIG)
    )
    plt_with_weight(EG)


if __name__ == "__main__":
    main()
