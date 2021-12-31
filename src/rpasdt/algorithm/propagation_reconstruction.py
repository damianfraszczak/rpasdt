import math
import operator
import random
from functools import reduce, lru_cache
from typing import List, Set

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


def _check_node_in_external_network(node: int,
                                    infected_nodes: Set[int]) -> bool:
    """
    Return True if node is detected to send a rumor in other site
    False otherwise.

    """
    if random.random():
        return node in infected_nodes
    return False


def _compute_neighbors_probability(node: int, G: Graph) -> float:
    neighbors_probability = [
        G.nodes[node][NODE_INFECTION_PROBABILITY_ATTR]
        for node in nx.neighbors(G, node)
    ]
    return reduce(
        operator.mul, neighbors_probability,
        1,
    )


@lru_cache
def _get_shortest_path(G: Graph, source: int, target: int) -> List[int]:
    return nx.shortest_path(G, source=source, target=target)


def _check_if_node_is_on_path_between_infected_nodes(node: int,
                                                     G: Graph) -> bool:
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
        _check_if_node_is_on_path_between_infected_nodes(node=node, G=EG))
    node_in_external_network = int(_check_node_in_external_network(
        node=node, infected_nodes=config.real_infected_nodes
    ))

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

    nodes = _get_nodes_to_process(EG, config.threshold)
    for node in nodes:
        EG.nodes[node][
            NODE_INFECTION_PROBABILITY_ATTR] = _compute_node_recovery(EG=EG,
                                                                      node=node,
                                                                      config=config)
    _compute_edges_weights(EG)
    _remove_invalid_edges_and_nodes(EG, config.threshold)
    return EG


def plt_with_weight(G):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Qt5Agg")
    pos = nx.spring_layout(G,
                           seed=500)  # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


def create_sample_IG(G, number_to_remove=None):
    IG = G.copy()
    if not number_to_remove:
        number_to_remove = math.ceil(len(G.nodes) / 10)
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
