import itertools
import math
import operator
import random
from functools import lru_cache, reduce
from typing import List, Set

import matplotlib
import networkx as nx
from networkx import Graph

from rpasdt.algorithm.models import PropagationReconstructionConfig

matplotlib.use("Qt5Agg")

NODE_INFECTION_PROBABILITY_ATTR = "INFECTION_PROBABILITY"
WEIGHT_ATTR = "weight"


def _init_extended_network(G, IG) -> Graph:
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
    return (
        reduce(
            operator.add,
            neighbors_probability,
            0,
        )
        / len(neighbors_probability)
    )


@lru_cache
def _get_shortest_path(G: Graph, source: int, target: int) -> List[int]:
    return nx.shortest_path(G, source=source, target=target)


def _check_if_node_is_on_path_between_infected_nodes(node: int, G: Graph) -> float:
    neighbors = list(nx.neighbors(G, node))
    combination = list(itertools.combinations(neighbors, 2))
    sum = 0.0
    for n1, n2 in combination:
        sp = _get_shortest_path(G, source=n1, target=n2)
        if node in sp:
            sum += 1.0
    value = sum / len(combination) if sum > 0 else 0.0
    return value


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
    m_free = config.m_free
    return 1 / (1 + math.exp(-(m1 + m2 + m3 + m_free)))


def _remove_invalid_edges_and_nodes(EG, threshold):
    edges_to_remove = []
    for edge in EG.edges(data=True):
        data = edge[2]
        weight = data[WEIGHT_ATTR]
        if weight < threshold:
            edges_to_remove.append(edge)
    EG.remove_edges_from(edges_to_remove)
    EG.remove_nodes_from(list(nx.isolates(EG)))


def create_snapshot_IG(G, delete_ratio=None):
    """Create a snapshot of the network by removing a random number of nodes.

    :param G: The network
    :param ratio_to_remove: The ratio of nodes to remove. If None, remove 10%
    of the nodes.
    """
    IG = G.copy()
    if not delete_ratio:
        delete_ratio = 10
    delete_ratio = delete_ratio / 100 if delete_ratio > 1 else delete_ratio
    k = math.ceil(len(G.nodes) * delete_ratio)
    to_remove = list(sorted(random.sample(list(G.nodes), k=k)))
    IG.remove_nodes_from(to_remove)
    return IG, to_remove


def reconstruct_propagation(config: PropagationReconstructionConfig) -> Graph:
    EG = _init_extended_network(G=config.G, IG=config.IG)
    iter = 0
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
