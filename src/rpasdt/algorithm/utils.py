from typing import Set

import networkx as nx
import powerlaw
from networkx import Graph


def _neighbors_of_k_hops(
    G: Graph, node: int, neighbors: Set, current_level: int, max_level: int
):
    if current_level > max_level:
        return neighbors
    for nn in nx.neighbors(G, node):
        neighbors.add(nn)
        _neighbors_of_k_hops(
            G=G,
            node=nn,
            neighbors=neighbors,
            current_level=current_level + 1,
            max_level=max_level,
        )
    return neighbors


def neighbors_of_k_hops(G: Graph, node: int, k: int = 1) -> Set:
    return _neighbors_of_k_hops(
        G=G, node=node, neighbors=set(), current_level=1, max_level=k
    )


def nmi(partition1, partition2):
    def _prepare_partition(partition):
        return [
            x[1]
            for x in sorted(
                [
                    (node, nid)
                    for nid, cluster in enumerate(partition)
                    for node in partition[cluster]
                ],
                key=lambda x: x[0],
            )
        ]

    first_partition_c = _prepare_partition(partition1)
    second_partition_c = _prepare_partition(partition2)

    from sklearn.metrics import normalized_mutual_info_score

    return normalized_mutual_info_score(first_partition_c, second_partition_c)


def get_power_law(G):
    deg_dist = sorted([deg for node, deg in G.degree()], reverse=True)
    fit = powerlaw.Fit(deg_dist, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    return alpha, xmin
