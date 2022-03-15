from typing import Set

import networkx as nx
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
