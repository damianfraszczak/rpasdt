"""Community detection methods."""
import math
import sys
from collections import defaultdict
from typing import Dict, List, Union

import networkx as nx
from cdlib import algorithms
from networkx import Graph

from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import (
    community_similarity,
    delete_communities,
    find_small_communities,
    node_similarity,
)
from rpasdt.common.utils import (
    get_enum,
    get_function_default_kwargs,
    get_object_value,
)
from rpasdt.network.networkx_utils import get_grouped_nodes

thismodule = sys.modules[__name__]
NUMBER_OF_COMMUNITIES_KWARG_NAMES = ["k", "number_communities", "level"]


def _update_communities_kwarg(
    type: CommunityOptionEnum, kwargs: Dict, number_communities: int
):
    for name in NUMBER_OF_COMMUNITIES_KWARG_NAMES:
        if name in kwargs:
            kwargs[name] = (
                number_communities - 1
                if type == CommunityOptionEnum.GIRVAN_NEWMAN
                else number_communities
            )


def find_communities(
    type: Union[str, CommunityOptionEnum],
    graph: Graph,
    number_communities: int = 1,
    **alg_kwargs
) -> Dict[int, List[int]]:
    alg_function_name = get_enum(type, CommunityOptionEnum).value
    alg = getattr(algorithms, alg_function_name, None) or getattr(
        thismodule, alg_function_name
    )
    kwargs = {**get_function_default_kwargs(alg), **alg_kwargs, **{"g_original": graph}}
    _update_communities_kwarg(
        type=type, kwargs=kwargs, number_communities=number_communities
    )
    result = alg(**kwargs)
    return {
        index: community
        for index, community in enumerate(get_object_value(result, "communities"))
    }


def merge_communities_based_on_similarity(G, communities):
    def _sorted_communities(c):
        return sorted(c.items(), key=lambda k: len(k[1]), reverse=True)

    communities = {**communities}
    small_communities = find_small_communities(communities=communities)
    changed = True
    while small_communities and changed:
        changed = False
        small_c_number, small_c_nodes = _sorted_communities(small_communities)[0]

        best_community, best_rank = None, -1
        for c_number, c_nodes in communities.items():
            if c_number == small_c_number:
                continue
            cc_sim = community_similarity(G, small_c_nodes, c_nodes)
            if cc_sim > best_rank:
                best_rank = cc_sim
                best_community = c_number
        if best_community:
            communities[best_community].extend(small_c_nodes)
            delete_communities(
                communities=communities,
                communities_to_delete={small_c_number: small_c_nodes},
            )
            changed = True
        small_communities = find_small_communities(communities=communities)

    return communities


def merge_communities_based_on_louvain(G, communities):
    M = nx.quotient_graph(G, communities.values())
    return find_communities(graph=M, type=CommunityOptionEnum.LOUVAIN)


def merge_communities_based_on_modularity(G, communities):
    communities = {**communities}
    small_communities = find_small_communities(communities=communities)

    changed = True
    while changed:
        changed = False
        # dla kazdej malej spolecznosci
        # przypisz ja do wiekszej
        for sc_number, sc_nodes in small_communities.items():

            grouped_nodes = get_grouped_nodes(communities)
            for node in sc_nodes:
                grouped_nodes[node] = sc_number
            max_modularity = -100
            best_community = -1
            # dodac threshold
            for ctm_n, ctm_nodes in grouped_nodes.items():
                pass
        if changed:
            small_communities = find_small_communities(communities)
    return communities


def df_node_similarity(g_original: Graph, **kwargs):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )

    nodes_to_process = [node for node, centrality in sorted_by_degree]

    average_degree = sum(centrality for node, centrality in sorted_by_degree) / len(
        sorted_by_degree
    )

    communities = defaultdict(list)
    similarity_threshold = average_degree

    for node in nodes_to_process:
        if G.nodes[node]["community"]:
            current_community = G.nodes[node]["community"]
        else:
            current_community = max(communities.keys() or [0]) + 1
            G.nodes[node]["community"] = current_community
            communities[current_community].append(node)

        for node_n in G.neighbors(node):
            if G.nodes[node_n]["community"]:
                continue
            similarity = node_similarity(G, node, node_n)
            if similarity >= similarity_threshold:
                G.nodes[node_n]["community"] = current_community
                communities[current_community].append(node_n)
    communities = merge_communities_based_on_similarity(G, communities)
    return {"communities": communities.values()}
