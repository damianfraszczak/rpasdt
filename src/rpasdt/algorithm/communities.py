"""Community detection methods."""
import sys
from collections import defaultdict
from typing import Dict, List, Union, Optional

import networkx as nx
from cdlib import algorithms
from networkx import Graph

from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import (
    community_similarity,
    delete_communities,
    find_small_communities,
    modularity,
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
    kwargs = {**get_function_default_kwargs(alg), **alg_kwargs,
              **{"g_original": graph}}
    _update_communities_kwarg(
        type=type, kwargs=kwargs, number_communities=number_communities
    )
    result = alg(**kwargs)
    return {
        index: community
        for index, community in
        enumerate(get_object_value(result, "communities"))
    }


def merge_communities_based_on_similarity(G: Graph, communities,
                                          similarity_threshold,
                                          resolution=0.5):
    def _sorted_communities(c):
        return sorted(c.items(), key=lambda k: len(k[1]), reverse=True)

    communities = {**communities}
    small_communities = find_small_communities(communities=communities)
    changed = True
    while small_communities and changed:
        changed = False
        for small_c_number, small_c_nodes in list(small_communities.items()):
            best_community, best_community_small, best_rank = None, None, 0

            # similarities = defaultdict(list)
            # for c_number, c_nodes in communities.items():
            #     if c_number != small_c_number:
            #         similarities[community_similarity(G, small_c_nodes, c_nodes)].append(c_number)
            # max_similarity = max(similarities.keys())
            # max_similarity_communities = similarities.get(max_similarity) or []
            #
            # communities[small_c_number] = []
            # communities[small_c_number].extend(small_c_nodes)
            # for community_to_join in max_similarity_communities:
            #     communities[small_c_number].extend(communities[community_to_join])
            #     delete_communities(
            #         communities=communities,
            #         communities_to_delete={
            #             community_to_join: communities[community_to_join]
            #         },
            #     )
            #     small_communities.pop(small_c_number, None)
            #     changed = True
            # print(f"{max_similarity}-{similarities[max_similarity]}")

            for c_number, c_nodes in communities.items():
                if c_number == small_c_number:
                    continue
                cc_sim = community_similarity(G, small_c_nodes, c_nodes)
                if cc_sim > best_rank and cc_sim > similarity_threshold:
                    best_rank = cc_sim
                    best_community = c_number
                    best_community_small = small_c_number

            if best_community:
                communities[best_community].extend(
                    small_communities[best_community_small]
                )
                delete_communities(
                    communities=communities,
                    communities_to_delete={
                        best_community_small: communities[best_community_small]
                    },
                )
                small_communities.pop(best_community_small)
                changed = True
        small_communities = find_small_communities(
            communities=communities, resolution=resolution
        )

    if small_communities:
        communities = merge_communities_based_on_modularity(G, communities,
                                                            resolution)
    return communities


def merge_communities_based_on_louvain(G, communities, **kwargs):
    M = nx.quotient_graph(G, communities.values())
    return find_communities(graph=M, type=CommunityOptionEnum.LOUVAIN)


def merge_communities_based_on_modularity(G, communities, resolution):
    communities = {**communities}
    small_communities = find_small_communities(communities=communities,
                                               resolution=resolution)

    changed = True
    while small_communities and changed:
        print(communities)
        print(small_communities)
        changed = False
        for small_c_number, small_c_nodes in list(small_communities.items()):
            best_community, best_community_small, best_rank = None, None, -1
            for c_number, c_nodes in communities.items():
                if c_number == small_c_number:
                    continue
                grouped_nodes = get_grouped_nodes(communities)
                for node in small_c_nodes:
                    grouped_nodes[node] = c_number

                cc_modularity = modularity(partition=grouped_nodes, graph=G)
                if cc_modularity > best_rank:
                    best_rank = cc_modularity
                    best_community = c_number
                    best_community_small = small_c_number
            if best_community:
                communities[best_community].extend(
                    small_communities[best_community_small]
                )
                delete_communities(
                    communities=communities,
                    communities_to_delete={
                        best_community_small: communities[best_community_small]
                    },
                )
                small_communities.pop(best_community_small)
                changed = True
        small_communities = find_small_communities(
            communities=communities, resolution=resolution
        )

    return communities


def df_merge_communities(G, communities):
    # mergowanie malych do duzych na bazie podobienstwa
    communities = {**communities}
    small_communities = find_small_communities(communities=communities)
    big_communities = {
        key: value for key, value in communities.items() if
        key not in small_communities
    }
    while small_communities:
        best_community, best_community_small, best_rank = None, None, 0
        for small_c_number, small_c_nodes in small_communities.items():
            for c_number, c_nodes in big_communities.items():
                if c_number == small_c_number:
                    continue
                cc_sim = community_similarity(G, small_c_nodes, c_nodes)
                if cc_sim > best_rank:
                    best_rank = cc_sim
                    best_community = c_number
                    best_community_small = small_c_number
        if best_community:
            big_communities[best_community].extend(
                communities[best_community_small])
            small_communities.pop(best_community_small)

    small_communities2 = find_small_communities(communities=big_communities)
    big_communities2 = {
        key: value
        for key, value in communities.items()
        if key not in small_communities2
    }
    print(small_communities2)

    return big_communities


def df_node_similarity_2(g_original: Graph, resolution=None, **kwargs):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )

    average_degree = sum(
        centrality for node, centrality in sorted_by_degree) / len(
        sorted_by_degree
    )
    pass


def df_node_similarity(g_original: Graph,
                       similarity_threshold: Optional[float] = None,
                       resolution: Optional[float] = None, **kwargs):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )

    nodes_to_process = [node for node, centrality in sorted_by_degree]

    average_degree = sum(
        centrality for node, centrality in sorted_by_degree) / len(
        sorted_by_degree
    )
    if not resolution:
        resolution = 1 - average_degree

    if not similarity_threshold:
        similarity_threshold = average_degree

    communities = defaultdict(list)
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
    communities = merge_communities_based_on_similarity(G, communities,
                                                        resolution=resolution,
                                                        similarity_threshold=similarity_threshold)
    return {"communities": communities.values()}
