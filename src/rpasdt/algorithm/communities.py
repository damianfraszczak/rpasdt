"""Community detection methods."""
import math
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
from cdlib import algorithms
from networkx import Graph
from scipy.stats import tmean

from rpasdt.algorithm.similarity import (
    community_similarity,
    jaccard_node_similarity,
)
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.algorithm.utils import (
    delete_communities,
    find_small_communities,
    get_communities_size,
    modularity,
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
    **alg_kwargs,
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


# chwilowo zawsze sorensen
def merge_communities_similarity():
    pass


def merge_communities_modularity():
    pass


def merge_communities_df(
    G: Graph,
    communities: Dict,
    community_eval_threshold: float = 0.5,
    resolution: float = 0.5,
    max_iterations: [int] = math.inf,
):
    communities, small_communities = merge_communities(
        G=G,
        communities=communities,
        community_eval_function=merge_communities_similarity,
        community_eval_threshold=community_eval_threshold,
        resolution=resolution,
        max_iterations=max_iterations,
    )
    if small_communities:
        communities, small_communities = merge_communities(
            G=G,
            communities=communities,
            community_eval_function=merge_communities_modularity,
            community_eval_threshold=community_eval_threshold,
            resolution=resolution,
            max_iterations=max_iterations,
        )

    return communities


def merge_communities(
    G: Graph,
    communities: Dict,
    community_eval_function: Callable,
    community_eval_threshold: float = 0.5,
    resolution: float = 0.5,
    max_iterations: [int] = math.inf,
) -> Tuple[Dict, Dict]:
    communities = {**communities}
    small_communities = find_small_communities(
        communities=communities, resolution=resolution
    )

    current_iteration = 1
    changed = True
    while small_communities and changed and current_iteration <= max_iterations:
        current_iteration += 1
        changed = False
        for small_c_number, small_c_nodes in list(small_communities.items()):
            communities_evaluation = defaultdict(set)
            for c_number, c_nodes in communities.items():
                if c_number != small_c_number:
                    communities_evaluation[
                        community_eval_function(
                            G,
                            small_c_nodes,
                            c_nodes,
                        )
                    ].add(c_number)
            best_match = max(communities_evaluation.keys())
            best_match_communities = communities_evaluation.get(best_match) or []
            # ignore if it is hyb
            communities_number_without_compared = len(communities) - 1
            if (
                best_match <= similarity_threshold
                or len(best_match_communities) == communities_number_without_compared
            ):
                continue

            communities[small_c_number] = set()
            communities[small_c_number].update(small_c_nodes)
            for community_to_join in best_match_communities:
                communities[small_c_number].update(communities[community_to_join])
                delete_communities(
                    communities=communities,
                    communities_to_delete={
                        community_to_join: communities[community_to_join]
                    },
                )
                small_communities.pop(small_c_number, None)
                changed = True
        small_communities = find_small_communities(
            communities=communities, resolution=resolution
        )
    return communities, small_communities


def get_neighbour_communities(G, communities, community):
    partition = get_grouped_nodes(communities)
    neighbours = {nn for node in community for nn in nx.neighbors(G, node)}
    return {partition[node] for node in G if node in neighbours}


def merge_communities_based_on_similarity(
    G: Graph,
    communities,
    node_similarity_function,
    similarity_threshold,
    resolution=0.5,
):
    def _sorted_communities(c):
        return sorted(c.items(), key=lambda k: len(k[1]), reverse=True)

    def sm(communities, iteration=1):
        return find_small_communities(
            communities=communities,
            resolution=resolution,
            remove_outliers=False,
            iteration=1,
        )

    current_iteration = 0
    communities = {**communities}
    small_communities = sm(communities, current_iteration)

    changed = True
    while small_communities and changed:
        # print(
        #     f"C - {[len(com) for c, com in communities.items()]} - {communities}")
        # print(f"SM - {small_communities}")
        changed = False
        current_iteration += 1
        print(
            f"{current_iteration}-{len(small_communities)}-{get_communities_size(small_communities)}"
        )
        for small_c_number, small_c_nodes in list(small_communities.items()):
            best_community, best_community_small, best_rank = None, None, 0

            similarities = defaultdict(list)
            for c_number in get_neighbour_communities(
                G=G, communities=communities, community=small_c_nodes
            ):

                if c_number == small_c_number:
                    continue
                c_nodes = communities[c_number]
                similarities[
                    community_similarity(
                        G,
                        small_c_nodes,
                        c_nodes,
                        node_similarity_function=node_similarity_function,
                    )
                ].append(c_number)
            if not similarities:
                continue
            max_similarity = max(similarities.keys())
            max_similarity_communities = similarities.get(max_similarity) or []
            # ignore if it is hyb
            communities_number_without_compared = len(communities) - 1
            if (
                max_similarity <= similarity_threshold
                or len(max_similarity_communities)
                == communities_number_without_compared
            ):
                continue

            communities[small_c_number] = set()
            communities[small_c_number].update(small_c_nodes)

            for community_to_join in max_similarity_communities:
                communities[small_c_number].update(communities[community_to_join])
                delete_communities(
                    communities=communities,
                    communities_to_delete={
                        community_to_join: communities[community_to_join]
                    },
                )
                small_communities.pop(small_c_number, None)
                changed = True
            # print(f"{max_similarity}-{similarities[max_similarity]}")

            # for c_number, c_nodes in communities.items():
            #     if c_number == small_c_number:
            #         continue
            #     cc_sim = community_similarity(G, small_c_nodes,
            #                                   c_nodes,
            #                                   node_similarity_function=node_similarity_function)
            #     if cc_sim > best_rank and cc_sim > similarity_threshold:
            #         best_rank = cc_sim
            #         best_community = c_number
            #         best_community_small = small_c_number
            #
            # if best_community:
            #     communities[best_community].extend(
            #         small_communities[best_community_small]
            #     )
            #     delete_communities(
            #         communities=communities,
            #         communities_to_delete={
            #             best_community_small: communities[best_community_small]
            #         },
            #     )
            #     small_communities.pop(best_community_small)
            #     changed = True
        # small_communities = sm(communities, current_iteration)

    return communities


def merge_communities_based_on_louvain(G, communities, **kwargs):
    M = nx.quotient_graph(G, communities.values())
    return find_communities(graph=M, type=CommunityOptionEnum.LOUVAIN)


def merge_communities_based_on_modularity(
    G, communities, resolution, modularity_threshold, max_iterations
):
    communities = {**communities}

    def sm(communities, iteration):
        return find_small_communities(
            communities=communities,
            resolution=resolution,
            iteration=iteration,
        )

    current_iteration = 0
    changed = True

    small_communities = sm(communities, iteration=current_iteration)

    while small_communities and changed and current_iteration < max_iterations:
        changed = False
        current_iteration += 1
        # print(small_communities)
        for small_c_number, small_c_nodes in list(small_communities.items()):
            best_community, best_community_small, best_rank = None, None, -1
            for c_number in get_neighbour_communities(
                G=G, communities=communities, community=small_c_nodes
            ):

                if c_number == small_c_number:
                    continue
                grouped_nodes = get_grouped_nodes(communities)
                for node in small_c_nodes:
                    grouped_nodes[node] = c_number

                cc_modularity = modularity(partition=grouped_nodes, graph=G)

                if cc_modularity > best_rank and cc_modularity > 0:
                    best_rank = cc_modularity
                    best_community = c_number
                    best_community_small = small_c_number
            if best_community:
                communities[best_community].update(
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
        small_communities = sm(communities, iteration=current_iteration)

    return communities


def assign_community(G, communities, node, community=None):
    if G.nodes[node]["community"]:
        return G.nodes[node]["community"]
    current_community = community or max(communities.keys() or [0]) + 1
    G.nodes[node]["community"] = current_community
    communities[current_community].add(node)
    return current_community


def initial_communities_improved(
    g_original, similarity_threshold, node_similarity_function
):
    from scipy.stats import tmean

    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = dict(
        sorted(normalized_degree.items(), key=lambda x: x[1], reverse=True)
    )
    centralitites = [centrality for node, centrality in normalized_degree.items()]

    average_degree = tmean(centralitites)

    if not node_similarity_function:
        node_similarity_function = jaccard_node_similarity
    communities = defaultdict(set)

    biggest = [
        node
        for node, centrality in sorted_by_degree.items()
        if centrality > average_degree
    ]
    biggest_ratio = max(2, math.ceil(len(biggest) / 10))
    # biggest = biggest[:biggest_ratio]
    print(f"NEIGBHOURS {len(biggest)}")
    neighbours_of_biggest = defaultdict(set)
    for node in biggest:
        # tworze spolecznosci wokol najwiekszych plus doklejam do nich najmniejszego
        community = assign_community(G, communities, node)
        for small_node in G.neighbors(node):
            if G.degree[small_node] == 1:
                assign_community(G, communities, small_node, community)
            else:
                neighbours_of_biggest[small_node].add(node)
    print(f"BIG NEIG {len(neighbours_of_biggest)}")
    # neighbours_of_biggest = {
    #     key: value
    #     for key, value in neighbours_of_biggest.items()
    #     if sorted_by_degree[key] < average_degree
    # }
    print(f"AFTER BIG NEIG {len(neighbours_of_biggest)}")
    # dzialam od najmniejszych
    for small_node, big_neighbours in sorted(
        neighbours_of_biggest.items(), key=lambda k: len(k[1]), reverse=True
    ):

        best_sim, best_community = -1, -1
        for big_n in big_neighbours:
            community = G.nodes[big_n]["community"]
            new_community = [small_node]
            sim = community_similarity(
                G,
                communities[community],
                new_community,
                node_similarity_function=node_similarity_function,
            )
            if sim > best_sim and sim >= similarity_threshold:
                best_sim = sim
                best_community = community
        if best_community > -1:
            assign_community(G, communities, small_node, best_community)
    print("PRZUPISAYWANIE")
    for node in G:
        if not G.nodes[node]["community"]:
            assign_community(G, communities, node)

    return communities


def initial_communities2(g_original, similarity_threshold, node_similarity_function):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )
    centralitites = [centrality for node, centrality in normalized_degree.items()]

    average_degree = tmean(centralitites)

    nodes_to_process = [node for node, centrality in sorted_by_degree]

    if not node_similarity_function:
        node_similarity_function = jaccard_node_similarity
    communities = defaultdict(set)
    for node in nodes_to_process:
        current_community = assign_community(G, communities, node)
        for node_n in G.neighbors(node):
            if G.nodes[node_n]["community"]:
                continue
            if normalized_degree[node] > average_degree:

                similarity = node_similarity_function(G, node, node_n)
                if similarity >= similarity_threshold:
                    assign_community(G, communities, node_n, current_community)
    return communities


def initial_communities(g_original, similarity_threshold, node_similarity_function):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)
    sorted_by_degree = sorted(
        normalized_degree.items(), key=lambda x: x[1], reverse=True
    )

    nodes_to_process = [node for node, centrality in sorted_by_degree]

    if not node_similarity_function:
        node_similarity_function = jaccard_node_similarity
    communities = defaultdict(set)
    for node in nodes_to_process:
        current_community = assign_community(G, communities, node)

        for node_n in G.neighbors(node):
            if G.nodes[node_n]["community"]:
                continue
            if G.degree[node_n] == 1:
                assign_community(G, communities, node_n, current_community)
                continue
            similarity = node_similarity_function(G, node, node_n)
            print(f"{node}-{node_n}-{similarity}")
            if similarity >= similarity_threshold:
                assign_community(G, communities, node_n, current_community)

    return communities


def initial_communities_each_separate(
    g_original, similarity_threshold, node_similarity_function
):
    return {node: {node} for node in g_original}


def df_node_similarity(
    g_original: Graph,
    node_similarity_function: Optional[Callable] = None,
    similarity_threshold: Optional[float] = None,
    resolution: Optional[float] = None,
    max_iterations: Optional[int] = math.inf,
    **kwargs,
):
    G = g_original.copy()

    nx.set_node_attributes(G, None, "community")
    normalized_degree = nx.degree_centrality(G)

    average_degree = sum(
        centrality for node, centrality in normalized_degree.items()
    ) / len(normalized_degree)

    if not similarity_threshold:
        similarity_threshold = average_degree

    if not resolution:
        resolution = 1 - similarity_threshold

    if not node_similarity_function:
        node_similarity_function = jaccard_node_similarity

    communities = initial_communities(
        g_original,
        similarity_threshold=similarity_threshold,
        node_similarity_function=node_similarity_function,
    )
    print("INITAL DONE")
    # communities = merge_communities_based_on_similarity(
    #     G=G,
    #     communities=communities,
    #     node_similarity_function=node_similarity_function,
    #     similarity_threshold=similarity_threshold,
    #     resolution=resolution,
    #     # max_iterations=max_iterations
    # )
    # print("SIM DONE")
    # # # poprawic by te wybrane duze klastry zostaly i podpinal male do nich ciagel
    communities = merge_communities_based_on_modularity(
        G=G,
        communities=communities,
        modularity_threshold=similarity_threshold,
        resolution=resolution,
        max_iterations=0,
    )
    print("MOD DONE")
    return {"communities": communities.values()}
