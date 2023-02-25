import math
import statistics
from collections import defaultdict
from typing import Set

import networkx as nx
import numpy as np
import powerlaw
from networkx import Graph
from sklearn.metrics import normalized_mutual_info_score


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
    print(len(first_partition_c))
    print(len(second_partition_c))

    return normalized_mutual_info_score(first_partition_c, second_partition_c)


def get_power_law(G):
    deg_dist = sorted([deg for node, deg in G.degree()], reverse=True)
    fit = powerlaw.Fit(deg_dist, discrete=True)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    return alpha, xmin


last_size = 0


def modularity(partition, graph, weight="weight", resolution=1.0):
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] = deg.get(com, 0.0) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.0) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.0) + float(edge_weight) / 2.0

    res = 0.0
    for com in set(partition.values()):
        res += (inc.get(com, 0.0) / links) - resolution * (
            deg.get(com, 0.0) / (2.0 * links)
        ) ** 2
    return res


def get_communities_size(communities):
    return [len(nodes) for community, nodes in communities.items()]


def get_community_avg_size(communities, alg="median", remove_outliers=False):
    count_nodes = get_communities_size(communities)
    if remove_outliers:
        count_nodes = remove_min_max(count_nodes)
    community_avg_size = getattr(statistics, alg)(count_nodes)
    community_avg_size = math.ceil(community_avg_size)
    community_avg_size = max(community_avg_size, 2)
    return community_avg_size


def get_community_weighted_avg_size(communities):
    distribution = defaultdict(int)
    for nodes in communities.values():
        distribution[len(nodes)] += 1

    com_len = len(communities)

    distribution = {key: value / com_len for key, value in distribution.items()}

    return (
        sum(
            [
                len(nodes) * distribution[len(nodes)]
                for community, nodes in communities.items()
            ]
        )
        / com_len
    )


def filter_communities_by_size(communities, size, hard=True):
    small_c = dict(
        filter(
            lambda elem: len(elem[1]) < size if hard else len(elem[1]) <= size,
            communities.items(),
        )
    )
    return small_c


def find_small_communities(
    communities,
    alg="mean",
    remove_outliers=True,
    iteration=1,
    hard=True,
):
    # hard = False

    community_avg_size = get_community_avg_size(
        communities, alg=alg, remove_outliers=remove_outliers
    )

    return filter_communities_by_size(
        communities=communities, size=community_avg_size, hard=hard
    )


def delete_communities(communities, communities_to_delete):
    for to_delete in communities_to_delete.keys():
        communities.pop(to_delete, None)


def remove_min_max(data):
    min_d = min(data)
    max_d = max(data)
    removed = [val for val in data if val != min_d and val != max_d]
    if len(removed) > 1:
        return removed
    return data


def reject_outliers(data, m=2.0):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)

    return data[s < m]


def reject_outliers2(data, m=2.0):
    an_array = np.array(data)
    mean = np.mean(an_array)
    standard_deviation = np.std(an_array)
    distance_from_mean = abs(an_array - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = an_array[not_outlier]
    # print(f"{data}\n{no_outliers}")
    return no_outliers
