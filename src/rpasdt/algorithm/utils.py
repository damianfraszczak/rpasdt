import math
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import stats


def modularity(partition, graph, weight="weight", resolution=1.):
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
        res += (inc.get(com, 0.0) / links) - resolution*(
                deg.get(com, 0.0) / (2.0 * links)) ** 2
    return res


def get_communities_size(communities):
    return [len(nodes) for community, nodes in communities.items()]


def get_community_avg_size(communities, alg="tmean", remove_outliers=True):
    count_nodes = get_communities_size(communities)
    if remove_outliers:
        count_nodes = remove_min_max(count_nodes)

    return getattr(stats, alg)(count_nodes)


def get_community_weighted_avg_size(communities):
    distribution = defaultdict(int)
    for nodes in communities.values():
        distribution[len(nodes)] += 1

    com_len = len(communities)

    distribution = {key: value / com_len for key, value in
                    distribution.items()}

    return (
        sum(
            [
                len(nodes) * distribution[len(nodes)]
                for community, nodes in communities.items()
            ]
        )
        / com_len
    )


def find_small_communities_modularity(
    communities,
    resolution=0.5,
    alg="tmean",
    remove_outliers=True,
    iteration=1,
    hard=True,
):
    # hard = False

    community_avg_size = get_community_avg_size(
        communities, alg='tmean', remove_outliers=True
    )

    community_avg_size = math.ceil(community_avg_size)
    community_avg_size /= 2 ** (iteration - 1)
    community_avg_size = max(community_avg_size, 2)

    # <= dla modularity, < dla similarity
    return dict(
        filter(
            lambda elem: len(elem[1]) <= community_avg_size,
            communities.items(),
        )
    )


def find_small_communities(
    communities,
    resolution=0.5,
    alg="tmean",
    remove_outliers=True,
    iteration=1,
    hard=True,
):
    # hard = False

    community_avg_size = get_community_avg_size(
        communities, alg=alg, remove_outliers=remove_outliers
    )
    community_avg_size = max(community_avg_size, 2)

    # community_avg_size = (community_avg_size) / max(count_nodes)

    # print(f"{community_avg_size}-{resolution}")

    community_avg_size *= resolution
    # community_avg_size /= math.sqrt(iteration)

    community_avg_size = math.ceil( community_avg_size)
    community_avg_size = max(community_avg_size, 2)

    # <= dla modularity, < dla similarity
    return dict(
        filter(
            lambda elem: len(elem[1]) < community_avg_size
            if hard
            else len(elem[1]) <= community_avg_size,
            communities.items(),
        )
    )


def delete_communities(communities, communities_to_delete):
    for to_delete in communities_to_delete.keys():
        communities.pop(to_delete, None)


def get_avg_degree(G):
    normalized_degree = nx.degree_centrality(G)
    return sum(
        centrality for node, centrality in normalized_degree.items()) / len(
        normalized_degree
    )


def remove_min_max(data):
    min_d = min(data)
    max_d = max(data)
    removed = [val for val in data if val != min_d and val != max_d]
    if len(removed) > 1:
        print(f"{data}-{removed}")
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
    print(f"{data}\n{no_outliers}")
    return no_outliers
