import math
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import stats


def modularity(partition, graph, weight="weight"):
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
        res += (inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2
    return res


def get_communities_size(communities):
    return [len(nodes) for community, nodes in communities.items()]


def get_community_avg_size(communities, alg="tmean", remove_outliers=True):
    count_nodes = get_communities_size(communities)
    if remove_outliers:
        count_nodes = reject_outliers(count_nodes)

    return getattr(stats, alg)(count_nodes)


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


def find_small_communities(
    communities, resolution=0.5, alg="tmean", remove_outliers=True, iteration=1
):
    community_avg_size = get_community_avg_size(
        communities, alg=alg, remove_outliers=True
    )
    # community_avg_size = max(community_avg_size, 2)

    # community_avg_size = (community_avg_size) / max(count_nodes)

    # print(f"{community_avg_size}-{resolution}")

    community_avg_size *= resolution
    community_avg_size /= 2 ** (iteration - 1)
    community_avg_size = math.floor(community_avg_size)
    community_avg_size = max(community_avg_size, 2)

    # <= dla modularity, < dla similarity
    return dict(
        filter(lambda elem: len(elem[1]) <= community_avg_size, communities.items())
    )


def delete_communities(communities, communities_to_delete):
    for to_delete in communities_to_delete.keys():
        communities.pop(to_delete, None)


def get_avg_degree(G):
    normalized_degree = nx.degree_centrality(G)
    return sum(centrality for node, centrality in normalized_degree.items()) / len(
        normalized_degree
    )


def reject_outliers(data, m=2.0):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.0)
    return data[s < m]
