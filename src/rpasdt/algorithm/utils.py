import math
from collections import defaultdict

import networkx as nx


def community_similarity(G, c1, c2, node_similarity_function):
    coefficients = [node_similarity_function(G, a, b) for a in c1 for b
                    in c2]
    return sum(coefficients) / len(coefficients)


# algorytmy https://cran.r-project.org/web/packages/linkprediction/vignettes/proxfun.html
def sorensen_node_similarity(G, a, b):
    intersection_size = len(list(nx.common_neighbors(G, a, b)))
    a_size = G.degree[a]
    b_size = G.degree[b]

    return 2 * intersection_size / (a_size + b_size)


def jaccard_node_similarity(G, a, b):
    result = nx.jaccard_coefficient(G, [(a, b)])
    return next(result)[2]


def academic_adar_node_similarity(G, a, b):
    result = nx.adamic_adar_index(G, [(a, b)])
    return next(result)[2]


def hub_promoted_index_node_similarity(G, a, b):
    intersection_size = len(list(nx.common_neighbors(G, a, b)))
    a_size = G.degree[a]
    b_size = G.degree[b]

    return intersection_size / min(a_size, b_size)


def hub_depressed_index_node_similarity(G, a, b):
    intersection_size = len(list(nx.common_neighbors(G, a, b)))
    a_size = G.degree[a]
    b_size = G.degree[b]

    return intersection_size / max(a_size, b_size)

def leicht_holme_node_similarity(G, a, b):
    intersection_size = len(list(nx.common_neighbors(G, a, b)))
    a_size = G.degree[a]
    b_size = G.degree[b]

    return intersection_size / (a_size * b_size)

def resource_allocation_index_node_similarity(G, a, b):
    result = nx.resource_allocation_index(G, [(a, b)])
    return next(result)[2]


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
        res += (inc.get(com, 0.0) / links) - (
            deg.get(com, 0.0) / (2.0 * links)) ** 2
    return res


def get_community_avg_size(communities):
    return sum([len(nodes) for community, nodes in communities.items()]) / len(
        communities
    )


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


def find_small_communities(communities, resolution=0.5):
    community_avg_size = math.floor(
        get_community_avg_size(communities)) * resolution
    community_avg_size = max(community_avg_size, 2)

    # <= dla modularity, < dla similarity
    return dict(
        filter(lambda elem: len(elem[1]) <= community_avg_size,
               communities.items())
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
