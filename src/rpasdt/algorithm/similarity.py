import networkx as nx

from rpasdt.algorithm.utils import reject_outliers


def community_similarity(G, c1, c2, node_similarity_function):
    coefficients = [node_similarity_function(G, a, b) for a in c1 for b
                    in c2]
    coefficients = reject_outliers(coefficients)
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
