from typing import Optional, Dict, List

from community import community_louvain
from networkx import Graph, k_shell, k_core, community

from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import group_dict_by_values
from rpasdt.network.networkx_utils import map_networkx_communities_to_dict


def community_bipartition_local(graph: Graph) -> Dict[int, List[int]]:
    return community.kernighan_lin_bisection(graph)


def community_k_clique_local(graph: Graph, clique_size: int = 2) -> Dict[int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.k_clique_communities(graph, clique_size)])


def community_k_shell_local(graph: Graph, k: Optional[int] = None, core_size: Optional[int] = None) -> Dict[
    int, List[int]]:
    return map_networkx_communities_to_dict([c for c in k_shell(G=graph, k=k, core_number=core_size)])


def community_k_core_local(graph: Graph, k: Optional[int] = None, core_number: Optional[int] = None) -> Dict[
    int, List[int]]:
    return {0: k_shell(G=graph, k=k, core_number=core_number).nodes()}


def community_k_core_local(graph: Graph, k: Optional[int] = None, core_number: Optional[int] = None) -> Dict[
    int, List[int]]:
    return {0: k_core(G=graph, k=k, core_number=core_number).nodes()}


def community_louvain_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    return {community + 1: node for community, node in
            group_dict_by_values(community_louvain.best_partition(graph)).items()}


def community_girvan_newman_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    if not communities_count:
        communities_count = 1
    community_generator = community.girvan_newman(graph)
    for cc in community_generator:
        if len(cc) >= communities_count:
            break
    return map_networkx_communities_to_dict(cc)


def community_label_propagation_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.label_propagation_communities(graph)])


def community_greedy_modularity_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.greedy_modularity_communities(graph)])


def community_naive_greedy_modularity_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[
    int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.naive_greedy_modularity_communities(graph)])


def community_naive_modularity_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.naive_greedy_modularity_communities(graph)])


def community_tree_local(graph: Graph, communities_count: Optional[int] = None) -> Dict[int, List[int]]:
    return map_networkx_communities_to_dict([c for c in community.lukes_partitioning(graph)])


COMMUNITY_OPERATION_MAP = {
    CommunityOptionEnum.LOUVAIN: community_louvain_local,
    CommunityOptionEnum.GIRVAN_NEWMAN: community_girvan_newman_local,
    CommunityOptionEnum.BIPARTITION: community_bipartition_local,
    CommunityOptionEnum.GREEDY_MODULARITY: community_greedy_modularity_local,
    CommunityOptionEnum.NAIVE_MODULARITY: community_naive_modularity_local,
    CommunityOptionEnum.LABEL_PROPAGATION: community_label_propagation_local,
    CommunityOptionEnum.TREE: community_tree_local,
    CommunityOptionEnum.K_CLIQUE: community_k_clique_local,
    CommunityOptionEnum.K_CORE: community_k_clique_local,
}


def find_communities(
        alg: CommunityOptionEnum,
        graph: Graph,
        communities_count: Optional[int] = None, *args, **kwargs) -> Dict[int, List[int]]:
    community_alg = COMMUNITY_OPERATION_MAP.get(alg)
    if alg:
        return community_alg(graph=graph, communities_count=communities_count)
    return {}
