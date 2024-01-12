"""Community detection methods."""
import sys
from typing import Collection, Dict, List, Union

import networkx as nx
from cdlib import algorithms
from networkx import Graph

from rpasdt.algorithm.blocd import blocd as blocd_imp
from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import (
    get_enum,
    get_function_default_kwargs,
    get_object_value,
)

thismodule = sys.modules[__name__]
NUMBER_OF_COMMUNITIES_KWARG_NAMES = ["k", "number_communities", "level"]
MAX_NUMBER_OF_COMMUNITIES_KWARG_NAMES = ["kmax"]


def blocd(*args, **kwargs):
    return blocd_imp(*args, **kwargs)


def _update_communities_kwarg(
    graph: Graph, type: CommunityOptionEnum, kwargs: dict,
    number_communities: int
) -> dict:
    # correctly set the desired number of communities for given alg
    for name in NUMBER_OF_COMMUNITIES_KWARG_NAMES:
        if name in kwargs:
            kwargs[name] = (
                number_communities - 1
                if type == CommunityOptionEnum.GIRVAN_NEWMAN
                else number_communities
            )
    for name in MAX_NUMBER_OF_COMMUNITIES_KWARG_NAMES:
        if name in kwargs:
            current_val = kwargs[name]
            if current_val == 0:
                kwargs[name] = len(graph.nodes) - 1
    # remove empty collections from kwargs
    for key in set(kwargs.keys()):
        value = kwargs[key]
        if isinstance(value, Collection) and not value:
            kwargs.pop(key)
    return kwargs


COMMUNITY_CACHE = {}


def find_communities(
    type: Union[str, CommunityOptionEnum],
    graph: Graph,
    number_communities: int = 1,
    **alg_kwargs,
) -> Dict[int, List[int]]:
    alg_function_name = get_enum(type, CommunityOptionEnum).value
    cache_key = (
        f"{alg_function_name}-{nx.weisfeiler_lehman_graph_hash(graph)}-{alg_kwargs}"
    )

    result = COMMUNITY_CACHE.get(cache_key)
    if result:
        return result

    alg = getattr(algorithms, alg_function_name, None) or getattr(
        thismodule, alg_function_name
    )
    kwargs = {**get_function_default_kwargs(alg), **alg_kwargs,
              **{"g_original": graph}}
    _update_communities_kwarg(
        graph=graph, type=type, kwargs=kwargs,
        number_communities=number_communities
    )

    result = alg(**kwargs)
    communities = {
        index: community
        for index, community in
        enumerate(get_object_value(result, "communities"))
    }

    COMMUNITY_CACHE[cache_key] = communities

    return communities
