"""Community detection methods."""
from typing import Dict, List, Union

import networkx as nx
from cdlib import algorithms
from networkx import Graph

from rpasdt.algorithm.taxonomies import CommunityOptionEnum
from rpasdt.common.utils import get_enum, get_function_default_kwargs

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
    alg = getattr(algorithms, get_enum(type, CommunityOptionEnum).value)
    kwargs = {**get_function_default_kwargs(alg), **alg_kwargs, **{"g_original": graph}}
    _update_communities_kwarg(
        type=type, kwargs=kwargs, number_communities=number_communities
    )
    result = alg(**kwargs)
    return {index: community for index, community in enumerate(result.communities)}


G = nx.karate_club_graph()
