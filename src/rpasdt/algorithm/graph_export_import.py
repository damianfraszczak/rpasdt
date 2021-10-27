"""Graph export/import utilities."""
from typing import List, Union

import networkx as nx

from rpasdt.algorithm.taxonomies import GraphDataFormatEnum


def _fetch_lines(data: Union[str, List[str]]) -> List[str]:
    return data.split("\n") if isinstance(data, str) else data


GRAPH_EXPORTER = {
    GraphDataFormatEnum.MULTILINE_ADJLIST: lambda graph: list(
        nx.generate_multiline_adjlist(graph)
    )
}

GRAPH_IMPORTER = {
    GraphDataFormatEnum.MULTILINE_ADJLIST: lambda data: nx.parse_multiline_adjlist(
        iter(_fetch_lines(data))
    )
}
