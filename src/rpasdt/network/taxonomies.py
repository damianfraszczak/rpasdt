from rpasdt.common.enums import StringChoiceEnum


class NodeAttributeEnum(StringChoiceEnum):
    """Available node attributes."""

    COLOR = "COLOR"
    SIZE = "SIZE"
    EXTRA_LABEL = "EXTRA_LABEL"
    LABEL = "LABEL"
    SOURCE = "SOURCE"


class DistanceMeasureOptionEnum(StringChoiceEnum):
    pass


NETWORK_OPTIONS = {
    "bridge": ("Bridges", "Generate all bridges in a graph."),
    "cycle": (
        "Simple cycles",
        "Find simple cycles (elementary circuits) of a directed graph.",
    ),
    "degree_assortativity": (
        "Degree assortativity",
        "Compute degree assortativity of graph.",
    ),
    "average_neighbor_degree": (
        "Average neighbor degree",
        "Returns the average degree of the neighborhood of each node.",
    ),
    "k_nearest_neighbors": (
        "K-nearest neighbors",
        "Compute the average degree connectivity of graph.",
    ),
    "average_clustering": (
        "Average clustering",
        "Compute the average clustering coefficient.",
    ),
}
