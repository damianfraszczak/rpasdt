from rpasdt.common.enums import StringChoiceEnum


class GraphLayout(StringChoiceEnum):
    """Available graph rendering layouts."""

    CIRCULAR = ("CIRCULAR", "Draw the graph G with a circular layout.")
    KAMADA_KAWAI = (
        "KAMADA_KAWAI",
        "Draw the graph G with a Kamada-Kawai force-directed layout.",
    )
    PLANAR = ("PLANAR", "Draw a planar networkx graph with planar layout.")
    RANDOM = ("RANDOM", "Draw the graph G with a random layout.")
    SPECTRAL = ("SPECTRAL", "Draw the graph G with a spectral 2D layout.")
    SPRING = ("SPRING", "Draw the graph G with a spring layout.")
    SHELL = ("SHELL", "Draw networkx graph with shell layout.")


class DiffusionGraphNodeRenderTypeEnum(StringChoiceEnum):
    FULL = ("FULL", "Full graph")
    ONLY_INFECTED = ("ONLY_INFECTED", "Only infected nodes")


class GraphTypeEnum(StringChoiceEnum):
    """Available graph types and predefined topologies."""

    WATTS_STROGATZ = ("WATTS_STROGATZ_GRAPH", "Watts-Strogatz")

    BALANCED_TREE = ("BALANCED_TREE", "Balanced tree")
    COMPLETE = ("COMPLETE", "Complete graph")
    ERDOS_RENYI = ("ERDOS_RENYI", "Erdős-Rényi graph or a binomial graph")
    # Social Networks
    KARATE_CLUB = ("KARATE_CLUB_GRAPH", "Karate club")
    DAVIS_SOUTHERN = ("DAVIS_SOUTHERN", "Davis Southern women social network")
    FLORENTINE_FAMILIES = ("FLORENTINE_FAMILIES", "Florentine families graph")
    LES_MISERABLES = ("LES_MISERABLES", "Les Miserables graph")
    # COMMUNITIES
    CAVEMAN_GRAPH = ("CAVEMAN_GRAPH", "Caveman graph of l cliques of size k")
    CONNECTED_CAVEMAN_GRAPH = (
        "CONNECTED_CAVEMAN_GRAPH",
        "Connected Caveman graph of l cliques of size k",
    )


class DiffusionTypeEnum(StringChoiceEnum):
    """Available diffusion models."""

    SI = ("SI", "SI Model")
    SIS = ("SIS", "SIS Model")
    SIR = ("SIR", "SIR Model")
    SEIR = ("SEIR", "SEIR Model")
    SWIR = ("SWIR", "SWIR Model")
    THRESHOLD = ("THRESHOLD", "Threshold")
    INDEPENDENT_CASCADES = ("INDEPENDENT_CASCADES", "Independent Cascades")


class NodeStatusEnum(StringChoiceEnum):
    """Available node statuses."""

    SUSCEPTIBLE = "Susceptible"
    INFECTED = "Infected"
    RECOVERED = "Recovered"
    BLOCKED = "Blocked"


# mappign how the statuses from ndlib corresponds to this app
NodeStatusToNodeStatusCodeMap = {
    NodeStatusEnum.SUSCEPTIBLE: 0,
    NodeStatusEnum.INFECTED: 1,
}


class CentralityOptionEnum(StringChoiceEnum):
    DEGREE = ("degree", "Compute the degree centrality for nodes.")
    EIGENVECTOR = ("eigenvector", "Compute the eigenvector centrality for the graph G.")
    KATZ = ("katz", "Compute the Katz centrality for the nodes of the graph G.")
    CLOSENESS = ("closeness", "Compute closeness centrality for nodes.")
    BETWEENNESS = (
        "betweenness",
        "Compute the shortest-path betweenness centrality for nodes.",
    )
    EDGE_BETWEENNESS = (
        "edge_betweenness",
        ("Compute betweenness centrality for edges."),
    )
    HARMONIC = ("harmonic", "Compute harmonic centrality for nodes.")
    PERLOCATION = ("percolation", "Compute the percolation centrality for nodes.")
    TROPHIC = ("Trophic", "Compute the trophic levels of nodes.")
    VOTE_RANK = (
        "Vote rank",
        "Select a list of influential nodes in a graph using VoteRank algorithm",
    )
    PAGE_RANK = ("PageRank", "Returns the PageRank of the nodes in the graph.")


class CommunityOptionEnum(StringChoiceEnum):
    BIPARTITION = (
        "bipartition",
        "Finds 2 communities in a graph using the Kernighan method.",
    )
    LOUVAIN = ("louvain", "Find communities in graph using the Louvain method.")
    # centrality based
    GIRVAN_NEWMAN = (
        "girvan_newman",
        "Finds communities in a graph using the Girvan–Newman method.",
    )
    GREEDY_MODULARITY = (
        "greedy_modularity",
        "Finds communities in a graph using the Clauset-Newman-Moore greedy modularity maximization.",
    )
    NAIVE_MODULARITY = (
        (
            "naive_modularity",
            "Find communities in graph using the naive modularity maximization.",
        ),
    )
    LABEL_PROPAGATION = (
        "label_propagation",
        "Finds communities in a graph using the Label propagation method.",
    )
    TREE = (
        "tree",
        "Finds communities in a graph using the Lukes Algorithm for exact optimal weighted tree partitioning.",
    )
    K_CLIQUE = ("k_clique", "Finds communities in a graph using the K-Clique method.")
    K_CORE = ("k_core", "Finds core in a graph using the K-core method.")
    K_SHELL = ("k_shell", "Finds shell in a graph using the K-shell method.")
    K_CRUST = ("k_crust", "Finds crust in a graph using the K-shell method.")
    K_CORONA = ("k_corona", "Finds korona in a graph using the K-shell method.")
    K_MEANS = ("k_means", "Finds communities in a graph using the K-means method.")


class SourceSelectionOptionEnum(StringChoiceEnum):
    RANDOM = ("random", "Select sources in random way.")
    DEGREE = ("degree", "Select sources according to degree centrality metrics.")
    CLOSENESS = (
        "closeness",
        "Select sources according to closeness centrality metrics.",
    )
    BETWEENNESS = (
        "betweenness",
        "Select sources according to betweenness centrality metrics.",
    )


class SourceDetectionAlgorithm(StringChoiceEnum):
    DYNAMIC_AGE = ("dynamic_age", "Find sources with dynamic age algorithm.")
    NET_SLEUTH = ("net_sleuth", "Find sources with NetSleuth algorithm.")
    RUMOR_CENTER = ("rumor_center", "Find sources with Rumor center algorithm.")
    CENTRALITY_BASED = ("centrality", "Find sources with centrality based algorithm.")
    UNBIASED_CENTRALITY_BASED = (
        "unbiased_centrality",
        "Find sources with unbiased centrality based algorithm.",
    )
    COMMUNITY_CENTRALITY_BASED = (
        "community_centrality",
        "Find sources with community centrality based algorithm.",
    )
    COMMUNITY_UNBIASED_CENTRALITY_BASED = (
        "community_unbiased_centrality",
        "Find sources with community unbiased centrality based algorithm.",
    )
