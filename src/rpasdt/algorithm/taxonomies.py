"""Algorithms taxonomies."""
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
    """Available graph nodes rendering nodes."""

    FULL = ("FULL", "Full graph")
    ONLY_INFECTED = ("ONLY_INFECTED", "Only infected nodes")


class GraphDataFormatEnum(StringChoiceEnum):
    MULTILINE_ADJLIST = ("MULTILINE_ADJLIST", "Multiline adjacency list")


class GraphTypeEnum(StringChoiceEnum):
    """Available graph types and predefined topologies."""

    WATTS_STROGATZ = ("WATTS_STROGATZ", "Watts-Strogatz")
    BARABASI_ALBERT = ("BARABASI_ALBERT", "Barabási–Albert")
    BALANCED_TREE = ("BALANCED_TREE", "Balanced tree")
    COMPLETE = ("COMPLETE", "Complete graph")
    ERDOS_RENYI = ("ERDOS_RENYI", "Erdős-Rényi graph or a binomial graph")
    STAR = (
        "STAR",
        "The star graph consists of one center node connected to n outer nodes.",
    )
    # Social Networks
    KARATE_CLUB = ("KARATE_CLUB", "Karate club")
    DAVIS_SOUTHERN = ("DAVIS_SOUTHERN", "Davis Southern women social network")
    FLORENTINE_FAMILIES = ("FLORENTINE_FAMILIES", "Florentine families graph")
    LES_MISERABLES = ("LES_MISERABLES", "Les Miserables graph")
    # COMMUNITIES
    CAVEMAN_GRAPH = ("CAVEMAN_GRAPH", "Caveman graph of l cliques of size k")
    CONNECTED_CAVEMAN_GRAPH = (
        "CONNECTED_CAVEMAN_GRAPH",
        "Connected Caveman graph of l cliques of size k",
    )
    CUSTOM = ("CUSTOM", "Custom graph")


class DiffusionTypeEnum(StringChoiceEnum):
    """Available diffusion models."""

    SI = ("SI", "SI Model")
    SIS = ("SIS", "SIS Model")
    SIR = ("SIR", "SIR Model")
    SEIR = ("SEIR", "SEIR Model")
    SWIR = ("SWIR", "SWIR Model")
    THRESHOLD = ("THRESHOLD", "Threshold")
    GENERALISED_THRESHOLD = ("GENERALISED_THRESHOLD", "Generalised Threshold")
    KERTESZ_THRESHOLD = ("KERTESZ_THRESHOLD", "Kertesz Threshold")
    INDEPENDENT_CASCADES = ("INDEPENDENT_CASCADES", "Independent Cascades")
    VOTER = ("VOTER", "Voter")
    Q_VOTER = ("Q_VOTER", "Q-Voter")
    MAJORITY_RULE = ("MAJORITY_RULE", "Majority rule")
    SZNAJD = ("SZNAJD", "Sznajd")


class DiffusionSimulationMode(StringChoiceEnum):
    """Available diffusion simulation modes."""

    SINGLE = ("SINGLE", "Single iteration.")
    BUNCH = ("BUNCH", "Bunch of iterations.")
    TO_FULL_COVER = ("TO_FULL_COVER", "To cover the whole network.")


class NodeStatusEnum(StringChoiceEnum):
    """Available node statuses."""

    SUSCEPTIBLE = "Susceptible"
    INFECTED = "Infected"
    RECOVERED = "Recovered"
    BLOCKED = "Blocked"


#  Numbers based on cdlib.
NodeStatusToValueMapping = {
    NodeStatusEnum.SUSCEPTIBLE: 0,
    NodeStatusEnum.INFECTED: 1,
    NodeStatusEnum.RECOVERED: 2,
    NodeStatusEnum.BLOCKED: -1,
}


class CentralityOptionEnum(StringChoiceEnum):
    """Available centrality measures."""

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


class ClusteringOptionEnum(StringChoiceEnum):
    K_CLIQUE = ("k_clique", "Finds clusters in a graph using the K-Clique method.")
    K_CORE = ("k_core", "Finds clusters in a graph using the K-core method.")
    K_SHELL = ("k_shell", "Finds clusters in a graph using the K-shell method.")
    K_CRUST = ("k_crust", "Finds clusters in a graph using the K-shell method.")
    K_CORONA = ("k_corona", "Finds clusters in a graph using the K-shell method.")
    K_MEANS = ("k_means", "Finds clusters in a graph using the K-means method.")


class CommunityOptionEnum(StringChoiceEnum):
    """Available community detection methods."""

    AGDL = (
        "agdl",
        "AGDL is a graph-based agglomerative algorithm, for clustering high-dimensional data.",
    )
    ASYNC_FLUID = (
        "async_fluid",
        "Fluid Communities (FluidC) is based on the simple idea of fluids (i.e., communities) interacting in an environment (i.e., a non-complete graph), expanding and contracting.",
    )
    BELIEF = (
        "belief",
        "Belief community seeks the consensus of many high-modularity partitions.",
    )
    CPM = ("cpm", "CPM is a model where the quality function to optimize is:")
    CHINESEWHISPERS = (
        "chinesewhispers",
        "Fuzzy graph clustering that (i) creates an intermediate representation of the input graph, which reflects the â€śambiguityâ€ť of its nodes, and (ii) uses hard clustering to discover crisp clusters in such â€śdisambiguatedâ€ť intermediate graph.",
    )
    DER = ("der", "DER is a Diffusion Entropy Reducer graph clustering algorithm.")
    EDMOT = ("edmot", "The algorithm first creates the graph of higher order motifs.")
    EIGENVECTOR = (
        "eigenvector",
        "Newmans leading eigenvector method for detecting community structure based on modularity.",
    )
    EM = ("em", "EM is based on based on a mixture model.")
    GA = ("ga", "Genetic based approach to discover communities in social networks.")
    GDMP2 = (
        "gdmp2",
        "Gdmp2 is a method for identifying a set of dense subgraphs of a given sparse graph.",
    )
    GEMSEC = (
        "gemsec",
        "The procedure uses random walks to approximate the pointwise mutual information matrix obtained by pooling normalized adjacency matrix powers.",
    )
    GIRVAN_NEWMAN = (
        "girvan_newman",
        "The Girvan-Newman algorithm detects communities by progressively removing edges from the original graph.",
    )
    GREEDY_MODULARITY = (
        "greedy_modularity",
        "The CNM algorithm uses the modularity to find the communities strcutures.",
    )
    HEAD_TAIL = (
        "head_tail",
        "Identifying homogeneous communities in complex networks by applying head/tail breaks on edge betweenness given its heavy-tailed distribution.",
    )
    INFOMAP = ("infomap", "Infomap is based on ideas of information theory.")
    KCUT = ("kcut", "An Efficient Spectral Algorithm for Network Community Discovery.")
    LABEL_PROPAGATION = (
        "label_propagation",
        "The Label Propagation algorithm (LPA) detects communities using network structure alone.",
    )
    LEIDEN = (
        "leiden",
        "The Leiden algorithm is an improvement of the Louvain algorithm.",
    )
    LOUVAIN = ("louvain", "Louvain maximizes a modularity score for each community.")
    LSWL = (
        "lswl",
        "LSWL locally discovers networksâ€™ the communities precisely, deterministically, and quickly.",
    )
    LSWL_PLUS = (
        "lswl_plus",
        "LSWL+ is capable of finding a partition with overlapping communities or without them, based on user preferences.",
    )
    MARKOV_CLUSTERING = (
        "markov_clustering",
        "The Markov clustering algorithm (MCL) is based on simulation of (stochastic) flow in graphs.",
    )
    MCODE = (
        "mcode",
        "MCODE is the earliest seed-growth method for predicting protein complexes from PPI networks.",
    )
    MOD_M = (
        "mod_m",
        "Community Discovery algorithm designed to find local optimal community structures in large networks starting from a given source vertex.",
    )
    MOD_R = (
        "mod_r",
        "Community Discovery algorithm that infers the hierarchy of communities that enclose a given vertex by exploring the graph one vertex at a time.",
    )
    PARIS = (
        "paris",
        "Paris is a hierarchical graph clustering algorithm inspired by modularity-based clustering techniques.",
    )
    PYCOMBO = (
        "pycombo",
        "This is an implementation (for Modularity maximization) of the community detection algorithm called â€śComboâ€ť.",
    )
    RBER_POTS = (
        "rber_pots",
        "rber_pots is a model where the quality function to optimize is:",
    )
    RB_POTS = (
        "rb_pots",
        "Rb_pots is a model where the quality function to optimize is:",
    )
    RICCI_COMMUNITY = (
        "ricci_community",
        "Curvature is a geometric property to describe the local shape of an object.",
    )
    R_SPECTRAL_CLUSTERING = (
        "r_spectral_clustering",
        "Spectral clustering partitions the nodes of a graph into groups based upon the eigenvectors of the graph Laplacian.",
    )
    SCAN = (
        "scan",
        "SCAN (Structural Clustering Algorithm for Networks) is an algorithm which detects clusters, hubs and outliers in networks.",
    )
    SIGNIFICANCE_COMMUNITIES = (
        "significance_communities",
        "Significance_communities is a model where the quality function to optimize is:",
    )
    SPINGLASS = (
        "spinglass",
        "Spinglass relies on an analogy between a very popular statistical mechanic model called Potts spin glass, and the community structure.",
    )
    SURPRISE_COMMUNITIES = (
        "surprise_communities",
        "Surprise_communities is a model where the quality function to optimize is:",
    )
    WALKTRAP = ("walktrap", "walktrap is an approach based on random walks.")
    SBM_DL = (
        "sbm_dl",
        "Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models.",
    )
    SBM_DL_NESTED = (
        "sbm_dl_nested",
        "Efficient Monte Carlo and greedy heuristic for the inference of stochastic block models.",
    )
    SCD = (
        "scd",
        "The procedure greedily optimizes the approximate weighted community clustering metric.",
    )
    SPECTRAL = (
        "spectral",
        "SCD implements a Spectral Clustering algorithm for Communities Discovery.",
    )
    THRESHOLD_CLUSTERING = (
        "threshold_clustering",
        "Developed for semantic similarity networks, this algorithm specifically targets weighted and directed graphs.",
    )


class NetworkAnalysisOptionEnum(StringChoiceEnum):
    """Available network analysis measures."""

    DENSITY = ("density", "The density of a graph.")
    AVERAGE_CLUSTERING = ("average_clustering", "The average clustering of a graph.")
    SUMMARY = (
        "summary",
        "The summary includes the number of nodes and edges, or neighbours for a single node.",
    )


class SourceSelectionOptionEnum(StringChoiceEnum):
    """Available source selection methods."""

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
    PAGE_RANK = (
        "page_rank",
        "Select sources according to PageRank centrality metrics.",
    )


class SourceDetectionAlgorithm(StringChoiceEnum):
    """Available source detection methods."""

    DYNAMIC_AGE = ("dynamic_age", "Find sources with dynamic age algorithm.")
    NET_SLEUTH = ("net_sleuth", "Find sources with NetSleuth algorithm.")
    RUMOR_CENTER = ("rumor_center", "Find sources with Rumor center algorithm.")
    CENTRALITY_BASED = ("centrality", "Find sources with centrality based algorithm.")
    MULTIPLE_CENTRALITY_BASED = (
        "multiple_centrality",
        "Find sources with multiple centrality based algorithm.",
    )
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
    JORDAN_CENTER = ("jordan_center", "Find sources with Jordan center algorithm.")
