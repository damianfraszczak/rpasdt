from functools import partial

from PyQt5.QtWidgets import QMenu, QApplication, QStyle

from rpasdt.algorithm.taxonomies import CentralityOptionEnum, CommunityOptionEnum, \
    SourceDetectionAlgorithm
from rpasdt.gui.utils import create_action

# TODO Link Prediction

CENTRALITY_OPTIONS = {
    CentralityOptionEnum.DEGREE: (
    'Degree', 'Compute the degree centrality for nodes.'),
    CentralityOptionEnum.EIGENVECTOR: (
    'Eigenvector', 'Compute the eigenvector centrality for the graph G.'),
    CentralityOptionEnum.KATZ: (
    'Katz', 'Compute the Katz centrality for the nodes of the graph G.'),
    CentralityOptionEnum.CLOSENESS: (
    'Closeness', 'Compute closeness centrality for nodes.'),
    CentralityOptionEnum.BETWEENNESS: ('Betweenness',
                                       'Compute the shortest-path betweenness centrality for nodes.'),
    # CentralityOptionEnum.EDGE_BETWEENNESS: ('Edge betweenness', ('Compute betweenness centrality for edges.')),
    CentralityOptionEnum.HARMONIC: (
    'Harmonic Centrality', 'Compute harmonic centrality for nodes.'),
    CentralityOptionEnum.VOTE_RANK: (
        'Vote rank',
        'Select a list of influential nodes in a graph using VoteRank algorithm'),
    CentralityOptionEnum.PAGE_RANK: (
    'Page rank', 'Compute page rank centrality for nodes.'),
}

COMMUNITY_OPTIONS = {
    CommunityOptionEnum.BIPARTITION: (
        'Bipartitions',
        'Partition a graph into two blocks using the Kernighan–Lin algorithm.'),
    CommunityOptionEnum.LOUVAIN: (
    'Louvain', 'Find communities in graph using the Louvain method.'),
    CommunityOptionEnum.GIRVAN_NEWMAN: (
        'Girvan-Newman',
        'Finds communities in a graph using the Girvan–Newman method.'),
    CommunityOptionEnum.GREEDY_MODULARITY: (
        'Clauset-Newman-Moore greedy modularity',
        'Finds communities in a graph using the Clauset-Newman-Moore greedy modularity maximization.'),
    # CommunityOptionEnum.NAIVE_MODULARITY: (
    #     'Naive greedy modularity',
    #     'Find communities in graph using the greedy modularity maximization.'
    # ),
    CommunityOptionEnum.LABEL_PROPAGATION: (
        'Label propagation',
        'Generates community sets determined by label propagation'),
    CommunityOptionEnum.TREE: (
        'Tree partitioning',
        'Optimal partitioning of a weighted tree using the Lukes algorithm.'),
    CommunityOptionEnum.K_CLIQUE: ('K-Clique',
                                   'Find k-clique communities in graph using the percolation method.'),
    CommunityOptionEnum.K_CORE: (
    'K-Core', 'Finds core in a graph using the K-core method.'),
    CommunityOptionEnum.K_SHELL: (
    'K-Shell', 'Finds shell in a graph using the K-shell method.'),
    CommunityOptionEnum.K_CRUST: (
    'K-Crust', 'Finds crust in a graph using the K-shell method.'),
    CommunityOptionEnum.K_CORONA: (
    'K-Corona', 'Finds korona in a graph using the K-shell method.'),
    CommunityOptionEnum.K_MEANS: (
    'K-Means', 'Finds communities in a graph using the K-means method.'),
}
NETWORK_OPTIONS = {
    'bridge': ('Bridges', 'Generate all bridges in a graph.'),
    'cycle': ('Simple cycles',
              'Find simple cycles (elementary circuits) of a directed graph.'),
    'degree_assortativity': (
    'Degree assortativity', 'Compute degree assortativity of graph.'),
    'average_neighbor_degree': (
        'Average neighbor degree',
        'Returns the average degree of the neighborhood of each node.'),
    'k_nearest_neighbors': ('K-nearest neighbors',
                            'Compute the average degree connectivity of graph.'),
    'average_clustering': (
    'Average clustering', 'Compute the average clustering coefficient.')
}


def _create_actions(definition_map, handler: 'GraphController',
                    method_prefix='handler', parent=None):
    return [
        create_action(
            title=title,
            tooltip=tooltip,
            handler=getattr(handler, f'{method_prefix}_{method_name}', None),
            parent=parent
        ) for method_name, (title, tooltip) in definition_map.items()
    ]


def _create_menu(title, definition_map, handler, method_prefix, parent):
    menu = QMenu(title, parent)
    menu.addActions(
        _create_actions(definition_map, handler, method_prefix, parent))
    menu.setTitle(title)
    return menu


def create_analysis_action(parent, handler):
    action = create_action(
        title='Analysis',
        tooltip='Network analysis tools',
        icon=QApplication.style().standardIcon(QStyle.SP_ComputerIcon),
        parent=parent
    )
    menu = QMenu("Analysis", parent=parent)
    action.setMenu(menu)
    # centralities

    menu.addMenu(_create_menu(title="Centrality analysis",
                              definition_map=CENTRALITY_OPTIONS,
                              handler=handler,
                              method_prefix='handler_analysis_centrality',
                              parent=parent
                              ))
    menu.addSeparator()
    menu.addMenu(_create_menu(title="Community analysis",
                              definition_map=COMMUNITY_OPTIONS,
                              handler=handler,
                              method_prefix='handler_analysis_community',
                              parent=parent
                              ))
    # menu.addSeparator()
    # menu.addMenu(_create_menu(title="Network analysis",
    #                           definition_map=NETWORK_OPTIONS,
    #                           handler=handler,
    #                           method_prefix='handler_analysis_network',
    #                           parent=parent
    #                           ))
    # menu.addSeparator()
    return action


def create_diffusion_action(parent, handler):
    action = create_action(
        title='Diffusion',
        tooltip='Simulate rumour diffusion in the network',
        icon=QApplication.style().standardIcon(QStyle.SP_MediaVolume),
        parent=parent,
        handler=handler.handler_create_diffusion
    )
    return action


def create_edit_diffusion_action(parent, handler):
    action = create_action(
        title='Edit diffusion',
        tooltip='Edit diffusion model parameters',
        icon=QApplication.style().standardIcon(QStyle.SP_MediaVolume),
        parent=parent,
        handler=handler.handler_edit_diffusion
    )
    return action


def create_source_detection_action(parent, handler):
    action = create_action(
        title='Source detection',
        tooltip='Source detection algorithms',
        icon=QApplication.style().standardIcon(QStyle.SP_DialogHelpButton),
        parent=parent,
    )
    menu = QMenu("Source detection", parent=parent)
    action.setMenu(menu)
    for alg in SourceDetectionAlgorithm:
        menu.addAction(create_action(
            title=alg.title(),
            tooltip=alg.title(),
            icon=QApplication.style().standardIcon(QStyle.SP_TrashIcon),
            parent=parent,
            handler=partial(handler.handler_configure_source_detection, alg)
        ))
    return action


def create_source_selection_action(parent, handler):
    action = create_action(
        title='Source selection',
        tooltip='Source selection',
        icon=QApplication.style().standardIcon(QStyle.SP_DialogNoButton),
        parent=parent
    )
    menu = QMenu("Source selection", parent=parent)
    action.setMenu(menu)
    # centralities

    menu.addAction(create_action(
        title='Clear',
        tooltip='Clear all sources',
        icon=QApplication.style().standardIcon(QStyle.SP_TrashIcon),
        parent=parent,
        handler=handler.handler_clear_sources
    ))
    menu.addAction(create_action(
        title='Select',
        tooltip='Select sources automatically',
        icon=QApplication.style().standardIcon(QStyle.SP_DialogOkButton),
        parent=parent,
        handler=handler.handler_select_sources
    ))
    return action


def create_diffusion_simulation_actions(parent,
                                        controller: 'DiffusionGraphController'):
    return [
        create_action(
            title='Clear',
            tooltip='Clear simulation',
            icon=QApplication.style().standardIcon(QStyle.SP_MediaStop),
            parent=parent,
            handler=controller.diffusion_clear_handler,
        ),
        create_action(
            title='Play',
            tooltip='Execute single iteration',
            icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay),
            handler=controller.diffusion_execute_iteration_handler,
            parent=parent
        ),
        create_action(
            title='Execute batch iterations',
            tooltip='Run simulation in batch mode',
            icon=QApplication.style().standardIcon(QStyle.SP_MediaSeekForward),
            handler=controller.diffusion_execute_iteration_bunch_handler,
            parent=parent
        ),
        create_action(
            title='Execute simulation',
            tooltip='Perform the whole simulation process (All nodes infected)',
            icon=QApplication.style().standardIcon(QStyle.SP_MediaSkipForward),
            handler=controller.diffusion_execute_iteration_handler,
            parent=parent
        ),
    ]


def create_edit_graph_config_action(parent, handler):
    return create_action(
        parent=parent,
        title='Edit graph config',
        tooltip='Edit graph config',
        handler=handler.handler_edit_graph_config,
        icon=QApplication.style().standardIcon(QStyle.SP_FileIcon),

    )
