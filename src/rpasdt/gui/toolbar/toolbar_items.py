from functools import partial
from typing import Dict, Tuple, Union

from PyQt5.QtWidgets import QApplication, QMenu, QStyle

from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    NetworkAnalysisOptionEnum,
    SourceDetectionAlgorithm,
)
from rpasdt.common.enums import StringChoiceEnum
from rpasdt.common.utils import format_label
from rpasdt.gui.utils import create_action


def _get_toolbar_choices(enum: StringChoiceEnum):
    return {
        f"{enum.__class__.__name__}_{enum.name}": (format_label(enum.value), enum.label)
        for enum in enum
    }


# TODO Link Prediction
ANALYSIS_HANDLER_PREFIX = "handler_analysis_"
CENTRALITY_OPTIONS = _get_toolbar_choices(CentralityOptionEnum)
COMMUNITY_OPTIONS = _get_toolbar_choices(CommunityOptionEnum)

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

NETWORK_OPTIONS = _get_toolbar_choices(NetworkAnalysisOptionEnum)


def _create_actions(
    definition_map: Union[Dict, Tuple],
    handler: "GraphController",
    method_prefix="handler",
    parent=None,
):
    return [
        create_action(
            title=title,
            tooltip=tooltip,
            handler=getattr(handler, f"{method_prefix}{method_name}", None),
            parent=parent,
        )
        for method_name, (title, tooltip) in definition_map.items()
    ]


def _create_menu(
    title: str, definition_map: Union[Dict, Tuple], handler, method_prefix, parent
):
    menu = QMenu(title, parent)
    menu.addActions(_create_actions(definition_map, handler, method_prefix, parent))
    menu.setTitle(title)
    return menu


def create_analysis_action(parent, handler):
    action = create_action(
        title="Analysis",
        tooltip="Network analysis tools",
        icon=QApplication.style().standardIcon(QStyle.SP_ComputerIcon),
        parent=parent,
    )
    menu = QMenu("Analysis", parent=parent)
    action.setMenu(menu)
    # centralities

    menu.addMenu(
        _create_menu(
            title="Centrality analysis",
            definition_map=CENTRALITY_OPTIONS,
            handler=handler,
            method_prefix=ANALYSIS_HANDLER_PREFIX,
            parent=parent,
        )
    )
    menu.addSeparator()
    menu.addMenu(
        _create_menu(
            title="Community analysis",
            definition_map=COMMUNITY_OPTIONS,
            handler=handler,
            method_prefix=ANALYSIS_HANDLER_PREFIX,
            parent=parent,
        )
    )
    menu.addSeparator()
    menu.addMenu(
        _create_menu(
            title="Network analysis",
            definition_map=NETWORK_OPTIONS,
            handler=handler,
            method_prefix=ANALYSIS_HANDLER_PREFIX,
            parent=parent,
        )
    )
    menu.addSeparator()
    return action


def create_diffusion_action(parent, handler):
    action = create_action(
        title="Diffusion",
        tooltip="Simulate rumour diffusion in the network",
        icon=QApplication.style().standardIcon(QStyle.SP_MediaVolume),
        parent=parent,
        handler=handler.handler_create_diffusion,
    )
    return action


def create_edit_diffusion_action(parent, handler):
    action = create_action(
        title="Edit diffusion",
        tooltip="Edit diffusion model parameters",
        icon=QApplication.style().standardIcon(QStyle.SP_MediaVolume),
        parent=parent,
        handler=handler.handler_edit_diffusion,
    )
    return action


def create_source_detection_action(parent, handler):
    action = create_action(
        title="Source detection",
        tooltip="Source detection algorithms",
        icon=QApplication.style().standardIcon(QStyle.SP_DialogHelpButton),
        parent=parent,
    )
    menu = QMenu("Source detection", parent=parent)
    action.setMenu(menu)
    for alg in SourceDetectionAlgorithm:
        title = " ".join(alg.value.split("_")).title()
        menu.addAction(
            create_action(
                title=title,
                tooltip=title,
                icon=QApplication.style().standardIcon(QStyle.SP_TrashIcon),
                parent=parent,
                handler=partial(handler.handler_configure_source_detection, alg),
            )
        )
    return action


def create_diffusion_plots_action(parent, controller):
    action = create_action(
        title="Plots",
        tooltip="Diffusion specific plots",
        icon=QApplication.style().standardIcon(QStyle.SP_DirIcon),
        parent=parent,
    )
    menu = QMenu("Plots", parent=parent)
    action.setMenu(menu)
    plots = {
        "Trends": controller.handler_plot_diffusion_trend,
        "Prevalence": controller.handler_plot_diffusion_prevalence,
    }
    for title, handler in plots.items():
        menu.addAction(
            create_action(
                title=title,
                tooltip=title,
                parent=parent,
                handler=handler,
            )
        )
    return action


def create_source_selection_action(parent, handler):
    action = create_action(
        title="Source selection",
        tooltip="Source selection",
        icon=QApplication.style().standardIcon(QStyle.SP_DialogNoButton),
        parent=parent,
    )
    menu = QMenu("Source selection", parent=parent)
    action.setMenu(menu)
    # centralities

    menu.addAction(
        create_action(
            title="Clear",
            tooltip="Clear all sources",
            icon=QApplication.style().standardIcon(QStyle.SP_TrashIcon),
            parent=parent,
            handler=handler.handler_clear_sources,
        )
    )
    menu.addAction(
        create_action(
            title="Select",
            tooltip="Select sources automatically",
            icon=QApplication.style().standardIcon(QStyle.SP_DialogOkButton),
            parent=parent,
            handler=handler.handler_select_sources,
        )
    )
    return action


def create_diffusion_simulation_actions(parent, controller: "DiffusionGraphController"):
    return [
        create_action(
            title="Clear",
            tooltip="Clear simulation",
            icon=QApplication.style().standardIcon(QStyle.SP_MediaStop),
            parent=parent,
            handler=controller.diffusion_clear_handler,
        ),
        create_action(
            title="Play",
            tooltip="Execute single iteration",
            icon=QApplication.style().standardIcon(QStyle.SP_MediaPlay),
            handler=controller.diffusion_execute_iteration_handler,
            parent=parent,
        ),
        create_action(
            title="Execute batch iterations",
            tooltip="Run simulation in batch mode",
            icon=QApplication.style().standardIcon(QStyle.SP_MediaSeekForward),
            handler=controller.diffusion_execute_iteration_bunch_handler,
            parent=parent,
        ),
        create_action(
            title="Execute simulation",
            tooltip="Perform the whole simulation process (All nodes infected)",
            icon=QApplication.style().standardIcon(QStyle.SP_MediaSkipForward),
            handler=controller.diffusion_execute_iteration_handler,
            parent=parent,
        ),
    ]


def create_edit_graph_config_action(parent, handler):
    return create_action(
        parent=parent,
        title="Edit graph config",
        tooltip="Edit graph config",
        handler=handler.handler_edit_graph_config,
        icon=QApplication.style().standardIcon(QStyle.SP_FileIcon),
    )


def create_export_experiment_action(parent, handler):
    return create_action(
        parent=parent,
        title="Export experiment",
        tooltip="Export experiment",
        handler=handler.handler_export_experiment,
        icon=QApplication.style().standardIcon(QStyle.SP_DriveFDIcon),
    )
