from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT

from rpasdt.gui.toolbar.toolbar_items import (
    create_analysis_action,
    create_diffusion_action,
    create_diffusion_plots_action,
    create_diffusion_simulation_actions,
    create_edit_diffusion_action,
    create_edit_graph_config_action,
    create_export_experiment_action,
    create_source_detection_action,
    create_source_selection_action,
)


class DefaultNetworkGraphToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent, coordinates=False):
        super().__init__(canvas, parent, coordinates)


class AnalysisNetworkGraphToolbar(DefaultNetworkGraphToolbar):
    def __init__(self, canvas, parent, toolbar_handler=None):
        super().__init__(canvas, parent)

        self.addAction(
            create_edit_graph_config_action(parent=self, handler=toolbar_handler)
        )
        self.addAction(create_analysis_action(parent=self, handler=toolbar_handler))


class MainNetworkGraphToolbar(AnalysisNetworkGraphToolbar):
    def __init__(self, canvas, parent, toolbar_handler):
        super().__init__(canvas, parent, toolbar_handler)
        self.addAction(
            create_export_experiment_action(parent=self, handler=toolbar_handler)
        )
        self.addAction(
            create_source_selection_action(parent=self, handler=toolbar_handler)
        )
        self.addAction(create_diffusion_action(parent=self, handler=toolbar_handler))


class DiffusionNetworkGraphToolbar(AnalysisNetworkGraphToolbar):
    def __init__(self, canvas, parent, toolbar_handler):
        super().__init__(canvas, parent, toolbar_handler)
        self.addAction(
            create_edit_diffusion_action(parent=self, handler=toolbar_handler)
        )
        self.addActions(
            create_diffusion_simulation_actions(parent=self, controller=toolbar_handler)
        )
        self.addAction(
            create_diffusion_plots_action(parent=self, controller=toolbar_handler)
        )
        self.addAction(
            create_source_detection_action(parent=self, handler=toolbar_handler)
        )


class SourceDetectionGraphToolbar(AnalysisNetworkGraphToolbar):
    def __init__(self, canvas, parent, toolbar_handler=None):
        super().__init__(canvas, parent, toolbar_handler)
