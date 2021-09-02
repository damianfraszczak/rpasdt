from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from networkx import Graph

from rpasdt.algorithm.centralities import compute_centrality
from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.network_analysis import compute_network_analysis
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    NetworkAnalysisOptionEnum,
)
from rpasdt.common.enums import StringChoiceEnum
from rpasdt.common.utils import format_label
from rpasdt.controller.graph import GraphController
from rpasdt.gui.analysis.analysis import AnalysisDialog
from rpasdt.gui.analysis.centrality import CentralityGraphPanel
from rpasdt.gui.analysis.communities import CommunityGraphPanel
from rpasdt.gui.analysis.models import AnalysisData, CommunityAnalysisData
from rpasdt.gui.dynamic_form.models import DynamicFormConfig, FormFieldConfig
from rpasdt.gui.form_utils import CommunityTypeToFormFieldsConfigMap
from rpasdt.gui.utils import (
    run_long_task,
    show_alert_dialog,
    show_dynamic_dialog,
)
from rpasdt.model.experiment import GraphConfig

ANALYSIS_HANDLER_PREFIX = "handler_analysis_"


class AnalysisGraphController(GraphController):
    def __init__(
        self,
        window: "MainWindow",
        graph: Graph,
        graph_config: GraphConfig,
        data: AnalysisData,
    ):
        super().__init__(window, graph, graph_config)
        self.data = data


@dataclass
class AnalysisEnumHandler:
    enum: Type[StringChoiceEnum]
    alg_executor: Callable
    title: str
    result_handler: Callable
    result_handler_per_type: Optional[Dict[str, Callable]] = None
    alg_form_field_config: Dict[str, Optional[Dict[str, FormFieldConfig]]] = None


class AnalysisControllerMixin:
    def __init__(self) -> None:
        super().__init__()
        self.handlers: Dict[str, AnalysisEnumHandler] = {}

    def __getattribute__(self, name: str) -> Any:

        if ANALYSIS_HANDLER_PREFIX in name:
            full_alg_name = name.replace(ANALYSIS_HANDLER_PREFIX, "")
            enum_name, alg_name = full_alg_name.split("_", 1)
            handlers = super().__getattribute__("handlers")
            handler_info = handlers.get(enum_name)
            if handler_info:
                alg_enum = handler_info.enum[alg_name]
                return super().__getattribute__("_configure_and_run_alg")(
                    handler_info=handler_info, alg_enum=alg_enum
                )
        return super().__getattribute__(name)

    def _configure_and_run_alg(
        self, handler_info: AnalysisEnumHandler, alg_enum: StringChoiceEnum
    ):

        title = format_label(alg_enum.value)
        label = format_label(title, handler_info.title)

        def _run_task():
            alg_kwargs = {}
            alg_form_config = (
                handler_info.alg_form_field_config.get(alg_enum)
                if handler_info.alg_form_field_config
                else None
            )
            if alg_form_config:
                form_config = DynamicFormConfig(
                    field_config=alg_form_config,
                    title=f"Configure {title}",
                )
                alg_kwargs = show_dynamic_dialog(object=alg_kwargs, config=form_config)
            alg_impl = lambda graph: handler_info.alg_executor(  # noqa: E731
                graph=graph, type=alg_enum, **alg_kwargs
            )
            return run_long_task(
                title=f"Computing {label}",
                function=alg_impl,
                function_args=[self.graph],
                callback=lambda data: handler_info.result_handler(
                    title=title, data=data
                ),
            )

        return _run_task

    def _get_handler_key(self, enum: Type[StringChoiceEnum]):
        return enum.__name__

    def add_handler(self, handler_info: AnalysisEnumHandler):
        key = self._get_handler_key(handler_info.enum)
        self.handlers[key] = handler_info

    def show_community_result(self, data, title, *args, **kwargs):
        self.show_dialog(
            AnalysisDialog(
                controller=AnalysisGraphController(
                    window=self.window,
                    graph=self.graph,
                    graph_config=self.graph_config,
                    data=CommunityAnalysisData(graph=self.graph, communities_data=data),
                ),
                title=title,
                graph_panel=CommunityGraphPanel,
            )
        )

    def show_analysis_result(self, data, title, *args, **kwargs):
        self.show_dialog(
            AnalysisDialog(
                controller=AnalysisGraphController(
                    window=self.window,
                    graph=self.graph,
                    graph_config=self.graph_config,
                    data=AnalysisData(graph=self.graph, nodes_data=data),
                ),
                title=title,
                graph_panel=CentralityGraphPanel,
            )
        )

    def show_value_result(self, data, title, *args, **kwargs):
        show_alert_dialog(title=title, text=f"{data}")


class CommunityAnalysisControllerMixin(AnalysisControllerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.add_handler(
            AnalysisEnumHandler(
                enum=CommunityOptionEnum,
                alg_executor=find_communities,
                title="Community",
                result_handler=self.show_community_result,
                alg_form_field_config=CommunityTypeToFormFieldsConfigMap,
            )
        )


class CentralityAnalysisControllerMixin(AnalysisControllerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.add_handler(
            AnalysisEnumHandler(
                enum=CentralityOptionEnum,
                alg_executor=compute_centrality,
                title="Centrality",
                result_handler=self.show_analysis_result,
            )
        )


class NetworkAnalysisControllerMixin(AnalysisControllerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.add_handler(
            AnalysisEnumHandler(
                enum=NetworkAnalysisOptionEnum,
                alg_executor=compute_network_analysis,
                title="Network",
                result_handler=self.show_value_result,
            )
        )
