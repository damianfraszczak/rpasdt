"""Models."""
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

from dataclasses_json import dataclass_json
from networkx import Graph

from rpasdt.algorithm.taxonomies import (
    DiffusionGraphNodeRenderTypeEnum,
    DiffusionTypeEnum,
    GraphDataFormatEnum,
    GraphLayout,
    GraphTypeEnum,
)
from rpasdt.model.constants import (
    ESTIMATED_SOURCE_NODE_COLOR,
    INFECTED_NODE_COLOR,
    NODE_COLOR,
    NODE_LABEL_COLOR,
    NODE_SIZE,
    RECOVERED_NODE_COLOR,
    SOURCE_NODE_COLOR,
)
from rpasdt.network.taxonomies import NodeAttributeEnum


@dataclass
class GraphConfig:
    """The graph rendering configuration."""

    node_color: str = field(default=NODE_COLOR)
    node_size: int = field(default=NODE_SIZE)
    node_label_font_color: str = field(default=NODE_LABEL_COLOR)
    display_node_labels: bool = field(default=True)
    display_node_extra_labels: bool = field(default=True)
    graph_position: Optional[Dict[int, Tuple]] = field(default=None)
    graph_layout: GraphLayout = field(default=GraphLayout.SPRING)
    # diffusion
    graph_node_rendering_type: DiffusionGraphNodeRenderTypeEnum = field(
        default=DiffusionGraphNodeRenderTypeEnum.FULL
    )
    source_node_color: str = field(default=SOURCE_NODE_COLOR)
    recovered_node_color: str = field(default=RECOVERED_NODE_COLOR)
    susceptible_node_color: str = field(default=NODE_COLOR)
    infected_node_color: str = field(default=INFECTED_NODE_COLOR)
    estimated_source_node_color: str = field(default=ESTIMATED_SOURCE_NODE_COLOR)

    def clone(self):
        return copy(self)

    def clear(self):
        self.graph_position = None


@dataclass
class Experiment:
    """The initial configured situation."""

    name: str = field(default="Experiment")
    graph_type: GraphTypeEnum = field(default=GraphTypeEnum.WATTS_STROGATZ)
    graph_type_properties: Dict = field(default_factory=dict)
    graph: Graph = field(default_factory=Graph)
    graph_config: GraphConfig = field(default_factory=GraphConfig)


@dataclass_json
@dataclass
class ExperimentExportModel:
    name: str
    graph_type: GraphTypeEnum
    # @dataclass_json has a problem with parsing when type Dict is used here
    graph_type_properties: Any
    graph_data: str
    graph_data_format: GraphDataFormatEnum
    graph_config: GraphConfig


class SimulationStep:
    start: Union[int, float]
    end: Union[int, float]
    range: Union[int, float] = 1


@dataclass
class DiffusionExperiment:
    source_graph: Graph = field(default_factory=Graph)
    diffusion_graph: Graph = field(default_factory=Graph)
    graph_config: GraphConfig = field(default_factory=GraphConfig)
    diffusion_type: DiffusionTypeEnum = field(default=DiffusionTypeEnum.SI)
    diffusion_model_properties: Dict = field(default_factory=dict)
    diffusion_iteration_bunch: int = field(default=200)

    @property
    def source_nodes(self):
        return [
            node_index
            for node_index, data in self.source_graph.nodes(data=True)
            if data.get(NodeAttributeEnum.SOURCE)
        ]
