"""Models."""
from copy import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from networkx import Graph

from rpasdt.algorithm.taxonomies import (
    DiffusionGraphNodeRenderTypeEnum,
    DiffusionTypeEnum,
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

    node_color: str = NODE_COLOR
    node_size: int = NODE_SIZE
    node_label_font_color: str = NODE_LABEL_COLOR
    display_node_labels: bool = True
    display_node_extra_labels: bool = True
    graph_position: Dict[int, Tuple] = None
    graph_layout: GraphLayout = GraphLayout.SPRING
    # diffusion
    graph_node_rendering_type: DiffusionGraphNodeRenderTypeEnum = (
        DiffusionGraphNodeRenderTypeEnum.FULL
    )
    source_node_color: str = SOURCE_NODE_COLOR
    recovered_node_color: str = RECOVERED_NODE_COLOR
    susceptible_node_color: str = NODE_COLOR
    infected_node_color: str = INFECTED_NODE_COLOR
    estimated_source_node_color: str = ESTIMATED_SOURCE_NODE_COLOR

    def clone(self):
        return copy(self)

    def clear(self):
        self.graph_position = None


@dataclass
class DiffusionExperiment:
    source_graph: Graph
    diffusion_graph: Graph = Graph()
    graph_config: GraphConfig = GraphConfig()
    diffusion_type: DiffusionTypeEnum = DiffusionTypeEnum.SI
    diffusion_model_properties: Optional[Dict] = None
    diffusion_iteration_bunch: int = 200

    @property
    def source_nodes(self):
        return [
            node_index
            for node_index, data in self.source_graph.nodes(data=True)
            if data.get(NodeAttributeEnum.SOURCE)
        ]


@dataclass
class Experiment:
    """The initial configured situation."""

    name: str = "Karate club simulation"
    graph_type: GraphTypeEnum = GraphTypeEnum.KARATE_CLUB
    graph_type_properties: Dict = None
    graph: Graph = None
    graph_config: GraphConfig = GraphConfig()
