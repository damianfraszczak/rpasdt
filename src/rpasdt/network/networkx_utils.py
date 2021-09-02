from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib as mpl
from ndlib.models import DiffusionModel
from networkx import Graph

from rpasdt.algorithm.taxonomies import GraphTypeEnum
from rpasdt.gui.dynamic_form.models import DynamicFormConfig, FormFieldConfig
from rpasdt.gui.form_utils import (
    GraphTypeToFormFieldsConfigMap,
    get_diffusion_model_fields_config,
)
from rpasdt.gui.utils import show_dynamic_dialog
from rpasdt.network.models import NodeAttribute


def get_graph_form_field_config(
    graph_type: GraphTypeEnum,
) -> Dict[str, FormFieldConfig]:
    return GraphTypeToFormFieldsConfigMap.get(graph_type, {})


def get_graph_default_properties(graph_type: GraphTypeEnum) -> Dict[str, Any]:
    return {
        field_name: field_config.default_value
        for field_name, field_config in get_graph_form_field_config(graph_type).items()
    }


def get_diffusion_model_default_properties(
    diffusion_model: DiffusionModel,
) -> Dict[str, Any]:
    return {
        field_name: field_config.default_value
        for field_name, field_config in get_diffusion_model_fields_config(
            diffusion_model
        ).items()
    }


def create_node_network_dict(
    graph: Graph,
    data_key: str,
    default_val: Optional[Any] = None,
    skip_empty: bool = False,
):
    return {
        n: graph.nodes[n].get(data_key, default_val) or ("" if skip_empty else n)
        for n in graph.nodes
    }


def create_node_network_array(
    graph: Graph, data_key: str, default_val: Optional[Any] = None
):
    return [graph.nodes[n].get(data_key, default_val) or n for n in graph.nodes]


def map_networkx_communities_to_dict(communities: Tuple[Set[int]]) -> Dict[int, int]:
    return {
        index + 1: community for index, community in enumerate(map(sorted, communities))
    }


def get_community_index(community):
    return community - 1


def get_grouped_nodes(data):
    grouped_nodes = {}
    for community, nodes in data.items():
        for node in nodes:
            grouped_nodes[node] = community
    return grouped_nodes


def get_nodes_color(nodes_value: List[int], cmap=None):
    low, *_, high = sorted(nodes_value) if len(nodes_value) > 1 else (1, 1)
    norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    return [mapper.to_rgba(i) for i in nodes_value]


def set_node_attributes(graph: Graph, attributes: List[NodeAttribute], nodes_list=None):
    nodes_list = nodes_list or list(graph.nodes())
    for node_index in nodes_list:
        for attr in attributes:
            graph.nodes[node_index][attr.key] = attr.value


def show_graph_config_dialog(
    graph_type: GraphTypeEnum, graph_type_properties: Dict
) -> Optional[Dict]:
    if graph_type_properties:
        graph_type_properties = show_dynamic_dialog(
            object=graph_type_properties,
            config=DynamicFormConfig(
                field_config=get_graph_form_field_config(graph_type),
                title=f"Edit {graph_type.label} properties",
            ),
        )
    return graph_type_properties
