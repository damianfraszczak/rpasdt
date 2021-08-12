import random
from typing import Dict

from ndlib.models import DiffusionModel

from rpasdt.algorithm.taxonomies import GraphDataFormatEnum, GraphTypeEnum
from rpasdt.common.utils import eval_if_str
from rpasdt.gui.dynamic_form.models import (
    DynamicFormConfig,
    FieldInputType,
    FormFieldConfig,
)
from rpasdt.gui.dynamic_form.utils import format_field_label
from rpasdt.model.constants import NODE_COLOR, NODE_LABEL_COLOR, NODE_SIZE
from rpasdt.network.taxonomies import NodeAttributeEnum

GRAPH_CONFIG_FIELD_CONFIG = {
    "node_color": FormFieldConfig(
        field_name="node_color",
        label=format_field_label("Node color"),
        type=FieldInputType.COLOR,
        default_value=NODE_COLOR,
    ),
    "node_label_font_color": FormFieldConfig(
        field_name="node_label_font_color",
        label=format_field_label("Node label color"),
        type=FieldInputType.COLOR,
        default_value=NODE_LABEL_COLOR,
    ),
}

GraphTypeToFormFieldsConfigMap = {
    GraphTypeEnum.WATTS_STROGATZ: {
        "n": FormFieldConfig(
            field_name="n",
            default_value=100,
            type=FieldInputType.INTEGER,
            help_text="The number of nodes",
        ),
        "k": FormFieldConfig(
            field_name="k",
            default_value=4,
            type=FieldInputType.INTEGER,
            help_text="Each node is connected to k nearest neighbors in ring topology",
        ),
        "p": FormFieldConfig(
            field_name="p",
            default_value=0.5,
            type=FieldInputType.DOUBLE,
            help_text="The probability of rewiring each edge",
        ),
    },
    GraphTypeEnum.BALANCED_TREE: {
        "r": FormFieldConfig(
            field_name="r",
            default_value=2,
            type=FieldInputType.INTEGER,
            help_text="Branching factor of the tree; each node will have r children",
        ),
        "h": FormFieldConfig(
            field_name="h",
            default_value=5,
            type=FieldInputType.INTEGER,
            help_text="Height of the tree",
        ),
    },
    GraphTypeEnum.COMPLETE: {
        "n": FormFieldConfig(
            field_name="n",
            default_value=5,
            type=FieldInputType.INTEGER,
            help_text="If n is an integer, nodes are from range(n). If n is a container of nodes, those nodes appear in the graph",
        ),
    },
    GraphTypeEnum.ERDOS_RENYI: {
        "n": FormFieldConfig(
            field_name="n",
            default_value=5,
            type=FieldInputType.INTEGER,
            help_text="The number of nodes.",
            inner_type=int,
        ),
        "p": FormFieldConfig(
            field_name="p",
            default_value=0.5,
            type=FieldInputType.DOUBLE,
            help_text="Probability for edge creation",
            inner_type=float,
        ),
        "seed": FormFieldConfig(
            field_name="seed",
            default_value=100,
            type=FieldInputType.INTEGER,
            help_text="Indicator of random number generation state",
            inner_type=int,
        ),
    },
    GraphTypeEnum.CAVEMAN_GRAPH: {
        "l": FormFieldConfig(
            field_name="l",
            default_value=5,
            type=FieldInputType.INTEGER,
            help_text="Cliques number",
            inner_type=int,
        ),
        "k": FormFieldConfig(
            field_name="k",
            default_value=3,
            type=FieldInputType.INTEGER,
            help_text="Cliques size",
            inner_type=int,
        ),
    },
    GraphTypeEnum.CONNECTED_CAVEMAN_GRAPH: {
        "l": FormFieldConfig(
            field_name="l",
            default_value=5,
            type=FieldInputType.INTEGER,
            help_text="Cliques number",
            inner_type=int,
        ),
        "k": FormFieldConfig(
            field_name="k",
            default_value=3,
            type=FieldInputType.INTEGER,
            help_text="Cliques size",
            inner_type=int,
        ),
    },
    GraphTypeEnum.CUSTOM: {
        "graph_data_format": FormFieldConfig(
            field_name="graph_data_format",
            default_value=GraphDataFormatEnum.MULTILINE_ADJLIST,
            type=FieldInputType.COMBOBOX,
            help_text="Graph data format",
            options=GraphDataFormatEnum.choices,
            inner_type=GraphDataFormatEnum,
        ),
        "file_path": FormFieldConfig(
            field_name="file_path",
            type=FieldInputType.FILE,
            help_text="Graph file path",
            inner_type=str,
        ),
    },
}

NodeAttributeFormFieldsConfig = {
    NodeAttributeEnum.COLOR: FormFieldConfig(
        field_name=NodeAttributeEnum.COLOR,
        default_value=NODE_COLOR,
        type=FieldInputType.COLOR,
        label="Node color",
    ),
    NodeAttributeEnum.SIZE: FormFieldConfig(
        field_name=NodeAttributeEnum.SIZE,
        default_value=NODE_SIZE,
        type=FieldInputType.INTEGER,
        label="Node size",
    ),
    NodeAttributeEnum.EXTRA_LABEL: FormFieldConfig(
        field_name=NodeAttributeEnum.EXTRA_LABEL,
        type=FieldInputType.SINGLE_TEXT,
        label="Node extra label",
    ),
    NodeAttributeEnum.LABEL: FormFieldConfig(
        field_name=NodeAttributeEnum.LABEL,
        type=FieldInputType.SINGLE_TEXT,
        label="Node name",
    ),
    NodeAttributeEnum.SOURCE: FormFieldConfig(
        field_name=NodeAttributeEnum.SOURCE,
        type=FieldInputType.CHECKBOX,
        label="Is source ?",
    ),
}


def get_graph_form_config(graph_type: GraphTypeEnum):
    return DynamicFormConfig(
        field_config=GraphTypeToFormFieldsConfigMap[graph_type],
        title=f"Configure {GraphTypeEnum[graph_type]}",
    )


def get_node_edit_config(node_index: int):
    return DynamicFormConfig(
        title=f"Edit node {node_index}", field_config=NodeAttributeFormFieldsConfig
    )


def get_diffusion_model_fields_config(
    diffusion_model: DiffusionModel,
) -> Dict[str, FormFieldConfig]:
    parameters = diffusion_model.get_model_parameters().get("model")
    result = {}
    for field, details in parameters.items():
        range = eval_if_str(details.get("range"))
        default_val = details.get(
            "default", random.uniform(range[0], range[1]) if range else 0
        )
        result[field] = FormFieldConfig(
            field_name=field,
            range=range,
            type=FieldInputType.DOUBLE,
            default_value=default_val,
        )
    return result


def get_diffusion_model_form_config(diffusion_model: DiffusionModel, title: str = None):
    if not title:
        title = f"Edit {diffusion_model.name} diffusion model params"
    return DynamicFormConfig(
        title=title,
        field_config=get_diffusion_model_fields_config(diffusion_model=diffusion_model),
    )
