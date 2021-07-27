import random
from typing import Dict, List, Optional

from ndlib.models import DiffusionModel
from ndlib.models.epidemics import (
    IndependentCascadesModel,
    SEIRModel,
    SIModel,
    SIRModel,
    SISModel,
    SWIRModel,
    ThresholdModel,
)
from ndlib.models.ModelConfig import Configuration
from networkx import Graph

from rpasdt.algorithm.taxonomies import DiffusionTypeEnum, NodeStatusEnum
from rpasdt.common.utils import eval_if_str

DiffusionTypeToDiffusionModelMap = {
    DiffusionTypeEnum.SI: SIModel,
    DiffusionTypeEnum.SIS: SISModel,
    DiffusionTypeEnum.SIR: SIRModel,
    DiffusionTypeEnum.SEIR: SEIRModel,
    DiffusionTypeEnum.SWIR: SWIRModel,
    DiffusionTypeEnum.THRESHOLD: ThresholdModel,
    DiffusionTypeEnum.INDEPENDENT_CASCADES: IndependentCascadesModel,
}


def get_diffusion_model_default_params(diffusion_model: DiffusionModel) -> Dict:
    parameters = diffusion_model.get_model_parameters().get("model")
    result = {}
    for field, details in parameters.items():
        range = eval_if_str(details.get("range"))
        default_val = details.get(
            "default", random.uniform(range[0], range[1]) if range else 0
        )
        result[field] = default_val
    return result


def get_and_init_diffusion_model(
    graph: Graph,
    diffusion_type: DiffusionTypeEnum,
    source_nodes: List[int],
    model_params: Optional[Dict] = None,
):
    diffusion_model = DiffusionTypeToDiffusionModelMap[diffusion_type](graph)
    config = Configuration()
    model_params = model_params or get_diffusion_model_default_params(diffusion_model)
    for key, value in model_params.items():
        config.add_model_parameter(param_name=key, param_value=value)
    config.add_model_initial_configuration(NodeStatusEnum.INFECTED, source_nodes)
    diffusion_model.set_initial_status(config)
    return diffusion_model, model_params
