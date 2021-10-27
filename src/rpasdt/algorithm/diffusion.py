"""Diffusion models configuration."""
import random
from typing import Dict, List, Optional

from ndlib.models import DiffusionModel
from ndlib.models.epidemics import (
    GeneralisedThresholdModel,
    IndependentCascadesModel,
    KerteszThresholdModel,
    SEIRModel,
    SIModel,
    SIRModel,
    SISModel,
    SWIRModel,
    ThresholdModel,
)
from ndlib.models.ModelConfig import Configuration
from ndlib.models.opinions import (
    MajorityRuleModel,
    QVoterModel,
    SznajdModel,
    VoterModel,
)
from networkx import Graph

from rpasdt.algorithm.taxonomies import (
    DiffusionTypeEnum,
    NodeStatusEnum,
    NodeStatusToValueMapping,
)
from rpasdt.common.utils import eval_if_str

DiffusionTypeToDiffusionModelMap = {
    DiffusionTypeEnum.SI: SIModel,
    DiffusionTypeEnum.SIS: SISModel,
    DiffusionTypeEnum.SIR: SIRModel,
    DiffusionTypeEnum.SEIR: SEIRModel,
    DiffusionTypeEnum.SWIR: SWIRModel,
    DiffusionTypeEnum.THRESHOLD: ThresholdModel,
    DiffusionTypeEnum.KERTESZ_THRESHOLD: KerteszThresholdModel,
    DiffusionTypeEnum.GENERALISED_THRESHOLD: GeneralisedThresholdModel,
    DiffusionTypeEnum.INDEPENDENT_CASCADES: IndependentCascadesModel,
    DiffusionTypeEnum.VOTER: VoterModel,
    DiffusionTypeEnum.Q_VOTER: QVoterModel,
    DiffusionTypeEnum.MAJORITY_RULE: MajorityRuleModel,
    DiffusionTypeEnum.SZNAJD: SznajdModel,
}
# ndlib diffusion model configuration kwargs
NDLIB_MODEL_KWARG = "model"
NDLIB_RANGE_KWARG = "range"
NDLIB_DEFAULT_KWARG = "default"


def get_diffusion_model_default_params(diffusion_model: DiffusionModel) -> Dict:
    """Return default configuration for provided diffusion model."""
    parameters = diffusion_model.get_model_parameters().get(NDLIB_MODEL_KWARG)
    result = {}
    for field, details in parameters.items():
        range = eval_if_str(details.get(NDLIB_RANGE_KWARG))
        default_val = details.get(
            NDLIB_DEFAULT_KWARG, random.uniform(range[0], range[1]) if range else 0
        )
        result[field] = default_val
    return result


def get_and_init_diffusion_model(
    graph: Graph,
    diffusion_type: DiffusionTypeEnum,
    source_nodes: List[int],
    model_params: Optional[Dict] = None,
):
    """Create and initialize diffusion model based on provided config."""
    diffusion_model = DiffusionTypeToDiffusionModelMap[diffusion_type](graph)
    config = Configuration()
    model_params = model_params or get_diffusion_model_default_params(diffusion_model)
    for key, value in model_params.items():
        config.add_model_parameter(param_name=key, param_value=value)
    config.add_model_initial_configuration(NodeStatusEnum.INFECTED, source_nodes)
    diffusion_model.set_initial_status(config)
    return diffusion_model, model_params


def get_nodes_by_diffusion_status(
    diffusion_model: DiffusionModel = None,
    node_status: NodeStatusEnum = NodeStatusEnum.INFECTED,
) -> List[int]:
    """Return nodes with required status."""
    return (
        [
            key
            for key, value in diffusion_model.status.items()
            if value == NodeStatusToValueMapping[node_status]
        ]
        if diffusion_model
        else []
    )
