from ndlib.models.epidemics import SIModel

from rpasdt.algorithm.taxonomies import DiffusionTypeEnum

DiffusionTypeToDiffusionModelMap = {DiffusionTypeEnum.SI: SIModel}
