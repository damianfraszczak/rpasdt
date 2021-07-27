from dataclasses import dataclass
from typing import Optional

from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    CommunityOptionEnum,
    SourceSelectionOptionEnum,
)


@dataclass
class NetworkSourceSelectionConfig:
    number_of_sources: int = 1
    algorithm: SourceSelectionOptionEnum = SourceSelectionOptionEnum.RANDOM


@dataclass
class SourceDetectionConfig:
    number_of_sources: Optional[int] = 1


@dataclass
class CentralityBasedSourceDetectionConfig(SourceDetectionConfig):
    centrality_algorithm: CentralityOptionEnum = CentralityOptionEnum.DEGREE


@dataclass
class UnbiasedCentralityBasedSourceDetectionConfig(
    CentralityBasedSourceDetectionConfig
):
    r: float = 0.85


@dataclass
class CommunitiesBasedSourceDetectionConfig(SourceDetectionConfig):
    communities_algorithm: CommunityOptionEnum = CommunityOptionEnum.GIRVAN_NEWMAN


@dataclass
class CentralityCommunityBasedSourceDetectionConfig(
    CommunitiesBasedSourceDetectionConfig
):
    centrality_algorithm: CentralityOptionEnum = CentralityOptionEnum.DEGREE


@dataclass
class UnbiasedCentralityCommunityBasedSourceDetectionConfig(
    CentralityCommunityBasedSourceDetectionConfig
):
    r: float = 0.85


@dataclass
class SingleSourceDetectionEvaluation:
    error_distance: int
    TP: int
    FP: int
    FN: int


@dataclass
class ExperimentSourceDetectionEvaluation:
    avg_error_distance: int
    recall: float
    precision: float
    f1score: float
