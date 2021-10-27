"""Common source detection methods."""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

from networkx import Graph

from rpasdt.algorithm.communities import find_communities
from rpasdt.algorithm.models import (
    CommunitiesBasedSourceDetectionConfig,
    SingleSourceDetectionEvaluation,
    SourceDetectionConfig,
)
from rpasdt.algorithm.source_detection_evaluation import (
    compute_source_detection_evaluation,
)
from rpasdt.common.utils import camel_case_split, sort_dict_by_value


class SourceDetector(ABC):
    CONFIG_CLASS = SourceDetectionConfig

    def __init__(
        self, G: Graph, IG: Graph, config: Optional[SourceDetectionConfig] = None
    ):
        self.G = G
        self.IG = IG
        self._config = config

    @property
    def config(self):
        if not self._config:
            self._config = self.create_config()
        return self._config

    def create_config(self):
        return self.CONFIG_CLASS()

    @abstractmethod
    def estimate_sources(self) -> Dict[int, Union[int, Dict[int, float]]]:
        pass

    def process_estimation(self, result: Dict[int, float]):
        return sort_dict_by_value(result)[: self.config.number_of_sources]

    @cached_property
    def detected_sources(self) -> Union[int, List[int]]:
        result = [source for source, _ in self.detected_sources_estimation]
        return result[0] if len(result) == 1 else result

    @cached_property
    def detected_sources_estimation(self) -> List[Tuple[int, float]]:
        return self.process_estimation(self.estimate_sources())

    def evaluate_sources(
        self, real_sources: List[int]
    ) -> SingleSourceDetectionEvaluation:
        return compute_source_detection_evaluation(
            G=self.IG, real_sources=real_sources, detected_sources=self.detected_sources
        )

    def __str__(self) -> str:
        return " ".join(camel_case_split(self.__class__.__name__))


class CommunityBasedSourceDetector(SourceDetector, ABC):
    CONFIG_CLASS = CommunitiesBasedSourceDetectionConfig

    def __init__(
        self,
        G: Graph,
        IG: Graph,
        config: Optional[CommunitiesBasedSourceDetectionConfig] = None,
    ):
        super().__init__(G, IG, config)

    @cached_property
    def communities(self) -> Dict[int, List[int]]:

        return (
            {0: self.IG}
            if self.config.number_of_sources == 1
            else find_communities(
                type=self.config.communities_algorithm,
                graph=self.IG,
                number_communities=self.config.number_of_sources,
            )
        )

    def estimate_sources(self) -> Dict[int, Dict[int, float]]:
        return {
            cluster: self.find_sources_in_community(self.IG.subgraph(nodes))
            for cluster, nodes in self.communities.items()
        }

    @abstractmethod
    def find_sources_in_community(self, graph: Graph):
        pass

    def process_estimation(self, result: Dict[int, float]) -> Union[int, List[int]]:
        # in one community there is exactly one source
        # [0] get first from sorted, [0] get node index
        return [
            sort_dict_by_value(nodes_dict)[0]
            for cluster, nodes_dict in self.estimate_sources().items()
        ]
