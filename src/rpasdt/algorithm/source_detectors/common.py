"""Common source detection methods."""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

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
from rpasdt.common.utils import (
    camel_case_split,
    normalize_dict_values,
    sort_dict_by_value,
)


class SourceDetector(ABC):
    CONFIG_CLASS = SourceDetectionConfig

    def __init__(
        self, G: Graph, IG: Graph, config: Optional[SourceDetectionConfig] = None
    ):
        self.G = G
        self.IG = IG
        self._config = config
        self.validate_config()

    def validate_config(self):
        if self.config.source_threshold and self.config.number_of_sources:
            raise Exception(
                "Only one of `source_threshold` or `number_of_sources` is " "required."
            )

    @property
    def config(self):
        if not self._config:
            self._config = self.create_config()
        return self._config

    def create_config(self):
        return self.CONFIG_CLASS()

    @abstractmethod
    def estimate_sources(self) -> Dict[int, Union[int, Dict[int, float]]]:
        """Compute each node score to be source."""
        pass

    def process_estimation(self, result: Dict[int, float]) -> List[Tuple[int, float]]:
        """Return estimated sources with scores."""

        if self.config.normalize_results:
            result = normalize_dict_values(result)
        result = sort_dict_by_value(result)
        max_estimation = result[0][1]
        if self.config.source_threshold:
            return [
                (node, value)
                for node, value in result
                if max_estimation - value <= self.config.source_threshold
            ]
        else:
            return result[: self.config.number_of_sources or 1]

    @cached_property
    def detected_sources(self) -> Union[int, List[int]]:
        """Return estimated sources."""
        result = [source for source, _ in self.detected_sources_estimation]
        return result[0] if len(result) == 1 else result

    @cached_property
    def detected_sources_estimation(self) -> List[Tuple[int, float]]:
        return self.process_estimation(self.estimate_sources())

    def get_additional_data_for_source_evaluation(self) -> Dict[str, Any]:
        return {}

    def evaluate_sources(
        self, real_sources: List[int]
    ) -> SingleSourceDetectionEvaluation:
        return compute_source_detection_evaluation(
            G=self.IG,
            real_sources=real_sources,
            detected_sources=self.detected_sources,
            additional_data=self.get_additional_data_for_source_evaluation(),
        )

    def __str__(self) -> str:
        name: str = " ".join(camel_case_split(self.__class__.__name__))
        return f"{name}-{self.config}"


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

    def process_estimation(self, result: Dict[int, float]) -> List[Tuple[int, float]]:
        nodes = {}
        for cluster, nodes_dict in self.estimate_sources().items():
            nodes.update(super().process_estimation(nodes_dict))
        return sort_dict_by_value(nodes)

    def get_additional_data_for_source_evaluation(self) -> Dict[str, Any]:
        return {"communities": self.communities}
