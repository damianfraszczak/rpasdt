"""Common source detection methods."""
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional, Union

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
        self._node_estimations = {}
        self._normalized_node_estimations = {}
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
    def estimate_sources(
        self, G: Graph, IG: Graph
    ) -> Dict[int, Union[int, Dict[int, float]]]:
        """Compute each node score to be source."""
        pass

    def process_estimation(self, result: Dict[int, float]) -> Dict[int, float]:
        result = sort_dict_by_value(result)
        self._node_estimations.update(result)
        if self.config.normalize_results:
            result = normalize_dict_values(result)
        self._normalized_node_estimations.update(result)
        return result

    def retrieve_sources_from_estimation(
        self, result: Dict[int, float]
    ) -> Dict[int, float]:
        """Return estimated sources with scores."""
        max_estimation = max(result.values())
        if self.config.source_threshold:
            return {
                node: value
                for node, value in result.items()
                if max_estimation - value <= self.config.source_threshold and value >= 0
            }
        else:
            return {
                node: value
                for node, value in list(result.items())[
                    : self.config.number_of_sources or 1
                ]
            }

    @property
    def detected_sources(self) -> Union[int, List[int]]:
        """Return estimated sources."""
        result = [source for source, _ in self.detected_sources_estimation.items()]
        return result[0] if len(result) == 1 else result

    @property
    def detected_sources_estimation(self) -> Dict[int, float]:
        estimation = self.estimate_sources(G=self.G, IG=self.IG)
        processed_estimation = self.process_estimation(estimation)
        return self.retrieve_sources_from_estimation(processed_estimation)

    def get_additional_data_for_source_evaluation(self) -> Dict[str, Any]:
        return {
            "node_estimations": sort_dict_by_value(self._node_estimations),
            "normalized_node_estimations": sort_dict_by_value(
                self._normalized_node_estimations
            ),
        }

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

    @property
    def detected_sources_estimation(self) -> Dict[int, float]:
        sources = {}
        for cluster, nodes in self.communities.items():
            estimation = self.estimate_sources(G=self.G, IG=self.IG.subgraph(nodes))
            processed_estimation = self.process_estimation(estimation)
            sources.update(self.retrieve_sources_from_estimation(processed_estimation))
        return sort_dict_by_value(sources)

    def get_additional_data_for_source_evaluation(self) -> Dict[str, Any]:

        return {
            **super().get_additional_data_for_source_evaluation(),
            "communities": self.communities,
            "estimation_per_community": self.get_estimation_per_community(),
        }

    def get_estimation_per_community(self):
        estimation_per_community = {}
        for cluster, nodes in self.communities.items():
            estimation_per_community[cluster] = {
                node: self._node_estimations[node] for node in nodes
            }
        return estimation_per_community
