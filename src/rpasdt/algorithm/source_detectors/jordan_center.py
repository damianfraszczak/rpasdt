"""Jordan center source detection method."""
from collections import defaultdict

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


class JordanCenterCommunityBasedSourceDetector(CommunityBasedSourceDetector):
    def find_sources_in_community(self, graph: Graph):

        IG = graph
        nodes = list(self.G.nodes())
        N = len(IG.nodes())
        tall = defaultdict(int)

        infected_nodes_neighbours = defaultdict(list)
        infected_nodes_neighbours.update(
            {
                infected_node: list(nx.all_neighbors(self.G, infected_node))
                for infected_node in IG.nodes()
            }
        )

        STOP = 0
        t = 1

        while STOP == 0:
            nodes_with_infected_neighbours = defaultdict(list)
            for node in nodes:
                for neighbour in nx.all_neighbors(self.G, node):
                    nodes_with_infected_neighbours[neighbour] = list(
                        set(
                            nodes_with_infected_neighbours[neighbour]
                            + infected_nodes_neighbours[node]
                        )
                    )

            # dla wszystkich wezlow
            for node in nodes:
                # liczba zarazonych swoich sasiadow +
                infected_nodes_sum = len(
                    list(
                        set(
                            infected_nodes_neighbours[node]
                            + nodes_with_infected_neighbours[node]
                        )
                    )
                ) - len(infected_nodes_neighbours[node])
                infected_nodes_neighbours[node] = list(
                    set(
                        infected_nodes_neighbours[node]
                        + nodes_with_infected_neighbours[node]
                    )
                )

                tall[node] = tall[node] + t * infected_nodes_sum
                # if multiple communities the value can be higher
                if len(infected_nodes_neighbours[node]) >= N:
                    STOP = 1

            t = t + 1
        return {
            node: tall[node]
            for node in nodes
            if len(infected_nodes_neighbours[node]) >= N
        }
