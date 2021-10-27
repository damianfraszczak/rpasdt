"""Rumor Center source detection method."""
from random import shuffle
from typing import List

import networkx as nx
from networkx import Graph

from rpasdt.algorithm.source_detectors.common import (
    CommunityBasedSourceDetector,
)


def get_root(tree: Graph, node: int):
    return tree.in_edges(node)[0][0] if tree.in_edges(node) else None


def get_children(tree: Graph, node: int):
    return [out_edge[1] for out_edge in tree.out_edges(node)]


def get_leaves(tree: Graph):
    return [
        x for x in tree.nodes() if tree.out_degree(x) == 0 and tree.in_degree(x) == 1
    ]


def children_processed(tree: Graph, node: int, processed_nodes: List[int]) -> bool:
    children = get_children(tree, node)
    return len(children) == 0 or all(child in processed_nodes for child in children)


class RumorCenterCommunityBasedSourceDetector(CommunityBasedSourceDetector):
    def find_sources_in_community(self, graph: Graph):
        nodes = list(graph)
        shuffle(nodes)
        N = len(nodes)
        rumorCentrality = {}

        for sourceNode in nodes:
            bfs_tree = nx.bfs_tree(graph, sourceNode, False)
            messages_up = {}
            messages_down = {}
            leaves = get_leaves(bfs_tree)

            while not children_processed(
                bfs_tree, sourceNode, list(messages_down.keys())
            ):
                for node in nodes:
                    if node in leaves:
                        messages_up[node] = 1
                        messages_down[node] = 1
                    elif node != sourceNode:
                        if children_processed(
                            bfs_tree, node, list(messages_down.keys())
                        ):
                            node_children = get_children(bfs_tree, node)
                            msg_top = 0
                            msg_down = 1
                            for child in node_children:
                                msg_top = msg_top + messages_up[child]
                                msg_down = msg_down * messages_down[child]
                            messages_up[node] = msg_top + 1
                            messages_down[node] = messages_up[node] * msg_down

            source_children = get_children(bfs_tree, sourceNode)
            r = 1.0
            for child in source_children:
                r = r / messages_down[child]
            r = r / N
            rumorCentrality[sourceNode] = r
        return rumorCentrality
