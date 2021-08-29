from dataclasses import dataclass

from rpasdt.network.taxonomies import NodeAttributeEnum


@dataclass
class NodeAttribute:
    key: NodeAttributeEnum
    value: any
