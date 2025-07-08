from typing import Dict, List, Set


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, function: Dict, index: int, children: Set[int], embeddings) -> None:
        self.function = function        # If the node is leaf, then keeps the function Dict. Otherwise it keeps cluster description
        self.index = index
        self.children = children
        self.embeddings = embeddings


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
