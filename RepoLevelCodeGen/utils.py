import logging
import re
import os
import ast
from typing import Dict, List, Set

import numpy as np
import tiktoken
from scipy import spatial

from .tree_structures import Node, Tree

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def reverse_mapping(layer_to_nodes: Dict[int, List[Node]]) -> Dict[Node, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_nodes.items():
        for node in nodes:
            node_to_layer[node.index] = layer
    return node_to_layer

def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        if isinstance(node.function, dict):
            text += f"{' '.join(node.function['function_description'].splitlines())}"
        else:
            text += f"{' '.join(node.function.splitlines())}"
        text += "\n\n"
    return text


def get_nodes_by_func_names(func_names: List[str], tree: Tree, repo_functions: Dict):
    """
    Retrieves nodes from the tree that match the given function names.
    
    Args:
        func_names (List[str]): List of function names to search for in the tree.
        tree (Tree): The tree structure containing all nodes.
        repo_functions (Dict): Dictionary mapping function names to their implementations.
        
    Returns:
        List: Context information extracted from the matching nodes.
    """
    nodes = []
    for fname in func_names:
        for node in get_node_list(tree.all_nodes):
            if isinstance(node.function, dict):
                if node.function["body"] == repo_functions[fname].body:
                    nodes.append(node)
    return get_context(nodes)


def indices_of_nearest_neighbors_from_distances(distances: List[float]) -> np.ndarray:
    """
    Returns the indices of nearest neighbors sorted in ascending order of distance.

    Args:
        distances (List[float]): A list of distances between embeddings.

    Returns:
        np.ndarray: An array of indices sorted by ascending distance.
    """
    return np.argsort(distances)


def extract_functions_from_repo(repo_path):
    """
    Returns the list of dictionary that contains functions in the repository.
    Constructor functions (__init__) are excluded from the results.

    Args:
        repo_path (str): Path to the repository.

    Returns:
        List: An array of functions, excluding constructors.
    """
    function_info = []
    for root, _, files in os.walk(repo_path):
        # Skip the tests folder at the root level only
        root_relative_path = os.path.relpath(root, repo_path)
        if root_relative_path == 'tests' or root_relative_path.startswith('tests/') or root_relative_path == 'test' or root_relative_path.startswith('test/'):
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    source_code = f.read()
                try:
                    tree = ast.parse(source_code)
                    
                    # Add parent references to all nodes
                    for parent in ast.walk(tree):
                        for child in ast.iter_child_nodes(parent):
                            setattr(child, 'parent', parent)
                    
                    class_stack = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            class_stack.append(node)
                            continue
                            
                        if isinstance(node, ast.FunctionDef):
                            # Skip constructor functions
                            if node.name == '__init__':
                                continue
                                
                            function_source = ast.get_source_segment(source_code, node)
                            
                            # Find the closest class ancestor
                            current = node
                            parent_class = None
                            while hasattr(current, 'parent'):
                                current = current.parent
                                if isinstance(current, ast.ClassDef):
                                    parent_class = current
                                    break
                            
                            info = {
                                'file': file_path,
                                'function': node.name,
                                'line_number': node.lineno,
                                'body': function_source,
                                'is_method': False
                            }
                            
                            if parent_class:
                                info['is_method'] = True
                                info['class_name'] = parent_class.name
                            
                            function_info.append(info)
                            
                        if isinstance(node, ast.ClassDef) and class_stack:
                            class_stack.pop()
                            
                except SyntaxError:
                    continue
    return function_info


def remove_function_from_tree(tree: Tree, function_body: str):
    """
    Removes the description of a function from the tree so that it cannot be retrieved in Code Generation
    
    Args:
        tree (Tree): Tree where the function is to be removed from the
        function_body (str): the function to be removed from the tree
        
    Returns:
        List: The nodes that are altered
    """
    removed_nodes = []
    
    for node in tree.layer_to_nodes[0]:
        if node.function["body"].strip() == function_body.strip():
            removed_nodes.append(node)
            emb_name = next(iter(node.embeddings.items()))[0]
            emb_size = len(next(iter(node.embeddings.values())))  # Get size of original embedding
            tree.all_nodes[node.index].embeddings = {emb_name: float('inf') * np.ones(emb_size)}  # Using a large value to ensure it's not selected
            tree.all_nodes[node.index].function["function_description"] = None
            tree.all_nodes[node.index].function["body"] = None
    return removed_nodes


def put_function_back_tree(tree: Tree, removed_nodes):
    """
    Alters the tree by putting the function description of the nodes in the tree
    
    Args:
        Tree: Tree where the function is to be put back
        List: The nodes that are altered
    
    Returns:
    """
    
    for node in removed_nodes:
        tree.all_nodes[node.index].embeddings = node.embeddings
        tree.all_nodes[node.index].function["function_description"] = node.function["function_description"]
        tree.all_nodes[node.index].function["body"] = node.function["body"]
        

def get_context(node_list: List[Node]) -> str:
    """
    Generates a list that contains functions along with their descriptions from a list of nodes.
    NOTE: It returns only the leaf nodes

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Context data [{"function": "", "description": "", "evidence": ""}, ...].
    """
    
    context = []
    for node in node_list:
        if isinstance(node.function, dict): # It is a leaf node
            context_item = {
                "function": node.function['body'], 
                "description": node.function['function_description'],
                "file": node.function['file'],
                "function_name": node.function['function'],
                "is_method": node.function['is_method'],
            }
            if node.function['is_method']:
                context_item["class_name"] = node.function['class_name']
            
            # Add evidence if available
            if hasattr(node, 'evidence'):
                context_item["evidence"] = node.evidence
                
            context.append(context_item)
    return context

def nodes_content_list(node_list: List[Node]) -> List[str]:
    """
    Extracts the function content from list of nodes
    
    Args:
        node_list (List[Node]): List of nodes.
        
    Returns:
        List[str]: A list of texts that are descriptions or summarizations of the functions.
    """
    nodes_content=[]
    for node in node_list:
        if isinstance(node.function, dict):
            if node.function['function_description']:
                nodes_content.append(str("Description: " + node.function['function_description'] + "\n\t" + "Implementation: " + node.function['body']))
            else:
                nodes_content.append(str(node.function['function_description']))
        else:
            nodes_content.append(str("Summary: " + node.function))
    return nodes_content

def path_to_module(path: str) -> str:
    """
    Extracts the module path from the path to the file.
    example: './dataset/RepoExec/test-apps/apimd/apimd/parser.py' -> 'apimd.parser'
    """
    parts = path.strip('/').split('/')
    try:
        start = parts.index('test-apps') + 2
        module_path = '.'.join(parts[start:])
        return module_path.removesuffix('.py').rstrip('.')
    except ValueError:
        return ''

def add_imports(context: List[Dict]):
    """
    Inserts the imports into the answer.
    """
    
    for item in context:
        if item["is_method"]:
            import_statement = f"from {path_to_module(item['file'])} import {item['class_name']}"
        else:
            import_statement = f"from {path_to_module(item['file'])} import {item['function_name']}"
        item['import_statement'] = import_statement

def add_constructors(context, repo_functions):
    """
    Adds the constructors to the context.
    """
    for item in context:
        if item['is_method'] and (f"{item['class_name']}.__init__" in repo_functions):
            constructor = repo_functions[f"{item['class_name']}.__init__"].body
            item['constructor'] = constructor