import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import get_node_list, get_text

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        max_node_in_cluster=5,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        if max_node_in_cluster > reduction_dimension:
            raise ValueError("max_node_in_cluster must not be larger than reduction_dimension")
        self.reduction_dimension = reduction_dimension
        self.max_node_in_cluster = max_node_in_cluster
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Maximum Nodes in a Cluster: {self.max_node_in_cluster}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.max_node_in_cluster = config.max_node_in_cluster
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
        clustering_method: str = "GMM",
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)
        
        def process_cluster(summarized_text, cluster, new_level_nodes, current_level_nodes, next_node_index):
            """
                This function creates a new node and adds it to the next layer if the summarized text indicates a clear commonality among the descriptions. 
                If the summary does not provide a clear commonality, the function does not create a new node and instead moves the children nodes up one layer.

                Args:
                    summarized_text (str): The summary.
                    cluster (List[Node]): The list of nodes in the current cluster.
                    new_level_nodes (Dict[int, Node]): A dictionary to hold the newly created nodes for the next level.
                    current_level_nodes (Dict[int, Node]): A dictionary of nodes in the current level.
                    next_node_index (int): The index for the next node to be created.

                Returns:
                    int: The updated next_node_index after processing the cluster. This function modifies new_level_nodes and current_level_nodes in place.
            """
            # New node will not be created. The children move one layer up
            if summarized_text == "No clear commonality can be found among the descriptions.":
                for node in cluster:
                    new_level_nodes[node.index] = node
                    if node.index in current_level_nodes:
                        del current_level_nodes[node.index]
            elif len(cluster) == 1:
                new_level_nodes[cluster[0].index] = cluster[0]
                if cluster[0].index in current_level_nodes:
                        del current_level_nodes[cluster[0].index]
            else:
                __, new_parent_node = self.create_node(
                    next_node_index, summarized_text, {node.index for node in cluster}
                )
                new_level_nodes[next_node_index] = new_parent_node
                next_node_index += 1
            return next_node_index

        def get_summary(cluster, summarization_length):
            """
            Generates a summary for a given cluster of nodes by extracting their text and summarizing it.

            Args:
                cluster (List[Node]): A list of nodes to summarize.
                summarization_length (int): The maximum number of tokens for the generated summary.

            Returns:
                Tuple[List[Node], str]: The original cluster of nodes and the summarized text.
            """
            node_texts = get_text(cluster)
            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            return cluster, summarized_text
            
        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                max_node_in_cluster=self.max_node_in_cluster,
                clustering_method=clustering_method,
                **self.clustering_params,
            )

            summarization_length = self.summarization_length

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    # Submit all tasks and get futures
                    futures = [
                        executor.submit(get_summary, cluster, summarization_length)
                        for cluster in clusters
                    ]
                    
                    # Collect results all at once after computation is done
                    for future in futures:
                        cluster, summarized_text = future.result()
                        next_node_index = process_cluster(summarized_text, cluster, new_level_nodes, current_level_nodes, next_node_index)

            else:
                for cluster in clusters:
                    cluster, summarized_text = get_summary(
                        cluster,
                        summarization_length,
                    )
                    next_node_index = process_cluster(summarized_text, cluster, new_level_nodes, current_level_nodes, next_node_index)

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        return current_level_nodes
