import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
import hdbscan
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_distances

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)     # When the text gets longer this part makes problem. See what it does: https://umap-learn.readthedocs.io/en/latest/parameters.html#n-neighbors
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 150, min_clusters: int = 8, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    # max_clusters=len(embeddings)
    # if it is local clusterin, min_clustering must not apply
    if len(embeddings) <= min_clusters:
        n_clusters = np.arange(1, max_clusters)
    else:
        n_clusters = np.arange(min_clusters, max_clusters)
    bics = []
    for n in n_clusters:
        print(f"fitting GMM with {n} clusters")
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering_GMM(
    embeddings: np.ndarray, dim: int, threshold: float, max_node_in_cluster: int, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) -2))
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= max_node_in_cluster + 1: # dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, min(dim, len(global_cluster_embeddings_) - 2), max_node_in_cluster - 1
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters

def perform_clustering_HDBSCAN(
    embeddings: np.ndarray, dim: int, threshold: float, max_node_in_cluster: int, verbose: bool = False
) -> List[np.ndarray]:
    # First reduce dimensionality using UMAP
    reduced_embeddings_global = global_cluster_embeddings(embeddings, min(dim, len(embeddings) - 2))
    
    # Compute cosine distance matrix and ensure it's float64
    distance_matrix = cosine_distances(reduced_embeddings_global).astype(np.float64)
    
    # Perform HDBSCAN clustering with precomputed distances
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,  # Minimum size of clusters
        min_samples=1,       # Minimum samples in neighborhood
        metric='precomputed',  # Using precomputed distance matrix
        cluster_selection_epsilon=0.0,  # No epsilon-based cluster selection
        prediction_data=True  # Enable prediction for new points
    )
    
    # Fit the clusterer with the distance matrix
    clusterer.fit(distance_matrix)
    
    # Get cluster labels and probabilities
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    
    # Convert HDBSCAN output to our expected format
    # HDBSCAN uses -1 for noise points, we'll treat them as their own clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if verbose:
        logging.info(f"Global Clusters: {n_clusters}")
    
    # Initialize clusters list
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    
    # Process each cluster
    for i, label in enumerate(unique_labels):
        # Get indices of points in this cluster
        cluster_indices = np.where(labels == label)[0]
        
        if len(cluster_indices) == 0:
            continue
            
        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]
        
        if verbose:
            logging.info(f"Nodes in Global Cluster {i}: {len(cluster_embeddings)}")
        
        # If cluster is small enough, no need for local clustering
        if len(cluster_embeddings) <= max_node_in_cluster + 1:
            for idx in cluster_indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], i)
            continue
        
        # Perform local clustering for large clusters
        reduced_embeddings_local = local_cluster_embeddings(
            cluster_embeddings, 
            min(dim, len(cluster_embeddings) - 2), 
            max_node_in_cluster - 1
        )
        
        # Compute cosine distance matrix for local clustering and ensure it's float64
        local_distance_matrix = cosine_distances(reduced_embeddings_local).astype(np.float64)
        
        # Local HDBSCAN clustering with precomputed distances
        local_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='precomputed',  # Using precomputed distance matrix
            cluster_selection_epsilon=0.0,
            prediction_data=True
        )
        
        local_clusterer.fit(local_distance_matrix)
        local_labels = local_clusterer.labels_
        local_unique_labels = np.unique(local_labels)
        
        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {len(local_unique_labels)}")
        
        # Assign local cluster labels
        for j, local_label in enumerate(local_unique_labels):
            local_indices = np.where(local_labels == local_label)[0]
            for local_idx in local_indices:
                original_idx = cluster_indices[local_idx]
                all_local_clusters[original_idx] = np.append(
                    all_local_clusters[original_idx], 
                    i * 1000 + j  # Create unique local cluster IDs
                )
    
    if verbose:
        logging.info(f"Total Clusters: {len(np.unique(np.concatenate(all_local_clusters)))}")
    
    return all_local_clusters

class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        clustering_method: str = "GMM",
        reduction_dimension: int = 10,
        max_node_in_cluster: int = 5,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering
        print("clustering performing")
        if clustering_method == "GMM":
            clusters = perform_clustering_GMM(
                embeddings, dim=reduction_dimension, max_node_in_cluster=max_node_in_cluster, threshold=threshold
            )
        elif clustering_method == "HDBSCAN":
            clusters = perform_clustering_HDBSCAN(
                embeddings, dim=reduction_dimension, max_node_in_cluster=max_node_in_cluster, threshold=threshold
            )
        else:
            raise ValueError(f"Invalid clustering method: {clustering_method}")
        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.function["function_description"])) if isinstance(node.function, dict) else len(tokenizer.encode(node.function)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
