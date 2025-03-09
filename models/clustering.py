import numpy as np
import anndata
import scanpy as sc
from typing import Dict, Any
from sklearn.metrics import silhouette_score


def cluster_patients(input_file: str, method: str = "leiden") -> Dict[str, Any]:
    """
    Cluster patients using scanpy's clustering implementations.
    
    Args:
        input_file: Path to the AnnData file containing gene expression data
        method: Clustering method to use (leiden, louvain, etc.)
        
    Returns:
        Dictionary with clustering results
    """
    adata = anndata.read_h5ad(input_file)
    
    # Make sure we have a PCA representation (required for clustering apparently, it lowers dimensionality)
    if 'X_pca' not in adata.obsm:
        print("Computing PCA representation...")
        sc.pp.pca(adata)
   
    # Note: It finds which cells/patients are similar to each other based on their gene expression patterns, 
    # creating a network of relationships that the clustering algorithms then use to identify groups 
    if 'neighbors' not in adata.uns:
        print("Computing neighborhood graph...")
        sc.pp.neighbors(adata)
    
    # Run clustering with the specified method
    if method == "leiden":
        sc.tl.leiden(adata, resolution=1.0)
        clusters = adata.obs['leiden']
    
    elif method == "louvain":
        sc.tl.louvain(adata, resolution=1.0)
        clusters = adata.obs['louvain']
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Calculate metrics: how many patients are in each cluster by counting the occurrences of each cluster label, 
    # returning a dictionary where keys are cluster IDs and values are the number of patients in each cluster.
    cluster_sizes = clusters.value_counts().to_dict()
    
    # Calculate silhouette score: measures how similar a patient is to its own cluster compared to other clusters.
    # A score of 1 means the patient is very similar to its own cluster and very different from others.
    # A score of -1 means the patient is very different from its own cluster and similar to others.
    try:
        if 'X_pca' in adata.obsm:
            s_score = silhouette_score(adata.obsm['X_pca'], clusters)
    except Exception as e:
        print(f"Warning: Could not compute silhouette score: {e}")
        s_score = 0.0
    
    return {
        "num_clusters": len(cluster_sizes),
        "sizes": cluster_sizes,
        "silhouette_score": s_score,
        "cluster_assignments": clusters.to_dict() #patient/cell to their assigned cluster number
    }