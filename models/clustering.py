import anndata
import scanpy as sc
from typing import Dict, Any
from sklearn.metrics import silhouette_score
from monitoring.telemetry import setup_monitoring
import logfire
import numpy as np

setup_monitoring()

def cluster_patients(embeddings: np.ndarray, method: str = "leiden") -> Dict[str, Any]:
    """
    Cluster patients based on their embeddings.
    
    Args:
        embeddings: numpy array of shape (n_patients, n_dimensions)
        method: clustering method to use (leiden, louvain)
    """
    logfire.info('Clustering patients')
    
    # Create anndata object from embeddings
    adata = sc.AnnData(embeddings)
    
    with logfire.span('Computing neighborhood graph'):
        sc.pp.neighbors(adata, use_rep='X')  # Use embeddings directly
    
    with logfire.span(f'Running {method} clustering'):
        if method == "leiden":
            sc.tl.leiden(adata, resolution=1.0)
            clusters = adata.obs['leiden']
        elif method == "louvain":
            sc.tl.louvain(adata, resolution=1.0)
            clusters = adata.obs['louvain']
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
    
    # Calculate metrics
    cluster_sizes = clusters.value_counts().to_dict()
    
    try:
        s_score = silhouette_score(embeddings, clusters)
        if s_score < 0.3:
            logfire.warn('Low silhouette score detected: {score:.2f}', 
                        score=s_score)
    except Exception as e:
        logfire.error('Failed to compute silhouette score: {error}', 
                     error=str(e))
        s_score = 0.0
    
    return {
        "num_clusters": len(cluster_sizes),
        "sizes": cluster_sizes,
        "silhouette_score": s_score,
        "cluster_assignments": clusters.to_dict()
    }