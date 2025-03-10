import anndata
import scanpy as sc
from typing import Dict, Any
from sklearn.metrics import silhouette_score
from monitoring.telemetry import setup_monitoring
import logfire

setup_monitoring()

def cluster_patients(input_file: str, method: str = "leiden") -> Dict[str, Any]:
    """
    Cluster patients using scanpy's clustering implementations.
    
    Args:
        input_file: Path to the AnnData file containing gene expression data
        method: Clustering method to use (leiden, louvain, etc.)
        
    Returns:
        Dictionary with clustering results
    """
    with logfire.span('Clustering patients', method=method) as span:
        try:
            adata = anndata.read_h5ad(input_file)
             
            # Log preprocessing steps
            with logfire.span('Computing PCA'):
                if 'X_pca' not in adata.obsm:
                    sc.pp.pca(adata)
                    span.set_attribute('pca_variance_ratio', 
                                     adata.uns['pca']['variance_ratio'].tolist())
            
            with logfire.span('Computing neighborhood graph'):
                if 'neighbors' not in adata.uns:
                    sc.pp.neighbors(adata)
            
            # Run clustering
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
                s_score = silhouette_score(adata.obsm['X_pca'], clusters)
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
                "cluster_assignments": clusters.to_dict() #patient/cell to their assigned cluster number
            }
            
        except Exception as e:
            logfire.exception('Clustering failed')
            raise