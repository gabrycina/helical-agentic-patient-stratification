import numpy as np
from typing import Any, Dict
import anndata
from helical.models.scgpt.model import scGPT 
from helical.models.geneformer.model import Geneformer
from monitoring.telemetry import monitor_model_execution


# example class but this could be done https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py
class EmbeddingModel:
    """Base class for gene expression embedding models."""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, input_file: str) -> np.ndarray:
        """Process gene expression data and return embeddings."""
        raise NotImplementedError("Subclasses must implement process method")

# example class but this could be done https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py
class HelicalScGPTModel(EmbeddingModel):
    """Implementation using Helical's scGPT model."""
    
    def __init__(self):
        super().__init__("scgpt")
        self.model = scGPT()
    
    @monitor_model_execution
    def process(self, input_file: str) -> np.ndarray:
        """Process gene expression data using Helical's scGPT model."""
        print(f"Processing {input_file} with Helical scGPT model")
        
        adata = anndata.read_h5ad(input_file)
        
        # Ensure data is in the format scGPT expects
        if 'gene_ids' not in adata.var or 'gene' not in adata.var:
            raise ValueError("Dataset must contain 'gene_ids' and 'gene' columns in .var")
            
        self.model.config["emb_mode"] = "cls"
        embeddings = self.model.get_embeddings(adata)
        
        return embeddings


# example class but this could be done https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py
class HelicalGeneformerModel(EmbeddingModel):
    """Implementation using Helical's Geneformer model."""
    
    def __init__(self):
        super().__init__("geneformer")
        self.model = Geneformer()
    
    def process(self, input_file: str) -> np.ndarray:
        """Process gene expression data using Helical's Geneformer model."""
        print(f"Processing {input_file} with Helical Geneformer model")
        
        adata = anndata.read_h5ad(input_file)
        processed_data = self.model.process_data(adata, gene_names="gene_name")
        embeddings = self.model.get_embeddings(processed_data)
        
        return embeddings


def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Factory function to get the appropriate embedding model."""
    models = {
        "scgpt": HelicalScGPTModel(),
        "geneformer": HelicalGeneformerModel(),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown embedding model: {model_name}")
    
    return models[model_name] 