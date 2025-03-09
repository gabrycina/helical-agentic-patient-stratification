import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from config import settings
from models.gene_expression import get_embedding_model
from models.clustering import cluster_patients

#Headsup: BaseModel would be overkill since this class just holds configuration values. 
#The other classes use BaseModel because they need Pydantic's validation and serialization features for API responses
@dataclass
class StratificationDependencies():
    """Dependencies for the patient stratification agent."""
    input_file: str
    embedding_model: str = settings.DEFAULT_EMBEDDING_MODEL
    clustering_method: str = settings.DEFAULT_CLUSTERING_MODEL


class PatientSubgroup(BaseModel):
    """Model representing a patient subgroup."""
    group_id: int = Field(..., description="Unique identifier for the subgroup")
    size: int = Field(..., description="Number of patients in this subgroup")
    marker_genes: List[str] = Field(..., description="Specific genes that are uniquely active (or inactive) in a group of patients, acting like a fingerprint to identify that group") # Characteristics traits we may say
    pathways: List[str] = Field(..., description="Enriched biological pathways")
    drugs: List[str] = Field([], description="Potential drugs targeting this subgroup")


class StratificationResult(BaseModel):
    """Result of patient stratification."""
    patient_count: int = Field(..., description="Total number of patients analyzed")
    subgroups: List[PatientSubgroup] = Field(..., description="Identified patient subgroups")
    summary: str = Field(..., description="Summary of findings and recommendations")


# Define the main stratification agent 
# (is LLM right for this? Yes, i believe it must be in order to orchestrate the workflow)
stratification_agent = Agent(
    settings.LLM_MODEL,
    deps_type=StratificationDependencies,
    result_type=StratificationResult,
    system_prompt="""
    You are an expert oncologist specializing in precision medicine. 
    Your task is to stratify cancer patients based on their gene expression data and
    identify personalized treatment options for each subgroup.
    
    Analyze the data carefully, identify clinically relevant subgroups, and
    recommend appropriate targeted therapies based on the molecular profiles.
    """
)


@stratification_agent.system_prompt
async def data_context(ctx: RunContext[StratificationDependencies]) -> str:
    """Add context about the dataset being analyzed."""
    return f"""
    You are analyzing gene expression data from the file: {ctx.deps.input_file}
    Using embedding model: {ctx.deps.embedding_model}
    Using clustering method: {ctx.deps.clustering_method}
    """


@stratification_agent.tool
async def generate_embeddings(ctx: RunContext[StratificationDependencies]) -> Dict[str, Any]:
    """Generate embeddings from gene expression data to create a latent representation of patient profiles."""
    input_file = ctx.deps.input_file
    model_name = ctx.deps.embedding_model
    
    print(f"Generating embeddings using {model_name} model from {input_file}")
    
    # TODO: In a real implementation, this would call actual embedding models OR (suggested) use the actual embedding model
    # added to the available models of pydantic-ai through custom code. As in here: https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py
    embedding_model = get_embedding_model(model_name)
    embeddings = embedding_model.process(input_file)
    
    return {
        "num_patients": embeddings.shape[0],
        "embedding_dimensions": embeddings.shape[1],
        "model_used": model_name
    }


@stratification_agent.tool
async def identify_patient_clusters(
    ctx: RunContext[StratificationDependencies], 
) -> Dict[str, Any]:
    """
    Identify distinct patient clusters based on gene expression embeddings.
    """
    input_file = ctx.deps.input_file
    clustering_method = ctx.deps.clustering_method
    
    print(f"Clustering patients using {clustering_method}")
    
    #TODO: This would call the actual clustering implementation
    clusters = cluster_patients(input_file, clustering_method)
    
    return {
        "num_clusters": clusters["num_clusters"],
        "cluster_sizes": clusters["sizes"],
        "silhouette_score": clusters["silhouette_score"]
    }


async def main():
    """Run the stratification agent on a sample dataset."""
    deps = StratificationDependencies(
        input_file="sample_breast_cancer_data.h5ad",
    )
    
    result = await stratification_agent.run(
        "Stratify breast cancer patients and identify personalized treatment options.",
        deps=deps
    )
    
    print(result.data)


if __name__ == "__main__":
    asyncio.run(main()) 