import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import logfire

from config import settings
from models.gene_expression import get_embedding_model
from models.clustering import cluster_patients
from monitoring.telemetry import setup_monitoring

# Initialize monitoring at startup
setup_monitoring()

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
    marker_genes: List[str] = Field(..., description="Specific genes that are uniquely active (or inactive) in a group of patients, acting like a fingerprint to identify that group") # Characteristics traits we may say
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
    """Generate embeddings from gene expression data."""
    input_file = ctx.deps.input_file
    model_name = ctx.deps.embedding_model
    
    logfire.info('Starting embedding generation with {model}', model=model_name)
    
    # TODO: In a real implementation, this would call actual embedding models OR (suggested) use the actual embedding model
    # added to the available models of pydantic-ai through custom code. As in here: https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py
    embedding_model = get_embedding_model(model_name)
    embeddings = embedding_model.process(input_file)
    
    logfire.info('Completed embedding generation: {shape} embeddings created', 
                 shape=embeddings.shape)
    
    
    # TODO: Store embeddings in memory
    ctx.memory['embeddings'] = embeddings
    
    return {
        "num_patients": embeddings.shape[0],
        "embedding_dimensions": embeddings.shape[1],
        "model_used": model_name
    }


@stratification_agent.tool
async def identify_patient_clusters(
    ctx: RunContext[StratificationDependencies], 
) -> Dict[str, Any]:
    """Identify distinct patient clusters based on gene expression embeddings."""
    clustering_method = ctx.deps.clustering_method
    
    # Get embeddings from memory
    if 'embeddings' not in ctx.memory:
        raise ValueError("No embeddings found. Run generate_embeddings first.")
    
    embeddings = ctx.memory['embeddings']
    
    logfire.info('Starting patient clustering using {method}', method=clustering_method)
    
    #TODO: This would call the actual clustering implementation
    clusters = cluster_patients(embeddings, clustering_method)
    
    logfire.info('Completed clustering: identified {num} clusters with silhouette score {score}', 
                 num=clusters["num_clusters"],
                 score=clusters["silhouette_score"])
    
    return clusters


async def main():
    """Run the stratification agent on a sample dataset."""
    logfire.info('Starting patient stratification pipeline')
    
    deps = StratificationDependencies(
        input_file="sample_breast_cancer_data.h5ad",
    )
    
    result = await stratification_agent.run(
        "Stratify breast cancer patients and identify personalized treatment options.",
        deps=deps
    )
    
    logfire.info('Completed stratification pipeline')
    print(result.data)


if __name__ == "__main__":
    asyncio.run(main()) 