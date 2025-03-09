from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Configuration settings for the patient stratification application."""
    
    APP_NAME: str = "Helical Patient Stratification API"
    API_V1_STR: str = "/api/v1"
    DEFAULT_EMBEDDING_MODEL: str = "scgpt"
    DEFAULT_CLUSTERING_MODEL: str = "leiden"
    LLM_MODEL: str = "openai:gpt-4o"
    OPENAI_API_KEY: str = Field("", env="OPENAI_API_KEY")
    DATA_DIR: Path = Path("data")
    RESULTS_DIR: Path = Path("results")
    MAX_RETRIES: int = 3
    
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True) 