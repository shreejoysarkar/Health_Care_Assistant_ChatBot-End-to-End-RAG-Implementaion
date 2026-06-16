"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # huggingface Configuration
    HF_TOKEN: str

    # Qdrant Cloud Configuration
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    # Collection Settings
    collection_name: str = "medical_rag"

    # Embedding Pipeline Settings
    embedding_batch_size: int = 64
    embedding_max_length: int = 8192
    upsert_batch_size: int = 100

    # Document Processing Settings
    chunk_size: int = 1024
    chunk_overlap: int = 200

    # Model Configuration
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "llama3.1"
    llm_temperature: float = 0.0

    # Retrieval Settings
    retrieval_k: int = 4

    # Re-ranking Settings
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_initial_k: int = 20
    reranker_max_length: int = 1024

    # Logging
    log_level: str = "INFO"

    # RAGAS Evaluation Settings
    enable_ragas_evaluation: bool = True
    ragas_timeout_seconds: float = 30.0
    ragas_log_results: bool = True
    ragas_llm_model: str | None = None 
    ragas_llm_temperature: float | None = None 
    ragas_embedding_model: str | None = None 

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Application Info
    app_name: str = "Health Care Assistant"
    app_version: str = "0.1.0"

    # Data Directory
    data_directory: str = "Data/Input"



@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()