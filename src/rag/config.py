from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingBackend = Literal["local", "openai", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    chunk_max_chars: int = Field(default=1500, gt=0)

    embedding_backend: EmbeddingBackend = "local"
    embedding_dim: int = Field(default=384, gt=0, le=16000)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
