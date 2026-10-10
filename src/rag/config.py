from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    embedding_dim: int = Field(default=1536, gt=0, le=16000)
    chunk_max_chars: int = Field(default=1500, gt=0)
