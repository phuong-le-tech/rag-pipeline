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
    # CPU threads for the local model; None lets ONNX Runtime decide (can oversubscribe).
    embedding_threads: int | None = Field(default=None, gt=0)

    def embedding_model_name(self) -> str:
        return {
            "local": self.local_embedding_model,
            "openai": self.openai_embedding_model,
            "fake": "fake",
        }[self.embedding_backend]

    def pipeline_fingerprint(self) -> str:
        """Identifies how chunks were produced; a change forces re-ingestion of every doc."""
        return ":".join(
            (
                self.embedding_backend,
                self.embedding_model_name(),
                str(self.embedding_dim),
                str(self.chunk_max_chars),
            )
        )
