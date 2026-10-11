from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag.ingest.loaders import LOADER_VERSION

EmbeddingBackend = Literal["local", "openai", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", extra="ignore", populate_by_name=True
    )

    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    chunk_max_chars: int = Field(default=1500, gt=0)

    embedding_backend: EmbeddingBackend = "local"
    embedding_dim: int = Field(default=384, gt=0, le=16000)
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    # CPU threads for local models; None lets ONNX Runtime decide (can oversubscribe).
    embedding_threads: int | None = Field(default=None, gt=0)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Free OpenRouter model by default (cost); anthropic stays available for quality runs.
    llm_backend: Literal["anthropic", "openrouter", "fake"] = "openrouter"
    llm_model: str = "claude-sonnet-5-5"  # used by the anthropic backend
    llm_max_tokens: int = Field(default=1024, gt=0)
    # Per-request timeout. SDK defaults are ~10 min, so one stuck connection stalls a batch job.
    llm_timeout_s: float = Field(default=60.0, gt=0)
    # Free models share upstream capacity; gemma-4 was rate-limited on 2026-10-10, nemotron wasn't.
    openrouter_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Keys come from env or .env without the RAG_ prefix. SecretStr hides them in reprs.
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")

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
                f"loader{LOADER_VERSION}",
            )
        )
