from rag.config import Settings
from rag.embed.base import Embedder
from rag.embed.fake import FakeEmbedder
from rag.embed.local import LocalEmbedder
from rag.embed.openai_embedder import OpenAIEmbedder


def make_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``RAG_EMBEDDING_BACKEND``. Models load lazily."""
    match settings.embedding_backend:
        case "local":
            return LocalEmbedder(
                dim=settings.embedding_dim, model_name=settings.local_embedding_model
            )
        case "openai":
            return OpenAIEmbedder(dim=settings.embedding_dim, model=settings.openai_embedding_model)
        case "fake":
            return FakeEmbedder(dim=settings.embedding_dim)
    raise ValueError(f"unknown embedding backend {settings.embedding_backend!r}")
