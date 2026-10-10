import time
from pathlib import Path
from typing import Annotated

import anthropic
import openai
import psycopg
import typer
from pgvector.psycopg import register_vector

from rag.config import Settings
from rag.db import connect, init_db
from rag.embed import make_embedder
from rag.generate.answer import answer_question
from rag.generate.generator import make_generator
from rag.ingest.chunking import chunk_markdown
from rag.ingest.loaders import iter_documents
from rag.ingest.pipeline import ingest
from rag.retrieval.rerank import make_reranker
from rag.retrieval.retriever import Retriever, RetrieverConfig

app = typer.Typer(help="RAG pipeline commands.", no_args_is_help=True)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
LLM_KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
PREVIEW_CHARS = 120


def _database_unreachable(exc: Exception) -> typer.Exit:
    typer.secho(f"Cannot reach database: {exc}", fg="red", err=True)
    typer.echo("Is it running? Try: docker compose up -d", err=True)
    return typer.Exit(1)


@app.command("ask")
def ask_cmd(
    question: Annotated[str, typer.Argument(help="Question about Kubernetes.")],
    k: Annotated[int, typer.Option("-k", help="Passages sent to the LLM.")] = 5,
    rerank: Annotated[bool, typer.Option(help="Rerank candidates before answering.")] = True,
    generated: Annotated[bool, typer.Option(help="Include auto-generated reference pages.")] = True,
) -> None:
    """Answer a question from the docs, with [n] citations to the passages used."""
    settings = Settings()
    config = RetrieverConfig(k=k, candidates=max(50, k), include_generated=generated, rerank=rerank)
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc

    with conn:
        register_vector(conn)
        reranker = make_reranker(settings) if rerank else None
        retriever = Retriever(conn, make_embedder(settings), reranker)
        try:
            result = answer_question(
                question,
                retriever,
                make_generator(settings),
                config,
                on_text=lambda piece: typer.echo(piece, nl=False),
            )
        except (anthropic.AnthropicError, openai.OpenAIError) as exc:
            key = LLM_KEY_NAMES.get(settings.llm_backend, "the provider API key")
            typer.secho(f"\nLLM call failed ({settings.llm_backend}): {exc}", fg="red", err=True)
            typer.echo(f"Check {key} in .env (never commit it).", err=True)
            raise typer.Exit(1) from exc

    if result.refused and not result.retrieved:
        typer.echo(result.text)
    typer.echo("")
    if result.invalid_citations:
        cited = ", ".join(f"[{n}]" for n in result.invalid_citations)
        typer.secho(f"Warning: answer cites passages that don't exist: {cited}", fg="yellow")
    if result.sources:
        typer.secho("\nSources", bold=True)
        for source in result.sources:
            hit = source.hit
            typer.echo(f"[{source.number}] {hit.heading or hit.title}\n    {hit.url}")


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the pgvector extension, tables and indexes."""
    try:
        init_db(Settings())
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc
    typer.secho("Schema ready.", fg="green")


@app.command("ingest")
def ingest_cmd(
    path: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Corpus root, e.g. data/k8s-docs.")
    ],
    prune: Annotated[
        bool, typer.Option(help="Delete indexed documents no longer under PATH.")
    ] = True,
) -> None:
    """Load, chunk, embed and store documents. Unchanged documents are skipped.

    Always point PATH at the corpus root: sources are stored relative to it, and --prune
    removes everything not found under it.
    """
    settings = Settings()
    docs = list(iter_documents(path))
    if not docs:
        typer.secho(f"No documents found under {path}", fg="yellow")
        raise typer.Exit(1)

    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc

    fingerprint = settings.pipeline_fingerprint()
    typer.echo(f"{len(docs)} documents, pipeline {fingerprint}")
    started = time.perf_counter()
    with conn, typer.progressbar(length=len(docs), label="Ingesting") as bar:
        register_vector(conn)
        stats = ingest(
            conn,
            docs,
            make_embedder(settings),
            settings.chunk_max_chars,
            fingerprint,
            prune=prune,
            on_progress=lambda _i: bar.update(1),
        )
    elapsed = time.perf_counter() - started
    typer.secho(
        f"added {stats.added}, updated {stats.updated}, skipped {stats.skipped}, "
        f"deleted {stats.deleted} ({stats.chunks} chunks) in {elapsed:.1f}s",
        fg="green",
    )


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Natural-language question or keywords.")],
    mode: Annotated[str, typer.Option(help="dense | keyword | hybrid")] = "hybrid",
    k: Annotated[int, typer.Option("-k", help="Number of results.")] = 5,
    generated: Annotated[bool, typer.Option(help="Include auto-generated reference pages.")] = True,
    rerank: Annotated[
        bool, typer.Option(help="Re-score candidates with the local cross-encoder.")
    ] = False,
) -> None:
    """Search the index and print ranked chunks (no LLM involved)."""
    try:
        config = RetrieverConfig(
            mode=mode,
            k=k,
            candidates=max(50, k),
            include_generated=generated,
            rerank=rerank,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    settings = Settings()
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc
    with conn:
        register_vector(conn)
        reranker = make_reranker(settings) if rerank else None
        hits = Retriever(conn, make_embedder(settings), reranker).retrieve(query, config)

    if not hits:
        typer.secho("No results.", fg="yellow")
        return
    for rank, hit in enumerate(hits, start=1):
        snippet = hit.content[:PREVIEW_CHARS].replace("\n", " ")
        tag = " [generated]" if hit.generated else ""
        typer.secho(f"{rank}. {hit.score:.4f}  {hit.heading or hit.title}{tag}", bold=True)
        typer.echo(f"   {hit.url}")
        typer.echo(f"   {snippet}")


@app.command("chunk")
def chunk_cmd(
    path: Annotated[Path, typer.Argument(exists=True, help="File or folder of .md/.txt docs.")],
    preview: Annotated[bool, typer.Option(help="Print the first chars of each chunk.")] = False,
) -> None:
    """Chunk documents locally and report stats (no database needed)."""
    max_chars = Settings().chunk_max_chars
    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    docs = [p for p in paths if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    if not docs:
        typer.secho(f"No {sorted(SUPPORTED_SUFFIXES)} files under {path}", fg="yellow")
        raise typer.Exit(1)

    total = 0
    for doc in docs:
        chunks = chunk_markdown(doc.read_text(encoding="utf-8", errors="replace"), max_chars)
        total += len(chunks)
        typer.echo(f"{doc}: {len(chunks)} chunks")
        if preview:
            for c in chunks:
                snippet = c.content[:PREVIEW_CHARS].replace("\n", " ")
                typer.echo(f"  [{c.ord}] {c.heading or '(no heading)'} | {snippet}")
    typer.secho(f"{len(docs)} docs, {total} chunks (max_chars={max_chars})", fg="green")
