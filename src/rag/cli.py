from pathlib import Path
from typing import Annotated

import psycopg
import typer

from rag.config import Settings
from rag.db import init_db
from rag.ingest.chunking import chunk_markdown

app = typer.Typer(help="RAG pipeline commands.", no_args_is_help=True)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
PREVIEW_CHARS = 120


@app.command("init-db")
def init_db_cmd() -> None:
    """Create the pgvector extension, tables and indexes."""
    try:
        init_db(Settings())
    except psycopg.OperationalError as exc:
        typer.secho(f"Cannot reach database: {exc}", fg="red", err=True)
        typer.echo("Is it running? Try: docker compose up -d", err=True)
        raise typer.Exit(1) from exc
    typer.secho("Schema ready.", fg="green")


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
