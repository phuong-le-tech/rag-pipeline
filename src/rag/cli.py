import json
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import anthropic
import openai
import psycopg
import typer
from pgvector.psycopg import register_vector

from rag.agent.loop import (
    AgentResult,
    ToolCallEvent,
    ToolResultEvent,
    agent_limits,
    run_agent,
)
from rag.agent.tools import DEFAULT_K, SearchTool
from rag.config import Settings
from rag.db import connect, init_db
from rag.embed import make_embedder
from rag.evals.answers import (
    RetryingGenerator,
    agent_answerer,
    answers_markdown,
    evaluate_answers,
    pipeline_answerer,
    write_answer_results,
)
from rag.evals.drafting import draft_items, sample_chunks
from rag.evals.golden import load_golden, unknown_sources
from rag.evals.judge import load_unanswerable
from rag.evals.review import pending_drafts, record_decision
from rag.evals.runner import (
    ABLATIONS,
    breakdown_markdown,
    evaluate,
    to_markdown,
    write_results,
)
from rag.generate.answer import Answer, answer_question
from rag.generate.generator import make_generator, make_tool_generator
from rag.ingest.chunking import chunk_markdown
from rag.ingest.loaders import iter_documents
from rag.ingest.pipeline import ingest
from rag.retrieval.rerank import make_reranker
from rag.retrieval.retriever import Retriever, RetrieverConfig
from rag.retrieval.rewrite import QueryRewriter

app = typer.Typer(help="RAG pipeline commands.", no_args_is_help=True)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
LLM_KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
PREVIEW_CHARS = 120


def llm_error_hint(exc: Exception, settings: Settings, agent: bool = False) -> str:
    """Actionable next step for an LLM failure, by kind of error rather than one generic hint."""
    key = LLM_KEY_NAMES.get(settings.llm_backend, "the provider API key")
    if isinstance(exc, anthropic.RateLimitError | openai.RateLimitError) and "per-day" in str(exc):
        return (
            "The daily free-model quota is used up (it applies to every free model): wait for "
            "the reset, add OpenRouter credits, or switch to RAG_LLM_BACKEND=anthropic."
        )
    if isinstance(exc, anthropic.RateLimitError | openai.RateLimitError):
        hint = "The model is rate-limited: retry in a minute"
        if settings.llm_backend == "openrouter":
            model_var = "RAG_AGENT_MODEL" if agent else "RAG_OPENROUTER_MODEL"
            hint += f", or pick another free model with {model_var}"
        return hint + "."
    if isinstance(exc, anthropic.AuthenticationError | openai.AuthenticationError):
        return f"Authentication failed: check {key} in .env (never commit it)."
    if isinstance(exc, anthropic.APIConnectionError | openai.APIConnectionError):
        return "Could not reach the provider: check your network connection."
    return f"Check {key} and the model name for backend '{settings.llm_backend}'."


def _database_unreachable(exc: Exception) -> typer.Exit:
    typer.secho(f"Cannot reach database: {exc}", fg="red", err=True)
    typer.echo("Is it running? Try: docker compose up -d", err=True)
    return typer.Exit(1)


def ask_config(k: int, rerank: bool, generated: bool, rewrite: bool) -> RetrieverConfig:
    """Defaults for `rag ask`, chosen from the n=94 evaluation (docs/06-evaluation.md):
    hybrid + rerank + at most one chunk per page. Rewriting stays opt-in: it helped
    hand-written questions but not others, and costs an LLM call per query."""
    return RetrieverConfig(
        mode="hybrid",
        k=k,
        candidates=max(50, k),
        include_generated=generated,
        rerank=rerank,
        max_per_source=1,
        rewrite=rewrite,
    )


def _print_sources(answer: Answer) -> None:
    typer.echo("")
    if answer.invalid_citations:
        cited = ", ".join(f"[{n}]" for n in answer.invalid_citations)
        typer.secho(f"Warning: answer cites passages that don't exist: {cited}", fg="yellow")
    if answer.sources:
        typer.secho("\nSources", bold=True)
        for source in answer.sources:
            hit = source.hit
            typer.echo(f"[{source.number}] {hit.heading or hit.title}\n    {hit.url}")


def _echo_search(event: ToolResultEvent, call: ToolCallEvent) -> None:
    k = (call.arguments or {}).get("k", DEFAULT_K)
    head = f'→ {event.name}("{event.query}", k={k})'
    if event.duplicate:
        refs = ", ".join(f"[{n}]" for n in event.numbers)
        note = f"duplicate, see {refs}" if refs else "duplicate, no passages"
        typer.secho(f"{head}  {note}", fg="yellow")
        return
    typer.echo(f"{head}  {event.ms / 1000:.1f} s")
    for number, hit in list(zip(event.numbers, event.hits, strict=True))[:3]:
        typer.echo(f"   [{number}] {hit.heading or hit.title} — {hit.url}")


def _run_agent_with_trace(
    question: str, settings: Settings, retriever: Retriever, config: RetrieverConfig
) -> AgentResult:
    search = SearchTool(retriever, config)
    call: ToolCallEvent | None = None
    result: AgentResult | None = None
    for event in run_agent(question, make_tool_generator(settings), search, agent_limits(settings)):
        if isinstance(event, AgentResult):
            result = event
        elif isinstance(event, ToolCallEvent):
            call = event
            if event.error:
                typer.secho(f"→ {event.name}  {event.error}", fg="yellow")
        elif call is not None and not event.error:
            _echo_search(event, call)
    if result is None:
        raise AssertionError("run_agent always ends with an AgentResult")
    return result


def _counters(result: AgentResult) -> str:
    if result.input_tokens is None and result.output_tokens is None:
        tokens = "n/a"
    else:
        tokens = f"{(result.input_tokens or 0) + (result.output_tokens or 0):,}"
    return (
        f"{result.llm_calls} LLM calls · {result.tool_calls} searches · {tokens} tokens · "
        f"stopped: {result.stopped}"
    )


@app.command("ask")
def ask_cmd(
    question: Annotated[str, typer.Argument(help="Question about Kubernetes.")],
    k: Annotated[int, typer.Option("-k", help="Passages sent to the LLM.")] = 5,
    rerank: Annotated[bool, typer.Option(help="Rerank candidates before answering.")] = True,
    generated: Annotated[bool, typer.Option(help="Include auto-generated reference pages.")] = True,
    rewrite: Annotated[
        bool, typer.Option(help="LLM rewrites the question for search (one extra LLM call).")
    ] = False,
    agent: Annotated[
        bool, typer.Option(help="The LLM decides what to search; prints each step.")
    ] = False,
) -> None:
    """Answer a question from the docs, with [n] citations to the passages used."""
    if agent and rewrite:
        raise typer.BadParameter(
            "--agent cannot be combined with --rewrite: the agent writes its own queries."
        )
    settings = Settings()
    config = ask_config(k=k, rerank=rerank, generated=generated, rewrite=rewrite)
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc

    agent_result: AgentResult | None = None
    with conn:
        register_vector(conn)
        reranker = make_reranker(settings) if rerank else None
        rewriter = QueryRewriter(make_generator(settings)) if rewrite else None
        retriever = Retriever(conn, make_embedder(settings), reranker, rewriter)
        try:
            if agent:
                agent_result = _run_agent_with_trace(question, settings, retriever, config)
                answer = agent_result.answer
            else:
                answer = answer_question(
                    question,
                    retriever,
                    make_generator(settings),
                    config,
                    on_text=lambda piece: typer.echo(piece, nl=False),
                )
        except (anthropic.AnthropicError, openai.OpenAIError) as exc:
            # The agent may run on its own backend, so name that one in the message and hint.
            failed = settings
            if agent:
                failed = settings.model_copy(
                    update={"llm_backend": settings.agent_backend or settings.llm_backend}
                )
            typer.secho(f"\nLLM call failed ({failed.llm_backend}): {exc}", fg="red", err=True)
            typer.echo(llm_error_hint(exc, failed, agent=agent), err=True)
            raise typer.Exit(1) from exc

    if agent_result is not None:
        typer.echo(f"\n{answer.text}")  # the agent does not stream, so the text comes here
    elif answer.refused and not answer.retrieved:
        typer.echo(answer.text)
    _print_sources(answer)
    if agent_result is not None:
        typer.echo(_counters(agent_result))


@app.command("eval")
def eval_cmd(
    golden: Annotated[Path, typer.Option(help="Golden set (JSONL).")] = Path("evals/golden.jsonl"),
    config: Annotated[
        list[str] | None, typer.Option("--config", help="Ablation name (repeatable). Default: all.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions.")] = None,
    out: Annotated[Path, typer.Option(help="Results directory.")] = Path("evals/results"),
    corpus: Annotated[Path, typer.Option(help="Corpus root (for CORPUS_REF).")] = Path(
        "data/k8s-docs"
    ),
) -> None:
    """Score retrieval configs on the golden set: hit@k, recall@k, MRR, nDCG@10, latency."""
    names = config or list(ABLATIONS)
    unknown = [n for n in names if n not in ABLATIONS]
    if unknown:
        raise typer.BadParameter(f"unknown config {unknown}; choose from {list(ABLATIONS)}")
    items = load_golden(golden)[:limit]
    settings = Settings()
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc

    with conn:
        register_vector(conn)
        known = {row[0] for row in conn.execute("SELECT source FROM documents")}
        missing = unknown_sources(items, known)
        if missing:
            typer.secho(f"Golden sources not in the index: {missing}", fg="red", err=True)
            raise typer.Exit(1)
        needs_reranker = any(ABLATIONS[n].rerank for n in names)
        reranker = make_reranker(settings) if needs_reranker else None
        needs_rewriter = any(ABLATIONS[n].rewrite for n in names)
        rewriter = (
            QueryRewriter(RetryingGenerator(make_generator(settings))) if needs_rewriter else None
        )
        retriever = Retriever(conn, make_embedder(settings), reranker, rewriter)
        reports = []
        try:
            for name in names:
                typer.echo(f"Evaluating {name} on {len(items)} questions...", err=True)
                reports.append(evaluate(retriever, items, ABLATIONS[name], name=name))
        except (anthropic.AnthropicError, openai.OpenAIError) as exc:
            typer.secho(f"LLM call failed (query rewriting): {exc}", fg="red", err=True)
            typer.echo(llm_error_hint(exc, settings), err=True)
            raise typer.Exit(1) from exc

    ref_file = corpus / "CORPUS_REF"
    meta = {
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_ref": ref_file.read_text().strip() if ref_file.exists() else "unknown",
        "fingerprint": settings.pipeline_fingerprint(),
        "reranker_model": settings.reranker_model,
        "n_questions": len(items),
        "golden": str(golden),
    }
    typer.echo(to_markdown(reports))
    typer.echo("By question origin (human-written vs LLM-drafted):")
    typer.echo(breakdown_markdown(reports, items, group=lambda item: item.origin))
    path = write_results(out, reports, meta)
    typer.secho(f"Saved {path}", fg="green")


@app.command("eval-answers")
def eval_answers_cmd(
    config: Annotated[
        str, typer.Option(help="Retrieval ablation to answer with.")
    ] = "hybrid+rerank",
    golden: Annotated[Path, typer.Option(help="Golden set (JSONL).")] = Path("evals/golden.jsonl"),
    unanswerable: Annotated[Path, typer.Option(help="Questions to refuse.")] = Path(
        "evals/unanswerable.jsonl"
    ),
    judge_backend: Annotated[
        str | None, typer.Option(help="anthropic | openrouter | fake (default: RAG_LLM_BACKEND).")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N of each set.")] = None,
    out: Annotated[Path, typer.Option(help="Results directory.")] = Path("evals/results"),
    corpus: Annotated[Path, typer.Option(help="Corpus root (for CORPUS_REF).")] = Path(
        "data/k8s-docs"
    ),
    resume: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Checkpoint of an interrupted run to continue."
        ),
    ] = None,
    mode: Annotated[
        str, typer.Option(help="pipeline (fixed retrieve-then-answer) | agent (LLM searches).")
    ] = "pipeline",
    k: Annotated[
        int | None, typer.Option(min=1, help="Passages per search (default: the config's k).")
    ] = None,
    judge_model: Annotated[
        str | None, typer.Option(help="Judge model id (default: the judge backend's model).")
    ] = None,
    judge_max_tokens: Annotated[
        int,
        typer.Option(
            min=1,
            help="Output token budget for the judge; reasoning models think before answering.",
        ),
    ] = 8192,
    price_in: Annotated[float, typer.Option(help="USD per million input tokens.")] = 0.0,
    price_out: Annotated[float, typer.Option(help="USD per million output tokens.")] = 0.0,
) -> None:
    """Generate answers, grade them with an LLM judge, check citations and refusals."""
    if config not in ABLATIONS:
        raise typer.BadParameter(f"unknown config {config!r}; choose from {list(ABLATIONS)}")
    if mode not in ("pipeline", "agent"):
        raise typer.BadParameter(f"unknown mode {mode!r}; choose pipeline or agent")
    if mode == "agent" and k is not None:
        raise typer.BadParameter("--k applies to pipeline mode only; the agent picks k per search")
    settings = Settings()
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = resume or out / f"{timestamp.replace(':', '-')}-answers.partial.jsonl"
    typer.echo(f"Checkpoint: {checkpoint}", err=True)
    judge_backend_name = judge_backend or settings.llm_backend
    judge_updates: dict[str, str | int] = {
        "llm_backend": judge_backend_name,
        "llm_max_tokens": judge_max_tokens,
    }
    if judge_model:
        model_field = {"openrouter": "openrouter_model", "anthropic": "llm_model"}.get(
            judge_backend_name
        )
        if model_field:
            judge_updates[model_field] = judge_model
    judge_settings = settings.model_copy(update=judge_updates)
    golden_items = load_golden(golden)[:limit]
    refuse_items = load_unanswerable(unanswerable)[:limit] if unanswerable.exists() else []
    retriever_config = ABLATIONS[config]
    if k is not None:
        retriever_config = replace(
            retriever_config, k=k, candidates=max(retriever_config.candidates, k)
        )
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc

    with conn:
        register_vector(conn)
        reranker = make_reranker(settings) if retriever_config.rerank else None
        retriever = Retriever(conn, make_embedder(settings), reranker)
        try:
            if mode == "agent":
                answerer = agent_answerer(
                    lambda: RetryingGenerator(make_tool_generator(settings)),
                    SearchTool(retriever, retriever_config),
                    agent_limits(settings),
                )
            else:
                answerer = pipeline_answerer(
                    retriever, RetryingGenerator(make_generator(settings)), retriever_config
                )
            report = evaluate_answers(
                golden_items,
                refuse_items,
                answerer,
                make_generator(judge_settings),
                on_progress=lambda done, total: typer.echo(f"  {done}/{total}", err=True),
                price_in=price_in,
                price_out=price_out,
                checkpoint=checkpoint,
            )
        except (anthropic.AnthropicError, openai.OpenAIError) as exc:
            typer.secho(f"LLM call failed: {exc}", fg="red", err=True)
            typer.echo(llm_error_hint(exc, settings), err=True)
            # Re-running without --resume would start a new checkpoint and spend the quota again.
            typer.echo(f"Resume with: --resume {checkpoint}", err=True)
            raise typer.Exit(1) from exc

    ref_file = corpus / "CORPUS_REF"
    meta = {
        "timestamp": timestamp,
        "corpus_ref": ref_file.read_text().strip() if ref_file.exists() else "unknown",
        "fingerprint": settings.pipeline_fingerprint(),
        "retrieval": config,
        "mode": mode,
        # In agent mode the model picks k per search, so the --k value is not recorded.
        "k": None if mode == "agent" else retriever_config.k,
        "llm": _llm_label(settings),
        "judge": _llm_label(judge_settings),
        "judge_max_tokens": judge_max_tokens,
        "prices": {"in": price_in, "out": price_out},
    }
    if mode == "agent":
        agent_settings = settings.model_copy(
            update={"llm_backend": settings.agent_backend or settings.llm_backend}
        )
        meta["agent_llm"] = _llm_label(agent_settings, settings.agent_model)
        meta["agent_limits"] = asdict(agent_limits(settings))
        meta["agent_k"] = "model-chosen per call (default 5, max 10)"
    typer.echo(answers_markdown(report))
    path = write_answer_results(out, report, meta)
    typer.secho(f"Saved {path}", fg="green")


def _llm_label(settings: Settings, model: str | None = None) -> str:
    match settings.llm_backend:
        case "anthropic":
            return f"anthropic:{model or settings.llm_model}"
        case "openrouter":
            return f"openrouter:{model or settings.openrouter_model}"
    return settings.llm_backend


@app.command("review-drafts")
def review_drafts_cmd(
    drafts: Annotated[Path, typer.Option(help="LLM drafts to review.")] = Path(
        "evals/drafts.jsonl"
    ),
    golden: Annotated[Path, typer.Option(help="Golden set to append to.")] = Path(
        "evals/golden.jsonl"
    ),
    reviewed: Annotated[Path, typer.Option(help="Decision ledger (enables resume).")] = Path(
        "evals/drafts-reviewed.txt"
    ),
) -> None:
    """Review drafted questions one by one: [y]es, [e]dit, [n]o, [q]uit (resumable)."""
    pending = pending_drafts(drafts, reviewed)
    counts = {"accepted": 0, "rejected": 0}
    for position, row in enumerate(pending, start=1):
        typer.secho(f"\n[{position}/{len(pending)}] {row['id']}", fg="cyan")
        typer.secho(f"Q: {row['question']}", bold=True)
        typer.echo(f"A: {row.get('reference_answer') or '-'}")
        typer.echo(f"Source: {', '.join(row['relevant_sources'])}")
        typer.echo(f"Tags: {', '.join(row.get('tags', []))}")
        choice = typer.prompt("[y]es / [e]dit / [n]o / [q]uit", default="y").strip().lower()
        if choice.startswith("q"):
            break
        edited = typer.prompt("New question") if choice.startswith("e") else None
        decision = "rejected" if choice.startswith("n") else "accepted"
        record_decision(row, decision, golden, reviewed, edited_question=edited)
        counts[decision] += 1
    left = len(pending_drafts(drafts, reviewed))
    typer.secho(
        f"{counts['accepted']} accepted, {counts['rejected']} rejected, {left} left.", fg="green"
    )


@app.command("draft-questions")
def draft_questions_cmd(
    n: Annotated[int, typer.Option(help="Chunks to sample (one per page).")] = 100,
    seed: Annotated[str, typer.Option(help="Sampling seed (same seed = same sample).")] = "v1",
    out: Annotated[Path, typer.Option(help="Drafts file to review.")] = Path("evals/drafts.jsonl"),
) -> None:
    """Draft golden questions with the configured LLM, for human review before use."""
    settings = Settings()
    try:
        conn = connect(settings)
    except psycopg.OperationalError as exc:
        raise _database_unreachable(exc) from exc
    with conn:
        samples = sample_chunks(conn, n=n, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_rows = (
        [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
        if out.exists()
        else []
    )
    if done_rows:
        typer.echo(f"Resuming: {len(done_rows)} drafts already in {out}", err=True)
    typer.echo(f"Drafting from {len(samples)} chunks with {settings.llm_backend}...", err=True)
    with out.open("a", encoding="utf-8") as sink:

        def save(row: dict) -> None:
            sink.write(json.dumps(row) + "\n")
            sink.flush()  # every draft survives a crash or Ctrl-C

        try:
            draft_items(
                samples,
                make_generator(settings),
                skip_ids={row["id"] for row in done_rows},
                on_row=save,
                on_progress=lambda done, kept: typer.echo(
                    f"  {done}/{len(samples)} sampled, {kept} new", err=True
                ),
            )
        except (anthropic.AnthropicError, openai.OpenAIError) as exc:
            typer.secho(f"LLM call failed: {exc}", fg="red", err=True)
            typer.echo(llm_error_hint(exc, settings) + " Re-run to resume.", err=True)
            raise typer.Exit(1) from exc

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    flagged = sum("high-overlap" in row["tags"] for row in rows)
    typer.secho(
        f"{len(rows)} drafts in {out} ({flagged} tagged high-overlap). "
        "Review them before merging into the golden set.",
        fg="green",
    )


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
