import pytest

from rag.evals.drafting import (
    ChunkSample,
    Draft,
    draft_items,
    draft_prompt,
    lexical_overlap,
    parse_draft,
    with_retry,
)
from rag.generate.generator import FakeGenerator


def sample(i=1, source="concepts/storage/volumes.md", section="concepts"):
    return ChunkSample(
        chunk_id=i,
        source=source,
        section=section,
        heading="Volumes > emptyDir",
        content="An emptyDir volume is created when a Pod is assigned to a node and is "
        "deleted when the Pod is removed from that node.",
    )


# --- parsing --------------------------------------------------------------------------------


def test_parse_draft_reads_json_object():
    text = '{"question": "When is emptyDir data deleted?", "answer": "When the Pod is removed."}'

    assert parse_draft(text) == Draft("When is emptyDir data deleted?", "When the Pod is removed.")


def test_parse_draft_finds_json_inside_prose_or_code_fence():
    text = 'Sure!\n```json\n{"question": "Q?", "answer": "A."}\n```\nHope this helps.'

    assert parse_draft(text) == Draft("Q?", "A.")


@pytest.mark.parametrize(
    "text",
    ['{"skip": true}', "no json here", '{"question": "", "answer": "x"}', '{"answer": "x"}'],
)
def test_parse_draft_returns_none_for_skips_and_garbage(text):
    assert parse_draft(text) is None


# --- overlap --------------------------------------------------------------------------------


def test_lexical_overlap_is_share_of_question_content_words_in_passage():
    passage = "An emptyDir volume is deleted when the Pod is removed from the node."

    copied = lexical_overlap("When is an emptyDir volume deleted from the node?", passage)
    paraphrased = lexical_overlap("Does scratch space survive if my pod goes away?", passage)

    assert copied > 0.7
    assert paraphrased < 0.3
    assert lexical_overlap("the a of", passage) == 0.0


# --- retry ----------------------------------------------------------------------------------


class Flaky:
    def __init__(self, failures, exc=None):
        self.failures = failures
        self.exc = exc or RuntimeError("429")
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc
        return "ok"


def test_with_retry_backs_off_exponentially_then_succeeds():
    sleeps = []
    fn = Flaky(failures=2)

    result = with_retry(
        fn, retry_on=(RuntimeError,), attempts=4, base_delay=1.0, sleep=sleeps.append
    )

    assert result == "ok"
    assert sleeps == [1.0, 2.0]


def test_with_retry_gives_up_after_attempts():
    fn = Flaky(failures=10)

    with pytest.raises(RuntimeError):
        with_retry(fn, retry_on=(RuntimeError,), attempts=3, base_delay=0.1, sleep=lambda _s: None)
    assert fn.calls == 3


def test_with_retry_does_not_retry_other_errors():
    fn = Flaky(failures=1, exc=ValueError("bad"))

    with pytest.raises(ValueError):
        with_retry(fn, retry_on=(RuntimeError,), attempts=3, base_delay=0.1, sleep=lambda _s: None)
    assert fn.calls == 1


# --- prompt + drafting loop -----------------------------------------------------------------


def test_draft_prompt_contains_passage_and_asks_for_json():
    system, user = draft_prompt(sample())

    assert "emptyDir" in user and "Volumes > emptyDir" in user
    assert "JSON" in system and "skip" in system


def test_draft_items_builds_golden_rows_tags_overlap_and_skips():
    gen = ScriptedGenerator(
        [
            '{"question": "Does scratch space survive if my pod goes away?", "answer": "No."}',
            '{"skip": true}',
            '{"question": "When is an emptyDir volume deleted from the node?", '
            '"answer": "On removal."}',
        ]
    )
    samples = [sample(1), sample(2, source="tasks/x.md", section="tasks"), sample(3)]

    rows = draft_items(samples, gen, sleep=lambda _s: None)

    assert [r["id"] for r in rows] == ["llm-0001", "llm-0003"]
    first, second = rows
    assert first["relevant_sources"] == ["concepts/storage/volumes.md"]
    assert first["origin"] == "llm"
    assert first["tags"] == ["llm-draft", "concepts"]
    assert "high-overlap" in second["tags"]
    assert first["reference_answer"] == "No."


def test_draft_items_streams_rows_and_resumes_by_skipping_done_ids():
    gen = ScriptedGenerator(['{"question": "Second question here?", "answer": "A."}'])
    streamed = []

    rows = draft_items(
        [sample(1), sample(2)],
        gen,
        sleep=lambda _s: None,
        skip_ids={"llm-0001"},
        on_row=streamed.append,
    )

    assert [r["id"] for r in rows] == ["llm-0002"]
    assert streamed == rows
    assert len(gen.calls) == 1  # the already-drafted sample cost no LLM call


@pytest.mark.integration
def test_sample_chunks_is_deterministic_one_per_page_with_generated_share(conn):
    from rag.embed import FakeEmbedder
    from rag.evals.drafting import sample_chunks
    from rag.ingest.loaders import Document
    from rag.ingest.pipeline import ingest

    body = "\n\n".join(f"Paragraph {j} " + "word " * 80 for j in range(4))
    docs = [
        Document(f"s/p{i}.md", "u", f"P{i}", "s", f"# P{i}\n\n{body}", f"h{i}", i >= 8)
        for i in range(10)
    ]
    ingest(conn, docs, FakeEmbedder(8), max_chars=500, fingerprint="f")

    first = sample_chunks(conn, n=5, seed="s1", generated_share=0.4)
    again = sample_chunks(conn, n=5, seed="s1", generated_share=0.4)
    other = sample_chunks(conn, n=5, seed="s2", generated_share=0.4)

    assert first == again
    assert [s.chunk_id for s in first] != [s.chunk_id for s in other]
    assert len({s.source for s in first}) == 5
    assert sum(s.source in {"s/p8.md", "s/p9.md"} for s in first) == 2


class ScriptedGenerator(FakeGenerator):
    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.answers.pop(0)
