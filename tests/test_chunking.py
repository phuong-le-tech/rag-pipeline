from rag.ingest.chunking import Chunk, chunk_markdown


def test_returns_empty_list_for_blank_text():
    assert chunk_markdown("   \n\n  ", max_chars=500) == []


def test_splits_on_headings_and_tracks_heading_path():
    text = "# Guide\nIntro.\n\n## Install\nRun it.\n\n## Usage\nCall it.\n"

    chunks = chunk_markdown(text, max_chars=500)

    assert [c.heading for c in chunks] == ["Guide", "Guide > Install", "Guide > Usage"]
    assert [c.content for c in chunks] == ["Intro.", "Run it.", "Call it."]


def test_sibling_heading_replaces_previous_branch():
    text = "# A\n## B\nb\n### C\nc\n## D\nd\n"

    headings = [c.heading for c in chunk_markdown(text, max_chars=500)]

    assert headings == ["A > B", "A > B > C", "A > D"]


def test_packs_paragraphs_up_to_max_chars_without_splitting_them():
    para = "x" * 40
    text = "# T\n" + "\n\n".join([para] * 5)

    chunks = chunk_markdown(text, max_chars=100)

    assert all(len(c.content) <= 100 for c in chunks)
    assert sum(c.content.count(para) for c in chunks) == 5
    assert len(chunks) == 3


def test_oversized_paragraph_is_hard_split():
    text = "# T\n" + "y" * 250

    chunks = chunk_markdown(text, max_chars=100)

    assert [len(c.content) for c in chunks] == [100, 100, 50]


def test_ignores_hash_lines_inside_code_fences():
    text = "# Real\n```bash\n# not a heading\necho hi\n```\n"

    chunks = chunk_markdown(text, max_chars=500)

    assert len(chunks) == 1
    assert chunks[0].heading == "Real"
    assert "# not a heading" in chunks[0].content


def test_chunks_are_numbered_in_order():
    text = "# A\none\n\n# B\ntwo\n"

    chunks = chunk_markdown(text, max_chars=500)

    assert chunks == [
        Chunk(ord=0, heading="A", content="one"),
        Chunk(ord=1, heading="B", content="two"),
    ]
