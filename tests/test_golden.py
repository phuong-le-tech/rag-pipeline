import json
from pathlib import Path

import pytest

from rag.evals.golden import GoldenItem, load_golden, unknown_sources


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


ROW = {
    "id": "rollback-1",
    "question": "How do I roll back a deployment?",
    "relevant_sources": ["concepts/workloads/controllers/deployment.md"],
    "reference_answer": "kubectl rollout undo deployment/<name>",
    "tags": ["howto", "kubectl"],
    "origin": "human",
}


def test_load_golden_parses_items_and_skips_blank_lines(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(ROW) + "\n\n", encoding="utf-8")

    items = load_golden(path)

    assert items == [
        GoldenItem(
            id="rollback-1",
            question="How do I roll back a deployment?",
            relevant_sources=("concepts/workloads/controllers/deployment.md",),
            reference_answer="kubectl rollout undo deployment/<name>",
            tags=("howto", "kubectl"),
            origin="human",
        )
    ]


def test_optional_fields_default(tmp_path):
    row = {"id": "a", "question": "q?", "relevant_sources": ["x.md"], "origin": "llm"}

    (item,) = load_golden(write_jsonl(tmp_path / "g.jsonl", [row]))

    assert item.reference_answer is None
    assert item.tags == ()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"question": "  "}, "question"),
        ({"relevant_sources": []}, "relevant_sources"),
        ({"origin": "robot"}, "origin"),
    ],
)
def test_invalid_rows_raise_with_line_number(tmp_path, change, message):
    path = write_jsonl(tmp_path / "g.jsonl", [ROW, {**ROW, "id": "b", **change}])

    with pytest.raises(ValueError, match=rf"line 2.*{message}"):
        load_golden(path)


def test_duplicate_ids_rejected(tmp_path):
    path = write_jsonl(tmp_path / "g.jsonl", [ROW, ROW])

    with pytest.raises(ValueError, match="duplicate id 'rollback-1'"):
        load_golden(path)


def test_bad_json_reports_line(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(ROW) + "\n{not json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        load_golden(path)


def test_unknown_sources_lists_typos_against_index():
    items = [
        GoldenItem("a", "q", ("real.md", "typo.md"), None, (), "human"),
        GoldenItem("b", "q", ("real.md",), None, (), "human"),
    ]

    assert unknown_sources(items, known={"real.md"}) == {"a": ("typo.md",)}


def test_load_golden_accepts_multi_origin(tmp_path):
    (item,) = load_golden(write_jsonl(tmp_path / "g.jsonl", [{**ROW, "origin": "multi"}]))

    assert item.origin == "multi"


MULTI_DRAFTS = Path(__file__).resolve().parents[1] / "evals" / "drafts-multi.jsonl"
CORPUS = Path(__file__).resolve().parents[1] / "data" / "k8s-docs"


def test_multi_drafts_are_well_formed():
    items = load_golden(MULTI_DRAFTS)

    assert len(items) == 15
    assert len({item.id for item in items}) == len(items)
    assert {item.origin for item in items} == {"multi"}
    assert all(len(item.relevant_sources) >= 2 for item in items)


def test_multi_draft_sources_exist_in_corpus():
    if not CORPUS.is_dir():
        pytest.skip("data/k8s-docs not checked out")
    items = load_golden(MULTI_DRAFTS)

    missing = {
        item.id: [s for s in item.relevant_sources if not (CORPUS / s).is_file()] for item in items
    }

    assert {k: v for k, v in missing.items() if v} == {}
