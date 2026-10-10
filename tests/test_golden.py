import json

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
