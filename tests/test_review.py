import json

from typer.testing import CliRunner

from rag import cli
from rag.evals.golden import load_golden
from rag.evals.review import pending_drafts, record_decision

runner = CliRunner()


def draft(i, question):
    return {
        "id": f"llm-{i:04d}",
        "question": question,
        "relevant_sources": [f"concepts/p{i}.md"],
        "reference_answer": f"answer {i}",
        "tags": ["llm-draft", "concepts"],
        "origin": "llm",
    }


def setup_files(tmp_path, n=3):
    drafts = tmp_path / "drafts.jsonl"
    drafts.write_text(
        "".join(json.dumps(draft(i, f"Question {i}?")) + "\n" for i in range(1, n + 1))
    )
    return drafts, tmp_path / "golden.jsonl", tmp_path / "reviewed.txt"


def review_args(drafts, golden, reviewed):
    return [
        "review-drafts",
        "--drafts",
        str(drafts),
        "--golden",
        str(golden),
        "--reviewed",
        str(reviewed),
    ]


def test_pending_drafts_skips_already_reviewed(tmp_path):
    drafts, _, reviewed = setup_files(tmp_path)
    reviewed.write_text("llm-0002\trejected\n")

    assert [d["id"] for d in pending_drafts(drafts, reviewed)] == ["llm-0001", "llm-0003"]


def test_record_decision_appends_accepted_rows_to_golden(tmp_path):
    _, golden, reviewed = setup_files(tmp_path)

    record_decision(draft(1, "Q?"), "accepted", golden, reviewed, edited_question=None)
    record_decision(draft(2, "Q2?"), "rejected", golden, reviewed, edited_question=None)
    record_decision(draft(3, "Old?"), "accepted", golden, reviewed, edited_question="New?")

    items = load_golden(golden)
    assert [i.id for i in items] == ["llm-0001", "llm-0003"]
    assert "reviewed" in items[0].tags and "edited" not in items[0].tags
    assert items[1].question == "New?" and "edited" in items[1].tags
    assert reviewed.read_text().splitlines() == [
        "llm-0001\taccepted",
        "llm-0002\trejected",
        "llm-0003\taccepted",
    ]


def test_review_command_accepts_edits_rejects_and_quits(tmp_path):
    drafts, golden, reviewed = setup_files(tmp_path, n=4)
    keys = "y\ne\nBetter question?\nn\nq\n"

    result = runner.invoke(cli.app, review_args(drafts, golden, reviewed), input=keys)

    assert result.exit_code == 0, result.output
    assert "Question 1?" in result.output and "concepts/p1.md" in result.output
    assert [i.question for i in load_golden(golden)] == ["Question 1?", "Better question?"]
    assert len(reviewed.read_text().splitlines()) == 3
    assert "1 left" in result.output


def test_review_command_resumes_where_it_stopped(tmp_path):
    drafts, golden, reviewed = setup_files(tmp_path, n=2)
    runner.invoke(cli.app, review_args(drafts, golden, reviewed), input="y\nq\n")

    result = runner.invoke(cli.app, review_args(drafts, golden, reviewed), input="y\n")

    assert "Question 1?" not in result.output
    assert "Question 2?" in result.output
    assert len(load_golden(golden)) == 2
