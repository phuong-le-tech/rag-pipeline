import json

import pytest

from rag.evals.judge import (
    Judgement,
    citation_stats,
    judge_prompt,
    load_unanswerable,
    parse_judgement,
)
from rag.retrieval.search import Hit


def hit(i, content):
    return Hit(i, 1.0, f"s{i}.md", f"https://k8s/{i}", "T", f"Page {i}", content, False)


HITS = [hit(1, "Use kubectl rollout undo to revert."), hit(2, "History is capped by a limit.")]


# --- prompt ---------------------------------------------------------------------------------


def test_judge_prompt_includes_passages_answer_and_reference():
    system, user = judge_prompt("How to roll back?", HITS, "Run undo [1].", "kubectl rollout undo")

    assert "[1] Page 1" in user and "rollout undo to revert" in user
    assert "Run undo [1]." in user
    assert "kubectl rollout undo" in user
    assert "JSON" in system and "supported" in system


def test_judge_prompt_without_reference_skips_correctness():
    _, user = judge_prompt("q", HITS, "a [1].", None)

    assert "Reference answer" not in user


# --- parsing --------------------------------------------------------------------------------


def test_parse_judgement_computes_faithfulness_and_lists_unsupported():
    text = json.dumps(
        {
            "claims": [
                {"text": "Use kubectl rollout undo.", "supported": True},
                {"text": "It lists every pod.", "supported": False},
                {"text": "History is capped.", "supported": True},
            ],
            "correct": "partial",
        }
    )

    judgement = parse_judgement(text)

    assert judgement == Judgement(
        faithfulness=pytest.approx(2 / 3),
        n_claims=3,
        unsupported=("It lists every pod.",),
        correct="partial",
    )


def test_parse_judgement_tolerates_prose_and_missing_correctness():
    text = 'Here you go:\n```json\n{"claims": [{"text": "x", "supported": true}]}\n```'

    judgement = parse_judgement(text)

    assert judgement.faithfulness == 1.0
    assert judgement.correct is None


def test_parse_judgement_with_no_claims_is_fully_faithful():
    assert parse_judgement('{"claims": [], "correct": "no"}').faithfulness == 1.0


@pytest.mark.parametrize("text", ["not json", '{"claims": "oops"}', '{"correct": "maybe"}'])
def test_parse_judgement_returns_none_on_garbage(text):
    assert parse_judgement(text) is None


# --- citation stats -------------------------------------------------------------------------


def test_citation_stats_counts_cited_sentences_and_invalid_numbers():
    answer = "Run undo [1]. Then check status. Use the limit [2]. Also see [9]."

    stats = citation_stats(answer, num_sources=2)

    assert stats == {"sentences": 4, "cited_sentences": 3, "invalid_citations": 1}


def test_citation_stats_ignores_code_blocks_and_bullets_markers():
    answer = "Run this [1]:\n\n```shell\nkubectl rollout undo deploy/x\n```\n\n- Check it [1]."

    stats = citation_stats(answer, num_sources=1)

    assert stats["sentences"] == 2
    assert stats["cited_sentences"] == 2


# --- unanswerable set -----------------------------------------------------------------------


def test_load_unanswerable_reads_questions(tmp_path):
    path = tmp_path / "u.jsonl"
    path.write_text(
        '{"id": "u1", "question": "How do I set an AWS Lambda timeout?", "origin": "human"}\n\n'
    )

    (item,) = load_unanswerable(path)

    assert item.id == "u1"
    assert item.question.startswith("How do I set")


def test_load_unanswerable_rejects_duplicates(tmp_path):
    row = '{"id": "u1", "question": "q", "origin": "human"}\n'
    path = tmp_path / "u.jsonl"
    path.write_text(row + row)

    with pytest.raises(ValueError, match="duplicate"):
        load_unanswerable(path)
