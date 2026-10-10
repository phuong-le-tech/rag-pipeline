import pytest

from rag.retrieval.fusion import rrf


def test_item_ranked_high_in_both_lists_wins():
    dense = [1, 2, 3]
    keyword = [2, 4, 1]

    fused = rrf([dense, keyword], k=60)

    assert [cid for cid, _ in fused] == [2, 1, 4, 3]


def test_scores_follow_reciprocal_rank_formula():
    fused = dict(rrf([[10, 20], [20]], k=60))

    assert fused[10] == pytest.approx(1 / 61)
    assert fused[20] == pytest.approx(1 / 62 + 1 / 61)


def test_limit_truncates_and_empty_lists_are_fine():
    assert rrf([[1, 2, 3], []], k=60, limit=2) == rrf([[1, 2, 3]], k=60)[:2]
    assert rrf([], k=60) == []


def test_ties_break_by_first_appearance_for_determinism():
    fused = rrf([[1], [2]], k=60)

    assert [cid for cid, _ in fused] == [1, 2]


def test_duplicate_within_one_list_counts_once_at_best_rank():
    fused = dict(rrf([[7, 7, 8]], k=60))

    assert fused[7] == pytest.approx(1 / 61)
    assert fused[8] == pytest.approx(1 / 63)


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        rrf([[1]], k=0)
