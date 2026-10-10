import math

import pytest

from rag.evals.metrics import hit_at_k, mean, ndcg_at_k, recall_at_k, reciprocal_rank

# Ranked list of retrieved chunks, represented by their source page (duplicates = same page).
RANKED = ["a.md", "x.md", "b.md", "a.md", "y.md"]
RELEVANT = {"a.md", "b.md"}


def test_hit_at_k_is_one_when_any_relevant_page_in_top_k():
    assert hit_at_k(RANKED, RELEVANT, k=1) == 1.0
    assert hit_at_k(["x.md", "y.md"], RELEVANT, k=2) == 0.0


def test_recall_at_k_counts_distinct_relevant_pages_found():
    assert recall_at_k(RANKED, RELEVANT, k=2) == 0.5
    assert recall_at_k(RANKED, RELEVANT, k=3) == 1.0


def test_reciprocal_rank_uses_first_relevant_chunk():
    assert reciprocal_rank(RANKED, RELEVANT) == 1.0
    assert reciprocal_rank(["x.md", "y.md", "b.md"], RELEVANT) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x.md"], RELEVANT) == 0.0


def test_ndcg_rewards_relevant_pages_near_the_top():
    perfect = ndcg_at_k(["a.md", "b.md", "x.md"], RELEVANT, k=3)
    late = ndcg_at_k(["x.md", "a.md", "b.md"], RELEVANT, k=3)

    assert perfect == pytest.approx(1.0)
    assert 0 < late < perfect


def test_ndcg_counts_each_relevant_page_once_so_it_never_exceeds_one():
    assert ndcg_at_k(["a.md", "a.md", "a.md"], {"a.md"}, k=3) == pytest.approx(1.0)


def test_ndcg_exact_value():
    # gains at ranks 2 and 3 -> DCG = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    expected = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))

    assert ndcg_at_k(["x.md", "a.md", "b.md"], RELEVANT, k=3) == pytest.approx(expected)


@pytest.mark.parametrize("fn", [hit_at_k, recall_at_k, ndcg_at_k])
def test_metrics_reject_bad_input(fn):
    with pytest.raises(ValueError):
        fn(RANKED, RELEVANT, k=0)
    with pytest.raises(ValueError):
        fn(RANKED, set(), k=3)


def test_mean_of_empty_is_nan_free_zero():
    assert mean([]) == 0.0
    assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)
