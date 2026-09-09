import math

from eval.metrics import (
    citation_coverage,
    grounded_answer_quality,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_relevance,
)


def test_recall_at_k_uses_relevant_document_ids():
    assert recall_at_k(["a", "b", "c"], {"b", "d"}, 2) == 0.5
    assert recall_at_k(["a", "b", "d"], {"b", "d"}, 3) == 1.0


def test_recall_at_k_is_one_when_no_relevant_items_exist():
    assert recall_at_k(["a", "b"], set(), 2) == 1.0


def test_reciprocal_rank_is_first_relevant_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["b", "a", "c"], {"b", "c"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_at_k_rewards_correct_order():
    ideal = ndcg_at_k(["a", "b", "c"], {"a": 2, "b": 1, "c": 0}, 3)
    reversed_order = ndcg_at_k(["b", "a", "c"], {"a": 2, "b": 1, "c": 0}, 3)
    assert math.isclose(ideal, 1.0)
    assert 0.0 < reversed_order < ideal


def test_ndcg_at_k_handles_missing_relevance_labels():
    assert ndcg_at_k(["a", "unknown"], {"a": 2}, 2) == 1.0


def test_existing_metrics_remain_deterministic():
    assert retrieval_relevance(["A relevant chunk"], ["A relevant chunk"]) == 1.0
    assert citation_coverage("Answer [Source 1]", ["Source 1"]) == 1.0
    assert grounded_answer_quality("Retention is seven years.", ["seven years"]) == 1.0
