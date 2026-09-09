from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    retrieval_relevance: float
    citation_coverage: float
    grounded_answer_quality: float
    recall_at_1: float
    recall_at_3: float
    mrr: float
    ndcg_at_3: float


def retrieval_relevance(retrieved_texts: list[str], expected_texts: list[str]) -> float:
    if not expected_texts:
        return 1.0
    expected = {normalize(text) for text in expected_texts}
    retrieved = {normalize(text) for text in retrieved_texts}
    return len(expected & retrieved) / len(expected)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of known relevant items retrieved in the first k results."""
    if not relevant_ids:
        return 1.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result; zero when none is found."""
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, float], k: int) -> float:
    """Normalized discounted cumulative gain using fixed graded relevance labels."""
    ranked = ranked_ids[:k]
    dcg = sum(
        (2.0 ** relevance.get(item_id, 0.0) - 1.0) / math.log2(rank + 1)
        for rank, item_id in enumerate(ranked, start=1)
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(
        (2.0 ** score - 1.0) / math.log2(rank + 1)
        for rank, score in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg else 1.0


def citation_coverage(answer: str, factual_source_ids: list[str]) -> float:
    if not factual_source_ids:
        return 1.0
    cited = set(re.findall(r"\[Source\s+(\d+)\]", answer))
    expected = {source.removeprefix("Source ") for source in factual_source_ids}
    return len(cited & expected) / len(expected)


def grounded_answer_quality(answer: str, expected_facts: list[str]) -> float:
    """Simple deterministic fact-coverage metric for local regression tests."""
    if not expected_facts:
        return 1.0
    normalized_answer = normalize(answer)
    matched = sum(normalize(fact) in normalized_answer for fact in expected_facts)
    return matched / len(expected_facts)


def normalize(value: str) -> str:
    return " ".join(value.lower().split())
