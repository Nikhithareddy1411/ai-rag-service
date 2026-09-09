from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    retrieval_relevance: float
    citation_coverage: float
    grounded_answer_quality: float


def retrieval_relevance(retrieved_texts: list[str], expected_texts: list[str]) -> float:
    if not expected_texts:
        return 1.0
    expected = {normalize(text) for text in expected_texts}
    retrieved = {normalize(text) for text in retrieved_texts}
    return len(expected & retrieved) / len(expected)


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
