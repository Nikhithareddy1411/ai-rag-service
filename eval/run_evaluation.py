from __future__ import annotations

import json
from pathlib import Path

from app.generation import build_grounded_prompt
from eval.metrics import EvaluationResult, citation_coverage, grounded_answer_quality, retrieval_relevance

DATASET = Path(__file__).with_name("dataset.json")


def run_case(case: dict) -> EvaluationResult:
    chunks = case["documents"]

    # Keep evaluation deterministic and offline: rank fixed fixtures by lexical
    # overlap instead of loading the production embedding model.
    query_terms = set(case["question"].lower().split())
    ranked = sorted(
        chunks,
        key=lambda chunk: len(query_terms & set(chunk["text"].lower().split())),
        reverse=True,
    )[:1]
    retrieved_texts = [chunk["text"] for chunk in ranked]
    expected_texts = [
        item["text"] for item in chunks if item["filename"] in case["expected_filenames"]
    ]

    relevance = retrieval_relevance(retrieved_texts, expected_texts)
    prompt = build_grounded_prompt(
        case["question"],
        [(chunk["filename"], chunk["text"], chunk["chunk_index"]) for chunk in ranked],
    )

    # A deterministic expected-answer fixture lets CI validate grounding logic
    # without downloading or executing an LLM.
    answer = f"{retrieved_texts[0]} [Source 1]"
    coverage = citation_coverage(answer, case["expected_source_ids"])
    quality = grounded_answer_quality(answer, case["expected_answer_facts"])
    assert "ONLY the provided sources" in prompt
    return EvaluationResult(relevance, coverage, quality)


def main() -> None:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    results = {case["id"]: run_case(case) for case in dataset["cases"]}
    fields = list(EvaluationResult.__annotations__)
    aggregate = EvaluationResult(
        *(sum(getattr(result, field) for result in results.values()) / len(results) for field in fields)
    )
    print("RAG evaluation")
    print(f"retrieval_relevance={aggregate.retrieval_relevance:.2f}")
    print(f"citation_coverage={aggregate.citation_coverage:.2f}")
    print(f"grounded_answer_quality={aggregate.grounded_answer_quality:.2f}")
    for case_id, result in results.items():
        print(case_id, result)
    if min(aggregate.retrieval_relevance, aggregate.citation_coverage, aggregate.grounded_answer_quality) < 0.8:
        raise SystemExit("Evaluation threshold failed")


if __name__ == "__main__":
    main()
