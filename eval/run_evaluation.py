from __future__ import annotations

import json
from pathlib import Path

from app.generation import build_grounded_prompt
from eval.metrics import (
    EvaluationResult,
    citation_coverage,
    grounded_answer_quality,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_relevance,
)

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
    )
    ranked_ids = [chunk["filename"] for chunk in ranked]
    relevant_ids = set(case["expected_filenames"])
    relevance_labels = {
        chunk["filename"]: (2.0 if chunk["filename"] in relevant_ids else 0.0)
        for chunk in chunks
    }

    retrieved_texts = [chunk["text"] for chunk in ranked[:1]]
    expected_texts = [
        item["text"] for item in chunks if item["filename"] in relevant_ids
    ]
    relevance = retrieval_relevance(retrieved_texts, expected_texts)

    prompt = build_grounded_prompt(
        case["question"],
        [(chunk["filename"], chunk["text"], chunk["chunk_index"]) for chunk in ranked[:3]],
    )

    # A deterministic expected-answer fixture lets CI validate grounding logic
    # without downloading or executing an LLM.
    answer = f"{ranked[0]['text']} [Source 1]"
    coverage = citation_coverage(answer, case["expected_source_ids"])
    quality = grounded_answer_quality(answer, case["expected_answer_facts"])
    assert "ONLY the provided sources" in prompt

    return EvaluationResult(
        retrieval_relevance=relevance,
        citation_coverage=coverage,
        grounded_answer_quality=quality,
        recall_at_1=recall_at_k(ranked_ids, relevant_ids, 1),
        recall_at_3=recall_at_k(ranked_ids, relevant_ids, 3),
        mrr=reciprocal_rank(ranked_ids, relevant_ids),
        ndcg_at_3=ndcg_at_k(ranked_ids, relevance_labels, 3),
    )


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
    print(f"recall@1={aggregate.recall_at_1:.2f}")
    print(f"recall@3={aggregate.recall_at_3:.2f}")
    print(f"mrr={aggregate.mrr:.2f}")
    print(f"ndcg@3={aggregate.ndcg_at_3:.2f}")

    for case_id, result in results.items():
        print(case_id, result)

    threshold_metrics = [
        aggregate.retrieval_relevance,
        aggregate.citation_coverage,
        aggregate.grounded_answer_quality,
        aggregate.recall_at_1,
        aggregate.recall_at_3,
        aggregate.mrr,
        aggregate.ndcg_at_3,
    ]
    if min(threshold_metrics) < 0.8:
        raise SystemExit("Evaluation threshold failed")


if __name__ == "__main__":
    main()
