from eval.metrics import citation_coverage, grounded_answer_quality, retrieval_relevance


def test_retrieval_relevance_scores_expected_source_hits():
    assert retrieval_relevance(["Policy retains records for seven years."], ["Policy retains records for seven years."]) == 1.0
    assert retrieval_relevance(["Unrelated access policy."], ["Policy retains records for seven years."]) == 0.0


def test_citation_coverage_detects_expected_source_markers():
    assert citation_coverage("Records are retained for seven years. [Source 1]", ["Source 1"]) == 1.0
    assert citation_coverage("Records are retained for seven years.", ["Source 1"]) == 0.0


def test_grounded_answer_quality_scores_expected_facts():
    assert grounded_answer_quality("Records are retained for seven years. [Source 1]", ["seven years"]) == 1.0
    assert grounded_answer_quality("Records are retained for one year. [Source 1]", ["seven years"]) == 0.0
