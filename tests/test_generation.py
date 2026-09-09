from app.generation import LocalLLM, build_grounded_prompt


def test_grounded_prompt_contains_question_sources_and_citation_rules():
    prompt = build_grounded_prompt(
        "What is the retention period?",
        [("policy.md", "Records are retained for seven years.", 2)],
    )
    assert "What is the retention period?" in prompt
    assert "[Source 1]" in prompt
    assert "Records are retained for seven years." in prompt
    assert "using ONLY the provided sources" in prompt
    assert "Do not invent facts" in prompt


def test_local_llm_is_lazy_and_uses_injected_generator(monkeypatch):
    llm = LocalLLM("test-model")
    calls = []

    class FakeGenerator:
        def __call__(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return [{"generated_text": " Grounded answer [Source 1] "}]

    monkeypatch.setattr(llm, "_pipeline", FakeGenerator())
    answer = llm.generate("context")

    assert answer == "Grounded answer [Source 1]"
    assert calls[0][0] == "context"
    assert calls[0][1]["max_new_tokens"] > 0
    assert calls[0][1]["do_sample"] is False
