from __future__ import annotations

from functools import lru_cache

from app.config import settings


class LocalLLM:
    """Lazy local text-generation provider backed by Hugging Face Transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.llm_model_name
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                task="text-generation",
                model=self.model_name,
                device=settings.llm_device,
            )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        generator = self._get_pipeline()
        kwargs = {
            "max_new_tokens": settings.llm_max_new_tokens,
            "do_sample": settings.llm_do_sample,
            "return_full_text": False,
        }
        if settings.llm_do_sample:
            kwargs["temperature"] = settings.llm_temperature
        result = generator(prompt, **kwargs)
        return result[0]["generated_text"].strip()


@lru_cache
def get_llm() -> LocalLLM:
    return LocalLLM()


def build_grounded_prompt(question: str, chunks: list[tuple[str, str, int]]) -> str:
    sources = "\n\n".join(
        f"[Source {i}] filename={filename} chunk={chunk_index}\n{text}"
        for i, (filename, text, chunk_index) in enumerate(chunks, start=1)
    )
    return f"""You are a grounded question-answering assistant.
Answer the user's question using ONLY the provided sources.
If the sources do not contain enough information, say: "I don't have enough information in the provided sources."
Do not invent facts, citations, filenames, or source numbers.
Cite every factual claim using the source marker format [Source N].
Keep the answer concise and directly answer the question.

User question:
{question}

Sources:
{sources}

Answer with inline citations such as [Source 1]."""
