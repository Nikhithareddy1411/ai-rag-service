import numpy as np
import pytest

from app.embeddings import TransformerEmbedding


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.ones(384, dtype=np.float32) / np.sqrt(384)
        return np.tile(np.ones(384, dtype=np.float32) / np.sqrt(384), (len(texts), 1))


def test_embedding_is_normalized_and_has_expected_dimension():
    provider = TransformerEmbedding("fake", 384)
    provider._model = FakeModel()

    vector = provider.embed("compliance reporting")

    assert len(vector) == 384
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_batch_embedding_returns_one_vector_per_text():
    provider = TransformerEmbedding("fake", 384)
    provider._model = FakeModel()

    vectors = provider.embed_many(["one", "two", "three"])

    assert len(vectors) == 3
    assert all(len(vector) == 384 for vector in vectors)


def test_model_dimension_mismatch_is_rejected():
    class WrongDimensionModel(FakeModel):
        def get_sentence_embedding_dimension(self):
            return 768

    provider = TransformerEmbedding("fake", 384)
    provider._model = WrongDimensionModel()

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        _ = provider.model
