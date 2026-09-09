from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


class TransformerEmbedding:
    """Local Sentence Transformer embedding provider.

    The model is downloaded once from Hugging Face on first use and then runs
    locally. Embeddings are normalized so dot product is equivalent to cosine
    similarity in pgvector.
    """

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            actual_dimension = self._model.get_sentence_embedding_dimension()
            if actual_dimension != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: configured={self.dimension}, "
                    f"model={actual_dimension}"
                )
        return self._model

    def embed(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.astype("float32").tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


@lru_cache
def get_embedding_model() -> TransformerEmbedding:
    return TransformerEmbedding(
        model_name=settings.embedding_model_name,
        dimension=settings.embedding_dimension,
    )


embedding_model = get_embedding_model()
