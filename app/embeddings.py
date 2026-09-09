import hashlib
import math
import re

import numpy as np

from app.config import settings


class LocalHashEmbedding:
    """Dependency-free deterministic embedding baseline.

    The provider is intentionally replaceable. Production deployments can swap this
    implementation for a hosted or transformer embedding model without changing the
    retrieval API.
    """

    def __init__(self, dimension: int | None = None):
        self.dimension = dimension or settings.embedding_dimension

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[idx] += sign
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm:
            vector /= norm
        return vector.tolist()


embedding_model = LocalHashEmbedding()
