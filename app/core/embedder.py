from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: Iterable[str], batch_size: int = 128) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.astype(np.float32)
