from typing import List

import chromadb
import numpy as np
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_documents(
        self,
        ids: List[str],
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[dict],
    ) -> None:
        max_batch_size = self.client.get_max_batch_size()
        all_embeddings = embeddings.tolist()

        for start in range(0, len(ids), max_batch_size):
            end = start + max_batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=all_embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
