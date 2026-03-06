from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    result: str
    dominant_cluster: int
    membership: np.ndarray


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.86, candidate_clusters: int = 3) -> None:
        self.similarity_threshold = similarity_threshold
        self.candidate_clusters = candidate_clusters
        self._entries: List[CacheEntry] = []
        self._cluster_index: Dict[int, List[int]] = {}
        self._hit_count = 0
        self._miss_count = 0
        self._lock = RLock()

    @staticmethod
    def _cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    def lookup(
        self,
        query_embedding: np.ndarray,
        membership: np.ndarray,
    ) -> Optional[dict]:
        with self._lock:
            top_clusters = np.argsort(membership)[::-1][: self.candidate_clusters]
            candidate_ids: List[int] = []
            seen = set()
            for cluster in top_clusters:
                for idx in self._cluster_index.get(int(cluster), []):
                    if idx not in seen:
                        seen.add(idx)
                        candidate_ids.append(idx)

            best_idx = -1
            best_score = -1.0
            for idx in candidate_ids:
                score = self._cosine_sim(query_embedding, self._entries[idx].embedding)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0 and best_score >= self.similarity_threshold:
                self._hit_count += 1
                match = self._entries[best_idx]
                return {
                    "cache_hit": True,
                    "matched_query": match.query,
                    "similarity_score": best_score,
                    "result": match.result,
                    "dominant_cluster": match.dominant_cluster,
                }

            self._miss_count += 1
            return None

    def add(
        self,
        query: str,
        query_embedding: np.ndarray,
        result: str,
        dominant_cluster: int,
        membership: np.ndarray,
    ) -> None:
        with self._lock:
            entry = CacheEntry(
                query=query,
                embedding=query_embedding.copy(),
                result=result,
                dominant_cluster=dominant_cluster,
                membership=membership.copy(),
            )
            idx = len(self._entries)
            self._entries.append(entry)

            top_clusters = np.argsort(membership)[::-1][: self.candidate_clusters]
            for cluster in top_clusters:
                cluster_id = int(cluster)
                self._cluster_index.setdefault(cluster_id, []).append(idx)

    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            rate = (self._hit_count / total) if total else 0.0
            return {
                "total_entries": len(self._entries),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(rate, 3),
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cluster_index.clear()
            self._hit_count = 0
            self._miss_count = 0
