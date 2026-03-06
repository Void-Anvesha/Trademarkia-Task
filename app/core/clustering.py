from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class ClusterSelectionResult:
    n_clusters: int
    bic_by_k: Dict[int, float]


class FuzzyClusterer:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model: GaussianMixture | None = None
        self.selection_result: ClusterSelectionResult | None = None

    def select_and_fit(
        self,
        embeddings: np.ndarray,
        k_values: List[int] | None = None,
    ) -> ClusterSelectionResult:
        if k_values is None:
            k_values = list(range(8, 31, 2))

        bic_scores: Dict[int, float] = {}
        best_bic = float("inf")
        best_model: GaussianMixture | None = None
        best_k = k_values[0]

        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="diag",
                random_state=self.random_state,
                reg_covar=1e-5,
                max_iter=300,
            )
            gmm.fit(embeddings)
            bic = float(gmm.bic(embeddings))
            bic_scores[k] = bic
            if bic < best_bic:
                best_bic = bic
                best_model = gmm
                best_k = k

        if best_model is None:
            raise RuntimeError("Failed to fit fuzzy clustering model.")

        self.model = best_model
        self.selection_result = ClusterSelectionResult(n_clusters=best_k, bic_by_k=bic_scores)
        return self.selection_result

    def membership_distribution(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FuzzyClusterer has not been fitted.")
        return self.model.predict_proba(embeddings)

    def dominant_cluster(self, embedding: np.ndarray) -> Tuple[int, np.ndarray]:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        probs = self.membership_distribution(embedding)[0]
        return int(np.argmax(probs)), probs
