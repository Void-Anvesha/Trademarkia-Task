import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from app.config import settings
from app.core.clustering import FuzzyClusterer
from app.core.dataset import load_dataset
from app.core.embedder import Embedder
from app.core.vector_store import VectorStore


def _top_terms_for_cluster(texts: list[str], max_features: int = 2000, top_n: int = 12) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return []
    vec = TfidfVectorizer(stop_words="english", max_features=max_features)
    x = vec.fit_transform(texts)
    scores = np.asarray(x.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = np.argsort(scores)[::-1][:top_n]
    return terms[order].tolist()


def build_artifacts(force_rebuild: bool = False) -> Dict[str, Any]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    marker_file = settings.artifacts_dir / "ready.json"
    if marker_file.exists() and not force_rebuild:
        return json.loads(marker_file.read_text(encoding="utf-8"))

    docs = load_dataset()
    embedder = Embedder(settings.embedding_model_name)
    embeddings = embedder.embed_texts([doc.text for doc in docs])

    clusterer = FuzzyClusterer(random_state=42)
    selection = clusterer.select_and_fit(embeddings)
    memberships = clusterer.membership_distribution(embeddings)
    dominant = memberships.argmax(axis=1)

    vector_store = VectorStore(
        persist_dir=str(settings.cache_dir),
        collection_name=settings.chroma_collection_name,
    )
    ids = [doc.doc_id for doc in docs]
    texts = [doc.text for doc in docs]
    metadatas = []
    for doc, cluster_id, probs in zip(docs, dominant, memberships):
        metadatas.append(
            {
                "target": doc.target,
                "target_name": doc.target_name,
                "dominant_cluster": int(cluster_id),
                "cluster_entropy": float(-np.sum(probs * np.log(probs + 1e-9))),
            }
        )

    vector_store.upsert_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)

    joblib.dump(clusterer.model, settings.artifacts_dir / "gmm.joblib")
    np.save(settings.artifacts_dir / "doc_embeddings.npy", embeddings)
    with (settings.artifacts_dir / "documents.jsonl").open("w", encoding="utf-8") as f:
        for doc, cluster_id, probs in zip(docs, dominant, memberships):
            row = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "target": doc.target,
                "target_name": doc.target_name,
                "dominant_cluster": int(cluster_id),
                "membership": [float(v) for v in probs],
            }
            f.write(json.dumps(row) + "\n")

    cluster_summary = {}
    for cluster_id in range(selection.n_clusters):
        c_texts = [d.text for d, c in zip(docs, dominant) if c == cluster_id]
        cluster_summary[str(cluster_id)] = {
            "size": len(c_texts),
            "top_terms": _top_terms_for_cluster(c_texts),
        }

    sorted_membership = np.sort(memberships, axis=1)
    confidence_margin = sorted_membership[:, -1] - sorted_membership[:, -2]
    boundary_doc_indices = np.argsort(confidence_margin)[:50]
    boundary_examples = [docs[int(i)].text[:280] for i in boundary_doc_indices]

    report = {
        "num_documents": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "selected_clusters": selection.n_clusters,
        "bic_by_k": selection.bic_by_k,
        "cluster_summary": cluster_summary,
        "boundary_examples": boundary_examples,
    }
    (settings.artifacts_dir / "cluster_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    marker_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def artifacts_ready() -> bool:
    return (settings.artifacts_dir / "ready.json").exists()
