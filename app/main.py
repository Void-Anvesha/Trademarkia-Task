from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.core.embedder import Embedder
from app.core.pipeline import artifacts_ready, build_artifacts
from app.core.semantic_cache import SemanticCache
from app.core.vector_store import VectorStore


class QueryRequest(BaseModel):
    query: str


def _build_result_from_retrieval(retrieval: dict) -> str:
    docs = retrieval.get("documents", [[]])[0]
    metas = retrieval.get("metadatas", [[]])[0]
    if not docs:
        return "No relevant document found."

    lines = []
    for idx, (doc, meta) in enumerate(zip(docs, metas), start=1):
        topic = meta.get("target_name", "unknown") if meta else "unknown"
        snippet = doc[:300].replace("\n", " ")
        lines.append(f"{idx}. ({topic}) {snippet}")
    return "\n".join(lines)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)

    if not artifacts_ready():
        build_artifacts(force_rebuild=False)

    embedder = Embedder(settings.embedding_model_name)
    gmm = joblib.load(settings.artifacts_dir / "gmm.joblib")
    vector_store = VectorStore(
        persist_dir=str(settings.cache_dir),
        collection_name=settings.chroma_collection_name,
    )
    cache = SemanticCache(
        similarity_threshold=settings.default_cache_similarity_threshold,
        candidate_clusters=settings.cache_candidate_clusters,
    )

    app.state.embedder = embedder
    app.state.gmm = gmm
    app.state.vector_store = vector_store
    app.state.semantic_cache = cache
    yield


app = FastAPI(title="20 Newsgroups Semantic Cache API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "service": "20 Newsgroups Semantic Cache API"}


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q_vec = app.state.embedder.embed_query(query)
    q_membership = app.state.gmm.predict_proba([q_vec])[0]
    dominant_cluster = int(q_membership.argmax())

    hit = app.state.semantic_cache.lookup(q_vec, q_membership)
    if hit is not None:
        return {
            "query": query,
            **hit,
        }

    retrieval = app.state.vector_store.query(
        query_embedding=q_vec,
        n_results=settings.retrieval_top_k,
        where={"dominant_cluster": dominant_cluster},
    )
    result = _build_result_from_retrieval(retrieval)

    app.state.semantic_cache.add(
        query=query,
        query_embedding=q_vec,
        result=result,
        dominant_cluster=dominant_cluster,
        membership=q_membership,
    )

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": dominant_cluster,
    }


@app.get("/cache/stats")
async def cache_stats():
    return app.state.semantic_cache.stats()


@app.delete("/cache")
async def clear_cache():
    app.state.semantic_cache.clear()
    return {"message": "Cache cleared."}
