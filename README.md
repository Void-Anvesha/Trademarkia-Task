<<<<<<< HEAD
# 20 Newsgroups Semantic Search + Cluster-Aware Semantic Cache

This project implements an end-to-end lightweight semantic retrieval system for the 20 Newsgroups corpus with:

1. Embedding + vector DB indexing
2. Fuzzy clustering (distributional membership, not hard labels)
3. A first-principles semantic cache (no Redis/Memcached/cache libraries)
4. A FastAPI service exposing live cache behavior

## Tech Choices

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
  - Good quality/speed tradeoff for ~20K short-medium documents.
- **Vector DB**: `ChromaDB` (persistent local storage)
  - Lightweight, easy filtered retrieval by metadata (dominant cluster).
- **Fuzzy Clustering**: `GaussianMixture` with probability outputs
  - Each document gets a membership distribution across clusters.

## Project Structure

```text
app/
  main.py
  config.py
  core/
    dataset.py
    embedder.py
    clustering.py
    vector_store.py
    semantic_cache.py
    pipeline.py
scripts/
  build_index.py
requirements.txt
Dockerfile
docker-compose.yml
```

## Setup (venv)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Build Index + Clusters

```bash
python -m scripts.build_index
```

This will:
- Download/load the 20 Newsgroups data
- Clean and filter text noise
- Compute embeddings
- Select cluster count via BIC across candidate values
- Fit fuzzy clustering model and store membership distributions
- Store vectors in ChromaDB
- Generate `data/artifacts/cluster_report.json`

## Run API (single uvicorn command)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If artifacts are missing, startup automatically builds them once.

## Endpoints

### `POST /query`

Request:
```json
{ "query": "How does gun policy relate to politics?" }
```

Response (shape):
```json
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
```

### `GET /cache/stats`

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### `DELETE /cache`

Flushes cache and resets stats.

## Tunable Semantic Cache Decision

Primary tunable: **similarity threshold** in `app/config.py` (`default_cache_similarity_threshold`).

- Lower threshold => more hits, greater risk of semantic drift (false hits)
- Higher threshold => fewer hits, better precision, lower cache utility

Suggested mini-study:
1. Run a fixed set of paraphrased queries.
2. Test thresholds: `0.78, 0.82, 0.86, 0.90, 0.93`.
3. Compare hit rate vs qualitative relevance.
4. Explain the behavior change, not just “best” value.

## Cluster Interpretation Artifacts

`data/artifacts/cluster_report.json` includes:
- Selected number of clusters (BIC-based)
- Cluster-level top TF-IDF terms
- Boundary examples (high uncertainty docs)

This supports explaining semantic meaning, overlap, and uncertainty regions.

## Docker (Bonus)

Build and run:

```bash
docker build -t tm-semantic-cache .
docker run -p 8000:8000 tm-semantic-cache
```

or

```bash
docker-compose up --build
```

## Submission Checklist

- Push this project to your GitHub repository
- Deploy API (e.g., Render/Railway/Fly/Azure) and collect project URL
- Grant repo access to `recruitments@trademarkia.com`
- Submit both links in:
  - https://forms.gle/4RpHZpAi8rbG9QCE8
=======
# Trademarkia-Task
“This project is a semantic search API built on the 20 Newsgroups dataset.” “It combines embeddings, fuzzy clustering, vector retrieval, and a semantic cache.” “The main goal is relevant retrieval + faster repeated responses through caching.”
>>>>>>> 70bc7117eeffa4759172b8925020444c95da620b
