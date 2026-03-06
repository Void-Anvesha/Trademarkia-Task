from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = data_dir / "artifacts"
    cache_dir: Path = data_dir / "cache"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_collection_name: str = "twenty_newsgroups"

    default_cache_similarity_threshold: float = 0.86
    cache_candidate_clusters: int = 3
    retrieval_top_k: int = 5


settings = Settings()
