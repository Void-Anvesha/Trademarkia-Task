import re
from dataclasses import dataclass
from typing import List

from sklearn.datasets import fetch_20newsgroups


_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")


@dataclass
class Document:
    doc_id: str
    text: str
    target: int
    target_name: str


def clean_text(raw_text: str) -> str:
    text = raw_text.lower()
    # URLs and email strings are high-frequency boilerplate in this dataset and
    # can dominate embedding space without adding semantic topic signal.
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Keep punctuation that might carry discourse style; do not aggressively
    # strip symbols so technical posts still retain domain texture.
    text = _WS_RE.sub(" ", text).strip()
    return text


def load_dataset(min_chars: int = 80) -> List[Document]:
    # remove=(headers, footers, quotes) drops obvious metadata and quoted reply
    # chains that otherwise leak thread structure into embeddings.
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
    )

    documents: List[Document] = []
    for idx, (text, target) in enumerate(zip(dataset.data, dataset.target)):
        cleaned = clean_text(text)
        # Very short fragments are often signatures/noise and degrade cluster
        # separation; keep a conservative threshold.
        if len(cleaned) < min_chars:
            continue
        documents.append(
            Document(
                doc_id=str(idx),
                text=cleaned,
                target=int(target),
                target_name=dataset.target_names[int(target)],
            )
        )

    return documents
