# external/index_build.py
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sp_normalize


def _tok(s: str) -> List[str]:
    import re
    return re.findall(r"[A-Za-z0-9\-]+", (s or "").lower())


def build_indices(chunks: List[Dict[str, Any]], artifacts_dir: Path) -> None:
    """
    Build and persist TF-IDF (L2-normalized) and BM25 indices.
    Saves: tfidf.pkl, tfidf_X.pkl, bm25.pkl.
    """
    texts = [c.get("text", "") for c in chunks]

    tfidf = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"[A-Za-z0-9\-]{2,}",
        stop_words="english",
        max_features=100_000,
    )
    X = tfidf.fit_transform(texts)
    X = sp_normalize(X, norm="l2", axis=1, copy=False)

    tokens = [_tok(t) for t in texts]
    bm25 = BM25Okapi(tokens)

    (artifacts_dir / "tfidf.pkl").write_bytes(pickle.dumps(tfidf))
    (artifacts_dir / "tfidf_X.pkl").write_bytes(pickle.dumps(X))
    (artifacts_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25))
