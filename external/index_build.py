# external/index_build.py
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sp_normalize


def _tok1(text: str) -> List[str]:
    """Simple alnum tokenizer → unigrams."""
    import re

    return re.findall(r"[A-Za-z0-9\-]+", (text or "").lower())


def _tok12(text: str) -> List[str]:
    """Unigrams + bigrams (space-joined)."""
    unis = _tok1(text)
    bigs = [f"{unis[i]} {unis[i+1]}" for i in range(len(unis) - 1)]
    return unis + bigs


def _build_tfidf(chunks: List[Dict[str, Any]]) -> Tuple[TfidfVectorizer, Any]:
    """
    Train a 1–2 gram TF-IDF on chunk texts, return (vectorizer, X).
    X is L2-normalized row-wise.
    """
    texts = [c.get("text", "") for c in chunks]
    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z0-9\-]+",
        lowercase=True,
        strip_accents="ascii",
        min_df=2,
    )
    X = vect.fit_transform(texts)
    X = sp_normalize(X, norm="l2", axis=1, copy=False)
    return vect, X


def _build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """
    Train a BM25 over 1–2 gram tokens for each chunk.
    """
    docs = [_tok1(c.get("text", "")) for c in chunks]
    return BM25Okapi(docs)


def build_indices(chunks: List[Dict[str, Any]], artifacts_dir: Path) -> None:
    """
    Build and persist sparse indices required by search:
      - TF-IDF (1–2 grams): tfidf.pkl, tfidf_X.pkl
      - BM25 (1–2 grams):   bm25.pkl
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ---- TF-IDF (1–2 grams)
    tfidf, X = _build_tfidf(chunks)
    with (artifacts_dir / "tfidf.pkl").open("wb") as f:
        pickle.dump(tfidf, f)
    with (artifacts_dir / "tfidf_X.pkl").open("wb") as f:
        pickle.dump(X, f)

    # ---- BM25 (1–2 grams)
    bm25 = _build_bm25(chunks)
    with (artifacts_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
