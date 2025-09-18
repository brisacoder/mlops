from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


def build_indices(chunks: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = [c["text"] for c in chunks]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_df=0.9, min_df=2, strip_accents="unicode"
    )
    X = tfidf.fit_transform(corpus)

    def tok(s: str):
        return [t for t in s.lower().split() if t.isalpha() or t.isalnum()]

    bm25 = BM25Okapi([tok(t) for t in corpus])

    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pickle.dump(tfidf, (out_dir / "tfidf.pkl").open("wb"))
    pickle.dump(X, (out_dir / "tfidf_X.pkl").open("wb"))
    pickle.dump(bm25, (out_dir / "bm25.pkl").open("wb"))
    return {
        "tfidf": str(out_dir / "tfidf.pkl"),
        "bm25": str(out_dir / "bm25.pkl"),
        "chunks": str(out_dir / "chunks.json"),
    }
