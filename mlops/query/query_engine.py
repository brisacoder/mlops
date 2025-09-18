"""Query engine over processed document artifacts.

Provides:
    * :class:`RunArtifacts` loader for a run directory
    * :func:`bm25_rank` scoring utility
    * :func:`answer_query` high-level query interface returning top sections
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple
import math
import re

TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


def tokenize(text: str) -> List[str]:
    """Tokenize a string for retrieval operations.

    Args:
        text: Input text.

    Returns:
        List of lowercased tokens (>=2 letters).
    """
    return [t.lower() for t in TOKEN_RE.findall(text)]


class RunArtifacts:
    """Loader for a single run directory's artifacts.

    Attributes:
        run_dir: Path object of run directory.
        pages: List of page dicts.
        sections: List of section dicts.
        tfidf: TF-IDF JSON structure if present.
        bm25: BM25 JSON structure if present.
        topics: Topics JSON structure if present.
        entities: Entities + relations JSON structure if present.
        images: List of image metadata dicts.
    """
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.pages = self._load_json("pages.json")
        self.sections = self._load_json("sections.json")
        self.tfidf = self._load_json("tfidf.json")
        self.bm25 = self._load_json("bm25.json")
        self.topics = self._load_json("topics.json")
        self.entities = self._load_json("entities.json")
        self.images = self._load_json("images.json") if (self.run_dir/"images.json").exists() else []

    def _load_json(self, name: str):
        path = self.run_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def section_texts(self) -> List[str]:
        """Convenience accessor for section texts."""
        return [s['text'] for s in self.sections]


def bm25_rank(
    query: str,
    bm25_data: Dict[str, Any],
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Tuple[int, float]]:
    """Rank documents using BM25 given serialized index data.

    Args:
        query: Query string.
        bm25_data: JSON-style BM25 structure with tokens & stats.
        k1: BM25 saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        List of (doc_index, score) sorted by descending score.
    """
    if not bm25_data:
        return []
    docs_tokens = bm25_data['docs_tokens']
    doc_len = bm25_data['doc_len']
    avg_len = bm25_data['avg_len']
    term_doc_freq = bm25_data['term_doc_freq']
    N = len(docs_tokens)
    q_tokens = tokenize(query)
    scores = [0.0] * N
    for q in q_tokens:
        df = term_doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for i, toks in enumerate(docs_tokens):
            freq = toks.count(q)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (doc_len[i] / avg_len))
            scores[i] += idf * (freq * (k1 + 1)) / denom
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked


def answer_query(artifacts: RunArtifacts, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Answer a query using BM25 over section texts.

    Args:
        artifacts: Loaded :class:`RunArtifacts` instance.
        query: Natural language question / keywords.
        top_k: Number of top sections to return.

    Returns:
        Dict with keys: query, answers (list), entities (optional top entity counts).
    """
    ranked = bm25_rank(query, artifacts.bm25)
    top = ranked[:top_k]
    answers = []
    for idx, score in top:
        sec = artifacts.sections[idx]
        snippet = sec['text'][:500]
        answers.append({
            "section_id": sec['id'],
            "title": sec['title'],
            "score": score,
            "snippet": snippet,
        })
    related_entities = []
    if artifacts.entities and 'entities' in artifacts.entities:
        ent_counts = artifacts.entities['entities']['counts']
        related_entities = sorted(ent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"query": query, "answers": answers, "entities": related_entities}
