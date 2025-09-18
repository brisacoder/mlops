"""Retrieval indexing utilities.

Implements lightweight TF-IDF and BM25 creation without external vector DBs.
Data structures are intentionally simple JSON-serializable forms so that
artifacts can be inspected or reloaded easily for experimentation.

Main concepts:
    * :func:`build_tfidf` produces sparse TF-IDF vectors in a list-of-dicts format.
    * :class:`BM25Index` holds tokenized documents and statistics for scoring.
    * :func:`bm25_score` scores a query against a :class:`BM25Index`.
    * :func:`build_and_persist_indexes` orchestrates building & persisting both indices.
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict

TOKEN_RE = re.compile(r"[A-Za-z]{2,}")

@dataclass
class BM25Index:
    """Container for BM25 corpus statistics.

    Attributes:
        doc_len: List of token counts per document.
        avg_len: Average document length used in BM25 normalization.
        term_doc_freq: Mapping term -> number of documents containing term.
        docs_tokens: Tokenized documents (list of token lists).
    """
    doc_len: List[int]
    avg_len: float
    term_doc_freq: Dict[str, int]
    docs_tokens: List[List[str]]


def tokenize(text: str) -> List[str]:
    """Regex tokenize a string.

    Args:
        text: Raw text.

    Returns:
        List of lowercased alphanumeric-ish tokens (>=2 chars).
    """
    return [t.lower() for t in TOKEN_RE.findall(text)]


def build_tfidf(docs: List[str], max_features: int = 6000) -> Dict[str, Any]:
    """Build a simplified TF-IDF representation.

    A manual implementation to avoid adding another dependency; for large-scale
    production usage consider scikit-learn's `TfidfVectorizer`.

    Args:
        docs: List of document strings.
        max_features: Maximum vocabulary size selected by document frequency.

    Returns:
        Dict with keys:
            vocab: List of chosen vocabulary terms in index order.
            docs: List of sparse vectors (dict index->tfidf weight).
            idf: Mapping term -> IDF value.
    """
    tokenized = [tokenize(d) for d in docs]
    term_freq_global: Counter[str] = Counter()
    for toks in tokenized:
        term_freq_global.update(set(toks))  # doc freq counting
    # Select top features by doc freq
    vocab = [t for t, _ in term_freq_global.most_common(max_features)]
    vocab_index = {t: i for i, t in enumerate(vocab)}

    doc_term_counts = []
    for toks in tokenized:
        c = Counter(toks)
        row = {vocab_index[t]: freq for t, freq in c.items() if t in vocab_index}
        doc_term_counts.append(row)

    n_docs = len(docs)
    idf = {}
    for term in vocab:
        df = term_freq_global[term]
        idf[term] = math.log((n_docs + 1) / (df + 1)) + 1.0

    # Store sparse representation
    tfidf_docs = []
    for row in doc_term_counts:
        doc_vec = {int(idx): (freq * idf[vocab[idx]]) for idx, freq in row.items()}
        tfidf_docs.append(doc_vec)

    return {"vocab": vocab, "docs": tfidf_docs, "idf": idf}


def build_bm25(docs: List[str]) -> BM25Index:
    """Construct a :class:`BM25Index` from raw documents.

    Args:
        docs: List of document strings.

    Returns:
        An instance of :class:`BM25Index` with tokenized content and statistics.
    """
    tokenized = [tokenize(d) for d in docs]
    doc_len = [len(toks) for toks in tokenized]
    avg_len = sum(doc_len) / max(1, len(doc_len))
    term_doc_freq_counter: Counter[str] = Counter()
    for toks in tokenized:
        term_doc_freq_counter.update(set(toks))
    term_doc_freq: Dict[str, int] = dict(term_doc_freq_counter)
    return BM25Index(doc_len=doc_len, avg_len=avg_len, term_doc_freq=term_doc_freq, docs_tokens=tokenized)


def bm25_score(query: str, index: BM25Index, k1: float = 1.5, b: float = 0.75) -> List[Tuple[int, float]]:
    """Score a query against the documents using BM25.

    Args:
        query: Query string.
        index: Pre-built :class:`BM25Index`.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        List of tuples (doc_index, score) sorted by descending score.
    """
    q_tokens = tokenize(query)
    scores = [0.0] * len(index.docs_tokens)
    N = len(index.docs_tokens)
    for q in q_tokens:
        df = index.term_doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        for doc_id, toks in enumerate(index.docs_tokens):
            freq = toks.count(q)
            if freq == 0:
                continue
            denom = freq + k1 * (1 - b + b * (index.doc_len[doc_id] / index.avg_len))
            score = idf * (freq * (k1 + 1)) / denom
            scores[doc_id] += score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked


def build_and_persist_indexes(
    run_dir: str | Path,
    sections: List[Dict[str, Any]],
    pages: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build and persist TF-IDF and BM25 indexes over section texts.

    Args:
        run_dir: Target directory for persistence.
        sections: Section dictionaries each containing 'text'.
        pages: Currently unused (reserved for potential page-level future indexing).

    Returns:
        Dictionary with keys 'tfidf' and 'bm25' (JSON serializable forms).
    """
    run_dir = Path(run_dir)
    section_texts = [s['text'] for s in sections]
    tfidf = build_tfidf(section_texts)
    bm25 = build_bm25(section_texts)
    (run_dir / "tfidf.json").write_text(json.dumps(tfidf, indent=2), encoding="utf-8")
    bm25_serializable = asdict(bm25)
    (run_dir / "bm25.json").write_text(json.dumps(bm25_serializable, indent=2), encoding="utf-8")
    return {"tfidf": tfidf, "bm25": bm25_serializable}
