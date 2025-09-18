"""Topic modeling utilities.

Provides lightweight helpers for running Latent Dirichlet Allocation (LDA) with
scikit-learn both on the entire document and per-section. The design avoids
Prefect decorators to keep functions reusable and independently testable.

Main exposed functions:
    * :func:`fit_lda` – Train an LDA model for given texts (list of documents).
    * :func:`compute_topics_overall_and_sections` – Aggregate topics overall and per-section and persist artifacts.
"""
from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

STOP_WORDS = set([
    'the', 'a', 'an', 'and', 'or', 'in', 'on', 'for', 'to', 'of', 'is', 'are', 'was', 'were',
    'it', 'that', 'this', 'with', 'as', 'by', 'from'
])

 
@dataclass
class LDATopic:
    """Represents a single discovered topic.

    Attributes:
        topic_id: Sequential topic index as produced by LDA.
        terms: Top terms (ordered) for this topic.
        weights: Unnormalized component weights corresponding to ``terms``.
    """
    topic_id: int
    terms: List[str]
    weights: List[float]


def _tokenizer(text: str) -> List[str]:
    """Tokenize text for LDA.

    A very small regex-based tokenizer removing short tokens and stop words.

    Args:
        text: Input string.

    Returns:
        List of normalized (lowercased) tokens.
    """
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    return [t for t in tokens if t not in STOP_WORDS]


def fit_lda(
    texts: List[str],
    n_topics: int = 8,
    max_features: int = 4000,
    random_state: int = 42,
    min_token_count: int = 3,
) -> Dict[str, Any]:
    """Fit an LDA model over the provided documents with safeguards.

    Adds defensive checks so tiny / stopword-only inputs do not raise
    scikit-learn errors. When the vocabulary would be empty, returns an
    empty topic list instead of failing.

    Args:
        texts: List of documents (each a string). Each string is treated as one document.
        n_topics: Number of latent topics.
        max_features: Maximum vocabulary size for the CountVectorizer.
        random_state: Random seed for reproducibility.
        min_token_count: Minimum total token count across all documents required to attempt LDA.

    Returns:
        Dictionary with keys:
            topics: List of serialized LDATopic (may be empty)
            doc_topic_distrib: List of per-document topic distributions (may be empty)
            skipped: Optional bool flag when modeling skipped
            reason: Optional explanation when skipped
    """
    if not texts:
        return {"topics": [], "doc_topic_distrib": [], "skipped": True, "reason": "no texts provided"}

    # Tokenize all texts first to gauge viability
    tokenized_docs = [[t for t in _tokenizer(doc)] for doc in texts]
    total_tokens = sum(len(d) for d in tokenized_docs)
    if total_tokens < min_token_count:
        return {
            "topics": [],
            "doc_topic_distrib": [],
            "skipped": True,
            "reason": f"insufficient tokens ({total_tokens} < {min_token_count})",
            "top_terms": _simple_term_summary(tokenized_docs),
        }

    # Build vectorizer; silence token_pattern warning since we supply tokenizer
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None",
        )
        vectorizer = CountVectorizer(
            tokenizer=lambda x: x,
            preprocessor=None,
            lowercase=False,
            max_features=max_features,
        )
        # vectorizer expects raw documents; we already tokenized, so join with spaces
        raw_docs = [" ".join(toks) for toks in tokenized_docs]
        try:
            dtm = vectorizer.fit_transform(raw_docs)
        except ValueError as exc:
            # Covers empty vocabulary edge
            return {
                "topics": [],
                "doc_topic_distrib": [],
                "skipped": True,
                "reason": f"vectorizer failure: {exc}",
                "top_terms": _simple_term_summary(tokenized_docs),
            }

    if dtm.shape[1] == 0:
        return {
            "topics": [],
            "doc_topic_distrib": [],
            "skipped": True,
            "reason": "empty vocabulary",
            "top_terms": _simple_term_summary(tokenized_docs),
        }

    n_topics_eff = min(n_topics, max(1, dtm.shape[1]))
    lda = LatentDirichletAllocation(n_components=n_topics_eff, random_state=random_state)
    doc_topic = lda.fit_transform(dtm)
    vocab = vectorizer.get_feature_names_out()
    topics: List[LDATopic] = []
    for topic_idx, comp in enumerate(lda.components_):
        top_indices = comp.argsort()[::-1][:15]
        terms = [str(vocab[i]) for i in top_indices]
        weights = [float(comp[i]) for i in top_indices]
        topics.append(LDATopic(topic_id=topic_idx, terms=terms, weights=weights))
    return {
        "topics": [asdict(t) for t in topics],
        "doc_topic_distrib": doc_topic.tolist(),
        "skipped": False,
        "top_terms": _simple_term_summary(tokenized_docs),
    }


class TermCount(TypedDict):
    term: str
    count: int


def _simple_term_summary(tokenized_docs: List[List[str]], top_n: int = 10) -> list[TermCount]:
    """Return a simple term frequency summary as a fallback.

    Args:
        tokenized_docs: List of token lists per document.
        top_n: Number of most common terms to include.

    Returns:
        List of dictionaries with 'term' and 'count'.
    """
    counter: Counter[str] = Counter()
    for toks in tokenized_docs:
        counter.update(toks)
    summary: list[TermCount] = [
        TermCount(term=str(term), count=int(count)) for term, count in counter.most_common(top_n)
    ]
    return summary


def compute_topics_overall_and_sections(
    run_dir: str | Path,
    sections: List[Dict[str, Any]],
    pages: List[Dict[str, Any]],
    n_topics_overall: int = 10,
    n_topics_section: int = 5,
) -> Dict[str, Any]:
    """Compute LDA topics for the whole document and each section, then persist.

    Args:
        run_dir: Target run directory where `topics.json` will be written.
        sections: List of section dicts (must contain 'text' and 'id').
        pages: List of page dicts (used to assemble overall corpus text).
        n_topics_overall: Number of topics for single overall model.
        n_topics_section: Number of topics for each per-section model.

    Returns:
        Dictionary with keys:
            overall: Output of :func:`fit_lda` on the combined document text.
            sections: Mapping section id -> topic model output.
    """
    run_dir = Path(run_dir)
    overall_text = "\n".join(p['text'] for p in pages)
    overall = fit_lda([overall_text], n_topics=n_topics_overall)

    per_section = {}
    for s in sections:
        section_text = s.get('text', '') or ''
        section_topics = fit_lda([section_text], n_topics=n_topics_section)
        per_section[str(s.get('id'))] = section_topics

    data = {"overall": overall, "sections": per_section}
    (run_dir / "topics.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
