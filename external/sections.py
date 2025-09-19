# external/sections.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


@dataclass
class Section:
    """
    A contiguous span of chunks that belong together topically.
    """
    id: int
    chunk_ids: List[int]
    pages: List[int]
    title: str
    preview: str
    topic_words: List[str]


def _first_line(s: str) -> str:
    s = (s or "").strip().splitlines()[0] if s else ""
    return s[:120]


def _make_sections_from_embeddings(
    chunks: List[Dict[str, Any]],
    emb: np.ndarray,
    sim_drop: float = 0.40,
    min_len: int = 3,
    max_len: int = 60,
) -> List[List[int]]:
    """
    Segment the document by breaking at large semantic drops between adjacent chunks.
    Assumes 'emb' rows are L2-normalized.
    """
    if not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[0] != len(chunks):
        raise ValueError("Embeddings must be (n_chunks, d) and match chunks length")

    sims = (emb[:-1] * emb[1:]).sum(axis=1)
    breaks = [0]
    cur_len = 1
    for i, cs in enumerate(sims, start=1):
        force_break = cur_len >= max_len
        large_drop = (cs < sim_drop) and (cur_len >= min_len)
        if force_break or large_drop:
            breaks.append(i)
            cur_len = 1
        else:
            cur_len += 1
    breaks.append(len(chunks))

    groups: List[List[int]] = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        span = list(range(a, b))
        if span:
            groups.append(span)
    return groups


def _lda_topic_words_per_section(
    chunks: List[Dict[str, Any]],
    sections: List[List[int]],
    n_topics: int = 16,
    topic_words_per_section: int = 12,
    max_features: int = 40000,
) -> Tuple[List[List[str]], CountVectorizer, LatentDirichletAllocation]:
    """
    Fit LDA on the whole corpus, then average per-chunk topic distributions within each section
    and convert that mixture into a ranked list of concrete words.
    """
    docs = [c.get("text", "") for c in chunks]
    vect = CountVectorizer(
        lowercase=True,
        token_pattern=r"[A-Za-z0-9\-]{2,}",
        stop_words="english",
        max_features=max_features,
    )
    X = vect.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="batch",
        random_state=42,
        max_iter=20,
        evaluate_every=0,
        n_jobs=-1,
    )
    lda.fit(X)
    theta = lda.transform(X)  # (n_chunks, n_topics)

    beta = lda.components_
    beta = beta / beta.sum(axis=1, keepdims=True)
    vocab = np.array(vect.get_feature_names_out())

    section_words: List[List[str]] = []
    for span in sections:
        mix = theta[span].mean(axis=0)
        word_scores = mix @ beta
        top_idx = np.argsort(-word_scores)[:topic_words_per_section]
        words = [w for w in vocab[top_idx]]
        section_words.append(words)

    return section_words, vect, lda


def build_sections_artifacts(
    chunks: List[Dict[str, Any]],
    chunk_embs: np.ndarray,
    artifacts_dir: Path,
    sim_drop: float = 0.40,
    min_len: int = 3,
    max_len: int = 60,
    n_topics: int = 16,
    topic_words_per_section: int = 12,
) -> List[Section]:
    """
    Create a section index and persist:
      - sections.json      (metadata + topic words)
      - section_embs.npy   (mean embedding per section)
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    groups = _make_sections_from_embeddings(chunks, chunk_embs, sim_drop, min_len, max_len)
    topics_per_section, _, _ = _lda_topic_words_per_section(
        chunks, groups, n_topics=n_topics, topic_words_per_section=topic_words_per_section
    )

    sections: List[Section] = []
    section_embs = []
    for sid, (span, words) in enumerate(zip(groups, topics_per_section)):
        pages = sorted({p for i in span for p in chunks[i].get("pages", [])})
        text0 = chunks[span[0]].get("text", "")
        sec = Section(
            id=sid,
            chunk_ids=span,
            pages=pages,
            title=_first_line(text0),
            preview=text0.strip()[:160],
            topic_words=words,
        )
        sections.append(sec)
        section_embs.append(chunk_embs[span].mean(axis=0))

    (artifacts_dir / "sections.json").write_text(
        json.dumps([asdict(s) for s in sections], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(artifacts_dir / "section_embs.npy", np.vstack(section_embs).astype("float32"))
    return sections
