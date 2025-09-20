# build.py
"""
Build the searchable artifacts for a car manual (or any large PDF).

Pipeline
--------
1) Extract PDF text and images (delegates to external.pdf_extract.extract_pdf).
2) Chunk text into page-aware segments (stabilized for long tokens).
3) Build sparse indices (TF-IDF 1–2 grams + BM25 1–2 grams).
4) Embed:
   - Image captions → caption_embs.npy
   - Chunks → chunk_embs.npy
5) Topic model (LDA) over chunks → dominant topic per chunk.
6) Sectioning: contiguous chunks sharing the dominant topic → sections.json.
7) Section embeddings: mean of member chunk embeddings → section_embs.npy.
8) Persist detailed timing metrics (per-run JSON and daily JSONL).

Outputs
-------
artifacts/
  ├── manifest.json
  ├── chunks.json
  ├── tfidf.pkl
  ├── tfidf_X.pkl
  ├── bm25.pkl
  ├── caption_index.json
  ├── caption_embs.npy
  ├── chunk_embs.npy
  ├── sections.json
  ├── section_embs.npy
  └── metrics/
      ├── runs/<UTC>/timings.json
      └── builds-YYYYMMDD.jsonl

Usage
-----
python build.py --pdf ./data/manual.pdf --artifacts ./artifacts

Environment
-----------
ARTIFACTS_DIR             # optional, default "./artifacts"
MIN_CHARS / MAX_CHARS     # chunk size (defaults 400 / 1200)
CHUNK_OVERLAP             # overlap in tokens (default 80)
LDA_TOPICS                # number of topics for LDA (default 24)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from external.index_build import build_indices
from external.pdf_extract import extract_pdf
from external.runtime import load_models


# --------------------------- Configuration ---------------------------

DEFAULT_MIN_CHARS = int(os.getenv("MIN_CHARS", "400"))
DEFAULT_MAX_CHARS = int(os.getenv("MAX_CHARS", "1200"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
DEFAULT_LDA_TOPICS = int(os.getenv("LDA_TOPICS", "24"))

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


# --------------------------- Data types ---------------------------

@dataclass
class Chunk:
    """A chunk of text with associated page numbers (1-based)."""
    id: str
    pages: List[int]
    text: str


# --------------------------- Helpers ---------------------------

def set_up_logger() -> logging.Logger:
    """Configure a root logger for console output."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logging.getLogger("build")


def ensure_dir(p: Path) -> None:
    """Create a directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def tok1(text: str) -> List[str]:
    """Simple alnum tokenizer (lowercased)."""
    import re
    return re.findall(r"[A-Za-z0-9\-]+", (text or "").lower())


def now_utc_iso() -> str:
    """UTC timestamp in ISO 8601 with tzinfo."""
    return dt.datetime.now(dt.UTC).isoformat()


# --------------------------- Chunking (robust) ---------------------------

def chunk_pages(
    manifest: Dict[str, Any],
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Robust page-aware chunking that never deadlocks.

    Rules
    -----
    - Accumulate tokens until adding the next would exceed `max_chars`.
    - If the *very first* token is longer than `max_chars`, hard-split it.
    - On normal flushes, carry up to `overlap` tail tokens **only if**
      the next token would still fit; otherwise drop the overlap to ensure progress.
    """
    logger = logging.getLogger("build")

    pages = manifest.get("pages", [])
    page_texts: List[Tuple[int, str]] = [(p["page"], p.get("text", "") or "") for p in pages]

    chunks: List[Chunk] = []
    buf: List[str] = []
    start_page: int | None = None
    cur_chars = 0

    def _buf_chars(tokens: List[str]) -> int:
        # approximate char count with single spaces between tokens
        return sum(len(t) for t in tokens) + max(0, len(tokens) - 1)

    def emit(end_page: int, carry_next: Optional[str] = None) -> None:
        """
        Flush current buffer as a chunk and seed the next buffer with overlap,
        but only if the overlap still leaves room for `carry_next`.
        """
        nonlocal buf, start_page, cur_chars, chunks

        if not buf:
            start_page = None
            cur_chars = 0
            return

        text = " ".join(buf).strip()
        if not text:
            buf = []
            start_page = None
            cur_chars = 0
            return

        cid = f"c{len(chunks):04d}"
        pages_span = list(range(start_page or end_page, end_page + 1))
        chunks.append(Chunk(id=cid, pages=pages_span, text=text))

        # Prepare next buffer with overlap, but only if it leaves room
        if overlap > 0:
            tail = buf[-overlap:]
        else:
            tail = []

        if carry_next is not None and tail:
            # If keeping the tail would block adding the very next token, drop it.
            if _buf_chars(tail) + (1 if tail else 0) + len(carry_next) > max_chars:
                logger.debug("Dropping overlap to ensure progress at page %s", end_page)
                tail = []

        buf = tail
        cur_chars = _buf_chars(buf)
        start_page = None

    def add_token(tok: str) -> None:
        nonlocal buf, cur_chars
        if buf:
            cur_chars += 1  # space
        buf.append(tok)
        cur_chars += len(tok)

    def hard_split(tok: str, page_num: int) -> None:
        """Split a single very-long token so the pointer advances."""
        nonlocal start_page, buf, cur_chars
        i = 0
        while i < len(tok):
            seg = tok[i : i + max_chars]
            if start_page is None:
                start_page = page_num
            buf = []
            cur_chars = 0
            add_token(seg)
            emit(end_page=page_num)  # force out immediately
            i += max_chars

    logger.info("2) Chunking…")
    for page_num, text in page_texts:
        text = text.strip()
        if not text:
            continue

        tokens = text.split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]

            # Start page marker
            if start_page is None:
                start_page = page_num

            # Case A: pathological long token
            if len(tok) > max_chars:
                hard_split(tok, page_num)
                i += 1
                continue

            # Case B: next token wouldn't fit
            if cur_chars > 0 and cur_chars + 1 + len(tok) > max_chars:
                emit(end_page=page_num, carry_next=tok)
                # After emit we re-check the *same* tok against (possibly empty) buffer
                # but we will definitely be able to add it now (or hard-split).
                if start_page is None:
                    start_page = page_num
                # If buffer still too full (due to overlap) we will immediately emit again,
                # but emit() above already dropped overlap that blocked this tok.

            # Add token (fits, or buffer is empty)
            if cur_chars == 0 or cur_chars + 1 + len(tok) <= max_chars:
                add_token(tok)
                i += 1
                # Opportunistic flush near boundary: if next token would push over
                if cur_chars >= min_chars and i < len(tokens):
                    nxt = tokens[i]
                    if len(nxt) > max_chars or cur_chars + 1 + len(nxt) > max_chars:
                        emit(end_page=page_num, carry_next=nxt)

        # Heartbeat every 25 pages
        if page_num % 25 == 0:
            logger.info("… chunking progress: processed %d pages", page_num)

    # Final emit if meaningful content remains
    if buf and cur_chars >= min_chars and page_texts:
        emit(end_page=page_texts[-1][0])

    return chunks


# --------------------------- LDA + Sections ---------------------------

def lda_topics_over_chunks(
    chunks: Sequence[Chunk],
    n_topics: int = DEFAULT_LDA_TOPICS,
    max_features: int = 25000,
) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray, List[List[str]]]:
    """
    Train LDA on chunk texts (bag-of-words with 1–2 grams) and return:
    - fitted LDA model
    - fitted CountVectorizer
    - doc_topic (n_chunks x n_topics) array
    - topic_words: list of top words per topic
    """
    texts = [c.text for c in chunks]
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        token_pattern=r"[A-Za-z0-9\-]+",
        max_features=max_features,
        lowercase=True,
    )
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method="online",
        random_state=0,
        n_jobs=-1,
        evaluate_every=-1,
    )
    doc_topic = lda.fit_transform(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    topic_words: List[List[str]] = []
    weights = lda.components_  # [n_topics, vocab]
    for k in range(n_topics):
        top_idx = np.argsort(-weights[k])[:12]
        topic_words.append([str(feature_names[i]) for i in top_idx])

    return lda, vectorizer, doc_topic, topic_words


def build_sections_from_topics(
    chunks: Sequence[Chunk],
    doc_topic: np.ndarray,
    topic_words: List[List[str]],
) -> List[Dict[str, Any]]:
    """
    Create sections by merging contiguous chunks that share the same
    dominant LDA topic. Section title is derived from the topic's top words.
    """
    if not chunks:
        return []

    z = np.argmax(doc_topic, axis=1)  # dominant topic id per chunk

    sections: List[Dict[str, Any]] = []
    cur_topic = int(z[0])
    cur_ids: List[str] = [chunks[0].id]

    for i in range(1, len(chunks)):
        t = int(z[i])
        if t == cur_topic:
            cur_ids.append(chunks[i].id)
        else:
            title = " ".join(topic_words[cur_topic][:5])
            sections.append(
                {
                    "id": f"s{len(sections):04d}",
                    "title": title,
                    "topic_words": topic_words[cur_topic],
                    "chunk_ids": cur_ids.copy(),
                }
            )
            cur_topic = t
            cur_ids = [chunks[i].id]

    title = " ".join(topic_words[cur_topic][:5])
    sections.append(
        {
            "id": f"s{len(sections):04d}",
            "title": title,
            "topic_words": topic_words[cur_topic],
            "chunk_ids": cur_ids.copy(),
        }
    )
    return sections


def embed_sections_mean(
    sections: Sequence[Dict[str, Any]],
    chunks: Sequence[Chunk],
    chunk_embs: np.ndarray,
) -> np.ndarray:
    """Compute section embeddings as the mean of member chunk embeddings."""
    idx_map = {c.id: i for i, c in enumerate(chunks)}
    d = chunk_embs.shape[1] if chunk_embs.size else 384
    out = np.zeros((len(sections), d), dtype="float32")
    for i, sec in enumerate(sections):
        ids = [idx_map[cid] for cid in sec.get("chunk_ids", []) if cid in idx_map]
        if ids:
            out[i] = np.mean(chunk_embs[ids], axis=0)
        else:
            out[i] = 0.0
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return (out / norms).astype("float32")


# --------------------------- Captions ---------------------------

def collect_captions(manifest: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return (files, texts) for images that have non-empty captions."""
    files: List[str] = []
    texts: List[str] = []
    for pg in manifest.get("pages", []):
        for im in pg.get("images", []):
            cap = (im.get("caption") or "").strip()
            if not cap:
                continue
            files.append(str(im.get("file")))
            texts.append(cap)
    return files, texts


# --------------------------- Embedding (SentenceTransformer) ---------------------------

def batch_encode(
    st_model,
    texts: Sequence[str],
    batch_size: int = 128,
    desc: str = "",
) -> np.ndarray:
    """
    Encode texts with SentenceTransformer, normalized embeddings, with visible tqdm.
    Adjust batch_size if you OOM (64) or have more VRAM (256).
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, 384), dtype="float32")

    out: List[np.ndarray] = []
    steps = (n + batch_size - 1) // batch_size
    for i in tqdm(range(0, n, batch_size), total=steps, desc=desc):
        seg = texts[i : i + batch_size]
        embs = st_model.encode(
            seg,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        out.append(embs.astype("float32"))
    return np.vstack(out)


# --------------------------- Metrics ---------------------------

def ms_since(t0: float) -> float:
    """Return milliseconds since t0."""
    return round((time.perf_counter() - t0) * 1000.0, 3)


def write_run_metrics(art_dir: Path, run_dir: Path, metrics: Dict[str, Any]) -> None:
    """Persist per-run detailed metrics and a daily JSONL roll-up."""
    ensure_dir(run_dir)
    (run_dir / "timings.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    daily = art_dir / "metrics" / f"builds-{dt.datetime.now(dt.UTC).strftime('%Y%m%d')}.jsonl"
    ensure_dir(daily.parent)
    with daily.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


# --------------------------- Main flow ---------------------------

def main() -> None:
    """Entry point to build all artifacts for a given PDF."""
    parser = argparse.ArgumentParser(description="Build artifacts for manual search.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to input PDF.")
    parser.add_argument(
        "--artifacts",
        type=str,
        default=os.getenv("ARTIFACTS_DIR", "./artifacts"),
        help="Artifacts directory (default: ./artifacts or $ARTIFACTS_DIR).",
    )
    parser.add_argument(
        "--topics", type=int, default=DEFAULT_LDA_TOPICS, help="Number of LDA topics."
    )
    parser.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Minimum chars per chunk."
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Maximum chars per chunk."
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Token overlap between chunks."
    )
    args = parser.parse_args()

    log = set_up_logger()

    pdf_path = Path(args.pdf).resolve()
    art_dir = Path(args.artifacts).resolve()
    ensure_dir(art_dir)
    ensure_dir(art_dir / "metrics")

    run_dir = art_dir / "metrics" / "runs" / dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    ensure_dir(run_dir)

    log.info("Artifacts dir: %s", art_dir)
    log.info("PDF: %s", pdf_path)

    # Models
    models_t0 = time.perf_counter()
    loaded = load_models(log)
    st_model = loaded.st_model
    t_models_ms = ms_since(models_t0)
    log.info(
        "Models loaded | spaCy=%.3fs | ST=%.3fs | device=%s | model=%s",
        loaded.t_spacy_load_s,
        loaded.t_st_load_s,
        "CUDA" if loaded.using_gpu else "CPU",
        getattr(st_model, "model_card_data", None) or getattr(st_model, "__class__", type("X", (), {})).__name__,
    )

    # 1) Extract
    t0 = time.perf_counter()
    manifest = extract_pdf(pdf_path, art_dir)
    t_extract_ms = ms_since(t0)
    n_pages = len(manifest.get("pages", []))
    n_images = sum(len(p.get("images", [])) for p in manifest.get("pages", []))
    (art_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    log.info("1) Extracted PDF | pages=%d images=%d | %.3fs", n_pages, n_images, t_extract_ms / 1000.0)

    # 2) Chunk
    log.info("2) Chunking…")
    t0 = time.perf_counter()
    chunks = chunk_pages(
        manifest,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    chunks_payload = [{"id": c.id, "pages": c.pages, "text": c.text} for c in chunks]
    (art_dir / "chunks.json").write_text(json.dumps(chunks_payload), encoding="utf-8")
    t_chunk_ms = ms_since(t0)
    log.info(
        "2) Chunked | chunks=%d (min=%d, max=%d, ovl=%d) | %.3fs",
        len(chunks),
        args.min_chars,
        args.max_chars,
        args.overlap,
        t_chunk_ms / 1000.0,
    )

    # 3) Indices
    log.info("3) Building sparse indices (TF-IDF + BM25)…")
    t0 = time.perf_counter()
    build_indices(chunks_payload, art_dir)
    t_sparse_ms = ms_since(t0)
    log.info("3) Text indices built (TF-IDF, BM25) | %.3fs", t_sparse_ms / 1000.0)

    # 4) Embeddings
    # 4a) Captions
    cap_files, cap_texts = collect_captions(manifest)
    log.info("4a) Embedding captions… n=%d", len(cap_texts))
    t0 = time.perf_counter()
    if cap_texts:
        caption_embs = batch_encode(st_model, cap_texts, desc="Embedding captions", batch_size=128)
    else:
        caption_embs = np.zeros((0, 384), dtype="float32")
    np.save(art_dir / "caption_embs.npy", caption_embs)
    (art_dir / "caption_index.json").write_text(
        json.dumps({"files": cap_files, "texts": cap_texts}), encoding="utf-8"
    )
    t_caps_ms = ms_since(t0)
    log.info("4a) Captions embedded | n=%d | %.3fs", len(cap_texts), t_caps_ms / 1000.0)

    # 4b) Chunks
    chunk_texts = [c.text for c in chunks]
    log.info("4b) Embedding chunks… n=%d", len(chunk_texts))
    t0 = time.perf_counter()
    chunk_embs = batch_encode(st_model, chunk_texts, desc="Embedding chunks", batch_size=128)
    np.save(art_dir / "chunk_embs.npy", chunk_embs)
    t_chunk_emb_ms = ms_since(t0)
    log.info("4b) Chunk embeddings | n=%d | %.3fs", len(chunk_texts), t_chunk_emb_ms / 1000.0)

    # 5) LDA & sections
    log.info("5) LDA / Sectioning… topics=%d", args.topics)
    t0 = time.perf_counter()
    _, _, doc_topic, topic_words = lda_topics_over_chunks(chunks, n_topics=args.topics)
    sections = build_sections_from_topics(chunks, doc_topic, topic_words)
    (art_dir / "sections.json").write_text(json.dumps(sections), encoding="utf-8")
    t_lda_ms = ms_since(t0)
    log.info("5) Sections from LDA | sections=%d | %.3fs", len(sections), t_lda_ms / 1000.0)

    # 6) Section embeddings
    log.info("6) Section embeddings…")
    t0 = time.perf_counter()
    section_embs = embed_sections_mean(sections, chunks, chunk_embs)
    np.save(art_dir / "section_embs.npy", section_embs)
    t_secemb_ms = ms_since(t0)
    log.info("6) Section embeddings | %.3fs", t_secemb_ms / 1000.0)

    # 7) Metrics
    total_ms = (
        (t_models_ms + t_extract_ms + t_chunk_ms + t_sparse_ms + t_caps_ms + t_chunk_emb_ms + t_lda_ms + t_secemb_ms)
    )
    metrics = {
        "type": "build",
        "ts": now_utc_iso(),
        "pdf": str(pdf_path),
        "artifacts_dir": str(art_dir),
        "pages": n_pages,
        "images": n_images,
        "chunks": len(chunks),
        "sections": len(sections),
        "settings": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "overlap": args.overlap,
            "lda_topics": args.topics,
        },
        "timings_ms": {
            "models_load": t_models_ms,
            "extract": t_extract_ms,
            "chunk": t_chunk_ms,
            "sparse_indices": t_sparse_ms,
            "embed_captions": t_caps_ms,
            "embed_chunks": t_chunk_emb_ms,
            "lda_sections": t_lda_ms,
            "section_embs": t_secemb_ms,
            "total": total_ms,
        },
    }
    write_run_metrics(art_dir, run_dir, metrics)
    log.info(
        "DONE | total %.3fs (models %.3fs + extract %.3fs + chunk %.3fs + sparse %.3fs + "
        "cap_emb %.3fs + chunk_emb %.3fs + lda %.3fs + sec_emb %.3fs)",
        total_ms / 1000.0,
        ms_since(models_t0) / 1000.0,  # includes model load only once
        t_extract_ms / 1000.0,
        t_chunk_ms / 1000.0,
        t_sparse_ms / 1000.0,
        t_caps_ms / 1000.0,
        t_chunk_emb_ms / 1000.0,
        t_lda_ms / 1000.0,
        t_secemb_ms / 1000.0,
    )


if __name__ == "__main__":
    main()
