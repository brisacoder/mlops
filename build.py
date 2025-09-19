#!/usr/bin/env python3
# build.py
from __future__ import annotations

import json
import os
import platform
import socket
import time
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from dotenv import load_dotenv

from external.pdf_extract import extract_pdf
from external.chunking import split_into_chunks
from external.index_build import build_indices
from external.entity_index import build_entity_index
from external.sections import build_sections_artifacts
from external.runtime import load_models

import hnswlib  # required (no guards)

# ---------- env ----------
load_dotenv()
ART_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()
PDF_PATH = Path(os.getenv("PDF_PATH", "./data/manual.pdf")).resolve()

CHUNK_MIN = int(os.getenv("CHUNK_MIN", "400"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "1200"))

HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "200"))
HNSW_EF_QUERY = int(os.getenv("HNSW_EF_QUERY", "50"))

EMB_BATCH = int(os.getenv("EMB_BATCH", "256"))
EMB_MIN_BATCH = int(os.getenv("EMB_MIN_BATCH", "8"))

SECT_SIM_DROP = float(os.getenv("SECT_SIM_DROP", "0.40"))
SECT_MIN_LEN = int(os.getenv("SECT_MIN_LEN", "3"))
SECT_MAX_LEN = int(os.getenv("SECT_MAX_LEN", "60"))
LDA_TOPICS = int(os.getenv("LDA_TOPICS", "16"))
LDA_WORDS_PER_SECTION = int(os.getenv("LDA_WORDS_PER_SECTION", "12"))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build")


@dataclass
class HostInfo:
    """Static run metadata about host and software versions."""
    hostname: str
    os: str
    python: str
    torch: str
    cuda_available: bool
    cuda_device: str
    cuda_capability: str


def gather_host_info() -> HostInfo:
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cap = ""
    if torch.cuda.is_available():
        maj, minr = torch.cuda.get_device_capability(0)
        cap = f"{maj}.{minr}"
    return HostInfo(
        hostname=socket.gethostname(),
        os=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_device=dev,
        cuda_capability=cap,
    )


class EventsLogger:
    """
    Writes two files:
      - run-<ts>.json  (final summary)
      - events-<ts>.jsonl (each step with start/end/duration and details)
    """
    def __init__(self, artifacts_dir: Path) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        met = artifacts_dir / "metrics"
        met.mkdir(parents=True, exist_ok=True)
        self.summary_path = met / f"run-{ts}.json"
        self.events_path = met / f"events-{ts}.jsonl"
        self._summary: Dict[str, Any] = {}
        self._start_abs = time.perf_counter()

    def set_summary(self, data: Dict[str, Any]) -> None:
        self._summary.update(data)

    def record_event(self, name: str, start_s: float, end_s: float, **fields: Any) -> None:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "start_s": round(start_s, 6),
            "end_s": round(end_s, 6),
            "dur_s": round(end_s - start_s, 6),
        }
        event.update(fields)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def finalize(self) -> None:
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(self._summary, f, ensure_ascii=False, indent=2)


def embed_with_adaptive_bs(st_model, texts: List[str], start_bs: int, min_bs: int, logger: logging.Logger) -> np.ndarray:
    """
    Encode texts with SentenceTransformer using adaptive batch size to avoid CUDA OOM.
    Returns float32, L2-normalized embeddings.
    """
    bs = max(min(start_bs, len(texts) or 1), min_bs)
    while bs >= min_bs:
        try:
            t0 = time.perf_counter()
            vecs = st_model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            logger.info("Embeddings ok | batch=%s | n=%s | %.2fs", bs, len(texts), time.perf_counter() - t0)
            return vecs
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                logger.warning("GPU OOM at batch=%s; retry with %s", bs, bs // 2)
                torch.cuda.empty_cache()
                bs //= 2
                continue
            raise
    # Fallback CPU
    logger.warning("Falling back to CPU embeddings (slow)")
    return st_model.encode(
        texts,
        device="cpu",
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def build_hnsw(vecs: np.ndarray, save_path: Path, m: int, ef_c: int, ef_q: int) -> None:
    """Build and persist HNSW (cosine) index for given vectors."""
    dim = vecs.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=vecs.shape[0], ef_construction=ef_c, M=m)
    idx.add_items(vecs, np.arange(vecs.shape[0]))
    idx.set_ef(ef_q)
    idx.save_index(str(save_path))


def main() -> None:
    # -------- metadata + files --------
    ART_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Artifacts: %s", ART_DIR)
    log.info("PDF: %s", PDF_PATH)

    host = gather_host_info()
    events = EventsLogger(ART_DIR)
    t_run0 = time.perf_counter()

    # -------- models --------
    t0 = time.perf_counter()
    loaded = load_models(log)  # spaCy + SentenceTransformer
    t1 = time.perf_counter()
    events.record_event(
        "models.load",
        t0, t1,
        spacy_load_s=round(loaded.t_spacy_load_s, 6),
        st_load_s=round(loaded.t_st_load_s, 6),
        device="cuda" if loaded.using_gpu else "cpu",
    )

    # -------- extract --------
    t0 = time.perf_counter()
    manifest = extract_pdf(PDF_PATH, ART_DIR)
    t1 = time.perf_counter()
    pages = len(manifest["pages"])
    images_total = sum(len(p["images"]) for p in manifest["pages"])
    events.record_event(
        "extract.pdf",
        t0, t1,
        pages=pages,
        images=images_total,
    )
    log.info("1) Extracted | pages=%s images=%s | %.3fs", pages, images_total, t1 - t0)

    # -------- chunking --------
    t0 = time.perf_counter()
    chunks = split_into_chunks(manifest["pages"], min_chars=CHUNK_MIN, max_chars=CHUNK_MAX)
    t1 = time.perf_counter()
    (ART_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (ART_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    events.record_event(
        "chunking.split",
        t0, t1,
        chunks=len(chunks),
        min_chars=CHUNK_MIN,
        max_chars=CHUNK_MAX,
    )
    log.info("2) Chunked | chunks=%s (min=%s, max=%s) | %.3fs", len(chunks), CHUNK_MIN, CHUNK_MAX, t1 - t0)

    # -------- text indices --------
    t0 = time.perf_counter()
    build_indices(chunks, ART_DIR)
    t1 = time.perf_counter()
    events.record_event("index.tfidf_bm25.build", t0, t1)
    log.info("3) Indices (TF-IDF, BM25) | %.3fs", t1 - t0)

    # -------- entities --------
    t0 = time.perf_counter()
    t_entities = build_entity_index(chunks, ART_DIR, batch_size=32, nlp=loaded.nlp)
    t1 = time.perf_counter()
    events.record_event(
        "entities.ner",
        t0, t1,
        spaCy_batch=32,
        elapsed_reported_s=round(t_entities, 6),
    )
    log.info("4) Entity index | %.3fs", t_entities)

    # -------- captions (embed + hnsw) --------
    cap_files, cap_texts = [], []
    for pg in manifest["pages"]:
        for im in pg["images"]:
            cap_files.append(im["file"])
            cap_texts.append((im.get("caption") or "").strip())
    (ART_DIR / "caption_index.json").write_text(
        json.dumps({"files": cap_files, "texts": cap_texts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if cap_texts:
        t0 = time.perf_counter()
        cap_embs = embed_with_adaptive_bs(loaded.st_model, cap_texts, EMB_BATCH, EMB_MIN_BATCH, log)
        np.save(ART_DIR / "caption_embs.npy", cap_embs)
        t1 = time.perf_counter()
        events.record_event(
            "embed.captions",
            t0, t1,
            n=len(cap_texts),
            batch_start=EMB_BATCH,
            batch_min=EMB_MIN_BATCH,
            dim=int(cap_embs.shape[1]),
        )
        log.info("5a) Captions embedded | %.3fs", t1 - t0)

        t0 = time.perf_counter()
        build_hnsw(cap_embs, ART_DIR / "hnsw_captions.bin", HNSW_M, HNSW_EF_CONSTRUCT, HNSW_EF_QUERY)
        t1 = time.perf_counter()
        events.record_event(
            "ann.hnsw.captions",
            t0, t1,
            M=HNSW_M,
            ef_construct=HNSW_EF_CONSTRUCT,
            ef_query=HNSW_EF_QUERY,
            n=len(cap_texts),
        )
        log.info("5b) HNSW (captions) | %.3fs", t1 - t0)
    else:
        log.info("5) No captions; skipping embeddings and HNSW.")

    # -------- chunks (embed + hnsw) --------
    chunk_texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    ch_embs = embed_with_adaptive_bs(loaded.st_model, chunk_texts, EMB_BATCH, EMB_MIN_BATCH, log)
    np.save(ART_DIR / "chunk_embs.npy", ch_embs)
    t1 = time.perf_counter()
    events.record_event(
        "embed.chunks",
        t0, t1,
        n=len(chunk_texts),
        batch_start=EMB_BATCH,
        batch_min=EMB_MIN_BATCH,
        dim=int(ch_embs.shape[1]),
    )
    log.info("6a) Chunks embedded | %.3fs", t1 - t0)

    t0 = time.perf_counter()
    build_hnsw(ch_embs, ART_DIR / "hnsw_chunks.bin", HNSW_M, HNSW_EF_CONSTRUCT, HNSW_EF_QUERY)
    t1 = time.perf_counter()
    events.record_event(
        "ann.hnsw.chunks",
        t0, t1,
        M=HNSW_M,
        ef_construct=HNSW_EF_CONSTRUCT,
        ef_query=HNSW_EF_QUERY,
        n=len(chunk_texts),
    )
    log.info("6b) HNSW (chunks) | %.3fs", t1 - t0)

    # -------- sections + LDA --------
    t0 = time.perf_counter()
    build_sections_artifacts(
        chunks=chunks,
        chunk_embs=ch_embs,
        artifacts_dir=ART_DIR,
        sim_drop=SECT_SIM_DROP,
        min_len=SECT_MIN_LEN,
        max_len=SECT_MAX_LEN,
        n_topics=LDA_TOPICS,
        topic_words_per_section=LDA_WORDS_PER_SECTION,
    )
    t1 = time.perf_counter()
    # Read back to report counts
    sections = json.loads((ART_DIR / "sections.json").read_text(encoding="utf-8"))
    events.record_event(
        "sections.lda",
        t0, t1,
        sections=len(sections),
        sim_drop=SECT_SIM_DROP,
        min_len=SECT_MIN_LEN,
        max_len=SECT_MAX_LEN,
        lda_topics=LDA_TOPICS,
        lda_words=LDA_WORDS_PER_SECTION,
    )
    log.info("6c) Sections + LDA | sections=%s | %.3fs", len(sections), t1 - t0)

    # -------- summary --------
    t_run1 = time.perf_counter()
    total_with_models = t_run1 - t_run0
    # Recompute total without models by subtracting the first event (models.load)
    # We recorded start/end for each event; for clarity compute by summing all durations except models.
    # But simpler: capture models duration directly from load_models timings:
    models_total = float(loaded.t_spacy_load_s + loaded.t_st_load_s)

    events.set_summary(
        {
            "run_started_utc": datetime.now(timezone.utc).isoformat(),
            "pdf_path": str(PDF_PATH),
            "artifacts_dir": str(ART_DIR),
            "host": asdict(host),
            "models": {
                "device": "cuda" if loaded.using_gpu else "cpu",
                "spacy_trf_s": round(loaded.t_spacy_load_s, 3),
                "sentence_transformers_s": round(loaded.t_st_load_s, 3),
            },
            "params": {
                "chunk_min": CHUNK_MIN,
                "chunk_max": CHUNK_MAX,
                "embed_batch_start": EMB_BATCH,
                "embed_batch_min": EMB_MIN_BATCH,
                "hnsw": {"M": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCT, "ef_query": HNSW_EF_QUERY},
                "sections": {
                    "sim_drop": SECT_SIM_DROP,
                    "min_len": SECT_MIN_LEN,
                    "max_len": SECT_MAX_LEN,
                    "lda_topics": LDA_TOPICS,
                    "lda_words_per_section": LDA_WORDS_PER_SECTION,
                },
            },
            "counts": {
                "pages": pages,
                "images": images_total,
                "chunks": len(chunks),
                "sections": len(sections),
            },
            "totals": {
                "total_with_models_s": round(total_with_models, 3),
                "models_load_s": round(models_total, 3),
                "total_without_models_s": round(total_with_models - models_total, 3),
            },
        }
    )
    events.finalize()
    log.info(
        "✅ Build complete | total=%.3fs (w/ models) | models=%.3fs | flow=%.3fs",
        total_with_models,
        models_total,
        total_with_models - models_total,
    )


if __name__ == "__main__":
    main()
