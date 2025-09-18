#!/usr/bin/env python3
from __future__ import annotations
import os, json, time, logging
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

from external.pdf_extract import extract_pdf
from external.chunking import split_into_chunks
from external.index_build import build_indices
from external.entity_index import build_entity_index
from external.runtime import load_models

# Optional ANN on CPU
_HNSW_AVAILABLE = False
try:
    import hnswlib

    _HNSW_AVAILABLE = True
except Exception:
    _HNSW_AVAILABLE = False

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("build")

# ---------- env ----------
load_dotenv()
PDF_PATH = Path(os.getenv("PDF_PATH", "./data/manual.pdf")).resolve()
ART_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()
ART_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_MIN = int(os.getenv("CHUNK_MIN", "400"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "1200"))

# HNSW
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "200"))
HNSW_EF_QUERY = int(os.getenv("HNSW_EF_QUERY", "50"))


# ---------- tiny perf helper ----------
class Perf:
    def __init__(self):
        self.t = {}
        self._tmp = {}

    def start(self, name):
        self._tmp[name] = time.perf_counter()

    def stop(self, name):
        v = time.perf_counter() - self._tmp.pop(name, time.perf_counter())
        self.t[name] = round(v, 3)
        return v

    def write_json(self, path: Path, extra: dict):
        data = {**self.t, **extra}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


perf = Perf()


def _build_hnsw(vecs: np.ndarray, save_path: Path):
    if not _HNSW_AVAILABLE or vecs.size == 0:
        return
    dim = vecs.shape[1]
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(
        max_elements=vecs.shape[0], ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M
    )
    idx.add_items(vecs, np.arange(vecs.shape[0]))
    idx.set_ef(HNSW_EF_QUERY)
    idx.save_index(str(save_path))


def main():
    log.info(f"Artifacts dir: {ART_DIR}")
    log.info(f"PDF: {PDF_PATH}")

    # 0) Preload big models (and time it)
    perf.start("t_load_models")
    loaded = load_models(log)
    perf.stop("t_load_models")
    perf.t["t_load_spacy"] = round(loaded.t_spacy_load_s, 3)
    perf.t["t_load_st"] = round(loaded.t_st_load_s, 3)
    perf.t["using_gpu"] = loaded.using_gpu

    # 1) Extract
    perf.start("t_extract")
    manifest = extract_pdf(PDF_PATH, ART_DIR)
    t = perf.stop("t_extract")
    pages = len(manifest["pages"])
    images_total = sum(len(p["images"]) for p in manifest["pages"])
    log.info(f"1) Extracted PDF | pages={pages} images={images_total} | {t:.3f}s")

    # 2) Chunk
    perf.start("t_chunk")
    chunks = split_into_chunks(
        manifest["pages"], min_chars=CHUNK_MIN, max_chars=CHUNK_MAX
    )
    t = perf.stop("t_chunk")
    log.info(
        f"2) Chunked | chunks={len(chunks)} (min={CHUNK_MIN}, max={CHUNK_MAX}) | {t:.3f}s"
    )

    # 3) Indices
    perf.start("t_indexes")
    build_indices(chunks, ART_DIR)
    t = perf.stop("t_indexes")
    log.info(f"3) Text indices built (TF-IDF, BM25) | {t:.3f}s")

    # 4) Entities (spaCy TRF) — pass preloaded nlp to capture true time
    perf.start("t_entities")
    t_entities = build_entity_index(chunks, ART_DIR, batch_size=32, nlp=loaded.nlp)
    perf.stop(
        "t_entities"
    )  # value set by build_entity_index; we also keep the returned for clarity
    log.info(f"4) Entity index (spaCy trf) | {t_entities:.3f}s")

    # 5) Caption embeddings (SentenceTransformers) + optional HNSW
    cap_idx_path = ART_DIR / "caption_index.json"
    cap_embs_path = ART_DIR / "caption_embs.npy"
    cap_hnsw_path = ART_DIR / "hnsw_captions.bin"

    cap_files, cap_texts = [], []
    for pg in manifest["pages"]:
        for im in pg["images"]:
            cap_files.append(im["file"])
            cap_texts.append((im.get("caption") or "").strip())

    if len(cap_texts) > 0:
        cap_idx = {"files": cap_files, "texts": cap_texts}
        cap_idx_path.write_text(
            json.dumps(cap_idx, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        log.info(f"5a) Embedding {len(cap_texts)} captions on GPU…")
        perf.start("t_embed_captions")
        cap_embs = loaded.st_model.encode(
            cap_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
        ).astype("float32")
        np.save(cap_embs_path, cap_embs)
        t = perf.stop("t_embed_captions")
        log.info(f"5a) Captions embedded | {t:.3f}s")

        if _HNSW_AVAILABLE:
            perf.start("t_hnsw_captions")
            _build_hnsw(cap_embs, cap_hnsw_path)
            t = perf.stop("t_hnsw_captions")
            log.info(f"5b) HNSW (captions) built → {cap_hnsw_path.name} | {t:.3f}s")
        else:
            log.info("5b) HNSW (captions) skipped (hnswlib not installed)")
    else:
        log.info("5) No captions; skipping embeddings/HNSW")

    # 6) Chunk embeddings + optional HNSW
    chunk_texts = [c["text"] for c in chunks]
    log.info(f"6a) Embedding {len(chunk_texts)} chunks on GPU…")
    perf.start("t_embed_chunks")
    ch_embs = loaded.st_model.encode(
        chunk_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
    ).astype("float32")
    np.save(ART_DIR / "chunk_embs.npy", ch_embs)
    t = perf.stop("t_embed_chunks")
    log.info(f"6a) Chunks embedded | {t:.3f}s")

    if _HNSW_AVAILABLE and ch_embs.size > 0:
        perf.start("t_hnsw_chunks")
        _build_hnsw(ch_embs, ART_DIR / "hnsw_chunks.bin")
        t = perf.stop("t_hnsw_chunks")
        log.info(f"6b) HNSW (chunks) built → hnsw_chunks.bin | {t:.3f}s")
    else:
        log.info("6b) HNSW (chunks) skipped (missing hnswlib or no vectors)")

    # totals & persist metrics
    flow_total = sum(v for k, v in perf.t.items() if k.startswith("t_"))
    perf.write_json(
        ART_DIR / "metrics" / "run-latest.json",
        {
            "pages": pages,
            "images_total": images_total,
            "chunks": len(chunks),
            "flow_total_s": round(flow_total, 3),
        },
    )
    log.info("✅ Build complete.")


if __name__ == "__main__":
    main()
