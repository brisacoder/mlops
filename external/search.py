from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import spacy
import torch
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize as sp_normalize
from sentence_transformers import SentenceTransformer


# Try optional ANN lib (CPU). If missing, we’ll use NumPy brute-force.
_HNSW_AVAILABLE = False
try:
    import hnswlib

    _HNSW_AVAILABLE = True
except Exception:
    _HNSW_AVAILABLE = False

# ---- GPU embedder (we reuse your existing external.semantics if present)
try:
    from external.semantics import embed_texts  # uses CUDA if available
except Exception:
    # Minimal inline GPU embedder (fallback)
    _embedder = None

    def embed_texts(texts: List[str]) -> np.ndarray:
        global _embedder
        if _embedder is None:
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _embedder = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=device
            )
        vecs = _embedder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
        )
        return vecs.astype("float32")


# ===============================
# Config (env-tunable)
# ===============================
CAPTION_SIM_THRESHOLD = float(os.getenv("CAPTION_SIM_THRESHOLD", "0.45"))
CAND_TOPN = int(os.getenv("CAND_TOPN", "200"))
IMG_TOPN = int(os.getenv("IMG_TOPN", "50"))
ENTITY_BOOST = float(os.getenv("ENTITY_BOOST", "0.15"))

# HNSW params
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCT = int(os.getenv("HNSW_EF_CONSTRUCT", "200"))
HNSW_EF_QUERY = int(os.getenv("HNSW_EF_QUERY", "50"))

# Ford-style figure ids (E######) + classic Fig. N
FIG_REF_RX = re.compile(
    r"\b(?:Fig(?:ure)?\.?\s*\d+[A-Za-z\-]*)|(?:E\d{5,})", re.IGNORECASE
)


# ===============================
# Simple daily JSONL logger (per-query)
# ===============================
class _JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: Dict[str, Any]) -> None:
        event.setdefault(
            "ts", dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _make_queries_logger(artifacts_dir: Path) -> _JsonlLogger:
    fname = f"queries-{dt.datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    return _JsonlLogger(artifacts_dir / "metrics" / fname)


# ===============================
# Load & warm runtime (GPU emb + NER, CPU ANN)
# ===============================
def load_runtime(artifacts_dir: str | Path = "./artifacts") -> Dict[str, Any]:
    ART = Path(artifacts_dir)

    # ---- Text indices (TF-IDF / BM25 / chunks)
    chunks = json.loads((ART / "chunks.json").read_text(encoding="utf-8"))
    tfidf = pickle.load((ART / "tfidf.pkl").open("rb"))
    X = pickle.load((ART / "tfidf_X.pkl").open("rb"))  # CSR
    X = sp_normalize(X, norm="l2", axis=1, copy=False)
    bm25: BM25Okapi = pickle.load((ART / "bm25.pkl").open("rb"))
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))

    # ---- Captions: load/compute embeddings + HNSW
    cap_idx = {"files": [], "texts": []}
    cap_embs = np.zeros((0, 384), dtype="float32")
    cap_ann = None
    cap_idx_path = ART / "caption_index.json"
    cap_embs_path = ART / "caption_embs.npy"
    cap_hnsw_path = ART / "hnsw_captions.bin"

    if cap_idx_path.exists():
        cap_idx = json.loads(cap_idx_path.read_text(encoding="utf-8"))
        if cap_embs_path.exists():
            cap_embs = np.load(cap_embs_path)
        else:
            # compute & persist
            cap_embs = embed_texts(cap_idx["texts"])
            np.save(cap_embs_path, cap_embs)
        if _HNSW_AVAILABLE and cap_embs.shape[0] > 0:
            cap_ann = _load_or_build_hnsw(cap_embs, cap_hnsw_path)

    # ---- Chunks: load/compute embeddings + HNSW (for candidate gen)
    ch_embs = np.zeros((0, 384), dtype="float32")
    ch_ann = None
    ch_embs_path = ART / "chunk_embs.npy"
    ch_hnsw_path = ART / "hnsw_chunks.bin"

    if ch_embs_path.exists():
        ch_embs = np.load(ch_embs_path)
    else:
        # embed chunk texts (can be thousands; GPU makes this quick)
        ch_texts = [c["text"] for c in chunks]
        ch_embs = embed_texts(ch_texts)
        np.save(ch_embs_path, ch_embs)

    if _HNSW_AVAILABLE and ch_embs.shape[0] > 0:
        ch_ann = _load_or_build_hnsw(ch_embs, ch_hnsw_path)

    # ---- spaCy transformers NER (GPU)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    spacy.require_gpu()
    nlp_q = spacy.load(
        "en_core_web_trf", disable=["tagger", "lemmatizer", "morphologizer", "textcat"]
    )

    # ---- Logger
    q_logger = _make_queries_logger(ART)

    # ---- Closure
    def search_fn(q: str, top_k: int = 5) -> Dict[str, Any]:
        t0 = time.perf_counter()
        out = _search_fast(
            q=q,
            top_k=top_k,
            chunks=chunks,
            tfidf=tfidf,
            X=X,
            bm25=bm25,
            manifest=manifest,
            cap_idx=cap_idx,
            cap_embs=cap_embs,
            cap_ann=cap_ann,
            ch_embs=ch_embs,
            ch_ann=ch_ann,
            nlp_q=nlp_q,
            ART=ART,
            q_logger=q_logger,
        )
        out["_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return out

    return {
        "search_fn": search_fn,
        "chunks": chunks,
        "tfidf": tfidf,
        "X": X,
        "bm25": bm25,
        "manifest": manifest,
        "cap_idx": cap_idx,
        "cap_embs": cap_embs,
        "cap_ann": cap_ann,
        "ch_embs": ch_embs,
        "ch_ann": ch_ann,
        "nlp_q": nlp_q,
        "queries_logger": q_logger,
    }


# ===============================
# HNSW helpers (CPU)
# ===============================
def _load_or_build_hnsw(vecs: np.ndarray, path: Path):
    index = hnswlib.Index(space="cosine", dim=vecs.shape[1])
    if path.exists():
        index.load_index(str(path))
        index.set_ef(HNSW_EF_QUERY)
        return index
    index.init_index(
        max_elements=vecs.shape[0], ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M
    )
    index.add_items(vecs, np.arange(vecs.shape[0]))
    index.set_ef(HNSW_EF_QUERY)
    index.save_index(str(path))
    return index


def _ann_search(index, q_vec: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    # q_vec: (1, d) L2-normalized; cosine distance ≈ 1 - cosine
    labels, distances = index.knn_query(q_vec, k=topn)
    sims = 1.0 - distances[0]
    idxs = labels[0]
    return sims, idxs


# ===============================
# Core search (GPU emb + NER, CPU ANN)
# ===============================
def _search_fast(
    q: str,
    top_k: int,
    chunks,
    tfidf,
    X,
    bm25,
    manifest,
    cap_idx,
    cap_embs,
    cap_ann,
    ch_embs,
    ch_ann,
    nlp_q,
    ART: Path,
    q_logger: _JsonlLogger,
) -> Dict[str, Any]:

    timings: Dict[str, float] = {}

    # ---- TF-IDF cosine (fast sparse matmul)
    t = time.perf_counter()
    qv = tfidf.transform([q])  # 1 x vocab
    qv = sp_normalize(qv, norm="l2", copy=False)
    sim = qv.dot(X.T).A.ravel()  # 1 x chunks
    timings["tfidf_ms"] = _ms(t)

    # ---- BM25
    t = time.perf_counter()
    bm = bm25.get_scores(_tok(q))
    timings["bm25_ms"] = _ms(t)

    # ---- Chunk ANN (embed query on GPU; ANN on CPU)
    cand_embed = set()
    if ch_ann is not None and ch_embs.shape[0] > 0:
        t = time.perf_counter()
        qch = embed_texts([q]).astype("float32")  # (1, d), normalized
        sims, idxs = _ann_search(ch_ann, qch, CAND_TOPN)
        cand_embed = set(map(int, idxs.tolist()))
        timings["chunk_ann_ms"] = _ms(t)
    else:
        timings["chunk_ann_ms"] = 0.0

    # ---- Candidate union + fusion
    t = time.perf_counter()
    idx_tfidf = np.argsort(-sim)[:CAND_TOPN]
    idx_bm25 = np.argsort(-bm)[:CAND_TOPN]
    cand = np.array(sorted(set(idx_tfidf).union(set(idx_bm25)).union(cand_embed)))

    sim_n = _minmax(sim[cand])
    bm_n = _minmax(bm[cand])
    base = 0.6 * sim_n + 0.4 * bm_n
    timings["fusion_ms"] = _ms(t)

    # ---- Entity boost (spaCy transformers on GPU)
    t = time.perf_counter()
    ent_map = {}
    ent_path = ART / "entity_index.json"
    if ent_path.exists():
        ent_map = json.loads(ent_path.read_text(encoding="utf-8"))
        doc = nlp_q(q)
        qents = {f"{e.text.strip().lower()}|{e.label_}" for e in doc.ents}
        if qents:
            cid_to_local = {chunks[i]["id"]: j for j, i in enumerate(cand)}
            boost = np.zeros_like(base)
            for qe in qents:
                for cid in ent_map.get(qe, []):
                    j = cid_to_local.get(cid)
                    if j is not None:
                        boost[j] += ENTITY_BOOST
            score = base + boost
        else:
            score = base
    else:
        score = base
    timings["entity_boost_ms"] = _ms(t)

    order_local = np.argsort(-score)[:top_k]
    chosen = cand[order_local]

    # ---- Images: (1) exact figure refs, else (2) caption ANN on same pages
    res = []
    t_img_all = time.perf_counter()
    # Query embedding for captions (GPU)
    qcap = embed_texts([q]).astype("float32")  # (1, d)
    img_total = 0
    for local_i, i in enumerate(chosen):
        c = chunks[int(i)]
        fig_refs = _collect_fig_refs(c["text"])
        imgs_precise = []
        page_set = set(c["pages"][:2])

        for pg in manifest["pages"]:
            if page_set and pg["page"] not in page_set:
                continue
            for im in pg["images"]:
                fid = (im.get("figure_id") or "").lower()
                if fid and fid in fig_refs:
                    imgs_precise.append(im)

        if imgs_precise:
            imgs = imgs_precise[:4]
        else:
            imgs = _filter_images_for_chunk_semantic(
                q_vec=qcap,
                pages=c["pages"],
                manifest=manifest,
                cap_idx=cap_idx,
                cap_embs=cap_embs,
                cap_ann=cap_ann,
                sim_threshold=CAPTION_SIM_THRESHOLD,
                max_imgs=4,
                topn=IMG_TOPN,
            )

        img_total += len(imgs)
        res.append(
            {
                "chunk_id": c["id"],
                "score": float(score[order_local[local_i]]),
                "pages": c["pages"],
                "text": c["text"],
                "images": imgs,
            }
        )
    timings["images_ms"] = _ms(t_img_all)

    # ---- Log timings
    try:
        _make_queries_logger(ART).write(
            {
                "type": "query",
                "q": q,
                "top_k": top_k,
                "tfidf_ms": timings["tfidf_ms"],
                "bm25_ms": timings["bm25_ms"],
                "chunk_ann_ms": timings["chunk_ann_ms"],
                "fusion_ms": timings["fusion_ms"],
                "entity_boost_ms": timings["entity_boost_ms"],
                "images_ms": timings["images_ms"],
                "total_ms": round(sum(timings.values()), 3),
                "chosen_chunk_ids": [chunks[int(i)]["id"] for i in chosen],
                "chosen_pages": [chunks[int(i)]["pages"] for i in chosen],
                "images_selected": img_total,
                "ann": "hnswlib" if _HNSW_AVAILABLE else "numpy",
            }
        )
    except Exception:
        pass

    return {"query": q, "results": res}


# ===============================
# Caption search (GPU emb, ANN on CPU)
# ===============================
def _filter_images_for_chunk_semantic(
    q_vec: np.ndarray,
    pages: List[int],
    manifest: Dict[str, Any],
    cap_idx: Dict[str, Any],
    cap_embs: np.ndarray,
    cap_ann,  # hnswlib index or None
    sim_threshold: float,
    max_imgs: int = 4,
    topn: int = IMG_TOPN,
) -> List[Dict[str, Any]]:
    if q_vec is None or cap_embs.shape[0] == 0:
        return []
    page_set = set(pages[:2]) if pages else set()
    cand_files: List[str] = []
    for pg in manifest["pages"]:
        if page_set and pg["page"] not in page_set:
            continue
        for im in pg["images"]:
            cand_files.append(im["file"])
    if not cand_files:
        return []

    if cap_ann is not None:
        sims, idxs = _ann_search(cap_ann, q_vec, topn)
        ranked = [(float(sims[j]), int(idxs[j])) for j in range(len(idxs))]
    else:
        sims = (cap_embs @ q_vec.T).ravel()
        top = np.argsort(-sims)[:topn]
        ranked = [(float(sims[i]), int(i)) for i in top]

    files_set = set(cand_files)
    picked, seen = [], set()
    for score, row in ranked:
        if score < sim_threshold:
            continue
        f = cap_idx["files"][row]
        if f not in files_set or f in seen:
            continue
        seen.add(f)
        # locate metadata
        for pg in manifest["pages"]:
            if page_set and pg["page"] not in page_set:
                continue
            for im in pg["images"]:
                if im["file"] == f:
                    picked.append(im)
                    if len(picked) >= max_imgs:
                        return picked
                    break
    return picked


# ===============================
# Utilities
# ===============================
def _tok(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9\-]+", (s or "").lower())


def _collect_fig_refs(text: str) -> set[str]:
    return set(m.group(0).lower() for m in FIG_REF_RX.finditer(text or ""))


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)
