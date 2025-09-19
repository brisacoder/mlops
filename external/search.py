# external/search.py
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize as sp_normalize
from sentence_transformers import CrossEncoder

# Optional ANN (CPU)
_HNSW_AVAILABLE = False
try:
    import hnswlib  # type: ignore
    _HNSW_AVAILABLE = True
except Exception:
    _HNSW_AVAILABLE = False

from external.runtime import load_models

# -------- config --------
CAPTION_SIM_THRESHOLD = float(os.getenv("CAPTION_SIM_THRESHOLD", "0.45"))
CAND_TOPN              = int(os.getenv("CAND_TOPN", "200"))
IMG_TOPN               = int(os.getenv("IMG_TOPN", "80"))
ENTITY_BOOST           = float(os.getenv("ENTITY_BOOST", "0.15"))  # (unused; entity map optional)

# Fusion weights
W_TFIDF = float(os.getenv("W_TFIDF", "0.25"))
W_BM25  = float(os.getenv("W_BM25",  "0.25"))
W_EMB   = float(os.getenv("W_EMB",   "0.50"))

# Reranker
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CAND_N  = int(os.getenv("RERANK_CAND_N",  "100"))
RERANK_WEIGHT  = float(os.getenv("RERANK_WEIGHT", "0.75"))

# Figure refs, TOC suppression
FIG_REF_RX = re.compile(r"\b(?:Fig(?:ure)?\.?\s*\d+[A-Za-z\-]*)|(?:E\d{5,})", re.IGNORECASE)
TOC_WORDS = ("table of contents", "contents", "index", "glossary")


# -------- JSONL logger --------
class _JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: Dict[str, Any]) -> None:
        event.setdefault("ts", dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _make_queries_logger(artifacts_dir: Path) -> _JsonlLogger:
    fname = f"queries-{dt.datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    return _JsonlLogger(artifacts_dir / "metrics" / fname)


# -------- public loader --------
def load_runtime(artifacts_dir: str | Path = "./artifacts") -> Dict[str, Any]:
    ART = Path(artifacts_dir)

    # Core indices
    chunks = json.loads((ART / "chunks.json").read_text(encoding="utf-8"))
    tfidf = pickle.load((ART / "tfidf.pkl").open("rb"))
    X = pickle.load((ART / "tfidf_X.pkl").open("rb"))
    X = sp_normalize(X, norm="l2", axis=1, copy=False)
    bm25: BM25Okapi = pickle.load((ART / "bm25.pkl").open("rb"))
    manifest = json.loads((ART / "manifest.json").read_text(encoding="utf-8"))

    # Caption vectors + ANN
    cap_idx: Dict[str, Any] = {"files": [], "texts": []}
    cap_embs = np.zeros((0, 384), dtype="float32")
    cap_ann = None
    if (ART / "caption_index.json").exists():
        cap_idx = json.loads((ART / "caption_index.json").read_text(encoding="utf-8"))
        if (ART / "caption_embs.npy").exists():
            cap_embs = np.load(ART / "caption_embs.npy")
        if _HNSW_AVAILABLE and cap_embs.shape[0] > 0:
            cap_ann = _load_or_build_hnsw(cap_embs, ART / "hnsw_captions.bin")

    # Chunk vectors + ANN
    ch_embs = np.zeros((0, 384), dtype="float32")
    ch_ann = None
    if (ART / "chunk_embs.npy").exists():
        ch_embs = np.load(ART / "chunk_embs.npy")
        if _HNSW_AVAILABLE and ch_embs.shape[0] > 0:
            ch_ann = _load_or_build_hnsw(ch_embs, ART / "hnsw_chunks.bin")

    # Sections
    sections = json.loads((ART / "sections.json").read_text(encoding="utf-8")) if (ART / "sections.json").exists() else []
    sec_embs = np.load(ART / "section_embs.npy") if (ART / "section_embs.npy").exists() else np.zeros((0, ch_embs.shape[1] if ch_embs.size else 384), dtype="float32")

    # Models (one source of truth)
    loaded = load_models()
    nlp_q = loaded.nlp           # currently unused in this file (kept for parity/future)
    st_model = loaded.st_model

    # Dimension guard: runtime embedder must match artifacts
    def _get_dim(arr: np.ndarray) -> Optional[int]:
        return int(arr.shape[1]) if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.size > 0 else None

    art_dim = _get_dim(ch_embs) or _get_dim(cap_embs) or _get_dim(sec_embs)
    try:
        rt_dim = int(getattr(st_model, "get_sentence_embedding_dimension", lambda: None)() or 0)
    except Exception:
        rt_dim = 0
    if art_dim and rt_dim and art_dim != rt_dim:
        raise ValueError(
            f"Embedding dimension mismatch: artifacts={art_dim}, runtime={rt_dim}. "
            "Rebuild artifacts with the same SentenceTransformer you use at query time, "
            "or change runtime to the model used during build."
        )

    # Embed helper
    def _embed(texts: List[str]) -> np.ndarray:
        vecs = st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256)
        return vecs.astype("float32")

    # Reranker (GPU if available)
    import torch
    reranker = CrossEncoder(RERANKER_MODEL, device=("cuda" if torch.cuda.is_available() else "cpu"))

    q_logger = _make_queries_logger(ART)

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
            sections=sections,
            sec_embs=sec_embs,
            embed_fn=_embed,
            reranker=reranker,
            ART=ART,
            q_logger=q_logger,
        )
        out["_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return out

    return {
        "search_fn": search_fn,
        "chunks": chunks,
        "manifest": manifest,
        "spacy_query_model": getattr(nlp_q.meta, "name", "unknown") if hasattr(nlp_q, "meta") else "unknown",
    }


# -------- HNSW helpers --------
def _load_or_build_hnsw(vecs: np.ndarray, path: Path):
    index = hnswlib.Index(space="cosine", dim=vecs.shape[1])
    if path.exists():
        index.load_index(str(path))
        index.set_ef(int(os.getenv("HNSW_EF_QUERY", "50")))
        return index
    index.init_index(
        max_elements=vecs.shape[0],
        ef_construction=int(os.getenv("HNSW_EF_CONSTRUCT", "200")),
        M=int(os.getenv("HNSW_M", "16")),
    )
    index.add_items(vecs, np.arange(vecs.shape[0]))
    index.set_ef(int(os.getenv("HNSW_EF_QUERY", "50")))
    index.save_index(str(path))
    return index


def _ann_search(index, q_vec: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, distances = index.knn_query(q_vec, k=topn)
    sims = 1.0 - distances[0]
    idxs = labels[0]
    return sims, idxs


# -------- Core search --------
def _search_fast(
    q: str,
    top_k: int,
    chunks: List[Dict[str, Any]],
    tfidf,
    X,
    bm25: Any,
    manifest: Dict[str, Any],
    cap_idx: Dict[str, Any],
    cap_embs: np.ndarray,
    cap_ann: Any,
    ch_embs: np.ndarray,
    ch_ann: Any,
    sections: List[Dict[str, Any]],
    sec_embs: np.ndarray,
    embed_fn,
    reranker,
    ART: Path,
    q_logger: _JsonlLogger,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}

    # Section routing (semantic + topic/title lexical)
    def _tok(s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9\-]+", (s or "").lower())

    def _minmax(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        a = float(arr.min()); b = float(arr.max())
        if b - a <= 1e-9:
            return np.zeros_like(arr)
        return (arr - a) / (b - a)

    def _toc_index_penalty(text: str) -> float:
        if not text:
            return 0.0
        s = text.lower()
        pen = 0.0
        if any(w in s for w in TOC_WORDS): pen += 0.25
        if s.count("..") >= 10: pen += 0.2
        num = sum(ch.isdigit() for ch in s)
        if num > max(30, len(s) // 10): pen += 0.1
        return min(pen, 0.5)

    def _collect_fig_refs(text: str) -> set[str]:
        return set(m.group(0).lower() for m in FIG_REF_RX.finditer(text or ""))

    def _fallback_images_by_size(manifest, pages, max_imgs=4, min_wh=120, min_area=16000):
        page_set = set(pages[:2]) if pages else set()
        cands = []
        for pg in manifest["pages"]:
            if page_set and pg["page"] not in page_set:
                continue
            for im in pg["images"]:
                w = int(im.get("width", 0) or 0); h = int(im.get("height", 0) or 0)
                if w < min_wh or h < min_wh or (w * h) < min_area:
                    continue
                cands.append((w * h, im))
        cands.sort(key=lambda x: -x[0])
        return [im for _, im in cands[:max_imgs]]

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

        if _HNSW_AVAILABLE and cap_ann is not None:
            sims, idxs = _ann_search(cap_ann, q_vec, topn)
            ranked = [(float(sims[j]), int(idxs[j])) for j in range(len(idxs))]
        else:
            sims = (cap_embs @ q_vec.T).ravel()
            top = np.argsort(-sims)[:topn]
            ranked = [(float(sims[i]), int(i)) for i in top]

        files_set = set(cand_files)
        picked, seen = [], set()
        for score, row in ranked:
            if score < CAPTION_SIM_THRESHOLD:
                continue
            f = cap_idx["files"][row]
            if f not in files_set or f in seen:
                continue
            seen.add(f)
            # locate metadata for this file on the selected pages
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

    # --- pick best section(s)
    def _score_sections(query: str) -> List[int]:
        if sec_embs.size == 0:
            return []
        qv = embed_fn([query])  # (1,d)
        sem = (sec_embs @ qv.T).ravel()  # cosine
        q_terms = set(_tok(query))
        lex = []
        for s in sections:
            bag = set(s.get("topic_words", [])) | set(_tok(s.get("title", "")))
            lex.append(len(q_terms & bag))
        lex = np.array(lex, dtype="float32")
        lex = np.clip(lex, 0, 4) / 4.0
        score = 0.8 * _minmax(sem.astype("float32")) + 0.2 * _minmax(lex)
        return list(np.argsort(-score))

    sec_order = _score_sections(q)
    chosen_secs = sec_order[:1] if len(sec_order) else []
    allowed_chunk_ids = {i for sid in chosen_secs for i in sections[sid]["chunk_ids"]} if chosen_secs else set(range(len(chunks)))

    # --- TF-IDF
    t = time.perf_counter()
    qv_tfidf = tfidf.transform([q])
    qv_tfidf = sp_normalize(qv_tfidf, norm="l2", copy=False)
    sim = (qv_tfidf @ X.T).toarray().ravel()
    timings["tfidf_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # --- BM25
    t = time.perf_counter()
    bm = bm25.get_scores(_tok(q))
    timings["bm25_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # --- ANN candidate expansion (chunks)
    cand_embed: set[int] = set()
    if _HNSW_AVAILABLE and ch_ann is not None and ch_embs.shape[0] > 0:
        t = time.perf_counter()
        qch = embed_fn([q])
        sims, idxs = _ann_search(ch_ann, qch, CAND_TOPN)
        cand_embed = set(map(int, idxs.tolist()))
        timings["chunk_ann_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
    else:
        timings["chunk_ann_ms"] = 0.0

    # --- Candidate set (restricted to chosen section)
    idx_tfidf = np.argsort(-sim)[:CAND_TOPN]
    idx_bm25  = np.argsort(-bm)[:CAND_TOPN]
    cand = set(idx_tfidf).union(set(idx_bm25)).union(cand_embed)
    cand = np.array([i for i in cand if i in allowed_chunk_ids], dtype=int)
    if cand.size == 0:
        # fallback to global if section restriction was too strict
        cand = np.array(sorted(set(idx_tfidf).union(set(idx_bm25)).union(cand_embed)), dtype=int)
        if cand.size == 0:
            _safe_log_query(q_logger, q, top_k, timings, [], [], 0, ann="none")
            return {"query": q, "results": []}

    # --- Fusion (TF-IDF + BM25 + chunk-embedding cosine + tiny lexical overlap)
    t = time.perf_counter()
    sim_n = _minmax(sim[cand])
    bm_n  = _minmax(bm[cand])
    qch2  = embed_fn([q])
    emb_raw = (ch_embs[cand] @ qch2.T).ravel().astype("float32") if ch_embs.size else np.zeros_like(sim_n)
    emb_n   = _minmax(emb_raw)

    q_terms = set(_tok(q))
    overlap = np.array([len(q_terms & set(_tok(chunks[int(i)]["text"]))) for i in cand], dtype="float32")
    overlap = np.clip(overlap, 0, 3) / 3.0

    base = W_TFIDF * sim_n + W_BM25 * bm_n + W_EMB * emb_n + 0.05 * overlap
    penalties = np.array([_toc_index_penalty(chunks[int(i)]["text"]) for i in cand], dtype="float32")
    base = np.clip(base - penalties, 0.0, 1.0)
    timings["fusion_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # --- Cross-encoder rerank (top-N)
    t = time.perf_counter()
    topN = min(RERANK_CAND_N, cand.size)
    if topN > 0:
        order_fused = np.argsort(-base)[:topN]
        cand_top = cand[order_fused]
        texts_top = [chunks[int(i)]["text"] for i in cand_top]
        pairs = [(q, txt) for txt in texts_top]
        ce_scores = np.asarray(reranker.predict(pairs), dtype="float32")
        fused_sel = base[order_fused]
        fused_n = _minmax(fused_sel)
        ce_n = _minmax(ce_scores)
        mixed = (1.0 - RERANK_WEIGHT) * fused_n + RERANK_WEIGHT * ce_n
        local_rerank = np.argsort(-mixed)
        reranked = cand_top[local_rerank]
        tail = [i for i in cand.tolist() if i not in set(cand_top.tolist())]
        cand = np.concatenate([reranked, np.array(tail, dtype=int)]) if tail else reranked
        base_r = base.copy()
        base_r[order_fused] = mixed[local_rerank]
        base = base_r
    timings["rerank_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # --- Select top-k and attach images (figure-ids → caption ANN → size fallback)
    order_local = np.argsort(-base)[:top_k]
    chosen = cand[order_local]

    res = []
    t_img_all = time.perf_counter()
    qcap = embed_fn([q])  # (1,d)
    img_total = 0
    for local_i, i in enumerate(chosen):
        c = chunks[int(i)]
        fig_refs = _collect_fig_refs(c["text"])
        imgs_precise: List[Dict[str, Any]] = []
        page_set = set(c.get("pages", [])[:2])

        # 1) exact figure-id match on same pages
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
            # 2) caption-semantic on same pages
            imgs = _filter_images_for_chunk_semantic(
                q_vec=qcap,
                pages=c.get("pages", []),
                manifest=manifest,
                cap_idx=cap_idx,
                cap_embs=cap_embs,
                cap_ann=cap_ann,
                sim_threshold=CAPTION_SIM_THRESHOLD,
                max_imgs=4,
                topn=IMG_TOPN,
            )

        if not imgs:
            # 3) largest-by-size fallback on same pages
            imgs = _fallback_images_by_size(manifest, c.get("pages", []), max_imgs=4, min_wh=120, min_area=16000)

        img_total += len(imgs)
        res.append(
            {
                "chunk_id": c["id"],
                "score": float(base[order_local[local_i]]),
                "pages": c.get("pages", []),
                "text": c.get("text", ""),
                "images": imgs,
            }
        )
    timings["images_ms"] = round((time.perf_counter() - t_img_all) * 1000.0, 3)

    _safe_log_query(
        q_logger,
        q,
        top_k,
        timings,
        [chunks[int(i)]["id"] for i in chosen],
        [chunks[int(i)]["pages"] for i in chosen],
        img_total,
        ann=("hnswlib" if _HNSW_AVAILABLE and ch_ann is not None else "none"),
    )

    return {"query": q, "results": res}


# -------- logging helper --------
def _safe_log_query(
    q_logger: _JsonlLogger,
    q: str,
    top_k: int,
    timings: Dict[str, float],
    chosen_chunk_ids: List[str],
    chosen_pages: List[List[int]],
    images_selected: int,
    ann: str,
) -> None:
    try:
        q_logger.write(
            {
                "type": "query",
                "q": q,
                "top_k": top_k,
                "tfidf_ms": timings.get("tfidf_ms", 0.0),
                "bm25_ms": timings.get("bm25_ms", 0.0),
                "chunk_ann_ms": timings.get("chunk_ann_ms", 0.0),
                "fusion_ms": timings.get("fusion_ms", 0.0),
                "rerank_ms": timings.get("rerank_ms", 0.0),
                "images_ms": timings.get("images_ms", 0.0),
                "total_ms": round(sum(timings.values()), 3),
                "chosen_chunk_ids": chosen_chunk_ids,
                "chosen_pages": chosen_pages,
                "images_selected": images_selected,
                "ann": ann,
            }
        )
    except Exception:
        pass
