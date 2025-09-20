# external/search.py
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import normalize as sp_normalize
from sentence_transformers import CrossEncoder

from external.runtime import load_models
from external.nltk_setup import get_wordnet  # your proven bootstrap


# -----------------------------------------------------------
# Config
# -----------------------------------------------------------

DEBUG_RETURN = os.getenv("SEARCH_DEBUG_RETURN", "0").lower() in ("1", "true", "yes")

CAND_TOPN = int(os.getenv("CAND_TOPN", "200"))
# Keep TF-IDF wiring for backward compatibility, but we do not use it in fusion.
W_TFIDF = float(os.getenv("W_TFIDF", "0.0"))
W_BM25 = float(os.getenv("W_BM25", "0.35"))
W_EMB = float(os.getenv("W_EMB", "0.65"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CAND_N = int(os.getenv("RERANK_CAND_N", "100"))
RERANK_WEIGHT = float(os.getenv("RERANK_WEIGHT", "0.75"))

CAPTION_SIM_THRESHOLD = float(os.getenv("CAPTION_SIM_THRESHOLD", "0.45"))
IMG_TOPN = int(os.getenv("IMG_TOPN", "80"))
MIN_IMG_WH = int(os.getenv("MIN_IMG_WH", "120"))
MIN_IMG_AREA = int(os.getenv("MIN_IMG_AREA", "25000"))
FALLBACK_MAX_PAGES = int(os.getenv("FALLBACK_MAX_PAGES", "8"))

TOC_WORDS = ("table of contents", "contents", "index", "glossary")
TOC_TITLES = {"index", "table of contents", "contents", "glossary"}

FIG_REF_RX = re.compile(r"\\b(?:Fig(?:ure)?\\.?\\s*\\d+[A-Za-z\\-]*)|(?:E\\d{5,})", re.IGNORECASE)

# Optional ANN via hnswlib
_HNSW_AVAILABLE = False
try:
    import hnswlib  # type: ignore

    _HNSW_AVAILABLE = True
except Exception:
    _HNSW_AVAILABLE = False

# WordNet (mandatory via your helper)
WN = get_wordnet()


# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------


class JsonlLogger:
    """Append-only JSONL writer for query diagnostics."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: Dict[str, Any]) -> None:
        """Write a single JSON event with UTC timestamp."""
        event.setdefault(
            "ts", dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\\n")


def make_queries_logger(artifacts_dir: Path) -> JsonlLogger:
    """Create a JSONL logger file under artifacts/metrics for the current UTC day."""
    fname = f"queries-{dt.datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    return JsonlLogger(artifacts_dir / "metrics" / fname)


# -----------------------------------------------------------
# Tokenization helpers (unigram + bigram)
# -----------------------------------------------------------


def tok1(text: str) -> List[str]:
    """Tokenize into case-folded unigrams allowing letters, numbers and dashes."""
    return re.findall(r"[A-Za-z0-9\\-]+", (text or "").lower())


def tok12(text: str) -> List[str]:
    """Return unigrams + bigrams for better BM25 recall (index-time choice dependent)."""
    unis = tok1(text)
    bigs = [f"{unis[i]} {unis[i+1]}" for i in range(len(unis) - 1)]
    return unis + bigs


def q_tok12(query: str, expansions: Set[str]) -> List[str]:
    """
    Query tokens with unigrams+bigrams + expanded synonyms merged in.
    Synonyms are kept as unigrams; bigrams are derived only from surface query.
    """
    unis = tok1(query)
    bigs = [f"{unis[i]} {unis[i+1]}" for i in range(len(unis) - 1)]
    return list(set(unis + bigs + list(expansions)))


# -----------------------------------------------------------
# Generic utils
# -----------------------------------------------------------


def minmax(arr: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] with protection against zero span."""
    if arr.size == 0:
        return arr
    a = float(arr.min())
    b = float(arr.max())
    if b - a <= 1e-9:
        return np.zeros_like(arr)
    return (arr - a) / (b - a)


def toc_index_penalty(text: str) -> float:
    """
    Heuristic penalty for TOC/Index-like text:
    - presence of cue words (contents/index/glossary)
    - many leader dots
    - high digit density
    Caps at 0.5.
    """
    if not text:
        return 0.0
    s = text.lower()
    pen = 0.0
    if any(w in s for w in TOC_WORDS):
        pen += 0.25
    if s.count("..") >= 10:
        pen += 0.2
    num = sum(ch.isdigit() for ch in s)
    if num > max(30, len(s) // 10):
        pen += 0.1
    return float(min(pen, 0.5))


def collect_fig_refs(text: str) -> set[str]:
    """Collect figure reference strings like 'Fig. 3' or 'E12345' present in chunk text."""
    return set(m.group(0).lower() for m in FIG_REF_RX.finditer(text or ""))


def load_or_build_hnsw(vecs: np.ndarray, path: Path):
    """Create or load an HNSW index for cosine similarity over L2-normalized vectors."""
    index = hnswlib.Index(space="cosine", dim=vecs.shape[1])  # type: ignore[name-defined]
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


def ann_search(index, q_vec: np.ndarray, topn: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run ANN search and return (similarities, indices)."""
    labels, distances = index.knn_query(q_vec, k=topn)
    sims = 1.0 - distances[0]
    idxs = labels[0]
    return sims, idxs


# -----------------------------------------------------------
# POS-aware WordNet expansion (generic)
# -----------------------------------------------------------

# Map NLTK treebank POS tags to WordNet POS
_TB2WN = {
    "N": "n",
    "V": "v",
    "J": "a",  # adjectives (JJ)
    "R": "r",  # adverbs (RB)
}


def _tbtag(word_list: List[str]) -> List[Tuple[str, str]]:
    """
    Lightweight POS tag using NLTK. Your nltk_setup should ensure tagger data is present.
    """
    import nltk

    return nltk.pos_tag(word_list)


def _to_wordnet_pos(treebank_tag: str) -> Optional[str]:
    """Map a Penn Treebank tag initial to a WordNet POS code."""
    if not treebank_tag:
        return None
    key = treebank_tag[0]
    return _TB2WN.get(key)


def wordnet_pos_synonyms(tokens: Set[str]) -> Set[str]:
    """
    For each token, POS-tag it, get synonyms for the SAME POS only.
    Also add derivationally related forms for richer surface matches.
    """
    out: Set[str] = set()
    pairs = _tbtag(sorted(tokens))
    for w, tb in pairs:
        wn_pos = _to_wordnet_pos(tb)
        if not wn_pos:
            continue
        for syn in WN.synsets(w, pos=wn_pos):
            for lem in syn.lemmas():
                out.add(lem.name().replace("_", " ").lower())
                # derivationally related forms
                for dr in lem.derivationally_related_forms():
                    out.add(dr.name().replace("_", " ").lower())
    # keep only alnum-ish tokens
    return {t for t in out if re.fullmatch(r"[A-Za-z0-9\- ]{2,30}", t)}


def simple_inflections(tokens: Set[str]) -> Set[str]:
    """Very light inflectional variants (no hardcoded task verbs)."""
    base = set(tokens)
    for t in list(tokens):
        if len(t) <= 2:
            continue
        if t.endswith("e"):
            base.update({t[:-1] + "ing", t + "d", t + "s"})
        else:
            base.update({t + "ing", t + "ed", t + "s"})
    return base


def expand_terms(query: str) -> Set[str]:
    """
    Generic expansion:
      - take query tokens
      - POS-aware WordNet synonyms (same POS)
      - simple inflections
    No hardcoded domain lists.
    """
    q_terms = set(tok1(query))
    expanded = q_terms | wordnet_pos_synonyms(q_terms) | simple_inflections(q_terms)
    return {w for w in expanded if 2 <= len(w) <= 30}


# -----------------------------------------------------------
# Section scoring & texts
# -----------------------------------------------------------


def build_section_texts(sections: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Concatenate chunk texts per section using a robust ID→index mapping.
    Sections contain chunk_ids like 'c0000', 'c0001', ... so we can't int()-parse.
    """
    # Map chunk_id (e.g., 'c0007') to its row index in `chunks`
    id_to_idx = {c.get("id"): i for i, c in enumerate(chunks)}

    out: List[str] = []
    for sec in sections:
        buf: List[str] = []
        for cid in sec.get("chunk_ids", []):
            idx = id_to_idx.get(cid)

            # Fallback: tolerate raw numeric IDs or mixed formatting
            if idx is None and isinstance(cid, str):
                digits = "".join(ch for ch in cid if ch.isdigit())
                if digits.isdigit():
                    n = int(digits)
                    if 0 <= n < len(chunks) and chunks[n].get("id") == cid:
                        idx = n

            # Only append if we resolved to a valid index
            if idx is not None and 0 <= idx < len(chunks):
                text = chunks[idx].get("text", "")
                if text:
                    buf.append(text)

        out.append(" ".join(buf).strip())
    return out


def is_index_like_section(sec: Dict[str, Any]) -> bool:
    """Detect index/TOC-like sections based on title/topic_words stats."""
    title = (sec.get("title") or "").strip().lower()
    if title in TOC_TITLES or any(t in title for t in TOC_TITLES):
        return True
    words = " ".join(sec.get("topic_words", []) or [])
    dots = words.count("..")
    digits = sum(ch.isdigit() for ch in words)
    if dots >= 8 or digits > max(25, len(words) // 8):
        return True
    return False


def score_sections(
    query: str,
    sections: List[Dict[str, Any]],
    sec_embs: np.ndarray,
    embed_fn,
    sec_bm25: Optional[BM25Okapi],
    sec_texts_tokens: Optional[List[List[str]]],
) -> Tuple[List[int], Set[str], np.ndarray]:
    """
    3-way blend for routing:
      0.55 * semantic + 0.30 * section-BM25 + 0.15 * lexical(over title/topic_words)
    Penalties for TOC/Index. Require overlap in full section text to avoid generic sections.
    """
    if not sections or sec_embs.size == 0:
        return [], set(), np.zeros((0,), dtype="float32")

    expanded = expand_terms(query)
    q_terms = set(tok1(query)) | expanded

    # semantic
    qv = embed_fn([" ".join(sorted(tok1(query)))])
    sem = (sec_embs @ qv.T).ravel().astype("float32")
    sem_n = minmax(sem)

    # section BM25
    bm = np.zeros((len(sections),), dtype="float32")
    if sec_bm25 is not None and sec_texts_tokens:
        bm = sec_bm25.get_scores(list(q_terms))
    bm_n = minmax(bm)

    # lexical overlap (title + topic words)
    lex = []
    for s in sections:
        bag = set(s.get("topic_words", [])) | set(tok1(s.get("title", "")))
        lex.append(len(q_terms & bag))
    lex = np.array(lex, dtype="float32")
    lex_n = np.clip(lex, 0, 4) / 4.0

    score = 0.55 * sem_n + 0.30 * bm_n + 0.15 * lex_n
    pen = np.array([0.6 if is_index_like_section(s) else 0.0 for s in sections], dtype="float32")
    score = np.clip(score - pen, 0.0, 1.0)

    # Must have some overlap in section full text
    if sec_texts_tokens:
        has_any = np.zeros((len(sections),), dtype="float32")
        for i, toks in enumerate(sec_texts_tokens):
            if set(toks) & q_terms:
                has_any[i] = 1.0
        score = score * (0.15 + 0.85 * has_any)

    order = list(np.argsort(-score))
    return order, expanded, score


# -----------------------------------------------------------
# Image selection
# -----------------------------------------------------------


def fallback_images_by_size(
    manifest: Dict[str, Any],
    pages: Sequence[int],
    max_imgs: int = 4,
    min_wh: int = MIN_IMG_WH,
    min_area: int = MIN_IMG_AREA,
    take_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Pick largest images on the requested pages as a last resort."""
    if not pages:
        return []
    use_pages = list(pages) if take_pages is None else list(pages)[:max(take_pages, 1)]
    page_set = set(use_pages)
    cands: List[Tuple[int, Dict[str, Any]]] = []
    for pg in manifest.get("pages", []):
        if page_set and pg.get("page") not in page_set:
            continue
        for im in pg.get("images", []):
            w, h = int(im.get("width", 0) or 0), int(im.get("height", 0) or 0)
            if w < min_wh or h < min_wh or (w * h) < min_area:
                continue
            cands.append((w * h, im))
    cands.sort(key=lambda x: -x[0])
    return [im for _, im in cands[:max_imgs]]


def filter_images_semantic(
    q_vec: np.ndarray,
    pages: Sequence[int],
    manifest: Dict[str, Any],
    cap_idx: Dict[str, Any],
    cap_embs: np.ndarray,
    cap_ann: Any,
    sim_threshold: float,
    max_imgs: int = 4,
    topn: int = IMG_TOPN,
) -> List[Tuple[Dict[str, Any], str]]:
    """Choose images by caption similarity; fall back to size-limited picks."""
    picked: List[Tuple[Dict[str, Any], str]] = []
    seen: set[str] = set()
    page_set = set(pages) if pages else set()

    cand_files: set[str] = set()
    for pg in manifest.get("pages", []):
        if page_set and pg.get("page") not in page_set:
            continue
        for im in pg.get("images", []):
            cand_files.add(im.get("file"))

    has_caps = bool(cap_idx.get("texts")) and any((t or "").strip() for t in cap_idx.get("texts", []))
    if q_vec is not None and cap_embs.size > 0 and has_caps and cand_files:
        if _HNSW_AVAILABLE and cap_ann is not None:
            sims, idxs = ann_search(cap_ann, q_vec, topn)
            ranked = [(float(sims[j]), int(idxs[j])) for j in range(len(idxs))]
        else:
            sims = (cap_embs @ q_vec.T).ravel()
            top = np.argsort(-sims)[:topn]
            ranked = [(float(sims[i]), int(i)) for i in top]

        for score, row in ranked:
            if score < sim_threshold:
                continue
            f = cap_idx["files"][row]
            if f not in cand_files or f in seen:
                continue
            for pg in manifest.get("pages", []):
                if page_set and pg.get("page") not in page_set:
                    continue
                for im in pg.get("images", []):
                    if im.get("file") == f:
                        picked.append((im, "caption"))
                        seen.add(f)
                        if len(picked) >= max_imgs:
                            return picked
                        break

    if len(picked) < max_imgs:
        fill = fallback_images_by_size(
            manifest,
            list(pages),
            max_imgs=max_imgs - len(picked),
            min_wh=MIN_IMG_WH,
            min_area=MIN_IMG_AREA,
            take_pages=min(len(pages), FALLBACK_MAX_PAGES) if pages else None,
        )
        for im in fill:
            f = im.get("file")
            if f in seen:
                continue
            picked.append((im, "size"))
            seen.add(f)
            if len(picked) >= max_imgs:
                break

    return picked


def build_result_images(
    q: str,
    chunk: Dict[str, Any],
    manifest: Dict[str, Any],
    cap_idx: Dict[str, Any],
    cap_embs: np.ndarray,
    cap_ann: Any,
    embed_fn,
) -> Tuple[List[Dict[str, Any]], str]:
    """Resolve images associated with a chunk via figure id, caption semantic, or size."""
    fig_refs = collect_fig_refs(chunk.get("text", ""))
    pages = chunk.get("pages", [])
    page_set = set(pages)
    precise: List[Dict[str, Any]] = []
    for pg in manifest.get("pages", []):
        if page_set and pg.get("page") not in page_set:
            continue
        for im in pg.get("images", []):
            fid = (im.get("figure_id") or "").lower()
            if fid and fid in fig_refs:
                precise.append(im)
    if precise:
        return precise[:4], "figure-id"

    qcap = embed_fn([q])
    pairs = filter_images_semantic(
        q_vec=qcap,
        pages=pages,
        manifest=manifest,
        cap_idx=cap_idx,
        cap_embs=cap_embs,
        cap_ann=cap_ann,
        sim_threshold=CAPTION_SIM_THRESHOLD,
        max_imgs=4,
        topn=IMG_TOPN,
    )
    if pairs:
        return [im for im, _why in pairs], ",".join(sorted(set(why for _im, why in pairs)))

    imgs = fallback_images_by_size(
        manifest,
        pages,
        max_imgs=4,
        min_wh=MIN_IMG_WH,
        min_area=MIN_IMG_AREA,
        take_pages=min(len(pages), FALLBACK_MAX_PAGES) if pages else None,
    )
    return imgs, "size"


# -----------------------------------------------------------
# Core search
# -----------------------------------------------------------


def safe_log_query(
    q_logger: JsonlLogger,
    q: str,
    top_k: int,
    timings: Dict[str, float],
    chosen_chunk_ids: List[str],
    chosen_pages: List[List[int]],
    images_selected: int,
    ann: str,
    section_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort telemetry; swallow exceptions so search never fails due to logging."""
    try:
        payload = {
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
        if section_info:
            payload["section_routing"] = section_info
        q_logger.write(payload)
    except Exception:
        pass


def score_and_route_sections(
    q: str,
    sections: List[Dict[str, Any]],
    sec_embs: np.ndarray,
    embed_fn,
    sec_bm25: Optional[BM25Okapi],
    sec_texts_tokens: Optional[List[List[str]]],
) -> Tuple[List[int], Set[str]]:
    """Return top-3 section indices and expanded query terms for downstream use."""
    order, expanded, _ = score_sections(
        q, sections, sec_embs, embed_fn, sec_bm25, sec_texts_tokens
    )
    return order[:3], expanded  # route to top-3


def search_core(
    q: str,
    top_k: int,
    *,
    chunks: List[Dict[str, Any]],
    X,
    bm25: BM25Okapi,
    manifest: Dict[str, Any],
    cap_idx: Dict[str, Any],
    cap_embs: np.ndarray,
    cap_ann: Any,
    ch_embs: np.ndarray,
    ch_ann: Any,
    sections: List[Dict[str, Any]],
    sec_embs: np.ndarray,
    sec_bm25: Optional[BM25Okapi],
    sec_texts_tokens: Optional[List[List[str]]],
    embed_fn,
    reranker: CrossEncoder,
    q_logger: JsonlLogger,
    ann_label: str = "hnswlib",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end search:
      1) Route to likely sections
      2) Collect BM25 + ANN chunk candidates
      3) Apply fusion and TOC/Index kill-switch
      4) Cross-encoder rerank
      5) Pick tops, attach images, log telemetry
    """
    timings: Dict[str, float] = {}
    dbg: Dict[str, Any] = {}

    # ----- Section routing
    chosen_secs, expanded_terms = score_and_route_sections(
        q, sections, sec_embs, embed_fn, sec_bm25, sec_texts_tokens
    )

    # Map chunk id -> index once; build allowed index set from routed sections
    id_to_idx = {c.get("id"): i for i, c in enumerate(chunks)}
    if chosen_secs:
        allowed_chunk_idxs: Set[int] = set()
        for sid in chosen_secs:
            for cid in sections[sid].get("chunk_ids", []):
                idx = id_to_idx.get(cid)
                if idx is not None:
                    allowed_chunk_idxs.add(idx)
    else:
        allowed_chunk_idxs = set(range(len(chunks)))

    # ----- BM25 (index may be 1–2 grams; query must match that choice)
    t = time.perf_counter()
    bm = bm25.get_scores(q_tok12(q, expanded_terms))
    timings["bm25_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # ----- ANN candidates (optional)
    cand_embed: set[int] = set()
    if _HNSW_AVAILABLE and ch_ann is not None and ch_embs.shape[0] > 0:
        t = time.perf_counter()
        qch = embed_fn([q])
        _sims, idxs = ann_search(ch_ann, qch, CAND_TOPN)
        cand_embed = set(map(int, idxs.tolist()))
        timings["chunk_ann_ms"] = round((time.perf_counter() - t) * 1000.0, 3)
    else:
        timings["chunk_ann_ms"] = 0.0

    # Candidate pool within routed sections
    idx_bm25 = np.argsort(-bm)[:CAND_TOPN]
    cand = set(idx_bm25).union(cand_embed)
    cand_arr = np.array([i for i in cand if i in allowed_chunk_idxs], dtype=int)
    if cand_arr.size == 0:
        cand_arr = np.array(sorted(cand), dtype=int)
        if cand_arr.size == 0:
            safe_log_query(q_logger, q, top_k, timings, [], [], 0, ann=ann_label)
            return {"query": q, "results": [], "_debug": dbg} if debug else {"query": q, "results": []}

    # ----- Fusion + gates
    t = time.perf_counter()
    bm_n = minmax(bm[cand_arr])

    # Embedding similarity
    if ch_embs.size > 0:
        qch2 = embed_fn([q])
        emb_raw = (ch_embs[cand_arr] @ qch2.T).ravel().astype("float32")
    else:
        emb_raw = np.zeros_like(bm_n)
    emb_n = minmax(emb_raw)

    # lexical overlap gate (with expanded terms)
    exp_set = set(expanded_terms)
    overlap = np.array(
        [len(exp_set & set(tok1(chunks[int(i)]["text"]))) for i in cand_arr],
        dtype="float32",
    )
    overlap = np.clip(overlap, 0, 3) / 3.0

    base = W_BM25 * bm_n + W_EMB * emb_n + 0.05 * overlap

    # Kill switch for high-penalty candidates (TOC/Index hygiene)
    penalties = np.array(
        [toc_index_penalty(chunks[int(i)]["text"]) for i in cand_arr], dtype="float32"
    )
    kill_mask = penalties >= 0.35
    if kill_mask.any():
        cand_arr = cand_arr[~kill_mask]
        bm_n = bm_n[~kill_mask]
        emb_n = emb_n[~kill_mask]
        overlap = overlap[~kill_mask]
        base = base[~kill_mask]

    # If we killed everything, relax once to avoid empty answers
    if cand_arr.size == 0:
        relax = penalties < 0.50  # keep the least egregious pages
        cand_arr = np.array([i for j, i in enumerate(idx_bm25) if relax[j]])[:max(8, top_k)]
        if cand_arr.size == 0:
            cand_arr = idx_bm25[:max(8, top_k)]
        bm_n = minmax(bm[cand_arr])
        if ch_embs.size > 0:
            qch2 = embed_fn([q])
            emb_raw = (ch_embs[cand_arr] @ qch2.T).ravel().astype("float32")
        else:
            emb_raw = np.zeros_like(bm_n)
        emb_n = minmax(emb_raw)
        overlap = np.array(
            [len(exp_set & set(tok1(chunks[int(i)]["text"]))) for i in cand_arr],
            dtype="float32",
        )
        overlap = np.clip(overlap, 0, 3) / 3.0
        base = W_BM25 * bm_n + W_EMB * emb_n + 0.05 * overlap

    timings["fusion_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # ----- Rerank
    t = time.perf_counter()
    topN = min(RERANK_CAND_N, cand_arr.size)
    if topN > 0:
        order_fused = np.argsort(-base)[:topN]
        cand_top = cand_arr[order_fused]
        texts_top = [chunks[int(i)]["text"] for i in cand_top]
        pairs = [(q, txt) for txt in texts_top]
        ce_scores = np.asarray(reranker.predict(pairs), dtype="float32")
        fused_sel = base[order_fused]
        fused_n = minmax(fused_sel)
        ce_n = minmax(ce_scores)
        mixed = (1.0 - RERANK_WEIGHT) * fused_n + RERANK_WEIGHT * ce_n
        local_rerank = np.argsort(-mixed)
        reranked = cand_top[local_rerank]
        tail = [i for i in cand_arr.tolist() if i not in set(cand_top.tolist())]
        cand_arr = np.concatenate([reranked, np.array(tail, dtype=int)]) if tail else reranked
        base_r = base.copy()
        base_r[order_fused] = mixed[local_rerank]
        base = base_r
    timings["rerank_ms"] = round((time.perf_counter() - t) * 1000.0, 3)

    # ----- Select, build images, sort by page
    order_local = np.argsort(-base)[:top_k]
    chosen = cand_arr[order_local]
    results: List[Dict[str, Any]] = []
    img_total = 0

    t_img = time.perf_counter()
    for local_i, i in enumerate(chosen):
        c = chunks[int(i)]
        imgs, reason = build_result_images(
            q=q,
            chunk=c,
            manifest=manifest,
            cap_idx=cap_idx,
            cap_embs=cap_embs,
            cap_ann=cap_ann,
            embed_fn=embed_fn,
        )
        img_total += len(imgs)
        results.append(
            {
                "chunk_id": c.get("id"),
                "score": float(base[order_local[local_i]]),
                "pages": c.get("pages", []),
                "text": c.get("text", ""),
                "images": imgs,
                "img_reason": reason,
            }
        )
    timings["images_ms"] = round((time.perf_counter() - t_img) * 1000.0, 3)

    def _first_page(pgs: Sequence[int]) -> int:
        return min(pgs) if pgs else 10**9

    results.sort(key=lambda r: (_first_page(r["pages"]), -r["score"]))

    # Section info for logs/debug (includes LDA topic words)
    section_info: Dict[str, Any] = {}
    if sections and chosen_secs:
        section_info = {
            "picked_ids": chosen_secs,
            "picked_titles": [sections[i].get("title", "") for i in chosen_secs],
            "picked_topic_words": [sections[i].get("topic_words", []) for i in chosen_secs],
        }

    safe_log_query(
        q_logger,
        q,
        top_k,
        timings,
        [r["chunk_id"] for r in results],
        [r["pages"] for r in results],
        img_total,
        ann=ann_label,
        section_info=section_info,
    )

    out: Dict[str, Any] = {"query": q, "results": results}
    if debug:
        out["_debug"] = {
            "section_routing": {
                **section_info,
                "expanded_terms": sorted(list(expanded_terms))[:100],
            },
            "chosen": [
                {
                    "chunk_id": r["chunk_id"],
                    "pages": r["pages"],
                    "score": r["score"],
                    "img_reason": r.get("img_reason", ""),
                }
                for r in results
            ],
        }
    return out


# -----------------------------------------------------------
# Runtime wrapper
# -----------------------------------------------------------


def load_runtime(artifacts_dir: str | Path = "./artifacts") -> Dict[str, Any]:
    """
    Load all artifacts and return a dict with 'search_fn', 'chunks', and 'manifest'.
    Keeps TF-IDF artifacts for compatibility, but the search pipeline does not use TF-IDF.
    """
    art = Path(artifacts_dir)

    chunks = json.loads((art / "chunks.json").read_text(encoding="utf-8"))

    # Keep TF-IDF matrix load for backward compatibility (unused in fusion).
    X = pickle.load((art / "tfidf_X.pkl").open("rb"))
    X = sp_normalize(X, norm="l2", axis=1, copy=False)

    bm25: BM25Okapi = pickle.load((art / "bm25.pkl").open("rb"))
    manifest = json.loads((art / "manifest.json").read_text(encoding="utf-8"))

    cap_idx: Dict[str, Any] = {"files": [], "texts": []}
    cap_embs = np.zeros((0, 384), dtype="float32")
    cap_ann = None
    if (art / "caption_index.json").exists():
        cap_idx = json.loads((art / "caption_index.json").read_text(encoding="utf-8"))
        if (art / "caption_embs.npy").exists():
            cap_embs = np.load(art / "caption_embs.npy")
        if _HNSW_AVAILABLE and cap_embs.shape[0] > 0:
            cap_ann = load_or_build_hnsw(cap_embs, art / "hnsw_captions.bin")

    ch_embs = np.zeros((0, 384), dtype="float32")
    ch_ann = None
    if (art / "chunk_embs.npy").exists():
        ch_embs = np.load(art / "chunk_embs.npy")
        if _HNSW_AVAILABLE and ch_embs.shape[0] > 0:
            ch_ann = load_or_build_hnsw(ch_embs, art / "hnsw_chunks.bin")

    sections = (
        json.loads((art / "sections.json").read_text(encoding="utf-8"))
        if (art / "sections.json").exists()
        else []
    )
    sec_embs = (
        np.load(art / "section_embs.npy")
        if (art / "section_embs.npy").exists()
        else np.zeros((0, ch_embs.shape[1] if ch_embs.size else 384), dtype="float32")
    )

    # Section BM25 over concatenated texts (1–2 grams)
    sec_bm25 = None
    sec_texts_tokens = None
    if sections:
        sec_texts = build_section_texts(sections, chunks)
        sec_texts_tokens = [tok12(t) for t in sec_texts]
        sec_bm25 = BM25Okapi(sec_texts_tokens)

    loaded = load_models()
    st_model = loaded.st_model

    def _embed(texts: List[str]) -> np.ndarray:
        """Encode texts to L2-normalized embeddings (float32)."""
        vecs = st_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
        )
        return vecs.astype("float32")

    reranker = CrossEncoder(
        RERANKER_MODEL, device=("cuda" if torch.cuda.is_available() else "cpu")
    )

    q_logger = make_queries_logger(art)

    def search_fn(q: str, top_k: int = 5, debug: Optional[bool] = None) -> Dict[str, Any]:
        """Search API: returns dict with 'results' and optional '_debug' and '_latency_ms'."""
        dbg = DEBUG_RETURN if debug is None else bool(debug)
        t0 = time.perf_counter()
        out = search_core(
            q=q,
            top_k=top_k,
            chunks=chunks,
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
            sec_bm25=sec_bm25,
            sec_texts_tokens=sec_texts_tokens,
            embed_fn=_embed,
            reranker=reranker,
            q_logger=q_logger,
            ann_label="hnswlib",
            debug=dbg,
        )
        out["_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        return out

    return {"search_fn": search_fn, "chunks": chunks, "manifest": manifest}
