# external/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time, logging, os
import torch, spacy


@dataclass
class Loaded:
    nlp: "spacy.language.Language"
    st_model: "object"  # SentenceTransformer instance
    t_spacy_load_s: float
    t_st_load_s: float
    using_gpu: bool


def load_models(logger: Optional[logging.Logger] = None) -> Loaded:
    log = logger or logging.getLogger("runtime")
    # Torch GPU hint
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # --- spaCy transformers (prefer GPU; fall back to CPU if CuPy missing) ---
    use_gpu = False
    t0 = time.perf_counter()
    try:
        import cupy  # noqa

        spacy.require_gpu()
        use_gpu = True
        log.info("spaCy: using GPU (transformers)")
    except Exception as e:
        log.info(f"spaCy: GPU not available ({e}); using CPU")
    nlp = spacy.load(
        "en_core_web_trf", disable=["tagger", "lemmatizer", "morphologizer", "textcat"]
    )
    t_spacy = time.perf_counter() - t0

    # --- SentenceTransformers (GPU if available) ---
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t1 = time.perf_counter()
    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    t_st = time.perf_counter() - t1

    log.info(f"Models loaded | spaCy={t_spacy:.3f}s | ST={t_st:.3f}s | device={device}")
    return Loaded(
        nlp=nlp,
        st_model=st_model,
        t_spacy_load_s=t_spacy,
        t_st_load_s=t_st,
        using_gpu=(device == "cuda"),
    )
