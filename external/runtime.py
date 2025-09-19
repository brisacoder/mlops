# external/runtime.py
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import spacy
import torch
from sentence_transformers import SentenceTransformer

SPACY_GPU_ID = int(os.getenv("SPACY_GPU_ID", "0"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "multi-qa-mpnet-base-dot-v1")


@dataclass
class Loaded:
    """Container for loaded runtime models."""
    nlp: "spacy.language.Language"
    st_model: SentenceTransformer
    t_spacy_load_s: float
    t_st_load_s: float
    using_gpu: bool


def _load_spacy_gpu(gpu_id: int) -> "spacy.language.Language":
    """
    Load spaCy transformers pipeline on a specific CUDA GPU.
    NOTE: Do not try to call .to() on spaCy components; spacy.require_gpu()
    handles device placement for curated-transformers.
    """
    # This will raise if CUDA/CuPy is unavailable (as requested: no guards).
    spacy.require_gpu(gpu_id)

    # Optional: set Torch default device so any Torch ops in the process
    # (e.g., other libs) default to the same GPU.
    torch.set_default_device(f"cuda:{gpu_id}")

    # Load the transformer pipeline; components will run on the GPU selected above.
    nlp = spacy.load(
        "en_core_web_trf",
        disable=["tagger", "lemmatizer", "morphologizer", "textcat"],
    )
    return nlp


def load_models(logger: logging.Logger | None = None) -> Loaded:
    """
    Load spaCy (transformers) on CUDA and a SentenceTransformer on the same device.
    Crashes if dependencies/devices are not present (by design).
    """
    log = logger or logging.getLogger("runtime")
    torch.set_float32_matmul_precision("high")

    t0 = time.perf_counter()
    log.info("spaCy: using GPU %s (transformers)", SPACY_GPU_ID)
    nlp = _load_spacy_gpu(SPACY_GPU_ID)
    t_spacy = time.perf_counter() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t1 = time.perf_counter()
    st_model = SentenceTransformer(EMBED_MODEL, device=device)
    t_st = time.perf_counter() - t1

    log.info(
        "Models loaded | spaCy=%.3fs | ST=%.3fs | device=%s | model=%s",
        t_spacy, t_st, device, EMBED_MODEL,
    )

    return Loaded(
        nlp=nlp,
        st_model=st_model,
        t_spacy_load_s=t_spacy,
        t_st_load_s=t_st,
        using_gpu=(device == "cuda"),
    )
