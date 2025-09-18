# external/runtime.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import time, logging, os
import torch, spacy
import cupy  # noqa

# --- SentenceTransformers (GPU if available) ---
from sentence_transformers import SentenceTransformer

SPACY_GPU_ID = int(os.getenv("SPACY_GPU_ID", "0"))


@dataclass
class Loaded:
    nlp: "spacy.language.Language"
    st_model: "object"  # SentenceTransformer instance
    t_spacy_load_s: float
    t_st_load_s: float
    using_gpu: bool


# ---- spaCy transformers NER (GPU, *explicitly* pinned)
def _load_spacy_gpu(gpu_id: int):
    # 1) Select the exact GPU
    spacy.require_gpu(gpu_id)

    # 2) Make PyTorch default to the same device (prevents CPU tensors being created)
    #    Available in torch >= 2.1
    try:
        torch.set_default_device(f"cuda:{gpu_id}")
    except Exception:
        pass

    # 3) Load the transformer pipeline
    nlp = spacy.load(
        "en_core_web_trf",
        disable=["tagger", "lemmatizer", "morphologizer", "textcat"],
    )

    # 4) Force each relevant component to CUDA explicitly
    #    (curated-transformers exposes .model which is a torch.nn.Module)
    if "transformer" in nlp.pipe_names:
        trf = nlp.get_pipe("transformer")
        try:
            trf.model.to(f"cuda:{gpu_id}")
        except Exception:
            pass

    if "ner" in nlp.pipe_names:
        ner = nlp.get_pipe("ner")
        try:
            ner.model.to(f"cuda:{gpu_id}")
        except Exception:
            pass

    return nlp


def load_models(logger: Optional[logging.Logger] = None) -> Loaded:
    log = logger or logging.getLogger("runtime")
    # Torch GPU hint
    torch.set_float32_matmul_precision("high")

    # --- spaCy transformers (prefer GPU; fall back to CPU if CuPy missing) ---
    t0 = time.perf_counter()
    log.info("spaCy: using GPU (transformers)")
    nlp = _load_spacy_gpu(SPACY_GPU_ID)
    t_spacy = time.perf_counter() - t0

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
