from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

_model = None


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _model
    if _model is None:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _model = SentenceTransformer(model_name, device=device)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedder()
    vecs = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=256
    )
    return vecs.astype("float32")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T
