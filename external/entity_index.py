# external/entity_index.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import spacy


def build_entity_index(
    chunks: List[Dict[str, Any]],
    artifacts_dir: Path,
    batch_size: int = 32,
    nlp: "spacy.language.Language" | None = None,
) -> float:
    """
    Run NER over chunks (spaCy transformers) and persist an entity → chunk_ids map.
    Key format: "{text}|{label}". Saves entity_index.json.
    Returns: elapsed seconds (float) for logging.
    """
    import time

    t0 = time.perf_counter()
    nlp_local = nlp or spacy.load("en_core_web_trf")

    ent_map: Dict[str, List[str]] = {}
    texts = [c.get("text", "") for c in chunks]
    ids = [c.get("id") for c in chunks]

    for i in range(0, len(texts), batch_size):
        docs = list(nlp_local.pipe(texts[i : i + batch_size], batch_size=batch_size))
        for doc, cid in zip(docs, ids[i : i + batch_size]):
            for ent in doc.ents:
                key = f"{ent.text.strip().lower()}|{ent.label_}"
                ent_map.setdefault(key, []).append(cid)

    (artifacts_dir / "entity_index.json").write_text(json.dumps(ent_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return time.perf_counter() - t0
