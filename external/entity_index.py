# external/entity_index.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
import json, logging

log = logging.getLogger("build.entity_index")

def build_entity_index(
    chunks: List[Dict[str, Any]],
    out_dir: Path,
    batch_size: int = 32,
    nlp=None,                      # <-- pass preloaded spaCy here
) -> float:
    """
    Writes:
      artifacts/entity_index.json
      artifacts/chunk_entities.json
    Returns wall time (seconds) for the NER pass, including nlp.pipe.
    """
    from time import perf_counter
    out_dir.mkdir(parents=True, exist_ok=True)
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_trf", disable=["tagger","lemmatizer","morphologizer","textcat"])
        log.info("entity_index: loaded spaCy internally (no preload)")

    texts = [c["text"] for c in chunks]
    ids   = [c["id"]   for c in chunks]

    t0 = perf_counter()
    entity_index: Dict[str, List[str]] = {}
    chunk_entities: Dict[str, List[str]] = {}

    # transformers pipeline: keep n_process=1 (GPU or CPU)
    for doc, cid in zip(nlp.pipe(texts, batch_size=batch_size), ids):
        ents = []
        seen = set()
        for e in doc.ents:
            key = f"{e.text.strip().lower()}|{e.label_}"
            if key in seen:
                continue
            seen.add(key)
            ents.append(key)
            entity_index.setdefault(key, []).append(cid)
        chunk_entities[cid] = ents

    (out_dir / "entity_index.json").write_text(json.dumps(entity_index, ensure_ascii=False), encoding="utf-8")
    (out_dir / "chunk_entities.json").write_text(json.dumps(chunk_entities, ensure_ascii=False), encoding="utf-8")
    t = perf_counter() - t0
    log.info(f"entity_index: entities={len(entity_index)} unique | {t:.3f}s")
    return t
