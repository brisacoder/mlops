# external/chunking.py
from __future__ import annotations

from typing import Dict, Any, List


def split_into_chunks(
    pages: List[Dict[str, Any]],
    min_chars: int = 400,
    max_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Greedy, page-aware chunker. Keeps page bounds but merges lines until max_chars.
    Each chunk: {"id": "cXXXX", "pages": [p,...], "text": "..."}.
    """
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_pages: List[int] = []

    def flush() -> None:
        if not buf:
            return
        cid = f"c{len(chunks):04d}"
        chunks.append(
            {
                "id": cid,
                "pages": sorted(set(buf_pages)),
                "text": " ".join(buf).strip(),
            }
        )
        buf.clear()
        buf_pages.clear()

    for pg in pages:
        text = (pg.get("text") or "").strip()
        if not text:
            continue
        parts = [s.strip() for s in text.splitlines() if s.strip()]
        for line in parts:
            candidate = (" ".join(buf + [line])).strip()
            if len(candidate) > max_chars and len(" ".join(buf)) >= min_chars:
                flush()
                candidate = line
            buf.append(line)
            buf_pages.append(pg["page"])
        # if page ended and we have a large chunk, consider flushing
        if len(" ".join(buf)) >= max_chars:
            flush()
    # tail
    if buf:
        flush()
    return chunks
