from __future__ import annotations
from typing import List, Dict, Any
import re


def split_into_chunks(
    pages: List[Dict[str, Any]], min_chars=400, max_chars=1200
) -> List[Dict[str, Any]]:
    heading_rx = re.compile(r"^\s*(\d+(\.\d+)*)\s+[A-Z][A-Za-z].{0,80}$")
    chunks: List[Dict[str, Any]] = []
    buf, buf_pages = [], set()
    cid = 0

    def flush():
        nonlocal cid, buf, buf_pages, chunks
        text = "\n".join(buf).strip()
        if not text:
            buf.clear()
            buf_pages.clear()
            return
        if len(text) > max_chars:
            parts = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
            for part in parts:
                cid += 1
                chunks.append(
                    {"id": f"C{cid}", "text": part.strip(), "pages": sorted(buf_pages)}
                )
        else:
            cid += 1
            chunks.append({"id": f"C{cid}", "text": text, "pages": sorted(buf_pages)})
        buf.clear()
        buf_pages.clear()

    for p in pages:
        lines = (p["text"] or "").splitlines()
        for ln in lines:
            if heading_rx.match(ln) and len("\n".join(buf)) >= min_chars:
                flush()
            buf.append(ln)
            buf_pages.add(p["page"])
            if len("\n".join(buf)) >= max_chars:
                flush()
        if len("\n".join(buf)) >= (max_chars * 1.5):
            flush()
    flush()
    return chunks
