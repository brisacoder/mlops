from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
from PIL import Image

# Figure/caption patterns: classic "Fig. 5-2" or Ford-style "E416711"
FIG_RX = re.compile(r"(?:Fig(?:ure)?\.?\s*\d+[A-Za-z\-]*)|(?:^E\d{5,}$)", re.IGNORECASE)


def _overlap_1d(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, max(a1 - a0, b1 - b0))
    return inter / denom


def _blocks_with_bboxes(page: fitz.Page) -> List[Dict[str, Any]]:
    d = page.get_text("dict")
    lines = []
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            text = "".join([s.get("text", "") for s in l.get("spans", [])]).strip()
            if not text:
                continue
            x0 = min(s["bbox"][0] for s in l["spans"])
            y0 = min(s["bbox"][1] for s in l["spans"])
            x1 = max(s["bbox"][2] for s in l["spans"])
            y1 = max(s["bbox"][3] for s in l["spans"])
            size = sum(s.get("size", 0) for s in l["spans"]) / max(1, len(l["spans"]))
            lines.append({"text": text, "bbox": (x0, y0, x1, y1), "size": size})
    return lines


def _nearest_caption_for_image(img_bbox, lines) -> Tuple[str | None, str | None]:
    x0, y0, x1, y1 = img_bbox
    best = None
    best_dy = 1e9
    figure_id = None
    for L in lines:
        tx = L["bbox"]
        dy = min(abs(tx[1] - y1), abs(y0 - tx[3]))  # below or slightly above
        if _overlap_1d(x0, x1, tx[0], tx[2]) < 0.3:
            continue
        t = L["text"].strip()
        if len(t) > 180:
            continue
        has_fig = bool(FIG_RX.search(t))
        looks_caption = has_fig or t.endswith(".") or len(t) <= 80
        if not looks_caption:
            continue
        if dy < best_dy:
            best_dy = dy
            best = t
            m = FIG_RX.search(t)
            figure_id = m.group(0) if m else None
    return best, figure_id


def extract_pdf(pdf_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # thresholds from env (optional)
    ICON_MIN_WH = int(os.getenv("ICON_MIN_WH", "48"))
    ICON_MIN_AREA = int(os.getenv("ICON_MIN_AREA", "16000"))
    SAVE_AS_JPEG = os.getenv("SAVE_AS_JPEG", "true").lower() == "true"
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

    doc = fitz.open(pdf_path)
    pages: List[Dict[str, Any]] = []

    for pno in range(len(doc)):
        page = doc[pno]
        page_text = page.get_text("text") or ""
        lines = _blocks_with_bboxes(page)

        images_meta = []
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            info = doc.extract_image(xref)
            img_bytes = info["image"]
            try:
                img_bbox = page.get_image_bbox(xref)
            except Exception:
                img_bbox = (0, 0, 0, 0)

            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # skip icons/tiny images
            if (
                pil.width < ICON_MIN_WH
                or pil.height < ICON_MIN_WH
                or (pil.width * pil.height) < ICON_MIN_AREA
            ):
                continue

            base = f"p{pno+1}_{i+1}"
            if SAVE_AS_JPEG:
                out_path = img_dir / f"{base}.jpg"
                pil.save(out_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            else:
                out_path = img_dir / f"{base}.png"
                pil.save(out_path, format="PNG")

            caption, fig_id = _nearest_caption_for_image(img_bbox, lines)
            images_meta.append(
                {
                    "file": str(out_path),
                    "page": pno + 1,
                    "width": pil.width,
                    "height": pil.height,
                    "bbox": img_bbox,
                    "caption": caption,
                    "figure_id": fig_id,
                }
            )

        pages.append(
            {
                "page": pno + 1,
                "text": page_text,
                "images": images_meta,
            }
        )

    meta = {
        "pdf_path": str(pdf_path.resolve()),
        "pages": len(doc),
        "images_total": sum(len(p["images"]) for p in pages),
    }
    manifest = {"meta": meta, "pages": pages}

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
