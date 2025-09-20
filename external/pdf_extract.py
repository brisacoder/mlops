# external/pdf_extract.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import fitz  # PyMuPDF


def _nearest_caption_for(rect: "fitz.Rect", blocks: List[tuple]) -> str:
    """
    Choose a short, likely-caption line for an image based on spatial proximity.
    Strategy:
      - horizontally overlapping text blocks
      - within ~160 px vertically (favor blocks just below the image)
    """
    candidates: List[Tuple[float, str]] = []
    for b in blocks:
        bx0, by0, bx1, by1, btxt = b[0], b[1], b[2], b[3], b[4]
        if not btxt or not btxt.strip():
            continue
        # horizontal overlap
        horiz_overlap = min(rect.x1, bx1) - max(rect.x0, bx0)
        if horiz_overlap <= 0:
            continue
        # vertical gap (prefer below)
        vgap = min(abs(by0 - rect.y1), abs(rect.y0 - by1))
        if vgap > 160:
            continue
        penalty = 0.0 if by0 >= rect.y1 else 0.2  # prefer below
        score = vgap + penalty * 50.0
        candidates.append((score, btxt.strip()))

    candidates.sort(key=lambda t: t[0])
    cap = candidates[0][1] if candidates else ""
    return cap.splitlines()[0][:200]


def _to_pngable(pix: "fitz.Pixmap") -> "fitz.Pixmap":
    """
    Ensure the pixmap can be written as PNG:
      - Must be GRAY or RGB colorspace (alpha is OK, but we handle odd cases).
      - Convert anything else (Indexed, CMYK/DeviceN/Separation, etc.) to RGB.
    """
    cs = pix.colorspace
    # If there is no colorspace (stencil / mask), convert to GRAY.
    if cs is None:
        return fitz.Pixmap(fitz.csGRAY, pix)

    # If already GRAY or RGB (possibly with alpha), we keep it.
    if cs == fitz.csGRAY or cs == fitz.csRGB:
        return pix

    # Anything else -> convert to RGB explicitly.
    return fitz.Pixmap(fitz.csRGB, pix)


def _save_pixmap_safely(pix: "fitz.Pixmap", path: Path) -> Tuple[str, int, int]:
    """
    Save Pixmap as PNG if possible, otherwise fall back to JPEG.
    Returns (filename, width, height).
    """
    # 1) Make PNG-friendly copy
    p = _to_pngable(pix)

    # 2) Try PNG
    png_path = path.with_suffix(".png")
    try:
        p.save(str(png_path))  # infers PNG from extension
        return png_path.name, int(p.width), int(p.height)
    except Exception:
        # Some exotic alpha situations still fail. Try stripping alpha.
        try:
            if p.alpha:
                p_noa = fitz.Pixmap(p, 0)  # remove alpha channel
                p_noa.save(str(png_path))
                return png_path.name, int(p_noa.width), int(p_noa.height)
        except Exception:
            pass

    # 3) Fall back to JPEG (lossy but robust for weird inputs)
    jpg_path = path.with_suffix(".jpg")
    try:
        # If still non-RGB, force RGB for JPEG
        if p.colorspace != fitz.csRGB:
            p = fitz.Pixmap(fitz.csRGB, p)
        p.save(str(jpg_path))
        return jpg_path.name, int(p.width), int(p.height)
    finally:
        p = None  # release references


def extract_pdf(pdf_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Extract page text and images from a PDF.
    Saves images to artifacts_dir/images and returns a manifest:

    {
      "pages": [
        {"page": 1, "text": "...", "images": [
           {"file": "page001_im0.png", "page": 1, "width": 1024, "height": 768,
            "caption": "Turn clockwise to lock.", "figure_id": ""},
           ...
        ]},
        ...
      ]
    }
    """
    images_dir = artifacts_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            blocks = page.get_text("blocks") or []  # [(x0,y0,x1,y1,txt,block_no,...)]

            imgs_meta: List[Dict[str, Any]] = []
            for j, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                rect = page.get_image_bbox(img)
                base = f"page{i:03d}_im{j}"
                try:
                    pix = fitz.Pixmap(doc, xref)
                    fname, w, h = _save_pixmap_safely(pix, images_dir / base)
                    caption = _nearest_caption_for(rect, blocks)
                    imgs_meta.append(
                        {
                            "file": fname,
                            "page": i,
                            "width": w,
                            "height": h,
                            "caption": caption,
                            "figure_id": "",  # keep field for future explicit figure IDs
                        }
                    )
                finally:
                    pix = None  # ensure prompt release

            pages.append({"page": i, "text": text, "images": imgs_meta})

    return {"pages": pages}
