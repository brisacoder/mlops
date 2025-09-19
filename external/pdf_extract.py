# external/pdf_extract.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple

import fitz  # PyMuPDF


def _ensure_rgb(pix: "fitz.Pixmap") -> "fitz.Pixmap":
    """
    Ensure the pixmap is in an RGB-compatible colorspace so it can be saved as PNG.
    - If colorspace has >3 components (e.g., CMYK), convert to RGB.
    - If it's already RGB/Gray (with/without alpha), return as is.
    """
    cs = pix.colorspace
    # If colorspace is None (e.g., indexed) or has up to 3 comps, PNG is OK.
    if cs is None:
        return pix
    n = cs.n  # number of components
    if n <= 3:
        return pix
    # n > 3 (e.g., CMYK): convert
    return fitz.Pixmap(fitz.csRGB, pix)


def _save_image(pix: "fitz.Pixmap", out_dir: Path, base: str) -> Tuple[str, int, int]:
    """
    Save a PyMuPDF Pixmap to disk as PNG. Returns (filename, width, height).
    Converts to RGB when needed to avoid 'unsupported colorspace' errors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = _ensure_rgb(pix)
    fname = f"{base}.png"
    path = out_dir / fname
    rgb.save(str(path))  # PNG inferred from extension
    w, h = int(rgb.width), int(rgb.height)
    # Explicitly drop references to free memory in long loops
    if rgb is not pix:
        rgb = None  # noqa: F841
    return fname, w, h


def extract_pdf(pdf_path: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Extract per-page text and images. Saves images to artifacts_dir/images.

    Manifest format:
    {
      "pages": [
        {"page": 1, "text": "...", "images": [
           {"file": "page001_im0.png", "page": 1, "width": 1024, "height": 768,
            "caption": "", "figure_id": ""}, ...
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
            imgs_meta: List[Dict[str, Any]] = []
            # get_images(full=True) returns a list of tuples; index 0 is xref
            for j, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base = f"page{i:03d}_im{j}"
                try:
                    pix = fitz.Pixmap(doc, xref)
                    fname, w, h = _save_image(pix, images_dir, base)
                    imgs_meta.append(
                        {
                            "file": fname,
                            "page": i,
                            "width": w,
                            "height": h,
                            "caption": "",
                            "figure_id": "",
                        }
                    )
                finally:
                    # Make sure pixmap ref is dropped promptly
                    try:
                        pix = None  # noqa: F841
                    except Exception:
                        pass

            pages.append({"page": i, "text": text, "images": imgs_meta})

    return {"pages": pages}
