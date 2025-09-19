# app/ui.py
from __future__ import annotations
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

from external.search import load_runtime

load_dotenv()
ART = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()
runtime = load_runtime(ART)

# UI-only image size guard (sometimes a small icon slips through)
MIN_IMG_W = int(os.getenv("UI_MIN_IMG_W", "120"))
MIN_IMG_H = int(os.getenv("UI_MIN_IMG_H", "120"))

def _image_ok(img_path: Path) -> bool:
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            return (w >= MIN_IMG_W) and (h >= MIN_IMG_H)
    except Exception:
        return False

def respond(q: str, k: int):
    if not q.strip():
        return "Type a question like **how do I change the air filter?**", []

    res = runtime["search_fn"](q, top_k=k)
    if not res.get("results"):
        return "No results.", []

    # Helpers
    def _resolve_path(basename: str) -> Optional[Path]:
        """Try common extensions if the exact file isn't found."""
        if not basename:
            return None
        p = ART / "images" / basename
        if p.exists():
            return p
        # try alternate extensions
        stem = Path(basename).stem
        for ext in (".jpg", ".png", ".jpeg", ".webp"):
            alt = ART / "images" / f"{stem}{ext}"
            if alt.exists():
                return alt
        return None

    md_parts = []
    gallery_items = []     # list of (path, caption)
    seen_files = set()     # avoid duplicates across chunks

    for rank, r in enumerate(res["results"], start=1):
        text = (r.get("text") or "").strip()
        pages = r.get("pages") or []
        page_str = ", ".join(map(str, pages)) if pages else "?"

        # Render chunk text
        snippet = text[:2000] + ("…" if len(text) > 2000 else "")
        md_parts.append(f"### {rank}. Pages {page_str}\n{snippet}\n")

        # Collect images
        for im in r.get("images", []):
            # Quick metadata-based small-icon filter (faster than opening file)
            w = int(im.get("width", 0) or 0)
            h = int(im.get("height", 0) or 0)
            if (w and h) and (w < 120 or h < 120 or (w * h) < 16000):
                continue

            fname = Path(im.get("file", "")).name
            if not fname:
                continue

            img_path = _resolve_path(fname)
            if not img_path:
                continue
            # UI-level size guard (uses Pillow in your _image_ok)
            if not _image_ok(img_path):
                continue

            key = img_path.as_posix()
            if key in seen_files:
                continue
            seen_files.add(key)

            cap = (im.get("caption") or "").strip()
            if not cap:
                # reasonably informative fallback caption
                cap = f"page {im.get('page', pages[0] if pages else '?')}"

            gallery_items.append((key, cap))

    md = "\n\n".join(md_parts).strip() or "No result text."
    return md, gallery_items

with gr.Blocks(title="Car Manual Q&A") as demo:
    gr.Markdown("## Car Manual Q&A\nAsk questions and get relevant text + images from the manual.")
    with gr.Row():
        with gr.Column(scale=1):
            q = gr.Textbox(label="Ask the manual", placeholder="e.g., how do I change the air filter?", lines=2)
            k = gr.Slider(label="Top-k chunks", minimum=1, maximum=10, value=5, step=1)
            btn = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            answer = gr.Markdown(label="Answer")  # auto-sizing, renders headings nicely
            gallery = gr.Gallery(
                label="Images",
                show_label=True,
                columns=[3],        # grid
                height=700,         # tall enough
                object_fit="contain",
                preview=True,
            )

    btn.click(fn=respond, inputs=[q, k], outputs=[answer, gallery])
    q.submit(fn=respond, inputs=[q, k], outputs=[answer, gallery])  # enter to search

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
