from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from external.search import load_runtime

load_dotenv()
ART = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()
runtime = load_runtime(ART)


def respond(q):
    res = runtime["search_fn"](q, top_k=5)
    if not res["results"]:
        return "No results.", []
    answers = []
    imgs = []
    for r in res["results"]:
        answers.append(f"Pages {r['pages']}: {r['text'][:600]}…")
        for im in r["images"]:
            img_path = (ART / "images" / Path(im.get("file", "")).name).as_posix()
            imgs.append(img_path)
    return "\n\n---\n\n".join(answers), imgs



demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Ask the manual"),
    outputs=[gr.Textbox(label="Answer"), gr.Gallery(label="Images")],
    title="Car Manual Q&A",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)