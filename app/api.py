from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles

from external.search import load_runtime

load_dotenv()
ART = Path(os.getenv("ARTIFACTS_DIR", "./artifacts")).resolve()
IMG_ROOT = ART / "images"

state = {"runtime": None}


@asynccontextmanager
async def lifespan(app):
    state["runtime"] = load_runtime(ART)  # warm everything
    yield
    state["runtime"] = None


app = FastAPI(title="Manual Q&A", lifespan=lifespan)
app.mount("/images", StaticFiles(directory=str(IMG_ROOT)), name="images")


@app.get("/ask")
def ask(q: str = Query(..., description="User question"), k: int = 5):
    runtime = state["runtime"]
    res = runtime["search_fn"](q, top_k=k)
    # replace file paths with URLs
    for hit in res["results"]:
        for im in hit["images"]:
            p = Path(im["file"])
            im["url"] = f"/images/{p.name}"
            im.pop("file", None)
    return res
