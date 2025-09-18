from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional
import json, os, time, socket, platform, datetime as dt
import torch

def _now_iso():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()

class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def write(self, event: Dict[str, Any]) -> None:
        event.setdefault("ts", _now_iso())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

def make_run_logger(artifacts_dir: Path) -> JsonlLogger:
    metrics_dir = artifacts_dir / "metrics"
    fname = f"run-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.jsonl"
    return JsonlLogger(metrics_dir / fname)

def make_queries_logger(artifacts_dir: Path) -> JsonlLogger:
    metrics_dir = artifacts_dir / "metrics"
    fname = f"queries-{dt.datetime.utcnow().strftime('%Y%m%d')}.jsonl"
    return JsonlLogger(metrics_dir / fname)

class StageTimer:
    def __init__(self, logger: JsonlLogger, stage: str, meta: Optional[Dict[str,Any]] = None):
        self.logger = logger
        self.stage = stage
        self.meta = meta or {}
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        t_ms = (time.perf_counter() - self.t0) * 1000.0
        ev = {"type": "stage", "stage": self.stage, "t_ms": round(t_ms, 3)}
        ev.update(self.meta)
        if exc is not None:
            ev["error"] = repr(exc)
        self.logger.write(ev)

def log_env(logger: JsonlLogger, artifacts_dir: Path, pdf_path: Path) -> None:
    info = {
        "type": "env",
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }
    # torch / cuda (optional)
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
            info["torch"] = torch.__version__
    except Exception as e:
        info["cuda_probe_error"] = repr(e)
    logger.write(info)
