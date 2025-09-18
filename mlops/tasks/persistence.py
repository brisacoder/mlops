"""Persistence and run directory management utilities.

Responsibilities:
    * Create a uniquely named run directory under `artifacts/`
    * Persist lightweight JSON metadata
    * Provide helper for saving arbitrary JSON dictionaries
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import uuid
from typing import Any, Dict

ARTIFACTS_ROOT = Path("artifacts")


def create_run_directory(source_pdf: str | Path) -> Path:
    """Create a new run directory and metadata file.

    Directory naming convention: ``run_<UTC_YYYYMMDD_HHMMSS>_<8hex>``.

    Args:
        source_pdf: Path to the original PDF (recorded for provenance only).

    Returns:
        Path to the created run directory.
    """
    ARTIFACTS_ROOT.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{ts}_{uuid.uuid4().hex[:8]}"
    run_dir = ARTIFACTS_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source_pdf": str(source_pdf), "run_id": run_id, "created_utc": ts}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return run_dir


def save_json(path: str | Path, data: Dict[str, Any]):
    """Write a JSON dictionary to disk with UTF-8 encoding.

    Args:
        path: Destination file path (parent dirs must exist).
        data: Mapping to serialize as JSON.
    """
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
