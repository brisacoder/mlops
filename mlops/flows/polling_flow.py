"""Directory polling Prefect flow.

This provides an alternative to the external watchdog-based watcher. Instead
of reacting to filesystem events, it periodically (or ad‑hoc) scans a directory
for new PDF files and invokes the existing ``document_processing_flow`` on each
unprocessed file.

Intended usage patterns:
  1. Ad-hoc run (one scan):
        python -m mlops.flows.polling_flow --watch incoming_pdfs
  2. Scheduled via Prefect deployment (e.g., every 60s) to emulate a watcher.

State tracking:
  A JSON file (``artifacts/_processed.json`` by default) stores processed file
  absolute paths. You can safely delete it to force re-processing.

Why polling? Polling is simpler to host inside Prefect's orchestration layer
without relying on additional system-level file notification APIs and works
portably (e.g. containers, remote agents) where inotify / FSEvents may not be
available or desirable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set
import json
import time
import argparse
from prefect import flow, task, get_run_logger

from mlops.flows.document_flow import document_processing_flow

DEFAULT_STATE_FILE = Path("artifacts/_processed.json")


def _load_state(state_file: Path) -> Set[str]:
    if state_file.exists():
        try:
            return set(json.loads(state_file.read_text(encoding="utf-8")))
        except Exception:
            # Corrupt state; start fresh
            return set()
    return set()


def _save_state(state_file: Path, processed: Set[str]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(sorted(processed), indent=2), encoding="utf-8")


@task
def list_pdf_files(watch_dir: str) -> List[str]:
    """Return sorted list of absolute PDF file paths in directory."""
    p = Path(watch_dir).expanduser().resolve()
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.pdf") if f.is_file())


@task
def filter_new(files: List[str], processed: List[str]) -> List[str]:
    """Return subset of files not yet processed."""
    processed_set = set(processed)
    return [f for f in files if f not in processed_set]


@task
def process_files(new_files: List[str]) -> List[Dict[str, Any]]:
    """Invoke the document flow for each new file (sequential)."""
    results = []
    logger = get_run_logger()
    for path in new_files:
        logger.info("Processing new PDF: %s", path)
        try:
            res = document_processing_flow(path)  # subflow call
            results.append({"pdf": path, "run_dir": res["run_dir"]})
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed processing %s: %s", path, exc)
    return results


@task
def update_state(processed_state_file: str, prior: List[str], newly_processed: List[str]) -> str:
    """Persist updated processed set and return state file path."""
    state_path = Path(processed_state_file)
    processed_set = set(prior)
    processed_set.update(newly_processed)
    _save_state(state_path, processed_set)
    return str(state_path)


@flow(name="directory_polling")
def directory_polling_flow(
    watch_dir: str = "incoming_pdfs",
    state_file: str = str(DEFAULT_STATE_FILE),
    sleep_seconds: int | None = None,
    iterations: int = 1,
) -> Dict[str, Any]:
    """Poll a directory for new PDFs and process them.

    Args:
        watch_dir: Directory containing candidate PDF files.
        state_file: Path to JSON file with processed absolute paths.
        sleep_seconds: If provided (>0) and iterations > 1, sleep between scans
            to allow new files to arrive (useful for ad-hoc manual loop start).
        iterations: Number of polling cycles (default 1 for single scan). Use a
            deployment schedule instead of large iteration counts for true
            continuous operation.

    Returns:
        Dictionary with summary of processed files and state file path.
    """
    state_path = Path(state_file).expanduser().resolve()
    processed_existing = _load_state(state_path)
    all_newly_processed: Set[str] = set()
    all_results: List[Dict[str, Any]] = []
    logger = get_run_logger()

    for i in range(iterations):
        logger.info("Polling iteration %d of %d", i + 1, iterations)
        files = list_pdf_files(watch_dir)
        new_files = filter_new(files, list(processed_existing.union(all_newly_processed)))
        if not new_files:
            logger.info("No new PDFs found in %s", watch_dir)
        else:
            results = process_files(new_files)
            all_results.extend(results)
            all_newly_processed.update(new_files)
        if sleep_seconds and i < iterations - 1:
            time.sleep(sleep_seconds)

    final_processed = processed_existing.union(all_newly_processed)
    update_state(str(state_path), list(processed_existing), list(all_newly_processed))
    return {
        "processed_total": len(final_processed),
        "processed_this_run": list(all_newly_processed),
        "results": all_results,
        "state_file": str(state_path),
    }


def _cli():  # pragma: no cover - convenience wrapper
    parser = argparse.ArgumentParser(description="Run a single or iterative directory polling scan for PDFs.")
    parser.add_argument(
        "--watch",
        "-w",
        default="incoming_pdfs",
        help="Directory to scan for PDFs (default: incoming_pdfs)",
    )
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE), help="Path to processed state JSON file")
    parser.add_argument("--iterations", type=int, default=1, help="Number of polling loops (default: 1)")
    parser.add_argument(
        "--sleep",
        type=int,
        default=0,
        help="Seconds to sleep between iterations (only if iterations>1)",
    )
    args = parser.parse_args()
    result = directory_polling_flow(
        watch_dir=args.watch,
        state_file=args.state_file,
        iterations=args.iterations,
        sleep_seconds=args.sleep if args.sleep > 0 else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":  # pragma: no cover
    _cli()
