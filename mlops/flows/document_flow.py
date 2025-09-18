"""Prefect flow orchestrating document processing.
No business logic inline; wraps pure functions from tasks modules.
"""
from __future__ import annotations
from typing import Dict, Any
import time
import json
from pathlib import Path
from prefect import flow, task

from mlops.tasks.persistence import create_run_directory
from mlops.tasks.pdf_extraction import extract_text_and_images
from mlops.tasks.topic_modeling import compute_topics_overall_and_sections
from mlops.tasks.retrieval_index import build_and_persist_indexes
from mlops.tasks.ner_relations import run_ner_and_relations
from mlops.flows.flow_description import describe_flow  # re-export for compatibility


@task
def t_create_run_dir(pdf_path: str) -> str:
    """Create a new run directory.

    Args:
        pdf_path: Path to the input PDF.

    Returns:
        Path to the created run directory (string form for Prefect serialization).
    """
    run_dir = create_run_directory(pdf_path)
    return str(run_dir)


@task
def t_extract(pdf_path: str, run_dir: str) -> Dict[str, Any]:
    """Extract pages, sections, images.

    Args:
        pdf_path: Path to PDF.
        run_dir: Run directory where artifacts will be stored.
    """
    return extract_text_and_images(pdf_path, run_dir)


@task
def t_topics(run_dir: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Compute and persist overall + per-section LDA topics."""
    return compute_topics_overall_and_sections(run_dir, extracted['sections'], extracted['pages'])


@task
def t_indexes(run_dir: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Build TF-IDF & BM25 retrieval indexes over sections."""
    return build_and_persist_indexes(run_dir, extracted['sections'], extracted['pages'])


@task
def t_entities(run_dir: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Extract heuristic entities and build relationship graph."""
    return run_ner_and_relations(run_dir, extracted['sections'])


@flow(name="document_processing")
def document_processing_flow(pdf_path: str) -> Dict[str, Any]:
    """End-to-end document processing orchestration.

    Task graph (DAG):
        t_create_run_dir -> t_extract -> (t_topics, t_indexes, t_entities)

    Args:
        pdf_path: Path to input PDF file.

    Returns:
        Mapping containing run directory and outputs of downstream tasks.
    """
    timings: Dict[str, float] = {}
    t0_flow = time.perf_counter()

    # Run directory creation
    t0 = time.perf_counter()
    run_dir = t_create_run_dir(pdf_path)
    timings["t_create_run_dir"] = time.perf_counter() - t0

    # Extraction
    t0 = time.perf_counter()
    extracted = t_extract(pdf_path, run_dir)
    timings["t_extract"] = time.perf_counter() - t0

    # Parallel branches (executed sequentially here, but timed individually)
    t0 = time.perf_counter()
    topics = t_topics(run_dir, extracted)
    timings["t_topics"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    indexes = t_indexes(run_dir, extracted)
    timings["t_indexes"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    entities = t_entities(run_dir, extracted)
    timings["t_entities"] = time.perf_counter() - t0

    timings["flow_total"] = time.perf_counter() - t0_flow

    # Persist timings
    try:
        run_path = Path(str(run_dir))
        (run_path / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - non-fatal
        pass

    return {
        "run_dir": run_dir,
        "topics": topics,
        "indexes": indexes,
        "entities": entities,
        "timings": timings,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run document processing Prefect flow")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()
    result = document_processing_flow(args.pdf_path)
    print("Flow finished. Run dir:", result["run_dir"])
    print()
    print(describe_flow())
