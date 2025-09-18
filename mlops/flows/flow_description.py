"""Flow description utilities independent of Prefect.

This module intentionally contains ONLY static metadata describing the
document processing flow DAG so that tools (e.g. documentation builders
or CI environments) can introspect the pipeline structure without having
Prefect installed or importable.

Keeping this separate avoids hard dependencies on the orchestration
framework for simple topology queries.
"""
from __future__ import annotations


def describe_flow() -> str:
    """Return a human-readable textual description of the flow DAG.

    The flow (named ``document_processing``) consists of five tasks with
    a simple fan-out after extraction. Represented edges show data /
    dependency ordering only.

    DAG:
        t_create_run_dir -> t_extract -> (t_topics, t_indexes, t_entities)

    Returns:
        Multi-line string describing nodes and edges.
    """
    lines = [
        "Flow: document_processing",
        "Nodes (tasks): t_create_run_dir, t_extract, t_topics, t_indexes, t_entities",
        "Edges:",
        "  t_create_run_dir -> t_extract",
        "  t_extract -> t_topics",
        "  t_extract -> t_indexes",
        "  t_extract -> t_entities",
    ]
    return "\n".join(lines)


__all__ = ["describe_flow"]
