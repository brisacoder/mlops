"""Utility to print the document processing flow structure.

Works with or without Prefect installed:
  - If Prefect is available, it attempts to import the flow (no execution).
  - If Prefect is missing, it still prints the static DAG description.

Usage:
    python print_flow.py
"""
from __future__ import annotations

from mlops.flows.flow_description import describe_flow

PREFECT_AVAILABLE = False
FLOW_IMPORTED = False
FLOW_NAME = "document_processing"

try:  # pragma: no cover - optional branch
    import importlib
    importlib.import_module("prefect")  # quick presence check
    try:
        from mlops.flows.document_flow import document_processing_flow  # noqa: F401
        PREFECT_AVAILABLE = True
        FLOW_IMPORTED = True
    except Exception as exc:  # Prefect present but flow import failed
        PREFECT_AVAILABLE = True
        FLOW_IMPORTED = False
        _IMPORT_ERROR = exc  # type: ignore
except Exception:  # Prefect not installed
    PREFECT_AVAILABLE = False
    FLOW_IMPORTED = False
    _IMPORT_ERROR = None  # type: ignore


def main() -> None:
    """Print static DAG description plus diagnostics about Prefect availability."""
    print(describe_flow())
    if not PREFECT_AVAILABLE:
        print("\n[info] Prefect not installed. Displayed static description only.")
    elif PREFECT_AVAILABLE and not FLOW_IMPORTED:
        print("\n[warn] Prefect installed but flow import failed. Static description shown.")
        if '_IMPORT_ERROR' in globals() and _IMPORT_ERROR:
            print(f"       Import error: {_IMPORT_ERROR}")
    else:
        print(f"\n[prefect] Flow '{FLOW_NAME}' import verified (no execution performed).")


if __name__ == "__main__":  # pragma: no cover
    main()
