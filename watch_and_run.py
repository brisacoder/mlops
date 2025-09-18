"""Watch a folder for new PDF files and trigger the Prefect flow.

Usage:
    python watch_and_run.py                # watches ./artifacts (default)
    python watch_and_run.py --watch incoming_files

The watcher processes PDFs synchronously (one after another). Drop a file
and wait for completion logs before adding many more for best clarity.
"""
from __future__ import annotations
import time
from pathlib import Path
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from mlops.flows.document_flow import document_processing_flow

DEFAULT_DIR = Path("artifacts")
DEFAULT_DIR.mkdir(exist_ok=True)

class PDFCreatedHandler(FileSystemEventHandler):
    """Watchdog handler that triggers the Prefect flow on new PDFs."""

    def on_created(self, event):  # type: ignore[override]
        """Callback executed when a new filesystem entry is created."""
        if event.is_directory:
            return
        src = getattr(event, "src_path", None)
        if not src:
            return
        path = Path(str(src))
        if path.suffix.lower() == ".pdf":
            print(f"[Watcher] New PDF detected: {path}")
            try:
                document_processing_flow(str(path))
            except Exception as exc:  # noqa: BLE001 - intentional broad catch for CLI feedback
                print(f"[Watcher] Error processing {path}: {exc}")


def main():
    """Start the watchdog observer loop.

    Press Ctrl+C to stop. On detection of a new PDF the Prefect flow runs inline
    (synchronous) so rapid bursts of PDFs will process sequentially.
    """
    parser = argparse.ArgumentParser(
        description="Watch a directory for new PDFs and run the document processing flow."
    )
    parser.add_argument(
        "--watch",
        "-w",
        default=str(DEFAULT_DIR),
        help="Directory to watch for new PDF files (default: artifacts)",
    )
    args = parser.parse_args()

    watch_dir = Path(args.watch).expanduser().resolve()
    watch_dir.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    handler = PDFCreatedHandler()
    observer.schedule(handler, str(watch_dir), recursive=False)
    observer.start()
    print(f"Watching for new PDFs in {watch_dir} (Ctrl+C to stop)...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
