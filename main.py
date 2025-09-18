def main():
    """Print quick-start command references for the workflow."""
    print("mlops document workflow")
    print("Key commands:")
    print("  python watch_and_run.py                # start watcher")
    print("  python -m mlops.flows.document_flow <pdf>  # manual run")
    print("  python query_doc.py <run_dir> <query>  # query a processed run")


if __name__ == "__main__":
    main()
