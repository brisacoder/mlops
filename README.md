## Local PDF Processing Workflow (Prefect)

This project provides a simple local-only workflow for processing PDF documents:

1. Drop a PDF into `artifacts/` (or run the flow manually) 
2. A watcher triggers a Prefect flow
3. The flow extracts text, sections, and images
4. Topic modeling (LDA) overall and per-section
5. TF-IDF + BM25 indexes for retrieval
6. Lightweight Named Entity extraction + relationship graph
7. All run artifacts saved under a unique `artifacts/run_*` directory
8. A query CLI can answer questions over the processed document

All business logic lives in pure Python functions under `mlops/tasks/`, with Prefect wrappers in `mlops/flows/`. No databases or cloud storage required.

### Prefect usage

Prefect is only used for orchestration – all core logic stays in pure modules. The flow graph is:

```
t_create_run_dir -> t_extract -> (t_topics, t_indexes, t_entities)
```

Where each downstream branch produces its own JSON artifacts under the run directory. You can:

* Run the flow manually (see below)
* Start the watcher to trigger on file creation
* Print a textual DAG description:

```
python print_flow.py
```

To view more detailed execution metadata you may also run with Prefect's logging level set (e.g. `PREFECT_LOGGING_LEVEL=INFO`).

#### Using the Prefect UI (local server)

You can optionally run a local Prefect server which provides a web UI for flow & task runs.

1. Start server (in a new terminal):
	 ```
	 prefect server start
	 ```
	 UI will usually become available at http://127.0.0.1:4200

2. In the terminal where you run flows, point the client to that API:
	 ```
	 # PowerShell example
	 $Env:PREFECT_API_URL = "http://127.0.0.1:4200/api"
	 ```

     or 
     
     ```
    prefect profile create local-ui
    prefect profile use local-ui
    prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
    ```

3. Run the flow as before:
	 ```
	 python -m mlops.flows.document_flow path/to/file.pdf
	 ```

4. Refresh the UI – the flow run and task runs will appear with logs & timings.

Unset / revert to ephemeral mode by removing the environment variable:
```
Remove-Item Env:PREFECT_API_URL
```

#### Deployments & scheduling (optional)

To schedule or trigger flows from the UI you create a deployment and run a worker:

```
prefect deployment build mlops/flows/document_flow.py:document_processing_flow \
	-n doc-local -q default -o doc-deploy.yaml
prefect deployment apply doc-deploy.yaml
prefect worker start -p default
```

Trigger a deployment run:
```
prefect deployment run document_processing/doc-local --param pdf_path=path/to/file.pdf
```

#### Polling alternative to watchdog

Instead of the OS-level watcher (`watch_and_run.py`), a Prefect-native polling flow is provided: `directory_polling_flow` in `mlops/flows/polling_flow.py`.

Ad‑hoc single scan:
```
python -m mlops.flows.polling_flow --watch incoming_pdfs
```

Iterative polling (5 scans, 10s between):
```
python -m mlops.flows.polling_flow --watch incoming_pdfs --iterations 5 --sleep 10
```

Schedule it (every minute) via deployment:
```
prefect deployment build mlops/flows/polling_flow.py:directory_polling_flow \
	-n poll-incoming -q default --cron "* * * * *" \
	--param watch_dir="incoming_pdfs" -o poll-deploy.yaml
prefect deployment apply poll-deploy.yaml
prefect worker start -p default
```

Drop PDFs into the watched directory; each scheduled run processes only new files (tracked in `artifacts/_processed.json`).

#### Timings / performance artifact

The flow now records per-task and total wall-clock durations in `timings.json` inside each run directory:

```json
{
	"t_create_run_dir": 0.01,
	"t_extract": 0.84,
	"t_topics": 1.55,
	"t_indexes": 0.21,
	"t_entities": 0.04,
	"flow_total": 2.70
}
```

Use this to spot bottlenecks (e.g., large `t_extract` for image-heavy PDFs, large `t_topics` for long vocabularies). If using Prefect UI, durations also appear in the run metadata.

### Install

Python 3.13+ (uses `pyproject.toml` dependencies). Assuming [uv](https://github.com/astral-sh/uv) or pip:

```
pip install -e .
```

### Run the watcher

In one terminal:
```
python watch_and_run.py
```
Then copy or save a PDF into the `artifacts/` directory. A new run directory like `artifacts/run_20250101_123456_ab12cd34/` will appear containing JSON outputs and extracted images.

### Manual flow execution

```
python -m mlops.flows.document_flow path/to/file.pdf
```

### Artifacts produced

Inside each run directory:

* `meta.json` – provenance
* `pages.json` – per-page raw text
* `sections.json` – heuristic sections
* `images.json` – extracted image metadata + PNG files
* `topics.json` – overall + per-section LDA topics
* `tfidf.json` – sparse TF-IDF representation
* `bm25.json` – BM25 index data
* `entities.json` – named entities & co-occurrence graph

### Querying

After a run completes:
```
python query_doc.py artifacts/run_... "What is the main contribution?"
```
You will see top matching sections (BM25) plus frequently occurring entities.

### Project layout

```
mlops/
	tasks/                # Pure functions (no Prefect decorators)
		pdf_extraction.py
		topic_modeling.py
		retrieval_index.py
		ner_relations.py
		persistence.py
	flows/
		document_flow.py    # Prefect flow wrappers
	query/
		query_engine.py     # Query utilities
watch_and_run.py        # Filesystem watcher triggering the flow
query_doc.py            # CLI for querying a run
```

### Extending

Because each processing step is a pure function, you can reuse them independently or enhance them (e.g., replace heuristic NER with a model) without changing the flow wiring.

### Notes / Limitations

* Section splitter is heuristic; consider integrating structured parsers for complex docs.
* LDA quality depends on document length; for very short docs reduce topic counts.
* Simple entity extraction may over-generate; improve with statistical or neural NER later.
* No persistence beyond local JSON; integrate a vector DB only after validating usefulness.

### License

MIT

