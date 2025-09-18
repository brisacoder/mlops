manual_qa/
  .env
  data/
    manual.pdf
  artifacts/              # generated
  external/
    pdf_extract.py
    chunking.py
    index_build.py
    semantics.py
    entity_index.py
    search.py
  app/
    api.py
    ui.py
  build.py
  README.md


python -m spacy download en_core_web_trf