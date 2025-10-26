# Multimodal RAG — Local Runner

This repository contains an experimental research & development pipeline for a multimodal Retrieval-Augmented Generation (RAG) system. It supports ingesting PDFs (text, tables, figures), extracting structured layout with a VLM, falling back to pdfplumber for table extraction, enriching content with a structured LLM (Gemini), and creating text or multimodal embeddings (Voyage multmodal models). Knowledge units (KUs) are stored in MongoDB and can be queried with vector + hybrid search.

Key features
- PDF extraction: text, table, figure extraction via VLM with pdfplumber fallback for tables
- Figure analysis: image analysis text is captured and included in multimodal embeddings so the analysis is retrievable by queries
- Embeddings: text and multimodal embeddings with metadata (model name and embedding type) stored per KU
- Gemini structured completions: Pydantic schemas converted to JSON Schema and patched to be accepted by the API
- Resilient ingestion: unordered bulk inserts with duplicate-key handling

Who is this for
- Engineers and researchers who want a locally runnable multimodal RAG research pipeline that integrates VLM layout parsing, structured LLM enrichments, and multimodal embeddings.

Requirements
- Python 3.11
- MongoDB (Atlas or local) — a vector index is required if you want to use MongoDB vector search
- API keys for external services you intend to use (Voyage AI, Gemini, etc.)

Quick setup
1. Create & activate venv (PowerShell):

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your credentials

4. (Atlas) Create a vector search index for the `knowledge_units` collection. Map `embeddings.vector` as a knnVector with the model's output dimension (default 1024 in this repo).

Running

```powershell
python main.py
```

Notes about the current codebase
- Figure embeddings: the pipeline now appends the figure's "analysis" text into the multimodal embedding input (caption + "Detailed Analysis" text). This makes the analysis retrievable by text queries as well as by multimodal queries.
- Table extraction: the pipeline first uses the VLM to extract tables; when that fails or returns empty, a pdfplumber fallback is attempted.
- Gemini schema patching: some Pydantic models produce JSON Schema objects that Gemini rejects (empty object properties). The client now patches such cases by adding additionalProperties or converting maps to typed maps (e.g., metrics_mentioned -> map<string, number>).

Environment variables
See `.env.example` for a full list. At minimum you will need:
- `MONGO_URI` — MongoDB connection string
- `VOYAGE_API_KEY` — Embedding/reranker API key (Voyage)
- `GEMINI_API_KEY` — Gemini key (if used)

Troubleshooting and debugging
- If ingestion raises a BulkWriteError for duplicate keys, the handler now retries unordered bulk insert. Check logs to find skipped or existing KUs.
- If vector search returns no results, ensure you created the Atlas vector index and that the index dimensions match your embedding model.
- If VLM table extraction fails for a PDF, try running ingestion again and watch logs for the pdfplumber fallback path.

Contributing
- This is an R&D codebase; prefer small focused PRs. Add tests for any behavioral change (especially around schema patching and table extraction fallbacks).

License
- See repository LICENSE (if present).

Contact
- For questions, open an issue in the repo or reach out to the maintainers.
