# 🚀 Quick Start Guide

Get your Multimodal RAG system running in **5 minutes**!

## 📋 Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] MongoDB Atlas account created
- [ ] Voyage AI API key obtained
- [ ] Google Gemini API key obtained
- [ ] Alibaba DashScope API key obtained

# QUICKSTART

These are the minimal steps to run this Multimodal RAG project locally on Windows using PowerShell and Python 3.11.

Prerequisites
- Python 3.11
- Git (optional)
- A MongoDB instance (Atlas or local) reachable via connection string
- API keys for services you plan to use (Voyage AI for embeddings, Gemini for synthesis, etc.)

1) Create and activate a virtual environment (PowerShell)

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3) Create your .env file

```powershell
copy .env.example .env
# Edit .env with your keys and connection string (use any text editor)
```

Minimum environment variables
- MONGO_URI — MongoDB connection string
- VOYAGE_API_KEY — API key for embeddings (Voyage)
- GEMINI_API_KEY — API key for Google Gemini (if used)

See `.env.example` for the full list of optional variables and defaults.

4) Create MongoDB Atlas vector index (if using Atlas)

If you plan to use MongoDB Atlas vector search, create an index for the `knowledge_units` collection. In Atlas UI: Clusters → Search → Create Search Index. Use the JSON editor and make an index that maps `embeddings.vector` as a knnVector with the appropriate dimensions (the default embedding dimension used in this repo is 1024).

Example mapping (adjust dimensions to match your embedding model):

```json
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embeddings": {
        "fields": {
          "vector": {
            "dimensions": 1024,
            "similarity": "cosine",
            "type": "knnVector"
          }
        },
        "type": "document"
      }
    }
  }
}
```

Name the index (for example `vector_index`) and wait for it to build.

5) Ingest documents (example)

Place PDFs or other supported sources into the `documents/` folder. Then run:

```powershell
python main.py
```

Outputs
- Knowledge units (KUs) are written to MongoDB (see `src/storage/MongoDBHandler.py`).
- Query results are saved under `output/queries/` when you run the query flow.

Troubleshooting hints
- If ingestion raises duplicate-key BulkWriteError, the pipeline performs unordered bulk inserts and will continue; check logs to see if some KUs were skipped.
- If VLM table extraction fails for a PDF, the pipeline includes a pdfplumber fallback — try ingesting that PDF and check logs for the fallback path.
- For PowerShell activation issues you may need to set your execution policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Further reading
- See `README.md` for a longer description, env var list, and troubleshooting steps.
