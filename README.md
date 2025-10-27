# Multimodal RAG — Local Runner

This repository contains an experimental research and development pipeline for a multimodal Retrieval-Augmented Generation (RAG) system.
It supports ingesting PDFs (text, tables, figures), extracting structured layout with a VLM, falling back to pdfplumber for table extraction, enriching content with a structured LLM (Gemini), and creating text or multimodal embeddings (Voyage multimodal models).
Knowledge Units (KUs) are stored in MongoDB and can be queried using vector and hybrid search.


## Key Features

* **PDF Extraction:** Extract text, tables, and figures via VLM with pdfplumber fallback for tables.
* **Figure Analysis:** Extract image analysis text and include it in multimodal embeddings for better retrieval.
* **Embeddings:** Store text and multimodal embeddings with metadata (model name and embedding type) per KU.
* **Structured Completions:** Use Gemini with Pydantic schemas converted to JSON Schema for structured enrichment.
* **Resilient Ingestion:** Perform unordered bulk inserts with duplicate-key handling.


## Who Is This For

Researchers and engineers who want a **locally runnable multimodal RAG pipeline** that integrates:

* VLM layout parsing
* Structured LLM enrichments
* Multimodal embeddings and retrieval


## Requirements

* Python 3.11
* MongoDB (Atlas or local) — a vector index is required for vector search
* API keys for external services (Voyage AI, Gemini, etc.)

Quick setup: See details in **QUICKSTART.md**


## Configuration

**Configuration Overview:**
![Configuration Overview](images/config.png)


## PDF Processing Pipeline

* **Text:** extract → chunk
* **Figure:** parse → extract → analyze (enrich) → multimodal embedding
* **Table:** parse → extract

![PDF Process](images/pdf-process.png)


## Website Processing

Extract → text chunk
![Website Process](images/link-process.png)


## YouTube / Video Processing

Extract through API transcript →
Fallback: download video → speech-to-text model (STT) → transcript → text chunk

![YouTube Process](images/youtube-process.png)


## MongoDB Storage

**Source Collection:**
![Source Collection](images/source.png)

**Knowledge Unit Collection:**
![Knowledge Units](images/kus.png)


## Query Results

![Query Results](images/query-results.png)
![Retrieval Results](images/retrieval-results.png)


## Notes About the Current Codebase

* The pipeline first uses the VLM to extract tables; if that fails, pdfplumber is used as fallback.
* VLMs are not yet good at extracting complex tables; for difficult cases, combine with OCR to improve text and layout accuracy.
* Some GitHub contributor names may appear unusual due to work from different machines.



## Future Improvements

* Improve layout and parser extraction
* Define stricter schemas and validation
* Refactor and clean up codebase
* Fix automatic YouTube transcript extraction (faster than Whisper)
* Enhance visual-language query capabilities