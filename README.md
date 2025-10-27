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

Quick setup: See detail in QUICKSTART.md

Configuration: 
![Configuration Overview](images/config.png)


Notes about the current codebase
- Figure embeddings: the pipeline now appends the figure's "analysis" text into the multimodal embedding input (caption + "Detailed Analysis" text). This makes the analysis retrievable by text queries as well as by multimodal queries.
- Table extraction: the pipeline first uses the VLM to extract tables; when that fails or returns empty, a pdfplumber fallback is attempted.
- Gemini schema patching: some Pydantic models produce JSON Schema objects that Gemini rejects (empty object properties). The client now patches such cases by adding additionalProperties or converting maps to typed maps (e.g., metrics_mentioned -> map<string, number>).
- VLMs aren’t very good at extracting complex tables. For harder cases, should combine them with OCR to improve text and layout accuracy.
- Thầy thông cảm vì nhiều thành viên lạ tên trên github ạ, do em dùng máy khác nên v.


Future Improvements
- Enhance layout and parser extraction
- Define stricter schemas and validation
- Refactor and clean up codebase