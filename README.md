# 🚀 Multimodal RAG System

A production-ready, high-performance Retrieval-Augmented Generation (RAG) system supporting **multimodal content** from PDFs, videos, websites, and images.

## ✨ Key Features

- 📄 **Multi-source Support**: PDF, YouTube, local videos, websites, images
- 🎨 **Multimodal Embeddings**: Separate optimized embeddings for text, figures, and tables
- 🔍 **Advanced Retrieval**: Vector search + Reranking for high-quality results
- 🤖 **Best-in-class Models**:
  - **Qwen VL Max**: Content analysis and enrichment
  - **Voyage-3**: Text embedding
  - **Voyage-multimodal-3**: Figure/table embedding
  - **Voyage Rerank-2.5**: Result reranking
  - **Gemini 2.5 Flash**: Answer synthesis
- ⚡ **High Performance**: Async processing, batch operations, parallel extraction
- 🗄️ **Scalable Storage**: MongoDB with Atlas Vector Search

---

## 📁 Project Structure

```
multimodal-rag/
├── src/
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── Embedder.py              # Text + Multimodal embedders
│   │   └── content_embedder.py       # (deprecated, merged into Embedder.py)
│   │
│   ├── enrichment/
│   │   ├── __init__.py
│   │   ├── enricher.py               # Content enrichment orchestrator
│   │   ├── schema.py                 # Pydantic schemas for enrichment
│   │   └── clients/
│   │       ├── __init__.py
│   │       ├── base_client.py        # Abstract LLM client
│   │       ├── gemini_client.py      # Gemini client (enrichment + synthesis)
│   │       └── qwen_client.py        # Qwen VL Max client
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── pdf.py                    # PDF extraction with LayoutParser
│   │   ├── video.py                  # Video transcription with Whisper
│   │   ├── website.py                # Website content extraction
│   │   └── utils.py                  # Extraction utilities (PDFUtils, VideoUtils, etc.)
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── mongodb_handler.py        # Enhanced MongoDB operations
│   │   └── schema.py                 # Database schemas
│   │
│   ├── query/
│   │   ├── __init__.py
│   │   └── query_engine.py           # Query engine with reranking
│   │
│   ├── __init__.py
│   └── Pipeline.py                   # Main ingestion pipeline
│
├── documents/                         # Input documents folder
│   ├── pdfs/
│   ├── videos/
│   └── images/
│
├── temp/                              # Temporary files (auto-created)
│
├── .env                               # Environment variables
├── .env.example                       # Environment template
├── .gitignore
├── requirements.txt
├── setup.py                           # Package setup
├── main.py                            # Main entry point
└── README.md                          # This file
```

---

## 🛠️ Installation

### 1. Prerequisites

- Python 3.9+
- MongoDB Atlas account (for vector search)
- API Keys:
  - Voyage AI API Key
  - Google Gemini API Key
  - Alibaba DashScope API Key (for Qwen)

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag
```

### 3. Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Setup MongoDB Atlas Vector Search

1. Create a MongoDB Atlas cluster
2. Create a database named `multimodal_rag_db`
3. Create a **Vector Search Index** named `vector_index`:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embeddings.vector",
      "numDimensions": 1024,
      "similarity": "cosine"
    }
  ]
}
```

### 6. Configure Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# MongoDB
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority

# Voyage AI (for embeddings and reranking)
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxx

# Google Gemini (for answer synthesis)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxx

# Alibaba DashScope (for Qwen content analysis)
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxx
```

---

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from main import MultimodalRAGSystem, initialize_system

async def main():
    # Initialize system
    rag_system = await initialize_system()
    
    # Ingest documents
    sources = [
        {'type': 'pdf', 'path': 'documents/paper.pdf'},
        {'type': 'website', 'url': 'https://example.com/article'},
    ]
    await rag_system.ingest_sources(sources)
    
    # Query
    result = await rag_system.query(
        question="What are the main findings?",
        initial_k=20,  # Retrieve 20 candidates
        final_k=5      # Rerank to top 5
    )
    
    # Print results
    rag_system.print_query_result(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### Run Main Script

```bash
python main.py
```

---

## ⚙️ Configuration

### Model Configuration

Edit in `main.py`:

```python
# Text Embedding Model
text_embedder = TextEmbedder(
    api_key=os.getenv("VOYAGE_API_KEY"),
    model_name="voyage-3"  # Options: voyage-3, voyage-3-lite
)

# Multimodal Embedding Model
multimodal_embedder = MultimodalEmbedder(
    api_key=os.getenv("VOYAGE_API_KEY"),
    model_name="voyage-multimodal-3"
)

# Reranker Model
reranker = VoyageReranker(
    api_key=os.getenv("VOYAGE_API_KEY"),
    model_name="rerank-2.5"  # Options: rerank-2.5, rerank-2
)

# Gemini for Synthesis
gemini_client = GeminiClient(
    api_key=os.getenv("GEMINI_API_KEY"),
    model_name="gemini-2.5-pro",  # Options: gemini-2.0-flash-exp, gemini-1.5-pro
    temperature=0.7
)
```

### Pipeline Configuration

```python
pipeline = OptimizedPipeline(
    mongo_handler=mongo_handler,
    llm_client=qwen_client,
    content_embedder=content_embedder,
    max_workers=4,        # Parallel workers
    batch_size=16         # Batch size for enrichment/embedding
)
```

### Query Configuration

```python
result = await rag_system.query(
    question="Your question",
    initial_k=20,    # Initial retrieval candidates
    final_k=5,       # Final results after reranking
)
```

---

## 📊 Usage Examples

### 1. Ingest Multiple PDFs

```python
sources = [
    {'type': 'pdf', 'path': 'documents/paper1.pdf'},
    {'type': 'pdf', 'path': 'documents/paper2.pdf'},
    {'type': 'pdf', 'path': 'documents/paper3.pdf'},
]
results = await rag_system.ingest_sources(sources)
```

### 2. Ingest YouTube Videos

```python
sources = [
    {'type': 'youtube', 'url': 'https://youtube.com/watch?v=xxxxx'},
]
results = await rag_system.ingest_sources(sources)
```

### 3. Ingest Websites

```python
sources = [
    {'type': 'website', 'url': 'https://blog.example.com/article'},
    {'type': 'website', 'url': 'https://docs.example.com/guide'},
]
results = await rag_system.ingest_sources(sources)
```

### 4. Query with Different Parameters

```python
# High precision (more reranking)
result = await rag_system.query(
    question="Explain the transformer architecture",
    initial_k=50,
    final_k=10
)

# Fast response (less candidates)
result = await rag_system.query(
    question="What is the main topic?",
    initial_k=10,
    final_k=3
)
```

---

## 🔧 Advanced Features

### Custom Content Enrichment

```python
from src.enrichment.enricher import ContentEnricher
from src.enrichment.clients.qwen_client import QwenClient

enricher = ContentEnricher(
    client=QwenClient(api_key=os.getenv("DASHSCOPE_API_KEY")),
    max_concurrent=10  # Concurrent API calls
)

items = [
    {"type": "text", "data": "Your text here"},
    {"type": "table", "data": {"col1": [...], "col2": [...]}, "caption": "Table 1"},
]

enriched = await enricher.enrich_batch(items)
```

### Direct Vector Search

```python
from src.storage.mongodb_handler import MongoDBHandler

db = MongoDBHandler(os.getenv("MONGO_URI"))

# Search by vector
results = db.vector_search(
    query_vector=[0.1, 0.2, ...],  # Your embedding
    top_k=10,
    ku_types=["text_chunk", "figure"]  # Filter by type
)

# Hybrid search
results = db.hybrid_search(
    query_vector=[0.1, 0.2, ...],
    text_query="transformer architecture",
    top_k=10,
    vector_weight=0.7  # 70% vector, 30% text
)
```

### Monitor Database Statistics

```python
stats = db.get_statistics()
print(f"Total Sources: {stats['total_sources']}")
print(f"Total KUs: {stats['total_kus']}")
print(f"KUs by Type: {stats['kus_by_type']}")
```

---

## 🧪 Testing

### Run Unit Tests

```bash
pytest tests/
```

### Test Individual Components

```python
# Test PDF extraction
from src.extraction.pdf import PDFExtractor

extractor = PDFExtractor(use_layoutparser=True)
results = extractor.extract_parallel("test.pdf")
print(f"Extracted {len(results)} elements")

# Test embeddings
from src.embedding.Embedder import TextEmbedder

embedder = TextEmbedder(api_key=os.getenv("VOYAGE_API_KEY"))
vector = embedder.embed("Test text")
print(f"Vector dimension: {len(vector)}")
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. MongoDB Vector Search Not Working

**Error**: `Vector search returned 0 results`

**Solution**: Make sure you've created the Atlas Vector Search index:
- Go to MongoDB Atlas → Your Cluster → Search
- Create Search Index with name `vector_index`
- Use the vector configuration provided above

#### 2. LayoutParser Installation Fails

**Error**: `Cannot install detectron2`

**Solution**: LayoutParser is optional. Set `use_layoutparser=False`:
```python
extractor = PDFExtractor(use_layoutparser=False)
```

#### 3. Voyage API Rate Limit

**Error**: `429 Too Many Requests`

**Solution**: Reduce batch sizes:
```python
pipeline = OptimizedPipeline(
    batch_size=8,  # Smaller batches
    max_workers=2
)
```

#### 4. Out of Memory During Extraction

**Solution**: Process files one at a time:
```python
for source in sources:
    await rag_system.ingest_sources([source])
```

---

## 📈 Performance Optimization

### 1. Batch Processing

```python
# Good: Process in batches
pipeline = OptimizedPipeline(batch_size=16)

# Bad: Process one by one
pipeline = OptimizedPipeline(batch_size=1)
```

### 2. Parallel Workers

```python
# For CPU-heavy tasks
pipeline = OptimizedPipeline(max_workers=8)

# For API-heavy tasks
enricher = ContentEnricher(max_concurrent=20)
```

### 3. Query Caching

```python
query_engine = EnhancedQueryEngine(
    use_query_cache=True  # Cache identical queries
)
```

### 4. MongoDB Indexing

Ensure indexes are created:
```python
db.knowledge_units.create_index([("source_id", 1)])
db.knowledge_units.create_index([("ku_type", 1)])
```

---

## Security Best Practices

1. **Never commit `.env` file**
2. **Use environment variables for all secrets**
3. **Restrict MongoDB network access**
4. **Use read-only MongoDB user for queries**
5. **Rotate API keys regularly**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/multimodal-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/multimodal-rag/discussions)
- **Email**: your.email@example.com

---

## 🙏 Acknowledgments

- **Voyage AI** for embeddings and reranking
- **Google Gemini** for answer synthesis, content analysis
- **MongoDB** for vector database
- **PyMuPDF, Camelot** for PDF processing
- **Whisper** for video transcription

---

## 📊 Benchmarks

| Operation | Time (avg) | Notes |
|-----------|------------|-------|
| PDF Ingestion (10 pages) | ~30s | With LayoutParser |
| Video Transcription (5 min) | ~45s | Whisper base model |
| Query (with reranking) | ~2s | Including synthesis |
| Embedding (1000 tokens) | ~0.5s | Voyage-3 |

---

## 🗺️ Roadmap

- [ ] Support for more document formats (DOCX, PPTX)
- [ ] Image-to-image search
- [ ] Multi-language support
- [ ] Real-time streaming answers
- [ ] Web UI dashboard
- [ ] Docker deployment
- [ ] Kubernetes orchestration

---

**Built with ❤️ for the AI community**