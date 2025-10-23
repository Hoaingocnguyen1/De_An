# 🚀 Quick Start Guide

Get your Multimodal RAG system running in **5 minutes**!

## 📋 Prerequisites Checklist

- [ ] Python 3.9+ installed
- [ ] MongoDB Atlas account created
- [ ] Voyage AI API key obtained
- [ ] Google Gemini API key obtained
- [ ] Alibaba DashScope API key obtained

---

## ⚡ Installation Steps

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-rag.git
cd multimodal-rag

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# OR Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Get API Keys

#### Voyage AI (Embeddings + Reranking)
1. Visit https://www.voyageai.com/
2. Sign up and navigate to API Keys
3. Create new key → Copy it

#### Google Gemini (Answer Synthesis)
1. Visit https://makersuite.google.com/app/apikey
2. Create API key → Copy it

#### Alibaba DashScope (Qwen Content Analysis)
1. Visit https://dashscope.console.aliyun.com/
2. Register and create API key → Copy it

### 3. Setup MongoDB Atlas

```bash
# 1. Create free cluster at https://www.mongodb.com/cloud/atlas/register

# 2. Create database user:
#    - Database Access → Add New User
#    - Username: rag_user
#    - Password: (generate secure password)

# 3. Whitelist IP:
#    - Network Access → Add IP Address
#    - Allow access from anywhere: 0.0.0.0/0

# 4. Get connection string:
#    - Click "Connect" → "Connect your application"
#    - Copy connection string
#    - Replace <password> with your password
```

### 4. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env and add your keys:
nano .env  # or use your preferred editor
```

**Minimum required configuration:**

```env
MONGO_URI=mongodb+srv://rag_user:YOUR_PASSWORD@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
VOYAGE_API_KEY=pa-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 5. Create Vector Search Index

**IMPORTANT**: MongoDB requires manual index creation for vector search.

1. Go to MongoDB Atlas → Your Cluster
2. Click **"Search"** tab
3. Click **"Create Search Index"**
4. Choose **"JSON Editor"**
5. Select database: `multimodal_rag_db`, collection: `knowledge_units`
6. Paste this configuration:

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

7. Name it: `vector_index`
8. Click **"Create Search Index"**
9. Wait 1-2 minutes for index to be ready

---

## 🎯 First Run

### Prepare Test Documents

```bash
# Create documents folder
mkdir -p documents/pdfs

# Add your PDF files
# For testing, download a sample paper:
wget https://arxiv.org/pdf/2010.11929.pdf -O documents/pdfs/vision_transformer.pdf
```

### Run the System

```bash
python main.py
```

**Expected output:**

```
2024-01-15 10:30:00 - INFO - ✓ MongoDB connection successful
2024-01-15 10:30:01 - INFO - GeminiClient initialized with model: gemini-2.0-flash-exp
2024-01-15 10:30:01 - INFO - TextEmbedder initialized with Voyage AI model: voyage-3
2024-01-15 10:30:01 - INFO - ✓ Optimized pipeline initialized
2024-01-15 10:30:01 - INFO - ✓ Enhanced Query Engine ready with reranking
2024-01-15 10:30:01 - INFO - ✓ Multimodal RAG System initialized

================================================================================
STEP 1: INGESTING DOCUMENTS
================================================================================
2024-01-15 10:30:02 - INFO - ⚡ Processing pdf: documents/pdfs/vision_transformer.pdf
2024-01-15 10:30:05 - INFO -   [1/4] Extracting content in parallel...
2024-01-15 10:30:15 - INFO -   ✓ Extracted 45 raw objects
2024-01-15 10:30:15 - INFO -   [2/4] Enriching content in batches...
2024-01-15 10:30:25 - INFO -   ✓ Enriched 45 objects
2024-01-15 10:30:25 - INFO -   [3/4] Generating embeddings in batches...
2024-01-15 10:30:35 - INFO -   ✓ Generated embeddings for 52 knowledge units
2024-01-15 10:30:35 - INFO -   [4/4] Finalizing Knowledge Units...
2024-01-15 10:30:36 - INFO - ✓ Completed documents/pdfs/vision_transformer.pdf in 34.2s - 52 KUs created.

================================================================================
STEP 2: QUERYING THE SYSTEM
================================================================================
2024-01-15 10:30:40 - INFO -   [1/4] Embedding query with voyage-3
2024-01-15 10:30:41 - INFO -   [2/4] Retrieving top-20 candidates...
2024-01-15 10:30:42 - INFO -   [3/4] Reranking with rerank-2.5...
2024-01-15 10:30:43 - INFO -   ✓ Reranked to top-5 results
2024-01-15 10:30:43 - INFO -   [4/4] Synthesizing answer with GeminiClient
2024-01-15 10:30:45 - INFO - ✓ Query completed successfully

================================================================================
ANSWER:
================================================================================
The Vision Transformer (ViT) introduces several key architectural innovations...
```

---

## 🧪 Test Individual Components

### Test MongoDB Connection

```python
from src.storage.mongodb_handler import MongoDBHandler
import os
from dotenv import load_dotenv

load_dotenv()
db = MongoDBHandler(os.getenv("MONGO_URI"))

if db.test_connection():
    print("✓ MongoDB connected!")
    stats = db.get_statistics()
    print(f"Total KUs: {stats['total_kus']}")
else:
    print("✗ Connection failed!")
```

### Test Embeddings

```python
from src.embedding.Embedder import TextEmbedder
import os

embedder = TextEmbedder(api_key=os.getenv("VOYAGE_API_KEY"))
vector = embedder.embed("This is a test")
print(f"✓ Vector dimension: {len(vector)}")
print(f"First 5 values: {vector[:5]}")
```

### Test PDF Extraction

```python
from src.extraction.pdf import PDFExtractor

extractor = PDFExtractor()
results = extractor.extract_parallel("documents/pdfs/test.pdf")
print(f"✓ Extracted {len(results)} elements")

# Count by type
from collections import Counter
types = Counter(r['type'] for r in results)
print(f"Types: {dict(types)}")
```

---

## 🎓 Next Steps

### 1. Customize Configuration

Edit `main.py` to adjust:

```python
# Processing parameters
pipeline = OptimizedPipeline(
    max_workers=8,      # Increase for more parallel processing
    batch_size=32       # Increase for faster embedding
)

# Query parameters
result = await rag_system.query(
    question="Your question",
    initial_k=50,       # Retrieve more candidates
    final_k=10          # Keep more final results
)
```

### 2. Add More Documents

```python
SOURCES_TO_INGEST = [
    {'type': 'pdf', 'path': 'documents/paper1.pdf'},
    {'type': 'pdf', 'path': 'documents/paper2.pdf'},
    {'type': 'website', 'url': 'https://arxiv.org/abs/2010.11929'},
    {'type': 'youtube', 'url': 'https://youtube.com/watch?v=xxxxx'},
]
```

### 3. Explore Advanced Features

```python
# Get database statistics
stats = db.get_statistics()
print(stats)

# List all sources
sources = db.list_sources(status="completed")

# Hybrid search
results = db.hybrid_search(
    query_vector=vector,
    text_query="transformer architecture",
    vector_weight=0.7
)
```

---

## ❓ Troubleshooting

### Issue: "Vector search returned 0 results"

**Solution**: Check if vector index is created in MongoDB Atlas.

```python
# Test vector search
from src.storage.mongodb_handler import MongoDBHandler
db = MongoDBHandler(os.getenv("MONGO_URI"))

test_vector = [0.1] * 1024  # Dummy vector
results = db.vector_search(test_vector, top_k=5)

if not results:
    print("❌ Vector index not configured correctly!")
    print("Please create the index in MongoDB Atlas (see step 5)")
else:
    print(f"✓ Vector search works! Found {len(results)} results")
```

### Issue: "Failed to get embeddings from Voyage AI"

**Solutions**:
1. Check API key is correct
2. Check internet connection
3. Verify API quota: https://www.voyageai.com/dashboard

### Issue: "Qwen API Error"

**Solution**: Verify DashScope API key and region:
```python
import dashscope
dashscope.api_key = "your-key"

# Test connection
response = dashscope.Generation.call(
    model="qwen-turbo",
    prompt="Hello"
)
print(response)
```

### Issue: Out of memory

**Solution**: Reduce batch size and workers:
```python
pipeline = OptimizedPipeline(
    max_workers=2,
    batch_size=4
)
```

---

## 📞 Need Help?

- **Documentation**: See [README.md](README.md)
- **Issues**: https://github.com/yourusername/multimodal-rag/issues
- **Discord**: [Join our community](#)

---

**Ready to build amazing RAG applications! 🎉**