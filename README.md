# 📚 Thesys Embedder Space — "The Librarian"

> A standalone, high-performance semantic search and embedding service with code-aware chunking and caching.

---

## Overview

The **Embedder Space** is the knowledge retrieval backbone of the [Thesys](https://github.com/Transcendental-Programmer/Thesys) multi-agent system. It can operate **independently** as a semantic search API for any codebase, or as part of the full pipeline.

**What it does**: Embeds code and text into dense vectors, performs semantic search over indexed repositories, and provides intelligent code-aware chunking that respects function/class boundaries.

**Model**: `all-MiniLM-L6-v2` (22M parameters, FP16, ~90MB RAM). Blazing fast on CPU — processes 1000+ embeddings/second.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                       EMBEDDER SPACE                              │
│                                                                   │
│  ┌──────────┐   ┌──────────────────┐   ┌───────────────────────┐ │
│  │ FastAPI   │──▶│ Smart Chunker    │──▶│ Embedding Engine      │ │
│  │ Endpoints │   │ (code-boundary   │   │ (sentence-transformers│ │
│  └──────────┘   │  aware)          │   │  + batch processing)  │ │
│                  └──────────────────┘   └──────────┬────────────┘ │
│                                                    │              │
│                                          ┌─────────▼──────────┐   │
│                                          │ FAISS Index         │   │
│                                          │ (in-memory /        │   │
│                                          │  persistent)        │   │
│                                          └─────────┬──────────┘   │
│                                                    │              │
│                                          ┌─────────▼──────────┐   │
│                                          │ LRU Cache Layer     │   │
│                                          │ (avoid re-embedding │   │
│                                          │  unchanged code)    │   │
│                                          └────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

### Internal Pipeline

1. **Smart Chunking** (for `/index` and `/chunk` endpoints):
   - Parses source files using regex-based boundary detection (class/function/method definitions).
   - Each chunk is a logical unit: one function, one class, one config block.
   - Chunks include a **header** (file path + parent class/function name) for context.
   - Max chunk size: 512 tokens. Chunks > 512 tokens are split at line boundaries with 50-token overlap.

2. **Batch Embedding**:
   - Uses `sentence-transformers` with batch processing for throughput.
   - On 2 vCPU: ~200 embeddings/second for code chunks.

3. **FAISS Indexing**:
   - Uses `IndexFlatIP` (inner product) for exact search — fast enough for <100K vectors.
   - For larger repos, automatically switches to `IndexIVFFlat` (approximate, ~10x faster).

4. **Cache Layer**:
   - LRU cache keyed by `hash(text)` → embedding vector.
   - Cache hit = <0.1ms response (no model inference).
   - Cache is warmed on startup if a persistent index exists.

---

## API Endpoints

### `POST /embed`
Get embedding vectors for text/code snippets.

**Request:**
```json
{
  "texts": [
    "def calculate_tax(income, rate):\n    return income * rate",
    "class UserAuthentication:\n    def __init__(self, db):\n        self.db = db"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.023, -0.156, 0.089, ...],
    [0.045, -0.201, 0.112, ...]
  ],
  "dimensions": 384,
  "metadata": {
    "model": "all-MiniLM-L6-v2",
    "cached_count": 0,
    "latency_ms": 12
  }
}
```

### `POST /chunk`
Intelligently chunk a source file into semantic units.

**Request:**
```json
{
  "file_path": "auth/middleware.py",
  "content": "import jwt\nfrom functools import wraps\n\nclass AuthMiddleware:\n    def __init__(self, secret):\n        self.secret = secret\n\n    def require_auth(self, f):\n        @wraps(f)\n        def decorated(*args, **kwargs):\n            token = request.headers.get('Authorization')\n            ...\n            return f(*args, **kwargs)\n        return decorated\n\n    def generate_token(self, user_id):\n        return jwt.encode({'user_id': user_id}, self.secret)\n",
  "language": "python"
}
```

**Response:**
```json
{
  "chunks": [
    {
      "id": "auth/middleware.py::imports",
      "content": "import jwt\nfrom functools import wraps",
      "type": "imports",
      "start_line": 1,
      "end_line": 2
    },
    {
      "id": "auth/middleware.py::AuthMiddleware.__init__",
      "content": "class AuthMiddleware:\n    def __init__(self, secret):\n        self.secret = secret",
      "type": "method",
      "start_line": 4,
      "end_line": 6,
      "parent": "AuthMiddleware"
    },
    {
      "id": "auth/middleware.py::AuthMiddleware.require_auth",
      "content": "    def require_auth(self, f):\n        @wraps(f)\n        def decorated(*args, **kwargs):\n            token = request.headers.get('Authorization')\n            ...\n            return f(*args, **kwargs)\n        return decorated",
      "type": "method",
      "start_line": 8,
      "end_line": 14,
      "parent": "AuthMiddleware"
    }
  ],
  "total_chunks": 3,
  "metadata": {
    "language": "python",
    "chunking_strategy": "code_boundary"
  }
}
```

### `POST /index`
Index an entire repository for semantic search.

**Request:**
```json
{
  "repo_path": "/workspace/my-project",
  "include_patterns": ["*.py", "*.js", "*.ts"],
  "exclude_patterns": ["**/node_modules/**", "**/__pycache__/**"]
}
```

**Response:**
```json
{
  "status": "indexed",
  "files_processed": 47,
  "chunks_created": 312,
  "index_size_mb": 0.46,
  "latency_ms": 8500
}
```

### `POST /search`
Semantic search over an indexed repository.

**Request:**
```json
{
  "query": "function that validates JWT tokens",
  "top_k": 5,
  "min_score": 0.3
}
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "auth/middleware.py::AuthMiddleware.require_auth",
      "file_path": "auth/middleware.py",
      "content": "def require_auth(self, f):\n    ...",
      "score": 0.87,
      "start_line": 8,
      "end_line": 14
    },
    {
      "chunk_id": "utils/token.py::verify_token",
      "file_path": "utils/token.py",
      "content": "def verify_token(token, secret):\n    ...",
      "score": 0.72,
      "start_line": 15,
      "end_line": 25
    }
  ],
  "metadata": {
    "query_embedding_ms": 3,
    "search_ms": 1,
    "total_indexed_chunks": 312
  }
}
```

### `GET /health`
Returns index status, cache stats, memory usage.

---

## Smart Chunking Strategy

| Source Element | Chunking Rule |
|---------------|--------------|
| **Imports** | Grouped into single chunk per file |
| **Functions** | One chunk per function (including decorators + docstring) |
| **Classes** | One chunk per method; class-level attributes in separate chunk |
| **Config files** | One chunk per top-level key (YAML/JSON) |
| **Markdown** | One chunk per heading section |
| **Long functions** (>512 tokens) | Split at logical line breaks with 50-token overlap |

---

## Deployment

### Requirements
- Python 3.10+
- ~500MB RAM (model + small index)
- 2 vCPU minimum

### Quick Start (Local)
```bash
cd spaces/embedder
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7863
# Model auto-downloads on first request
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `FAISS_INDEX_PATH` | `./index/faiss.idx` | Persistent index path |
| `CACHE_SIZE` | `10000` | LRU cache max entries |
| `MAX_CHUNK_TOKENS` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap tokens for split chunks |
| `BATCH_SIZE` | `64` | Embedding batch size |

---

## Project Structure

```
spaces/embedder/
├── app.py                 # FastAPI application + endpoints
├── requirements.txt       # Dependencies
├── Dockerfile             # HF Spaces deployment
│
├── core/
│   ├── embedder.py        # Sentence-transformer wrapper + batching
│   ├── chunker.py         # Code-aware smart chunking logic
│   ├── indexer.py         # FAISS index management
│   ├── cache.py           # LRU cache layer
│   └── schemas.py         # Pydantic models for request/response
│
└── tests/
    ├── test_chunker.py    # Verify code boundary detection
    ├── test_embedder.py   # Verify embedding dimensions
    ├── test_indexer.py    # Verify search accuracy
    └── test_app.py        # API endpoint tests
```

---

## Standalone Usage Examples

### Index and Search a Project
```python
import httpx

# Index your repository
httpx.post("http://localhost:7863/index", json={
    "repo_path": "/path/to/my-project",
    "include_patterns": ["*.py"]
})

# Search for relevant code
results = httpx.post("http://localhost:7863/search", json={
    "query": "database connection pooling",
    "top_k": 3
}).json()

for r in results["results"]:
    print(f"[{r['score']:.2f}] {r['file_path']} (L{r['start_line']}-{r['end_line']})")
    print(f"  {r['content'][:100]}...")
```

### Chunk a File for Analysis
```bash
curl -X POST http://localhost:7863/chunk \
  -H "Content-Type: application/json" \
  -d '{"file_path": "utils.py", "content": "...", "language": "python"}'
```
