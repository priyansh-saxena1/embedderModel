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
