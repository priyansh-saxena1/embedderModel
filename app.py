"""
Embedder Space — FastAPI application entry point.

Endpoints are wired up progressively as each core module lands.
Full implementation lives in PRs: feat/schemas → feat/cache →
feat/chunker → feat/embedder → feat/indexer → feat/api.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Embedder Space",
    description="Semantic search and embedding service with code-aware chunking.",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok", "message": "Embedder Space is running."}
