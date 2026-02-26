"""
core/schemas.py — Pydantic v2 models for all API request/response payloads.

Covers:
  - /embed   (EmbedRequest, EmbedResponse, EmbedMetadata)
  - /chunk   (ChunkRequest, Chunk, ChunkResponse, ChunkMetadata)
  - /index   (IndexRequest, IndexResponse)
  - /search  (SearchRequest, SearchResult, SearchResponse, SearchMetadata)
  - /health  (HealthResponse)
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# /embed
# ---------------------------------------------------------------------------


class EmbedRequest(BaseModel):
    """Request body for POST /embed."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        description="One or more code/text snippets to embed.",
        examples=[["def foo(): pass", "class Bar: ..."]]
    )


class EmbedMetadata(BaseModel):
    model: str = Field(..., description="Sentence-transformer model name used.")
    cached_count: int = Field(
        0, ge=0, description="Number of embeddings served from cache."
    )
    latency_ms: float = Field(..., ge=0, description="Total request latency in ms.")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]] = Field(
        ..., description="Dense embedding vectors, one per input text."
    )
    dimensions: int = Field(..., gt=0, description="Dimensionality of each vector.")
    metadata: EmbedMetadata

    @model_validator(mode="after")
    def _check_lengths(self) -> "EmbedResponse":
        for vec in self.embeddings:
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Embedding length {len(vec)} != declared dimensions {self.dimensions}"
                )
        return self


# ---------------------------------------------------------------------------
# /chunk
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "java", "go",
    "rust", "c", "cpp", "yaml", "json", "markdown", "unknown",
}


class ChunkRequest(BaseModel):
    """Request body for POST /chunk."""

    file_path: str = Field(
        ..., description="Relative path of the source file (used as chunk ID prefix)."
    )
    content: str = Field(..., description="Full text content of the file.")
    language: str = Field(
        "unknown",
        description=f"Source language. One of: {', '.join(sorted(SUPPORTED_LANGUAGES))}.",
    )


class Chunk(BaseModel):
    """A single semantic chunk produced by the smart chunker."""

    id: str = Field(
        ...,
        description="Unique chunk identifier: '<file_path>::<symbol_path>'.",
    )
    content: str = Field(..., description="Source text of the chunk.")
    type: str = Field(
        ...,
        description=(
            "Chunk classification: 'imports' | 'function' | 'class' | "
            "'method' | 'config' | 'section' | 'block'."
        ),
    )
    start_line: int = Field(..., ge=1, description="1-based start line in source file.")
    end_line: int = Field(..., ge=1, description="1-based end line in source file.")
    parent: Optional[str] = Field(
        None, description="Name of the enclosing class or function, if any."
    )
    token_count: Optional[int] = Field(
        None, ge=0, description="Approximate token count of this chunk."
    )

    @model_validator(mode="after")
    def _end_gte_start(self) -> "Chunk":
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line ({self.start_line})"
            )
        return self


class ChunkMetadata(BaseModel):
    language: str
    chunking_strategy: str = Field(
        "code_boundary",
        description="Strategy used: 'code_boundary' | 'line_split' | 'heading'.",
    )


class ChunkResponse(BaseModel):
    chunks: List[Chunk]
    total_chunks: int = Field(..., ge=0)
    metadata: ChunkMetadata

    @model_validator(mode="after")
    def _total_matches(self) -> "ChunkResponse":
        if self.total_chunks != len(self.chunks):
            raise ValueError(
                f"total_chunks ({self.total_chunks}) != len(chunks) ({len(self.chunks)})"
            )
        return self


# ---------------------------------------------------------------------------
# /index
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Request body for POST /index."""

    repo_path: str = Field(
        ..., description="Absolute path to the repository root on the server."
    )
    include_patterns: List[str] = Field(
        default=["*.py", "*.js", "*.ts"],
        description="Glob patterns for files to include.",
    )
    exclude_patterns: List[str] = Field(
        default=["**/node_modules/**", "**/__pycache__/**", "**/.git/**"],
        description="Glob patterns for paths to exclude.",
    )
    force_reindex: bool = Field(
        False,
        description="If True, drops any existing index and re-indexes from scratch.",
    )


class IndexResponse(BaseModel):
    status: str = Field(..., description="'indexed' | 'error'")
    files_processed: int = Field(..., ge=0)
    chunks_created: int = Field(..., ge=0)
    index_size_mb: float = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str = Field(
        ..., min_length=1, description="Natural-language or code query."
    )
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return.")
    min_score: float = Field(
        0.0, ge=0.0, le=1.0, description="Minimum cosine similarity score threshold."
    )


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    content: str
    score: float = Field(..., ge=0.0, le=1.0)
    start_line: int = Field(..., ge=1)
    end_line: int = Field(..., ge=1)


class SearchMetadata(BaseModel):
    query_embedding_ms: float = Field(..., ge=0)
    search_ms: float = Field(..., ge=0)
    total_indexed_chunks: int = Field(..., ge=0)


class SearchResponse(BaseModel):
    results: List[SearchResult]
    metadata: SearchMetadata


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class CacheStats(BaseModel):
    size: int = Field(..., ge=0, description="Current number of cached embeddings.")
    max_size: int = Field(..., ge=0)
    hit_rate: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str = Field("ok", description="'ok' | 'degraded' | 'error'")
    index_loaded: bool = False
    total_indexed_chunks: int = Field(0, ge=0)
    cache: Optional[CacheStats] = None
    memory_mb: Optional[float] = Field(None, ge=0)
    model: Optional[str] = None
