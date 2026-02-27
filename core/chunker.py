"""
core/chunker.py — Code-boundary-aware smart chunker.

Supported languages / strategies:
  python       — regex-based, detects def/class/import blocks
  javascript   — regex-based, detects function/class/import blocks
  typescript   — same patterns as javascript + type/interface/enum
  java, go,
  rust, cpp,
  c            — generic brace-based function/class detection
  yaml         — one chunk per top-level key
  json         — one chunk per top-level key
  markdown     — one chunk per heading section
  unknown      — whole file as single block; split if > MAX_CHUNK_TOKENS

Environment variables (read once at import, overrideable via constructor):
  MAX_CHUNK_TOKENS  — maximum tokens per chunk before splitting (default: 512)
  CHUNK_OVERLAP     — overlap tokens when splitting long chunks (default: 50)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from core.schemas import Chunk, ChunkMetadata, ChunkResponse

_MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Approximate tokenisation: split on whitespace + punctuation (no model needed).
_TOKEN_RE = re.compile(r"\S+")


def _approx_token_count(text: str) -> int:
    """Fast, model-agnostic token count estimate (word-piece approximation)."""
    return len(_TOKEN_RE.findall(text))


def _split_long_chunk(
    chunk: Chunk,
    max_tokens: int = _MAX_CHUNK_TOKENS,
    overlap: int = _CHUNK_OVERLAP,
) -> List[Chunk]:
    """Split a single oversized chunk into smaller pieces at line boundaries."""
    lines = chunk.content.splitlines(keepends=True)
    pieces: List[Chunk] = []
    current_lines: List[str] = []
    current_tokens = 0
    part_index = 0
    start_line = chunk.start_line

    for i, line in enumerate(lines):
        line_tokens = _approx_token_count(line)
        if current_tokens + line_tokens > max_tokens and current_lines:
            # Emit current piece
            content = "".join(current_lines)
            end_line = start_line + len(current_lines) - 1
            pieces.append(
                Chunk(
                    id=f"{chunk.id}#part{part_index}",
                    content=content,
                    type=chunk.type,
                    start_line=start_line,
                    end_line=end_line,
                    parent=chunk.parent,
                    token_count=current_tokens,
                )
            )
            part_index += 1
            # Overlap: keep last `overlap` tokens worth of lines
            overlap_lines: List[str] = []
            overlap_tokens = 0
            for prev_line in reversed(current_lines):
                t = _approx_token_count(prev_line)
                if overlap_tokens + t > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_tokens += t
            current_lines = overlap_lines
            current_tokens = overlap_tokens
            start_line = end_line - len(overlap_lines) + 1

        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        content = "".join(current_lines)
        end_line = start_line + len(current_lines) - 1
        suffix = f"#part{part_index}" if part_index > 0 else ""
        pieces.append(
            Chunk(
                id=f"{chunk.id}{suffix}",
                content=content,
                type=chunk.type,
                start_line=start_line,
                end_line=end_line,
                parent=chunk.parent,
                token_count=current_tokens,
            )
        )

    return pieces if pieces else [chunk]


def _maybe_split(chunk: Chunk, max_tokens: int, overlap: int) -> List[Chunk]:
    tokens = chunk.token_count if chunk.token_count is not None else _approx_token_count(chunk.content)
    if tokens > max_tokens:
        return _split_long_chunk(chunk, max_tokens, overlap)
    return [chunk]


# ---------------------------------------------------------------------------
# Language-specific chunkers
# ---------------------------------------------------------------------------


@dataclass
class _RawChunk:
    """Intermediate representation before building the final Chunk schema."""

    name: str
    content_lines: List[str]
    start_line: int
    end_line: int
    chunk_type: str
    parent: Optional[str] = None


# --- Python ------------------------------------------------------------------

_PY_IMPORT_RE = re.compile(r"^(?:import |from )")
_PY_CLASS_RE = re.compile(r"^class\s+(\w+)")
_PY_DEF_RE = re.compile(r"^(\s*)def\s+(\w+)")
_PY_DECO_RE = re.compile(r"^\s*@")


def _chunk_python(file_path: str, lines: List[str]) -> List[Chunk]:
    """Regex-based Python chunker."""
    chunks: List[Chunk] = []
    n = len(lines)
    i = 0

    # --- collect import block (contiguous import/from lines at top of file) ---
    import_lines: List[str] = []
    import_start = 1
    while i < n:
        stripped = lines[i].lstrip()
        if _PY_IMPORT_RE.match(stripped) or stripped == "" or stripped.startswith("#"):
            import_lines.append(lines[i])
            i += 1
        else:
            break

    # Trim trailing blank lines from imports
    while import_lines and import_lines[-1].strip() == "":
        import_lines.pop()

    if import_lines:
        content = "".join(import_lines)
        chunks.append(
            Chunk(
                id=f"{file_path}::imports",
                content=content,
                type="imports",
                start_line=import_start,
                end_line=import_start + len(import_lines) - 1,
                parent=None,
                token_count=_approx_token_count(content),
            )
        )

    # --- walk remaining lines for class / function definitions ---------------
    current_class: Optional[str] = None
    current_class_indent: int = -1

    while i < n:
        line = lines[i]
        stripped = line.lstrip()
        indent = len(line) - len(line.lstrip())

        # Update class tracking
        cls_match = _PY_CLASS_RE.match(stripped)
        if cls_match and indent == 0:
            current_class = cls_match.group(1)
            current_class_indent = indent
            i += 1
            continue

        # Reset class when we return to module level with non-class content
        if current_class and indent == 0 and not stripped.startswith("#") and stripped != "":
            if not cls_match:
                current_class = None
                current_class_indent = -1

        def_match = _PY_DEF_RE.match(line)
        if def_match:
            func_indent = len(def_match.group(1))
            func_name = def_match.group(2)

            # Collect decorators above this def
            deco_start = i
            while deco_start > 0 and _PY_DECO_RE.match(lines[deco_start - 1]):
                deco_start -= 1

            # Collect body: all lines with indent > func_indent  or blank lines
            body_lines = list(lines[deco_start:i + 1])  # include def line
            j = i + 1
            while j < n:
                next_line = lines[j]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip()) if next_stripped else func_indent + 4
                if next_stripped == "":
                    body_lines.append(next_line)
                    j += 1
                elif next_indent > func_indent:
                    body_lines.append(next_line)
                    j += 1
                else:
                    break

            # Trim trailing blanks
            while body_lines and body_lines[-1].strip() == "":
                body_lines.pop()

            content = "".join(body_lines)
            end_line = deco_start + len(body_lines)  # 0-based end line index
            chunk_type = "method" if current_class and func_indent > 0 else "function"

            chunk = Chunk(
                id=f"{file_path}::{current_class + '.' if current_class else ''}{func_name}",
                content=content,
                type=chunk_type,
                start_line=deco_start + 1,  # 1-based
                end_line=end_line,
                parent=current_class,
                token_count=_approx_token_count(content),
            )
            chunks.append(chunk)
            i = j
            continue

        i += 1

    return chunks


# --- JavaScript / TypeScript -------------------------------------------------

_JS_IMPORT_RE = re.compile(r"^(?:import |const .+ = require\(|export \{)")
_JS_CLASS_RE = re.compile(r"^(?:export\s+)?(?:default\s+)?class\s+(\w+)")
_JS_FUNC_RE = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)"
    r"|^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?"
    r"(?:function|\([^)]*\)\s*=>|\w+\s*=>)"
)
_TS_TYPE_RE = re.compile(r"^(?:export\s+)?(?:interface|type|enum)\s+(\w+)")


def _collect_block(lines: List[str], start: int) -> Tuple[List[str], int]:
    """Collect a brace-delimited block starting at `start`."""
    block = [lines[start]]
    depth = lines[start].count("{") - lines[start].count("}")
    j = start + 1
    while j < len(lines) and depth > 0:
        block.append(lines[j])
        depth += lines[j].count("{") - lines[j].count("}")
        j += 1
    return block, j


def _chunk_js_ts(file_path: str, lines: List[str], language: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    n = len(lines)
    i = 0
    current_class: Optional[str] = None

    # Import block
    import_lines: List[str] = []
    while i < n and (_JS_IMPORT_RE.match(lines[i]) or lines[i].strip() == ""):
        import_lines.append(lines[i])
        i += 1
    while import_lines and import_lines[-1].strip() == "":
        import_lines.pop()
    if import_lines:
        content = "".join(import_lines)
        chunks.append(Chunk(
            id=f"{file_path}::imports", content=content, type="imports",
            start_line=1, end_line=len(import_lines),
            token_count=_approx_token_count(content),
        ))

    while i < n:
        line = lines[i]
        stripped = line.strip()

        cls_match = _JS_CLASS_RE.match(stripped)
        if cls_match:
            current_class = cls_match.group(1)
            i += 1
            continue

        func_match = _JS_FUNC_RE.match(stripped)
        ts_type_match = _TS_TYPE_RE.match(stripped) if language == "typescript" else None

        if func_match or ts_type_match:
            name = ""
            if func_match:
                name = func_match.group(1) or func_match.group(2) or "anonymous"
                chunk_type = "method" if current_class else "function"
            else:
                name = ts_type_match.group(1)  # type: ignore[union-attr]
                chunk_type = "class"

            if "{" in line:
                block_lines, end_idx = _collect_block(lines, i)
            else:
                block_lines = [lines[i]]
                end_idx = i + 1

            content = "".join(block_lines)
            chunks.append(Chunk(
                id=f"{file_path}::{current_class + '.' if current_class else ''}{name}",
                content=content,
                type=chunk_type,
                start_line=i + 1,
                end_line=i + len(block_lines),
                parent=current_class,
                token_count=_approx_token_count(content),
            ))
            i = end_idx
            continue

        i += 1

    return chunks


# --- YAML / JSON -------------------------------------------------------------

def _chunk_yaml(file_path: str, content: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    try:
        data = yaml.safe_load(content)
    except Exception:
        # Fallback: whole file as one block
        return [Chunk(id=f"{file_path}::root", content=content, type="config",
                      start_line=1, end_line=content.count("\n") + 1,
                      token_count=_approx_token_count(content))]

    if not isinstance(data, dict):
        return [Chunk(id=f"{file_path}::root", content=content, type="config",
                      start_line=1, end_line=content.count("\n") + 1,
                      token_count=_approx_token_count(content))]

    # Reconstruct per-key using yaml.dump so each chunk is valid YAML
    line_cursor = 1
    raw_lines = content.splitlines(keepends=True)

    for key in data:
        key_str = str(key)
        snippet = yaml.dump({key: data[key]}, default_flow_style=False, allow_unicode=True)
        num_lines = snippet.count("\n") or 1
        chunks.append(Chunk(
            id=f"{file_path}::{key_str}",
            content=snippet,
            type="config",
            start_line=line_cursor,
            end_line=line_cursor + num_lines - 1,
            token_count=_approx_token_count(snippet),
        ))
        line_cursor += num_lines

    return chunks


def _chunk_json(file_path: str, content: str) -> List[Chunk]:
    try:
        data = json.loads(content)
    except Exception:
        return [Chunk(id=f"{file_path}::root", content=content, type="config",
                      start_line=1, end_line=content.count("\n") + 1,
                      token_count=_approx_token_count(content))]

    if not isinstance(data, dict):
        return [Chunk(id=f"{file_path}::root", content=content, type="config",
                      start_line=1, end_line=content.count("\n") + 1,
                      token_count=_approx_token_count(content))]

    chunks: List[Chunk] = []
    line_cursor = 1
    for key, value in data.items():
        snippet = json.dumps({key: value}, indent=2, ensure_ascii=False)
        num_lines = snippet.count("\n") + 1
        chunks.append(Chunk(
            id=f"{file_path}::{key}",
            content=snippet,
            type="config",
            start_line=line_cursor,
            end_line=line_cursor + num_lines - 1,
            token_count=_approx_token_count(snippet),
        ))
        line_cursor += num_lines

    return chunks


# --- Markdown ----------------------------------------------------------------

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")


def _chunk_markdown(file_path: str, lines: List[str]) -> List[Chunk]:
    chunks: List[Chunk] = []
    section_lines: List[str] = []
    section_start = 1
    section_name = "preamble"
    i = 0

    def flush(name: str, content_lines: List[str], start: int, end: int) -> None:
        while content_lines and content_lines[-1].strip() == "":
            content_lines.pop()
        if not content_lines:
            return
        content = "".join(content_lines)
        chunks.append(Chunk(
            id=f"{file_path}::{name.replace(' ', '_').lower()[:64]}",
            content=content,
            type="section",
            start_line=start,
            end_line=end,
            token_count=_approx_token_count(content),
        ))

    for i, line in enumerate(lines, start=1):
        m = _MD_HEADING_RE.match(line)
        if m:
            flush(section_name, section_lines, section_start, i - 1)
            section_lines = [line]
            section_start = i
            section_name = m.group(2).strip()
        else:
            section_lines.append(line)

    flush(section_name, section_lines, section_start, i)
    return chunks


# --- Generic (brace-based) ---------------------------------------------------

_GENERIC_FUNC_RE = re.compile(
    r"(?:^|\s)(?:func|function|def|void|int|string|bool|auto|public|private|protected|static)?\s*(\w+)\s*\([^)]*\)\s*(?:->.*?)?\{"
)


def _chunk_generic(file_path: str, lines: List[str]) -> List[Chunk]:
    """Fallback chunker for C, Go, Rust, Java, etc. — collects brace blocks."""
    chunks: List[Chunk] = []
    n = len(lines)
    i = 0

    while i < n:
        line = lines[i]
        m = _GENERIC_FUNC_RE.search(line)
        if m and "{" in line:
            name = m.group(1) or "block"
            block_lines, end_idx = _collect_block(lines, i)
            content = "".join(block_lines)
            chunks.append(Chunk(
                id=f"{file_path}::{name}",
                content=content,
                type="function",
                start_line=i + 1,
                end_line=i + len(block_lines),
                token_count=_approx_token_count(content),
            ))
            i = end_idx
            continue
        i += 1

    # If nothing matched, emit whole file as a block
    if not chunks:
        content = "".join(lines)
        chunks.append(Chunk(
            id=f"{file_path}::module",
            content=content,
            type="block",
            start_line=1,
            end_line=len(lines),
            token_count=_approx_token_count(content),
        ))

    return chunks


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXT_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".mdx": "markdown",
}


def detect_language(file_path: str) -> str:
    """Infer language from file extension; returns 'unknown' if not recognised."""
    ext = Path(file_path).suffix.lower()
    return _EXT_LANGUAGE_MAP.get(ext, "unknown")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class SmartChunker:
    """Stateless chunker — all state lives in env-vars / constructor params."""

    def __init__(
        self,
        max_chunk_tokens: int = _MAX_CHUNK_TOKENS,
        chunk_overlap: int = _CHUNK_OVERLAP,
    ) -> None:
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        file_path: str,
        content: str,
        language: str = "unknown",
    ) -> ChunkResponse:
        """Chunk *content* and return a :class:`ChunkResponse`.

        If *language* is ``'unknown'`` the language is inferred from the
        file extension of *file_path*.
        """
        if language == "unknown":
            language = detect_language(file_path)

        raw_chunks = self._dispatch(file_path, content, language)

        # Apply token-based splitting for oversized chunks
        final: List[Chunk] = []
        for chunk in raw_chunks:
            final.extend(_maybe_split(chunk, self.max_chunk_tokens, self.chunk_overlap))

        # De-duplicate IDs (can happen if two functions share a name)
        seen: dict[str, int] = {}
        deduped: List[Chunk] = []
        for chunk in final:
            if chunk.id in seen:
                seen[chunk.id] += 1
                deduped.append(chunk.model_copy(update={"id": f"{chunk.id}#{seen[chunk.id]}"}))
            else:
                seen[chunk.id] = 0
                deduped.append(chunk)

        strategy = self._strategy_name(language)
        return ChunkResponse(
            chunks=deduped,
            total_chunks=len(deduped),
            metadata=ChunkMetadata(language=language, chunking_strategy=strategy),
        )

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _dispatch(self, file_path: str, content: str, language: str) -> List[Chunk]:
        lines = content.splitlines(keepends=True)
        if language == "python":
            return _chunk_python(file_path, lines)
        if language in ("javascript",):
            return _chunk_js_ts(file_path, lines, "javascript")
        if language == "typescript":
            return _chunk_js_ts(file_path, lines, "typescript")
        if language == "yaml":
            return _chunk_yaml(file_path, content)
        if language == "json":
            return _chunk_json(file_path, content)
        if language == "markdown":
            return _chunk_markdown(file_path, lines)
        if language in ("java", "go", "rust", "c", "cpp"):
            return _chunk_generic(file_path, lines)
        # unknown / fallback
        chunk = Chunk(
            id=f"{file_path}::module",
            content=content,
            type="block",
            start_line=1,
            end_line=len(lines) or 1,
            token_count=_approx_token_count(content),
        )
        return _maybe_split(chunk, self.max_chunk_tokens, self.chunk_overlap)

    @staticmethod
    def _strategy_name(language: str) -> str:
        if language in ("python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"):
            return "code_boundary"
        if language in ("yaml", "json"):
            return "config_key"
        if language == "markdown":
            return "heading"
        return "line_split"
