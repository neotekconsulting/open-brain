#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open Brain MCP Server — FastMCP implementation.

Exposes tools:
  - semantic_search(query, limit?, type_filter?)
  - list_recent(limit?, since?)
  - stats()
  - capture_thought(text, type?, people?, topics?, action_items?)

Transport: stdio for Claude Desktop, HTTP for generic JSON-RPC clients.

Environment:
  - DATABASE_URL   Postgres connection string (e.g. postgres://user:pass@host/db)
  - OLLAMA_URL     Ollama base URL (e.g. http://ollama:11434)
  - MCP_PORT       HTTP port (default: 8000)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import asyncpg
import httpx
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config (env var sourcing supports defaults from docker-compose defaults)
# ---------------------------------------------------------------------------

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


DATABASE_URL: str = _env("DATABASE_URL", "postgres://openbrain:changeme@postgres:5432/openbrain")
OLLAMA_URL: str = (url := _env("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
MCP_PORT: int = int(_env("MCP_PORT", "8000"))

# Parse PG host for a lightweight "connected?" check
_PG_HOST = urlparse(DATABASE_URL).hostname or "localhost"
_PG_PORT = urlparse(DATABASE_URL).port or 5432


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

app = FastMCP("brain-server", version="0.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_pool() -> asyncpg.Pool:
    """Create (or return cached) connection pool. AsyncPG caches pools per
    DSN, but we make the creation explicit here for clarity and retryability."""
    return await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=10,
        command_timeout=15,
    )


def _validate_content(content: str) -> None:
    if not content or not content.strip():
        raise ValueError("`text` must be a non-empty, non-whitespace string.")


async def _pg_version(pool: asyncpg.Pool) -> str:
    row = await pool.fetchrow("SELECT version() AS v")
    return (row["v"] or "").split(" on ")[0] if row else "unknown"


async def _embed_one(text: str) -> List[float]:
    """Call local Ollama embedding endpoint for a single text and return the
    embedding vector as a list of floats."""
    async with httpx.AsyncClient(
        base_url=OLLAMA_URL,
        timeout=httpx.Timeout(30, connect=10.0),
    ) as client:
        resp = await client.post(
            "/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@app.tool()
async def semantic_search(
    query: str,
    limit: int = 10,
    type_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Semantic similarity search over captured thoughts using pgvector cosine
    distance.

    Args:
        query:          Free-text search query.
        limit:          Max results to return (1-100). Default 10.
        type_filter:    Optional thought type to restrict results.

    Returns:
        List of matching thoughts with distance, thought_type, topics,
        people, action_items, and created_at.
    """
    _validate_content(query)
    limit = max(1, min(100, int(limit)))

    embedding = await _embed_one(query)

    pool = await _get_pool()
    async with pool.acquire() as conn:
        sql = """
            SELECT
                t.id,
                t.raw_text,
                t.thought_type,
                t.created_at,
                t.source,
                t.metadata,
                e.vector <=> $1::vector AS cosine_distance
            FROM thoughts t
            JOIN embeddings e ON e.thought_id = t.id
            WHERE ($2::text IS NULL OR t.thought_type = $2::text)
            ORDER BY e.vector <=> $1::vector
            LIMIT $3;
        """
        rows = await conn.fetch(sql, embedding, type_filter, limit)

    results: List[Dict[str, Any]] = []
    for row in rows:
        meta = dict(row["metadata"] or {})
        results.append({
            "id": str(row["id"]),
            "text": row["raw_text"],
            "type": row["thought_type"],
            "source": row["source"],
            "topics": meta.get("topics", []),
            "people": meta.get("people", []),
            "action_items": meta.get("action_items", []),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "distance": round(row["cosine_distance"], 6),
        })
    return results


@app.tool()
async def list_recent(
    limit: int = 20,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List recently captured thoughts (creation-time desc).

    Args:
        limit:  Max results to return (1-200). Default 20.
        since:  ISO-8601 timestamp or date string (e.g. '2026-06-01') to only
                include thoughts created after this point.

    Returns:
        List of thoughts ordered newest-first.
    """
    limit = max(1, min(200, int(limit)))

    # Parse filter
    since_dt: Optional[datetime] = None
    if since:
        # Accept ISO or simple date strings gracefully
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                since_dt = datetime.strptime(since, fmt)
                break
            except ValueError:
                continue
        if since_dt is None:
            raise ValueError(
                f"Could not parse `since` value '{since}'. "
                "Expected YYYY-MM-DD or ISO-8601."
            )

    pool = await _get_pool()
    async with pool.acquire() as conn:
        sql = """
            SELECT
                id, raw_text, thought_type, created_at, source, metadata
            FROM thoughts
            WHERE ($1::timestamptz IS NULL OR created_at > $1::timestamptz)
            ORDER BY created_at DESC
            LIMIT $2;
        """
        rows = await conn.fetch(sql, since_dt, limit)

    results: List[Dict[str, Any]] = []
    for row in rows:
        meta = dict(row["metadata"] or {})
        results.append({
            "id": str(row["id"]),
            "text": row["raw_text"],
            "type": row["thought_type"],
            "source": row["source"],
            "topics": meta.get("topics", []),
            "people": meta.get("people", []),
            "action_items": meta.get("action_items", []),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        })
    return results


@app.tool()
async def stats() -> Dict[str, Any]:
    """Return aggregate counts across the brain store.

    Returns:
        Dict with total_thoughts, total_embeddings, thoughts_by_type,
        first_thought_at, latest_thought_at.
    """
    pool = await _get_pool()
    async with pool.acquire() as conn:
        row_total = await conn.fetchrow(
            "SELECT count(*) AS c FROM thoughts"
        )
        row_embed = await conn.fetchrow(
            "SELECT count(*) AS c FROM embeddings"
        )
        row_types = await conn.fetch(
            """
            SELECT thought_type, count(*) AS count
            FROM thoughts
            WHERE thought_type IS NOT NULL
            GROUP BY thought_type
            ORDER BY count DESC;
            """
        )
        row_bounds = await conn.fetchrow(
            """
            SELECT
                min(created_at) AS first_at,
                max(created_at) AS last_at
            FROM thoughts;
            """
        )

    types: Dict[str, int] = {r["thought_type"]: r["count"] for r in (row_types or [])}
    return {
        "total_thoughts": int(row_total["c"]) if row_total else 0,
        "total_embeddings": int(row_embed["c"]) if row_embed else 0,
        "thoughts_by_type": types,
        "first_thought_at": (
            row_bounds["first_at"].isoformat()
            if row_bounds and row_bounds.get("first_at")
            else None
        ),
        "latest_thought_at": (
            row_bounds["last_at"].isoformat()
            if row_bounds and row_bounds.get("last_at")
            else None
        ),
    }


@app.tool()
async def capture_thought(
    text: str,
    type: Optional[str] = None,
    people: Optional[List[str]] = None,
    topics: Optional[List[str]] = None,
    action_items: Optional[List[str]] = None,
    source: Optional[str] = None,
    evidence_basis: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture a new thought into the brain. Optionally embed it via Ollama so it
    becomes retrievable by `semantic_search`.

    Args:
        text:            Thought body (required, must be non-empty).
        type:            Optional classifier tag, e.g. "insight", "decision".
        people:          Optional list of people/actors mentioned.
        topics:          Optional list of topic tags.
        action_items:    Optional list of actionable items.
        source:          Optional source label (meeting, transcript, etc.).
        evidence_basis:  Optional short justification string.

    Returns:
        Dict with the inserted thought id, created_at, and embedding status.
    """
    _validate_content(text)

    metadata: Dict[str, Any] = {}
    if people:
        metadata["people"] = list(people)
    if topics:
        metadata["topics"] = list(topics)
    if action_items:
        metadata["action_items"] = list(action_items)
    if metadata:
        # Only include non-empty metadata
        pass

    pool = await _get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """
                INSERT INTO thoughts
                    (raw_text, thought_type, source, metadata, evidence_basis)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                RETURNING id, created_at;
                """,
                text,
                type,
                source,
                json.dumps(metadata) if metadata else None,
                evidence_basis,
            )

            thought_id = row["id"]
            created_at = row["created_at"]

            # Generate embedding via Ollama if available
            embedding_ok = False
            embedding_model = "nomic-embed-text"
            try:
                vector = await _embed_one(text)
                if vector:
                    await conn.execute(
                        """
                        INSERT INTO embeddings (thought_id, vector, model)
                        VALUES ($1, $2::vector, $3);
                        """,
                        thought_id,
                        str(vector),
                        embedding_model,
                    )
                    embedding_ok = True
            except Exception:
                # Best-effort; record that embedding failed but thought was stored
                embedding_ok = False

    return {
        "id": str(thought_id),
        "created_at": created_at.isoformat() if created_at else None,
        "type": type,
        "source": source,
        "topics": topics or [],
        "people": people or [],
        "action_items": action_items or [],
        "embedding_model": embedding_model if embedding_ok else None,
        "embedding_generated": embedding_ok,
    }


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """Serialize dict as indented JSON and write atomically."""
    import os as _os
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
    _os.replace(tmp, path)


def _extension_abs_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def write_claude_desktop_config(output_dir: Optional[str] = None) -> str:
    """Write `claude_desktop_config.json` for Claude Desktop with the brain
    server configured for stdio transport.

    Output path: `{output_dir}/claude_desktop_config.json` or the mcp-brain
    directory by default.

    Returns:
        Absolute path of the written file.
    """
    here = output_dir or _extension_abs_dir()
    out = os.path.join(here, "claude_desktop_config.json")

    payload = {
        "mcpServers": {
            "open-brain": {
                "command": "python",
                "args": ["-m", "server.main"],
                "cwd": _extension_abs_dir(),
                "env": {
                    "DATABASE_URL": DATABASE_URL,
                    "OLLAMA_URL": OLLAMA_URL,
                },
            }
        }
    }
    _write_json(out, payload)
    return out


def write_cursor_mcp_config(output_dir: Optional[str] = None) -> str:
    """Write `mcp.json` for Cursor IDE with the brain server as an HTTP
    transport endpoint.

    Output path: `{output_dir}/mcp.json` or the mcp-brain directory by default.

    Returns:
        Absolute path of the written file.
    """
    here = output_dir or _extension_abs_dir()
    out = os.path.join(here, "mcp.json")

    host = "127.0.0.1"
    port = MCP_PORT
    payload = {
        "mcpServers": {
            "open-brain": {
                "type": "http",
                "url": f"http://{host}:{port}/mcp",
                "transport": {
                    "type": "http",
                    "endpoint": "/mcp",
                },
                "env": {
                    "DATABASE_URL": DATABASE_URL,
                    "OLLAMA_URL": OLLAMA_URL,
                },
            }
        }
    }
    _write_json(out, payload)
    return out


# ---------------------------------------------------------------------------
# Entrypoint & transport selection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Open Brain MCP Server")
    parser.add_argument(
        "transport",
        choices=["stdio", "http"],
        nargs="?",
        default="stdio",
        help="Transport for FastMCP: 'stdio' for Claude Desktop, 'http' for JSON-RPC.",
    )
    parser.add_argument(
        "--write-claude-config",
        action="store_true",
        help="Write claude_desktop_config.json and exit.",
    )
    parser.add_argument(
        "--write-cursor-config",
        action="store_true",
        help="Write mcp.json for Cursor and exit.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory to write config files into (default: mcp-brain/).",
    )
    args = parser.parse_args()

    if args.write_claude_config:
        p = write_claude_desktop_config(args.config_dir)
        print(f"Wrote: {p}", file=sys.stderr)
    elif args.write_cursor_config:
        p = write_cursor_mcp_config(args.config_dir)
        print(f"Wrote: {p}", file=sys.stderr)
    elif args.transport == "stdio":
        try:
            app.run(transport="stdio")
        except KeyboardInterrupt:
            pass
    elif args.transport == "http":
        try:
            app.run(transport="http", host="0.0.0.0", port=MCP_PORT)
        except KeyboardInterrupt:
            pass
