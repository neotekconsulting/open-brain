#!/usr/bin/env python3
"""End-to-end smoke test for the Open Brain stack.

Exercises the local docker-compose stack:
1) Inserts 5 thoughts via the ingestion API.
2) Searches each stored thought via the MCP server.
3) Asserts that retrieval returns a relevant result above the distance threshold.

Run:
  python tests/smoke_test.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import asyncpg
import httpx


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def _die(msg: str) -> "SystemExit":
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


SMOKE_THOUGHTS = [
    "I prefer Rust for building high-performance services.",
    "We should migrate to Postgres for the new features.",
    "The containerized Docker runtime lets us ship faster.",
    "We are exposing a FastAPI for our ingestion needs.",
    "This test illustrates the embedding pipeline works end-to-end.",
]


def _build_database_url() -> str:
    user = os.environ.get("POSTGRES_USER", "openbrain")
    password = os.environ.get("POSTGRES_PASSWORD", "changeme")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "openbrain")
    return f"postgres://{user}:{password}@{host}:{port}/{db}"


def _load_config() -> dict:
    return {
        "DATABASE_URL": os.environ.get(
            "DATABASE_URL", _build_database_url()
        ),
        "OLLAMA_URL": os.environ.get(
            "OLLAMA_URL", "http://localhost:11434"
        ).rstrip("/"),
        "MCP_URL": os.environ.get(
            "SMOKE_MCP_URL", "http://localhost:8000"
        ).rstrip("/"),
        "INGESTION_URL": os.environ.get(
            "SMOKE_INGESTION_URL", "http://localhost:8080"
        ).rstrip("/"),
    }


def _wait_for_http(url: str, *, timeout: float = 120.0, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=httpx.Timeout(5)) as client:
                r = client.get(url)
                if r.status_code < 500:
                    return
                last_err = f"status={r.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(interval)
    _die(f"Timeout waiting for {url}: {last_err}")


async def _wait_for_postgres(dsn: str, *, timeout: float = 120.0, interval: float = 2.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            conn = await asyncpg.connect(dsn=dsn, timeout=5)
            await conn.execute("SELECT 1")
            await conn.close()
            return
        except Exception as exc:
            last_err = str(exc)
        await asyncio.sleep(interval)
    _die(f"Timeout waiting for Postgres at {dsn}: {last_err}")


def _ingest_thoughts(cfg: dict, thoughts: list[str]) -> list[dict]:
    base = cfg["INGESTION_URL"]
    _info("Waiting for ingestion API /health...")
    _wait_for_http(f"{base}/health")

    results: list[dict] = []
    with httpx.Client(timeout=120) as client:
        for payload in thoughts:
            r = client.post(f"{base}/capture", json={"raw_text": payload})
            try:
                r.raise_for_status()
                body = r.json()
            except Exception as exc:
                _die(f"Ingestion request failed for payload={payload!r}: {exc}")

            try:
                cid = body["captured_id"]
            except KeyError:
                _die(f"Ingestion response missing captured_id: {body}")

            embedded = body.get("embedding_generated", False)
            results.append(body)
            snippet = (payload or "")[:30]
            print(f"PASS ingest: id={cid} embedded={embedded} text={snippet!r}")
            time.sleep(0.2)
    return results


def _parse_mcp_payload(resp: httpx.Response) -> dict:
    """Parse an MCP streamable-HTTP response.

    The transport may reply either with a plain JSON body or with an SSE stream
    (``text/event-stream``) that carries the JSON-RPC message in ``data:`` lines.
    """
    ctype = resp.headers.get("content-type", "")
    text = resp.text
    if "text/event-stream" in ctype:
        data_parts: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                data_parts.append(stripped[len("data:"):].strip())
        raw = "".join(data_parts).strip()
    else:
        raw = text.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


async def _mcp_post(
    client: httpx.AsyncClient,
    base: str,
    method: str,
    params: dict,
    *,
    session_id: str | None = None,
    request_id: int | None = None,
) -> httpx.Response:
    """Send one JSON-RPC message to the MCP streamable-HTTP endpoint."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    payload: dict = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        payload["id"] = request_id
    return await client.post(f"{base}/mcp", json=payload, headers=headers)


async def _call_mcp_search(cfg: dict, query: str, limit: int = 5) -> list:
    base = cfg["MCP_URL"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(120)) as client:
        # 1) MCP handshake: initialize, then send the initialized notification.
        init = await _mcp_post(
            client,
            base,
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "open-brain-smoke", "version": "1.0"},
            },
            request_id=1,
        )
        try:
            init.raise_for_status()
        except Exception as exc:
            _die(f"MCP initialize failed: {exc}")
        session_id = init.headers.get("mcp-session-id")
        await _mcp_post(
            client,
            base,
            "notifications/initialized",
            {},
            session_id=session_id,
        )

        # 2) Invoke the semantic_search tool.
        r = await _mcp_post(
            client,
            base,
            "tools/call",
            {
                "name": "semantic_search",
                "arguments": {"query": query, "limit": limit},
            },
            session_id=session_id,
            request_id=2,
        )
        try:
            r.raise_for_status()
        except Exception as exc:
            _die(f"MCP search request failed for query={query!r}: {exc}")

        body = _parse_mcp_payload(r)
        search_results: list = []
        if isinstance(body, dict):
            result_obj = body.get("result") or {}
            content = (result_obj.get("content") or [{}])[0]
            if isinstance(content, dict) and content.get("type") == "text":
                try:
                    search_results = json.loads(content.get("text") or "[]")
                except json.JSONDecodeError as exc:
                    _die(f"Invalid search result JSON for query={query!r}: {exc}")
            elif isinstance(content, dict):
                search_results = content.get("results") or []
            # Fallback: some servers also return parsed data under structuredContent.
            if not search_results and isinstance(result_obj.get("structuredContent"), dict):
                sc = result_obj["structuredContent"]
                search_results = sc.get("result") or sc.get("results") or []

        if not isinstance(search_results, list):
            _die(f"Unexpected non-list search results for query={query!r}")
        return search_results


def _assert_result(query: str, result: list, *, max_distance: float = 0.75) -> bool:
    if not result:
        print(f"FAIL search: query={query!r} returned zero results")
        return False

    top = result[0]
    text = str(top.get("text") or top.get("raw_text") or top.get("rawText") or "")
    distance = top.get("distance")
    distance = distance if isinstance(distance, (int, float)) else None

    passed = True
    parts: list[str] = [f"query={query!r}", f"results={len(result)}"]
    parts.append(f"top_text={text[:40]!r}")
    if distance is not None:
        parts.append(f"distance={distance:.4f}")

    if distance is not None and distance > max_distance:
        print("FAIL search: " + " ".join(parts))
        passed = False
    else:
        print("PASS search: " + " ".join(parts))

    return passed


def _run_lint() -> bool:
    lint_targets = [
        os.path.join(REPO_ROOT, "ingestion-api", "app", "main.py"),
        os.path.join(REPO_ROOT, "mcp-brain", "server.py"),
        os.path.join(REPO_ROOT, "tests", "smoke_test.py"),
    ]
    for path in lint_targets:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        if "`json`" in src and "json.dumps" in src:
            _die(f"Shadowing detected in {path}: remove the `json = ...` binding before running.")
        if "json = __import__('json')" in src:
            _die(f"Shadowing detected in {path}: replace `json = __import__('json')` with `import json`.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Open Brain smoke test")
    parser.add_argument("--run-lint", action="store_true", help="Run local lint checks before executing")
    parser.add_argument("--thoughts-file", default=os.path.join(SCRIPT_DIR, "smoke_thoughts.json"), help="Path to JSON list of thought strings")
    args = parser.parse_args()

    if args.run_lint:
        _info("Running pre-flight lint checks")
        _run_lint()

    cfg = _load_config()

    _info(f"Using ingestion={cfg['INGESTION_URL']}")
    _info(f"Using MCP={cfg['MCP_URL']}")
    _info(f"Using DATABASE_URL={cfg['DATABASE_URL']}")
    _info(f"Using OLLAMA_URL={cfg['OLLAMA_URL']}")

    _info("Waiting for Postgres...")
    asyncio.run(_wait_for_postgres(cfg["DATABASE_URL"]))
    _info("Postgres reachable")

    _info("Waiting for MCP health...")
    _wait_for_http(f"{cfg['MCP_URL']}/health")
    _info("MCP server reachable")

    _info("Waiting for ingestion API health...")
    _wait_for_http(f"{cfg['INGESTION_URL']}/health")
    _info("Ingestion API reachable")

    _info("Waiting for Ollama...")
    _wait_for_http(f"{cfg['OLLAMA_URL']}/")
    _info("Ollama reachable")

    thoughts = list(SMOKE_THOUGHTS)
    try:
        with open(args.thoughts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            thoughts = [str(item) for item in data]
    except Exception as exc:
        _info(f"Using built-in thoughts ({exc})")

    if len(thoughts) < 5:
        _info(f"Less than 5 thoughts loaded; padding with defaults.")
        while len(thoughts) < 5:
            thoughts.append(f"Smoke test thought #{len(thoughts)+1}.")

    selected = thoughts[:5]

    _info("Inserting 5 test thoughts...")
    _ingest_thoughts(cfg, selected)

    _info("Running semantic searches...")
    passed = True
    for thought in selected:
        results = asyncio.run(_call_mcp_search(cfg, thought))
        if not _assert_result(thought, results):
            passed = False
        time.sleep(0.1)

    if passed:
        print("PASS: smoke test succeeded")
        return 0
    print("FAIL: one or more smoke assertions failed")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(1)
