# Open Brain

`neotekconsulting/open-brain` is a local-first brain extension for software teams.
It captures thoughts into Postgres with vector embeddings via Ollama and exposes
them through a FastAPI ingestion API and a FastMCP brain server.

**Phase 1 scope:** automated capture with embeddings, similarity search, and MCP
tooling so agent-based clients can query stored memories. All services run in
Docker Compose.

## Features

- Thought capture through a typed REST API (`POST /capture`) with automatic
  embedding generation using `nomic-embed-text` (768-dim).
- Structured metadata extraction through a local LLM (`llama3.2` by default)
  producing `type`, `people`, `topics`, and `action_items`.
- pgvector-backed semantic search with an HNSW index for cosine similarity.
- MCP server exposing four tools: `semantic_search`, `list_recent`, `stats`,
  and `capture_thought`, usable from Claude Desktop, Cursor, or any JSON-RPC
  client.
- End-to-end smoke test for validating ingestion and retrieval.
- Fully declarative local stack with persistent Postgres and Ollama volumes.

## Architecture

```
                         +----------+
                         |  Client  |
                         +----+-----+
                              |
                  +-----------+-----------+
                  |                       |
              HTTP /capture           JSON-RPC /mcp
                  |                       |
          +-------+-------+      +-------+-------+
          | ingestion-api |      |   mcp-brain   |
          |  FastAPI      |      |  FastMCP      |
          +---+-----------+      +--+----+--------+
              |                     |    |
              +----------+----------+    |
                         |               |
                     Embeddings           |
                         |               |
                    +----+-----+         |
                    |  ollama  |         |
                    +---------+         |
                         |               |
                    Embedding            |
                    Storage             |
                         |               |
              +----------+----------+    |
              |                     |    |
        +------+-----+      +-------+----+----+
        | postgres   |      | postgres      |
        | pgvector   |      | pgvector      |
        | :5432      |      | :5432         |
        +------------+      +---------------+
```


## Prerequisites

- **Docker** and **Docker Compose** (version 2+ plugin).
- **Ollama** container is provided by Compose; you do not need a host install.
- 8 GB+ RAM recommended for `llama3.2` and `nomic-embed-text` plus Postgres.
- To run the smoke test from your host you also need **Python 3.10+** with the
  `asyncpg` and `httpx` packages (`pip install asyncpg httpx`).

## Environment

Copy `.env` from the repo root or create one with at least:

```bash
POSTGRES_USER=openbrain
POSTGRES_PASSWORD=changeme
POSTGRES_DB=openbrain
POSTGRES_PORT=5432
OLLAMA_PORT=11434
MCP_PORT=8000
INGESTION_PORT=8080
```

The Compose file defines safe defaults for every required variable, so an empty
`.env` is valid for local experiments.

## Quickstart

```bash
# clone the repo
git clone https://github.com/neotekconsulting/open-brain.git
cd open-brain

# optional: create/edit .env to override defaults

# start the stack; the first run pulls the Ollama models (nomic-embed-text and
# llama3.2) via the one-shot `ollama-init` service, which can take several
# minutes before the API containers start
docker compose up -d

# watch the model pull finish on the first run
docker compose logs -f ollama-init

# verify services are up
docker compose ps
curl http://localhost:8080/health   # ingestion API -> {"status": "ok"}
```

The MCP brain server speaks streamable HTTP (JSON-RPC) at
`http://localhost:8000/mcp`; it has no plain `GET /health` route.

Stop the stack with `docker compose down`. Data persists in the `pgdata` and
`ollama` named volumes; add `-v` (`docker compose down -v`) to wipe them — this
is required if you change `schema.sql`, which only initializes while `pgdata`
is empty.

## Smoke test

The smoke test runs on your **host** (not in a container) and exercises the
whole stack end to end.

First, start the stack and let the models finish downloading (see Quickstart).
Then install the host dependencies once:

```bash
pip install asyncpg httpx
```

Run the test from the repo root:

```bash
python tests/smoke_test.py
```

It waits for Postgres, Ollama, the ingestion API, and the MCP server, inserts
five built-in test thoughts via `POST /capture`, then performs an MCP
`initialize` handshake and calls the `semantic_search` tool over streamable
HTTP for each thought. It asserts a relevant top result per query under a
cosine-distance threshold (default `0.75`). On success it prints
`PASS: smoke test succeeded` and exits `0`.

### Smoke test configuration

All variables are optional; defaults target the Compose stack on `localhost`:

- `SMOKE_INGESTION_URL` — ingestion base URL (default `http://localhost:8080`).
- `SMOKE_MCP_URL` — MCP base URL (default `http://localhost:8000`).
- `OLLAMA_URL` — Ollama base URL (default `http://localhost:11434`).
- `DATABASE_URL` — full Postgres DSN, or set `POSTGRES_USER`,
  `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, and `POSTGRES_DB` to
  have one built automatically.
- `--thoughts-file PATH` — optional JSON array of thought strings to use
  instead of the built-ins.

### Smoke test troubleshooting

- **`/capture` returns 502 on the first run.** The Ollama models are still
  downloading. Wait until `docker compose logs ollama-init` prints `success`,
  then re-run the test.
- **Ollama port `11434` is already in use.** A host Ollama is already running.
  Publish the container's Ollama on a free port and point the test at it:

  ```bash
  OLLAMA_PORT=11435 docker compose up -d
  OLLAMA_URL=http://localhost:11435 python tests/smoke_test.py
  ```

  PowerShell:

  ```powershell
  $env:OLLAMA_PORT='11435'; docker compose up -d
  $env:OLLAMA_URL='http://localhost:11435'; python tests/smoke_test.py
  ```
- **Stale results or schema.** `schema.sql` only runs while `pgdata` is empty.
  Reset with `docker compose down -v`, then start the stack again.

## API Docs

### Ingestion API

Base URL: `http://localhost:8080`

#### `POST /capture`

Captures a thought, generates an embedding, and stores metadata.

Request body:

```json
{
  "raw_text": "We should migrate to Postgres for the new features.",
  "source": "meeting-transcript",
  "thought_type": "decision",
  "evidence_basis": "Discussed during retro",
  "visibility_verified_by_human_at": "2026-06-09"
}
```

Response:

```json
{
  "captured_id": "...",
  "status": "captured",
  "metadata": {}
}
```

#### `GET /health`

Returns `{"status": "ok"}` when the API is ready.

## MCP Tools

The MCP brain server runs at `http://localhost:8000` with FastMCP tool
resolution at `/mcp`. The following tools are available in both HTTP and stdio
transports.

### `semantic_search`

Semantic similarity search using generated embeddings.

| Argument        | Type    | Default | Notes                              |
|-----------------|---------|---------|------------------------------------|
| `query`         | string  | —       | Free-text query                    |
| `limit`         | integer | 10      | Max results (1-100)                |
| `type_filter`   | string  | `null`  | Optional thought type restriction  |

Returns a list of matching thoughts including `distance`, `source`, `topics`,
`people`, and `action_items`.

### `list_recent`

Returns recently captured thoughts ordered by creation time descending.

| Argument | Type   | Default | Notes                                      |
|----------|--------|---------|--------------------------------------------|
| `limit`  | integer| 20      | Max results (1-200)                        |
| `since`  | string | `null`  | ISO-8601 timestamp or date filter          |

### `stats`

Aggregate counts and date bounds for stored thoughts.

Returns `total_thoughts`, `total_embeddings`, `thoughts_by_type`,
`first_thought_at`, and `latest_thought_at`.

### `capture_thought`

Persists a thought through the MCP server.

| Argument         | Type    | Default | Notes                   |
|------------------|---------|---------|-------------------------|
| `text`           | string  | —       | Required, non-empty     |
| `type`           | string  | `null`  | Optional classifier     |
| `people`         | array   | `[]`    | Optional                |
| `topics`         | array   | `[]`    | Optional                |
| `action_items`   | array   | `[]`    | Optional                |
| `source`         | string  | `null`  | Optional source label   |
| `evidence_basis` | string  | `null`  | Optional justification  |

## Known Issues (Phase 1)

- Metadata extraction is best-effort. Ollama generation can return non-JSON
  text; the ingestion API logs a warning and continues without metadata.
- Embedding retrieval is coupled to `nomic-embed-text` (768 dimensions). Models
  with different output sizes require code changes.
- The MCP stdio transport is supported by FastMCP but is not exercised by the
  integration tests in this phase.
- Postgres must be reachable for the MCP server to start; there is no retry
  back-off in the initial pool creation path.
- First-run Ollama model pulls add significant startup latency.

## Development

```bash
# rebuild after code changes
docker compose up -d --build
```
