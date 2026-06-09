# MCP Brain Server

FastMCP server exposingOpen Brain tools.

## Environment variables

- `DATABASE_URL` (required): Postgres connection string.
- `OLLAMA_URL` (required): Ollama HTTP endpoint.
- `MCP_PORT`: HTTP port for JSON-RPC clients. Defaults to `8000`.

## Transport

### stdio (Claude Desktop)
```bash
python server.py stdio
```

### HTTP (generic JSON-RPC clients)
```bash
python server.py http
```

HTTP endpoint is served at `http://0.0.0.0:${MCP_PORT}/mcp`.

## Tools

- `semantic_search(query, limit, type_filter)`
- `list_recent(limit, since)`
- `stats()`
- `capture_thought(text, type, people, topics, action_items)`

## Config writers

```bash
python server.py --write-claude-config
python server.py --write-cursor-config
```
```
