import os
import uuid
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Open Brain Ingestion API")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgres://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@postgres:5432/{os.environ['POSTGRES_DB']}",
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
INGESTION_PORT = int(os.environ.get("INGESTION_PORT", 8080))

EMBED_MODEL = "nomic-embed-text"
METADATA_MODEL = "llama3.2"  # small, reasonable default for structure extraction


class CaptureRequest(BaseModel):
    raw_text: str
    source: Optional[str] = None
    thought_type: Optional[str] = None
    visibility_verified_by_human_at: Optional[str] = None
    evidence_basis: Optional[str] = None


class CaptureResponse(BaseModel):
    captured_id: str
    status: str
    metadata: Optional[dict] = None


async def get_ollama_embedding(text: str, client: httpx.AsyncClient) -> list[float]:
    """Call Ollama embedding API. Expects 768-dim output for nomic-embed-text."""
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    resp = await client.post(url, json=payload, timeout=120.0)
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding")
    if not embedding or len(embedding) != 768:
        raise ValueError(f"Unexpected embedding dimensions for {EMBED_MODEL}: {len(embedding) if embedding else 'None'}")
    return embedding


async def extract_metadata(raw_text: str, client: httpx.AsyncClient) -> dict:
    """Use a small LLM via Ollama to extract structured metadata."""
    system_prompt = (
        "Extract structured metadata from the following thought/diary text. "
        "Return a JSON object only, with keys: type, people (list of strings), "
        "topics (list of strings), action_items (list of strings). "
        "If a value is unknown, use an empty list."
    )
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": METADATA_MODEL,
        "prompt": f"{system_prompt}\n\nText:\n{raw_text}\n\nJSON:",
        "stream": False,
        "format": "json",
    }
    resp = await client.post(url, json=payload, timeout=180.0)
    resp.raise_for_status()
    data = resp.json().get("response", {})
    if isinstance(data, str):
        import json as _json
        try:
            metadata = _json.loads(data)
        except Exception:
            metadata = {}
    else:
        metadata = data or {}
    metadata.setdefault("type", None)
    metadata.setdefault("people", [])
    metadata.setdefault("topics", [])
    metadata.setdefault("action_items", [])
    return metadata


@app.on_event("startup")
async def startup() -> None:
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
    logger.info("Database pool created.")


@app.on_event("shutdown")
async def shutdown() -> None:
    if getattr(app.state, "db_pool", None) is not None:
        await app.state.db_pool.close()
        logger.info("Database pool closed.")


@app.post("/capture", response_model=CaptureResponse)
async def capture(req: CaptureRequest) -> CaptureResponse:
    if not req.raw_text.strip():
        raise HTTPException(status_code=400, detail="raw_text must not be empty")

    captured_id = str(uuid.uuid4())
    async with httpx.AsyncClient() as client:
        # 1) Embedding
        try:
            embedding = await get_ollama_embedding(req.raw_text, client)
        except Exception as exc:
            logger.exception("Embedding failed")
            raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

        # 2) Metadata extraction (best-effort)
        metadata: dict = {}
        try:
            metadata = await extract_metadata(req.raw_text, client)
        except Exception as exc:
            logger.warning("Metadata extraction failed; continuing without metadata: %s", exc)

    # 3) Persist
    async with app.state.db_pool.acquire() as conn:
        # Build JSON metadata without passing source twice since source lives on thoughts table
        json_metadata = {k: v for k, v in metadata.items() if k not in ("source", "raw_text")}
        if req.source:
            json_metadata["source"] = req.source

        thought_id = await conn.fetchval(
            """
            INSERT INTO thoughts (id, raw_text, thought_type, source, metadata, evidence_basis, visibility_verified_by_human_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            uuid.UUID(captured_id),
            req.raw_text,
            metadata.get("type"),
            req.source,
            json_metadata,
            req.evidence_basis,
            req.visibility_verified_by_human_at,
        )
        await conn.execute(
            """
            INSERT INTO embeddings (thought_id, vector, model)
            VALUES ($1, $2, $3)
            """,
            thought_id,
            embedding,
            EMBED_MODEL,
        )

    return CaptureResponse(captured_id=captured_id, status="captured", metadata=metadata)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
