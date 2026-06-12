import asyncio
import json
import os
import sys
import uuid
from urllib.parse import urlparse

import asyncpg
import httpx


def _load_env(path: str) -> None:
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_this_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_this_dir, "..", ".env")):
    _load_env(os.path.join(_this_dir, "..", ".env"))

CAPTURE_URL = os.environ.get("CAPTURE_URL", "http://localhost:8080/capture")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"postgres://{os.environ.get('POSTGRES_USER', 'openbrain')}:{os.environ.get('POSTGRES_PASSWORD', 'changeme')}@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'openbrain')}",
)


async def _build_inline_tags(metadata: dict) -> str:
    tags = []
    if metadata.get("type"):
        tags.append(f"#type:{metadata['type']}")
    if metadata.get("people"):
        tags.append(f"#people:{','.join(metadata['people'])}")
    if metadata.get("topics"):
        tags.append(f"#topics:{','.join(metadata['topics'])}")
    if metadata.get("action_items"):
        tags.append(f"#action:{','.join(metadata['action_items'])}")
    return ("\n" + " ".join(tags)) if tags else ""


async def _get_ollama_embedding(text: str, client: httpx.AsyncClient) -> list[float]:
    parsed = urlparse(os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    url = f"{parsed.scheme}://{parsed.netloc}/api/embeddings"
    resp = await client.post(url, json={"model": "nomic-embed-text", "prompt": text}, timeout=120.0)
    resp.raise_for_status()
    embedding = resp.json().get("embedding")
    if not embedding or len(embedding) != 768:
        raise ValueError("Unexpected embedding dimensions")
    return embedding


async def main() -> int:
    if os.environ.get("SKIP_BACKFILL"):
        print("SKIP_BACKFILL set; skipping.")
        return 0

    pool = await asyncpg.create_pool(DATABASE_URL)
    if pool is None:
        raise RuntimeError("Failed to create database pool")

    rows = await pool.fetch("SELECT id, raw_text, thought_type, metadata, source, evidence_basis, visibility_verified_by_human_at FROM thoughts ORDER BY created_at ASC")
    updated = 0
    skipped = 0

    async with httpx.AsyncClient() as client:
        for row in rows:
            thought_id = row["id"]
            metadata = row["metadata"] or {}
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            tagged_text = (row["raw_text"] or "") + _build_inline_tags(metadata)
            if tagged_text == row["raw_text"]:
                skipped += 1
                continue

            embedding = await _get_ollama_embedding(tagged_text, client)
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE thoughts SET raw_text = $1 WHERE id = $2",
                        tagged_text,
                        thought_id,
                    )
                    await conn.execute(
                        "UPDATE embeddings SET vector = $1::vector WHERE thought_id = $2",
                        str(embedding),
                        thought_id,
                    )
            updated += 1
            print(f"updated {updated} / {len(rows)}: {thought_id}")

    await pool.close()
    print(f"backfill complete: updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
