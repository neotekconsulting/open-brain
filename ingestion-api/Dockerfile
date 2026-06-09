# Grabs DATABASE_URL/OLLAMA_URL from Compose env injection.
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE} AS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

FROM ${BASE_IMAGE}
WORKDIR /app

COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY app ./app

# Compose already defines port mapping; default aligned with docker-compose default.
ENV INGESTION_PORT=8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${INGESTION_PORT:-8080}"]
