# Open Brain — MCP Brain Server
# Build runtime for:
#  - asyncpg (Postgres async driver)
#  - httpx (Ollama HTTP client)
#  - fastmcp (MCP framework)
#
# Transport:
#  - stdio via default CMD
#  - HTTP when overridden with transport=http

FROM python:3.12-slim

# Prefer reproducible installs; apt only for build/runtime libs if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY server.py ./

ENV DATABASE_URL="postgres://openbrain:changeme@postgres:5432/openbrain" \
    OLLAMA_URL="http://ollama:11434" \
    MCP_PORT="8000"

EXPOSE 8000

# Default transport: stdio (Claude Desktop)
# To run HTTP:
#   docker run ... mcp-brain python server.py http
CMD ["python", "server.py", "stdio"]
