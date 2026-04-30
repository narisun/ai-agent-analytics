# Build context: monorepo root (docker build -f agents/analytics-agent/Dockerfile .)
# After Phase 5 carve-out, build context is the agent's own repo root.
ARG BASE_TAG=3.11-sdk0.4.0
FROM ghcr.io/narisun/ai-python-base:${BASE_TAG}

WORKDIR /app

COPY agents/analytics-agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/analytics-agent/src/ /app/src/

USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
