# ai-agent-analytics

Enterprise AI Platform — Analytics Agent (LangGraph + FastAPI orchestrator).

## Quick start

```bash
pip install -r requirements.txt
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## Requirements files

- **`requirements.txt`** — full dependency list including the SDK pinned via `git+ssh`. Used for local development and CI.
- **`requirements-runtime.txt`** — only the agent-specific runtime extras (uvicorn, jinja2). Used by the Dockerfile because the base image already provides the SDK.

This split exists so the Docker build doesn't need git or SSH inside the image (the SDK is baked into the base image at SDK release time).

## Build the container

```bash
docker build -t ai-agent-analytics:dev .
docker run --rm -p 8000:8000 ai-agent-analytics:dev
```

The Dockerfile inherits from `ghcr.io/narisun/ai-python-base:3.11-sdk0.4.0`,
which has the platform SDK pre-installed.

## Local SDK development

If you're modifying the SDK alongside this agent:

```bash
pip install -e ../ai-platform-sdk
```

(Editable install overrides the git+ssh pin in `requirements.txt`.)

## CI

CI installs the SDK via git+https (rewriting the SSH pin for CI use only).
This works because `narisun/ai-platform-sdk` is a public repo.
