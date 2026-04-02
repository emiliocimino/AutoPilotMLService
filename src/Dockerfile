FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
LABEL authors="emiliocimino"

WORKDIR /home

COPY .. .

RUN uv sync --project pyproject.toml

CMD [".venv/bin/fastapi", "run", "app.py", "--port", "8010"]
