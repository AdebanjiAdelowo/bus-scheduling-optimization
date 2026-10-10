# API + dashboard image. Data is not baked in: mount ./data and ./results, or set BUSOPT_SYNTHETIC=1
# to run on the built-in synthetic toy network.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra dashboard --extra agent
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --extra dashboard --extra agent

EXPOSE 8000 8501
# API by default; for the dashboard: docker run ... uv run streamlit run src/busopt/dashboard/app.py --server.address 0.0.0.0
CMD ["uv", "run", "uvicorn", "busopt.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
