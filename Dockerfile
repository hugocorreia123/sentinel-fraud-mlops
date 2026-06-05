# Single-container deployment for Hugging Face Spaces.
# Runs both FastAPI (port 8000, internal) and Streamlit (port 7860, public).
# Models load from local snapshots; no MLflow, no Redis required.

FROM python:3.12-slim

# OS deps for LightGBM (libgomp) + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 supervisor curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. Pin to runtime essentials — exclude heavy ones (mlflow, locust, jupyter)
# that we don't need on Spaces.
RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache \
       --index-url https://download.pytorch.org/whl/cpu \
       torch \
    && uv pip install --system --no-cache \
       fastapi uvicorn pydantic loguru \
       lightgbm \
       polars pandas numpy redis prometheus-client requests streamlit

# Copy only what the runtime needs
COPY apps/ apps/
COPY models/ models/
COPY data_pipeline/features/ data_pipeline/features/
RUN touch data_pipeline/__init__.py apps/__init__.py

# Supervisor config to run both processes
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Hugging Face Spaces requires the container to listen on port 7860
EXPOSE 7860

# Configure runtime: load models from disk, point Streamlit at the in-container FastAPI
ENV MODEL_SOURCE=local \
    SENTINEL_API_URL=http://localhost:8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "-n"]
