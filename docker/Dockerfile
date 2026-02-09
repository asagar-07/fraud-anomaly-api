FROM python:3.12-slim

WORKDIR /app

# System deps (minimal; add build-essential only if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code + artifacts needed for inference
COPY src ./src
COPY artifacts ./artifacts

# Ensure imports work with src/ layout
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "fraud.api.app:app", "--host", "0.0.0.0", "--port", "8000"]