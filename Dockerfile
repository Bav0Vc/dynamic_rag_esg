FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch first so pip doesn't pull the large CUDA wheel
# when sentence-transformers or transformers resolves it as a dependency
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code (data/ & evaluation/results/ are mounted as volumes)
COPY config/     ./config/
COPY pipeline/   ./pipeline/
COPY orchestration/ ./orchestration/
COPY evaluation/ ./evaluation/

CMD ["python", "-m", "orchestration.run_pipeline"]
