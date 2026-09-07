FROM python:3.11-slim

WORKDIR /app

# Install minimal OS dependencies for network operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install CPU-only PyTorch AND CPU-only torchvision
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Copy and install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source, data assets, and evaluation harness
COPY src/ ./src/
COPY data/ ./data/
COPY evals/ ./evals/

EXPOSE 8501
EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/kinoagent/app.py", "--server.port=8501", "--server.address=0.0.0.0"]