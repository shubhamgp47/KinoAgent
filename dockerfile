FROM python:3.11-slim

WORKDIR /app

# 1. System packages: curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Set default port for local testing
ENV PORT=8501
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 3. Pre-install CPU-only PyTorch to avoid CUDA bloat
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# 4. Copy dependency specification and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy only runtime code and static fixtures (exclude evals)[cite: 1, 2]
COPY src/ ./src/
COPY data/ ./data/

# 6. Documentation ports (Streamlit + Prometheus)[cite: 4]
EXPOSE 8501
EXPOSE 8000

# 7. Dynamic Healthcheck mapped to active PORT
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# 8. Production entrypoint with shell expansion
CMD ["sh", "-c", "streamlit run src/kinoagent/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false --server.enableCORS=false --server.enableXsrfProtection=false"]