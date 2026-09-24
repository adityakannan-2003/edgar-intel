FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Where sentence-transformers caches the model. Baked into the image below
    # so a cold start does not download 90 MB before serving its first request.
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    # Container filesystems are ephemeral; a model download on every boot is a
    # slow start and an outbound dependency at the worst possible moment.
    HF_HUB_OFFLINE=0

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

# Torch from the CPU index, before anything can pull the default wheel.
#
# This matters more than it looks. `pip install '.[models]'` resolves torch from
# PyPI, which is the CUDA build: ~2.5 GB of GPU runtime in an image that will
# never see a GPU. The CPU wheel is roughly a tenth of that, and the embedder
# and cross-encoder both run on CPU by design.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install ".[serving,obs,models]"

# `.[models]` is not optional for a deployed instance, which is what the previous
# version of this file got wrong. Queries are embedded at request time with
# MiniLM-L6-v2 (384-d, matching the indexed vectors), so without
# sentence-transformers every /search in dense or hybrid mode raises ImportError
# while /health and /ready both reported fine. /ready now embeds a probe string.

# Pre-download the embedding and reranking models into the image.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Non-root. The process needs no write access to anything but /tmp.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app /opt/hf
USER app

EXPOSE 8000

# Readiness, not liveness: this checks the database, the index and the embedder,
# so an image built without the models extra fails here instead of in production.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "edgar_intel.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
