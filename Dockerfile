# ── Base image ────────────────────────────────────────────────────────────────
# Default: CUDA 12.4 — stable, supports RTX 30xx / 40xx (Ampere / Ada).
#
# For Blackwell (RTX 50xx) swap to:
#   FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# and change the torch install line to use --index-url .../nightly/cu128
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Keeps EasyOCR model cache inside the container's home dir
# (mount a volume here to persist across rebuilds)
ENV EASYOCR_MODULE_PATH=/root/.EasyOCR

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        wget curl \
    && rm -rf /var/lib/apt/lists/*

# Use python3.11 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# ── PyTorch (CUDA 12.4 / RTX 30-40xx) ────────────────────────────────────────
# Installed before requirements.txt so Docker cache layer is reused
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu124

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source ────────────────────────────────────────────────────────────────────
COPY config.py ocr_engine.py server.py llms.txt openapi.yaml ./

EXPOSE 8000

# Note: start.sh is NOT used in Docker (it is interactive / tunnel-focused).
# The tunnel (ngrok / cloudflared) runs on the HOST pointing to this port.
CMD ["python", "server.py"]
