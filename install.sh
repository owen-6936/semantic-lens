#!/usr/bin/env bash
# install.sh — native install for Linux / macOS
# Sets up a virtual environment and installs the correct PyTorch variant.
# Run once, then use ./start.sh to launch the server.

set -e

# ── OS check ─────────────────────────────────────────────────────────────────

OS="$(uname -s)"
case "$OS" in
    Linux*)  OS=linux ;;
    Darwin*) OS=macos ;;
    *)
        echo "ERROR: Unsupported OS: $OS"
        echo "       On Windows, use Docker (see compose.yml) or WSL2."
        exit 1
        ;;
esac

# ── Python check (3.11+) ─────────────────────────────────────────────────────

PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c "import sys; print(sys.version_info >= (3,11))" 2>/dev/null)
        if [ "$VER" = "True" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.11+ not found."
    if [ "$OS" = "linux" ]; then
        echo "       sudo dnf install python3.11   (Fedora/RHEL)"
        echo "       sudo apt install python3.11   (Debian/Ubuntu)"
    else
        echo "       brew install python@3.11"
    fi
    exit 1
fi

echo "[install] Using $($PYTHON --version)"

# ── Virtual environment ───────────────────────────────────────────────────────

if [ ! -d ".venv" ]; then
    echo "[install] Creating virtual environment ..."
    "$PYTHON" -m venv .venv
fi
source .venv/bin/activate

# ── GPU / PyTorch variant selection ──────────────────────────────────────────

echo ""
echo "  Select your GPU (determines PyTorch CUDA variant):"
echo ""
echo "  [1] NVIDIA Blackwell  (RTX 5090 / 5080 / 5070 etc.)  — CUDA 12.8 nightly"
echo "  [2] NVIDIA Ada / Ampere  (RTX 4090–3060 etc.)        — CUDA 12.4 stable"
echo "  [3] NVIDIA older  (RTX 2080 / 1080 etc.)             — CUDA 11.8 stable"
echo "  [4] No GPU / CPU only"
echo ""
read -rp "  Choice [1-4]: " GPU_CHOICE

case "$GPU_CHOICE" in
    1)
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
        TORCH_EXTRA="--pre"
        GPU_LABEL="Blackwell — CUDA 12.8 nightly"
        ;;
    2)
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_EXTRA=""
        GPU_LABEL="Ada/Ampere — CUDA 12.4 stable"
        ;;
    3)
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        TORCH_EXTRA=""
        GPU_LABEL="Older NVIDIA — CUDA 11.8 stable"
        ;;
    *)
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
        TORCH_EXTRA=""
        GPU_LABEL="CPU only"
        ;;
esac

echo ""
echo "[install] Installing PyTorch ($GPU_LABEL) ..."
# shellcheck disable=SC2086
pip install --quiet $TORCH_EXTRA torch torchvision --index-url "$TORCH_INDEX"

# ── Remaining dependencies ────────────────────────────────────────────────────

echo "[install] Installing server dependencies ..."
pip install --quiet -r requirements.txt

# ── .env ─────────────────────────────────────────────────────────────────────

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[install] Created .env — edit it to set your API key and tunnel config."
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "[install] Done."
echo ""
echo "  Next steps:"
echo "    1. Edit .env — set API_KEY, NGROK_DOMAIN or CF_TUNNEL"
echo "    2. Run: ./start.sh"
echo ""
