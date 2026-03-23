#!/usr/bin/env bash
# start.sh — launch the OCR server, optionally expose via ngrok or cloudflared
# Usage: ./start.sh [port]

set -e

# ── Load .env first (exports all vars into the shell environment) ────────────

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "[start] Created .env from .env.example — edit it to configure your tunnel."
fi

set -a
# shellcheck source=.env
source .env
set +a

# Command-line port overrides .env PORT
PORT=${1:-${PORT:-8000}}

# ── Virtualenv ───────────────────────────────────────────────────────────────

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# ── Tunnel selection ─────────────────────────────────────────────────────────

echo ""
echo "  Expose server publicly?"
echo "  [1] ngrok"
echo "  [2] cloudflared"
echo "  [3] No — local only"
echo ""
read -rp "  Choice [1/2/3]: " TUNNEL_CHOICE

case "$TUNNEL_CHOICE" in
    1)
        if ! command -v ngrok &>/dev/null; then
            echo "[start] ERROR: ngrok not found."
            echo "        Install: https://ngrok.com/download"
            exit 1
        fi
        TUNNEL="ngrok"
        ;;
    2)
        if ! command -v cloudflared &>/dev/null; then
            echo "[start] ERROR: cloudflared not found."
            echo "        Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
            exit 1
        fi
        TUNNEL="cloudflared"
        ;;
    *)
        TUNNEL="none"
        ;;
esac

# ── Start server in background ───────────────────────────────────────────────

echo ""
echo "[start] Starting OCR server on port $PORT ..."
python server.py &
SERVER_PID=$!

cleanup() {
    echo ""
    echo "[start] Shutting down server (PID $SERVER_PID) ..."
    kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for the health endpoint to respond — model loading can take a while
# on first run (model download). Polls every 2s for up to 2 minutes.
echo -n "[start] Waiting for server to be ready"
READY=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "[start] ERROR: server process exited unexpectedly. Check output above."
        exit 1
    fi
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo " ready!"
        READY=1
        break
    fi
    echo -n "."
    sleep 2
done

if [ "$READY" -eq 0 ]; then
    echo ""
    echo "[start] ERROR: server did not become ready within 120s."
    exit 1
fi

# ── Launch tunnel (foreground — prints the public URL) ───────────────────────

case "$TUNNEL" in
    ngrok)
        if [ -n "$NGROK_DOMAIN" ]; then
            # Strip any https:// or http:// prefix — ngrok --domain wants hostname only
            NGROK_HOST="${NGROK_DOMAIN#https://}"
            NGROK_HOST="${NGROK_HOST#http://}"
            echo "[start] ngrok stable URL: https://$NGROK_HOST"
            ngrok http --domain="$NGROK_HOST" "$PORT"
        else
            echo "[start] WARNING: NGROK_DOMAIN not set — URL will change on every restart."
            echo "[start]          Get your free static domain: https://dashboard.ngrok.com/domains"
            echo "[start]          Then set NGROK_DOMAIN=your-name.ngrok-free.app in .env"
            echo ""
            ngrok http "$PORT"
        fi
        ;;
    cloudflared)
        if [ -n "$CF_TUNNEL" ]; then
            echo "[start] cloudflared named tunnel: $CF_TUNNEL (stable URL)"
            cloudflared tunnel run "$CF_TUNNEL"
        else
            echo "[start] WARNING: CF_TUNNEL not set — URL will change on every restart."
            echo "[start]          One-time setup for a stable URL:"
            echo "[start]            cloudflared tunnel login"
            echo "[start]            cloudflared tunnel create semantic-lens"
            echo "[start]          Then set CF_TUNNEL=semantic-lens in .env"
            echo ""
            cloudflared tunnel --url "http://localhost:$PORT"
        fi
        ;;
    none)
        echo "[start] Running locally at http://localhost:$PORT"
        echo "[start] Press Ctrl+C to stop."
        wait "$SERVER_PID"
        ;;
esac
