# semantic-lens

Low-latency OCR HTTP server for gaming automation. Returns text detections with bounding-box coordinates ready for `adb shell input tap`.

- **GPU-first** — CUDA with automatic CPU fallback
- **Blackwell-ready** — SM 10.x support via PyTorch 2.7+ / CUDA 12.8
- **ADB-ready output** — every detection includes `tap_x` / `tap_y` center coordinates
- **Two input modes** — multipart upload (fastest) or JSON + base64

---

## Requirements

- Python 3.11+
- PyTorch with the correct CUDA variant for your GPU (see below)
- `ngrok` or `cloudflared` if you want public access (optional)

---

## Installation

```bash
git clone <repo>
cd semantic-lens

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 1. Install PyTorch — pick the right variant for your GPU:

# Blackwell (RTX 50xx) — CUDA 12.8, PyTorch 2.7 nightly
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Ada / Ampere (RTX 30/40xx) — CUDA 12.4, stable
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install everything else
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
$EDITOR .env
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `WORKERS` | `1` | Uvicorn workers (keep 1 — model is not fork-safe) |
| `LANGUAGES` | `en` | Comma-separated EasyOCR language codes e.g. `en,ch_sim` |
| `GPU` | `true` | Attempt CUDA; falls back to CPU automatically |
| `API_KEY` | *(empty)* | Auth key sent in `X-Api-Key` header. Leave empty to disable. |
| `CONFIDENCE_THRESHOLD` | `0.4` | Default minimum confidence (0–1) |
| `MAX_IMAGE_BYTES` | `10485760` | Max upload size (bytes) — default 10 MB |

---

## Running

```bash
./start.sh          # prompts for ngrok / cloudflared / local
./start.sh 9000     # custom port
```

The startup prompt:

```
  Expose server publicly?
  [1] ngrok
  [2] cloudflared
  [3] No — local only
```

Choosing **1** or **2** starts the server in the background and the tunnel in the foreground. The public URL is printed by the tunnel tool. Ctrl+C stops both.

The script polls `/health` in a 2s loop (up to 2 min) before launching the tunnel, so the tunnel is only opened once the model is fully loaded and the server is accepting requests.

### Stable public URLs

Without extra config both tunnel tools assign a **random URL on every restart**. To get a fixed URL:

**ngrok** (free, easiest):

1. Sign up at ngrok.com → grab your authtoken

   ```bash
   ngrok config add-authtoken <your-token>
   ```

2. Claim your one free static domain at `https://dashboard.ngrok.com/domains`
3. Set it in `.env`:

   ```
   NGROK_DOMAIN=your-name.ngrok-free.app
   ```

**cloudflared** (free, no account needed for the tunnel itself):

```bash
cloudflared tunnel login
cloudflared tunnel create semantic-lens   # creates a stable UUID-based URL
```

Then set in `.env`:

```
CF_TUNNEL=semantic-lens
```

The stable URL is shown the first time you run the named tunnel and stays the same forever.

---

## API

Interactive docs at `/docs` (Swagger) and `/redoc`.
Raw spec: `/openapi.json` (JSON) · `/spec` (YAML).

### Authentication

If `API_KEY` is set, include it in every request:

```
X-Api-Key: your-key-here
```

### Endpoints

#### `GET /health`

Returns server status and active compute device.

```json
{ "status": "ok", "device": "cuda:0" }
```

#### `GET /info`

Returns GPU details (name, compute capability, VRAM).

```json
{
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 5090",
  "compute_capability": "10.0",
  "vram_total_mb": 32768
}
```

#### `POST /ocr/upload` *(multipart — recommended)*

Full OCR on an uploaded image file.

```bash
curl -X POST http://localhost:8000/ocr/upload \
  -H "X-Api-Key: yourkey" \
  -F "image=@screenshot.png" \
  -F "confidence_threshold=0.5"
```

#### `POST /ocr/find/upload` *(multipart — recommended)*

Find a specific text string and return its screen coordinates.

```bash
curl -X POST http://localhost:8000/ocr/find/upload \
  -H "X-Api-Key: yourkey" \
  -F "image=@screenshot.png" \
  -F "query=OK" \
  -F "confidence_threshold=0.4" \
  -F "case_sensitive=false"
```

#### `POST /ocr` *(JSON / base64)*

Full OCR with base64-encoded image in the request body.

```bash
curl -X POST http://localhost:8000/ocr \
  -H "X-Api-Key: yourkey" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64>",
    "confidence_threshold": 0.4
  }'
```

#### `POST /ocr/find` *(JSON / base64)*

Find text using base64 image.

```bash
curl -X POST http://localhost:8000/ocr/find \
  -H "X-Api-Key: yourkey" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64>",
    "query": "Start",
    "confidence_threshold": 0.4,
    "case_sensitive": false
  }'
```

### Response format

All OCR endpoints return the same shape:

```json
{
  "detections": [
    {
      "text": "OK",
      "confidence": 0.9812,
      "tap_x": 540,
      "tap_y": 1205,
      "bbox": {
        "x1": 490, "y1": 1190,
        "x2": 590, "y2": 1220,
        "center_x": 540, "center_y": 1205,
        "width": 100, "height": 30
      }
    }
  ],
  "count": 1,
  "processing_time_ms": 42.3,
  "device": "cuda:0",
  "image_w": 1080,
  "image_h": 1920
}
```

`tap_x` / `tap_y` are the center of the bounding box — use them directly with ADB:

```bash
adb shell input tap 540 1205
```

---

## ADB workflow example

```bash
# 1. Take a screenshot of your device
adb exec-out screencap -p > screen.png

# 2. Ask the OCR server where the "Battle" button is
RESULT=$(curl -s -X POST http://localhost:8000/ocr/find/upload \
  -H "X-Api-Key: yourkey" \
  -F "image=@screen.png" \
  -F "query=Battle")

# 3. Extract coordinates and tap
X=$(echo $RESULT | python3 -c "import sys,json; d=json.load(sys.stdin)['detections']; print(d[0]['tap_x']) if d else exit(1)")
Y=$(echo $RESULT | python3 -c "import sys,json; d=json.load(sys.stdin)['detections']; print(d[0]['tap_y']) if d else exit(1)")
adb shell input tap $X $Y
```

---

## Supported languages

EasyOCR supports 80+ languages. Pass a comma-separated list in `LANGUAGES`:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `en` | English | `ch_sim` | Chinese Simplified |
| `ja` | Japanese | `ko` | Korean |
| `ar` | Arabic | `fr` | French |
| `de` | German | `es` | Spanish |

Full list: [EasyOCR supported languages](https://www.jaided.ai/easyocr/)

---

## Testing

The test suite runs without a GPU and without downloading any models — heavy dependencies (`torch`, `easyocr`) are mocked via `tests/conftest.py`.

```bash
# Install test dependencies (no torch required)
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

**Test coverage:**

| File | What is tested |
|------|---------------|
| `tests/test_config.py` | `Settings` defaults, env-var overrides, `language_list` parsing |
| `tests/test_ocr_engine.py` | `BBox`, `Detection`, `OCRResult` dataclasses; `from_bytes` / `from_base64` image helpers |
| `tests/test_server.py` | All HTTP endpoints, auth enforcement, 413 size limit, ADB coord presence, find-method routing |

CI runs the suite on Python 3.11 and 3.12 for every push/PR (see `.github/workflows/ci.yml`).

---

## Misc

- `GET /robots.txt` — disallows all crawlers
- `GET /llms.txt` — LLM-readable API summary
- `GET /spec` — OpenAPI spec as YAML
