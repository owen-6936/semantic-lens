# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [1.1.0] - 2026-03-23

### Added

- Web UI (`/ui`) — dark-themed OCR tester with drag-and-drop image upload,
  canvas bounding-box overlay, color-keyed detection cards, and one-click
  ADB tap command copy
- `GET /` redirects to `/ui/`
- `install.sh` — interactive native installer for Linux/macOS with GPU
  variant selection (Blackwell / Ada-Ampere / older NVIDIA / CPU)
- `Dockerfile` — CUDA 12.4 base image; swap to `cu128` nightly for Blackwell
- `compose.yml` — Docker Compose with NVIDIA GPU passthrough and EasyOCR
  model volume cache
- `CHANGELOG.md` (this file)
- `.github/workflows/ci.yml` — lint, syntax, YAML validation, Dockerfile lint,
  and pytest matrix (Python 3.11 + 3.12)
- `tests/` — 49 pytest tests covering `config.py`, `ocr_engine.py`, and all
  HTTP endpoints; torch/easyocr mocked so CI never needs a GPU build
- `requirements-test.txt` — lightweight test dependencies (no torch/easyocr)
- `.github/workflows/codeql.yml` — weekly CodeQL security scan
- Stable tunnel URLs: `NGROK_DOMAIN` and `CF_TUNNEL` in `.env` persist the
  public address across restarts
- `GET /llms.txt` endpoint — serves `llms.txt` to LLM clients
- `GET /spec` endpoint — serves OpenAPI spec as YAML

### Fixed

- Model download failure (`ContentTooShortError`) — replaced EasyOCR's
  `urllib.urlretrieve` with a `wget -c` / `curl -C -` resumable downloader
  that picks up interrupted downloads instead of failing after 70+ MB
- `NGROK_DOMAIN` with `https://` prefix caused pydantic validation error —
  added field to `Settings` and strip protocol prefix in `start.sh`
- `start.sh` used `sleep 2` before launching tunnel — replaced with a
  `/health` poll loop (2 s intervals, 2 min timeout) for reliability

### Changed

- `start.sh` now sources `.env` before prompting, so `PORT` from `.env`
  is respected and tunnel vars are available
- `start.sh` warns explicitly when tunnel vars are unset and explains
  one-time setup steps

---

## [1.0.0] - 2026-03-23

### Added

- FastAPI OCR server with EasyOCR backend (PyTorch-based)
- Auto CUDA / CPU device detection — Blackwell (SM 10.x) supported via
  PyTorch 2.7+ / CUDA 12.8 nightly
- `POST /ocr/upload` and `POST /ocr` — full OCR, multipart and JSON variants
- `POST /ocr/find/upload` and `POST /ocr/find` — substring search returning
  only matching detections
- `GET /health` and `GET /info` — server status and GPU details
- `GET /robots.txt` — disallows all crawlers
- `openapi.yaml` — static OpenAPI 3.1 spec
- `llms.txt` — LLM-readable API reference
- `config.py` — pydantic-settings config from `.env`
- `ocr_engine.py` — `OCREngine` class with `run()` and `find()` methods
- ADB-ready `tap_x` / `tap_y` on every detection (bbox center coordinates)
- Optional `X-Api-Key` authentication header
- `start.sh` — interactive launch script with ngrok / cloudflared prompt
- `requirements.txt`, `.env.example`, `README.md`
