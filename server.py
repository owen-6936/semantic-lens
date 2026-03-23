"""
OCR HTTP server — FastAPI
Supports JSON (base64) and multipart (raw bytes) image uploads.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import uvicorn
import yaml
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Settings, get_settings
from ocr_engine import OCREngine, OCRResult

# ---------------------------------------------------------------------------
# App lifecycle — load model once
# ---------------------------------------------------------------------------

_engine: OCREngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    cfg = get_settings()
    _engine = OCREngine(languages=cfg.language_list, prefer_gpu=cfg.gpu)
    yield
    _engine = None


app = FastAPI(
    title="OCR Server",
    description="Low-latency OCR with CUDA (Blackwell-ready) / CPU fallback. Returns text + x,y coords for ADB.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the web UI at /ui  (no auth — UI handles the key itself via JS)
_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/ui", StaticFiles(directory=_static, html=True), name="ui")

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui/")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

def require_api_key(
    x_api_key: Annotated[Optional[str], Header()] = None,
    settings: Settings = Depends(get_settings),
):
    if not settings.api_key:
        return  # auth disabled
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


AuthDep = Annotated[None, Depends(require_api_key)]


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class BBoxOut(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    width: int
    height: int


class DetectionOut(BaseModel):
    text: str
    confidence: float
    bbox: BBoxOut

    # Shorthand for ADB: `adb shell input tap <tap_x> <tap_y>`
    tap_x: int
    tap_y: int


class OCRResponse(BaseModel):
    detections: list[DetectionOut]
    count: int
    processing_time_ms: float
    device: str
    image_w: int
    image_h: int


def _to_response(result: OCRResult) -> OCRResponse:
    dets = [
        DetectionOut(
            text=d.text,
            confidence=d.confidence,
            bbox=BBoxOut(**d.bbox.__dict__),
            tap_x=d.bbox.center_x,
            tap_y=d.bbox.center_y,
        )
        for d in result.detections
    ]
    return OCRResponse(
        detections=dets,
        count=len(dets),
        processing_time_ms=result.processing_time_ms,
        device=result.device,
        image_w=result.image_w,
        image_h=result.image_h,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_size(data: bytes, max_bytes: int):
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds {max_bytes // 1024 // 1024} MB limit",
        )


# ---------------------------------------------------------------------------
# JSON body schemas
# ---------------------------------------------------------------------------

class OCRJsonRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image (with or without data-URI prefix)")
    confidence_threshold: float = Field(0.4, ge=0.0, le=1.0)


class FindJsonRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    query: str = Field(..., description="Text to search for (substring match)")
    confidence_threshold: float = Field(0.4, ge=0.0, le=1.0)
    case_sensitive: bool = False


# ---------------------------------------------------------------------------
# Endpoints — info / health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health(_: AuthDep):
    return {"status": "ok", "device": _engine.device if _engine else "not loaded"}


@app.get("/info", tags=["meta"])
async def info(_: AuthDep):
    if _engine is None:
        raise HTTPException(500, "Engine not initialised")
    return _engine.device_info


@app.get("/robots.txt", include_in_schema=False)
async def robots():
    """Block all crawlers — this is a private API."""
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


@app.get("/llms.txt", include_in_schema=False)
async def llms_txt():
    """LLM-readable API reference."""
    f = Path(__file__).parent / "llms.txt"
    return FileResponse(f, media_type="text/plain")


@app.get("/spec", include_in_schema=False)
async def openapi_yaml():
    """OpenAPI spec as YAML (JSON is at /openapi.json, UI at /docs)."""
    spec = app.openapi()
    return Response(
        content=yaml.dump(spec, sort_keys=False, allow_unicode=True),
        media_type="application/yaml",
    )


# ---------------------------------------------------------------------------
# Endpoints — multipart (fastest, no base64 overhead)
# ---------------------------------------------------------------------------

@app.post("/ocr/upload", response_model=OCRResponse, tags=["ocr"])
async def ocr_upload(
    _: AuthDep,
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.4),
    settings: Settings = Depends(get_settings),
):
    """Full OCR via multipart file upload — lowest latency option."""
    data = await image.read()
    _check_size(data, settings.max_image_bytes)
    img = OCREngine.from_bytes(data)
    result = _engine.run(img, confidence_threshold)
    return _to_response(result)


@app.post("/ocr/find/upload", response_model=OCRResponse, tags=["ocr"])
async def find_upload(
    _: AuthDep,
    image: UploadFile = File(...),
    query: str = Form(...),
    confidence_threshold: float = Form(0.4),
    case_sensitive: bool = Form(False),
    settings: Settings = Depends(get_settings),
):
    """Find specific text in image and return its coordinates. Upload variant."""
    data = await image.read()
    _check_size(data, settings.max_image_bytes)
    img = OCREngine.from_bytes(data)
    result = _engine.find(img, query, confidence_threshold, case_sensitive)
    return _to_response(result)


# ---------------------------------------------------------------------------
# Endpoints — JSON / base64 (handy for scripting)
# ---------------------------------------------------------------------------

@app.post("/ocr", response_model=OCRResponse, tags=["ocr"])
async def ocr_json(
    body: OCRJsonRequest,
    _: AuthDep,
    settings: Settings = Depends(get_settings),
):
    """Full OCR via JSON body with base64 image."""
    raw = body.image
    if "," in raw:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    _check_size(data, settings.max_image_bytes)
    img = OCREngine.from_bytes(data)
    result = _engine.run(img, body.confidence_threshold)
    return _to_response(result)


@app.post("/ocr/find", response_model=OCRResponse, tags=["ocr"])
async def find_json(
    body: FindJsonRequest,
    _: AuthDep,
    settings: Settings = Depends(get_settings),
):
    """Find specific text, return coordinates for ADB tap. JSON variant."""
    raw = body.image
    if "," in raw:
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    _check_size(data, settings.max_image_bytes)
    img = OCREngine.from_bytes(data)
    result = _engine.find(img, body.query, body.confidence_threshold, body.case_sensitive)
    return _to_response(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "server:app",
        host=cfg.host,
        port=cfg.port,
        workers=cfg.workers,
        log_level="info",
    )
