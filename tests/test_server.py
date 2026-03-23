"""Tests for server.py — all HTTP endpoints."""

import base64
import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from PIL import Image

# conftest.py has already mocked torch + easyocr before these imports
import server
from server import app
from ocr_engine import BBox, Detection, OCRResult
from config import Settings, get_settings


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_png(w: int = 10, h: int = 10) -> bytes:
    """Return a minimal RGB PNG as raw bytes (no disk I/O)."""
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_result(text: str = "OK", count: int = 1) -> OCRResult:
    dets = [
        Detection(
            text=text,
            confidence=0.95,
            bbox=BBox(
                x1=10,
                y1=10,
                x2=110,
                y2=40,
                center_x=60,
                center_y=25,
                width=100,
                height=30,
            ),
        )
        for _ in range(count)
    ]
    return OCRResult(
        detections=dets,
        processing_time_ms=12.3,
        device="cpu",
        image_w=200,
        image_h=100,
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def engine(monkeypatch):
    """Inject a mock OCR engine so tests never need real inference."""
    mock = MagicMock()
    mock.device = "cpu"
    mock.device_info = {"device": "cpu"}
    mock.run.return_value = _make_result()
    mock.find.return_value = _make_result()
    monkeypatch.setattr(server, "_engine", mock)
    return mock


@pytest.fixture()
def client():
    """TestClient without lifespan — engine is injected separately."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def authed_client(monkeypatch):
    """Client + settings override requiring X-Api-Key: testkey."""

    def _settings():
        s = Settings()
        s.api_key = "testkey"
        return s

    app.dependency_overrides[get_settings] = _settings
    c = TestClient(app, raise_server_exceptions=True)
    yield c
    app.dependency_overrides.clear()


# ── root / redirect ───────────────────────────────────────────────────────────


def test_root_redirects_to_ui(client, engine):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/ui/"


# ── meta endpoints ────────────────────────────────────────────────────────────


def test_health_returns_ok(client, engine):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_info_returns_device(client, engine):
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["device"] == "cpu"


def test_robots_txt_disallows_all(client, engine):
    r = client.get("/robots.txt")
    assert r.status_code == 200
    assert "Disallow: /" in r.text
    assert "User-agent: *" in r.text


def test_spec_returns_yaml(client, engine):
    r = client.get("/spec")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/yaml")
    assert r.text.startswith("openapi:")


def test_llms_txt_returns_text(client, engine):
    r = client.get("/llms.txt")
    assert r.status_code == 200
    assert "semantic-lens" in r.text.lower() or "ocr" in r.text.lower()


# ── auth ──────────────────────────────────────────────────────────────────────


def test_health_no_auth_when_disabled(client, engine):
    """When API_KEY is empty auth is disabled — no key required."""
    r = client.get("/health")
    assert r.status_code == 200


def test_health_correct_key(authed_client, engine):
    r = authed_client.get("/health", headers={"X-Api-Key": "testkey"})
    assert r.status_code == 200


def test_health_wrong_key_rejected(authed_client, engine):
    r = authed_client.get("/health", headers={"X-Api-Key": "wrongkey"})
    assert r.status_code == 401


def test_health_missing_key_rejected(authed_client, engine):
    r = authed_client.get("/health")
    assert r.status_code == 401


def test_ocr_upload_requires_key(authed_client, engine):
    r = authed_client.post(
        "/ocr/upload", files={"image": ("x.png", _make_png(), "image/png")}
    )
    assert r.status_code == 401


# ── POST /ocr/upload ──────────────────────────────────────────────────────────


def test_ocr_upload_returns_detections(client, engine):
    r = client.post(
        "/ocr/upload",
        files={"image": ("screen.png", _make_png(), "image/png")},
        data={"confidence_threshold": "0.4"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["detections"][0]["text"] == "OK"


def test_ocr_upload_response_has_adb_fields(client, engine):
    r = client.post("/ocr/upload", files={"image": ("x.png", _make_png(), "image/png")})
    assert r.status_code == 200
    det = r.json()["detections"][0]
    assert "tap_x" in det
    assert "tap_y" in det
    assert det["tap_x"] == 60
    assert det["tap_y"] == 25


def test_ocr_upload_response_has_bbox(client, engine):
    r = client.post("/ocr/upload", files={"image": ("x.png", _make_png(), "image/png")})
    bbox = r.json()["detections"][0]["bbox"]
    for field in ("x1", "y1", "x2", "y2", "center_x", "center_y", "width", "height"):
        assert field in bbox


def test_ocr_upload_response_has_meta(client, engine):
    r = client.post("/ocr/upload", files={"image": ("x.png", _make_png(), "image/png")})
    body = r.json()
    assert "processing_time_ms" in body
    assert "device" in body
    assert "image_w" in body
    assert "image_h" in body


def test_ocr_upload_too_large_rejected(client, engine, monkeypatch):
    def _small_limit():
        s = Settings()
        s.max_image_bytes = 5  # 5 bytes
        return s

    app.dependency_overrides[get_settings] = _small_limit
    r = client.post("/ocr/upload", files={"image": ("x.png", _make_png(), "image/png")})
    app.dependency_overrides.clear()
    assert r.status_code == 413


# ── POST /ocr (JSON / base64) ─────────────────────────────────────────────────


def test_ocr_json_returns_detections(client, engine):
    r = client.post(
        "/ocr",
        json={"image": _b64(_make_png()), "confidence_threshold": 0.4},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_ocr_json_strips_data_uri_prefix(client, engine):
    b64 = "data:image/png;base64," + _b64(_make_png())
    r = client.post("/ocr", json={"image": b64})
    assert r.status_code == 200


# ── POST /ocr/find/upload ─────────────────────────────────────────────────────


def test_find_upload_returns_detections(client, engine):
    r = client.post(
        "/ocr/find/upload",
        files={"image": ("x.png", _make_png(), "image/png")},
        data={"query": "OK", "confidence_threshold": "0.4", "case_sensitive": "false"},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_find_upload_calls_find_method(client, engine):
    client.post(
        "/ocr/find/upload",
        files={"image": ("x.png", _make_png(), "image/png")},
        data={"query": "Battle"},
    )
    engine.find.assert_called_once()
    args = engine.find.call_args
    assert args[0][1] == "Battle"  # second positional arg is the query


# ── POST /ocr/find (JSON) ─────────────────────────────────────────────────────


def test_find_json_returns_detections(client, engine):
    r = client.post(
        "/ocr/find",
        json={"image": _b64(_make_png()), "query": "OK"},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_find_json_case_sensitive_passed(client, engine):
    client.post(
        "/ocr/find",
        json={"image": _b64(_make_png()), "query": "OK", "case_sensitive": True},
    )
    _, kwargs = engine.find.call_args
    assert kwargs.get("case_sensitive") is True or engine.find.call_args[0][3] is True
