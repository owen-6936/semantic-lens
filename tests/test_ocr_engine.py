"""Tests for ocr_engine.py — dataclasses and image decode helpers.

OCREngine.__init__ is NOT tested here (requires real EasyOCR/torch).
We test the pure-Python components that run in CI without GPU deps.
"""

import base64
import io
import numpy as np
import pytest
from PIL import Image

# conftest.py mocks torch + easyocr before this import
from ocr_engine import BBox, Detection, OCRResult, OCREngine


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_png(w: int = 20, h: int = 10, color=(128, 64, 32)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── BBox ─────────────────────────────────────────────────────────────────────


def test_bbox_fields():
    b = BBox(
        x1=10, y1=20, x2=110, y2=50, center_x=60, center_y=35, width=100, height=30
    )
    assert b.x1 == 10
    assert b.y1 == 20
    assert b.x2 == 110
    assert b.y2 == 50
    assert b.center_x == 60
    assert b.center_y == 35
    assert b.width == 100
    assert b.height == 30


def test_bbox_center_is_midpoint():
    b = BBox(x1=0, y1=0, x2=100, y2=40, center_x=50, center_y=20, width=100, height=40)
    assert b.center_x == (b.x1 + b.x2) // 2
    assert b.center_y == (b.y1 + b.y2) // 2


# ── Detection ─────────────────────────────────────────────────────────────────


def test_detection_fields():
    bbox = BBox(x1=0, y1=0, x2=50, y2=20, center_x=25, center_y=10, width=50, height=20)
    d = Detection(text="Hello", confidence=0.9, bbox=bbox)
    assert d.text == "Hello"
    assert d.confidence == pytest.approx(0.9)
    assert d.bbox is bbox


def test_detection_confidence_range():
    bbox = BBox(x1=0, y1=0, x2=1, y2=1, center_x=0, center_y=0, width=1, height=1)
    d = Detection(text="x", confidence=0.9999, bbox=bbox)
    assert 0.0 <= d.confidence <= 1.0


# ── OCRResult ─────────────────────────────────────────────────────────────────


def test_ocr_result_fields():
    result = OCRResult(
        detections=[],
        processing_time_ms=42.0,
        device="cpu",
        image_w=640,
        image_h=480,
    )
    assert result.detections == []
    assert result.processing_time_ms == pytest.approx(42.0)
    assert result.device == "cpu"
    assert result.image_w == 640
    assert result.image_h == 480


def test_ocr_result_mutable_detections():
    result = OCRResult(
        detections=[], processing_time_ms=0, device="cpu", image_w=1, image_h=1
    )
    bbox = BBox(x1=0, y1=0, x2=1, y2=1, center_x=0, center_y=0, width=1, height=1)
    result.detections.append(Detection(text="A", confidence=0.8, bbox=bbox))
    assert len(result.detections) == 1


# ── OCREngine.from_bytes ──────────────────────────────────────────────────────


def test_from_bytes_returns_ndarray():
    arr = OCREngine.from_bytes(_make_png())
    assert isinstance(arr, np.ndarray)


def test_from_bytes_shape():
    arr = OCREngine.from_bytes(_make_png(w=20, h=10))
    assert arr.shape == (10, 20, 3)  # H × W × C


def test_from_bytes_dtype():
    arr = OCREngine.from_bytes(_make_png())
    assert arr.dtype == np.uint8


def test_from_bytes_color_preserved():
    arr = OCREngine.from_bytes(_make_png(w=1, h=1, color=(255, 0, 128)))
    assert arr[0, 0, 0] == 255
    assert arr[0, 0, 1] == 0
    assert arr[0, 0, 2] == 128


# ── OCREngine.from_base64 ─────────────────────────────────────────────────────


def test_from_base64_plain():
    b64 = base64.b64encode(_make_png(w=4, h=4)).decode()
    arr = OCREngine.from_base64(b64)
    assert arr.shape == (4, 4, 3)


def test_from_base64_with_data_uri_prefix():
    raw = _make_png(w=4, h=4)
    b64 = "data:image/png;base64," + base64.b64encode(raw).decode()
    arr = OCREngine.from_base64(b64)
    assert arr.shape == (4, 4, 3)


def test_from_base64_wrong_prefix_stripped():
    """Any prefix before the comma is stripped regardless of mimetype."""
    raw = _make_png(w=2, h=2)
    b64 = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    arr = OCREngine.from_base64(b64)
    assert isinstance(arr, np.ndarray)
