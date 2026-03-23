"""
OCR engine wrapping EasyOCR with CUDA (Blackwell-ready) / CPU fallback.
Model is loaded once at startup and reused for every request.
"""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import easyocr
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Resumable model downloader — replaces EasyOCR's urllib-based downloader
# which has no resume support and fails on large files over slow connections.
# ---------------------------------------------------------------------------


def _resumable_download_and_unzip(
    url: str, filename: str, model_storage_directory: str, verbose: bool = True
) -> None:
    model_dir = Path(model_storage_directory)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / filename
    if target.exists():
        return  # already have it

    zip_path = model_dir / "temp.zip"

    if shutil.which("wget"):
        # -c  : resume if partial file exists
        # -nv : print URL + result only (not verbose mode for cleaner server logs)
        cmd = ["wget", "-c", "-O", str(zip_path), url]
        if not verbose:
            cmd = ["wget", "-c", "-nv", "-O", str(zip_path), url]
    elif shutil.which("curl"):
        # -L  : follow redirects   -C - : resume automatically
        cmd = ["curl", "-L", "-C", "-", "-o", str(zip_path), url]
        if not verbose:
            cmd = ["curl", "-L", "-C", "-", "-s", "-o", str(zip_path), url]
    else:
        raise RuntimeError(
            "wget and curl not found. Install either to enable model downloads:\n"
            "  sudo dnf install wget   (Fedora/RHEL)\n"
            "  sudo apt install wget   (Debian/Ubuntu)"
        )

    print(f"[OCR] Downloading {filename} ...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed (exit {result.returncode}): {url}")

    print(f"[OCR] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir)
    zip_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int
    width: int
    height: int


@dataclass
class Detection:
    text: str
    confidence: float
    bbox: BBox


@dataclass
class OCRResult:
    detections: list[Detection]
    processing_time_ms: float
    device: str
    image_w: int
    image_h: int


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OCREngine:
    def __init__(self, languages: list[str], prefer_gpu: bool = True):
        self.device, self._use_gpu = self._detect_device(prefer_gpu)
        print(f"[OCR] Using device: {self.device}")
        print(
            f"[OCR] Loading EasyOCR ({', '.join(languages)}) — this may take a moment on first run ..."
        )
        self.reader = self._load_reader(languages)
        print("[OCR] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.4,
    ) -> OCRResult:
        h, w = image.shape[:2]
        t0 = time.perf_counter()
        raw = self.reader.readtext(image, detail=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        for bbox_pts, text, conf in raw:
            if conf < confidence_threshold:
                continue
            xs = [int(p[0]) for p in bbox_pts]
            ys = [int(p[1]) for p in bbox_pts]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            detections.append(
                Detection(
                    text=text,
                    confidence=round(float(conf), 4),
                    bbox=BBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        center_x=(x1 + x2) // 2,
                        center_y=(y1 + y2) // 2,
                        width=x2 - x1,
                        height=y2 - y1,
                    ),
                )
            )

        return OCRResult(
            detections=detections,
            processing_time_ms=round(elapsed_ms, 2),
            device=self.device,
            image_w=w,
            image_h=h,
        )

    def find(
        self,
        image: np.ndarray,
        query: str,
        confidence_threshold: float = 0.4,
        case_sensitive: bool = False,
    ) -> OCRResult:
        """Return only detections whose text contains *query*."""
        result = self.run(image, confidence_threshold)
        needle = query if case_sensitive else query.lower()
        result.detections = [
            d
            for d in result.detections
            if needle in (d.text if case_sensitive else d.text.lower())
        ]
        return result

    # ------------------------------------------------------------------
    # Image decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def from_bytes(data: bytes) -> np.ndarray:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    @staticmethod
    def from_base64(b64: str) -> np.ndarray:
        # Strip optional data-URI prefix: "data:image/png;base64,..."
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        return OCREngine.from_bytes(base64.b64decode(b64))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_reader(self, languages: list[str]) -> easyocr.Reader:
        # Swap EasyOCR's urllib downloader for our wget/curl-based one which
        # supports resume (-c / -C -) so interrupted downloads pick up where
        # they left off instead of failing after 70+ MB.
        import easyocr.utils as _eu

        _eu.download_and_unzip = _resumable_download_and_unzip
        return easyocr.Reader(languages, gpu=self._use_gpu, verbose=False)

    @staticmethod
    def _detect_device(prefer_gpu: bool) -> tuple[str, bool]:
        if prefer_gpu and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            print(f"[OCR] GPU: {name}  (SM {major}.{minor})")
            if major >= 10:
                print(
                    "[OCR] Blackwell architecture detected — ensure CUDA 12.8+ and PyTorch 2.7+"
                )
            return "cuda:0", True
        if prefer_gpu:
            print("[OCR] CUDA not available — falling back to CPU")
        return "cpu", False

    @property
    def device_info(self) -> dict:
        info: dict = {"device": self.device}
        if self._use_gpu:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            info["compute_capability"] = f"{major}.{minor}"
            info["vram_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2
            )
        return info
