# tests/conftest.py
# Mock torch and easyocr before any project imports.
# This allows all tests to run in CI without installing GPU-specific packages.

import sys
from unittest.mock import MagicMock

# ── torch mock ────────────────────────────────────────────────────────────────
_torch = MagicMock()
_torch.cuda.is_available.return_value = False  # force CPU path in all tests
_torch.cuda.get_device_name.return_value = "MockGPU"
_torch.cuda.get_device_capability.return_value = (8, 6)
_torch.cuda.get_device_properties.return_value = MagicMock(total_memory=8 * 1024**3)
sys.modules["torch"] = _torch

# ── easyocr mock ──────────────────────────────────────────────────────────────
_easyocr = MagicMock()
sys.modules["easyocr"] = _easyocr
sys.modules["easyocr.utils"] = MagicMock()
