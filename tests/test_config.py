"""Tests for config.py — Settings and helpers."""
import os
import pytest
from config import Settings


def test_default_host():
    s = Settings()
    assert s.host == "0.0.0.0"


def test_default_port():
    s = Settings()
    assert s.port == 8000


def test_default_gpu_true():
    s = Settings()
    assert s.gpu is True


def test_default_api_key_empty():
    s = Settings()
    assert s.api_key == ""


def test_default_confidence_threshold():
    s = Settings()
    assert s.confidence_threshold == pytest.approx(0.4)


def test_default_max_image_bytes():
    s = Settings()
    assert s.max_image_bytes == 10 * 1024 * 1024


def test_language_list_single():
    s = Settings(languages="en")
    assert s.language_list == ["en"]


def test_language_list_multiple():
    s = Settings(languages="en,ch_sim")
    assert s.language_list == ["en", "ch_sim"]


def test_language_list_strips_spaces():
    s = Settings(languages="en, ch_sim , ja")
    assert s.language_list == ["en", "ch_sim", "ja"]


def test_language_list_ignores_empty_segments():
    s = Settings(languages="en,,ja")
    assert s.language_list == ["en", "ja"]


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    s = Settings()
    assert s.port == 9999


def test_ngrok_domain_default_empty(monkeypatch):
    monkeypatch.delenv("NGROK_DOMAIN", raising=False)
    s = Settings(_env_file=None)
    assert s.ngrok_domain == ""


def test_cf_tunnel_default_empty(monkeypatch):
    monkeypatch.delenv("CF_TUNNEL", raising=False)
    s = Settings(_env_file=None)
    assert s.cf_tunnel == ""


def test_extra_env_vars_ignored(monkeypatch):
    """Pydantic should accept only declared fields — no crash on unknown vars."""
    monkeypatch.setenv("SOME_RANDOM_VAR", "value")
    # Should not raise
    Settings()
