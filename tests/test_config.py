"""Tests for loading settings from environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import DEFAULT_MODEL_NAME, DEFAULT_MODEL_REVISION, load_settings

ENV_KEYS = (
    "BIRD_MODEL_NAME",
    "BIRD_MODEL_REVISION",
    "BIRD_MAX_UPLOAD_MB",
    "BIRD_LOW_CONFIDENCE_THRESHOLD",
    "BIRD_MAX_IMAGE_PIXELS",
    "BIRD_HOST",
    "BIRD_PORT",
    "BIRD_LOG_LEVEL",
    "BIRD_ALLOWED_HOSTS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any BIRD_* variables inherited from the developer's shell."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.env")
    assert settings.model_name == DEFAULT_MODEL_NAME
    assert settings.model_revision == DEFAULT_MODEL_REVISION
    assert settings.max_image_pixels == 64_000_000
    assert settings.allowed_hosts == ("127.0.0.1", "localhost")
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.low_confidence_threshold == 0.5
    assert settings.host == "127.0.0.1"


def test_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIRD_MAX_UPLOAD_MB", "2")
    monkeypatch.setenv("BIRD_LOW_CONFIDENCE_THRESHOLD", "0.7")
    monkeypatch.setenv("BIRD_MODEL_REVISION", "abc123")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.max_upload_bytes == 2 * 1024 * 1024
    assert settings.low_confidence_threshold == 0.7
    assert settings.model_revision == "abc123"


def test_env_file_is_read_and_process_env_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BIRD_PORT=9001\nBIRD_HOST=0.0.0.0\n")
    monkeypatch.setenv("BIRD_HOST", "127.0.0.1")
    settings = load_settings(env_file)
    assert settings.port == 9001
    assert settings.host == "127.0.0.1"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("BIRD_LOW_CONFIDENCE_THRESHOLD", "1.5"),
        ("BIRD_LOW_CONFIDENCE_THRESHOLD", "high"),
        ("BIRD_MAX_UPLOAD_MB", "0"),
        ("BIRD_PORT", "70000"),
        ("BIRD_MAX_UPLOAD_MB", "inf"),
        ("BIRD_MAX_UPLOAD_MB", "nan"),
        ("BIRD_LOG_LEVEL", "VERBOSE"),
        ("BIRD_MAX_IMAGE_PIXELS", "1.5"),
    ],
)
def test_invalid_values_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: str
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        load_settings(tmp_path / "missing.env")


def test_blank_values_fall_back_to_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BIRD_MODEL_NAME", "   ")
    monkeypatch.setenv("BIRD_ALLOWED_HOSTS", "")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.model_name == DEFAULT_MODEL_NAME
    assert settings.allowed_hosts == ("127.0.0.1", "localhost")


def test_allowed_hosts_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIRD_ALLOWED_HOSTS", "localhost, birds.lan ,")
    settings = load_settings(tmp_path / "missing.env")
    assert settings.allowed_hosts == ("localhost", "birds.lan")
