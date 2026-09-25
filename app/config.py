"""Application settings loaded from environment variables.

Values are read from the process environment first, then from an optional
``.env`` file in the current working directory. Every setting has a safe
default.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

DEFAULT_MODEL_NAME = "ozzyonfire/bird-species-classifier"
# Pinned commit of the model repository, so every install gets the same weights.
DEFAULT_MODEL_REVISION = "d4d80527be1343dacfe67af84d064ca6f9b7547e"
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")
BYTES_PER_MB = 1024 * 1024
LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration for the app."""

    model_name: str = DEFAULT_MODEL_NAME
    model_revision: str = DEFAULT_MODEL_REVISION
    max_upload_bytes: int = 10 * BYTES_PER_MB
    low_confidence_threshold: float = 0.5
    max_image_pixels: int = 64_000_000
    top_k: int = 5
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS

    def __post_init__(self) -> None:
        """Reject values that would make the app behave incorrectly."""
        if not self.model_name.strip():
            raise ValueError("BIRD_MODEL_NAME must not be empty.")
        if not self.model_revision.strip():
            raise ValueError("BIRD_MODEL_REVISION must not be empty.")
        if self.max_upload_bytes <= 0:
            raise ValueError("BIRD_MAX_UPLOAD_MB must be greater than 0.")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("BIRD_LOW_CONFIDENCE_THRESHOLD must be between 0 and 1.")
        if self.max_image_pixels <= 0:
            raise ValueError("BIRD_MAX_IMAGE_PIXELS must be greater than 0.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if not 0 < self.port < 65536:
            raise ValueError("BIRD_PORT must be between 1 and 65535.")
        if self.log_level not in LOG_LEVELS:
            raise ValueError(f"BIRD_LOG_LEVEL must be one of {', '.join(sorted(LOG_LEVELS))}.")
        if not self.allowed_hosts:
            raise ValueError("BIRD_ALLOWED_HOSTS must list at least one host name.")


def _merged_environment(env_file: Path | None) -> dict[str, str]:
    """Combine ``.env`` values with the process environment (process wins)."""
    merged: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        merged.update({k: v for k, v in dotenv_values(env_file).items() if v is not None})
    merged.update(os.environ)
    return merged


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    """Read a float setting, raising a clear error if it is malformed."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number, got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number, got {raw!r}.")
    return value


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    """Read an integer setting, raising a clear error if it is malformed."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a whole number, got {raw!r}.") from exc


def _get_str(env: Mapping[str, str], key: str, default: str) -> str:
    """Read a string setting, falling back to the default when blank."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _get_list(env: Mapping[str, str], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated list setting, falling back to the default when blank."""
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Build a :class:`Settings` object from the environment and ``.env`` file."""
    env = _merged_environment(env_file)
    max_upload_mb = _get_float(env, "BIRD_MAX_UPLOAD_MB", 10.0)
    return Settings(
        model_name=_get_str(env, "BIRD_MODEL_NAME", DEFAULT_MODEL_NAME),
        model_revision=_get_str(env, "BIRD_MODEL_REVISION", DEFAULT_MODEL_REVISION),
        max_upload_bytes=int(max_upload_mb * BYTES_PER_MB),
        low_confidence_threshold=_get_float(env, "BIRD_LOW_CONFIDENCE_THRESHOLD", 0.5),
        max_image_pixels=_get_int(env, "BIRD_MAX_IMAGE_PIXELS", 64_000_000),
        host=_get_str(env, "BIRD_HOST", "127.0.0.1"),
        port=_get_int(env, "BIRD_PORT", 8000),
        log_level=_get_str(env, "BIRD_LOG_LEVEL", "INFO").upper(),
        allowed_hosts=_get_list(env, "BIRD_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS),
    )
