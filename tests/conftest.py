"""Shared pytest fixtures: a fake classifier, app factory, and image helpers.

The fake classifier returns fixed predictions so the tests never download
or run the real model.
"""

from __future__ import annotations

import io
import os
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.classifier import Prediction
from app.config import Settings
from app.main import create_app

# Deliberately unsorted, with raw upper-case labels and six entries,
# so tests can check ordering, clean-up, and truncation to five.
DEFAULT_RAW_PREDICTIONS: tuple[Prediction, ...] = (
    Prediction(label="MOURNING DOVE", confidence=0.05),
    Prediction(label="AMERICAN GOLDFINCH", confidence=0.82),
    Prediction(label="HOUSE FINCH", confidence=0.06),
    Prediction(label="CEDAR WAXWING", confidence=0.01),
    Prediction(label="BLACK-THROATED SPARROW", confidence=0.03),
    Prediction(label="EVENING GROSBEAK", confidence=0.02),
)

# Host name that Starlette's TestClient sends in the Host header.
TEST_HOST = "testserver"


def with_test_host(settings: Settings) -> Settings:
    """Return settings that also accept the TestClient's Host header."""
    return replace(settings, allowed_hosts=(*settings.allowed_hosts, TEST_HOST))


class FakeClassifier:
    """A stand-in for the real model that returns fixed predictions."""

    def __init__(self, predictions: tuple[Prediction, ...] = DEFAULT_RAW_PREDICTIONS) -> None:
        """Store the predictions to return and track calls."""
        self.predictions = predictions
        self.calls = 0

    def predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        """Return the fixed predictions regardless of the image."""
        assert image.mode == "RGB"
        self.calls += 1
        return list(self.predictions)


class FailingClassifier:
    """A classifier whose inference always raises."""

    def predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        """Raise an internal error that must never reach the user."""
        raise RuntimeError("secret internal failure /path/to/weights.bin")


def make_image_bytes(
    image_format: str = "JPEG",
    size: tuple[int, int] = (64, 48),
    mode: str = "RGB",
    noise: bool = False,
) -> bytes:
    """Create an in-memory image in the requested format."""
    if noise:
        image = Image.frombytes(mode, size, os.urandom(size[0] * size[1] * len(mode)))
    else:
        image = Image.new(mode, size, color=(200, 150, 50) if mode == "RGB" else None)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def test_settings() -> Settings:
    """Settings with defaults suitable for tests."""
    return Settings()


@pytest.fixture
def make_client() -> Iterator[Callable[..., TestClient]]:
    """Factory that builds a started TestClient with a chosen classifier and settings."""
    clients: list[TestClient] = []

    def _make(
        classifier: object | None = None,
        settings: Settings | None = None,
        fail_to_load: bool = False,
    ) -> TestClient:
        chosen = classifier if classifier is not None else FakeClassifier()

        def loader(_: Settings) -> object:
            if fail_to_load:
                raise RuntimeError("model download failed")
            return chosen

        app = create_app(settings=with_test_host(settings or Settings()), loader=loader)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    """A TestClient backed by the default fake classifier."""
    return make_client()


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A small valid JPEG image."""
    return make_image_bytes("JPEG")
