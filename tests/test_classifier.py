"""Unit tests for the model-independent logic in ``app.classifier``."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.classifier import (
    CorruptImageError,
    ImageTooLargeError,
    Prediction,
    UnsupportedImageTypeError,
    classify,
    clean_label,
    decode_image,
    is_low_confidence,
)
from app.config import Settings
from app.main import create_app, default_loader
from tests.conftest import FakeClassifier, make_image_bytes, with_test_host

MAX_PIXELS = 1_000_000


class TestCleanLabel:
    """Display-name clean-up of raw model labels."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AMERICAN GOLDFINCH", "American Goldfinch"),
            ("BLACK-THROATED SPARROW", "Black-Throated Sparrow"),
            ("  BALD   EAGLE ", "Bald Eagle"),
            ("snowy_egret", "Snowy Egret"),
            ("ANNAS HUMMINGBIRD", "Annas Hummingbird"),
            ("", ""),
        ],
    )
    def test_clean_label(self, raw: str, expected: str) -> None:
        assert clean_label(raw) == expected


class TestClassify:
    """Ranking, truncation, and label clean-up in ``classify``."""

    def test_sorted_truncated_and_cleaned(self) -> None:
        image = Image.new("RGB", (8, 8))
        results = classify(FakeClassifier(), image, top_k=5)
        assert [r.label for r in results] == [
            "American Goldfinch",
            "House Finch",
            "Mourning Dove",
            "Black-Throated Sparrow",
            "Evening Grosbeak",
        ]
        assert [r.confidence for r in results] == [0.82, 0.06, 0.05, 0.03, 0.02]

    def test_confidence_is_clamped(self) -> None:
        fake = FakeClassifier((Prediction("A", 1.0000001), Prediction("B", -0.1)))
        results = classify(fake, Image.new("RGB", (8, 8)), top_k=5)
        assert [r.confidence for r in results] == [1.0, 0.0]


class TestIsLowConfidence:
    """The low-confidence rule."""

    def test_below_threshold(self) -> None:
        assert is_low_confidence([Prediction("A", 0.49)], 0.5) is True

    def test_at_threshold_is_not_low(self) -> None:
        assert is_low_confidence([Prediction("A", 0.5)], 0.5) is False

    def test_empty_is_low(self) -> None:
        assert is_low_confidence([], 0.5) is True


class TestDecodeImage:
    """Validation and normalisation of uploaded image bytes."""

    @pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
    def test_allowed_formats_decode_to_rgb(self, image_format: str) -> None:
        image = decode_image(make_image_bytes(image_format), MAX_PIXELS)
        assert image.mode == "RGB"
        assert image.size == (64, 48)

    def test_rgba_png_converted_to_rgb(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGBA", (10, 10), (255, 0, 0, 128)).save(buffer, format="PNG")
        assert decode_image(buffer.getvalue(), MAX_PIXELS).mode == "RGB"

    def test_exif_orientation_applied(self) -> None:
        image = Image.new("RGB", (40, 20), "blue")
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise.
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", exif=exif.tobytes())
        assert decode_image(buffer.getvalue(), MAX_PIXELS).size == (20, 40)

    def test_text_is_rejected(self) -> None:
        with pytest.raises(UnsupportedImageTypeError):
            decode_image(b"just some text", MAX_PIXELS)

    @pytest.mark.parametrize(("image_format", "mode"), [("GIF", "P"), ("BMP", "RGB")])
    def test_other_image_formats_rejected(self, image_format: str, mode: str) -> None:
        with pytest.raises(UnsupportedImageTypeError):
            decode_image(make_image_bytes(image_format, mode=mode), MAX_PIXELS)

    def test_truncated_image_is_corrupt(self) -> None:
        data = make_image_bytes("JPEG", size=(200, 200), noise=True)
        with pytest.raises(CorruptImageError):
            decode_image(data[: len(data) // 2], MAX_PIXELS)

    def test_too_many_pixels_rejected(self) -> None:
        with pytest.raises(ImageTooLargeError):
            decode_image(make_image_bytes("PNG", size=(20, 20)), max_image_pixels=399)

    def test_pillow_decompression_bomb_error_mapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Over 2x Pillow's own limit, Image.open raises DecompressionBombError.
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
        with pytest.raises(ImageTooLargeError):
            decode_image(make_image_bytes("PNG", size=(20, 20)), MAX_PIXELS)

    def test_multi_picture_jpeg_accepted(self) -> None:
        frames = [Image.new("RGB", (30, 20), "red"), Image.new("RGB", (30, 20), "blue")]
        buffer = io.BytesIO()
        frames[0].save(buffer, format="MPO", save_all=True, append_images=frames[1:])
        image = decode_image(buffer.getvalue(), MAX_PIXELS)
        assert image.mode == "RGB"
        assert image.size == (30, 20)

    @pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
    def test_large_images_are_downscaled(self, image_format: str) -> None:
        data = make_image_bytes(image_format, size=(4000, 3000))
        image = decode_image(data, MAX_PIXELS * 20)
        assert max(image.size) <= 1024
        assert image.mode == "RGB"


@pytest.mark.integration
def test_real_model_end_to_end() -> None:
    """Load the real Hugging Face model and identify a generated image.

    Skipped by default; run with ``pytest -m integration``. The first run
    downloads the model (~34 MB).
    """
    app = create_app(settings=with_test_host(Settings()), loader=default_loader)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["model_loaded"] is True
        image = make_image_bytes("JPEG", size=(224, 224), noise=True)
        response = client.post("/api/identify", files={"file": ("noise.jpg", image, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    assert all(0.0 <= p["confidence"] <= 1.0 for p in body["predictions"])
    top = body["top_prediction"]["species"]
    assert top == top.strip() and top != top.upper()
