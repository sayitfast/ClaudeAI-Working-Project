"""Model loading, image decoding, and prediction logic.

This module knows nothing about HTTP. The web layer in ``app.main`` calls
these functions and translates the exceptions defined here into responses.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Decoders Pillow may try when identifying an upload. Nothing else is ever run.
OPEN_FORMATS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
# Formats accepted after identification. "MPO" is a multi-picture JPEG, used by
# many phone cameras; Pillow reports it separately but it is still a JPEG file.
ALLOWED_FORMATS: frozenset[str] = frozenset({"JPEG", "MPO", "PNG", "WEBP"})
# The model works at 224 px, so large photos are shrunk to this size right
# after decoding to keep memory use low.
WORKING_SIZE: tuple[int, int] = (1024, 1024)


class ImageValidationError(Exception):
    """Base class for problems with an uploaded image."""


class UnsupportedImageTypeError(ImageValidationError):
    """The data is not a JPEG, PNG, or WEBP image."""


class CorruptImageError(ImageValidationError):
    """The data looks like a supported image but cannot be decoded."""


class ImageTooLargeError(ImageValidationError):
    """The image has more pixels than the configured limit."""


@dataclass(frozen=True)
class Prediction:
    """A single species guess with a confidence between 0 and 1."""

    label: str
    confidence: float


class Classifier(Protocol):
    """Anything that can turn an RGB image into raw species predictions."""

    def predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        """Return up to ``top_k`` raw predictions for ``image``."""
        ...


class HuggingFaceClassifier:
    """Wraps a ``transformers`` image-classification pipeline."""

    def __init__(self, pipe: Any) -> None:
        """Store the pipeline and a lock that serialises inference calls."""
        self._pipe = pipe
        self._lock = threading.Lock()

    def predict(self, image: Image.Image, top_k: int) -> list[Prediction]:
        """Run the model and return raw predictions (labels not yet cleaned)."""
        with self._lock:
            raw_results = self._pipe(image, top_k=top_k)
        return [
            Prediction(label=str(item["label"]), confidence=float(item["score"]))
            for item in raw_results
        ]


def select_device() -> int | str:
    """Pick a GPU (CUDA or Apple MPS) when available, otherwise the CPU."""
    import torch

    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_classifier(model_name: str, revision: str) -> HuggingFaceClassifier:
    """Download (or load from cache) the model and build a classifier.

    Heavy imports happen here so that importing this module stays cheap,
    which keeps the unit tests fast and free of model downloads.
    """
    from transformers import pipeline

    device = select_device()
    logger.info("Loading model %s@%s on device %s", model_name, revision, device)
    pipe = pipeline(
        task="image-classification",
        model=model_name,
        revision=revision,
        device=device,
    )
    return HuggingFaceClassifier(pipe)


def configure_pillow(max_image_pixels: int) -> None:
    """Set Pillow's decompression-bomb guard to the configured pixel limit.

    ``Image.MAX_IMAGE_PIXELS`` is a process-wide Pillow setting. It is set once
    at startup as a second line of defence; ``decode_image`` also enforces the
    limit itself on every upload.
    """
    Image.MAX_IMAGE_PIXELS = max_image_pixels


def _open_image(data: bytes) -> Image.Image:
    """Identify the image format from its bytes, allowing only safe formats."""
    try:
        image = Image.open(io.BytesIO(data), formats=list(OPEN_FORMATS))
    except UnidentifiedImageError as exc:
        raise UnsupportedImageTypeError("Not a JPEG, PNG, or WEBP image.") from exc
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError("Image dimensions are too large.") from exc
    except Exception as exc:
        raise CorruptImageError("Image header could not be read.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise UnsupportedImageTypeError("Not a JPEG, PNG, or WEBP image.")
    return image


def _check_dimensions(image: Image.Image, max_image_pixels: int) -> None:
    """Reject images whose width x height exceeds the pixel limit."""
    width, height = image.size
    if width <= 0 or height <= 0:
        raise CorruptImageError("Image has invalid dimensions.")
    if width * height > max_image_pixels:
        raise ImageTooLargeError("Image dimensions are too large.")


def decode_image(data: bytes, max_image_pixels: int) -> Image.Image:
    """Validate and decode uploaded bytes into an upright RGB image.

    Large images are shrunk to fit within ``WORKING_SIZE``. For multi-frame
    files only the first frame is used.

    Raises:
        UnsupportedImageTypeError: the bytes are not JPEG, PNG, or WEBP.
        ImageTooLargeError: the image exceeds ``max_image_pixels``.
        CorruptImageError: the image cannot be fully decoded.
    """
    image = _open_image(data)
    _check_dimensions(image, max_image_pixels)
    try:
        if image.format in ("JPEG", "MPO"):
            # Let the JPEG decoder scale down while decoding (much less memory).
            image.draft("RGB", WORKING_SIZE)
        image.load()
        image.thumbnail(WORKING_SIZE)
        upright = ImageOps.exif_transpose(image)
        return upright.convert("RGB")
    except Exception as exc:
        raise CorruptImageError("Image data could not be decoded.") from exc


def clean_label(raw_label: str) -> str:
    """Turn a raw model label like ``"AMERICAN GOLDFINCH"`` into title case.

    Hyphenated words keep their hyphen with each part capitalised,
    e.g. ``"BLACK-THROATED SPARROW"`` becomes ``"Black-Throated Sparrow"``.
    """
    words = raw_label.replace("_", " ").split()
    return " ".join("-".join(part.capitalize() for part in word.split("-")) for word in words)


def _clamp_confidence(value: float) -> float:
    """Keep a confidence value inside the closed range [0, 1]."""
    return min(max(value, 0.0), 1.0)


def classify(classifier: Classifier, image: Image.Image, top_k: int) -> list[Prediction]:
    """Return the top ``top_k`` predictions, highest confidence first, with clean labels."""
    raw_predictions = classifier.predict(image, top_k)
    ranked = sorted(raw_predictions, key=lambda p: p.confidence, reverse=True)[:top_k]
    return [
        Prediction(label=clean_label(p.label), confidence=_clamp_confidence(p.confidence))
        for p in ranked
    ]


def is_low_confidence(predictions: list[Prediction], threshold: float) -> bool:
    """Return True when there is no prediction or the best one is below ``threshold``."""
    return not predictions or predictions[0].confidence < threshold
