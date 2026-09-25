"""FastAPI application: startup lifespan, middleware, and HTTP routes.

Start it with ``python -m app.main`` (uses BIRD_HOST / BIRD_PORT) or with
``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.classifier import (
    Classifier,
    CorruptImageError,
    ImageTooLargeError,
    Prediction,
    UnsupportedImageTypeError,
    classify,
    configure_pillow,
    decode_image,
    is_low_confidence,
    load_classifier,
)
from app.config import Settings, load_settings
from app.schemas import ErrorResponse, HealthResponse, IdentifyResponse, SpeciesPrediction

logger = logging.getLogger("app.main")

STATIC_DIR = Path(__file__).resolve().parent / "static"
READ_CHUNK_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024
BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
# How many photos may be decoded and classified at the same time. Keeps
# memory use predictable when several requests arrive together.
MAX_CONCURRENT_JOBS = 2

MSG_NO_FILE = "No photo was uploaded. Please choose an image file and try again."
MSG_TOO_MANY_FILES = "Please upload just one photo at a time."
MSG_EMPTY_FILE = "The uploaded file is empty. Please choose a different photo."
MSG_BAD_REQUEST = "The request was not understood. Please upload one image in the 'file' field."
MSG_UNSUPPORTED = "That file is not a supported image. Please upload a JPEG, PNG, or WEBP photo."
MSG_CORRUPT = "That image appears to be damaged or incomplete. Please try a different photo."
MSG_DIMENSIONS = (
    "That image has too many pixels. Please resize it to a smaller resolution and try again."
)
MSG_MODEL_UNAVAILABLE = (
    "The bird identification model is not available right now. Please try again later."
)
MSG_PREDICTION_FAILED = "The model could not process this photo. Please try a different one."
MSG_INTERNAL = "Something went wrong on our side. Please try again."
MSG_CROSS_ORIGIN = "Requests from other websites are not allowed."

SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (
        b"content-security-policy",
        b"default-src 'self'; img-src 'self' blob:; object-src 'none'; "
        b"base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
    ),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
)
NO_STORE_HEADER: tuple[bytes, bytes] = (b"cache-control", b"no-store")

ClassifierLoader = Callable[[Settings], Classifier]


def too_large_message(max_bytes: int) -> str:
    """Build the user-facing message for an upload over the size limit."""
    max_mb = max_bytes / (1024 * 1024)
    return f"That file is too large. The maximum size is {max_mb:g} MB."


def _header(scope: Scope, name: bytes) -> str | None:
    """Return the first value of a request header, decoded as Latin-1."""
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class SecurityHeadersMiddleware:
    """ASGI middleware that adds security headers to every HTTP response.

    API responses also get ``Cache-Control: no-store``.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Add any missing security headers when the response starts."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = list(SECURITY_HEADERS)
        if scope["path"].startswith("/api/"):
            extra.append(NO_STORE_HEADER)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name.lower() for name, _ in headers}
                headers.extend(h for h in extra if h[0] not in present)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class UnhandledErrorMiddleware:
    """ASGI middleware that turns unexpected exceptions into a generic 500.

    It sits inside ``SecurityHeadersMiddleware`` so these responses still get
    security headers. Details are logged, never sent to the client.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Run the app, replacing any unhandled exception with a safe response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        response_started = False

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception:
            logger.exception("Unhandled error on %s", scope.get("path", "?"))
            if response_started:
                raise
            response = JSONResponse(status_code=500, content={"detail": MSG_INTERNAL})
            await response(scope, receive, send)


class SameOriginGuardMiddleware:
    """ASGI middleware that blocks state-changing requests from other websites.

    A web page on another site could otherwise make the visitor's browser send
    photos to this local server. Requests are rejected with 403 when the browser
    marks them as cross-site, or when their ``Origin`` does not match the
    ``Host`` they were sent to. Clients that send neither header (curl,
    scripts) are allowed.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream ASGI app."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject cross-origin POST/PUT/PATCH/DELETE requests."""
        if (
            scope["type"] == "http"
            and scope["method"] in BODY_METHODS
            and self._is_cross_origin(scope)
        ):
            response = JSONResponse(status_code=403, content={"detail": MSG_CROSS_ORIGIN})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _is_cross_origin(scope: Scope) -> bool:
        """Return True when browser headers show the request came from another origin."""
        if _header(scope, b"sec-fetch-site") == "cross-site":
            return True
        origin = _header(scope, b"origin")
        if origin is None:
            return False
        host = _header(scope, b"host")
        origin_host = urlsplit(origin).netloc
        return not origin_host or origin_host.lower() != (host or "").lower()


class BodySizeLimitMiddleware:
    """ASGI middleware that rejects request bodies over a byte limit with 413.

    The body is read in chunks and counting stops as soon as the limit is
    passed, so an oversized upload is never fully read into memory.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        """Wrap the downstream ASGI app with a maximum body size."""
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Buffer the body up to the limit, then replay it to the app."""
        if scope["type"] != "http" or scope["method"] not in BODY_METHODS:
            await self.app(scope, receive, send)
            return

        declared = self._declared_length(scope)
        if declared is not None and declared > self.max_body_bytes:
            await self._reject(scope, receive, send)
            return

        chunks: list[bytes] = []
        total = 0
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_body_bytes:
                await self._reject(scope, receive, send)
                return
            chunks.append(body)
            more_body = message.get("more_body", False)

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _declared_length(scope: Scope) -> int | None:
        """Return the Content-Length header as an int, or None if absent/invalid."""
        value = _header(scope, b"content-length")
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Send a 413 JSON response."""
        upload_limit = max(self.max_body_bytes - MULTIPART_OVERHEAD_BYTES, 1)
        response = JSONResponse(
            status_code=413,
            content={"detail": too_large_message(upload_limit)},
            headers={"Connection": "close"},
        )
        await response(scope, receive, send)


def default_loader(settings: Settings) -> Classifier:
    """Load the real Hugging Face model described by ``settings``."""
    return load_classifier(settings.model_name, settings.model_revision)


def _try_load_classifier(loader: ClassifierLoader, settings: Settings) -> Classifier | None:
    """Load the model, returning None (and logging why) if it fails."""
    try:
        classifier = loader(settings)
    except Exception:
        logger.exception("Model failed to load; /api/identify will return 503.")
        return None
    logger.info("Model loaded and ready.")
    return classifier


async def read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload in chunks, raising 413 as soon as it exceeds ``max_bytes``."""
    buffer = bytearray()
    while True:
        chunk = await upload.read(min(READ_CHUNK_BYTES, max_bytes + 1 - len(buffer)))
        if not chunk:
            return bytes(buffer)
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(status_code=413, detail=too_large_message(max_bytes))


async def decode_upload(data: bytes, max_image_pixels: int) -> Image.Image:
    """Decode image bytes off the event loop, mapping failures to HTTP errors."""
    try:
        return await run_in_threadpool(decode_image, data, max_image_pixels)
    except UnsupportedImageTypeError as exc:
        raise HTTPException(status_code=415, detail=MSG_UNSUPPORTED) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=413, detail=MSG_DIMENSIONS) from exc
    except CorruptImageError as exc:
        raise HTTPException(status_code=400, detail=MSG_CORRUPT) from exc


async def predict_species(
    classifier: Classifier, image: Image.Image, top_k: int
) -> list[Prediction]:
    """Run inference in a worker thread, mapping failures to a 500 error."""
    try:
        predictions = await run_in_threadpool(classify, classifier, image, top_k)
    except Exception as exc:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail=MSG_PREDICTION_FAILED) from exc
    if not predictions:
        logger.error("Model returned no predictions.")
        raise HTTPException(status_code=500, detail=MSG_PREDICTION_FAILED)
    return predictions


def build_response(predictions: list[Prediction], threshold: float) -> IdentifyResponse:
    """Convert predictions into the public response model."""
    items = [SpeciesPrediction(species=p.label, confidence=p.confidence) for p in predictions]
    return IdentifyResponse(
        top_prediction=items[0],
        predictions=items,
        low_confidence=is_low_confidence(predictions, threshold),
    )


async def _read_single_upload(files: list[UploadFile] | None, max_bytes: int) -> bytes:
    """Check exactly one file was sent, then read it within the size limit."""
    if not files:
        raise HTTPException(status_code=400, detail=MSG_NO_FILE)
    try:
        if len(files) > 1:
            raise HTTPException(status_code=400, detail=MSG_TOO_MANY_FILES)
        data = await read_upload_limited(files[0], max_bytes)
    finally:
        for upload in files:
            await upload.close()
    if not data:
        raise HTTPException(status_code=400, detail=MSG_EMPTY_FILE)
    return data


async def serve_index() -> FileResponse:
    """Serve the single-page upload UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


async def health(request: Request) -> HealthResponse:
    """Report whether the service is up and the model is loaded."""
    loaded = request.app.state.classifier is not None
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        max_upload_bytes=settings.max_upload_bytes,
    )


async def identify(
    request: Request, file: list[UploadFile] | None = File(default=None)
) -> IdentifyResponse:
    """Identify the bird species in one uploaded JPEG, PNG, or WEBP photo."""
    classifier: Classifier | None = request.app.state.classifier
    settings: Settings = request.app.state.settings
    if classifier is None:
        raise HTTPException(status_code=503, detail=MSG_MODEL_UNAVAILABLE)
    data = await _read_single_upload(file, settings.max_upload_bytes)
    async with request.app.state.work_slots:
        image = await decode_upload(data, settings.max_image_pixels)
        predictions = await predict_species(classifier, image, settings.top_k)
    return build_response(predictions, settings.low_confidence_threshold)


async def handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a friendly 400 instead of FastAPI's detailed 422 body."""
    logger.info("Rejected malformed request to %s", request.url.path)
    return JSONResponse(status_code=400, content={"detail": MSG_BAD_REQUEST})


def _register_routes(app: FastAPI) -> None:
    """Attach all HTTP routes and the static files mount."""
    error_responses = {code: {"model": ErrorResponse} for code in (400, 403, 413, 415, 500, 503)}
    app.add_api_route("/", serve_index, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route("/api/health", health, methods=["GET"], response_model=HealthResponse)
    app.add_api_route(
        "/api/identify",
        identify,
        methods=["POST"],
        response_model=IdentifyResponse,
        responses=error_responses,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _configure_logging(level: str) -> None:
    """Set up log formatting and quiet noisy third-party loggers."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _add_middleware(app: FastAPI, settings: Settings) -> None:
    """Install middleware. The last one added is the outermost."""
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD_BYTES,
    )
    app.add_middleware(SameOriginGuardMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(UnhandledErrorMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)


def create_app(
    settings: Settings | None = None,
    loader: ClassifierLoader = default_loader,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: configuration; read from the environment when omitted.
        loader: function that builds the classifier at startup. Tests pass a
            fake loader so the real model is never downloaded.
    """
    resolved = settings if settings is not None else load_settings()
    _configure_logging(resolved.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_pillow(resolved.max_image_pixels)
        app.state.work_slots = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        app.state.classifier = await run_in_threadpool(_try_load_classifier, loader, resolved)
        yield
        app.state.classifier = None

    app = FastAPI(
        title="Bird Species Identifier",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = resolved
    app.state.classifier = None
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    _add_middleware(app, resolved)
    _register_routes(app)
    return app


def run() -> None:
    """Start Uvicorn using the host and port from the settings."""
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
