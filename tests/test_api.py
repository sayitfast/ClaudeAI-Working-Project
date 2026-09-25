"""HTTP-level tests for the FastAPI app, using a fake classifier."""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.classifier import Prediction
from app.config import Settings
from tests.conftest import FailingClassifier, FakeClassifier, make_image_bytes

IDENTIFY_URL = "/api/identify"


def upload(client: TestClient, data: bytes, name: str = "bird.jpg", mime: str = "image/jpeg"):
    """POST a single file to the identify endpoint."""
    return client.post(IDENTIFY_URL, files={"file": (name, data, mime)})


class TestIdentifySuccess:
    """Happy-path identification."""

    def test_returns_top_prediction(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = upload(client, jpeg_bytes)
        assert response.status_code == 200
        body = response.json()
        assert body["top_prediction"] == {"species": "American Goldfinch", "confidence": 0.82}
        assert body["low_confidence"] is False

    def test_returns_top_five_highest_first(self, client: TestClient, jpeg_bytes: bytes) -> None:
        body = upload(client, jpeg_bytes).json()
        confidences = [p["confidence"] for p in body["predictions"]]
        assert len(confidences) == 5
        assert confidences == sorted(confidences, reverse=True)
        assert body["predictions"][0] == body["top_prediction"]
        assert "Cedar Waxwing" not in [p["species"] for p in body["predictions"]]

    def test_labels_are_cleaned(self, client: TestClient, jpeg_bytes: bytes) -> None:
        body = upload(client, jpeg_bytes).json()
        species = [p["species"] for p in body["predictions"]]
        assert species == [
            "American Goldfinch",
            "House Finch",
            "Mourning Dove",
            "Black-Throated Sparrow",
            "Evening Grosbeak",
        ]

    @pytest.mark.parametrize(
        ("image_format", "mime", "name"),
        [("PNG", "image/png", "bird.png"), ("WEBP", "image/webp", "bird.webp")],
    )
    def test_accepts_png_and_webp(
        self, client: TestClient, image_format: str, mime: str, name: str
    ) -> None:
        response = upload(client, make_image_bytes(image_format), name=name, mime=mime)
        assert response.status_code == 200

    def test_ignores_misleading_extension_for_real_image(self, client: TestClient) -> None:
        response = upload(client, make_image_bytes("PNG"), name="bird.txt", mime="text/plain")
        assert response.status_code == 200


class TestLowConfidence:
    """The low_confidence flag."""

    def test_flag_set_when_top_is_below_threshold(
        self, make_client: Callable[..., TestClient], jpeg_bytes: bytes
    ) -> None:
        fake = FakeClassifier((Prediction("SNOWY EGRET", 0.30), Prediction("GREAT EGRET", 0.25)))
        body = upload(make_client(classifier=fake), jpeg_bytes).json()
        assert body["low_confidence"] is True
        assert body["top_prediction"]["species"] == "Snowy Egret"

    def test_threshold_is_configurable(
        self, make_client: Callable[..., TestClient], jpeg_bytes: bytes
    ) -> None:
        client = make_client(settings=Settings(low_confidence_threshold=0.9))
        body = upload(client, jpeg_bytes).json()
        assert body["low_confidence"] is True


class TestUploadValidation:
    """Rejection of bad uploads with the right status codes."""

    def test_missing_file_returns_400(self, client: TestClient) -> None:
        response = client.post(IDENTIFY_URL)
        assert response.status_code == 400
        assert "No photo" in response.json()["detail"]

    def test_text_field_instead_of_file_returns_400(self, client: TestClient) -> None:
        response = client.post(IDENTIFY_URL, data={"file": "not a file"})
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        response = upload(client, b"")
        assert response.status_code == 400
        assert "empty" in response.json()["detail"]

    def test_oversize_file_returns_413(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(settings=Settings(max_upload_bytes=2048))
        big_png = make_image_bytes("PNG", size=(100, 100), noise=True)
        assert len(big_png) > 2048
        response = upload(client, big_png, name="big.png", mime="image/png")
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_oversize_body_rejected_before_parsing(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(settings=Settings(max_upload_bytes=1024))
        response = upload(client, b"x" * (200 * 1024))
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

    def test_text_file_renamed_to_jpg_returns_415(self, client: TestClient) -> None:
        response = upload(client, b"hello, I am not a bird photo\n" * 10)
        assert response.status_code == 415
        assert "JPEG, PNG, or WEBP" in response.json()["detail"]

    def test_gif_returns_415(self, client: TestClient) -> None:
        response = upload(client, make_image_bytes("GIF", mode="P"), name="bird.gif")
        assert response.status_code == 415

    def test_corrupt_image_returns_400(self, client: TestClient) -> None:
        full = make_image_bytes("JPEG", size=(200, 200), noise=True)
        response = upload(client, full[: len(full) // 2])
        assert response.status_code == 400
        assert "damaged" in response.json()["detail"]

    def test_too_many_pixels_returns_413(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(settings=Settings(max_image_pixels=100))
        response = upload(client, make_image_bytes("PNG", size=(20, 20)), mime="image/png")
        assert response.status_code == 413
        assert "pixels" in response.json()["detail"]


class TestModelUnavailable:
    """Behaviour when the model fails to load or to predict."""

    def test_identify_returns_503(
        self, make_client: Callable[..., TestClient], jpeg_bytes: bytes
    ) -> None:
        client = make_client(fail_to_load=True)
        response = upload(client, jpeg_bytes)
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]

    def test_health_reports_model_not_loaded(self, make_client: Callable[..., TestClient]) -> None:
        body = make_client(fail_to_load=True).get("/api/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False

    def test_inference_error_is_hidden(
        self, make_client: Callable[..., TestClient], jpeg_bytes: bytes
    ) -> None:
        response = upload(make_client(classifier=FailingClassifier()), jpeg_bytes)
        assert response.status_code == 500
        text = response.text
        assert "secret" not in text
        assert "Traceback" not in text
        assert "weights.bin" not in text


class TestHealthAndPages:
    """Health endpoint, static page, and security headers."""

    def test_health_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["max_upload_bytes"] == 10 * 1024 * 1024

    def test_index_page_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Bird Species Identifier" in response.text

    @pytest.mark.parametrize("path", ["/static/app.js", "/static/styles.css"])
    def test_static_assets_served(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 200

    @pytest.mark.parametrize("path", ["/", "/api/health", "/static/app.js", "/missing"])
    def test_security_headers_present(self, client: TestClient, path: str) -> None:
        headers = client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        csp = headers["content-security-policy"]
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_security_headers_on_errors(self, make_client: Callable[..., TestClient]) -> None:
        client = make_client(settings=Settings(max_upload_bytes=1024))
        response = upload(client, b"x" * (200 * 1024))
        assert response.status_code == 413
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in response.headers

    def test_api_docs_disabled(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


class TestSizeLimitEdges:
    """Upload size limit at the boundary and without Content-Length."""

    def test_chunked_oversize_body_returns_413(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(settings=Settings(max_upload_bytes=1024))

        def body() -> Iterator[bytes]:
            for _ in range(200):
                yield b"x" * 1024

        response = client.post(
            IDENTIFY_URL,
            content=body(),
            headers={"content-type": "multipart/form-data; boundary=abc"},
        )
        assert response.status_code == 413

    def test_file_exactly_at_limit_is_not_rejected_for_size(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(settings=Settings(max_upload_bytes=4096))
        response = upload(client, b"x" * 4096)
        assert response.status_code == 415

    def test_file_one_byte_over_limit_returns_413(
        self, make_client: Callable[..., TestClient]
    ) -> None:
        client = make_client(settings=Settings(max_upload_bytes=4096))
        response = upload(client, b"x" * 4097)
        assert response.status_code == 413


class TestRequestShape:
    """Multiple files, empty model output, and unexpected errors."""

    def test_multiple_files_returns_400(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = client.post(
            IDENTIFY_URL,
            files=[
                ("file", ("a.jpg", jpeg_bytes, "image/jpeg")),
                ("file", ("b.jpg", jpeg_bytes, "image/jpeg")),
            ],
        )
        assert response.status_code == 400
        assert "one photo" in response.json()["detail"]

    def test_multi_picture_jpeg_accepted(self, client: TestClient) -> None:
        frames = [Image.new("RGB", (30, 20), "red"), Image.new("RGB", (30, 20), "blue")]
        buffer = io.BytesIO()
        frames[0].save(buffer, format="MPO", save_all=True, append_images=frames[1:])
        assert upload(client, buffer.getvalue()).status_code == 200

    def test_empty_predictions_return_500(
        self, make_client: Callable[..., TestClient], jpeg_bytes: bytes
    ) -> None:
        response = upload(make_client(classifier=FakeClassifier(())), jpeg_bytes)
        assert response.status_code == 500
        assert "could not process" in response.json()["detail"]

    def test_unhandled_error_is_generic_and_has_headers(self, client: TestClient) -> None:
        async def boom() -> None:
            raise RuntimeError("secret internal detail")

        client.app.add_api_route("/boom", boom, methods=["GET"])
        response = client.get("/boom")
        assert response.status_code == 500
        assert "secret" not in response.text
        assert response.json()["detail"] == "Something went wrong on our side. Please try again."
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "content-security-policy" in response.headers


class TestCrossSiteProtection:
    """Host allow-list and same-origin checks."""

    def test_unknown_host_rejected(self, client: TestClient) -> None:
        response = client.get("/api/health", headers={"host": "evil.example"})
        assert response.status_code == 400

    def test_cross_site_fetch_rejected(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = client.post(
            IDENTIFY_URL,
            files={"file": ("bird.jpg", jpeg_bytes, "image/jpeg")},
            headers={"sec-fetch-site": "cross-site"},
        )
        assert response.status_code == 403
        assert response.headers["x-frame-options"] == "DENY"

    def test_foreign_origin_rejected(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = client.post(
            IDENTIFY_URL,
            files={"file": ("bird.jpg", jpeg_bytes, "image/jpeg")},
            headers={"origin": "https://evil.example"},
        )
        assert response.status_code == 403

    def test_null_origin_rejected(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = client.post(
            IDENTIFY_URL,
            files={"file": ("bird.jpg", jpeg_bytes, "image/jpeg")},
            headers={"origin": "null"},
        )
        assert response.status_code == 403

    def test_same_origin_allowed(self, client: TestClient, jpeg_bytes: bytes) -> None:
        response = client.post(
            IDENTIFY_URL,
            files={"file": ("bird.jpg", jpeg_bytes, "image/jpeg")},
            headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"},
        )
        assert response.status_code == 200


class TestCachingAndMethods:
    """Cache headers on API responses and HEAD support."""

    def test_api_responses_not_cached(self, client: TestClient) -> None:
        assert client.get("/api/health").headers["cache-control"] == "no-store"

    def test_head_on_index(self, client: TestClient) -> None:
        assert client.head("/").status_code == 200
