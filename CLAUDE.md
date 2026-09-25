# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The virtual environment in `.venv` must be built with a native arm64 Python (`/opt/homebrew/bin/python3.12`). The Intel Python at `/usr/local/bin/python3.12` can only install PyTorch 2.2.2, and transformers 5.x silently disables PyTorch below 2.5.

```bash
.venv/bin/python -m app.main                                  # run the app (reads BIRD_HOST/BIRD_PORT), http://127.0.0.1:8000
.venv/bin/uvicorn app.main:create_app --factory --port 8000   # alternative; there is no module-level `app`
.venv/bin/pytest                                              # unit tests (fake model; integration tests are skipped)
.venv/bin/pytest tests/test_api.py::TestUploadValidation::test_gif_returns_415   # single test
.venv/bin/pytest -m integration                               # real-model test (downloads ~34 MB on first run)
.venv/bin/ruff check . && .venv/bin/ruff format .
node --check app/static/app.js                                # only JS check available; there are no JS tests
```

`pyproject.toml` sets `addopts = "-m 'not integration'"`. Passing `-m integration` on the command line overrides it.

## Architecture

- `app/classifier.py` holds everything that doesn't involve HTTP: image decoding and validation, label clean-up, ranking, and model loading. It raises domain exceptions (`UnsupportedImageTypeError`, `CorruptImageError`, `ImageTooLargeError`). `app/main.py` maps them to 415, 400 and 413. Keep web code out of `classifier.py`.
- **Classifier seam.** `Classifier` is a Protocol whose `predict()` returns *raw* labels in any order. `classify()` sorts, truncates to `top_k`, cleans labels and clamps confidences. A fake therefore only needs to return raw predictions, and the ordering and clean-up logic is still tested.
- **App factory.** `create_app(settings, loader)` builds the app. The lifespan calls `loader(settings)` in a thread pool. If loading fails, `app.state.classifier` stays `None`: health reports `degraded` and `/api/identify` returns 503. Tests inject a fake through `loader`. There is deliberately no module-level `app`, so importing the module has no side effects.
- **App state.** Only these live on `app.state`: `settings` (a frozen dataclass), `classifier`, and `work_slots`. `work_slots` is an `asyncio.Semaphore(2)` that limits concurrent decode and inference.
- **Middleware order** (outermost first; `add_middleware` in `_add_middleware` is called in reverse):
  1. `SecurityHeadersMiddleware`: headers, plus `Cache-Control: no-store` on `/api/*`.
  2. `UnhandledErrorMiddleware`: generic 500s. It sits inside the headers middleware so those responses still get headers. Don't replace it with `add_exception_handler(Exception, ...)`, because Starlette serves that handler outside user middleware.
  3. `TrustedHostMiddleware`: `BIRD_ALLOWED_HOSTS`.
  4. `SameOriginGuardMiddleware`: 403 on `Sec-Fetch-Site: cross-site` or an `Origin` that doesn't match `Host`.
  5. `BodySizeLimitMiddleware`: buffers at most `max_upload_bytes` + 64 KB of multipart overhead, then replays the body.

  The size limit has to live in middleware because FastAPI parses the multipart body before the endpoint runs. `read_upload_limited` then enforces the exact per-file limit.
- **Image decoding.** `Image.open` is restricted to the JPEG, PNG and WEBP decoders. After opening, the format `MPO` (multi-picture JPEG from phones) is also accepted. Pixel count is checked before `load()`. JPEGs use `draft()`, and every image is thumbnailed to 1024 px *before* `exif_transpose`/`convert`, to bound memory. `configure_pillow` also sets Pillow's global `MAX_IMAGE_PIXELS` at startup.
- **Settings.** `app/config.py` merges `.env` from the current working directory with `os.environ` (the environment wins) and never mutates `os.environ`. Validation lives in `Settings.__post_init__`. The default model revision is pinned to a commit SHA.
- **Frontend.** `app/static/` has no build step. The CSP forbids inline scripts and styles, so styling goes in `styles.css` and dynamic widths are set through `element.style` in JS. All server data must be inserted with `textContent`. The page reads `max_upload_bytes` and `model_loaded` from `/api/health` at load time.

## Testing conventions

- Use the `make_client(classifier=..., settings=..., fail_to_load=...)` fixture in `tests/conftest.py`. It adds `testserver` to `allowed_hosts`. Any test that builds an app directly must call `with_test_host(settings)`, or every request gets a 400.
- `FakeClassifier` defaults to 6 unsorted, upper-case predictions, so tests can assert ordering, truncation and clean-up.
