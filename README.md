# Bird Species Identifier

A small web app that runs on your own computer. Upload a photo of a bird and it tells you
the most likely species, with a confidence score, plus the next four best guesses.

It uses the [`ozzyonfire/bird-species-classifier`](https://huggingface.co/ozzyonfire/bird-species-classifier)
model (EfficientNet, 525 species, MIT license). The model runs locally; your photos are
never sent anywhere else.

## Requirements

- Python 3.10 or newer (3.12 recommended)
- About 1 GB of free disk space for Python packages on macOS (mostly PyTorch). On Linux,
  the default PyTorch install includes NVIDIA CUDA libraries and can take several GB.
- An internet connection the **first time** you start the app, to download the model

On an Apple Silicon Mac, use a native (arm64) Python such as Homebrew's
`/opt/homebrew/bin/python3.12`. An Intel (x86_64) Python cannot install a recent
enough PyTorch.

## Setup

Run these commands from the project folder.

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

   On Windows use `.venv\Scripts\activate` instead.

3. Install the app and developer tools:

   ```bash
   pip install -r requirements-dev.txt
   ```

   To install only what is needed to run the app, use `pip install -r requirements.txt`.

4. (Optional) Create your own settings file:

   ```bash
   cp .env.example .env
   ```

## Run the app

With the virtual environment active:

```bash
python -m app.main
```

Then open **http://127.0.0.1:8000** in your browser.

Stop the app with `Ctrl+C`.

**First run:** the model (~34 MB) is downloaded from Hugging Face and cached in
`~/.cache/huggingface`. Later runs load it from the cache. To guarantee no network
access after the first download, start the app with `HF_HUB_OFFLINE=1 python -m app.main`.

You can also start it with Uvicorn directly:

```bash
uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Check that the model is loaded:

```bash
curl http://127.0.0.1:8000/api/health
```

A healthy response looks like
`{"status":"ok","model_loaded":true,"max_upload_bytes":10485760}`.

## API

| Method | Path            | Description                                                        |
|--------|-----------------|--------------------------------------------------------------------|
| GET    | `/`             | The upload page                                                    |
| GET    | `/api/health`   | Service status and whether the model is loaded                     |
| POST   | `/api/identify` | Identify a bird. Send one image as multipart form field `file`.    |

Example:

```bash
curl -F "file=@my-bird.jpg" http://127.0.0.1:8000/api/identify
```

Response:

```json
{
  "top_prediction": {"species": "American Goldfinch", "confidence": 0.998},
  "predictions": [
    {"species": "American Goldfinch", "confidence": 0.998},
    {"species": "Andean Siskin", "confidence": 0.0005}
  ],
  "low_confidence": false
}
```

`predictions` holds the top 5, highest first (shortened above).

Errors are returned as `{"detail": "<friendly message>"}` with these status codes:

| Status | When                                                             |
|--------|------------------------------------------------------------------|
| 400    | No file, more than one file, empty file, malformed request, damaged image, or an unknown `Host` header |
| 403    | The request came from a different website (see Security below)  |
| 413    | File larger than the size limit, or too many pixels              |
| 415    | Not a JPEG, PNG, or WEBP image (checked from the file contents)  |
| 500    | Unexpected error while processing (details are only in the logs) |
| 503    | The model is not loaded                                          |

## Configuration

Settings come from environment variables or a `.env` file in the **current working
directory** (environment variables win), so start the app from the project folder.
See `.env.example`.

| Variable                        | Default                             | Meaning                                                    |
|---------------------------------|-------------------------------------|------------------------------------------------------------|
| `BIRD_MODEL_NAME`               | `ozzyonfire/bird-species-classifier` | Hugging Face model to load                                 |
| `BIRD_MODEL_REVISION`           | `d4d80527be1343dacfe67af84d064ca6f9b7547e` | Model branch, tag, or commit hash (pinned by default) |
| `BIRD_MAX_UPLOAD_MB`            | `10`                                | Largest accepted upload, in MB                             |
| `BIRD_LOW_CONFIDENCE_THRESHOLD` | `0.5`                               | Top confidence below this sets `low_confidence` to `true` |
| `BIRD_MAX_IMAGE_PIXELS`         | `64000000`                          | Largest accepted image, in total pixels                    |
| `BIRD_HOST`                     | `127.0.0.1`                         | Address used by `python -m app.main`                       |
| `BIRD_PORT`                     | `8000`                              | Port used by `python -m app.main`                          |
| `BIRD_LOG_LEVEL`                | `INFO`                              | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`         |
| `BIRD_ALLOWED_HOSTS`            | `127.0.0.1,localhost`               | Comma-separated host names the server answers to           |

`BIRD_HOST` and `BIRD_PORT` only apply to `python -m app.main`; when you run `uvicorn`
directly, use its `--host` and `--port` options.

If you make the app reachable from other devices (for example `BIRD_HOST=0.0.0.0`), add
the name or IP address you will browse to into `BIRD_ALLOWED_HOSTS`, for example
`BIRD_ALLOWED_HOSTS=127.0.0.1,localhost,192.168.1.20`.

The app uses an NVIDIA GPU (CUDA) or Apple Silicon GPU (MPS) automatically when one is
available, and the CPU otherwise.

## Security

- The server listens on `127.0.0.1` only, unless you change `BIRD_HOST`.
- Uploads are limited in size while they are being received, and the file type is
  checked from the file's contents, not its name.
- Very large images are rejected, and the rest are shrunk before processing, to keep
  memory use low. At most two photos are processed at the same time.
- Requests whose `Host` header is not in `BIRD_ALLOWED_HOSTS` are rejected. Uploads
  sent by a web page on a different website are rejected with 403.
- Every response has security headers, including a Content-Security-Policy that
  allows same-origin resources only. Error messages never include internal details;
  those go to the server log.

## Tests and linting

With the virtual environment active:

```bash
pytest
```

The normal test run uses a fake model, so it is fast and needs no download.

Run the optional test that uses the real model (downloads it on first run):

```bash
pytest -m integration
```

Lint and format:

```bash
ruff check .
ruff format .
```

## Limits of the model

- It only knows **525 species**. A bird outside that list will be reported as the
  closest species it does know, often with low confidence.
- It works best with **clear, well-lit photos of a single bird** that fills much of the
  frame. Distant, blurry, or partly hidden birds, and photos with several birds, give
  less reliable results.
- It will always name a bird, even for a photo with no bird in it. Treat low-confidence
  results with caution.
- Some species names come from the training dataset and may have unusual spellings.

## Project layout

```
app/
  main.py         FastAPI app, lifespan, middleware, routes
  config.py       Settings from environment variables
  classifier.py   Model loading, image validation, prediction logic
  schemas.py      Pydantic response models
  static/         index.html, styles.css, app.js
tests/            pytest suite (fake model by default)
```
