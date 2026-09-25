/*
 * Bird Species Identifier front end.
 * Vanilla JavaScript, no dependencies. All server data is inserted with
 * textContent so model output can never be interpreted as HTML.
 */
"use strict";

(function () {
  const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];
  const DEFAULT_MAX_BYTES = 10 * 1024 * 1024;

  const FALLBACK_MESSAGES = {
    400: "That photo couldn't be read. Please try a different image.",
    413: "That photo is too large. Please choose a smaller image.",
    415: "That file type isn't supported. Please use a JPEG, PNG, or WEBP photo.",
    500: "Something went wrong while identifying the bird. Please try again.",
    503: "The bird identification model isn't ready right now. Please try again later.",
  };
  const NETWORK_ERROR = "Couldn't reach the server. Check that the app is still running and try again.";
  const GENERIC_ERROR = "Something unexpected happened. Please try again.";

  const els = {
    form: document.getElementById("upload-form"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    fileName: document.getElementById("file-name"),
    maxSize: document.getElementById("max-size"),
    preview: document.getElementById("preview"),
    previewImage: document.getElementById("preview-image"),
    previewCaption: document.getElementById("preview-caption"),
    identifyButton: document.getElementById("identify-button"),
    resetButton: document.getElementById("reset-button"),
    loading: document.getElementById("loading-message"),
    error: document.getElementById("error-message"),
    results: document.getElementById("results"),
    resultsHeading: document.getElementById("results-heading"),
    topSpecies: document.getElementById("top-species"),
    topConfidence: document.getElementById("top-confidence"),
    lowConfidenceNote: document.getElementById("low-confidence-note"),
    otherResults: document.getElementById("other-results"),
    announcement: document.getElementById("result-announcement"),
    serviceStatus: document.getElementById("service-status"),
  };

  const state = {
    file: null,
    previewUrl: null,
    busy: false,
    maxBytes: DEFAULT_MAX_BYTES,
  };

  /** Format a byte count as whole or one-decimal megabytes. */
  function formatMegabytes(bytes) {
    const mb = bytes / (1024 * 1024);
    return Number.isInteger(mb) ? String(mb) : mb.toFixed(1);
  }

  /** Format a 0-1 confidence as a percentage string. */
  function formatPercent(confidence) {
    const value = Math.max(0, Math.min(1, Number(confidence) || 0)) * 100;
    return (value >= 10 ? value.toFixed(0) : value.toFixed(1)) + "%";
  }

  /** Return true when the file looks like an allowed image type. */
  function hasAllowedType(file) {
    if (ALLOWED_TYPES.includes(file.type)) {
      return true;
    }
    const name = file.name.toLowerCase();
    return file.type === "" && ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext));
  }

  /** Check a file in the browser; returns an error message or null. */
  function validateFile(file) {
    if (!hasAllowedType(file)) {
      return FALLBACK_MESSAGES[415];
    }
    if (file.size === 0) {
      return "That file is empty. Please choose a different photo.";
    }
    if (file.size > state.maxBytes) {
      return "That photo is too large. The maximum size is " + formatMegabytes(state.maxBytes) + " MB.";
    }
    return null;
  }

  function showError(message) {
    els.error.textContent = message;
    els.error.hidden = false;
  }

  function clearError() {
    els.error.textContent = "";
    els.error.hidden = true;
  }

  function clearResults() {
    els.results.hidden = true;
    els.topSpecies.textContent = "";
    els.topConfidence.textContent = "";
    els.lowConfidenceNote.hidden = true;
    els.otherResults.replaceChildren();
    els.announcement.textContent = "";
  }

  function releasePreview() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    els.identifyButton.disabled = busy || !state.file;
    els.identifyButton.classList.toggle("is-loading", busy);
    els.identifyButton.setAttribute("aria-busy", String(busy));
    els.fileInput.disabled = busy;
    els.resetButton.disabled = busy;
    els.dropZone.classList.toggle("is-disabled", busy);
    els.dropZone.setAttribute("aria-disabled", String(busy));
    els.loading.hidden = !busy;
  }

  /** Forget the selected photo and hide its preview. */
  function clearSelection() {
    releasePreview();
    state.file = null;
    els.form.reset();
    els.previewImage.removeAttribute("src");
    els.previewImage.alt = "";
    els.previewCaption.textContent = "";
    els.preview.hidden = true;
    els.fileName.textContent = "No photo selected";
    els.identifyButton.disabled = true;
  }

  /** Show the chosen image and enable the Identify button. */
  function selectFile(file) {
    clearError();
    clearResults();
    const problem = validateFile(file);
    if (problem) {
      clearSelection();
      els.resetButton.hidden = true;
      showError(problem);
      return;
    }
    releasePreview();
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);
    els.previewImage.src = state.previewUrl;
    els.previewImage.alt = "Preview of your selected photo: " + file.name;
    els.previewCaption.textContent = "Selected photo: " + file.name;
    els.preview.hidden = false;
    els.fileName.textContent = file.name;
    els.identifyButton.disabled = false;
    els.resetButton.hidden = false;
  }

  /** Return the page to its initial empty state. */
  function reset() {
    clearSelection();
    els.resetButton.hidden = true;
    clearError();
    clearResults();
    setBusy(false);
    els.dropZone.focus();
  }

  function buildResultRow(prediction) {
    const item = document.createElement("li");
    item.className = "result-row";

    const name = document.createElement("span");
    name.className = "result-name";
    name.textContent = prediction.species;

    const percent = document.createElement("span");
    percent.className = "result-percent";
    percent.textContent = formatPercent(prediction.confidence);

    const bar = document.createElement("span");
    bar.className = "bar";
    bar.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = formatPercent(prediction.confidence);
    bar.appendChild(fill);

    item.append(name, percent, bar);
    return item;
  }

  function renderResults(data) {
    const top = data.top_prediction;
    els.topSpecies.textContent = top.species;
    els.topConfidence.textContent = formatPercent(top.confidence);
    els.lowConfidenceNote.hidden = !data.low_confidence;

    const others = Array.isArray(data.predictions) ? data.predictions.slice(1, 5) : [];
    els.otherResults.replaceChildren(...others.map(buildResultRow));
    els.results.hidden = false;
    els.announcement.textContent =
      "Result: " + top.species + ", " + formatPercent(top.confidence) + " confidence." +
      (data.low_confidence ? " The model is not sure about this photo." : "");
    els.resultsHeading.focus();
  }

  function isValidResult(data) {
    return (
      data !== null &&
      typeof data === "object" &&
      data.top_prediction &&
      typeof data.top_prediction.species === "string" &&
      Array.isArray(data.predictions)
    );
  }

  /** Pick a friendly message for a failed response. */
  async function errorMessageFor(response) {
    try {
      const body = await response.json();
      if (body && typeof body.detail === "string" && body.detail.length > 0) {
        return body.detail;
      }
    } catch (_err) {
      // Non-JSON error body; fall through to the default message.
    }
    return FALLBACK_MESSAGES[response.status] || GENERIC_ERROR;
  }

  async function identify() {
    if (!state.file || state.busy) {
      return;
    }
    clearError();
    clearResults();
    setBusy(true);

    const formData = new FormData();
    formData.append("file", state.file, state.file.name);

    try {
      const response = await fetch("/api/identify", { method: "POST", body: formData });
      if (!response.ok) {
        showError(await errorMessageFor(response));
        return;
      }
      const data = await response.json();
      if (!isValidResult(data)) {
        showError(GENERIC_ERROR);
        return;
      }
      renderResults(data);
    } catch (_err) {
      showError(NETWORK_ERROR);
    } finally {
      setBusy(false);
    }
  }

  /** Load server limits and warn if the model is not available. */
  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      if (!response.ok) {
        return;
      }
      const health = await response.json();
      if (Number.isFinite(health.max_upload_bytes) && health.max_upload_bytes > 0) {
        state.maxBytes = health.max_upload_bytes;
        els.maxSize.textContent = formatMegabytes(state.maxBytes);
      }
      if (health.model_loaded === false) {
        els.serviceStatus.textContent =
          "The bird identification model isn't loaded, so identification won't work right now. " +
          "Check the server logs and restart the app.";
        els.serviceStatus.hidden = false;
      }
    } catch (_err) {
      // The upload itself will report connection problems.
    }
  }

  function openFilePicker() {
    if (!state.busy) {
      els.fileInput.click();
    }
  }

  function bindEvents() {
    els.fileInput.addEventListener("change", () => {
      const file = els.fileInput.files && els.fileInput.files[0];
      // Clear the input so choosing the same file again still fires "change".
      els.fileInput.value = "";
      if (file) {
        selectFile(file);
      }
    });

    els.dropZone.addEventListener("click", openFilePicker);
    els.dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openFilePicker();
      }
    });

    ["dragenter", "dragover"].forEach((type) => {
      els.dropZone.addEventListener(type, (event) => {
        event.preventDefault();
        if (!state.busy) {
          els.dropZone.classList.add("is-dragover");
        }
      });
    });
    ["dragleave", "dragend"].forEach((type) => {
      els.dropZone.addEventListener(type, () => els.dropZone.classList.remove("is-dragover"));
    });
    els.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("is-dragover");
      if (state.busy) {
        return;
      }
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || files.length === 0) {
        return;
      }
      if (files.length > 1) {
        showError("Please drop just one photo at a time.");
        return;
      }
      selectFile(files[0]);
    });

    // Stop a photo dropped outside the drop zone from navigating away.
    window.addEventListener("dragover", (event) => event.preventDefault());
    window.addEventListener("drop", (event) => event.preventDefault());

    els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      identify();
    });
    els.resetButton.addEventListener("click", reset);
  }

  bindEvents();
  checkHealth();
})();
