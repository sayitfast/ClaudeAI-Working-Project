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

  // The server shrinks every photo to fit within 1024 px, so a larger crop
  // would only make the upload bigger.
  const CROP_MAX_OUTPUT_PX = 1024;
  const CROP_JPEG_QUALITY = 0.92;
  // Smallest crop box on screen, in pixels, so the corner handles never overlap.
  const CROP_MIN_PX = 48;
  // How far one arrow-key press moves or resizes the crop box, in screen pixels.
  const CROP_KEY_STEP_PX = 10;
  // A drag on the photo must travel this far before it starts drawing a new box.
  const CROP_DRAW_THRESHOLD_PX = 4;
  // Starting crop box: the middle 80% of the photo. Crop boxes are stored as
  // fractions (0-1) of the photo's size so they survive layout changes.
  const DEFAULT_CROP = Object.freeze({ x: 0.1, y: 0.1, w: 0.8, h: 0.8 });
  const ARROW_KEYS = new Map([
    ["ArrowLeft", [-1, 0]],
    ["ArrowRight", [1, 0]],
    ["ArrowUp", [0, -1]],
    ["ArrowDown", [0, 1]],
  ]);

  const FALLBACK_MESSAGES = {
    400: "That photo couldn't be read. Please try a different image.",
    413: "That photo is too large. Please choose a smaller image.",
    415: "That file type isn't supported. Please use a JPEG, PNG, or WEBP photo.",
    500: "Something went wrong while identifying the bird. Please try again.",
    503: "The bird identification model isn't ready right now. Please try again later.",
  };
  const NETWORK_ERROR = "Couldn't reach the server. Check that the app is still running and try again.";
  const GENERIC_ERROR = "Something unexpected happened. Please try again.";
  const CROP_ERROR = "This photo couldn't be cropped in your browser. You can still identify it without cropping.";

  const els = {
    form: document.getElementById("upload-form"),
    dropZone: document.getElementById("drop-zone"),
    fileInput: document.getElementById("file-input"),
    fileName: document.getElementById("file-name"),
    maxSize: document.getElementById("max-size"),
    preview: document.getElementById("preview"),
    previewImage: document.getElementById("preview-image"),
    previewCaption: document.getElementById("preview-caption"),
    cropStage: document.getElementById("crop-stage"),
    cropLayer: document.getElementById("crop-layer"),
    cropBox: document.getElementById("crop-box"),
    cropHelp: document.getElementById("crop-help"),
    cropStatus: document.getElementById("crop-status"),
    cropControls: document.getElementById("crop-controls"),
    cropButton: document.getElementById("crop-button"),
    applyCropButton: document.getElementById("apply-crop-button"),
    cancelCropButton: document.getElementById("cancel-crop-button"),
    undoCropButton: document.getElementById("undo-crop-button"),
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
    file: null, // The photo that will be uploaded: the original or its cropped copy.
    originalFile: null,
    originalUrl: null,
    croppedUrl: null,
    cropRect: null, // The applied crop box, or null when the full photo is used.
    cropping: false,
    cropWorking: false,
    draft: null, // The crop box being edited while cropping.
    drag: null,
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

  function releaseCroppedPreview() {
    if (state.croppedUrl) {
      URL.revokeObjectURL(state.croppedUrl);
      state.croppedUrl = null;
    }
  }

  function releasePreview() {
    releaseCroppedPreview();
    if (state.originalUrl) {
      URL.revokeObjectURL(state.originalUrl);
      state.originalUrl = null;
    }
  }

  /** Show, hide, enable, and disable the buttons to match the current state. */
  function updateControls() {
    const hasPhoto = state.originalFile !== null;
    const isCropped = hasPhoto && state.file !== state.originalFile;
    els.identifyButton.disabled = state.busy || state.cropping || !state.file;
    els.cropControls.hidden = !hasPhoto;
    els.cropHelp.hidden = !state.cropping;
    els.cropButton.hidden = state.cropping;
    els.cropButton.textContent = isCropped ? "Change crop" : "Crop photo";
    els.cropButton.disabled = state.busy;
    els.applyCropButton.hidden = !state.cropping;
    els.applyCropButton.disabled = state.cropWorking;
    els.cancelCropButton.hidden = !state.cropping;
    els.undoCropButton.hidden = state.cropping || !isCropped;
    els.undoCropButton.disabled = state.busy;
  }

  function setBusy(busy) {
    state.busy = busy;
    updateControls();
    els.identifyButton.classList.toggle("is-loading", busy);
    els.identifyButton.setAttribute("aria-busy", String(busy));
    els.fileInput.disabled = busy;
    els.resetButton.disabled = busy;
    els.dropZone.classList.toggle("is-disabled", busy);
    els.dropZone.setAttribute("aria-disabled", String(busy));
    els.loading.hidden = !busy;
  }

  /** Show the original photo or its cropped copy in the preview. */
  function showPreview(useCropped) {
    const name = state.originalFile.name;
    const url = useCropped ? state.croppedUrl : state.originalUrl;
    if (els.previewImage.getAttribute("src") !== url) {
      els.previewImage.src = url;
    }
    els.previewImage.alt = (useCropped ? "Cropped preview of your selected photo: " : "Preview of your selected photo: ") + name;
    els.previewCaption.textContent = (useCropped ? "Cropped photo: " : "Selected photo: ") + name;
    els.fileName.textContent = state.file === state.originalFile ? name : name + " (cropped)";
    els.preview.hidden = false;
  }

  /** Leave crop mode without changing which photo will be uploaded. */
  function stopCropping() {
    state.cropping = false;
    state.draft = null;
    state.drag = null;
    els.cropLayer.hidden = true;
  }

  /** Forget the selected photo and hide its preview. */
  function clearSelection() {
    stopCropping();
    releasePreview();
    state.file = null;
    state.originalFile = null;
    state.cropRect = null;
    els.form.reset();
    els.previewImage.removeAttribute("src");
    els.previewImage.alt = "";
    els.previewCaption.textContent = "";
    els.preview.hidden = true;
    els.fileName.textContent = "No photo selected";
    els.cropStatus.textContent = "";
    updateControls();
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
    stopCropping();
    releasePreview();
    state.file = file;
    state.originalFile = file;
    state.cropRect = null;
    state.originalUrl = URL.createObjectURL(file);
    showPreview(false);
    els.cropStatus.textContent = "";
    updateControls();
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

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  /** Return the smallest allowed crop box, as fractions of the photo's on-screen size. */
  function minCropSize(bounds) {
    return {
      w: bounds.width > 0 ? Math.min(1, CROP_MIN_PX / bounds.width) : 1,
      h: bounds.height > 0 ? Math.min(1, CROP_MIN_PX / bounds.height) : 1,
    };
  }

  /** Grow a box to the minimum size if needed and keep it inside the photo. */
  function fitRect(rect, min) {
    const w = clamp(rect.w, min.w, 1);
    const h = clamp(rect.h, min.h, 1);
    return { x: clamp(rect.x, 0, 1 - w), y: clamp(rect.y, 0, 1 - h), w: w, h: h };
  }

  function moveRect(rect, dx, dy, min) {
    return fitRect({ x: rect.x + dx, y: rect.y + dy, w: rect.w, h: rect.h }, min);
  }

  /** Drag one corner of a box ("nw", "ne", "sw" or "se") while the opposite corner stays put. */
  function resizeRect(rect, corner, dx, dy, min) {
    let left = rect.x;
    let top = rect.y;
    let right = rect.x + rect.w;
    let bottom = rect.y + rect.h;
    if (corner.includes("w")) {
      left = clamp(left + dx, 0, right - min.w);
    }
    if (corner.includes("e")) {
      right = clamp(right + dx, left + min.w, 1);
    }
    if (corner.includes("n")) {
      top = clamp(top + dy, 0, bottom - min.h);
    }
    if (corner.includes("s")) {
      bottom = clamp(bottom + dy, top + min.h, 1);
    }
    return fitRect({ x: left, y: top, w: right - left, h: bottom - top }, min);
  }

  /** Return the box with corners at two points. */
  function rectBetween(a, b, min) {
    return fitRect(
      { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(b.x - a.x), h: Math.abs(b.y - a.y) },
      min
    );
  }

  /** Convert a pointer position to fractions of the photo's on-screen size. */
  function pointInPhoto(event, bounds) {
    return {
      x: clamp((event.clientX - bounds.left) / bounds.width, 0, 1),
      y: clamp((event.clientY - bounds.top) / bounds.height, 0, 1),
    };
  }

  function describeCrop(rect) {
    return (
      "Crop area is " + Math.round(rect.w * 100) + "% of the photo's width and " +
      Math.round(rect.h * 100) + "% of its height."
    );
  }

  /** Place the crop layer exactly over the photo and draw the crop box inside it. */
  function renderCropBox() {
    if (!state.cropping) {
      return;
    }
    const image = els.previewImage;
    const layer = els.cropLayer.style;
    layer.left = image.offsetLeft + "px";
    layer.top = image.offsetTop + "px";
    layer.width = image.offsetWidth + "px";
    layer.height = image.offsetHeight + "px";
    const box = els.cropBox.style;
    box.left = state.draft.x * 100 + "%";
    box.top = state.draft.y * 100 + "%";
    box.width = state.draft.w * 100 + "%";
    box.height = state.draft.h * 100 + "%";
  }

  /** Show the full photo with a crop box the user can move and resize. */
  function startCrop() {
  if (!state.originalFile || state.busy || state.cropping || state.cropWorking) {
      return;
    }
    clearError();
    state.cropping = true;
    state.draft = Object.assign({}, state.cropRect || DEFAULT_CROP);
    showPreview(false);
    els.cropLayer.hidden = false;
    updateControls();
    renderCropBox();
    els.cropBox.focus();
  }

  /** Leave crop mode and keep using whichever photo was chosen before. */
  function cancelCrop() {
    if (!state.cropping) {
      return;
    }
    stopCropping();
    showPreview(state.croppedUrl !== null);
    updateControls();
    els.cropButton.focus();
  }

  /** Name the cropped copy after the original, e.g. "robin.png" becomes "robin-cropped.jpg". */
  function croppedName(name) {
    const dot = name.lastIndexOf(".");
    return (dot > 0 ? name.slice(0, dot) : name) + "-cropped.jpg";
  }

  /** Copy the boxed part of a loaded image onto a canvas and encode it as a JPEG file. */
  async function cropToFile(image, rect, name) {
    const sourceWidth = image.naturalWidth;
    const sourceHeight = image.naturalHeight;
    if (!sourceWidth || !sourceHeight) {
      throw new Error("The photo has not been decoded.");
    }
    const sx = Math.round(rect.x * sourceWidth);
    const sy = Math.round(rect.y * sourceHeight);
    const sw = clamp(Math.round(rect.w * sourceWidth), 1, sourceWidth - sx);
    const sh = clamp(Math.round(rect.h * sourceHeight), 1, sourceHeight - sy);
    const scale = Math.min(1, CROP_MAX_OUTPUT_PX / Math.max(sw, sh));

    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(sw * scale));
    canvas.height = Math.max(1, Math.round(sh * scale));
    const context = canvas.getContext("2d");
    context.imageSmoothingQuality = "high";
    context.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", CROP_JPEG_QUALITY);
    });
    if (!blob) {
      throw new Error("The cropped photo could not be encoded.");
    }
    return new File([blob], name, { type: "image/jpeg" });
  }

  /** Cut the boxed area out of the original photo and use it for identification. */
  async function applyCrop() {
    if (!state.cropping || state.cropWorking) {
      return;
    }
    const original = state.originalFile;
    const rect = Object.assign({}, state.draft);
    state.cropWorking = true;
    updateControls();
    let cropped = null;
    try {
      // The preview shows the original photo while cropping, so it is the source.
      await els.previewImage.decode();
      cropped = await cropToFile(els.previewImage, rect, croppedName(original.name));
    } catch (_err) {
      // Reported below.
    }
    state.cropWorking = false;

    // The user may have cancelled or chosen another photo while this ran.
    if (!state.cropping || state.originalFile !== original) {
      updateControls();
      return;
    }
    const problem = cropped ? validateFile(cropped) : CROP_ERROR;
    if (problem) {
      updateControls();
      showError(problem);
      return;
    }
    stopCropping();
    releaseCroppedPreview();
    state.file = cropped;
    state.cropRect = rect;
    state.croppedUrl = URL.createObjectURL(cropped);
    clearError();
    clearResults();
    showPreview(true);
    updateControls();
    els.cropStatus.textContent = "Crop applied. The cropped photo will be used to identify the bird.";
    els.cropButton.focus();
  }

  /** Go back to uploading the full, uncropped photo. */
  function undoCrop() {
    if (state.cropping || state.busy || state.file === state.originalFile) {
      return;
    }
    state.file = state.originalFile;
    state.cropRect = null;
    showPreview(false);
    releaseCroppedPreview();
    clearError();
    clearResults();
    updateControls();
    els.cropStatus.textContent = "Crop removed. The full photo will be used to identify the bird.";
    els.cropButton.focus();
  }

  function onCropPointerDown(event) {
    if (!state.cropping || state.drag || event.button !== 0) {
      return;
    }
    const bounds = els.cropLayer.getBoundingClientRect();
    if (bounds.width === 0 || bounds.height === 0) {
      return;
    }
    event.preventDefault();
    let mode = "draw";
    if (event.target.dataset.handle) {
      mode = event.target.dataset.handle;
    } else if (event.target === els.cropBox) {
      mode = "move";
    }
    state.drag = {
      mode: mode,
      pointerId: event.pointerId,
      bounds: bounds,
      startX: event.clientX,
      startY: event.clientY,
      startRect: Object.assign({}, state.draft),
      anchor: pointInPhoto(event, bounds),
      drawing: false,
    };
    els.cropLayer.setPointerCapture(event.pointerId);
    els.cropBox.focus({ preventScroll: true });
  }

  function onCropPointerMove(event) {
    const drag = state.drag;
    if (!drag || event.pointerId !== drag.pointerId) {
      return;
    }
    const offsetX = event.clientX - drag.startX;
    const offsetY = event.clientY - drag.startY;
    const dx = offsetX / drag.bounds.width;
    const dy = offsetY / drag.bounds.height;
    const min = minCropSize(drag.bounds);
    if (drag.mode === "move") {
      state.draft = moveRect(drag.startRect, dx, dy, min);
    } else if (drag.mode === "draw") {
      if (!drag.drawing && Math.hypot(offsetX, offsetY) < CROP_DRAW_THRESHOLD_PX) {
        return;
      }
      drag.drawing = true;
      state.draft = rectBetween(drag.anchor, pointInPhoto(event, drag.bounds), min);
    } else {
      state.draft = resizeRect(drag.startRect, drag.mode, dx, dy, min);
    }
    renderCropBox();
  }

  function endCropDrag(event) {
    if (state.drag && event.pointerId === state.drag.pointerId) {
      state.drag = null;
    }
  }

  function onCropKeydown(event) {
    if (!state.cropping) {
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      applyCrop();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      cancelCrop();
      return;
    }
    const direction = ARROW_KEYS.get(event.key);
    const bounds = els.cropLayer.getBoundingClientRect();
    if (!direction || bounds.width === 0 || bounds.height === 0) {
      return;
    }
    event.preventDefault();
    const dx = (direction[0] * CROP_KEY_STEP_PX) / bounds.width;
    const dy = (direction[1] * CROP_KEY_STEP_PX) / bounds.height;
    const min = minCropSize(bounds);
    state.draft = event.shiftKey
      ? resizeRect(state.draft, "se", dx, dy, min)
      : moveRect(state.draft, dx, dy, min);
    renderCropBox();
    els.cropStatus.textContent = describeCrop(state.draft);
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
    if (!state.file || state.busy || state.cropping) {
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

    els.cropButton.addEventListener("click", startCrop);
    els.applyCropButton.addEventListener("click", applyCrop);
    els.cancelCropButton.addEventListener("click", cancelCrop);
    els.undoCropButton.addEventListener("click", undoCrop);
    els.cropBox.addEventListener("keydown", onCropKeydown);
    els.cropLayer.addEventListener("pointerdown", onCropPointerDown);
    els.cropLayer.addEventListener("pointermove", onCropPointerMove);
    els.cropLayer.addEventListener("pointerup", endCropDrag);
    els.cropLayer.addEventListener("pointercancel", endCropDrag);
    // Keep the crop layer over the photo when the photo or the page is resized.
    const resizeObserver = new ResizeObserver(renderCropBox);
    resizeObserver.observe(els.previewImage);
    resizeObserver.observe(els.cropStage);

    els.form.addEventListener("submit", (event) => {
      event.preventDefault();
      identify();
    });
    els.resetButton.addEventListener("click", reset);
  }

  bindEvents();
  checkHealth();
})();
