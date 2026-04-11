(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const dropZone = $("dropZone");
  const fileInput = $("fileInput");
  const preview = $("prev");
  const fileName = $("fname");
  const readyTag = $("uploadReady");
  const submitButton = $("btnSubmit");
  const form = $("mainForm");
  const hiddenMode = $("modeInput");
  const hiddenPrimitives = $("primJson");

  const fillParams = $("fillParams");
  const outlineParams = $("outlineParams");
  const modeToggle = $("modeToggle");
  if (hiddenMode) hiddenMode.value = "fill";
  if (fillParams) fillParams.hidden = false;
  if (outlineParams) outlineParams.hidden = true;
  if (modeToggle) modeToggle.hidden = true;

  function setSliderValue(id) {
    const input = $(id);
    const value = $(id + "Val");
    if (!input || !value) return;
    value.textContent = input.value;
    input.addEventListener("input", () => {
      value.textContent = input.value;
    });
  }

  ["numPrims", "imageScale"].forEach(setSliderValue);

  function syncShapeLabels() {
    document.querySelectorAll(".shape-check").forEach((label) => {
      const checkbox = label.querySelector('input[type="checkbox"]');
      if (!checkbox) return;
      label.classList.toggle("active", checkbox.checked);
    });
  }

  const shapeInputs = Array.from(document.querySelectorAll('.shape-check input[type="checkbox"]'));
  shapeInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (!shapeInputs.some((node) => node.checked)) {
        input.checked = true;
      }
      syncShapeLabels();
    });
  });
  syncShapeLabels();

  function attachFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;

    if (fileName) fileName.textContent = file.name;
    if (preview) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
    }
    if (readyTag) {
      readyTag.hidden = false;
      readyTag.textContent = "已选择图片";
    }
    if (dropZone) dropZone.classList.add("ready");
    if (submitButton) submitButton.classList.add("ready");
  }

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) attachFile(fileInput.files[0]);
    });

    dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      dropZone.classList.remove("drag-over");
      if (event.dataTransfer.files && event.dataTransfer.files[0]) {
        attachFile(event.dataTransfer.files[0]);
      }
    });

    document.addEventListener("paste", (event) => {
      const items = (event.clipboardData || window.clipboardData || {}).items || [];
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].type && items[i].type.indexOf("image") !== -1) {
          attachFile(items[i].getAsFile());
          break;
        }
      }
    });
  }

  if (form) {
    form.addEventListener("submit", () => {
      if (hiddenMode) hiddenMode.value = "fill";
      if (hiddenPrimitives) hiddenPrimitives.value = "";
    });
  }
})();
