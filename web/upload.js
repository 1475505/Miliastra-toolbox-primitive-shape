(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const PRESETS = {
    circle: [
      { key: "coin", label: "冒险币", hint: "默认圆形装饰物", name: "冒险币", type_id: 10005009 },
      { key: "geo_badge", label: "岩元素徽章", hint: "低负载常用预设", name: "岩元素徽章", type_id: 20001285, rot_z: 90, rot_y_add: 90 },
      { key: "electro_badge", label: "雷元素徽章", hint: "低负载常用预设", name: "雷元素徽章", type_id: 20001281, rot_z: 90, rot_y_add: 90 },
      { key: "anemo_badge", label: "风元素徽章", hint: "低负载常用预设", name: "风元素徽章", type_id: 20001287, rot_z: 90, rot_y_add: 90 },
      { key: "custom", label: "自定义", hint: "手动填写参数", name: "自定义圆形" },
    ],
    rect: [
      { key: "wood_box", label: "木质箱子", hint: "常用矩形元件", name: "木质箱子", type_id: 20001224 },
      { key: "geo_cube", label: "岩元素立方体", hint: "体块感更强", name: "岩元素立方体", type_id: 20001034 },
      { key: "stone_wall", label: "石质墙体(黄)", hint: "适合描边堆叠", name: "石质墙体(黄)", type_id: 20001869 },
      { key: "green_platform", label: "积木平台(绿)", hint: "大尺寸矩形平台", name: "积木平台(绿)", type_id: 10005014 },
      { key: "custom", label: "自定义", hint: "手动填写参数", name: "自定义矩形" },
    ],
  };

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
  const outlineLink = $("outlineLink");
  const topbarSubtitle = document.querySelector(".topbar-subtitle");
  const shapeSectionTitle = $("shapeSectionTitle");
  const fillShapeSection = $("fillShapeSection");
  const primitiveListSection = $("primitiveListSection");
  const primitiveList = $("primitiveList");
  const primitiveEmpty = $("primitiveEmpty");
  const primitiveCountHint = $("primitiveCountHint");
  const addCirclePrimitiveBtn = $("addCirclePrimitiveBtn");
  const addRectPrimitiveBtn = $("addRectPrimitiveBtn");
  const shapeHint = $("shapeHint");

  let currentMode = "fill";

  function getPresetList(shape) {
    return PRESETS[shape] || PRESETS.circle;
  }

  function getPreset(shape, key) {
    return getPresetList(shape).find((preset) => preset.key === key) || getPresetList(shape)[0];
  }

  function inferPresetKey(shape, config) {
    if (config.preset_key) return config.preset_key;
    const presets = getPresetList(shape);
    const matched = presets.find((preset) => {
      if (preset.key === "custom") return false;
      return (
        Number(preset.type_id || 0) === Number(config.type_id || 0) &&
        Number(preset.rot_z || 0) === Number(config.rot_z || 0) &&
        Number(preset.rot_y_add || 0) === Number(config.rot_y_add || 0)
      );
    });
    return matched ? matched.key : "custom";
  }

  function fillNumberField(input, value) {
    input.value = value === undefined || value === null || value === "" ? "" : String(value);
  }

  function updatePrimitiveEmptyState() {
    if (!primitiveList || !primitiveEmpty || !primitiveCountHint) return;
    const count = primitiveList.querySelectorAll(".primitive-card").length;
    primitiveEmpty.hidden = count > 0;
    primitiveCountHint.textContent = count > 0
      ? `当前共 ${count} 个装饰物元件，导出时会保留类型 ID 与旋转参数。`
      : "建议至少保留一种元件类型。";
  }

  function updatePrimitiveCardTone(card) {
    const shape = card.querySelector('[data-field="shape"]').value;
    const badge = card.querySelector(".primitive-badge");
    if (!badge) return;
    badge.textContent = shape === "rect" ? "矩形元件" : "圆形元件";
    badge.dataset.shape = shape;
  }

  function updatePresetOptions(card, nextPresetKey) {
    const shape = card.querySelector('[data-field="shape"]').value;
    const presetSelect = card.querySelector('[data-field="preset"]');
    const presetOptions = getPresetList(shape);
    presetSelect.innerHTML = presetOptions
      .map((preset) => `<option value="${preset.key}">${preset.label}</option>`)
      .join("");
    presetSelect.value = nextPresetKey && presetOptions.some((preset) => preset.key === nextPresetKey)
      ? nextPresetKey
      : presetOptions[0].key;
    updatePrimitiveCardTone(card);
  }

  function updatePrimitiveMeta(card) {
    const shape = card.querySelector('[data-field="shape"]').value;
    const presetKey = card.querySelector('[data-field="preset"]').value;
    const preset = getPreset(shape, presetKey);
    const meta = card.querySelector(".primitive-meta");
    if (!meta) return;
    meta.textContent = preset && preset.hint
      ? `${preset.hint}。可以继续微调类型 ID、资源 ID 和旋转。`
      : "当前参数将写入装饰物导出结果。";
  }

  function applyPreset(card, presetKey, preserveName) {
    const shape = card.querySelector('[data-field="shape"]').value;
    const preset = getPreset(shape, presetKey);
    const nameInput = card.querySelector('[data-field="name"]');
    const assetInput = card.querySelector('[data-field="image_asset_ref"]');
    const typeIdInput = card.querySelector('[data-field="type_id"]');
    const elementTypeIdInput = card.querySelector('[data-field="element_type_id"]');
    const rotZInput = card.querySelector('[data-field="rot_z"]');
    const rotYInput = card.querySelector('[data-field="rot_y_add"]');

    if (!preserveName || !nameInput.value.trim()) {
      nameInput.value = preset.name || (shape === "rect" ? "自定义矩形" : "自定义圆形");
    }
    fillNumberField(assetInput, preset.image_asset_ref);
    fillNumberField(typeIdInput, preset.type_id);
    fillNumberField(elementTypeIdInput, preset.element_type_id);
    fillNumberField(rotZInput, preset.rot_z);
    fillNumberField(rotYInput, preset.rot_y_add);
    updatePrimitiveMeta(card);
  }

  function createPrimitiveField(label, field) {
    const wrap = document.createElement("label");
    wrap.className = "primitive-field";

    const title = document.createElement("span");
    title.className = "primitive-field-label";
    title.textContent = label;

    wrap.appendChild(title);
    wrap.appendChild(field);
    return wrap;
  }

  function createPrimitiveRow(config) {
    const initialShape = config.shape === "rect" ? "rect" : "circle";
    const card = document.createElement("article");
    card.className = "primitive-card";

    const head = document.createElement("div");
    head.className = "primitive-card-head";

    const badge = document.createElement("span");
    badge.className = "primitive-badge";

    const presetSelect = document.createElement("select");
    presetSelect.className = "form-select";
    presetSelect.dataset.field = "preset";

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "btn-chip primitive-remove";
    removeBtn.textContent = "删除";

    head.appendChild(badge);
    head.appendChild(presetSelect);
    head.appendChild(removeBtn);

    const grid = document.createElement("div");
    grid.className = "primitive-grid";

    const shapeSelect = document.createElement("select");
    shapeSelect.className = "form-select";
    shapeSelect.dataset.field = "shape";
    shapeSelect.innerHTML = `
      <option value="circle">圆形</option>
      <option value="rect">矩形</option>
    `;
    shapeSelect.value = initialShape;

    const nameInput = document.createElement("input");
    nameInput.type = "text";
    nameInput.className = "form-input";
    nameInput.dataset.field = "name";
    nameInput.placeholder = "元件名称";

    const assetInput = document.createElement("input");
    assetInput.type = "number";
    assetInput.className = "form-input";
    assetInput.dataset.field = "image_asset_ref";
    assetInput.placeholder = "例如 100002";

    const typeIdInput = document.createElement("input");
    typeIdInput.type = "number";
    typeIdInput.className = "form-input";
    typeIdInput.dataset.field = "type_id";
    typeIdInput.placeholder = "例如 20001285";

    const elementTypeIdInput = document.createElement("input");
    elementTypeIdInput.type = "number";
    elementTypeIdInput.className = "form-input";
    elementTypeIdInput.dataset.field = "element_type_id";
    elementTypeIdInput.placeholder = "留空则跟随类型 ID";

    const rotZInput = document.createElement("input");
    rotZInput.type = "number";
    rotZInput.step = "1";
    rotZInput.className = "form-input";
    rotZInput.dataset.field = "rot_z";
    rotZInput.placeholder = "0";

    const rotYInput = document.createElement("input");
    rotYInput.type = "number";
    rotYInput.step = "1";
    rotYInput.className = "form-input";
    rotYInput.dataset.field = "rot_y_add";
    rotYInput.placeholder = "0";

    [
      createPrimitiveField("形状", shapeSelect),
      createPrimitiveField("名称", nameInput),
      createPrimitiveField("图片资源 ID", assetInput),
      createPrimitiveField("类型 ID", typeIdInput),
      createPrimitiveField("元件类型 ID", elementTypeIdInput),
      createPrimitiveField("Z 轴旋转", rotZInput),
      createPrimitiveField("Y 轴附加旋转", rotYInput),
    ].forEach((field) => grid.appendChild(field));

    const meta = document.createElement("p");
    meta.className = "primitive-meta";

    card.appendChild(head);
    card.appendChild(grid);
    card.appendChild(meta);

    const presetKey = inferPresetKey(initialShape, config);
    updatePresetOptions(card, presetKey);
    applyPreset(card, presetKey, false);

    if (config.name) nameInput.value = config.name;
    fillNumberField(assetInput, config.image_asset_ref);
    fillNumberField(typeIdInput, config.type_id);
    fillNumberField(elementTypeIdInput, config.element_type_id);
    fillNumberField(rotZInput, config.rot_z);
    fillNumberField(rotYInput, config.rot_y_add);
    updatePrimitiveMeta(card);

    shapeSelect.addEventListener("change", () => {
      updatePresetOptions(card);
      applyPreset(card, card.querySelector('[data-field="preset"]').value, false);
      updatePrimitivesJson();
    });

    presetSelect.addEventListener("change", () => {
      applyPreset(card, presetSelect.value, false);
      updatePrimitivesJson();
    });

    [nameInput, assetInput, typeIdInput, elementTypeIdInput, rotZInput, rotYInput].forEach((input) => {
      input.addEventListener("input", updatePrimitivesJson);
      input.addEventListener("change", updatePrimitivesJson);
    });

    removeBtn.addEventListener("click", () => {
      card.remove();
      updatePrimitivesJson();
    });

    return card;
  }

  function addPrimitiveConfig(config) {
    if (!primitiveList) return;
    primitiveList.appendChild(createPrimitiveRow(config || { shape: "circle" }));
    updatePrimitivesJson();
  }

  function readOutlinePrimitives() {
    if (!primitiveList) return [];
    return Array.from(primitiveList.querySelectorAll(".primitive-card")).map((card) => {
      const primitive = {
        shape: card.querySelector('[data-field="shape"]').value,
        color: "#ffffff",
      };

      const name = card.querySelector('[data-field="name"]').value.trim();
      const imageAssetRef = card.querySelector('[data-field="image_asset_ref"]').value.trim();
      const typeId = card.querySelector('[data-field="type_id"]').value.trim();
      const elementTypeId = card.querySelector('[data-field="element_type_id"]').value.trim();
      const rotZ = card.querySelector('[data-field="rot_z"]').value.trim();
      const rotYAdd = card.querySelector('[data-field="rot_y_add"]').value.trim();

      if (name) primitive.name = name;
      if (imageAssetRef) primitive.image_asset_ref = Number(imageAssetRef);
      if (typeId) primitive.type_id = Number(typeId);
      if (elementTypeId) primitive.element_type_id = Number(elementTypeId);
      if (rotZ) primitive.rot_z = Number(rotZ);
      if (rotYAdd) primitive.rot_y_add = Number(rotYAdd);

      return primitive;
    });
  }

  function updatePrimitivesJson() {
    if (!hiddenPrimitives) return;

    const circleCheckbox = $("shapeCircle");
    const rectCheckbox = $("shapeRect");
    const triangleCheckbox = $("shapeTriangle");
    let primitives = [];

    if (currentMode === "outline") {
      primitives = readOutlinePrimitives();
    } else {
      if (circleCheckbox && circleCheckbox.checked) primitives.push({ shape: "circle", color: "#ffffff" });
      if (rectCheckbox && rectCheckbox.checked) primitives.push({ shape: "rect", color: "#ffffff" });
      if (triangleCheckbox && triangleCheckbox.checked) primitives.push({ shape: "triangle", color: "#ffffff" });
    }

    hiddenPrimitives.value = JSON.stringify(primitives);
    updatePrimitiveEmptyState();
  }

  function ensureOutlineDefault() {
    if (!primitiveList) return;
    if (!primitiveList.querySelector(".primitive-card")) {
      addPrimitiveConfig({ shape: "circle", preset_key: "coin" });
    }
  }

  function setMode(mode) {
    currentMode = mode;
    if (hiddenMode) hiddenMode.value = mode;
    if (fillParams) fillParams.hidden = mode !== "fill";
    if (outlineParams) outlineParams.hidden = mode !== "outline";
    if (fillShapeSection) fillShapeSection.hidden = mode !== "fill";
    if (primitiveListSection) primitiveListSection.hidden = mode !== "outline";

    if (shapeSectionTitle) {
      shapeSectionTitle.textContent = mode === "fill" ? "图元类型" : "装饰物参数";
    }
    if (topbarSubtitle) {
      topbarSubtitle.textContent = mode === "fill"
        ? "默认填充模式 · 默认仅圆形"
        : "装饰物拟合模式 · 使用元件参数生成轮廓";
    }
    if (outlineLink) {
      outlineLink.textContent = mode === "fill" ? "装饰物拟合" : "填充模式";
    }
    if (shapeHint) {
      shapeHint.textContent = mode === "fill"
        ? "默认只启用圆形；需要时再叠加矩形或三角形。"
        : "";
    }

    if (mode === "outline") ensureOutlineDefault();
    updatePrimitivesJson();
    syncShapeLabels();
  }

  function setSliderValue(id, formatter) {
    const input = $(id);
    const value = $(id + "Val");
    if (!input || !value) return;
    const renderValue = () => {
      value.textContent = formatter ? formatter(input.value) : input.value;
    };
    renderValue();
    input.addEventListener("input", renderValue);
  }

  function syncShapeLabels() {
    const fillShapeInputs = document.querySelectorAll("#fillShapeSection .shape-check input[type='checkbox']");
    fillShapeInputs.forEach((checkbox) => {
      const label = checkbox.closest(".shape-check");
      if (label) label.classList.toggle("active", checkbox.checked);
    });
  }

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

  setSliderValue("numPrims");
  setSliderValue("imageScale");
  setSliderValue("outputAlpha", (value) => value + "%");
  ["olPrimSize", "olSpacing", "olPrecision"].forEach((id) => setSliderValue(id));

  const numPrimsSlider = $("numPrims");
  const numPrimsManual = $("numPrimsManual");
  const numPrimsVal = $("numPrimsVal");
  if (numPrimsSlider && numPrimsManual) {
    numPrimsSlider.addEventListener("input", () => {
      numPrimsManual.value = numPrimsSlider.value;
    });
    numPrimsManual.addEventListener("input", () => {
      const value = Number.parseInt(numPrimsManual.value, 10);
      if (!Number.isNaN(value) && value >= 40 && value <= 2000) {
        numPrimsSlider.value = String(value);
      }
      if (numPrimsVal) numPrimsVal.textContent = numPrimsManual.value;
    });
  }

  if (outlineLink) {
    outlineLink.addEventListener("click", (event) => {
      event.preventDefault();
      setMode(currentMode === "fill" ? "outline" : "fill");
    });
  }

  if (addCirclePrimitiveBtn) {
    addCirclePrimitiveBtn.addEventListener("click", () => addPrimitiveConfig({ shape: "circle", preset_key: "coin" }));
  }

  if (addRectPrimitiveBtn) {
    addRectPrimitiveBtn.addEventListener("click", () => addPrimitiveConfig({ shape: "rect", preset_key: "wood_box" }));
  }

  const fillShapeInputs = Array.from(document.querySelectorAll("#fillShapeSection .shape-check input[type='checkbox']"));
  fillShapeInputs.forEach((input) => {
    input.addEventListener("change", () => {
      if (!fillShapeInputs.some((node) => node.checked)) {
        input.checked = true;
      }
      syncShapeLabels();
      updatePrimitivesJson();
    });
  });

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
      if (hiddenMode && !hiddenMode.value) hiddenMode.value = "fill";
      updatePrimitivesJson();
    });
  }

  syncShapeLabels();
  setMode("fill");
})();
