(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const PRESETS = {
    circle: [
      { key: "coin", label: "冒险币", hint: "默认圆形装饰物", name: "冒险币", type_id: 10005009, size: 1.0 },
      { key: "electro_badge", label: "雷元素徽章", hint: "低负载常用预设", name: "雷元素徽章", type_id: 20001281, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "pyro_badge", label: "火元素徽章", hint: "低负载常用预设", name: "火元素徽章", type_id: 20001282, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "dendro_badge", label: "草元素徽章", hint: "低负载常用预设", name: "草元素徽章", type_id: 20001283, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "cryo_badge", label: "冰元素徽章", hint: "低负载常用预设", name: "冰元素徽章", type_id: 20001284, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "geo_badge", label: "岩元素徽章", hint: "低负载常用预设", name: "岩元素徽章", type_id: 20001285, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "hydro_badge", label: "水元素徽章", hint: "低负载常用预设", name: "水元素徽章", type_id: 20001286, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "anemo_badge", label: "风元素徽章", hint: "低负载常用预设", name: "风元素徽章", type_id: 20001287, size: 0.3, rot_z: 90, rot_y_add: 90 },
      { key: "custom", label: "自定义", hint: "手动填写参数", name: "自定义圆形" },
    ],
    rect: [
      { key: "wood_box", label: "木质箱子", hint: "常用矩形元件", name: "木质箱子", type_id: 20001224, size: 1.0 },
      { key: "geo_cube", label: "石质元素立方体", hint: "体块感更强", name: "石质元素立方体", type_id: 20001034, size: 5.0 },
      { key: "wood_box_green", label: "木质箱子（绿）", hint: "彩色箱体预设", name: "木质箱子（绿）", type_id: 20001237, size: 1.5 },
      { key: "wood_box_blue", label: "木质箱子（蓝）", hint: "彩色箱体预设", name: "木质箱子（蓝）", type_id: 20001238, size: 1.5 },
      { key: "wood_box_purple", label: "木质箱子（紫）", hint: "彩色箱体预设", name: "木质箱子（紫）", type_id: 20001239, size: 1.5 },
      { key: "stone_wall_yellow", label: "石质墙体（黄）", hint: "适合描边堆叠", name: "石质墙体（黄）", type_id: 20001869, size: 3.0 },
      { key: "stone_wall_red", label: "石质墙体（红）", hint: "适合描边堆叠", name: "石质墙体（红）", type_id: 20001870, size: 3.0 },
      { key: "stone_wall_gray", label: "石质墙体（灰）", hint: "适合描边堆叠", name: "石质墙体（灰）", type_id: 20001872, size: 3.0 },
      { key: "water_cube", label: "水质立方体", hint: "常用立方体预设", name: "水质立方体", type_id: 20001874, size: 1.0 },
      { key: "cream_cube", label: "通常立方体（奶黄）", hint: "常用立方体预设", name: "通常立方体（奶黄）", type_id: 20001875, size: 1.0 },
      { key: "solid_cube_dark_blue", label: "坚固立方体（暗蓝）", hint: "常用立方体预设", name: "坚固立方体（暗蓝）", type_id: 20001876, size: 1.0 },
      { key: "ice_cube", label: "冰质立方体", hint: "常用立方体预设", name: "冰质立方体", type_id: 20001877, size: 1.0 },
      { key: "fire_cube", label: "火质立方体", hint: "常用立方体预设", name: "火质立方体", type_id: 20001878, size: 1.0 },
      { key: "electro_cube", label: "雷质立方体", hint: "常用立方体预设", name: "雷质立方体", type_id: 20001879, size: 1.0 },
      { key: "wood_low_cabinet", label: "矩形木质矮柜", hint: "细长矩形元件", name: "矩形木质矮柜", type_id: 20001082, size: 1.0 },
      { key: "block_cube_wood", label: "积木立方体（木色）", hint: "大尺寸积木元件", name: "积木立方体（木色）", type_id: 20001096, size: 6.0 },
      { key: "block_cube_dark", label: "积木立方体（深色）", hint: "大尺寸积木元件", name: "积木立方体（深色）", type_id: 20001097, size: 6.0 },
      { key: "block_cube_light", label: "积木立方体（浅色）", hint: "大尺寸积木元件", name: "积木立方体（浅色）", type_id: 20001100, size: 6.0 },
      { key: "stone_ceiling_white", label: "石质天花板（白）", hint: "大尺寸矩形平台", name: "石质天花板（白）", type_id: 20002146, size: 5.0 },
      { key: "wood_ceiling_black", label: "木质天花板（黑）", hint: "大尺寸矩形平台", name: "木质天花板（黑）", type_id: 20002121, size: 5.0 },
      { key: "green_platform", label: "积木平台（绿）", hint: "大尺寸矩形平台", name: "积木平台（绿）", type_id: 10005014, size: 5.0 },
      { key: "custom", label: "自定义", hint: "手动填写参数", name: "自定义矩形" },
    ],
  };

  const dropZone = $("dropZone");
  const fileInput = $("fileInput");
  const preview = $("prev");
  const fileName = $("fname");
  const imgSize = $("imgSize");
  const readyTag = $("uploadReady");
  const uploadWarning = $("uploadWarning");
  const submitButton = $("btnSubmit");
  const form = $("mainForm");
  const hiddenMode = $("modeInput");
  const hiddenPrimitives = $("primJson");

  const imageToolTab = $("imageToolTab");
  const classicToolTab = $("classicToolTab");
  const imageToolPage = $("imageToolPage");
  const classicToolPage = $("classicToolPage");
  const classicGiaForm = $("classicGiaForm");
  const classicDropZone = $("classicDropZone");
  const classicGiaInput = $("classicGiaInput");
  const classicGiaName = $("classicGiaName");
  const classicGiaReady = $("classicGiaReady");
  const classicGiaButton = $("btnConvertClassicGia");
  const giaDirectionInput = $("giaDirectionInput");
  const dirOverToClassic = $("dirOverToClassic");
  const dirClassicToOver = $("dirClassicToOver");
  const classicUploadTitle = $("classicUploadTitle");
  const classicDropText = $("classicDropText");
  const classicHint = $("classicHint");
  const classicToolTitle = $("classicToolTitle");
  const classicToolDesc = $("classicToolDesc");
  const classicSteps = $("classicSteps");

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
  let activeTool = "image";
  let activePreviewUrl = null;

  // ---- 本地模式（WebAssembly） ----
  const localModeToggle = $("localModeToggle");
  const engineBadge = $("engineBadge");
  const engineHint = $("engineHint");
  const engineStatus = $("engineStatus");
  const engineStatusText = $("engineStatusText");
  const localProgress = $("localProgress");
  const localProgressText = $("localProgressText");
  const localProgressPct = $("localProgressPct");
  const localProgressFill = $("localProgressFill");
  const enableTargetRes = $("enableTargetRes");
  const targetResRow = $("targetResRow");
  const targetWidthInput = $("targetWidth");
  const targetHeightInput = $("targetHeight");
  const targetResLock = $("targetResLock");
  const targetResHint = $("targetResHint");

  let localMode = false;
  let processing = false;
  let aspectLock = true;
  let aspectRatio = null; // width / height of the first image
  let lastImageDims = null; // dimensions of the most recently attached image

  function isLocalMode() {
    return Boolean(localModeToggle && localModeToggle.checked);
  }

  function downloadBlob(blob, name) {
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("failed to read blob"));
      reader.readAsDataURL(blob);
    });
  }

  async function saveBlob(blob, name) {
    if (window.pywebview && window.pywebview.api && typeof window.pywebview.api.save_bytes === "function") {
      const payload = await blobToDataUrl(blob);
      const result = await window.pywebview.api.save_bytes(payload, name);
      if (!result || !result.ok) {
        if (result && result.cancelled) return false;
        throw new Error("desktop save failed");
      }
      return true;
    }

    downloadBlob(blob, name);
    return true;
  }

  function filenameFromDisposition(disposition, fallback) {
    const value = disposition || "";
    const utf8Match = value.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match) {
      try {
        return decodeURIComponent(utf8Match[1]);
      } catch (error) {
        return utf8Match[1];
      }
    }
    const asciiMatch = value.match(/filename="?([^";]+)"?/i);
    return asciiMatch ? asciiMatch[1] : fallback;
  }

  function setTool(tool) {
    activeTool = tool;
    const isClassic = tool === "classic";
    if (imageToolPage) imageToolPage.hidden = isClassic;
    if (classicToolPage) classicToolPage.hidden = !isClassic;
    if (imageToolTab) imageToolTab.classList.toggle("active", !isClassic);
    if (classicToolTab) classicToolTab.classList.toggle("active", isClassic);
    if (outlineLink) outlineLink.hidden = isClassic;
    if (topbarSubtitle) {
      topbarSubtitle.textContent = isClassic
        ? "GIA模式转换"
        : (currentMode === "fill" ? "默认填充模式 · 默认仅圆形" : "装饰物拟合模式 · 使用元件参数生成轮廓");
    }
  }

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
    if (!preset) {
      meta.textContent = "当前参数将写入装饰物导出结果。";
      return;
    }

    const parts = [];
    if (preset.hint) parts.push(preset.hint);
    if (preset.type_id !== undefined) parts.push(`ID: ${preset.type_id}`);
    if (preset.size !== undefined) parts.push(`大小: ${preset.size}`);
    meta.textContent = parts.length > 0
      ? `${parts.join(" · ")}。可以继续微调类型 ID、资源 ID 和旋转。`
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

  async function readImageDimensions(file) {
    if (!file) return null;

    if (window.createImageBitmap) {
      try {
        const bitmap = await window.createImageBitmap(file);
        const size = { width: bitmap.width, height: bitmap.height };
        bitmap.close();
        return size;
      } catch (error) {
        // Fall back to Image() decoding below.
      }
    }

    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => {
        const size = { width: image.naturalWidth, height: image.naturalHeight };
        URL.revokeObjectURL(url);
        resolve(size);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(null);
      };
      image.src = url;
    });
  }

  function renderImageDimensions(dimensions) {
    const dimensionText = dimensions
      ? `${dimensions.width} × ${dimensions.height} px`
      : "分辨率读取失败";

    if (imgSize) {
      imgSize.textContent = "";
    }

    if (readyTag) {
      readyTag.hidden = false;
      readyTag.textContent = dimensions
        ? `已选择图片 · ${dimensionText}`
        : "已选择图片 · 分辨率读取失败";
    }
  }

  function formatFileSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  async function attachFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const isLarge = file.size > 3 * 1024 * 1024;
    if (uploadWarning) uploadWarning.hidden = !isLarge;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;

    if (fileName) fileName.textContent = file.name;
    if (imgSize) imgSize.textContent = formatFileSize(file.size);
    if (preview) {
      if (activePreviewUrl) {
        URL.revokeObjectURL(activePreviewUrl);
        activePreviewUrl = null;
      }
      const url = URL.createObjectURL(file);
      activePreviewUrl = url;
      preview.src = url;
      preview.hidden = false;
    }

    if (readyTag) {
      readyTag.hidden = false;
      readyTag.textContent = "已选择图片 · 读取分辨率中...";
    }

    const dimensions = await readImageDimensions(file);
    renderImageDimensions(dimensions);
    if (dimensions && dimensions.width > 0 && dimensions.height > 0) {
      aspectRatio = dimensions.width / dimensions.height;
      lastImageDims = dimensions;
      fillTargetResFromImage(dimensions.width, dimensions.height);
      syncTargetResFromAspect();
    }

    if (dropZone) dropZone.classList.add("ready");
    if (submitButton) submitButton.classList.add("ready");
  }

  function getClassicDirection() {
    return dirClassicToOver && dirClassicToOver.checked ? "classic_to_overlimit" : "overlimit_to_classic";
  }

  /* ================= 输出尺寸（缩放 / 指定分辨率 二选一） ================= */

  const segScale = $("segScale");
  const segTarget = $("segTarget");
  const scalePanel = $("scalePanel");
  const targetPanel = $("targetPanel");
  const outputSizeVal = $("outputSizeVal");
  const imageScaleInput = $("imageScale");
  let outputSizeMode = "scale"; // "scale" | "target"

  function syncTargetResFromAspect() {
    if (outputSizeMode !== "target" || !aspectLock || !aspectRatio) return;
    const w = Number.parseInt(targetWidthInput.value, 10);
    const h = Number.parseInt(targetHeightInput.value, 10);
    if (!Number.isFinite(w) && !Number.isFinite(h)) {
      targetWidthInput.value = String(Math.round(aspectRatio * 512));
      targetHeightInput.value = "512";
    } else if (Number.isFinite(w) && document.activeElement === targetWidthInput) {
      targetHeightInput.value = String(Math.max(16, Math.round(w / aspectRatio)));
    } else if (Number.isFinite(h) && document.activeElement === targetHeightInput) {
      targetWidthInput.value = String(Math.max(16, Math.round(h * aspectRatio)));
    }
  }

  function updateOutputSizeUi() {
    const isTarget = outputSizeMode === "target";
    if (segScale) segScale.classList.toggle("active", !isTarget);
    if (segTarget) segTarget.classList.toggle("active", isTarget);
    if (scalePanel) scalePanel.hidden = isTarget;
    if (targetPanel) targetPanel.hidden = !isTarget;

    // disabled 的字段不会随表单提交：切到「指定分辨率」时缩放按服务端默认 1.0，
    // 切回「缩放」时目标分辨率字段整体不提交。
    if (imageScaleInput) imageScaleInput.disabled = isTarget;
    if (enableTargetRes) {
      enableTargetRes.checked = isTarget;
      enableTargetRes.disabled = !isTarget;
    }
    [targetWidthInput, targetHeightInput].forEach((input) => {
      if (input) input.disabled = !isTarget;
    });

    if (outputSizeVal) {
      if (!isTarget) {
        outputSizeVal.textContent = `缩放 ×${imageScaleInput ? imageScaleInput.value : "1.0"}`;
      } else {
        const w = Number.parseInt(targetWidthInput.value, 10);
        const h = Number.parseInt(targetHeightInput.value, 10);
        outputSizeVal.textContent = Number.isFinite(w) && Number.isFinite(h) ? `${w} × ${h}` : "未设置";
      }
    }
  }

  function setOutputSizeMode(mode) {
    outputSizeMode = mode === "target" ? "target" : "scale";
    if (outputSizeMode === "target") {
      // 已有图片时直接填充其分辨率，否则按比例给出默认值
      if (lastImageDims) fillTargetResFromImage(lastImageDims.width, lastImageDims.height);
      else if (aspectLock) syncTargetResFromAspect();
    }
    updateOutputSizeUi();
  }

  // 「指定分辨率」模式下，上传图片后直接填充当前图片的分辨率
  function fillTargetResFromImage(width, height) {
    if (outputSizeMode !== "target") return;
    if (!targetWidthInput || !targetHeightInput) return;
    if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return;
    targetWidthInput.value = String(width);
    targetHeightInput.value = String(height);
    updateOutputSizeUi();
  }

  function getTargetResolution() {
    if (outputSizeMode !== "target") return null;
    const w = Number.parseInt(targetWidthInput.value, 10);
    const h = Number.parseInt(targetHeightInput.value, 10);
    if (!Number.isFinite(w) || !Number.isFinite(h)) return null;
    return {
      target_width: Math.max(16, Math.min(4096, w)),
      target_height: Math.max(16, Math.min(4096, h)),
    };
  }

  if (segScale) segScale.addEventListener("click", () => setOutputSizeMode("scale"));
  if (segTarget) segTarget.addEventListener("click", () => setOutputSizeMode("target"));
  [targetWidthInput, targetHeightInput].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", () => {
      syncTargetResFromAspect();
      updateOutputSizeUi();
    });
  });
  if (imageScaleInput) {
    imageScaleInput.addEventListener("input", updateOutputSizeUi);
  }
  if (targetResLock) {
    targetResLock.addEventListener("click", () => {
      aspectLock = !aspectLock;
      targetResLock.classList.toggle("active", aspectLock);
      targetResLock.textContent = aspectLock ? "等比" : "自由";
      if (aspectLock) syncTargetResFromAspect();
      updateOutputSizeUi();
    });
  }

  /* ================= 本地模式 ================= */

  function updateEngineUi() {
    const on = isLocalMode();
    localMode = on;
    try { localStorage.setItem("shaper.localMode", on ? "1" : "0"); } catch (error) { /* ignore */ }

    if (engineBadge) {
      engineBadge.textContent = on ? "本地引擎 · WASM" : "云端引擎";
      engineBadge.dataset.mode = on ? "local" : "cloud";
    }
    if (engineHint) {
      engineHint.textContent = on
        ? "本地模式：速度更快，计算在浏览器内完成。"
        : "云端模式：由服务器完成拟合计算。";
    }
    if (engineStatus) engineStatus.hidden = !on;

    if (fileInput) {
      if (on) fileInput.removeAttribute("required");
      else fileInput.setAttribute("required", "");
    }

    // 本地模式仅支持填充模式：切回 fill 并隐藏装饰物切换
    if (outlineLink) outlineLink.hidden = on;
    if (on && currentMode === "outline") setMode("fill");

    if (!on && localProgress) localProgress.hidden = true;

    if (on && window.LocalFit) {
      if (engineStatusText) engineStatusText.textContent = "本地引擎加载中…（首次约几秒）";
      window.LocalFit.ensureReady().then((ready) => {
        if (!engineStatusText) return;
        if (!isLocalMode()) return;
        const workers = window.LocalFit.readyWorkerCount ? window.LocalFit.readyWorkerCount() : 1;
        engineStatusText.textContent = ready
          ? (workers > 1 ? `本地引擎就绪（${workers} 核并行）` : "本地引擎就绪")
          : "本地引擎加载失败，可切换回云端模式";
        if (engineStatus) engineStatus.dataset.state = ready ? "ready" : "error";
      });
      if (engineStatus) engineStatus.dataset.state = "loading";
    }
  }

  /* ================= 本地处理参数 ================= */

  function readAllowedShapes() {
    const shapes = [];
    if ($("shapeCircle") && $("shapeCircle").checked) shapes.push("circle");
    if ($("shapeRect") && $("shapeRect").checked) shapes.push("rect");
    if ($("shapeTriangle") && $("shapeTriangle").checked) shapes.push("triangle");
    return shapes.length > 0 ? shapes : ["circle"];
  }

  function buildUnifiedConfig() {
    const manualPrims = Number.parseInt(($("numPrimsManual") || {}).value, 10);
    const sliderPrims = Number.parseInt(($("numPrims") || {}).value, 10);
    const numPrimitives = Number.isFinite(manualPrims)
      ? Math.max(40, Math.min(3000, manualPrims))
      : Math.max(40, Math.min(3000, Number.isFinite(sliderPrims) ? sliderPrims : 400));
    const config = {
      mode: "fill",
      num_primitives: numPrimitives,
      mask_threshold: 127,
      detail_scale: 1.0,
      image_scale: Number.parseFloat(($("imageScale") || {}).value) || 1.0,
      output_alpha: (Number.parseFloat(($("outputAlpha") || {}).value) || 100) / 100,
      enable_png_mode: Boolean($("enablePngMode") && $("enablePngMode").checked),
      allowed_shapes: readAllowedShapes(),
      primitives: JSON.parse((hiddenPrimitives && hiddenPrimitives.value) || "[]"),
      origin: { type: "center" },
    };
    const target = getTargetResolution();
    if (target) {
      // 「指定分辨率」与「图片缩放」互斥：分辨率模式按缩放 1.0 处理
      config.image_scale = 1.0;
      Object.assign(config, target);
    }
    return config;
  }

  function setLocalProgress(text, pct) {
    if (!localProgress) return;
    localProgress.hidden = false;
    if (localProgressText) localProgressText.textContent = text;
    const clamped = Math.max(0, Math.min(100, Math.round(pct || 0)));
    if (localProgressPct) localProgressPct.textContent = clamped + "%";
    if (localProgressFill) localProgressFill.style.width = clamped + "%";
  }

  /* 本地模式 · 单图：WASM 拟合 → 寄存 → 跳结果页 */
  async function processSingleLocal() {
    if (processing) return;
    if (!window.LocalFit) {
      alert("本地引擎未加载");
      return;
    }
    const file = fileInput && fileInput.files && fileInput.files[0];
    if (!file) {
      alert("请先选择图片");
      return;
    }
    processing = true;
    if (submitButton) submitButton.disabled = true;
    updatePrimitivesJson();

    const config = buildUnifiedConfig();
    const ext = (file.name.match(/\.[a-z0-9]+$/i) || [""])[0].toLowerCase();
    config.source_filename = file.name;
    config.source_ext = ext;

    try {
      setLocalProgress("本地引擎准备中…", 0);
      const { result, sourceImageBase64 } = await window.LocalFit.fitOne(file, config, (done, total) => {
        setLocalProgress(`正在拟合（${done}/${total} 图元）`, (done / total) * 100);
      });
      setLocalProgress("正在寄存结果…", 100);
      const taskId = await window.LocalFit.registerResult(
        result, config, file.name.replace(/\.[a-z0-9]+$/i, ""), sourceImageBase64
      );
      window.location.href = `/result/${taskId}`;
    } catch (error) {
      alert(`本地拟合失败：${(error && error.message) || error}`);
      setLocalProgress("拟合失败", 0);
    } finally {
      processing = false;
      if (submitButton) submitButton.disabled = false;
    }
  }

  if (localModeToggle) {
    localModeToggle.addEventListener("change", updateEngineUi);
  }

  function attachClassicGia(file) {
    if (!file) return;
    const name = file.name || "";
    if (!name.toLowerCase().endsWith(".gia")) {
      alert("请上传 .gia 文件");
      return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    classicGiaInput.files = transfer.files;

    if (classicGiaName) classicGiaName.textContent = name;
    if (classicGiaReady) {
      classicGiaReady.hidden = false;
      const direction = getClassicDirection();
      const modeLabel = direction === "classic_to_overlimit" ? "经典模式" : "超限模式";
      classicGiaReady.textContent = `已选择 ${modeLabel} GIA · ${(file.size / 1024).toFixed(1)} KB`;
    }
    if (classicDropZone) classicDropZone.classList.add("ready");
    if (classicGiaButton) classicGiaButton.classList.add("ready");
  }

  setSliderValue("numPrims");
  setSliderValue("imageScale");
  setSliderValue("outputAlpha", (value) => value + "%");
  ["olPrimSize", "olSpacing", "olPrecision"].forEach((id) => setSliderValue(id));

  function updateSliderFill(input) {
    const min = Number(input.min) || 0;
    const max = Number(input.max) || 100;
    const value = Number(input.value) || 0;
    const percent = max > min ? ((value - min) / (max - min)) * 100 : 0;
    input.style.setProperty("--slider-fill", percent + "%");
  }
  document.querySelectorAll('input[type="range"]').forEach((input) => {
    updateSliderFill(input);
    input.addEventListener("input", () => updateSliderFill(input));
  });

  const numPrimsSlider = $("numPrims");
  const numPrimsManual = $("numPrimsManual");
  const numPrimsVal = $("numPrimsVal");
  if (numPrimsSlider && numPrimsManual) {
    numPrimsSlider.max = "1200";
    numPrimsSlider.addEventListener("input", () => {
      numPrimsManual.value = numPrimsSlider.value;
      if (numPrimsVal) numPrimsVal.textContent = numPrimsSlider.value;
    });
    numPrimsManual.addEventListener("input", () => {
      const value = Number.parseInt(numPrimsManual.value, 10);
      if (!Number.isNaN(value) && value >= 40 && value <= 1200) {
        numPrimsSlider.value = String(value);
      }
      if (numPrimsVal) numPrimsVal.textContent = numPrimsSlider.value;
    });
  }

  if (outlineLink) {
    outlineLink.addEventListener("click", (event) => {
      event.preventDefault();
      setMode(currentMode === "fill" ? "outline" : "fill");
    });
  }

  if (imageToolTab) {
    imageToolTab.addEventListener("click", () => setTool("image"));
  }

  if (classicToolTab) {
    classicToolTab.addEventListener("click", () => setTool("classic"));
  }

  function updateClassicToolUi() {
    const isOverToClassic = !dirClassicToOver || !dirClassicToOver.checked;
    if (giaDirectionInput) giaDirectionInput.value = isOverToClassic ? "overlimit_to_classic" : "classic_to_overlimit";
    if (classicUploadTitle) classicUploadTitle.textContent = isOverToClassic ? "超限模式 GIA" : "经典模式 GIA";
    if (classicDropText) {
      classicDropText.innerHTML = isOverToClassic
        ? "点击或拖拽上传超限模式 <strong>.gia</strong>"
        : "点击或拖拽上传经典模式 <strong>.gia</strong>";
    }
    if (classicHint) {
      classicHint.textContent = isOverToClassic
        ? "转换会为 GIA 写入经典模式标记，原始文件不会被修改。"
        : "转换会移除 GIA 中的经典模式标记，原始文件不会被修改。";
    }
    if (classicToolTitle) classicToolTitle.textContent = isOverToClassic ? "超限模式转经典模式" : "经典模式转超限模式";
    if (classicToolDesc) {
      classicToolDesc.innerHTML = isOverToClassic
        ? "上传现有超限模式 GIA，转换后会下载一个带 <code>_classic</code> 后缀的经典模式 GIA。"
        : "上传现有经典模式 GIA，转换后会下载一个带 <code>_overlimit</code> 后缀的超限模式 GIA。";
    }
    if (classicSteps) {
      classicSteps.innerHTML = isOverToClassic
        ? "<li>上传超限模式 .gia</li><li>写入经典模式标记</li><li>下载新的经典模式 .gia</li>"
        : "<li>上传经典模式 .gia</li><li>移除经典模式标记</li><li>下载新的超限模式 .gia</li>";
    }
    if (classicGiaButton) {
      classicGiaButton.textContent = isOverToClassic ? "导出经典模式 GIA" : "导出超限模式 GIA";
    }
  }

  if (dirOverToClassic) {
    dirOverToClassic.addEventListener("change", updateClassicToolUi);
  }
  if (dirClassicToOver) {
    dirClassicToOver.addEventListener("change", updateClassicToolUi);
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
    dropZone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
      }
    });
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
      if (activeTool !== "image") return;
      const items = (event.clipboardData || window.clipboardData || {}).items || [];
      const imageFiles = [];
      for (let i = 0; i < items.length; i += 1) {
        if (items[i].type && items[i].type.indexOf("image") !== -1) {
          const file = items[i].getAsFile();
          if (file) imageFiles.push(file);
        }
      }
      if (imageFiles.length === 0) return;
      attachFile(imageFiles[0]);
    });
  }

  if (form) {
    form.addEventListener("submit", (event) => {
      if (isLocalMode()) {
        event.preventDefault();
        processSingleLocal();
        return;
      }
      if (hiddenMode && !hiddenMode.value) hiddenMode.value = "fill";
      updatePrimitivesJson();
    });
  }

  if (classicDropZone && classicGiaInput) {
    classicDropZone.addEventListener("click", () => classicGiaInput.click());
    classicGiaInput.addEventListener("change", () => {
      if (classicGiaInput.files && classicGiaInput.files[0]) attachClassicGia(classicGiaInput.files[0]);
    });

    classicDropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      classicDropZone.classList.add("drag-over");
    });

    classicDropZone.addEventListener("dragleave", (event) => {
      event.preventDefault();
      classicDropZone.classList.remove("drag-over");
    });

    classicDropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      classicDropZone.classList.remove("drag-over");
      if (event.dataTransfer.files && event.dataTransfer.files[0]) {
        attachClassicGia(event.dataTransfer.files[0]);
      }
    });
  }

  if (classicGiaForm) {
    classicGiaForm.addEventListener("submit", (event) => {
      event.preventDefault();
      const direction = getClassicDirection();
      const sourceLabel = direction === "classic_to_overlimit" ? "经典模式" : "超限模式";
      if (!classicGiaInput || !classicGiaInput.files || !classicGiaInput.files[0]) {
        alert(`请先选择${sourceLabel} GIA 文件`);
        return;
      }

      const originalText = classicGiaButton ? classicGiaButton.textContent : "";
      if (classicGiaButton) {
        classicGiaButton.disabled = true;
        classicGiaButton.textContent = "转换中...";
      }

      const formData = new FormData(classicGiaForm);
      fetch("/convert_gia_mode", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
          if (!response.ok) {
            return response.text().then((text) => {
              throw new Error(text || `HTTP ${response.status}`);
            });
          }
          const defaultSuffix = direction === "classic_to_overlimit" ? "_overlimit.gia" : "_classic.gia";
          const filename = filenameFromDisposition(
            response.headers.get("Content-Disposition"),
            (classicGiaInput.files[0].name || "gia_mode.gia").replace(/\.gia$/i, defaultSuffix),
          );
          return response.blob().then((blob) => ({ blob, filename }));
        })
        .then(({ blob, filename }) => saveBlob(blob, filename))
        .catch((error) => alert(`转换失败: ${error && error.message ? error.message : error}`))
        .finally(() => {
          if (classicGiaButton) {
            classicGiaButton.disabled = false;
            classicGiaButton.textContent = originalText || (direction === "classic_to_overlimit" ? "导出超限模式 GIA" : "导出经典模式 GIA");
          }
        });
    });
  }

  syncShapeLabels();
  setMode("fill");
  setTool("image");
  updateClassicToolUi();

  // 恢复本地模式偏好并预热引擎
  let savedLocalMode = false;
  try {
    savedLocalMode = localStorage.getItem("shaper.localMode") === "1";
  } catch (error) { /* ignore */ }
  if (localModeToggle) {
    localModeToggle.checked = savedLocalMode;
    updateEngineUi();
  }
  updateOutputSizeUi();
})();
