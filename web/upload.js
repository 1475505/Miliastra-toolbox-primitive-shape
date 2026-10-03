(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

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
  const topbarSubtitle = document.querySelector(".topbar-subtitle");

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
    if (imageToolTab) imageToolTab.setAttribute("aria-pressed", String(!isClassic));
    if (classicToolTab) classicToolTab.setAttribute("aria-pressed", String(isClassic));
    if (topbarSubtitle) {
      topbarSubtitle.textContent = isClassic ? "GIA转换" : "默认填充模式 · 默认仅圆形";
    }
  }

  function updatePrimitivesJson() {
    // primitives_json 只承载填充模式的图元类型清单（历史「装饰物元件列表」已移除）
    if (!hiddenPrimitives) return;
    const primitives = [];
    if (($("shapeCircle") || {}).checked) primitives.push({ shape: "circle", color: "#ffffff" });
    if (($("shapeRect") || {}).checked) primitives.push({ shape: "rect", color: "#ffffff" });
    if (($("shapeTriangle") || {}).checked) primitives.push({ shape: "triangle", color: "#ffffff" });
    hiddenPrimitives.value = JSON.stringify(primitives);
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

  function updateUploadWarning() {
    const file = fileInput && fileInput.files && fileInput.files[0];
    const isLarge = file && file.size > 3 * 1024 * 1024;
    if (uploadWarning) uploadWarning.hidden = isLocalMode() || !isLarge;
  }

  async function attachFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateUploadWarning();

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
    if ($("dirGiaToLua") && $("dirGiaToLua").checked) return "gia_to_lua";
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
    updateUploadWarning();
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
    const alphaPercent = Number.parseFloat(($("outputAlpha") || {}).value);
    const config = {
      mode: "fill",
      num_primitives: numPrimitives,
      mask_threshold: 127,
      detail_scale: 1.0,
      image_scale: Number.parseFloat(($("imageScale") || {}).value) || 1.0,
      // 透明度滑块最小值为 0，是合法取值，不能用 `|| 100` 兜底——JS 会把 0 当假值
      // 静默改成 100，表现为「把透明度调到 0 却完全没效果」。
      output_alpha:
        (Number.isFinite(alphaPercent) ? Math.max(0, Math.min(100, alphaPercent)) : 100) / 100,
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

  /* 寄存阶段的进度文案：真实百分比 + 按当前速率外推的剩余时间（弱网下这段是主要耗时） */
  function registerProgress(loaded, total, startedAt) {
    const ratio = total > 0 ? loaded / total : 0;
    const elapsed = performance.now() - startedAt;
    let eta = "";
    if (loaded > 0 && loaded < total && elapsed > 300) {
      const remain = Math.max(1, Math.round((total - loaded) / (loaded / elapsed) / 1000));
      eta = remain >= 60
        ? `（约 ${Math.floor(remain / 60)} 分 ${remain % 60} 秒）`
        : `（约 ${remain} 秒）`;
    }
    setLocalProgress(`正在寄存结果… ${Math.round(ratio * 100)}%${eta}`, ratio * 100);
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
      const { result, sourceBlob, maskBase64 } = await window.LocalFit.fitOne(file, config, (done, total) => {
        setLocalProgress(`正在拟合（${done}/${total} 图元）`, (done / total) * 100);
      });
      // 寄存只传 elements（弱网下自动 gzip），源图与蒙版都走本地缓存
      const registerStartedAt = performance.now();
      const taskId = await window.LocalFit.registerResult(
        result,
        config,
        file.name.replace(/\.[a-z0-9]+$/i, ""),
        (loaded, total) => registerProgress(loaded, total, registerStartedAt)
      );

      // 源图存 IndexedDB：结果页立即可用作底图，并在后台补传服务端
      let cached = false;
      try {
        await window.LocalFit.idbSaveSourceImage(taskId, sourceBlob);
        cached = true;
      } catch (error) { /* 隐私模式/配额受限，走同步回退 */ }

      if (!cached) {
        // 回退：同步补传源图（带进度）。该阶段可容忍失败，超时/出错直接跳过，
        // 不阻塞进入结果页（底图缺失时结果页白底降级，导出不受影响）
        setLocalProgress("正在上传原图…", 0);
        try {
          await window.LocalFit.uploadSourceImage(taskId, sourceBlob, (done, total) => {
            setLocalProgress(`正在上传原图（${Math.round((done / total) * 100)}%）`, (done / total) * 100);
          });
        } catch (error) { /* 跳过源图补传 */ }
      }

      // 蒙版存 IndexedDB：寄存请求不再携带，结果页本地读取作叠加预览
      if (maskBase64) {
        try {
          await window.LocalFit.idbSaveMask(taskId, maskBase64);
        } catch (error) { /* 蒙版叠加预览降级为不可用，不影响查看与导出 */ }
      }
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

  if (imageToolTab) {
    imageToolTab.addEventListener("click", () => setTool("image"));
  }

  if (classicToolTab) {
    classicToolTab.addEventListener("click", () => setTool("classic"));
  }

  function updateClassicToolUi() {
    const isLua = getClassicDirection() === "gia_to_lua";
    if ($("btnCopyGiaLua")) $("btnCopyGiaLua").hidden = !isLua;
    if ($("giaLuaCopyStatus")) $("giaLuaCopyStatus").textContent = "";
    if ($("giaLuaMaskOption")) $("giaLuaMaskOption").hidden = !isLua;
    if ($("classicModeTips")) $("classicModeTips").hidden = isLua;
    if (isLua) {
      if (giaDirectionInput) giaDirectionInput.value = "gia_to_lua";
      if (classicUploadTitle) classicUploadTitle.textContent = "超限模式图片素材组 GIA";
      if (classicDropText) classicDropText.innerHTML = "点击或拖拽上传素材组 <strong>.gia</strong>";
      if (classicHint) classicHint.textContent = "支持单层静态图片素材组，保留键鼠布局、颜色、透明度与层序。文本、嵌套组、动态图片需先转为单层图片；组遮罩可处理裁剪后导出，或勾选下方选项忽略。";
      if (classicToolTitle) classicToolTitle.textContent = "素材组 GIA 转 Lua 绘制脚本";
      if (classicToolDesc) classicToolDesc.textContent = "上传超限素材组，下载或复制 Lua。只需一个图片控件模板，脚本自动切换静态图片。";
      if (classicSteps) classicSteps.innerHTML = "<li>准备一个客户端图片控件，设为「仅存为模板」。</li><li><strong>填写索引：</strong>将脚本 <code>IMAGE_PREFAB_ID = 0</code> 中的 <code>0</code> 改成该控件的<strong>控件模板索引ID</strong>（不是图片资产ID）。只改这一处。</li><li><strong>挂载脚本：</strong>创建专用空客户端容器节点，将下载或复制的 Lua 作为客户端脚本挂到该节点，进入运行预览即可绘制。</li><li>可选：<code>BASE_SCALE</code> 缩放，<code>OFFSET_X/Y</code> 平移。自定义图片需在当前关卡可用。</li>";
      if (classicGiaButton) classicGiaButton.textContent = "导出 Lua 绘制脚本";
      return;
    }
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
  if ($("dirGiaToLua")) $("dirGiaToLua").addEventListener("change", updateClassicToolUi);
  if (dirClassicToOver) {
    dirClassicToOver.addEventListener("change", updateClassicToolUi);
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
      const copyLua = direction === "gia_to_lua" && event.submitter && event.submitter.id === "btnCopyGiaLua";
      if ($("giaLuaCopyStatus")) $("giaLuaCopyStatus").textContent = "";
      const sourceLabel = direction === "classic_to_overlimit" ? "经典模式" : "超限模式";
      if (!classicGiaInput || !classicGiaInput.files || !classicGiaInput.files[0]) {
        alert(`请先选择${sourceLabel} GIA 文件`);
        return;
      }

      const originalText = classicGiaButton ? classicGiaButton.textContent : "";
      if ($("btnCopyGiaLua")) $("btnCopyGiaLua").disabled = true;
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
          const defaultSuffix = direction === "gia_to_lua" ? ".lua" : direction === "classic_to_overlimit" ? "_overlimit.gia" : "_classic.gia";
          const filename = filenameFromDisposition(
            response.headers.get("Content-Disposition"),
            (classicGiaInput.files[0].name || "gia_mode.gia").replace(/\.gia$/i, defaultSuffix),
          );
          return response.blob().then((blob) => ({ blob, filename }));
        })
        .then(async ({ blob, filename }) => {
          if (copyLua) {
            await window.ShaperClipboard.copy(await blob.text());
            if ($("giaLuaCopyStatus")) $("giaLuaCopyStatus").textContent = "Lua 已复制，可粘贴到客户端脚本。";
          } else {
            await saveBlob(blob, filename);
          }
        })
        .catch((error) => alert(`转换失败: ${error && error.message ? error.message : error}`))
        .finally(() => {
          if ($("btnCopyGiaLua")) $("btnCopyGiaLua").disabled = false;
          if (classicGiaButton) {
            classicGiaButton.disabled = false;
            classicGiaButton.textContent = originalText || (direction === "classic_to_overlimit" ? "导出超限模式 GIA" : "导出经典模式 GIA");
          }
        });
    });
  }

  syncShapeLabels();
  updatePrimitivesJson();
  setTool("image");
  updateClassicToolUi();

  // 恢复本地模式偏好并预热引擎：优先用户上次选择，无记录时用服务端默认（DEFAULT_FIT_MODE）
  let savedLocalMode = false;
  try {
    const saved = localStorage.getItem("shaper.localMode");
    if (saved === "1" || saved === "0") {
      savedLocalMode = saved === "1";
    } else {
      const serverDefault = (document.body.dataset.defaultFitMode || "").toLowerCase();
      savedLocalMode = serverDefault !== "cloud"; // 缺省本地模式
    }
  } catch (error) { /* ignore */ }
  if (localModeToggle) {
    localModeToggle.checked = savedLocalMode;
    updateEngineUi();
  }
  updateOutputSizeUi();
})();
