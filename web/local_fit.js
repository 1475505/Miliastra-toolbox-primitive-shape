/* Local (WebAssembly) fitting client.
 *
 * Mirrors the backend pipeline in the browser:
 *   primitive_backend._extract_image_and_mask / _prepare_transparent_target
 *   shaper_core.process_image_fill
 *   fill_shaper.results_to_elements
 * so the assembled result object can be registered via /register_result and
 * rendered by the existing result page unchanged.
 */
(function () {
  "use strict";

  const PNG_ALPHA_FIT_FLOOR = 0.2;
  const PNG_ALPHA_FIT_GAMMA = 1.6;
  const BACKGROUND_BLEED_PX = 4.0;
  const DEFAULT_IMAGE_ASSET_REFS = { circle: 100002, rect: 100001, triangle: 100003 };

  // Worker 池：每个 Worker 加载独立的 WASM 实例。Go js/wasm 运行时为单线程
  // （GOMAXPROCS=1），多核只能多实例并行。单图拟合采用与云端 `primitive -j N`
  // 相同的 Step 内候选并行：每步各实例独立搜索候选，主实例应用最优。
  const POOL_SIZE = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 4));
  const pool = []; // { worker, ready, bootError, pending: Map, jobSeq }
  const bootWaiters = [];
  let bootSettled = false;
  let engineBusy = false;

  function bootPool() {
    if (pool.length > 0) return;
    for (let i = 0; i < POOL_SIZE; i += 1) {
      const slot = {
        worker: new Worker("/web/wasm/fit_worker.js"),
        ready: false,
        bootError: null,
        pending: new Map(),
        jobSeq: 0,
      };
      slot.worker.onmessage = (event) => handleSlotMessage(slot, event);
      slot.worker.onerror = (event) => {
        slot.bootError = event.message || "worker 错误";
        checkBootSettled();
      };
      pool.push(slot);
    }
  }

  function handleSlotMessage(slot, event) {
    const data = event.data || {};
    if (data.type === "ready") {
      slot.ready = true;
      checkBootSettled();
      return;
    }
    if (data.type === "boot_error") {
      slot.bootError = data.message || "wasm 加载失败";
      checkBootSettled();
      return;
    }
    const job = slot.pending.get(data.jobId);
    if (!job) return;
    if (data.type === "progress") {
      job.onProgress(data.done, data.total);
      return;
    }
    slot.pending.delete(data.jobId);
    job.resolve(data);
  }

  function checkBootSettled() {
    if (bootSettled) return;
    const settled = pool.every((slot) => slot.ready || slot.bootError);
    if (!settled) return;
    bootSettled = true;
    const anyReady = pool.some((slot) => slot.ready);
    bootWaiters.splice(0).forEach((resolve) => resolve(anyReady));
  }

  function ensureReady() {
    bootPool();
    if (bootSettled) return Promise.resolve(pool.some((slot) => slot.ready));
    return new Promise((resolve) => bootWaiters.push(resolve));
  }

  function engineStatus() {
    const readyCount = pool.filter((slot) => slot.ready).length;
    if (readyCount > 0) return "ready";
    if (bootSettled) return "error:" + ((pool.find((slot) => slot.bootError) || {}).bootError || "加载失败");
    return "loading";
  }

  function readyWorkerCount() {
    return pool.filter((slot) => slot.ready).length;
  }

  function poolSize() {
    return POOL_SIZE;
  }

  /* RPC：向 slot 发消息并等待对应回包 */
  function rpc(slot, type, payload, transfer) {
    return new Promise((resolve, reject) => {
      const jobId = ++slot.jobSeq;
      const timeout = setTimeout(() => {
        slot.pending.delete(jobId);
        reject(new Error("本地引擎响应超时"));
      }, 30 * 60 * 1000);
      slot.pending.set(jobId, {
        resolve: (data) => { clearTimeout(timeout); resolve(data); },
        onProgress: (payload && payload.onProgress) || (() => {}),
      });
      const message = { type, jobId, ...payload };
      delete message.onProgress;
      slot.worker.postMessage(message, transfer || []);
    });
  }

  function loadImageFromFile(file) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const image = new Image();
      image.onload = () => {
        URL.revokeObjectURL(url);
        resolve(image);
      };
      image.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("图片解码失败"));
      };
      image.src = url;
    });
  }

  function drawToCanvas(source, width, height, background) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (background) {
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, width, height);
    }
    ctx.drawImage(source, 0, 0, width, height);
    return canvas;
  }

  function canvasToPngBase64(canvas) {
    const url = canvas.toDataURL("image/png");
    return url.slice(url.indexOf(",") + 1);
  }

  function base64ToImage(base64) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("预览图解码失败"));
      image.src = "data:image/png;base64," + base64;
    });
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const url = String(reader.result || "");
        resolve(url.slice(url.indexOf(",") + 1));
      };
      reader.onerror = () => reject(reader.error || new Error("读取文件失败"));
      reader.readAsDataURL(file);
    });
  }

  function packColor(hex, alpha) {
    const rgb = hexToRgb(hex);
    const a = Math.max(0, Math.min(255, Math.round(alpha * 255)));
    return (((a << 24) | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2]) >>> 0);
  }

  function hexToRgb(hexColor) {
    let value = String(hexColor || "#ffffff").trim().replace(/^#/, "");
    if (value.length === 3) value = value.split("").map((ch) => ch + ch).join("");
    if (value.length !== 6) return [255, 255, 255];
    return [0, 2, 4].map((i) => parseInt(value.slice(i, i + 2), 16));
  }

  function bgrToHex(bgr) {
    const r = Math.max(0, Math.min(255, Math.round(bgr[2])));
    const g = Math.max(0, Math.min(255, Math.round(bgr[1])));
    const b = Math.max(0, Math.min(255, Math.round(bgr[0])));
    return "#" + [r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("");
  }

  function resolveOrigin(config, width, height) {
    const origin = (config && config.origin) || {};
    if (origin.type === "custom") {
      const x = parseFloat(origin.x);
      const y = parseFloat(origin.y);
      return [Number.isFinite(x) ? x : width / 2, Number.isFinite(y) ? y : height / 2];
    }
    if (origin.type === "top_left") return [0, 0];
    return [width / 2, height / 2];
  }

  function sourceIsPng(config, fileName) {
    const ext = String((config && config.source_ext) || "").toLowerCase();
    if (ext) return ext === ".png";
    return String(fileName || "").toLowerCase().endsWith(".png");
  }

  /* ------- 图像预处理：与 primitive_backend 对齐 ------- */

  function prepareTarget(imageData, transparent) {
    const { width, height, data } = imageData;
    const rgba = new Uint8Array(width * height * 4);
    const alpha = new Uint8Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const a = data[i * 4 + 3];
      alpha[i] = a;
      const af = a / 255;
      rgba[i * 4] = Math.round(data[i * 4] * af + 255 * (1 - af));
      rgba[i * 4 + 1] = Math.round(data[i * 4 + 1] * af + 255 * (1 - af));
      rgba[i * 4 + 2] = Math.round(data[i * 4 + 2] * af + 255 * (1 - af));
      if (transparent) {
        // _compress_alpha_for_fitting: floor + gamma
        const compressed = Math.pow(Math.max(0, (af - PNG_ALPHA_FIT_FLOOR) / (1 - PNG_ALPHA_FIT_FLOOR)), PNG_ALPHA_FIT_GAMMA);
        rgba[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, compressed)) * 255);
      } else {
        rgba[i * 4 + 3] = 255;
      }
    }
    return { rgba, alpha };
  }

  function hasTransparentAlpha(imageData) {
    const data = imageData.data;
    for (let i = 3; i < data.length; i += 4) {
      if (data[i] < 255) return true;
    }
    return false;
  }

  function computeWorkSize(fullW, fullH, detailScale) {
    const fullMax = Math.max(fullW, fullH);
    const canvasLimit = Math.max(16, Math.min(Math.round(fullMax * detailScale), fullMax, 2048));
    const ratio = Math.min(1, canvasLimit / Math.max(fullMax, 1));
    return {
      width: Math.max(1, Math.round(fullW * ratio)),
      height: Math.max(1, Math.round(fullH * ratio)),
    };
  }

  /* ------- results_to_elements 的 JS 复刻 ------- */

  function resultsToElements(results, unitScale, imageCenter, primitivesConfig, outputAlpha) {
    const presetMap = {};
    (primitivesConfig || []).forEach((preset) => {
      if (preset && preset.shape) presetMap[preset.shape] = preset;
    });

    unitScale = Number(unitScale) || 1.0;
    const originX = imageCenter[0] * unitScale;
    const originY = -imageCenter[1] * unitScale;
    const elements = [];

    results.forEach((result, index) => {
      const cx = result.cx * unitScale;
      const cy = -result.cy * unitScale;
      const shapeKey = result.type;
      let preset, elementType, size;
      if (shapeKey === "circle") {
        preset = presetMap.circle || {};
        elementType = "ellipse";
        size = { rx: round4(result.rx * unitScale), ry: round4(result.ry * unitScale) };
      } else if (shapeKey === "triangle") {
        preset = presetMap.triangle || {};
        elementType = "triangle";
        const triWidth = (result.width || result.size || 1) * unitScale;
        const triHeight = (result.height || (result.size || 1) * Math.sqrt(3) / 2) * unitScale;
        size = { width: round4(triWidth), height: round4(triHeight) };
      } else {
        preset = presetMap.rect || {};
        elementType = "rectangle";
        size = { width: round4(2 * result.hw * unitScale), height: round4(2 * result.hh * unitScale) };
      }

      const colorHex = typeof result.color === "string" ? result.color : "#ffffff";
      let alpha = Number(result.alpha != null ? result.alpha : 1);
      if (outputAlpha != null) {
        alpha = Math.max(0, Math.min(1, alpha * Number(outputAlpha)));
      }
      const imageAssetRef = Number(
        preset.image_asset_ref || preset.asset_id || result.image_asset_ref ||
        DEFAULT_IMAGE_ASSET_REFS[shapeKey] || 100002
      );

      const element = {
        id: index,
        type: elementType,
        center: { x: round4(cx), y: round4(cy) },
        relative_position: { x: round4(cx - originX), y: round4(cy - originY) },
        relative: { x: round4(cx - originX), y: round4(cy - originY) },
        size: size,
        rotation: { x: 0, y: 0, z: round4(-Number(result.angle || 0)) },
        color: colorHex,
        alpha: round4(alpha),
        packed_color: packColor(colorHex, alpha),
        image_asset_ref: imageAssetRef,
      };

      if (preset.type_id != null) {
        element.type_id = preset.type_id;
        element.element_type_id = preset.type_id;
      }
      if (preset.element_type_id != null) {
        element.element_type_id = preset.element_type_id;
      }
      if (preset.rot_z != null) {
        element.rotation.z = round4(element.rotation.z + Number(preset.rot_z));
      }
      if (preset.rot_y_add != null) {
        element.rotation.y = Number(preset.rot_y_add);
      }
      if (preset.name) {
        element.name = String(preset.name);
      }
      elements.push(element);
    });

    return elements;
  }

  function round4(value) {
    return Math.round(Number(value) * 10000) / 10000;
  }

  /* ------- 目标分辨率重定标：与 shaper_core._rescale_fill_output 对齐 ------- */

  function rescaleResult(result, targetW, targetH) {
    const width = result.image_size.width;
    const height = result.image_size.height;
    const rx = targetW / width;
    const ry = targetH / height;
    const unitScale = Number(result.config.unit_scale) || 1.0;
    const originX = result.image_center.x * rx;
    const originY = result.image_center.y * ry;
    const originUnitsX = originX * unitScale;
    const originUnitsY = -originY * unitScale;

    result.elements.forEach((element) => {
      const newCx = round4(element.center.x * rx);
      const newCy = round4(element.center.y * ry);
      element.center = { x: newCx, y: newCy };
      const relative = { x: round4(newCx - originUnitsX), y: round4(newCy - originUnitsY) };
      element.relative = { ...relative };
      if (element.relative_position) element.relative_position = { ...relative };
      if (element.size) {
        const scaled = {};
        Object.keys(element.size).forEach((key) => {
          const factor = key === "width" || key === "rx" ? rx : ry;
          scaled[key] = round4(element.size[key] * factor);
        });
        element.size = scaled;
      }
    });

    if (result.mask) {
      result.mask.center = { x: round4(result.mask.center.x * rx), y: round4(result.mask.center.y * ry) };
      result.mask.size = { width: round4(result.mask.size.width * rx), height: round4(result.mask.size.height * ry) };
      if (result.mask.bbox_px) {
        result.mask.bbox_px = {
          x: Math.round(result.mask.bbox_px.x * rx),
          y: Math.round(result.mask.bbox_px.y * ry),
          width: Math.max(1, Math.round(result.mask.bbox_px.width * rx)),
          height: Math.max(1, Math.round(result.mask.bbox_px.height * ry)),
        };
      }
    }

    result.image_center = { x: originX, y: originY };
    result.image_size = { width: targetW, height: targetH };
    result.config.target_width = targetW;
    result.config.target_height = targetH;
  }

  /* ------- 单张图片的完整本地拟合流程 ------- */

  async function fitOne(file, rawConfig, onProgress) {
    if (engineBusy) throw new Error("本地引擎正忙");
    const ready = await ensureReady();
    if (!ready) {
      const reason = (pool.find((slot) => slot.bootError) || {}).bootError || "未知错误";
      throw new Error("本地引擎加载失败：" + reason);
    }
    engineBusy = true;
    try {
      return await fitOneInner(file, rawConfig, onProgress);
    } finally {
      engineBusy = false;
    }
  }

  async function fitOneInner(file, rawConfig, onProgress) {
    const config = { ...(rawConfig || {}) };
    const startedAt = performance.now();
    const image = await loadImageFromFile(file);
    const fullW = image.naturalWidth;
    const fullH = image.naturalHeight;

    // 全分辨率 ImageData（判定透明通道）
    const fullCanvas = drawToCanvas(image, fullW, fullH, null);
    const fullImageData = fullCanvas.getContext("2d").getImageData(0, 0, fullW, fullH);
    const hasAlpha = hasTransparentAlpha(fullImageData);
    const pngSource = sourceIsPng(config, file.name);
    const enablePngMode = Boolean(config.enable_png_mode);
    const transparentOutput = pngSource && hasAlpha && enablePngMode;
    const needsWhiteBackground = pngSource && hasAlpha && !enablePngMode;

    // 工作分辨率
    const detailScale = Math.max(0.25, Number(config.detail_scale) || 1.0);
    const work = computeWorkSize(fullW, fullH, detailScale);
    const workCanvas = drawToCanvas(image, work.width, work.height, null);
    const workImageData = workCanvas.getContext("2d").getImageData(0, 0, work.width, work.height);
    const { rgba, alpha } = prepareTarget(workImageData, transparentOutput);

    if (onProgress) onProgress(0, Number(config.num_primitives) || 400);

    const wasmConfig = {
      full_w: fullW,
      full_h: fullH,
      work_w: work.width,
      work_h: work.height,
      num_primitives: Math.max(1, Number(config.num_primitives) || 400),
      allowed_shapes: config.allowed_shapes || ["circle"],
      transparent: transparentOutput,
      mask_threshold: Math.max(1, Math.min(254, Number(config.mask_threshold) || 127)),
    };

    const readySlots = pool.filter((slot) => slot.ready);
    const resultJson = readySlots.length > 1
      ? await runDistributed(readySlots, wasmConfig, rgba, alpha, onProgress)
      : await runOneShot(readySlots[0], wasmConfig, rgba, alpha, onProgress);
    const fit = JSON.parse(resultJson);
    if (fit.error) throw new Error(fit.error);

    // ------- 组装与后端一致的结果对象 -------
    const unitScale = Math.max(0.1, Number(config.image_scale) || 1.0);
    const outputAlpha = config.output_alpha != null ? Number(config.output_alpha) : 1.0;
    const imageCenter = resolveOrigin(config, fullW, fullH);
    const elements = resultsToElements(
      fit.shapes || [], unitScale, imageCenter, config.primitives || [], outputAlpha
    );

    if (needsWhiteBackground) {
      const bgCenterX = (fullW / 2) * unitScale;
      const bgCenterY = -(fullH / 2) * unitScale;
      const originX = imageCenter[0] * unitScale;
      const originY = -imageCenter[1] * unitScale;
      elements.unshift({
        type: "rectangle",
        shape: "rect",
        center: { x: round4(bgCenterX), y: round4(bgCenterY) },
        relative: { x: round4(bgCenterX - originX), y: round4(bgCenterY - originY) },
        size: {
          width: round4((fullW + BACKGROUND_BLEED_PX * 2) * unitScale),
          height: round4((fullH + BACKGROUND_BLEED_PX * 2) * unitScale),
        },
        rotation: 0,
        color: "#ffffff",
        alpha: 1.0,
        packed_color: 0xffffffff,
        is_background: true,
      });
    }

    const bbox = fit.bbox || [0, 0, fullW, fullH];
    const maskWidth = Math.max(1, bbox[2] - bbox[0]);
    const maskHeight = Math.max(1, bbox[3] - bbox[1]);
    const maskCenterX = (bbox[0] + bbox[2]) / 2;
    const maskCenterY = (bbox[1] + bbox[3]) / 2;
    const maskEnabled = !transparentOutput;

    const result = {
      mode: "fill",
      image_center: { x: imageCenter[0], y: imageCenter[1] },
      image_size: { width: fullW, height: fullH },
      config: {
        mode: "fill",
        engine: "primitive-wasm",
        fill_variant: transparentOutput ? "png" : "mask",
        enable_png_mode: enablePngMode,
        source_is_png: pngSource,
        source_has_transparency: hasAlpha,
        output_has_transparency: transparentOutput,
        pixel_per_unit: round4(1 / unitScale),
        unit_scale: unitScale,
        num_primitives: wasmConfig.num_primitives,
        mask_threshold: wasmConfig.mask_threshold,
        image_scale: unitScale,
        allowed_shapes: wasmConfig.allowed_shapes,
      },
      mask: {
        enabled: maskEnabled,
        shape_type: "rectangle",
        coverage: round4(fit.coverage || 0),
        center: { x: round4(maskCenterX * unitScale), y: round4(-maskCenterY * unitScale) },
        size: { width: round4(maskWidth * unitScale), height: round4(maskHeight * unitScale) },
        bbox_px: { x: bbox[0], y: bbox[1], width: maskWidth, height: maskHeight },
      },
      elements_count: elements.length,
      elements: elements,
      image_base64: null,
      preview_base64: null,
      mask_base64: maskEnabled ? fit.mask_png || null : null,
      elapsed_seconds: 0,
    };

    // 目标分辨率
    let outW = fullW;
    let outH = fullH;
    const targetW = Number(config.target_width) || 0;
    const targetH = Number(config.target_height) || 0;
    if (targetW >= 16 && targetH >= 16 && (targetW !== fullW || targetH !== fullH)) {
      outW = targetW;
      outH = targetH;
      rescaleResult(result, outW, outH);
    }

    // 输出尺寸的 base64 图
    const browserCanvas = document.createElement("canvas");
    browserCanvas.width = outW;
    browserCanvas.height = outH;
    const browserCtx = browserCanvas.getContext("2d");
    if (!transparentOutput) {
      browserCtx.fillStyle = "#ffffff";
      browserCtx.fillRect(0, 0, outW, outH);
    }
    browserCtx.drawImage(image, 0, 0, outW, outH);
    result.image_base64 = canvasToPngBase64(browserCanvas);

    const previewImage = await base64ToImage(fit.preview_png);
    const previewCanvas = drawToCanvas(previewImage, outW, outH, transparentOutput ? null : "#ffffff");
    result.preview_base64 = canvasToPngBase64(previewCanvas);

    if (result.mask_base64) {
      const maskImage = await base64ToImage(result.mask_base64);
      const maskCanvas = drawToCanvas(maskImage, outW, outH, "#000000");
      result.mask_base64 = canvasToPngBase64(maskCanvas);
    }

    result.elapsed_seconds = Math.round(((performance.now() - startedAt) / 1000) * 100) / 100;
    return { result, sourceImageBase64: await fileToBase64(file) };
  }

  /* 单实例一次性拟合（兜底路径） */
  async function runOneShot(slot, config, rgba, alpha, onProgress) {
    const data = await rpc(
      slot,
      "fit",
      { config, rgba, alpha, onProgress },
      [rgba.buffer, alpha.buffer]
    );
    return data.json;
  }

  /* 多实例 Step 内候选并行：与云端 `primitive -j N` 同语义。
   * 每个 Step：各实例从同一画布状态独立搜索候选（wm = ceil(16/N) 轮），
   * 汇总取 energy 最小，主实例 Model.Add 应用并进入下一步。 */
  async function runDistributed(slots, config, rgba, alpha, onProgress) {
    const main = slots[0];

    // 1) 全部实例建立会话（各自持有 Target/Worker，搜索互不干扰）
    const initResults = await Promise.all(slots.map((slot) => rpc(
      slot,
      "init",
      { config, rgba: rgba.slice(), alpha: alpha.slice() }
    )));
    const initData = initResults.map((data) => JSON.parse(data.json));
    const failed = initData.find((item) => !item || item.error || !item.ok);
    if (failed) throw new Error("本地引擎初始化失败：" + ((failed && failed.error) || "未知错误"));

    const steps = initData[0].steps;
    const total = initData[0].total;
    const wm = Math.max(1, Math.ceil(16 / slots.length));

    // 2) 逐步并行搜索 → 主实例应用最优
    let state = await rpc(main, "state", {});
    if (state.error) throw new Error(state.error);
    for (let i = 0; i < total; i += 1) {
      const mode = steps[i];
      const searches = slots.map((slot, index) => {
        const payload = slot === main
          ? { rgba: null, score: state.score, mode, wm, nonce: index * 131 + i }
          : { rgba: state.rgba.slice(), score: state.score, mode, wm, nonce: index * 131 + i };
        return rpc(slot, "search", payload).then((data) => JSON.parse(data.json));
      });
      const candidates = (await Promise.all(searches))
        .filter((item) => item && !item.error && item.type && Number.isFinite(item.energy));
      if (candidates.length === 0) {
        throw new Error(`第 ${i + 1} 步候选搜索失败`);
      }
      const best = candidates.reduce((a, b) => (a.energy <= b.energy ? a : b));
      state = await rpc(main, "apply", { candidate: JSON.stringify(best) });
      if (state.error) throw new Error(state.error);
      if (onProgress) onProgress(state.done, total);
    }

    // 3) 主实例汇总输出
    const finishData = await rpc(main, "finish", {});
    return finishData.json;
  }

  /* ------- 寄存结果并跳转 ------- */

  async function registerResult(result, config, imageName, sourceImageBase64) {
    const response = await fetch("/register_result", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        result: result,
        config: config,
        image_name: imageName,
        image_base64: sourceImageBase64 || "",
      }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error("结果寄存失败: " + (text || ("HTTP " + response.status)));
    }
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || "结果寄存失败");
    return payload.task_id;
  }

  window.LocalFit = {
    ensureReady,
    engineStatus,
    readyWorkerCount,
    poolSize,
    fitOne,
    registerResult,
  };
})();
