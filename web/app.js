(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const data = window.RESULT;
  if (!data) return;

  const mode = data.mode || "fill";
  const fillVariant = (data.config && data.config.fill_variant) || "mask";
  const outputHasTransparency = Boolean(data.config && data.config.output_has_transparency);
  const taskImageName = (window.TASK_IMAGE_NAME || "").trim();
  const exportBaseName = (taskImageName.replace(/\.[^.]+$/, "").trim()) || "shaper_result";
  const imageWidth = data.image_size.width;
  const imageHeight = data.image_size.height;
  const pixelPerUnit = (data.config && data.config.pixel_per_unit) || (data.config && data.config.primitive_size) || 1;

  const canvas = $("mainCanvas");
  const ctx = canvas.getContext("2d");
  const originInputX = $("originX");
  const originInputY = $("originY");

  let scale = 1;
  let hoveredIndex = null;
  let selectedIndex = null;
  let origin = {
    x: data.image_center.x,
    y: data.image_center.y,
  };

  const assets = {
    base: null,
    mask: null,
  };

  function isBackgroundElement(element) {
    return Boolean(element && element.is_background);
  }

  function visibleElements() {
    return data.elements.filter((element) => !isBackgroundElement(element));
  }

  function renderOrderedElements() {
    const background = [];
    const foreground = [];
    data.elements.forEach((element) => {
      if (isBackgroundElement(element)) background.push(element);
      else foreground.push(element);
    });
    return background.concat(foreground);
  }

  function displayElementIndex(index) {
    let displayIndex = -1;
    for (let i = 0; i <= index; i += 1) {
      if (!isBackgroundElement(data.elements[i])) displayIndex += 1;
    }
    return Math.max(displayIndex, 0);
  }

  function exportFileName(ext) {
    return `${exportBaseName}.${ext}`;
  }

  function loadImage(base64) {
    return new Promise((resolve) => {
      if (!base64) {
        resolve(null);
        return;
      }
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = "data:image/png;base64," + base64;
    });
  }

  function cssColor(color, alphaOverride) {
    if (Array.isArray(color)) {
      const alpha = color.length >= 4 ? color[3] / 255 : alphaOverride;
      return `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;
    }
    if (typeof color === "string") {
      const value = color.trim();
      if (/^#([a-f0-9]{3}|[a-f0-9]{6})$/i.test(value)) {
        let hex = value.slice(1);
        if (hex.length === 3) hex = hex.split("").map((part) => part + part).join("");
        const r = parseInt(hex.slice(0, 2), 16);
        const g = parseInt(hex.slice(2, 4), 16);
        const b = parseInt(hex.slice(4, 6), 16);
        return `rgba(${r},${g},${b},${alphaOverride})`;
      }
      return value;
    }
    return `rgba(255, 204, 0, ${alphaOverride})`;
  }

  function normalizeType(type) {
    if (type === "circle") return "ellipse";
    if (type === "rect") return "rectangle";
    return type;
  }

  function shapeName(type) {
    const normalized = normalizeType(type);
    if (normalized === "ellipse") return "圆形";
    if (normalized === "rectangle") return "矩形";
    if (normalized === "triangle") return "三角形";
    return normalized || "未知";
  }

  function imagePointToElementCenter(element) {
    return {
      x: element.center.x * pixelPerUnit,
      y: -element.center.y * pixelPerUnit,
    };
  }

  function elementRotationRad(element) {
    const rot = element.rotation ? Number(element.rotation.z || 0) : 0;
    return -rot * Math.PI / 180;
  }

  function drawElementPath(element) {
    const type = normalizeType(element.type);

    if (type === "ellipse") {
      const rx = (element.size.rx || (element.size.width || 0) / 2) * pixelPerUnit;
      const ry = (element.size.ry || (element.size.height || 0) / 2) * pixelPerUnit;
      ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
      return;
    }

    if (type === "triangle") {
      const width = (element.size.width || 0) * pixelPerUnit;
      const triHeight = (element.size.height || 0) * pixelPerUnit || (width * Math.sqrt(3) / 2);
      const topY = -2 * triHeight / 3;
      const bottomY = triHeight / 3;
      ctx.moveTo(0, topY);
      ctx.lineTo(-width / 2, bottomY);
      ctx.lineTo(width / 2, bottomY);
      ctx.closePath();
      return;
    }

    const width = (element.size.width || 0) * pixelPerUnit;
    const height = (element.size.height || 0) * pixelPerUnit;
    ctx.rect(-width / 2, -height / 2, width, height);
  }

  function drawElement(element, index) {
    const center = imagePointToElementCenter(element);
    const selected = !isBackgroundElement(element) && index === selectedIndex;
    const highlighted = !isBackgroundElement(element) && (selected || (selectedIndex === null && index === hoveredIndex));

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(elementRotationRad(element));
    ctx.beginPath();
    drawElementPath(element);

    if ($("showFill").checked) {
      if (selected) ctx.fillStyle = "rgba(37, 99, 235, 0.42)";
      else if (highlighted) ctx.fillStyle = "rgba(59, 130, 246, 0.28)";
      else ctx.fillStyle = cssColor(element.color, Math.max(0.12, Number(element.alpha || 0.5)));
      ctx.fill();
    }

    if ($("showBorder").checked) {
      if (selected) {
        ctx.strokeStyle = "#1d4ed8";
        ctx.lineWidth = 2 / scale;
      } else if (highlighted) {
        ctx.strokeStyle = "#3b82f6";
        ctx.lineWidth = 1.5 / scale;
      } else {
        ctx.strokeStyle = cssColor(element.color, 0.95);
        ctx.lineWidth = 1 / scale;
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  function drawOriginCross() {
    if (!$("showOrigin").checked) return;
    const arm = 12 / scale;
    ctx.save();
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 1.5 / scale;
    ctx.beginPath();
    ctx.moveTo(origin.x - arm, origin.y);
    ctx.lineTo(origin.x + arm, origin.y);
    ctx.moveTo(origin.x, origin.y - arm);
    ctx.lineTo(origin.x, origin.y + arm);
    ctx.stroke();
    ctx.fillStyle = "#ef4444";
    ctx.font = `${11 / scale}px sans-serif`;
    ctx.fillText("原点", origin.x + arm + 3, origin.y - 3);
    ctx.restore();
  }

  function render() {
    const wrap = $("canvasWrap");
    if (!wrap) return;

    scale = Math.min(1, (wrap.clientWidth - 32) / imageWidth, (wrap.clientHeight - 32) / imageHeight);
    canvas.width = Math.round(imageWidth * scale);
    canvas.height = Math.round(imageHeight * scale);

    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, imageWidth, imageHeight);
    if (!outputHasTransparency) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, imageWidth, imageHeight);
    }

    if ($("showImage").checked && assets.base) {
      ctx.drawImage(assets.base, 0, 0, imageWidth, imageHeight);
    }

    if ($("showMask").checked && assets.mask) {
      ctx.save();
      ctx.globalAlpha = 0.22;
      ctx.drawImage(assets.mask, 0, 0, imageWidth, imageHeight);
      ctx.restore();
    }

    renderOrderedElements().forEach(drawElement);
    drawOriginCross();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function canvasToImagePoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * imageWidth / rect.width,
      y: (event.clientY - rect.top) * imageHeight / rect.height,
    };
  }

  function pointInElement(element, x, y) {
    const type = normalizeType(element.type);
    const center = imagePointToElementCenter(element);
    const angle = -elementRotationRad(element);
    const dx0 = x - center.x;
    const dy0 = y - center.y;
    const dx = dx0 * Math.cos(angle) - dy0 * Math.sin(angle);
    const dy = dx0 * Math.sin(angle) + dy0 * Math.cos(angle);

    if (type === "ellipse") {
      const rx = (element.size.rx || (element.size.width || 0) / 2) * pixelPerUnit;
      const ry = (element.size.ry || (element.size.height || 0) / 2) * pixelPerUnit;
      if (rx <= 0 || ry <= 0) return false;
      return (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry) <= 1;
    }

    if (type === "triangle") {
      const width = (element.size.width || 0) * pixelPerUnit;
      const triHeight = (element.size.height || 0) * pixelPerUnit || (width * Math.sqrt(3) / 2);
      const vertices = [
        { x: 0, y: -2 * triHeight / 3 },
        { x: -width / 2, y: triHeight / 3 },
        { x: width / 2, y: triHeight / 3 },
      ];
      const crosses = [];
      for (let i = 0; i < 3; i += 1) {
        const a = vertices[i];
        const b = vertices[(i + 1) % 3];
        crosses.push((dx - a.x) * (b.y - a.y) - (dy - a.y) * (b.x - a.x));
      }
      return crosses.every((value) => value >= 0) || crosses.every((value) => value <= 0);
    }

    const width = (element.size.width || 0) * pixelPerUnit;
    const height = (element.size.height || 0) * pixelPerUnit;
    return Math.abs(dx) <= width / 2 && Math.abs(dy) <= height / 2;
  }

  function hitTest(x, y) {
    for (let i = data.elements.length - 1; i >= 0; i -= 1) {
      if (isBackgroundElement(data.elements[i])) continue;
      if (pointInElement(data.elements[i], x, y)) return i;
    }
    return null;
  }

  function relativePosition(element) {
    return {
      x: element.center.x - origin.x / pixelPerUnit,
      y: element.center.y + origin.y / pixelPerUnit,
    };
  }

  function clearDetail() {
    const empty = $("infoEmpty");
    const panel = $("infoPanel");
    if (empty) empty.hidden = false;
    if (panel) panel.hidden = true;
    ["infoId", "infoType", "infoCenter", "infoRelative", "infoSize", "infoRotation"].forEach((id) => {
      const node = $(id);
      if (node) node.textContent = "—";
    });
  }

  function showDetail(element, index) {
    const empty = $("infoEmpty");
    const panel = $("infoPanel");
    if (empty) empty.hidden = true;
    if (panel) panel.hidden = false;

    const relative = relativePosition(element);
    const rotation = element.rotation ? Number(element.rotation.z || 0) : 0;
    const normalizedType = normalizeType(element.type);

    $("infoId").textContent = element.id != null ? element.id : displayElementIndex(index);
    $("infoType").textContent = shapeName(element.type);
    $("infoCenter").textContent = `(${element.center.x.toFixed(2)}, ${element.center.y.toFixed(2)})`;
    $("infoRelative").textContent = `(${relative.x.toFixed(2)}, ${relative.y.toFixed(2)})`;
    if (normalizedType === "ellipse") {
      $("infoSize").textContent = `rx=${Number(element.size.rx || 0).toFixed(2)} ry=${Number(element.size.ry || 0).toFixed(2)}`;
    } else if (normalizedType === "triangle") {
      $("infoSize").textContent = `base=${Number(element.size.width || 0).toFixed(2)} height=${Number(element.size.height || 0).toFixed(2)}`;
    } else {
      $("infoSize").textContent = `${Number(element.size.width || 0).toFixed(2)} × ${Number(element.size.height || 0).toFixed(2)}`;
    }
    $("infoRotation").textContent = `${rotation.toFixed(1)}°`;
  }

  function downloadBlob(blob, name) {
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(anchor.href);
  }

  function downloadBase64(base64, name) {
    fetch("data:image/png;base64," + base64)
      .then((response) => response.blob())
      .then((blob) => downloadBlob(blob, name));
  }

  function updateStats() {
    const elements = visibleElements();
    const ellipseCount = elements.filter((element) => normalizeType(element.type) === "ellipse").length;
    const triangleCount = elements.filter((element) => normalizeType(element.type) === "triangle").length;
    const rectCount = elements.length - ellipseCount - triangleCount;

    if ($("modeLabel")) $("modeLabel").textContent = mode === "fill" ? "填充拟合" : "轮廓描边";
    if ($("statMode")) $("statMode").textContent = mode === "fill" ? "填充拟合" : "轮廓描边";
    if ($("statTotal")) $("statTotal").textContent = String(elements.length);
    if ($("statEllipse")) $("statEllipse").textContent = String(ellipseCount);
    if ($("statRect")) $("statRect").textContent = String(rectCount);
    if ($("statTriangle")) $("statTriangle").textContent = String(triangleCount);
    if ($("statImgSize")) $("statImgSize").textContent = `${imageWidth}×${imageHeight}`;
    if ($("elemCountDisplay")) $("elemCountDisplay").textContent = `图元: ${elements.length}`;

    const fillRetry = $("retrySectionFill");
    const outlineRetry = $("retrySectionOutline");
    if (fillRetry) fillRetry.hidden = mode !== "fill";
    if (outlineRetry) outlineRetry.hidden = mode === "fill";

    const compare = $("previewCompare");
    if (compare) compare.hidden = !(mode === "fill" && data.preview_base64);
    if ($("previewImg") && data.preview_base64) $("previewImg").src = "data:image/png;base64," + data.preview_base64;
    if ($("originalThumb") && data.image_base64) $("originalThumb").src = "data:image/png;base64," + data.image_base64;
  }

  function applyVariantUi() {
    const modeText = mode === "fill"
      ? (fillVariant === "png" ? "PNG 填充" : "填充拟合")
      : "轮廓描边";
    if ($("modeLabel")) $("modeLabel").textContent = modeText;
    if ($("statMode")) $("statMode").textContent = modeText;
    if ($("showMask") && $("showMask").parentElement) {
      const hasMask = Boolean(data.mask_base64);
      if (!hasMask) $("showMask").checked = false;
      $("showMask").disabled = !hasMask;
      $("showMask").parentElement.style.opacity = hasMask ? "1" : "0.45";
    }
  }

  async function init() {
    originInputX.value = origin.x;
    originInputY.value = origin.y;
    updateStats();
    applyVariantUi();
    clearDetail();

    const [base, mask] = await Promise.all([
      loadImage(data.image_base64),
      loadImage(data.mask_base64),
    ]);
    assets.base = base;
    assets.mask = mask;
    render();
  }

  ["showImage", "showMask", "showFill", "showBorder", "showOrigin"].forEach((id) => {
    const node = $(id);
    if (node) node.addEventListener("change", render);
  });

  canvas.addEventListener("mousemove", (event) => {
    const point = canvasToImagePoint(event);
    const worldX = point.x / pixelPerUnit;
    const worldY = -point.y / pixelPerUnit;
    if ($("coordsDisplay")) $("coordsDisplay").textContent = `坐标: (${worldX.toFixed(2)}, ${worldY.toFixed(2)})`;

    const nextHovered = hitTest(point.x, point.y);
    if (nextHovered !== hoveredIndex) {
      hoveredIndex = nextHovered;
      render();
    }

    const tooltip = $("tooltip");
    if (!tooltip) return;
    if (hoveredIndex !== null && hoveredIndex !== selectedIndex) {
      const element = data.elements[hoveredIndex];
      const relative = relativePosition(element);
      const rotation = element.rotation ? Number(element.rotation.z || 0) : 0;
      tooltip.hidden = false;
      tooltip.innerHTML = [
        `<b>#${displayElementIndex(hoveredIndex)}</b> ${shapeName(element.type)}`,
        `中心: (${element.center.x.toFixed(2)}, ${element.center.y.toFixed(2)})`,
        `相对原点: (${relative.x.toFixed(2)}, ${relative.y.toFixed(2)})`,
        normalizeType(element.type) === "ellipse"
          ? `尺寸: rx=${Number(element.size.rx || 0).toFixed(2)} ry=${Number(element.size.ry || 0).toFixed(2)}`
          : normalizeType(element.type) === "triangle"
            ? `尺寸: base=${Number(element.size.width || 0).toFixed(2)} height=${Number(element.size.height || 0).toFixed(2)}`
            : `尺寸: ${Number(element.size.width || 0).toFixed(2)} × ${Number(element.size.height || 0).toFixed(2)}`,
        `旋转: ${rotation.toFixed(1)}°`,
      ].join("<br>");

      const wrapRect = $("canvasWrap").getBoundingClientRect();
      tooltip.style.left = `${event.clientX - wrapRect.left + 14}px`;
      tooltip.style.top = `${event.clientY - wrapRect.top + 14}px`;
    } else {
      tooltip.hidden = true;
    }
  });

  canvas.addEventListener("mouseleave", () => {
    hoveredIndex = null;
    if ($("tooltip")) $("tooltip").hidden = true;
    render();
  });

  canvas.addEventListener("click", (event) => {
    const point = canvasToImagePoint(event);
    selectedIndex = hitTest(point.x, point.y);
    hoveredIndex = null;
    render();
    if (selectedIndex === null) clearDetail();
    else showDetail(data.elements[selectedIndex], selectedIndex);
  });

  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    const point = canvasToImagePoint(event);
    origin = { x: point.x, y: point.y };
    originInputX.value = point.x.toFixed(1);
    originInputY.value = point.y.toFixed(1);
    if (selectedIndex !== null) showDetail(data.elements[selectedIndex], selectedIndex);
    render();
  });

  originInputX.addEventListener("change", () => {
    origin.x = Number(originInputX.value || 0);
    if (selectedIndex !== null) showDetail(data.elements[selectedIndex], selectedIndex);
    render();
  });

  originInputY.addEventListener("change", () => {
    origin.y = Number(originInputY.value || 0);
    if (selectedIndex !== null) showDetail(data.elements[selectedIndex], selectedIndex);
    render();
  });

  if ($("btnResetOrigin")) {
    $("btnResetOrigin").addEventListener("click", () => {
      origin = { x: data.image_center.x, y: data.image_center.y };
      originInputX.value = origin.x;
      originInputY.value = origin.y;
      if (selectedIndex !== null) showDetail(data.elements[selectedIndex], selectedIndex);
      render();
    });
  }

  if ($("btnExportJSON")) {
    $("btnExportJSON").addEventListener("click", () => {
      const originUnits = { x: origin.x / pixelPerUnit, y: -origin.y / pixelPerUnit };
      const payload = {
        image_name: exportBaseName || null,
        group_name: exportBaseName || null,
        origin: originUnits,
        image_size: data.image_size,
        config: window.TASK_CFG || data.config,
        mask: data.mask || null,
        elements: renderOrderedElements().map((element, index) => ({
          id: element.id != null ? element.id : index,
          type: normalizeType(element.type),
          shape: element.shape,
          center: element.center,
          relative: {
            x: +(element.center.x - originUnits.x).toFixed(4),
            y: +(element.center.y - originUnits.y).toFixed(4),
          },
          size: element.size,
          rotation: element.rotation,
          color: element.color,
          alpha: element.alpha,
          packed_color: element.packed_color,
          image_asset_ref: element.image_asset_ref,
          type_id: element.type_id,
          element_type_id: element.element_type_id,
          name: element.name,
        })),
      };
      downloadBlob(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), exportFileName("json"));
    });
  }

  if ($("btnExportPNG")) {
    $("btnExportPNG").addEventListener("click", () => {
      if (mode === "fill" && data.preview_base64) {
        downloadBase64(data.preview_base64, exportFileName("png"));
        return;
      }
      canvas.toBlob((blob) => {
        if (blob) downloadBlob(blob, exportFileName("png"));
      }, "image/png");
    });
  }

  if ($("btnExportGIAOverlimit")) {
    $("btnExportGIAOverlimit").addEventListener("click", () => {
      const taskId = window.TASK_ID || "";
      const qs = new URLSearchParams({ origin_x: origin.x, origin_y: origin.y }).toString();
      fetch(`/download_overlimit_gia/${encodeURIComponent(taskId)}?${qs}`)
        .then((response) => {
          if (!response.ok) {
            return response.text().then((text) => {
              throw new Error(text || `HTTP ${response.status}`);
            });
          }
          return response.blob();
        })
        .then((blob) => downloadBlob(blob, exportFileName("gia")))
        .catch((error) => alert(`导出失败: ${error && error.message ? error.message : error}`));
    });
  }

  window.addEventListener("resize", render);
  init();
})();
