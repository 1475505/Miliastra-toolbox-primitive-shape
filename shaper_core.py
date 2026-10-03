"""
Core image processing entry points for the web app.

Fill mode uses the mask-aware primitive fitter (`primitive_backend` + `fill_shaper`).
"""

from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np

import fill_shaper
import primitive_backend


logger = logging.getLogger(__name__)


def _encode_png_base64(image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("failed to encode png")
    return base64.b64encode(buf).decode("utf-8")


def _decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法解码图片")
    return image


def _resolve_origin(config, width, height, prefix=""):
    origin_cfg = config.get("origin", {}) if not prefix else {
        "type": config.get(f"{prefix}origin_type", "center"),
        "x": config.get(f"{prefix}origin_x", ""),
        "y": config.get(f"{prefix}origin_y", ""),
    }
    if origin_cfg.get("type") == "custom":
        return (
            float(origin_cfg.get("x", width / 2.0)),
            float(origin_cfg.get("y", height / 2.0)),
        )
    if origin_cfg.get("type") == "top_left":
        return (0.0, 0.0)
    return (width / 2.0, height / 2.0)


def _has_transparent_alpha(image):
    return (
        image.ndim == 3
        and image.shape[2] == 4
        and bool(np.any(image[:, :, 3] < 255))
    )


def _flatten_to_bgr(image, background=255.0):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float64) / 255.0
        base = image[:, :, :3].astype(np.float64)
        flattened = base * alpha + float(background) * (1.0 - alpha)
        return np.clip(np.rint(flattened), 0, 255).astype(np.uint8)
    return image[:, :, :3].copy()


def _prepare_browser_image(image, preserve_alpha):
    if preserve_alpha and _has_transparent_alpha(image):
        return image.copy()
    return _flatten_to_bgr(image)


def _source_is_png(config):
    source_ext = str((config or {}).get("source_ext", "")).strip().lower()
    if source_ext:
        return source_ext == ".png"
    source_name = str((config or {}).get("source_filename", "")).strip().lower()
    return source_name.endswith(".png")


def _extract_fill_image_and_mask(image, mask_threshold):
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = fill_shaper.extract_mask(image_bgr) > 0
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float64) / 255.0
        image_bgr = image[:, :, :3].astype(np.float64)
        image_bgr = image_bgr * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
        image_bgr = np.clip(np.rint(image_bgr), 0, 255).astype(np.uint8)
        mask = image[:, :, 3] >= int(mask_threshold)
    else:
        image_bgr = image[:, :, :3].copy()
        mask = fill_shaper.extract_mask(image_bgr) > 0
    return image_bgr, mask


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        height, width = mask.shape[:2]
        return 0, 0, width, height
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _fill_allowed_types(config):
    shape_map = {
        "circle": fill_shaper.ShapeType.CIRCLE,
        "rect": fill_shaper.ShapeType.RECT,
        "triangle": fill_shaper.ShapeType.TRIANGLE,
    }
    allowed = []
    for shape_name in config.get("allowed_shapes", ["circle"]):
        mapped = shape_map.get(str(shape_name).strip().lower())
        if mapped and mapped not in allowed:
            allowed.append(mapped)
    return allowed or [fill_shaper.ShapeType.CIRCLE]


def _resolve_target_resolution(config, width, height):
    """解析目标输出分辨率；未启用或非法时返回 None。"""
    try:
        target_width = int(config.get("target_width", 0) or 0)
        target_height = int(config.get("target_height", 0) or 0)
    except (TypeError, ValueError):
        return None
    target_width = max(0, min(4096, target_width))
    target_height = max(0, min(4096, target_height))
    if target_width <= 0 or target_height <= 0:
        return None
    if target_width == width and target_height == height:
        return None
    return target_width, target_height


def _scale_element_size_keys(size, ratio_x, ratio_y):
    scaled = {}
    for key, value in size.items():
        factor = ratio_x if key in ("width", "rx") else ratio_y
        try:
            scaled[key] = round(float(value) * factor, 4)
        except (TypeError, ValueError):
            scaled[key] = value
    return scaled


def _rescale_fill_output(result, target_width, target_height):
    """把 fill 结果从原始分辨率重定标到目标分辨率。

    所有以像素推导的字段（elements 坐标/尺寸、mask、预览图）都按
    (rx, ry) 缩放；unit_scale 语义不变（缩放发生在像素域）。
    """
    width = result["image_size"]["width"]
    height = result["image_size"]["height"]
    ratio_x = target_width / float(width)
    ratio_y = target_height / float(height)

    origin_x = float(result["image_center"]["x"]) * ratio_x
    origin_y = float(result["image_center"]["y"]) * ratio_y
    unit_scale = float(result["config"].get("unit_scale", 1.0))
    origin_units_x = origin_x * unit_scale
    origin_units_y = -origin_y * unit_scale

    for element in result["elements"]:
        center = element.get("center", {})
        new_cx = round(float(center.get("x", 0.0)) * ratio_x, 4)
        new_cy = round(float(center.get("y", 0.0)) * ratio_y, 4)
        element["center"] = {"x": new_cx, "y": new_cy}
        relative = {"x": round(new_cx - origin_units_x, 4), "y": round(new_cy - origin_units_y, 4)}
        element["relative"] = dict(relative)
        if "relative_position" in element:
            element["relative_position"] = dict(relative)
        if "size" in element:
            element["size"] = _scale_element_size_keys(element["size"], ratio_x, ratio_y)

    mask = result.get("mask")
    if mask:
        mask_center = mask.get("center", {})
        mask["center"] = {
            "x": round(float(mask_center.get("x", 0.0)) * ratio_x, 4),
            "y": round(float(mask_center.get("y", 0.0)) * ratio_y, 4),
        }
        mask_size = mask.get("size", {})
        mask["size"] = {
            "width": round(float(mask_size.get("width", 0.0)) * ratio_x, 4),
            "height": round(float(mask_size.get("height", 0.0)) * ratio_y, 4),
        }
        bbox = mask.get("bbox_px")
        if bbox:
            mask["bbox_px"] = {
                "x": int(round(bbox.get("x", 0) * ratio_x)),
                "y": int(round(bbox.get("y", 0) * ratio_y)),
                "width": max(1, int(round(bbox.get("width", 1) * ratio_x))),
                "height": max(1, int(round(bbox.get("height", 1) * ratio_y))),
            }

    def _resize_b64(key):
        encoded = result.get(key)
        if not encoded:
            return
        raw = base64.b64decode(encoded)
        image = _decode_image(raw)
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        result[key] = _encode_png_base64(resized)

    _resize_b64("image_base64")
    _resize_b64("preview_base64")
    _resize_b64("mask_base64")

    result["image_center"] = {"x": origin_x, "y": origin_y}
    result["image_size"] = {"width": target_width, "height": target_height}
    result["config"]["target_width"] = target_width
    result["config"]["target_height"] = target_height


def process_image(image_bytes, config=None):
    """处理入口。当前仅保留填充拟合（历史轮廓/装饰物模式已移除）。"""
    return process_image_fill(image_bytes, config or {})


def process_image_fill(image_bytes, config=None):
    if config is None:
        config = {}

    started = time.time()
    perf_started = time.perf_counter()
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]
    image_center = _resolve_origin(config, width, height)
    unit_scale = float(max(0.1, config.get("image_scale", 1.0)))
    output_alpha = float(config.get("output_alpha", 1.0))
    enable_png_mode = bool(config.get("enable_png_mode", False))
    source_is_png = _source_is_png(config)
    has_transparent_alpha = _has_transparent_alpha(image)
    png_with_transparency = source_is_png and has_transparent_alpha
    mask_threshold = int(max(1, min(254, config.get("mask_threshold", 127))))
    transparent_output = png_with_transparency and enable_png_mode
    needs_white_background = png_with_transparency and not enable_png_mode

    logger.info(
        "fill_process start image=%dx%d mode=%s png_mode=%s transparent_src=%s "
        "num_primitives=%s detail_scale=%s allowed_shapes=%s",
        width,
        height,
        config.get("mode", "fill"),
        enable_png_mode,
        has_transparent_alpha,
        config.get("num_primitives", 400),
        config.get("detail_scale", 1.0),
        config.get("allowed_shapes", ["circle"]),
    )

    if transparent_output:
        fit_variant = "png"
        mask_enabled = False
        coverage_for_bbox = image[:, :, 3] > 0
        browser_image = _prepare_browser_image(image, preserve_alpha=True)
    else:
        fit_variant = "mask"
        mask_enabled = True
        fit_image, mask = _extract_fill_image_and_mask(image, mask_threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        coverage_for_bbox = cleaned > 0
        browser_image = fit_image

    num_primitives = int(config.get("num_primitives", 400))
    primitive_result = primitive_backend.fit_image_with_primitive(
        image,
        config={
            "num_primitives": num_primitives,
            "allowed_shapes": config.get("allowed_shapes", ["circle"]),
            "mask_threshold": mask_threshold,
            "detail_scale": float(config.get("detail_scale", 1.0)),
            "transparent_output": transparent_output,
        },
    )
    results = primitive_result["results"]
    preview = primitive_result["preview"]

    elements = fill_shaper.results_to_elements(
        results,
        unit_scale,
        image_center,
        config.get("primitives", []),
        output_alpha=output_alpha,
    )

    # 默认模式（PNG模式关闭）：添加白色背景图元
    if needs_white_background:
        # 白色矩形背景，覆盖整个图像区域
        background_bleed_px = 4.0
        bg_center_x = (width / 2.0) * unit_scale
        bg_center_y = -(height / 2.0) * unit_scale
        origin_x = float(image_center[0]) * unit_scale
        origin_y = -float(image_center[1]) * unit_scale
        bg_element = {
            "type": "rectangle",
            "shape": "rect",
            "center": {
                "x": round(bg_center_x, 4),
                "y": round(bg_center_y, 4),
            },
            "relative": {
                "x": round(bg_center_x - origin_x, 4),
                "y": round(bg_center_y - origin_y, 4),
            },
            "size": {
                "width": round((width + background_bleed_px * 2.0) * unit_scale, 4),
                "height": round((height + background_bleed_px * 2.0) * unit_scale, 4),
            },
            "rotation": 0.0,
            "color": "#ffffff",
            "alpha": 1.0,
            "packed_color": 0xFFFFFFFF,  # 白色不透明
            "is_background": True,
        }
        elements.insert(0, bg_element)

    x0, y0, x1, y1 = _mask_bbox(coverage_for_bbox)
    mask_width = max(1, x1 - x0)
    mask_height = max(1, y1 - y0)
    mask_center_x = (x0 + x1) / 2.0
    mask_center_y = (y0 + y1) / 2.0
    elapsed = time.time() - started
    logger.info(
        "fill_process done elapsed=%.3fs elements=%d fit_variant=%s mask_bbox=%dx%d preview=%dx%d",
        time.perf_counter() - perf_started,
        len(elements),
        fit_variant,
        mask_width,
        mask_height,
        preview.shape[1],
        preview.shape[0],
    )

    result = {
        "mode": "fill",
        "image_center": {"x": image_center[0], "y": image_center[1]},
        "image_size": {"width": width, "height": height},
        "config": {
            "mode": "fill",
            "engine": "primitive",
            "fill_variant": fit_variant,
            "enable_png_mode": enable_png_mode,
            "source_is_png": source_is_png,
            "source_has_transparency": has_transparent_alpha,
            "output_has_transparency": transparent_output,
            "pixel_per_unit": round(1.0 / unit_scale, 6),
            "unit_scale": unit_scale,
            "num_primitives": num_primitives,
            "mask_threshold": mask_threshold,
            "image_scale": unit_scale,
            "allowed_shapes": config.get("allowed_shapes", ["circle"]),
        },
        "mask": {
            "enabled": mask_enabled,
            "shape_type": "rectangle",
            "coverage": round(float(np.mean(coverage_for_bbox)), 4),
            "center": {
                "x": round(mask_center_x * unit_scale, 4),
                "y": round(-mask_center_y * unit_scale, 4),
            },
            "size": {
                "width": round(mask_width * unit_scale, 4),
                "height": round(mask_height * unit_scale, 4),
            },
            "bbox_px": {
                "x": x0,
                "y": y0,
                "width": mask_width,
                "height": mask_height,
            },
        },
        "elements_count": len(elements),
        "elements": elements,
        "image_base64": _encode_png_base64(browser_image),
        "preview_base64": _encode_png_base64(preview),
        "mask_base64": _encode_png_base64((coverage_for_bbox.astype(np.uint8) * 255)) if mask_enabled else None,
        "elapsed_seconds": round(elapsed, 2),
    }

    target_resolution = _resolve_target_resolution(config, width, height)
    if target_resolution is not None:
        _rescale_fill_output(result, target_resolution[0], target_resolution[1])
        logger.info(
            "fill_process rescaled to target resolution %dx%d",
            target_resolution[0],
            target_resolution[1],
        )
    return result

