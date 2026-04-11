"""
Core image processing entry points for the web app.

Fill mode is the primary workflow and uses the new mask-aware primitive fitter.
Outline mode is kept for compatibility with the older path-walking pipeline.
"""

from __future__ import annotations

import base64
import math
import time

import cv2
import numpy as np
from shapely.geometry import Polygon

import fill_shaper
import final_shaper as fs


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


def _extract_fill_image_and_mask(image, mask_threshold):
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = fs.extract_mask(image_bgr) > 0
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float64) / 255.0
        image_bgr = image[:, :, :3].astype(np.float64)
        image_bgr = image_bgr * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
        image_bgr = np.clip(np.rint(image_bgr), 0, 255).astype(np.uint8)
        mask = image[:, :, 3] >= int(mask_threshold)
    else:
        image_bgr = image[:, :, :3].copy()
        mask = fs.extract_mask(image_bgr) > 0
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


def process_image(image_bytes, config=None):
    if config is None:
        config = {}

    mode = config.get("mode", "fill")
    if mode == "outline":
        return process_image_outline(image_bytes, config)
    return process_image_fill(image_bytes, config)


def process_image_fill(image_bytes, config=None):
    if config is None:
        config = {}

    started = time.time()
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]
    image_center = _resolve_origin(config, width, height)
    unit_scale = float(max(0.1, config.get("image_scale", 1.0)))
    output_alpha = float(config.get("output_alpha", 1.0))
    allowed_types = _fill_allowed_types(config)
    enable_png_mode = bool(config.get("enable_png_mode", False))
    has_transparent_alpha = _has_transparent_alpha(image)
    mask_threshold = int(max(1, min(254, config.get("mask_threshold", 127))))
    transparent_output = has_transparent_alpha and enable_png_mode
    needs_white_background = has_transparent_alpha and not enable_png_mode

    if transparent_output:
        fit_variant = "png"
        coverage_weights = image[:, :, 3].astype(np.float64) / 255.0
        mask = None
        mask_enabled = False
        output_alpha_weights = coverage_weights
        coverage_for_bbox = coverage_weights > 1e-6
        min_mask_coverage = 0.12
        preview_alpha_map = coverage_weights
        browser_image = _prepare_browser_image(image, preserve_alpha=True)
        fit_image = image
    else:
        fit_variant = "mask"
        fit_image, mask = _extract_fill_image_and_mask(image, mask_threshold)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        mask = cleaned > 0
        coverage_weights = mask.astype(np.float64)
        mask_enabled = True
        output_alpha_weights = None
        coverage_for_bbox = mask
        min_mask_coverage = 0.55
        preview_alpha_map = np.ones_like(coverage_weights, dtype=np.float64)
        browser_image = fit_image

    num_primitives = int(config.get("num_primitives", 400))
    active_area = float(np.sum(coverage_weights))
    approx_size = math.sqrt(max(active_area, 1.0) / max(num_primitives, 1))
    fit_config = {
        "num_primitives": num_primitives,
        "allowed_types": allowed_types,
        "candidates": 24,
        "hill_climb_iter": 48,
        "min_size": max(2.0, approx_size * 0.28),
        "max_size": max(12.0, approx_size * 2.6),
        "min_mask_coverage": min_mask_coverage,
        "spill_penalty": 10000.0,
    }

    results = fill_shaper.fit_primitives(
        fit_image,
        config=fit_config,
        mask=mask,
        coverage_weights=coverage_weights,
        output_alpha_weights=output_alpha_weights,
    )
    preview = fill_shaper.render_results(
        fit_image,
        results,
        mask=mask,
        coverage_weights=coverage_weights,
        output_alpha_map=preview_alpha_map,
    )

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

    return {
        "mode": "fill",
        "image_center": {"x": image_center[0], "y": image_center[1]},
        "image_size": {"width": width, "height": height},
        "config": {
            "mode": "fill",
            "engine": "fill-shaper",
            "fill_variant": fit_variant,
            "enable_png_mode": enable_png_mode,
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
        "elapsed_seconds": round(time.time() - started, 2),
    }


def process_image_outline(image_bytes, config=None):
    if config is None:
        config = {}

    started = time.time()
    image = _decode_image(image_bytes)
    height, width = image.shape[:2]
    image_center = _resolve_origin(config, width, height)

    primitive_size = max(3, min(200, config.get("primitive_size", 15)))
    min_size = max(2, int(primitive_size * 0.4))
    max_size = max(min_size + 2, int(primitive_size * 2.0))

    mask = fs.extract_mask(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    allowed_types = None
    type_colors = {}
    primitive_presets = {}  # Store full primitive config for later use
    # Support both primitives_json (legacy) and allowed_shapes (new)
    shape_list = config.get("primitives", [])
    if not shape_list and "allowed_shapes" in config:
        shape_list = [{"shape": s, "color": "#ffffff"} for s in config["allowed_shapes"]]
    if shape_list:
        picked = set()
        for primitive in shape_list:
            shape = primitive.get("shape")
            color = primitive.get("color")
            if shape == "circle":
                picked.add(fs.ShapeType.ELLIPSE)
                if color:
                    type_colors[fs.ShapeType.ELLIPSE] = color
                # Store full preset config (image_asset_ref, type_id, etc.)
                primitive_presets[fs.ShapeType.ELLIPSE] = primitive
            elif shape == "rect":
                picked.add(fs.ShapeType.RECTANGLE)
                if color:
                    type_colors[fs.ShapeType.RECTANGLE] = color
                # Store full preset config (image_asset_ref, type_id, etc.)
                primitive_presets[fs.ShapeType.RECTANGLE] = primitive
        allowed_types = list(picked) if picked else []

    fitting_cfg = fs.FittingConfig(
        min_size=min_size,
        max_size=max_size,
        spacing_ratio=config.get("spacing", 0.9),
        precision=max(0.0, min(1.0, config.get("precision", 0.3))),
        allowed_types=allowed_types,
    )

    all_elements = []
    image_area = width * height
    min_contour_area = max(100, image_area * 0.00005)

    for contour_index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)
        if cw > width * 0.95 and ch > height * 0.95:
            continue
        margin = 5
        if (x + cw) < margin or x > (width - margin) or (y + ch) < margin or y > (height - margin):
            continue

        elements = fs.fit_beads(contour_index, contour, mask, dist_map, fitting_cfg, image_center)
        points = contour.reshape(-1, 2).astype(np.float64)
        if len(points) >= 3:
            polygon = Polygon(points).simplify(1.0, preserve_topology=True)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_valid and polygon.area >= fitting_cfg.min_size ** 2:
                cum_arc, total_arc = fs.build_arc_length_index(points)
                elements = fs.fill_gaps(
                    elements,
                    contour_index,
                    points,
                    cum_arc,
                    total_arc,
                    polygon,
                    dist_map,
                    image.shape,
                    fitting_cfg,
                    image_center,
                )
                elements = fs.expand_elements(elements, polygon, fitting_cfg, dist_map)
        all_elements.extend(elements)

    all_elements = fs.suppress_overlap(all_elements, image.shape)
    for element in all_elements:
        shape_type = element.get("type")
        if shape_type in type_colors:
            element["color"] = type_colors[shape_type]

    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image_bgr = image

    origin_units = {"x": image_center[0] / primitive_size, "y": -image_center[1] / primitive_size}
    exported_elements = []
    for element in all_elements:
        item = {key: value for key, value in element.items() if not key.startswith("_")}
        cx = float(item["center"]["x"]) / primitive_size
        cy = -float(item["center"]["y"]) / primitive_size

        if "size" in item:
            for key in list(item["size"].keys()):
                item["size"][key] = round(float(item["size"][key]) / primitive_size, 4)

        rot_z = -float(item.get("rotation", 0))
        if item.get("type") == fs.ShapeType.RECTANGLE and "size" in item:
            rect_h = float(item["size"].get("height", 0))
            theta = math.radians(rot_z)
            cx += (rect_h * 0.5) * math.sin(theta)
            cy += -(rect_h * 0.5) * math.cos(theta)

        item["center"]["x"] = round(cx, 4)
        item["center"]["y"] = round(cy, 4)
        item["relative_position"] = {
            "x": round(cx - origin_units["x"], 4),
            "y": round(cy - origin_units["y"], 4),
        }
        item["rotation"] = {"x": 0, "y": 0, "z": round(rot_z, 4)}

        # Convert ShapeType enum to string for consistency with fill mode
        shape_type = item.get("type")
        if shape_type == fs.ShapeType.ELLIPSE:
            item["type"] = "ellipse"
            item["shape"] = "circle"
            preset = primitive_presets.get(fs.ShapeType.ELLIPSE, {})
        elif shape_type == fs.ShapeType.RECTANGLE:
            item["type"] = "rectangle"
            item["shape"] = "rect"
            preset = primitive_presets.get(fs.ShapeType.RECTANGLE, {})
        else:
            preset = {}

        # Apply primitive preset config (image_asset_ref, type_id, etc.)
        if preset:
            if preset.get("image_asset_ref"):
                item["image_asset_ref"] = int(preset["image_asset_ref"])
            type_id = preset.get("type_id")
            element_type_id = preset.get("element_type_id")
            if type_id is not None:
                item["type_id"] = int(type_id)
            if element_type_id is not None:
                item["element_type_id"] = int(element_type_id)
            elif type_id is not None:
                item["element_type_id"] = int(type_id)
            if preset.get("rot_z") is not None:
                item["rotation"]["z"] = round(float(item["rotation"]["z"]) + float(preset["rot_z"]), 4)
            if preset.get("rot_y_add") is not None:
                item["rotation"]["y"] = float(preset["rot_y_add"])
                item["rot_y_add"] = float(preset["rot_y_add"])
            if preset.get("name"):
                item["name"] = str(preset["name"])

        exported_elements.append(item)

    return {
        "mode": "outline",
        "image_center": {"x": image_center[0], "y": image_center[1]},
        "image_size": {"width": width, "height": height},
        "config": {
            "mode": "outline",
            "primitive_size": primitive_size,
            "pixel_per_unit": primitive_size,
            "spacing": fitting_cfg.spacing_ratio,
            "precision": fitting_cfg.precision,
        },
        "elements_count": len(exported_elements),
        "elements": exported_elements,
        "image_base64": _encode_png_base64(image_bgr),
        "mask_base64": _encode_png_base64(mask),
        "elapsed_seconds": round(time.time() - started, 2),
    }
