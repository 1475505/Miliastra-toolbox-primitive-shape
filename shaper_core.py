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
import primitive_backend


def _encode_png_base64(image):
    # Handle RGBA images to preserve alpha channel
    if image.ndim == 3 and image.shape[2] == 4:
        ok, buf = cv2.imencode(".png", image)
    else:
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
    primitive_fit = primitive_backend.fit_image_with_primitive(
        image,
        {
            "num_primitives": int(config.get("num_primitives", 400)),
            "allowed_shapes": config.get("allowed_shapes", ["circle"]),
            "mask_threshold": int(max(1, min(254, config.get("mask_threshold", 127)))),
            "detail_scale": float(max(0.25, config.get("detail_scale", 1.0))),
        },
    )
    results = primitive_fit["results"]
    preview = primitive_fit["preview"]
    image_bgr = primitive_fit["image_bgr"]
    image_rgba = primitive_fit.get("image_rgba", image_bgr)
    mask = primitive_fit["mask"]
    elements = fill_shaper.results_to_elements(
        results,
        unit_scale,
        image_center,
        config.get("primitives", []),
        output_alpha=output_alpha,
    )

    bbox = primitive_fit["bbox"]
    x0 = bbox["x"]
    y0 = bbox["y"]
    mask_width = bbox["width"]
    mask_height = bbox["height"]
    x1 = x0 + mask_width
    y1 = y0 + mask_height
    mask_center_x = (x0 + x1) / 2.0
    mask_center_y = (y0 + y1) / 2.0

    return {
        "mode": "fill",
        "image_center": {"x": image_center[0], "y": image_center[1]},
        "image_size": {"width": width, "height": height},
        "config": {
            "mode": "fill",
            "engine": "primitive-official",
            "pixel_per_unit": round(1.0 / unit_scale, 6),
            "unit_scale": unit_scale,
            "num_primitives": int(config.get("num_primitives", 400)),
            "fit_size": primitive_fit["fit_size"],
            "mask_threshold": int(max(1, min(254, config.get("mask_threshold", 127)))),
            "image_scale": unit_scale,
            "allowed_shapes": config.get("allowed_shapes", ["circle"]),
        },
        "mask": {
            "enabled": True,
            "shape_type": "rectangle",
            "coverage": round(float(np.mean(mask)), 4),
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
        "image_base64": _encode_png_base64(image_rgba),
        "preview_base64": _encode_png_base64(preview),
        "mask_base64": _encode_png_base64(mask.astype(np.uint8) * 255),
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
    if "primitives" in config and config["primitives"]:
        picked = set()
        for primitive in config["primitives"]:
            shape = primitive.get("shape")
            color = primitive.get("color")
            if shape == "circle":
                picked.add(fs.ShapeType.ELLIPSE)
                if color:
                    type_colors[fs.ShapeType.ELLIPSE] = color
            elif shape == "rect":
                picked.add(fs.ShapeType.RECTANGLE)
                if color:
                    type_colors[fs.ShapeType.RECTANGLE] = color
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
