from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import fill_shaper
import final_shaper as fs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMITIVE_SRC = os.path.join(BASE_DIR, "third_party", "primitive")

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}
SHAPE_MODE_MAP = {"triangle": 1, "rect": 5, "circle": 7}
SHAPE_ORDER = ("circle", "rect", "triangle")
PNG_ALPHA_FIT_FLOOR = 0.2
PNG_ALPHA_FIT_GAMMA = 1.6
MIN_VISIBLE_ALPHA_WEIGHT = 0.05


def _rgb_to_hex(color):
    rgb = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{channel:02x}" for channel in rgb.tolist())


def _hex_to_rgb(hex_color):
    value = str(hex_color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"invalid color: {hex_color}")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _pack_color(color_hex, alpha):
    rgb = _hex_to_rgb(color_hex)
    alpha_int = int(np.clip(round(alpha * 255.0), 0, 255))
    return (
        (alpha_int << 24)
        | (int(rgb[0]) << 16)
        | (int(rgb[1]) << 8)
        | int(rgb[2])
    )


def _ensure_bgra(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        return image.copy()
    return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2BGRA)


def ensure_primitive_binary():
    binary_name = "primitive.exe" if os.name == "nt" else "primitive"
    candidate_bins = [
        os.path.join(BASE_DIR, "tools", binary_name),
        os.path.join(BASE_DIR, "build", binary_name),
        os.path.join(BASE_DIR, "build", "bin", binary_name),
    ]
    existing_binary = None
    for path in candidate_bins:
        if os.path.exists(path):
            existing_binary = path
            break

    go_candidates = []
    go_from_path = shutil.which("go")
    if go_from_path:
        go_candidates.append(go_from_path)
    bundled_go = os.path.join(BASE_DIR, "tools", "go", "go", "bin", "go.exe" if os.name == "nt" else "go")
    if os.path.exists(bundled_go):
        go_candidates.append(bundled_go)
    if existing_binary and not _primitive_source_is_newer(existing_binary):
        return existing_binary
    if not go_candidates:
        raise FileNotFoundError(f"{binary_name} 缺失，且本地未找到 Go 工具链")
    if not os.path.exists(PRIMITIVE_SRC):
        raise FileNotFoundError("primitive 源码目录缺失")

    output_path = candidate_bins[-1]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    go_exe = go_candidates[0]
    subprocess.run(
        [go_exe, "mod", "tidy"],
        cwd=PRIMITIVE_SRC,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [go_exe, "build", "-o", output_path, "."],
        cwd=PRIMITIVE_SRC,
        check=True,
        capture_output=True,
        text=True,
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"{binary_name} 构建失败")
    return output_path


def _primitive_source_is_newer(binary_path):
    if not os.path.exists(binary_path):
        return True
    binary_mtime = os.path.getmtime(binary_path)
    for root, _, files in os.walk(PRIMITIVE_SRC):
        for name in files:
            if not name.endswith((".go", ".mod", ".sum")):
                continue
            source_path = os.path.join(root, name)
            if os.path.getmtime(source_path) > binary_mtime:
                return True
    return False


def _clean_mask(mask):
    mask = mask.astype(bool)
    if np.any(mask) and np.any(~mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned > 0
    if np.any(mask):
        return mask
    return np.ones_like(mask, dtype=bool)


def _compress_alpha_for_fitting(alpha_channel, floor=PNG_ALPHA_FIT_FLOOR, gamma=PNG_ALPHA_FIT_GAMMA):
    alpha = np.clip(alpha_channel.astype(np.float32) / 255.0, 0.0, 1.0)
    floor = float(np.clip(floor, 0.0, 0.95))
    gamma = float(max(0.1, gamma))
    compressed = np.clip((alpha - floor) / max(1e-6, 1.0 - floor), 0.0, 1.0)
    compressed = np.power(compressed, gamma)
    return np.clip(np.rint(compressed * 255.0), 0, 255).astype(np.uint8)


def _prepare_transparent_target(image):
    alpha = image[:, :, 3].astype(np.float32) / 255.0
    flattened_rgb = image[:, :, :3].astype(np.float32)
    flattened_rgb = flattened_rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
    target_image = np.empty_like(image)
    target_image[:, :, :3] = np.clip(np.rint(flattened_rgb), 0, 255).astype(np.uint8)
    target_image[:, :, 3] = _compress_alpha_for_fitting(image[:, :, 3])
    return target_image


def _extract_image_and_mask(image, mask_threshold, use_alpha_target=False):
    if image.ndim == 2:
        target_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = fs.extract_mask(target_image) > 0
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        flattened = image[:, :, :3].astype(np.float32)
        flattened = flattened * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
        flattened = np.clip(np.rint(flattened), 0, 255).astype(np.uint8)
        target_image = _prepare_transparent_target(image) if use_alpha_target else flattened
        mask = image[:, :, 3] >= mask_threshold
    else:
        target_image = image[:, :, :3].copy()
        mask = fs.extract_mask(target_image) > 0
    image_rgba = _ensure_bgra(image if image.ndim == 3 and image.shape[2] == 4 else target_image)
    return target_image, _clean_mask(mask), image_rgba


def _compute_bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _normalize_allowed_shapes(allowed_shapes):
    requested = [str(name).strip().lower() for name in (allowed_shapes or ["circle"])]
    normalized = []
    for name in SHAPE_ORDER:
        if name in requested and name not in normalized:
            normalized.append(name)
    for name in requested:
        if name in SHAPE_MODE_MAP and name not in normalized:
            normalized.append(name)
    return normalized or ["circle"]


def _build_shape_configs(allowed_shapes, num_primitives):
    normalized = _normalize_allowed_shapes(allowed_shapes)
    total = max(1, int(num_primitives))
    base = total // len(normalized)
    remainder = total % len(normalized)
    configs = []
    for index, shape_name in enumerate(normalized):
        count = base + (1 if index < remainder else 0)
        if count <= 0:
            continue
        configs.append((SHAPE_MODE_MAP[shape_name], count))
    return configs or [(SHAPE_MODE_MAP["circle"], total)]


def _local_name(tag):
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _parse_transform_numbers(value):
    return [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", value or "")]


def _parse_fill(node):
    color_hex = node.attrib.get("fill", "#ffffff")
    alpha = float(node.attrib.get("fill-opacity", "1") or 1.0)
    return color_hex, alpha


def _parse_triangle(node, scale_x, scale_y, offset_x, offset_y):
    points_raw = re.findall(r"-?\d+(?:\.\d+)?", node.attrib.get("points", ""))
    if len(points_raw) != 6:
        return None
    pts = np.array([float(value) for value in points_raw], dtype=np.float64).reshape(3, 2)
    color_hex, alpha = _parse_fill(node)
    pts[:, 0] = offset_x + pts[:, 0] * scale_x
    pts[:, 1] = offset_y + pts[:, 1] * scale_y
    center = np.mean(pts, axis=0)
    width = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
    height = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    edge = pts[1] - pts[0]
    angle = math.degrees(math.atan2(edge[1], edge[0])) + 90.0
    return {
        "type": "triangle",
        "cx": float(center[0]),
        "cy": float(center[1]),
        "width": max(width, 1.0),
        "size": max(width, height, 1.0),
        "height": max(height, 1.0),
        "angle": angle,
        "color": color_hex,
        "alpha": alpha,
        "packed_color": _pack_color(color_hex, alpha),
    }


def _parse_shape_node(node, scale_x, scale_y, offset_x, offset_y):
    tag = _local_name(node.tag)
    if tag == "ellipse":
        color_hex, alpha = _parse_fill(node)
        cx = offset_x + float(node.attrib.get("cx", "0")) * scale_x
        cy = offset_y + float(node.attrib.get("cy", "0")) * scale_y
        rx = float(node.attrib.get("rx", "0")) * scale_x
        ry = float(node.attrib.get("ry", "0")) * scale_y
        return {
            "type": "circle",
            "cx": cx,
            "cy": cy,
            "rx": rx,
            "ry": ry,
            "angle": 0.0,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    if tag == "rect":
        color_hex, alpha = _parse_fill(node)
        x = float(node.attrib.get("x", "0"))
        y = float(node.attrib.get("y", "0"))
        width = float(node.attrib.get("width", "0"))
        height = float(node.attrib.get("height", "0"))
        cx = offset_x + (x + width / 2.0) * scale_x
        cy = offset_y + (y + height / 2.0) * scale_y
        return {
            "type": "rect",
            "cx": cx,
            "cy": cy,
            "hw": max(width * scale_x / 2.0, 0.5),
            "hh": max(height * scale_y / 2.0, 0.5),
            "angle": 0.0,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    if tag == "polygon":
        return _parse_triangle(node, scale_x, scale_y, offset_x, offset_y)
    return None


def _parse_nested_group(group, scale_x, scale_y, offset_x, offset_y):
    numbers = _parse_transform_numbers(group.attrib.get("transform", ""))
    if len(numbers) < 5:
        return None
    tx, ty, angle, sx, sy = numbers[:5]
    child = next(iter(group), None)
    if child is None:
        return None
    child_tag = _local_name(child.tag)
    color_hex, alpha = _parse_fill(child)
    if child_tag == "ellipse":
        return {
            "type": "circle",
            "cx": offset_x + tx * scale_x,
            "cy": offset_y + ty * scale_y,
            "rx": max(sx * scale_x, 0.5),
            "ry": max(sy * scale_y, 0.5),
            "angle": angle,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    if child_tag == "rect":
        return {
            "type": "rect",
            "cx": offset_x + tx * scale_x,
            "cy": offset_y + ty * scale_y,
            "hw": max(sx * scale_x / 2.0, 0.5),
            "hh": max(sy * scale_y / 2.0, 0.5),
            "angle": angle,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    return None


def parse_primitive_svg(svg_path, scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    outer_group = root.find("svg:g", SVG_NS)
    if outer_group is None:
        return []

    results = []
    for child in outer_group:
        tag = _local_name(child.tag)
        if tag == "g":
            shape = _parse_nested_group(child, scale_x, scale_y, offset_x, offset_y)
        else:
            shape = _parse_shape_node(child, scale_x, scale_y, offset_x, offset_y)
        if shape is not None:
            results.append(shape)
    return results


def _result_to_shape(result):
    shape_type = str(result.get("type", "")).strip().lower()
    if shape_type == "circle":
        return fill_shaper.Circle(
            cx=float(result.get("cx", 0.0)),
            cy=float(result.get("cy", 0.0)),
            radius_x=float(result.get("rx", 0.5)),
            radius_y=float(result.get("ry", result.get("rx", 0.5))),
            angle=float(result.get("angle", 0.0)),
        )
    if shape_type == "rect":
        return fill_shaper.Rect(
            cx=float(result.get("cx", 0.0)),
            cy=float(result.get("cy", 0.0)),
            half_width=float(result.get("hw", 0.5)),
            half_height=float(result.get("hh", result.get("hw", 0.5))),
            angle=float(result.get("angle", 0.0)),
        )
    if shape_type == "triangle":
        width = float(result.get("width", result.get("size", 1.0)))
        height = float(result.get("height", width))
        return fill_shaper.Triangle(
            cx=float(result.get("cx", 0.0)),
            cy=float(result.get("cy", 0.0)),
            base_width=width,
            height=height,
            angle=float(result.get("angle", 0.0)),
        )
    return None


def _apply_alpha_weights_to_results(results, alpha_weights, width, height):
    weights = np.clip(np.asarray(alpha_weights, dtype=np.float64), 0.0, 1.0)
    if weights.shape != (height, width):
        raise ValueError("alpha_weights shape mismatch")

    weighted_results = []
    for result in results:
        weighted = dict(result)
        shape = _result_to_shape(weighted)
        if shape is None:
            weighted_results.append(weighted)
            continue

        opacity = fill_shaper._shape_opacity(shape, weights, width=width, height=height)
        if opacity <= MIN_VISIBLE_ALPHA_WEIGHT:
            alpha = 0.0
        else:
            alpha = float(np.clip(float(weighted.get("alpha", 1.0)) * opacity, 0.0, 1.0))
        weighted["alpha"] = alpha
        if isinstance(weighted.get("color"), str):
            weighted["packed_color"] = _pack_color(weighted["color"], alpha)
        weighted_results.append(weighted)
    return weighted_results


def _render_preview(preview_image, full_width, full_height, alpha_map=None, transparent_output=False):
    if preview_image is None:
        raise ValueError("primitive output preview is missing")

    if preview_image.ndim == 2:
        preview_rgba = cv2.cvtColor(preview_image, cv2.COLOR_GRAY2BGRA)
    elif preview_image.shape[2] == 4:
        preview_rgba = preview_image.copy()
    else:
        preview_rgba = cv2.cvtColor(preview_image, cv2.COLOR_BGR2BGRA)

    preview_rgba = cv2.resize(preview_rgba, (full_width, full_height), interpolation=cv2.INTER_LINEAR)
    if transparent_output:
        alpha = preview_rgba[:, :, 3]
        if alpha_map is not None:
            resized_alpha = cv2.resize(alpha_map, (full_width, full_height), interpolation=cv2.INTER_LINEAR)
            alpha = np.rint(alpha.astype(np.float32) * (resized_alpha.astype(np.float32) / 255.0)).astype(np.uint8)
        preview_rgba[:, :, 3] = alpha
        return preview_rgba

    preview_rgba[:, :, 3] = 255
    return preview_rgba


def fit_image_with_primitive(image, config=None):
    if config is None:
        config = {}

    primitive_exe = ensure_primitive_binary()
    mask_threshold = int(max(1, min(254, config.get("mask_threshold", 127))))
    num_primitives = int(max(1, config.get("num_primitives", 400)))
    transparent_output = bool(config.get("transparent_output", False))

    target_image, mask, image_rgba = _extract_image_and_mask(
        image,
        mask_threshold,
        use_alpha_target=transparent_output,
    )
    full_height, full_width = target_image.shape[:2]
    x0, y0, x1, y1 = _compute_bbox(mask)

    detail_scale = float(max(0.25, config.get("detail_scale", 1.0)))
    full_max_dim = max(full_width, full_height)
    canvas_limit = int(max(16, min(round(full_max_dim * detail_scale), full_max_dim, 2048)))
    resize_ratio = min(1.0, float(canvas_limit) / float(max(full_max_dim, 1)))
    work_width = max(1, int(round(full_width * resize_ratio)))
    work_height = max(1, int(round(full_height * resize_ratio)))
    output_size = max(work_width, work_height)

    if work_width != full_width or work_height != full_height:
        interpolation = cv2.INTER_AREA if resize_ratio < 1.0 else cv2.INTER_LINEAR
        work_image = cv2.resize(target_image, (work_width, work_height), interpolation=interpolation)
    else:
        work_image = target_image.copy()

    scale_x = full_width / float(work_width)
    scale_y = full_height / float(work_height)
    shape_configs = _build_shape_configs(config.get("allowed_shapes", ["circle"]), num_primitives)

    with tempfile.TemporaryDirectory(prefix="primitive_fit_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        svg_path = os.path.join(tmpdir, "output.svg")
        png_path = os.path.join(tmpdir, "output.png")
        cv2.imwrite(input_path, work_image)

        cmd = [
            primitive_exe,
            "-i",
            input_path,
            "-o",
            svg_path,
            "-o",
            png_path,
            "-a",
            "0",
            "-r",
            "0",
            "-s",
            str(output_size),
        ]
        if transparent_output:
            cmd.extend(["-bg", "ffffff00"])
        else:
            cmd.extend(["-bg", "ffffff"])
        for mode, count in shape_configs:
            cmd.extend(["-m", str(mode), "-n", str(count)])
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        results = parse_primitive_svg(svg_path, scale_x=scale_x, scale_y=scale_y)
        preview_image = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if preview_image is None:
            raise ValueError("primitive 输出 PNG 读取失败")

    alpha_map = image[:, :, 3] if transparent_output and image.ndim == 3 and image.shape[2] == 4 else None
    if alpha_map is not None and results:
        results = _apply_alpha_weights_to_results(
            results,
            alpha_map.astype(np.float64) / 255.0,
            full_width,
            full_height,
        )
    preview = _render_preview(
        preview_image,
        full_width,
        full_height,
        alpha_map=alpha_map,
        transparent_output=transparent_output,
    )
    return {
        "results": results,
        "preview": preview,
        "image_bgr": target_image[:, :, :3].copy() if target_image.ndim == 3 and target_image.shape[2] == 4 else target_image,
        "image_rgba": image_rgba,
        "mask": mask,
        "bbox": {
            "x": x0,
            "y": y0,
            "width": max(1, x1 - x0),
            "height": max(1, y1 - y0),
        },
        "fit_size": canvas_limit,
    }
