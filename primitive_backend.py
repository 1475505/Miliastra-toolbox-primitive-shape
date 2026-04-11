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

import final_shaper as fs


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRIMITIVE_SRC = os.path.join(BASE_DIR, "third_party", "primitive")

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


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


def _extract_image_and_mask(image, mask_threshold):
    if image.ndim == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = fs.extract_mask(image_bgr) > 0
    elif image.shape[2] == 4:
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        image_bgr = image[:, :, :3].astype(np.float32)
        image_bgr = image_bgr * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])
        image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
        mask = image[:, :, 3] >= mask_threshold
    else:
        image_bgr = image[:, :, :3].copy()
        mask = fs.extract_mask(image_bgr) > 0
    return image_bgr, _clean_mask(mask)


def _compute_bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _shape_set_for_allowed(allowed_shapes):
    shape_names = list(allowed_shapes or ["circle"])
    mapped = []
    for name in shape_names:
        if name == "circle":
            mapped.append("rotatedellipse")
        elif name == "rect":
            mapped.append("rotatedrect")
        elif name == "triangle":
            mapped.append("triangle")
    if not mapped:
        mapped = ["rotatedellipse"]
    return ",".join(dict.fromkeys(mapped))


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


def _parse_triangle(node, scale_factor, offset_x, offset_y):
    points_raw = re.findall(r"-?\d+(?:\.\d+)?", node.attrib.get("points", ""))
    if len(points_raw) != 6:
        return None
    pts = np.array([float(value) for value in points_raw], dtype=np.float64).reshape(3, 2)
    color_hex, alpha = _parse_fill(node)
    center = np.mean(pts, axis=0)
    width = float(np.max(pts[:, 0]) - np.min(pts[:, 0])) * scale_factor
    height = float(np.max(pts[:, 1]) - np.min(pts[:, 1])) * scale_factor
    edge = pts[1] - pts[0]
    angle = math.degrees(math.atan2(edge[1], edge[0])) + 90.0
    cx = offset_x + float(center[0]) * scale_factor
    cy = offset_y + float(center[1]) * scale_factor
    return {
        "type": "triangle",
        "cx": cx,
        "cy": cy,
        "size": max(width, height, 1.0),
        "height": max(height, 1.0),
        "angle": angle,
        "color": color_hex,
        "alpha": alpha,
        "packed_color": _pack_color(color_hex, alpha),
    }


def _parse_shape_node(node, scale_factor, offset_x, offset_y):
    tag = _local_name(node.tag)
    if tag == "ellipse":
        color_hex, alpha = _parse_fill(node)
        cx = offset_x + float(node.attrib.get("cx", "0")) * scale_factor
        cy = offset_y + float(node.attrib.get("cy", "0")) * scale_factor
        rx = float(node.attrib.get("rx", "0")) * scale_factor
        ry = float(node.attrib.get("ry", "0")) * scale_factor
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
        cx = offset_x + (x + width / 2.0) * scale_factor
        cy = offset_y + (y + height / 2.0) * scale_factor
        return {
            "type": "rect",
            "cx": cx,
            "cy": cy,
            "hw": max(width * scale_factor / 2.0, 0.5),
            "hh": max(height * scale_factor / 2.0, 0.5),
            "angle": 0.0,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    if tag == "polygon":
        return _parse_triangle(node, scale_factor, offset_x, offset_y)
    return None


def _parse_nested_group(group, scale_factor, offset_x, offset_y):
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
            "cx": offset_x + tx * scale_factor,
            "cy": offset_y + ty * scale_factor,
            "rx": max(sx * scale_factor, 0.5),
            "ry": max(sy * scale_factor, 0.5),
            "angle": angle,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    if child_tag == "rect":
        return {
            "type": "rect",
            "cx": offset_x + tx * scale_factor,
            "cy": offset_y + ty * scale_factor,
            "hw": max(sx * scale_factor / 2.0, 0.5),
            "hh": max(sy * scale_factor / 2.0, 0.5),
            "angle": angle,
            "color": color_hex,
            "alpha": alpha,
            "packed_color": _pack_color(color_hex, alpha),
        }
    return None


def parse_primitive_svg(svg_path, scale_factor=1.0, offset_x=0.0, offset_y=0.0):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    outer_group = root.find("svg:g", SVG_NS)
    if outer_group is None:
        return []

    results = []
    for child in outer_group:
        tag = _local_name(child.tag)
        if tag == "g":
            shape = _parse_nested_group(child, scale_factor, offset_x, offset_y)
        else:
            shape = _parse_shape_node(child, scale_factor, offset_x, offset_y)
        if shape is not None:
            results.append(shape)
    return results


def _render_preview_on_canvas(preview_crop, full_width, full_height, bbox):
    x0, y0, x1, y1 = bbox
    canvas = np.full((full_height, full_width, 3), 255, dtype=np.uint8)
    crop_h = max(y1 - y0, 1)
    crop_w = max(x1 - x0, 1)
    resized = cv2.resize(preview_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    canvas[y0:y1, x0:x1] = resized
    return canvas


def fit_image_with_primitive(image, config=None):
    if config is None:
        config = {}

    primitive_exe = ensure_primitive_binary()
    mask_threshold = int(max(1, min(254, config.get("mask_threshold", 127))))
    num_primitives = int(max(1, config.get("num_primitives", 400)))

    image_bgr, mask = _extract_image_and_mask(image, mask_threshold)
    full_height, full_width = image_bgr.shape[:2]
    x0, y0, x1, y1 = _compute_bbox(mask)
    crop_bgr = image_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop_bgr[~crop_mask] = 255

    crop_h, crop_w = crop_bgr.shape[:2]
    max_crop_dim = max(crop_w, crop_h)
    detail_scale = float(max(0.25, config.get("detail_scale", 1.2)))
    target_fit_size = int(round(max_crop_dim * detail_scale))
    fit_size = int(max(16, min(target_fit_size, 2048)))
    resize_scale = max_crop_dim / float(fit_size)
    coord_scale = resize_scale
    output_size = max(32, fit_size)
    shape_set = _shape_set_for_allowed(config.get("allowed_shapes", ["circle"]))

    with tempfile.TemporaryDirectory(prefix="primitive_fit_") as tmpdir:
        input_path = os.path.join(tmpdir, "input.png")
        svg_path = os.path.join(tmpdir, "output.svg")
        png_path = os.path.join(tmpdir, "output.png")
        cv2.imwrite(input_path, crop_bgr)

        cmd = [
            primitive_exe,
            "-i",
            input_path,
            "-o",
            svg_path,
            "-o",
            png_path,
            "-n",
            str(num_primitives),
            "-a",
            "128",
            "-r",
            str(fit_size),
            "-s",
            str(output_size),
            "-shapes",
            shape_set,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        results = parse_primitive_svg(
            svg_path,
            scale_factor=coord_scale,
            offset_x=float(x0),
            offset_y=float(y0),
        )
        preview_crop = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if preview_crop is None:
            raise ValueError("primitive 输出 PNG 读取失败")

    preview = _render_preview_on_canvas(preview_crop, full_width, full_height, (x0, y0, x1, y1))
    return {
        "results": results,
        "preview": preview,
        "image_bgr": image_bgr,
        "mask": mask,
        "bbox": {
            "x": x0,
            "y": y0,
            "width": max(1, x1 - x0),
            "height": max(1, y1 - y0),
        },
        "fit_size": fit_size,
    }
