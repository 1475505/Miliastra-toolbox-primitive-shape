"""
Mask-aware primitive fitting used by the fill mode.

The implementation keeps the public entry points used by the rest of the
project, but fixes the main issues in the previous version:
1. Candidates are evaluated against the foreground mask.
2. Output shape names match the rest of the app.
3. Colors are exported in a frontend/GIA-friendly format.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np


class ShapeType:
    CIRCLE = "circle"
    RECT = "rect"
    TRIANGLE = "triangle"


@dataclass
class FillConfig:
    num_primitives: int = 100
    candidates: int = 32
    hill_climb_iter: int = 64
    min_size: float = 4.0
    max_size: float = 96.0
    allowed_types: list[str] | None = None
    alpha_range: tuple[float, float] = (0.15, 1.0)
    min_mask_coverage: float = 0.7
    spill_penalty: float = 12000.0
    random_seed: int | None = None


def _empty_pixels():
    return (
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
    )


class Shape:
    shape_type = None

    def rasterize(self, width: int, height: int):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def mutate(self, rng: np.random.Generator, width: int, height: int, cfg: FillConfig):
        raise NotImplementedError


class Circle(Shape):
    shape_type = ShapeType.CIRCLE

    def __init__(self, cx: float, cy: float, radius_x: float, radius_y: float | None = None, angle: float = 0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.radius_x = float(radius_x)
        self.radius_y = float(radius_x if radius_y is None else radius_y)
        self.angle = float(angle)

    def copy(self):
        return Circle(self.cx, self.cy, self.radius_x, self.radius_y, self.angle)

    def mutate(self, rng, width, height, cfg):
        candidate = self.copy()
        step = max(cfg.min_size, min(cfg.max_size, max(self.radius_x, self.radius_y))) * rng.choice([0.35, 0.7, 1.0])
        roll = rng.random()
        if roll < 0.25:
            candidate.cx = float(np.clip(candidate.cx + rng.normal(0, step), 0, width - 1))
        elif roll < 0.5:
            candidate.cy = float(np.clip(candidate.cy + rng.normal(0, step), 0, height - 1))
        elif roll < 0.7:
            candidate.radius_x = float(np.clip(candidate.radius_x + rng.normal(0, step * 0.5), cfg.min_size, cfg.max_size))
        elif roll < 0.9:
            candidate.radius_y = float(np.clip(candidate.radius_y + rng.normal(0, step * 0.5), cfg.min_size, cfg.max_size))
        else:
            candidate.angle = float(candidate.angle + rng.normal(0, 14.0))
        return candidate

    def rasterize(self, width, height):
        radius_x = max(self.radius_x, 0.5)
        radius_y = max(self.radius_y, 0.5)
        angle = math.radians(self.angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        extent_x = math.sqrt((radius_x * cos_a) ** 2 + (radius_y * sin_a) ** 2) + 1
        extent_y = math.sqrt((radius_x * sin_a) ** 2 + (radius_y * cos_a) ** 2) + 1
        x0 = max(int(math.floor(self.cx - extent_x)), 0)
        x1 = min(int(math.ceil(self.cx + extent_x)), width)
        y0 = max(int(math.floor(self.cy - extent_y)), 0)
        y1 = min(int(math.ceil(self.cy + extent_y)), height)
        if x0 >= x1 or y0 >= y1:
            return _empty_pixels()

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - self.cx
        dy = ys - self.cy
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        norm = (local_x / radius_x) ** 2 + (local_y / radius_y) ** 2
        dist = np.sqrt(np.maximum(norm, 0.0))
        edge_scale = max(radius_x, radius_y)
        aa = np.clip((1.0 - dist) * edge_scale + 0.5, 0.0, 1.0)
        inside = aa > 0.0
        if not inside.any():
            return _empty_pixels()
        return ys[inside].astype(np.int32), xs[inside].astype(np.int32), aa[inside].astype(np.float64)


class Rect(Shape):
    shape_type = ShapeType.RECT

    def __init__(self, cx: float, cy: float, half_width: float, half_height: float, angle: float = 0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.half_width = float(half_width)
        self.half_height = float(half_height)
        self.angle = float(angle)

    def copy(self):
        return Rect(self.cx, self.cy, self.half_width, self.half_height, self.angle)

    def mutate(self, rng, width, height, cfg):
        candidate = self.copy()
        size_step = max(cfg.min_size, min(cfg.max_size, max(self.half_width, self.half_height))) * rng.choice([0.35, 0.7, 1.0])
        roll = rng.random()
        if roll < 0.25:
            candidate.cx = float(np.clip(candidate.cx + rng.normal(0, size_step), 0, width - 1))
        elif roll < 0.5:
            candidate.cy = float(np.clip(candidate.cy + rng.normal(0, size_step), 0, height - 1))
        elif roll < 0.7:
            candidate.half_width = float(np.clip(candidate.half_width + rng.normal(0, size_step * 0.35), cfg.min_size, cfg.max_size))
        elif roll < 0.9:
            candidate.half_height = float(np.clip(candidate.half_height + rng.normal(0, size_step * 0.35), cfg.min_size, cfg.max_size))
        else:
            candidate.angle = float(candidate.angle + rng.normal(0, 12.0))
        return candidate

    def rasterize(self, width, height):
        half_width = max(self.half_width, 0.5)
        half_height = max(self.half_height, 0.5)
        angle = math.radians(self.angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        extent = math.sqrt(half_width ** 2 + half_height ** 2) + 1
        x0 = max(int(math.floor(self.cx - extent)), 0)
        x1 = min(int(math.ceil(self.cx + extent)), width)
        y0 = max(int(math.floor(self.cy - extent)), 0)
        y1 = min(int(math.ceil(self.cy + extent)), height)
        if x0 >= x1 or y0 >= y1:
            return _empty_pixels()

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - self.cx
        dy = ys - self.cy
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a
        dist_x = half_width + 0.5 - np.abs(local_x)
        dist_y = half_height + 0.5 - np.abs(local_y)
        aa = np.clip(np.minimum(dist_x, dist_y), 0.0, 1.0)
        inside = aa > 0.0
        if not inside.any():
            return _empty_pixels()
        return ys[inside].astype(np.int32), xs[inside].astype(np.int32), aa[inside].astype(np.float64)


class Triangle(Shape):
    shape_type = ShapeType.TRIANGLE

    def __init__(self, cx: float, cy: float, base_width: float, height: float | None = None, angle: float = 0.0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.base_width = float(base_width)
        self.height = float(base_width if height is None else height)
        self.angle = float(angle)

    def copy(self):
        return Triangle(self.cx, self.cy, self.base_width, self.height, self.angle)

    def mutate(self, rng, width, height, cfg):
        candidate = self.copy()
        step = max(cfg.min_size, min(cfg.max_size, max(self.base_width, self.height))) * rng.choice([0.35, 0.7, 1.0])
        roll = rng.random()
        if roll < 0.25:
            candidate.cx = float(np.clip(candidate.cx + rng.normal(0, step), 0, width - 1))
        elif roll < 0.5:
            candidate.cy = float(np.clip(candidate.cy + rng.normal(0, step), 0, height - 1))
        elif roll < 0.7:
            candidate.base_width = float(np.clip(candidate.base_width + rng.normal(0, step * 0.45), cfg.min_size, cfg.max_size))
        elif roll < 0.9:
            candidate.height = float(np.clip(candidate.height + rng.normal(0, step * 0.45), cfg.min_size, cfg.max_size))
        else:
            candidate.angle = float(candidate.angle + rng.normal(0, 18.0))
        return candidate

    def rasterize(self, width, height):
        base_width = max(self.base_width, 0.5)
        tri_h = max(self.height, 0.5)
        local_vertices = np.array([
            [0.0, -2.0 * tri_h / 3.0],
            [-base_width / 2.0, tri_h / 3.0],
            [base_width / 2.0, tri_h / 3.0],
        ])

        angle = math.radians(self.angle)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world_vertices = local_vertices @ rotation.T
        world_vertices[:, 0] += self.cx
        world_vertices[:, 1] += self.cy

        x0 = max(int(math.floor(np.min(world_vertices[:, 0]) - 1)), 0)
        x1 = min(int(math.ceil(np.max(world_vertices[:, 0]) + 1)), width)
        y0 = max(int(math.floor(np.min(world_vertices[:, 1]) - 1)), 0)
        y1 = min(int(math.ceil(np.max(world_vertices[:, 1]) + 1)), height)
        if x0 >= x1 or y0 >= y1:
            return _empty_pixels()

        ys, xs = np.mgrid[y0:y1, x0:x1]
        points = np.stack([xs, ys], axis=-1).astype(np.float64)
        edge_a = world_vertices[1] - world_vertices[0]
        edge_b = world_vertices[2] - world_vertices[1]
        edge_c = world_vertices[0] - world_vertices[2]

        def cross(edge, origin):
            return (points[..., 0] - origin[0]) * edge[1] - (points[..., 1] - origin[1]) * edge[0]

        c1 = cross(edge_a, world_vertices[0])
        c2 = cross(edge_b, world_vertices[1])
        c3 = cross(edge_c, world_vertices[2])
        same_side = ((c1 >= 0) & (c2 >= 0) & (c3 >= 0)) | ((c1 <= 0) & (c2 <= 0) & (c3 <= 0))
        if not same_side.any():
            return _empty_pixels()

        def distance_to_segment(a, b):
            ap = points - a
            ab = b - a
            ab_len_sq = np.maximum(np.sum(ab * ab), 1e-8)
            t = np.clip((ap[..., 0] * ab[0] + ap[..., 1] * ab[1]) / ab_len_sq, 0.0, 1.0)
            closest = a + t[..., None] * ab
            return np.sqrt(np.sum((points - closest) ** 2, axis=-1))

        dist = np.minimum.reduce([
            distance_to_segment(world_vertices[0], world_vertices[1]),
            distance_to_segment(world_vertices[1], world_vertices[2]),
            distance_to_segment(world_vertices[2], world_vertices[0]),
        ])
        aa = np.clip(0.5 + dist, 0.0, 1.0)
        aa = np.where(same_side, np.minimum(aa, 1.0), 0.0)
        inside = aa > 0.0
        if not inside.any():
            return _empty_pixels()
        return ys[inside].astype(np.int32), xs[inside].astype(np.int32), aa[inside].astype(np.float64)


def _bgr_to_hex(color):
    bgr = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    rgb = bgr[::-1]
    return "#" + "".join(f"{channel:02x}" for channel in rgb.tolist())


def _hex_to_rgb(hex_color):
    value = str(hex_color).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"invalid color: {hex_color}")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


def _hex_to_bgr(hex_color):
    return _hex_to_rgb(hex_color)[::-1]


def _pack_color(color_hex, alpha):
    rgb = _hex_to_rgb(color_hex)
    alpha_int = int(np.clip(round(alpha * 255.0), 0, 255))
    return (
        (alpha_int << 24)
        | (int(rgb[0]) << 16)
        | (int(rgb[1]) << 8)
        | int(rgb[2])
    )


def _prepare_target(target_img, mask):
    if target_img.ndim == 2:
        target = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR).astype(np.float64)
    elif target_img.shape[2] == 4:
        alpha = target_img[:, :, 3:4].astype(np.float64) / 255.0
        target = target_img[:, :, :3].astype(np.float64)
        target = target * alpha + 255.0 * (1.0 - alpha)
    else:
        target = target_img[:, :, :3].astype(np.float64)

    canvas = np.full_like(target, 255.0, dtype=np.float64)
    canvas[~mask] = 255.0
    return target, canvas


def _resolve_coverage_weights(target_img, mask=None, coverage_weights=None):
    if coverage_weights is not None:
        weights = np.asarray(coverage_weights, dtype=np.float64)
    elif mask is not None:
        weights = np.asarray(mask, dtype=np.float64)
    elif target_img.ndim == 3 and target_img.shape[2] == 4:
        weights = target_img[:, :, 3].astype(np.float64) / 255.0
    else:
        weights = np.ones(target_img.shape[:2], dtype=np.float64)

    if weights.shape != target_img.shape[:2]:
        raise ValueError("coverage_weights shape mismatch")
    return np.clip(weights, 0.0, 1.0)


def _shape_opacity(shape, coverage_weights, width, height):
    ys, xs, alphas = shape.rasterize(width, height)
    if len(ys) == 0:
        return 0.0
    total = float(np.sum(alphas))
    if total <= 1e-8:
        return 0.0
    return float(np.sum(alphas * coverage_weights[ys, xs]) / total)


def _build_alpha_candidates(cfg):
    alpha_min = float(np.clip(cfg.alpha_range[0], 0.05, 1.0))
    alpha_max = float(np.clip(cfg.alpha_range[1], alpha_min, 1.0))
    return np.unique(np.linspace(alpha_min, alpha_max, 3).round(3))


def _sample_focus(error_map, mask, rng):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    masked_error = error_map[mask]
    top_k = min(512, masked_error.size)
    if top_k <= 1:
        idx = 0
    else:
        top_indices = np.argpartition(masked_error, -top_k)[-top_k:]
        weights = masked_error[top_indices] + 1e-6
        weights = weights / np.sum(weights)
        idx = int(rng.choice(top_indices, p=weights))
    return float(xs[idx]), float(ys[idx])


def _random_size(rng, cfg):
    low = max(float(cfg.min_size), 0.5)
    high = max(low, float(cfg.max_size))
    if low == high:
        return low
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def random_shape(rng, width, height, cfg, focus=None):
    allowed = cfg.allowed_types or [ShapeType.CIRCLE]
    shape_type = str(rng.choice(allowed))

    if focus is None:
        cx = rng.uniform(0, width - 1)
        cy = rng.uniform(0, height - 1)
    else:
        fx, fy = focus
        jitter = max(cfg.min_size * 1.2, cfg.max_size * 0.35)
        cx = float(np.clip(rng.normal(fx, jitter), 0, width - 1))
        cy = float(np.clip(rng.normal(fy, jitter), 0, height - 1))

    if shape_type == ShapeType.CIRCLE:
        radius_major = _random_size(rng, cfg)
        radius_minor = max(cfg.min_size * 0.6, min(cfg.max_size, radius_major * math.exp(rng.uniform(-0.8, 0.8))))
        return Circle(cx, cy, radius_major, radius_minor, rng.uniform(-180.0, 180.0))
    if shape_type == ShapeType.TRIANGLE:
        return Triangle(
            cx,
            cy,
            _random_size(rng, cfg),
            _random_size(rng, cfg),
            rng.uniform(-180.0, 180.0),
        )
    return Rect(cx, cy, _random_size(rng, cfg), _random_size(rng, cfg), rng.uniform(-90.0, 90.0))


def _shape_fixed_color(shape, fixed_color_map):
    if not fixed_color_map:
        return None
    return fixed_color_map.get(shape.shape_type)


def compute_color(target, canvas, coverage_weights, shape, alpha, fixed_color=None):
    ys, xs, alphas = shape.rasterize(target.shape[1], target.shape[0])
    if len(ys) == 0:
        return None

    if fixed_color is not None:
        return np.array(fixed_color, dtype=np.float64)

    shape_alpha = alpha * alphas
    pixel_weights = coverage_weights[ys, xs]
    valid = (shape_alpha > 1e-8) & (pixel_weights > 1e-8)
    if not np.any(valid):
        return None

    active_alpha = shape_alpha[valid]
    active_weights = pixel_weights[valid]
    base = canvas[ys[valid], xs[valid]]
    target_pixels = target[ys[valid], xs[valid]]
    weighted_sq = np.sum((active_alpha ** 2) * active_weights)
    if weighted_sq <= 1e-8:
        return None

    solved = np.sum(
        (active_alpha * active_weights)[:, None] * (target_pixels - base * (1.0 - active_alpha[:, None])),
        axis=0,
    ) / weighted_sq
    color = solved
    return np.clip(color, 0, 255)


def compute_score(target, canvas, coverage_weights, shape, color, alpha, cfg):
    ys, xs, alphas = shape.rasterize(target.shape[1], target.shape[0])
    if len(ys) == 0 or color is None:
        return float("inf"), None

    pixel_weights = coverage_weights[ys, xs]
    weighted = alpha * alphas
    if np.sum(weighted) <= 1e-8:
        return float("inf"), None

    coverage = float(np.sum(weighted * pixel_weights) / np.maximum(np.sum(weighted), 1e-8))
    if coverage < cfg.min_mask_coverage:
        return float("inf"), None

    active = (pixel_weights > 1e-8) & (weighted > 1e-8)
    if not np.any(active):
        return float("inf"), None

    inside_ys = ys[active]
    inside_xs = xs[active]
    inside_alpha = weighted[active]
    inside_weights = pixel_weights[active]
    old_pixels = canvas[inside_ys, inside_xs]
    new_pixels = old_pixels * (1.0 - inside_alpha[:, None]) + color * inside_alpha[:, None]
    old_error = target[inside_ys, inside_xs] - old_pixels
    new_error = target[inside_ys, inside_xs] - new_pixels
    delta = float(np.sum((new_error ** 2 - old_error ** 2) * inside_weights[:, None]))

    outside_alpha = weighted * (1.0 - pixel_weights)
    if outside_alpha.size:
        delta += float(np.sum(outside_alpha) * cfg.spill_penalty)

    payload = {
        "ys": inside_ys,
        "xs": inside_xs,
        "alphas": inside_alpha,
        "coverage": coverage,
    }
    return delta, payload


def hill_climb(target, canvas, coverage_weights, shape, color, alpha, cfg, rng, fixed_color=None):
    best_shape = shape
    best_color = color
    best_score, best_payload = compute_score(target, canvas, coverage_weights, shape, color, alpha, cfg)
    width = target.shape[1]
    height = target.shape[0]

    for _ in range(max(1, int(cfg.hill_climb_iter))):
        candidate_shape = best_shape.mutate(rng, width, height, cfg)
        candidate_color = compute_color(target, canvas, coverage_weights, candidate_shape, alpha, fixed_color=fixed_color)
        candidate_score, candidate_payload = compute_score(target, canvas, coverage_weights, candidate_shape, candidate_color, alpha, cfg)
        if candidate_score < best_score:
            best_shape = candidate_shape
            best_color = candidate_color
            best_score = candidate_score
            best_payload = candidate_payload

    return best_shape, best_color, best_score, best_payload


def _apply_payload(canvas, color, payload):
    if payload is None:
        return
    ys = payload["ys"]
    xs = payload["xs"]
    alphas = payload["alphas"]
    canvas[ys, xs] = canvas[ys, xs] * (1.0 - alphas[:, None]) + color * alphas[:, None]


def _serialize_shape(shape, color_hex, alpha, packed_color):
    if isinstance(shape, Circle):
        return {
            "type": ShapeType.CIRCLE,
            "cx": float(shape.cx),
            "cy": float(shape.cy),
            "rx": float(shape.radius_x),
            "ry": float(shape.radius_y),
            "angle": float(shape.angle),
            "color": color_hex,
            "alpha": float(alpha),
            "packed_color": int(packed_color),
        }
    if isinstance(shape, Triangle):
        return {
            "type": ShapeType.TRIANGLE,
            "cx": float(shape.cx),
            "cy": float(shape.cy),
            "size": float(shape.base_width),
            "width": float(shape.base_width),
            "height": float(shape.height),
            "angle": float(shape.angle),
            "color": color_hex,
            "alpha": float(alpha),
            "packed_color": int(packed_color),
        }
    return {
        "type": ShapeType.RECT,
        "cx": float(shape.cx),
        "cy": float(shape.cy),
        "hw": float(shape.half_width),
        "hh": float(shape.half_height),
        "angle": float(shape.angle),
        "color": color_hex,
        "alpha": float(alpha),
        "packed_color": int(packed_color),
    }


def fit_primitives(target_img, config=None, progress_callback=None, mask=None, fixed_color_map=None, coverage_weights=None, output_alpha_weights=None):
    """
    Fit an image using a sequence of simple primitives.

    Args:
        target_img: BGR/BGRA/GRAY image.
        config: FillConfig or dict.
        progress_callback: optional callback(step, total, message).
        mask: optional boolean foreground mask.
        fixed_color_map: optional shape-type -> RGB override.
        coverage_weights: optional float map in [0, 1] for soft foreground weighting.
        output_alpha_weights: optional float map in [0, 1] used to scale exported primitive alpha.

    Returns:
        list[dict]
    """
    if config is None:
        config = FillConfig()
    elif isinstance(config, dict):
        config = FillConfig(**config)

    coverage = _resolve_coverage_weights(target_img, mask=mask, coverage_weights=coverage_weights)
    render_mask = coverage > 1e-6
    target, canvas = _prepare_target(target_img, render_mask)
    alpha_candidates = _build_alpha_candidates(config)
    rng = np.random.default_rng(config.random_seed)
    started = time.time()
    results = []

    for step in range(int(config.num_primitives)):
        if progress_callback:
            progress_callback(step, int(config.num_primitives), f"拟合图元 {step + 1}/{config.num_primitives}")

        error_map = np.sum((target - canvas) ** 2, axis=2)
        error_map[~render_mask] = 0.0
        focus = _sample_focus(error_map, render_mask, rng)

        best_shape = None
        best_color = None
        best_alpha = None
        best_score = float("inf")
        best_payload = None

        for _ in range(max(1, int(config.candidates))):
            candidate_shape = random_shape(rng, target.shape[1], target.shape[0], config, focus=focus)
            fixed_color = _shape_fixed_color(candidate_shape, fixed_color_map)
            for alpha in alpha_candidates:
                candidate_color = compute_color(target, canvas, coverage, candidate_shape, float(alpha), fixed_color=fixed_color)
                candidate_score, candidate_payload = compute_score(target, canvas, coverage, candidate_shape, candidate_color, float(alpha), config)
                if candidate_score < best_score:
                    best_shape = candidate_shape
                    best_color = candidate_color
                    best_alpha = float(alpha)
                    best_score = candidate_score
                    best_payload = candidate_payload

        if best_shape is None or best_color is None or best_payload is None or not np.isfinite(best_score):
            continue

        fixed_color = _shape_fixed_color(best_shape, fixed_color_map)
        best_shape, best_color, best_score, best_payload = hill_climb(
            target,
            canvas,
            coverage,
            best_shape,
            best_color,
            best_alpha,
            config,
            rng,
            fixed_color=fixed_color,
        )

        if best_payload is None:
            continue

        _apply_payload(canvas, best_color, best_payload)

        color_hex = _bgr_to_hex(best_color)
        export_alpha = float(best_alpha)
        if output_alpha_weights is not None:
            shape_opacity = _shape_opacity(best_shape, output_alpha_weights, target.shape[1], target.shape[0])
            if shape_opacity <= 0.05:
                export_alpha = 0.0
        export_alpha = float(np.clip(export_alpha, 0.0, 1.0))
        packed_color = _pack_color(color_hex, export_alpha)

        results.append(_serialize_shape(best_shape, color_hex, export_alpha, packed_color))

    if progress_callback:
        progress_callback(
            int(config.num_primitives),
            int(config.num_primitives),
            f"完成，耗时 {time.time() - started:.1f}s",
        )

    return results


def render_results(target_img, results, mask=None, coverage_weights=None, output_alpha_map=None):
    coverage = _resolve_coverage_weights(target_img, mask=mask, coverage_weights=coverage_weights)
    render_mask = coverage > 1e-6
    target, canvas = _prepare_target(target_img, render_mask)
    _ = target

    for result in results:
        if result["type"] == ShapeType.CIRCLE:
            shape = Circle(result["cx"], result["cy"], result["rx"], result["ry"], result.get("angle", 0.0))
        elif result["type"] == ShapeType.TRIANGLE:
            tri_width = float(result.get("width", result.get("size", 1.0)))
            tri_height = float(result.get("height", tri_width * math.sqrt(3.0) / 2.0))
            shape = Triangle(result["cx"], result["cy"], tri_width, tri_height, result.get("angle", 0.0))
        else:
            shape = Rect(result["cx"], result["cy"], result["hw"], result["hh"], result.get("angle", 0.0))

        color = _hex_to_bgr(result["color"]) if isinstance(result.get("color"), str) else np.array(result["color"], dtype=np.float64)
        _, payload = compute_score(
            target=np.zeros_like(canvas),
            canvas=canvas,
            coverage_weights=np.ones_like(coverage, dtype=np.float64),
            shape=shape,
            color=color,
            alpha=float(result.get("alpha", 1.0)),
            cfg=FillConfig(spill_penalty=0.0, min_mask_coverage=0.0),
        )
        _apply_payload(canvas, color, payload)

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    rgba = np.zeros((*canvas.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = canvas
    if output_alpha_map is None:
        rgba[:, :, 3] = (render_mask * 255).astype(np.uint8)
    else:
        alpha_map = np.clip(np.asarray(output_alpha_map, dtype=np.float64), 0.0, 1.0)
        if alpha_map.shape != canvas.shape[:2]:
            raise ValueError("output_alpha_map shape mismatch")
        rgba[:, :, 3] = np.rint(alpha_map * 255.0).astype(np.uint8)
    return rgba


def results_to_elements(results, unit_scale, img_center, primitives_config=None, output_alpha=None):
    preset_map = {}
    if primitives_config:
        for preset in primitives_config:
            shape = preset.get("shape")
            if shape:
                preset_map[shape] = preset

    unit_scale = float(unit_scale or 1.0)
    origin_x = float(img_center[0]) * unit_scale
    origin_y = -float(img_center[1]) * unit_scale
    elements = []

    for index, result in enumerate(results):
        cx = float(result["cx"]) * unit_scale
        cy = -float(result["cy"]) * unit_scale
        shape_key = result["type"]
        if shape_key == ShapeType.CIRCLE:
            preset = preset_map.get("circle", {})
            element_type = "ellipse"
            size = {
                "rx": round(float(result["rx"]) * unit_scale, 4),
                "ry": round(float(result["ry"]) * unit_scale, 4),
            }
        elif shape_key == ShapeType.TRIANGLE:
            preset = preset_map.get("triangle", {})
            element_type = "triangle"
            tri_width = float(result.get("width", result.get("size", 1.0))) * unit_scale
            tri_height = float(result.get("height", float(result.get("size", 1.0)) * math.sqrt(3.0) / 2.0)) * unit_scale
            size = {
                "width": round(tri_width, 4),
                "height": round(tri_height, 4),
            }
        else:
            preset = preset_map.get("rect", {})
            element_type = "rectangle"
            size = {
                "width": round(2.0 * float(result["hw"]) * unit_scale, 4),
                "height": round(2.0 * float(result["hh"]) * unit_scale, 4),
            }

        color_hex = result["color"] if isinstance(result.get("color"), str) else _bgr_to_hex(result.get("color", [255, 255, 255]))
        # Preserve fitted alpha layering and only scale it when the user requests a global opacity change.
        if output_alpha is not None:
            alpha = float(np.clip(float(result.get("alpha", 1.0)) * float(output_alpha), 0.0, 1.0))
        else:
            alpha = float(result.get("alpha", 1.0))
        image_asset_ref = int(
            preset.get("image_asset_ref")
            or preset.get("asset_id")
            or result.get("image_asset_ref")
            or 100002
        )
        packed_color = _pack_color(color_hex, alpha)

        element = {
            "id": index,
            "type": element_type,
            "center": {"x": round(cx, 4), "y": round(cy, 4)},
            "relative_position": {
                "x": round(cx - origin_x, 4),
                "y": round(cy - origin_y, 4),
            },
            "relative": {
                "x": round(cx - origin_x, 4),
                "y": round(cy - origin_y, 4),
            },
            "size": size,
            "rotation": {
                "x": 0.0,
                "y": 0.0,
                "z": round(-float(result.get("angle", 0.0)), 4),
            },
            "color": color_hex,
            "alpha": round(alpha, 4),
            "packed_color": packed_color,
            "image_asset_ref": image_asset_ref,
        }

        type_id = preset.get("type_id")
        if type_id is not None:
            element["type_id"] = type_id
            element["element_type_id"] = type_id
        if preset.get("rot_z") is not None:
            element["rotation"]["z"] = round(element["rotation"]["z"] + float(preset["rot_z"]), 4)
        if preset.get("rot_y_add") is not None:
            element["rotation"]["y"] = float(preset["rot_y_add"])

        elements.append(element)

    return elements
