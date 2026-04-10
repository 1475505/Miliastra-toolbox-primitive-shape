"""
填充拟合工具 - Fill Shaper
基于 fogleman/primitive 的 Hill Climbing 算法
用圆形和矩形缩放拟合图片区域
"""

import cv2
import numpy as np
import math
import time


# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

class FillConfig:
    def __init__(self, num_primitives=100, candidates=32, hill_climb_iter=64,
                 min_size=2, max_size=128, allowed_types=None, alpha_range=(0.3, 1.0)):
        self.num_primitives = num_primitives
        self.candidates = candidates
        self.hill_climb_iter = hill_climb_iter
        self.min_size = min_size
        self.max_size = max_size
        self.allowed_types = allowed_types  # None = all, or list of 'circle'/'rect'
        self.alpha_range = alpha_range


class ShapeType:
    CIRCLE = "circle"
    RECT = "rect"


# ═══════════════════════════════════════════════════════
#  图形基类与实现
# ═══════════════════════════════════════════════════════

class Shape:
    """图形基类"""
    shape_type = None

    def rasterize(self, w, h):
        """返回 (ys, xs, alphas) 扫描线表示"""
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def mutate(self, rng, w, h, cfg):
        """随机变异"""
        raise NotImplementedError


class Circle(Shape):
    shape_type = ShapeType.CIRCLE

    def __init__(self, cx, cy, r):
        self.cx = cx
        self.cy = cy
        self.r = r

    def rasterize(self, w, h):
        r = max(self.r, 0.5)
        y0 = max(int(self.cy - r), 0)
        y1 = min(int(self.cy + r) + 1, h)
        x0 = max(int(self.cx - r), 0)
        x1 = min(int(self.cx + r) + 1, w)

        if y0 >= y1 or x0 >= x1:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dist_sq = (xs - self.cx) ** 2 + (ys - self.cy) ** 2
        mask = dist_sq <= r * r
        if not mask.any():
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        ys_flat = ys[mask].ravel()
        xs_flat = xs[mask].ravel()
        # 抗锯齿：边缘像素用平滑过渡
        dist = np.sqrt(dist_sq[mask])
        # 1px 抗锯齿过渡
        alphas = np.clip(r - dist, 0.0, 1.0)
        return ys_flat, xs_flat, alphas

    def copy(self):
        return Circle(self.cx, self.cy, self.r)

    def mutate(self, rng, w, h, cfg):
        s = rng.choice([0.5, 1.0, 2.0])
        c = self.copy()
        if rng.random() < 0.4:
            c.cx = np.clip(c.cx + rng.uniform(-1, 1) * s * 10, 0, w)
        elif rng.random() < 0.7:
            c.cy = np.clip(c.cy + rng.uniform(-1, 1) * s * 10, 0, h)
        else:
            c.r = np.clip(c.r + rng.uniform(-1, 1) * s * 5, cfg.min_size, cfg.max_size)
        return c


class Rect(Shape):
    shape_type = ShapeType.RECT

    def __init__(self, cx, cy, hw, hh, angle=0):
        self.cx = cx
        self.cy = cy
        self.hw = hw  # 半宽
        self.hh = hh  # 半高
        self.angle = angle  # 度

    def rasterize(self, w, h):
        hw = max(self.hw, 0.5)
        hh = max(self.hh, 0.5)
        angle_rad = math.radians(self.angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # 计算旋转后的包围盒
        max_extent = math.sqrt(hw * hw + hh * hh)
        y0 = max(int(self.cy - max_extent), 0)
        y1 = min(int(self.cy + max_extent) + 1, h)
        x0 = max(int(self.cx - max_extent), 0)
        x1 = min(int(self.cx + max_extent) + 1, w)

        if y0 >= y1 or x0 >= x1:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        ys, xs = np.mgrid[y0:y1, x0:x1]
        # 变换到局部坐标
        dx = xs - self.cx
        dy = ys - self.cy
        local_x = dx * cos_a + dy * sin_a
        local_y = -dx * sin_a + dy * cos_a

        # 在矩形内部
        inside = (np.abs(local_x) <= hw) & (np.abs(local_y) <= hh)
        if not inside.any():
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64)

        ys_flat = ys[inside].ravel()
        xs_flat = xs[inside].ravel()
        # 抗锯齿：基于到边缘的距离
        dist_x = hw - np.abs(local_x[inside])
        dist_y = hh - np.abs(local_y[inside])
        dist = np.minimum(dist_x, dist_y)
        alphas = np.clip(dist, 0.0, 1.0)
        return ys_flat, xs_flat, alphas

    def copy(self):
        return Rect(self.cx, self.cy, self.hw, self.hh, self.angle)

    def mutate(self, rng, w, h, cfg):
        s = rng.choice([0.5, 1.0, 2.0])
        c = self.copy()
        r = rng.random()
        if r < 0.25:
            c.cx = np.clip(c.cx + rng.uniform(-1, 1) * s * 10, 0, w)
        elif r < 0.5:
            c.cy = np.clip(c.cy + rng.uniform(-1, 1) * s * 10, 0, h)
        elif r < 0.7:
            c.hw = np.clip(c.hw + rng.uniform(-1, 1) * s * 5, cfg.min_size, cfg.max_size)
        elif r < 0.9:
            c.hh = np.clip(c.hh + rng.uniform(-1, 1) * s * 5, cfg.min_size, cfg.max_size)
        else:
            c.angle = c.angle + rng.uniform(-1, 1) * 15 * s
        return c


# ═══════════════════════════════════════════════════════
#  核心算法
# ═══════════════════════════════════════════════════════

def compute_color(target, canvas, shape, alpha):
    """计算最优颜色，使 shape 区域的 canvas + color*alpha 最接近 target"""
    ys, xs, alphas = shape.rasterize(target.shape[0], target.shape[1])
    if len(ys) == 0:
        return np.array([128, 128, 128], dtype=np.float64)

    # 加权平均
    a = alpha * alphas
    total_a = np.sum(a)
    if total_a < 1e-10:
        return np.array([128, 128, 128], dtype=np.float64)

    # 目标颜色 = (target - canvas * (1-alpha)) / alpha 的加权平均
    diff = target[ys, xs] - canvas[ys, xs] * (1.0 - alpha * alphas[:, np.newaxis])
    # 简化: 直接取 target 区域的加权平均颜色
    weighted = target[ys, xs] * a[:, np.newaxis]
    color = np.sum(weighted, axis=0) / total_a

    return np.clip(color, 0, 255)


def compute_score(target, canvas, shape, color, alpha):
    """计算添加 shape 后的 MSE 改善量 (负值 = 改善)"""
    ys, xs, alphas = shape.rasterize(target.shape[0], target.shape[1])
    if len(ys) == 0:
        return 0.0

    a = alpha * alphas
    new_canvas = canvas[ys, xs] * (1 - a[:, np.newaxis]) + color * a[:, np.newaxis]
    diff_old = target[ys, xs].astype(np.float64) - canvas[ys, xs].astype(np.float64)
    diff_new = target[ys, xs].astype(np.float64) - new_canvas.astype(np.float64)

    # 只计算受影响像素的 MSE 改善
    old_mse = np.sum(diff_old ** 2)
    new_mse = np.sum(diff_new ** 2)
    return new_mse - old_mse  # 负值表示改善


def hill_climb(target, canvas, shape, color, alpha, cfg, rng):
    """Hill climbing 优化单个图形"""
    best_shape = shape
    best_color = color
    best_score = compute_score(target, canvas, best_shape, best_color, alpha)
    h, w = target.shape[:2]

    for _ in range(cfg.hill_climb_iter):
        candidate = best_shape.mutate(rng, w, h, cfg)
        cand_color = compute_color(target, canvas, candidate, alpha)
        cand_score = compute_score(target, canvas, candidate, cand_color, alpha)
        if cand_score < best_score:
            best_shape = candidate
            best_color = cand_color
            best_score = cand_score

    return best_shape, best_color, best_score


def random_shape(rng, w, h, cfg):
    """随机生成一个图形"""
    types = cfg.allowed_types or [ShapeType.CIRCLE, ShapeType.RECT]
    t = rng.choice(types)

    if t == ShapeType.CIRCLE:
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        r = rng.uniform(cfg.min_size, cfg.max_size)
        return Circle(cx, cy, r)
    else:
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        hw = rng.uniform(cfg.min_size, cfg.max_size)
        hh = rng.uniform(cfg.min_size, cfg.max_size)
        angle = rng.uniform(-45, 45)
        return Rect(cx, cy, hw, hh, angle)


def fit_primitives(target_img, config=None, progress_callback=None):
    """
    主入口：用 Hill Climbing 算法拟合图片

    Args:
        target_img: numpy 数组 (H, W, 3) 或 (H, W, 4)，BGR 或 BGRA
        config: FillConfig 或 dict
        progress_callback: 可选回调 fn(step, total, message)

    Returns:
        list of dict: 每个元素包含 type, cx, cy, size, angle, color, alpha
    """
    if config is None:
        config = FillConfig()
    elif isinstance(config, dict):
        config = FillConfig(**config)

    t0 = time.time()

    # 处理 alpha 通道
    if len(target_img.shape) == 3 and target_img.shape[2] == 4:
        alpha_ch = target_img[:, :, 3].astype(np.float64) / 255.0
        target = target_img[:, :, :3].astype(np.float64)
        # 将透明区域设为白色背景混合
        target = target * alpha_ch[:, :, np.newaxis] + 255.0 * (1 - alpha_ch[:, :, np.newaxis])
    else:
        if len(target_img.shape) == 2:
            target = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR).astype(np.float64)
        else:
            target = target_img[:, :, :3].astype(np.float64)

    h, w = target.shape[:2]
    rng = np.random.default_rng()

    # 初始化 canvas 为平均颜色
    avg_color = np.mean(target.reshape(-1, 3), axis=0)
    canvas = np.full((h, w, 3), avg_color, dtype=np.float64)

    # 提取前景 mask (如果有 alpha 通道)
    if len(target_img.shape) == 3 and target_img.shape[2] == 4:
        fg_mask = target_img[:, :, 3] > 127
    else:
        fg_mask = np.ones((h, w), dtype=bool)

    results = []
    n = config.num_primitives

    for i in range(n):
        if progress_callback:
            progress_callback(i, n, f'拟合图元 {i+1}/{n}')

        # 生成 K 个候选
        best_shape = None
        best_color = None
        best_alpha = 0.5
        best_score = float('inf')

        for _ in range(config.candidates):
            shape = random_shape(rng, w, h, config)
            # 尝试不同 alpha
            for alpha in [0.5, 0.75, 1.0]:
                color = compute_color(target, canvas, shape, alpha)
                score = compute_score(target, canvas, shape, color, alpha)
                if score < best_score:
                    best_shape = shape
                    best_color = color
                    best_alpha = alpha
                    best_score = score

        # Hill climb 优化
        best_shape, best_color, _ = hill_climb(
            target, canvas, best_shape, best_color, best_alpha, config, rng)

        # 应用到 canvas
        ys, xs, alphas = best_shape.rasterize(h, w)
        if len(ys) > 0:
            a = best_alpha * alphas
            for c in range(3):
                canvas[ys, xs, c] = canvas[ys, xs, c] * (1 - a) + best_color[c] * a

        # 记录结果
        if isinstance(best_shape, Circle):
            results.append({
                'type': ShapeType.CIRCLE,
                'cx': float(best_shape.cx),
                'cy': float(best_shape.cy),
                'r': float(best_shape.r),
                'color': [int(c) for c in np.clip(best_color, 0, 255)],
                'alpha': float(best_alpha),
            })
        elif isinstance(best_shape, Rect):
            results.append({
                'type': ShapeType.RECT,
                'cx': float(best_shape.cx),
                'cy': float(best_shape.cy),
                'hw': float(best_shape.hw),
                'hh': float(best_shape.hh),
                'angle': float(best_shape.angle),
                'color': [int(c) for c in np.clip(best_color, 0, 255)],
                'alpha': float(best_alpha),
            })

    elapsed = time.time() - t0
    if progress_callback:
        progress_callback(n, n, f'完成，耗时 {elapsed:.1f}s')

    return results


def render_results(target_img, results):
    """将拟合结果渲染为可视化图片 (BGR)"""
    if len(target_img.shape) == 3 and target_img.shape[2] == 4:
        canvas = cv2.cvtColor(target_img, cv2.COLOR_BGRA2BGR).astype(np.float64)
    elif len(target_img.shape) == 2:
        canvas = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR).astype(np.float64)
    else:
        canvas = target_img[:, :, :3].astype(np.float64)

    h, w = canvas.shape[:2]
    # 从平均颜色开始
    avg_color = np.mean(canvas.reshape(-1, 3), axis=0)
    canvas = np.full((h, w, 3), avg_color, dtype=np.float64)

    for r in results:
        if r['type'] == ShapeType.CIRCLE:
            shape = Circle(r['cx'], r['cy'], r['r'])
        else:
            shape = Rect(r['cx'], r['cy'], r['hw'], r['hh'], r.get('angle', 0))

        ys, xs, alphas = shape.rasterize(h, w)
        if len(ys) == 0:
            continue
        a = r['alpha'] * alphas
        color = np.array(r['color'], dtype=np.float64)
        for c in range(3):
            canvas[ys, xs, c] = canvas[ys, xs, c] * (1 - a) + color[c] * a

    return np.clip(canvas, 0, 255).astype(np.uint8)


def results_to_elements(results, prim_size, img_center, primitives_config=None):
    """
    将 fill_shaper 结果转换为与 outline 模式兼容的 elements 格式

    Args:
        results: fit_primitives 的返回值
        prim_size: 基准图元大小 (px)
        img_center: (cx, cy) 原点像素坐标
        primitives_config: 前端传来的 primitives 配置列表

    Returns:
        list of dict: 兼容 shaper_core 输出格式的元素列表
    """
    # 构建 type -> preset 映射
    preset_map = {}
    if primitives_config:
        for p in primitives_config:
            shape = p.get('shape')
            if shape:
                preset_map[shape] = p

    elements = []
    ox = img_center[0] / prim_size
    oy = -img_center[1] / prim_size

    for i, r in enumerate(results):
        # 像素坐标 → 归一化坐标
        cx_norm = float(r['cx']) / prim_size
        cy_norm = -float(r['cy']) / prim_size

        elem = {
            'id': i,
            'type': r['type'],
            'center': {'x': round(cx_norm, 4), 'y': round(cy_norm, 4)},
            'relative_position': {
                'x': round(cx_norm - ox, 4),
                'y': round(cy_norm - oy, 4),
            },
            'color': r['color'],
            'alpha': r.get('alpha', 1.0),
        }

        if r['type'] == ShapeType.CIRCLE:
            elem['size'] = {
                'width': round(2 * r['r'] / prim_size, 4),
                'height': round(2 * r['r'] / prim_size, 4),
            }
            elem['rotation'] = {'x': 0, 'y': 0, 'z': 0}

            # 注入预设信息
            preset = preset_map.get('circle', {})
            if preset:
                type_id = preset.get('type_id')
                if type_id:
                    elem['type_id'] = type_id
                    elem['element_type_id'] = type_id
                if 'rot_z' in preset and preset['rot_z'] is not None:
                    elem['rotation']['z'] = preset['rot_z']
                if 'rot_y_add' in preset and preset['rot_y_add'] is not None:
                    elem['rotation']['y'] = preset['rot_y_add']

        elif r['type'] == ShapeType.RECT:
            hw_norm = r['hw'] / prim_size
            hh_norm = r['hh'] / prim_size
            elem['size'] = {
                'width': round(2 * hw_norm, 4),
                'height': round(2 * hh_norm, 4),
            }
            rot_z = -r.get('angle', 0)
            elem['rotation'] = {'x': 0, 'y': 0, 'z': round(rot_z, 4)}

            # 矩形中心偏移修正 (与 outline 模式一致)
            rect_h = float(elem['size'].get('height', 0))
            theta = math.radians(rot_z)
            cx_norm += (rect_h * 0.5) * math.sin(theta)
            cy_norm += -(rect_h * 0.5) * math.cos(theta)
            elem['center']['x'] = round(cx_norm, 4)
            elem['center']['y'] = round(cy_norm, 4)
            elem['relative_position']['x'] = round(cx_norm - ox, 4)
            elem['relative_position']['y'] = round(cy_norm - oy, 4)

            # 注入预设信息
            preset = preset_map.get('rect', {})
            if preset:
                type_id = preset.get('type_id')
                if type_id:
                    elem['type_id'] = type_id
                    elem['element_type_id'] = type_id
                if 'rot_z' in preset and preset['rot_z'] is not None:
                    elem['rotation']['z'] += preset['rot_z']
                if 'rot_y_add' in preset and preset['rot_y_add'] is not None:
                    elem['rotation']['y'] = preset['rot_y_add']

        elements.append(elem)

    return elements