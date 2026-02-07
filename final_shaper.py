"""
轮廓描边工具 - Shaper V3
P0: 曲率分段 + 局部最优拟合 (二分查找 + 距离场约束)
P2: 后处理优化 (间隙检测填充 + 图元迭代扩展)
"""

import cv2
import numpy as np
import json
import os
import argparse
import sys
import math
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, scale, translate

try:
    from scipy.ndimage import gaussian_filter1d
except ImportError:
    # 简易 fallback：均匀滑动平均
    def gaussian_filter1d(arr, sigma):
        k = max(int(sigma * 3), 1)
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        return np.convolve(arr, kernel, mode='same')


# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

class ShapeType:
    ELLIPSE = "ellipse"
    RECTANGLE = "rectangle"


class FittingConfig:
    def __init__(self, min_size=4, max_size=40, spacing_ratio=0.9,
                 aspect_ratio_limit=2.5, precision=0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.spacing_ratio = spacing_ratio
        self.aspect_ratio_limit = aspect_ratio_limit
        self.precision = precision

        # 精度相关的派生参数
        self.effective_aspect_limit = aspect_ratio_limit + (1.0 - precision) * 3.0
        self.rect_bonus = (1.0 - precision) * 1.5
        self.min_radius_for_stretch = min_size * (0.5 + precision * 0.5)

        # P0: 曲率分析阈值
        self.curvature_sigma = 5.0           # 曲率高斯平滑 σ
        self.straight_thresh = 0.012         # < 此值视为直线
        self.tight_thresh = 0.06             # ≥ 此值视为急弯

        # P2: 后处理
        self.gap_sample_step = 1.5           # 覆盖检测采样步长 (pt index)
        self.gap_dilate_px = 2               # 覆盖膨胀像素
        self.gap_fill_iterations = 3         # 间隙填充最大迭代轮数
        self.expand_growth_factors = (1.3, 1.2, 1.1)  # 扩展尝试倍率


# ═══════════════════════════════════════════════════════
#  P0-A: 曲率分析
# ═══════════════════════════════════════════════════════

def compute_curvature(pts, sigma=5.0):
    """
    计算闭合轮廓上每点的无符号曲率 (带高斯平滑)。
    pts: (N, 2) 轮廓点坐标
    返回: (N,) 曲率数组 (0=直线, 大=急弯)

    方法: 用参数微分公式
        κ = |x'y'' - y'x''| / (x'² + y'²)^{3/2}
    首尾拼接处理闭合, 高斯平滑消除噪声。
    """
    N = len(pts)
    if N < 5:
        return np.zeros(N)

    pad = max(int(3 * sigma), 3)
    # 首尾拼接处理闭合曲线
    x = np.concatenate([pts[-pad:, 0], pts[:, 0], pts[:pad, 0]])
    y = np.concatenate([pts[-pad:, 1], pts[:, 1], pts[:pad, 1]])
    x = gaussian_filter1d(x, sigma)
    y = gaussian_filter1d(y, sigma)

    # 一阶、二阶导数
    dx = np.gradient(x);  dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)

    # 曲率公式
    num = np.abs(dx * ddy - dy * ddx)
    den = np.maximum((dx ** 2 + dy ** 2) ** 1.5, 1e-10)
    k = (num / den)[pad:pad + N]
    return gaussian_filter1d(k, sigma * 0.4)


def classify_curvature(k, cfg):
    """将曲率分为三档: 0=straight, 1=curved, 2=tight"""
    labels = np.ones(len(k), dtype=np.int32)
    labels[k < cfg.straight_thresh] = 0
    labels[k >= cfg.tight_thresh] = 2
    return labels


SEG_NAMES = {0: 'straight', 1: 'curved', 2: 'tight'}


def merge_short_runs(labels, min_run=5):
    """将过短的连续同类段合并到前一个段，避免碎片化"""
    N = len(labels)
    if N == 0:
        return labels
    out = labels.copy()
    runs = []
    start = 0
    for i in range(1, N):
        if out[i] != out[start]:
            runs.append((start, i, out[start]))
            start = i
    runs.append((start, N, out[start]))

    for idx in range(1, len(runs)):
        s, e, t = runs[idx]
        if e - s < min_run:
            prev_t = runs[idx - 1][2]
            out[s:e] = prev_t
    return out


# ═══════════════════════════════════════════════════════
#  P0-B: 几何工具
# ═══════════════════════════════════════════════════════

def build_arc_length_index(pts):
    """
    构建轮廓点的累计弧长数组。
    返回 (cum_arc, total_length)
    cum_arc[i] = 从起点到第 i 个点的弧长
    """
    d = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    return cum, cum[-1]


def cursor_to_index(cursor, cum_arc, total_arc):
    """弧长位置 → 最近轮廓点下标 (O(log n) 二分查找)"""
    c = cursor % max(total_arc, 1e-10)
    idx = np.searchsorted(cum_arc, c)
    return min(idx, len(cum_arc) - 2)


def tangent_normal_at(pts, idx, poly):
    """
    基于轮廓点数组计算 idx 处的切线和指向内部的法线。
    直接操作 numpy 数组, 比反复调用 Shapely interpolate 快得多。
    """
    N = len(pts)
    i0 = (idx - 1) % N
    i1 = (idx + 1) % N
    t = pts[i1] - pts[i0]
    n = np.linalg.norm(t)
    t = t / max(n, 1e-10)

    # 法线: 逆时针轮廓中, 左侧 (-ty, tx) 通常指向内部
    normal = np.array([-t[1], t[0]])

    # 验证法线方向 (若不指向内部则翻转)
    px, py = pts[idx]
    test_pt = Point(px + normal[0] * 2, py + normal[1] * 2)
    if not poly.contains(test_pt):
        normal = -normal

    return t, normal


def max_inscribed_radius(px, py, normal, dist_map, rmin, rmax):
    """
    二分查找最大内切圆半径。
    沿内法线方向寻找圆心位置, 使 dist_map[center] ≥ r。
    相比线性扫描, 20 次迭代即可达 ~0.3px 精度。
    """
    h, w = dist_map.shape
    lo, hi = rmin, rmax
    best_r = rmin
    best_c = (px + normal[0] * rmin, py + normal[1] * rmin)

    for _ in range(20):
        mid = (lo + hi) * 0.5
        cx = px + normal[0] * mid
        cy = py + normal[1] * mid
        ix, iy = int(cx), int(cy)
        if 0 <= ix < w and 0 <= iy < h and dist_map[iy, ix] >= mid * 0.95:
            best_r = mid
            best_c = (cx, cy)
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.3:
            break

    return best_r, best_c


def make_ellipse_poly(cx, cy, rx, ry, angle_deg):
    """构造旋转椭圆的 Shapely Polygon"""
    c = Point(0, 0).buffer(1.0, resolution=32)
    c = scale(c, rx, ry)
    c = rotate(c, angle_deg, origin=(0, 0))
    return translate(c, cx, cy)


def make_rect_poly(cx, cy, w, h, angle_deg):
    """构造旋转矩形的 Shapely Polygon"""
    r = box(-w / 2, -h / 2, w / 2, h / 2)
    r = rotate(r, angle_deg, origin=(0, 0))
    return translate(r, cx, cy)


# ═══════════════════════════════════════════════════════
#  P0-C: 单点最优图元选择
# ═══════════════════════════════════════════════════════

def best_primitive_at(px, py, tangent, normal, dist_map, poly, cfg,
                      kappa, seg_label):
    """
    在轮廓点 (px, py) 处选择得分最高的图元。
    得分 = 图元面积 × 类型加成 (鼓励大图元覆盖更多轮廓)。

    流程:
    1. 二分查找最大内切圆半径 (距离场约束)
    2. 基础候选: 该半径的圆
    3. 根据曲率分段类型, 尝试沿切线拉伸为椭圆/矩形
    4. 用 Shapely 检验包含性 (不超出轮廓)
    5. 返回得分最高的候选
    """
    rmin = cfg.min_size / 2
    rmax = cfg.max_size / 2
    best_r, center = max_inscribed_radius(px, py, normal, dist_map, rmin, rmax)
    ang = math.degrees(math.atan2(tangent[1], tangent[0]))

    # ---- 基础候选: 圆 ----
    cpoly = Point(center).buffer(best_r)
    candidates = [{
        'type': ShapeType.ELLIPSE, 'center': center,
        'size': (best_r, best_r), 'rot': ang,
        'score': math.pi * best_r ** 2,
        'poly': cpoly, 'tr': best_r,
    }]

    # ---- 拉伸候选 (椭圆 / 矩形) ----
    if best_r >= cfg.min_radius_for_stretch:
        seg = SEG_NAMES.get(seg_label, 'curved')
        kf = max(0.0, 1.0 - kappa * 30)   # 曲率因子: 直线→1, 急弯→0

        if seg == 'straight':
            # 直线段: 激进拉伸, 偏好矩形
            max_stretch = cfg.effective_aspect_limit + kf * 2
            rect_bonus = cfg.rect_bonus + 1.5
        elif seg == 'curved':
            # 弯曲段: 中等拉伸, 偏好椭圆
            max_stretch = cfg.effective_aspect_limit + kf * 0.5
            rect_bonus = cfg.rect_bonus * 0.5
        else:  # tight
            # 急弯: 保守拉伸
            max_stretch = min(cfg.effective_aspect_limit, 2.0)
            rect_bonus = 0.0

        contain_t = 0.85 + cfg.precision * 0.15

        for asp in (1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, max_stretch):
            if asp > max_stretch:
                continue
            mr = best_r * asp

            # -- 椭圆候选 --
            try:
                ep = make_ellipse_poly(center[0], center[1], mr, best_r, ang)
                if ep.is_valid:
                    ia = poly.intersection(ep).area
                    ea = ep.area
                    if ea > 0 and ia >= ea * contain_t:
                        candidates.append({
                            'type': ShapeType.ELLIPSE, 'center': center,
                            'size': (mr, best_r), 'rot': ang,
                            'score': math.pi * mr * best_r,
                            'poly': ep, 'tr': mr,
                        })
            except Exception:
                pass

            # -- 矩形候选 --
            try:
                rw, rh = mr * 2, best_r * 2
                rp = make_rect_poly(center[0], center[1], rw, rh, ang)
                if rp.is_valid:
                    ia = poly.intersection(rp).area
                    ra = rp.area
                    if ra > 0 and ia >= ra * contain_t:
                        candidates.append({
                            'type': ShapeType.RECTANGLE, 'center': center,
                            'size': (rw, rh), 'rot': ang,
                            'score': rw * rh * (1.0 + rect_bonus),
                            'poly': rp, 'tr': rw / 2,
                        })
            except Exception:
                pass

    return max(candidates, key=lambda c: c['score'])


# ═══════════════════════════════════════════════════════
#  P0-D: 主铺设流程
# ═══════════════════════════════════════════════════════

def fit_beads(cidx, contour, mask, dist_map, cfg, img_center):
    """
    P0 核心: 曲率分段引导的串珠铺设。

    改进点:
    - 用原始轮廓点直接计算曲率 (避免大量 Shapely interpolate 调用)
    - 累计弧长索引 + 二分查找 O(log n) 定位
    - 曲率分段 (straight/curved/tight) 指导图元类型选择
    - 二分查找最大内切半径 (替代线性扫描)
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    N = len(pts)
    if N < 3:
        return []

    poly = Polygon(pts).simplify(1.0, preserve_topology=True)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if not poly.is_valid or poly.area < cfg.min_size ** 2:
        return []

    # ---- 曲率 & 分段 ----
    kappa = compute_curvature(pts, cfg.curvature_sigma)
    labels = classify_curvature(kappa, cfg)
    labels = merge_short_runs(labels, min_run=5)

    # ---- 弧长索引 ----
    cum_arc, total_arc = build_arc_length_index(pts)

    # ---- 统计分段比例 (调试用) ----
    n_straight = np.sum(labels == 0)
    n_curved = np.sum(labels == 1)
    n_tight = np.sum(labels == 2)

    # ---- 沿轮廓铺设 ----
    elements = []
    cursor = 0.0
    max_iter = int(total_arc / (cfg.min_size * 0.3)) + 300
    it = 0

    while cursor < total_arc and it < max_iter:
        it += 1
        idx = cursor_to_index(cursor, cum_arc, total_arc)
        px, py = pts[idx]
        tangent, normal = tangent_normal_at(pts, idx, poly)
        k = kappa[idx]
        lb = labels[idx]

        best = best_primitive_at(px, py, tangent, normal,
                                 dist_map, poly, cfg, k, lb)

        cx, cy = best['center']
        elem = {
            'id': f'{cidx}_{len(elements)}',
            'type': best['type'],
            'center': {'x': round(cx, 2), 'y': round(cy, 2)},
            'relative_position': {
                'x': round(cx - img_center[0], 2),
                'y': round(cy - img_center[1], 2),
            },
            'rotation': round(best['rot'], 2),
            '_poly': best['poly'],
            '_tr': best['tr'],
            '_cursor': cursor,
        }
        if best['type'] == ShapeType.ELLIPSE:
            rx, ry = best['size']
            elem['size'] = {'rx': round(rx, 2), 'ry': round(ry, 2)}
        else:
            w, h = best['size']
            elem['size'] = {'width': round(w, 2), 'height': round(h, 2)}

        elements.append(elem)

        # 步进: 当前图元沿切线方向的跨度
        step = best['tr'] * 2 * cfg.spacing_ratio
        cursor += max(step, cfg.min_size * 0.5)

    return elements


# ═══════════════════════════════════════════════════════
#  P2-A: 覆盖率检测 (渲染法, O(n+m))
# ═══════════════════════════════════════════════════════

def render_coverage_mask(elements, img_shape):
    """
    将所有已放置图元渲染到二值图 (填充), 用于快速覆盖判断。
    比逐点调用 Shapely contains/distance 快数个数量级。
    """
    h, w = img_shape[:2]
    cov = np.zeros((h, w), dtype=np.uint8)

    for elem in elements:
        cx = int(elem['center']['x'])
        cy = int(elem['center']['y'])
        ang = elem['rotation']

        if elem['type'] == ShapeType.ELLIPSE:
            s = elem['size']
            axes = (max(int(s['rx']), 1), max(int(s['ry']), 1))
            cv2.ellipse(cov, (cx, cy), axes, ang, 0, 360, 255, -1)
        elif elem['type'] == ShapeType.RECTANGLE:
            rw, rh = elem['size']['width'], elem['size']['height']
            rect = ((cx, cy), (max(rw, 1), max(rh, 1)), ang)
            bpts = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(cov, [bpts], 0, 255, -1)

    return cov


def detect_gaps(elements, contour_pts, cum_arc, total_arc,
                img_shape, cfg):
    """
    沿轮廓密集采样, 用渲染覆盖掩码检测未覆盖间隙。
    返回: list of (start_arc, end_arc, gap_length)
    """
    cov_mask = render_coverage_mask(elements, img_shape)

    # 膨胀: 容许图元边缘恰好贴合轮廓
    if cfg.gap_dilate_px > 0:
        k = np.ones((cfg.gap_dilate_px * 2 + 1,) * 2, dtype=np.uint8)
        cov_mask = cv2.dilate(cov_mask, k, iterations=1)

    h, w = img_shape[:2]
    N = len(contour_pts)
    step = max(int(cfg.gap_sample_step), 1)

    # 每隔 step 个轮廓点采样
    sample_idx = np.arange(0, N, step)
    covered = np.zeros(len(sample_idx), dtype=bool)

    for si, idx in enumerate(sample_idx):
        x, y = int(contour_pts[idx, 0]), int(contour_pts[idx, 1])
        if 0 <= x < w and 0 <= y < h:
            covered[si] = cov_mask[y, x] > 0

    # 找连续未覆盖运行段
    min_gap_samples = max(int(cfg.min_size * 0.5 / max(cfg.gap_sample_step, 0.5)), 2)
    gaps = []
    i = 0
    n = len(covered)
    while i < n:
        if not covered[i]:
            j = i
            while j < n and not covered[j]:
                j += 1
            if j - i >= min_gap_samples:
                s_arc = cum_arc[sample_idx[i]]
                e_arc = cum_arc[sample_idx[min(j - 1, n - 1)]]
                gaps.append((s_arc, e_arc, e_arc - s_arc))
            i = j
        else:
            i += 1

    return gaps


# ═══════════════════════════════════════════════════════
#  P2-B: 间隙填充
# ═══════════════════════════════════════════════════════

def fill_gaps(elements, cidx, contour_pts, cum_arc, total_arc,
              poly, dist_map, img_shape, cfg, img_center):
    """
    多轮迭代: 检测间隙 → 在间隙中均匀补入图元 → 重新检测。
    """
    current = list(elements)

    for iteration in range(cfg.gap_fill_iterations):
        gaps = detect_gaps(current, contour_pts, cum_arc, total_arc,
                           img_shape, cfg)
        if not gaps:
            break

        added = 0
        for g_start, g_end, g_len in gaps:
            if g_len < cfg.min_size * 0.4:
                continue
            n_fillers = max(1, int(g_len / cfg.min_size))
            for fi in range(n_fillers):
                frac = (fi + 0.5) / n_fillers
                fill_arc = g_start + frac * g_len
                idx = cursor_to_index(fill_arc, cum_arc, total_arc)
                px, py = contour_pts[idx]
                tangent, normal = tangent_normal_at(contour_pts, idx, poly)

                best = best_primitive_at(
                    px, py, tangent, normal,
                    dist_map, poly, cfg,
                    kappa=0.03, seg_label=1,
                )
                cx, cy = best['center']
                elem = {
                    'id': f'{cidx}_gf{iteration}_{added}',
                    'type': best['type'],
                    'center': {'x': round(cx, 2), 'y': round(cy, 2)},
                    'relative_position': {
                        'x': round(cx - img_center[0], 2),
                        'y': round(cy - img_center[1], 2),
                    },
                    'rotation': round(best['rot'], 2),
                    '_poly': best['poly'],
                    '_tr': best['tr'],
                    '_cursor': fill_arc,
                }
                if best['type'] == ShapeType.ELLIPSE:
                    rx, ry = best['size']
                    elem['size'] = {'rx': round(rx, 2), 'ry': round(ry, 2)}
                else:
                    w, h = best['size']
                    elem['size'] = {'width': round(w, 2), 'height': round(h, 2)}

                current.append(elem)
                added += 1

        if added == 0:
            break

    return current


# ═══════════════════════════════════════════════════════
#  P2-C: 图元扩展
# ═══════════════════════════════════════════════════════

def expand_elements(elements, poly, cfg):
    """
    尝试沿切线方向扩展每个图元的长轴/宽度。
    仅在扩展后仍被原始轮廓包含时接受, 贪心策略。
    """
    contain_t = 0.85 + cfg.precision * 0.15

    for elem in elements:
        if elem.get('_poly') is None:
            continue
        cx = elem['center']['x']
        cy = elem['center']['y']
        ang = elem['rotation']

        if elem['type'] == ShapeType.ELLIPSE:
            rx = elem['size']['rx']
            ry = elem['size']['ry']
            for g in cfg.expand_growth_factors:
                new_rx = rx * g
                ep = make_ellipse_poly(cx, cy, new_rx, ry, ang)
                try:
                    ia = poly.intersection(ep).area
                    ea = ep.area
                    if ea > 0 and ia >= ea * contain_t:
                        elem['size']['rx'] = round(new_rx, 2)
                        elem['_poly'] = ep
                        elem['_tr'] = new_rx
                        break
                except Exception:
                    pass

        elif elem['type'] == ShapeType.RECTANGLE:
            rw = elem['size']['width']
            rh = elem['size']['height']
            for g in cfg.expand_growth_factors:
                new_w = rw * g
                rp = make_rect_poly(cx, cy, new_w, rh, ang)
                try:
                    ia = poly.intersection(rp).area
                    ra = rp.area
                    if ra > 0 and ia >= ra * contain_t:
                        elem['size']['width'] = round(new_w, 2)
                        elem['_poly'] = rp
                        elem['_tr'] = new_w / 2
                        break
                except Exception:
                    pass

    return elements


# ═══════════════════════════════════════════════════════
#  渲染输出
# ═══════════════════════════════════════════════════════

def draw_results(image_shape, elements, out_shapes, out_overlay, input_img=None):
    if len(image_shape) == 2:
        h, w = image_shape
    else:
        h, w = image_shape[:2]

    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    if input_img is not None:
        if len(input_img.shape) == 2:
            overlay = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
        elif input_img.shape[2] == 4:
            overlay = cv2.cvtColor(input_img, cv2.COLOR_BGRA2BGR)
        else:
            overlay = input_img.copy()
    else:
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for elem in elements:
        cx = int(elem['center']['x'])
        cy = int(elem['center']['y'])
        ang = elem['rotation']

        fill_c = (255, 100, 50, 255)
        border_c = (255, 255, 255, 255)
        ov_c = (0, 0, 255)

        if elem['type'] == ShapeType.ELLIPSE:
            s = elem['size']
            axes = (max(int(s['rx']), 1), max(int(s['ry']), 1))
            cv2.ellipse(canvas, (cx, cy), axes, ang, 0, 360, fill_c, -1)
            cv2.ellipse(canvas, (cx, cy), axes, ang, 0, 360, border_c, 1)
            cv2.ellipse(overlay, (cx, cy), axes, ang, 0, 360, ov_c, 1)

        elif elem['type'] == ShapeType.RECTANGLE:
            rw, rh = elem['size']['width'], elem['size']['height']
            rect = ((cx, cy), (max(rw, 1), max(rh, 1)), ang)
            bpts = np.intp(cv2.boxPoints(rect))
            cv2.drawContours(canvas, [bpts], 0, fill_c, -1)
            cv2.drawContours(canvas, [bpts], 0, border_c, 1)
            cv2.drawContours(overlay, [bpts], 0, ov_c, 1)

    cv2.imwrite(out_shapes, canvas)
    cv2.imwrite(out_overlay, overlay)


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Shaper V3 – 曲率分段 + 后处理优化')
    ap.add_argument('image_path', nargs='?', default='genshin.png',
                    help='输入图片路径')
    ap.add_argument('--min_size', type=int, default=6,
                    help='最小图元尺寸 (px)')
    ap.add_argument('--max_size', type=int, default=30,
                    help='最大图元尺寸 (px)')
    ap.add_argument('--spacing', type=float, default=0.9,
                    help='间距系数 (0.9 = 10%% 重叠)')
    ap.add_argument('-p', '--precision', type=float, default=0.3,
                    help='精度 0.0-1.0 (低=矩形为主, 高=圆形为主)')
    args = ap.parse_args()

    # ── 加载图片 ──
    if not os.path.exists(args.image_path):
        imgs = [f for f in os.listdir('.')
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        if imgs:
            print(f'未找到 {args.image_path}, 使用: {imgs[0]}')
            args.image_path = imgs[0]
        else:
            print(f'Error: {args.image_path} 未找到')
            sys.exit(1)

    print(f'处理: {args.image_path}')
    img = cv2.imread(args.image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print('图片加载失败')
        sys.exit(1)

    h, w = img.shape[:2]
    img_center = (w / 2.0, h / 2.0)

    # ── Mask 提取 ──
    if len(img.shape) == 3 and img.shape[2] == 4:
        _, mask = cv2.threshold(img[:, :, 3], 127, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cfg = FittingConfig(
        min_size=args.min_size, max_size=args.max_size,
        spacing_ratio=args.spacing,
        precision=max(0.0, min(1.0, args.precision)),
    )
    print(f'精度={cfg.precision:.1f}  rect_bonus={cfg.rect_bonus:.2f}  '
          f'aspect_limit={cfg.effective_aspect_limit:.1f}')

    # ── 处理每个轮廓 ──
    all_elements = []
    img_area = w * h
    min_contour_area = max(100, img_area * 0.00005)

    for ci, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        # 过滤满幅边框
        if cw > w * 0.95 and ch > h * 0.95:
            continue
        # 过滤贴边伪影
        margin = 5
        if (x + cw) < margin or x > (w - margin) \
                or (y + ch) < margin or y > (h - margin):
            continue

        # ── P0: 曲率分段铺设 ──
        elems = fit_beads(ci, contour, mask, dist_map, cfg, img_center)

        # ── P2: 后处理 ──
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            poly = Polygon(pts).simplify(1.0, preserve_topology=True)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area >= cfg.min_size ** 2:
                cum_arc, total_arc = build_arc_length_index(pts)

                # P2-B: 间隙填充 (多轮迭代)
                elems = fill_gaps(
                    elems, ci, pts, cum_arc, total_arc,
                    poly, dist_map, img.shape, cfg, img_center)

                # P2-C: 图元扩展
                elems = expand_elements(elems, poly, cfg)

        all_elements.extend(elems)

    # ── 输出 ──
    output = {
        'image_center': {'x': img_center[0], 'y': img_center[1]},
        'image_size': {'width': w, 'height': h},
        'config': {
            'min_size': cfg.min_size, 'max_size': cfg.max_size,
            'spacing': cfg.spacing_ratio, 'precision': cfg.precision,
        },
        'elements_count': len(all_elements),
        'elements': [
            {k: v for k, v in e.items() if not k.startswith('_')}
            for e in all_elements
        ],
    }

    with open('result_C_data.json', 'w') as f:
        json.dump(output, f, indent=2)

    draw_results(img.shape, output['elements'],
                 'result_A_shapes_only.png', 'result_B_overlay.png', img)

    print(f'完成! 共 {len(all_elements)} 个图元')
    print('输出: result_A_shapes_only.png, result_B_overlay.png, result_C_data.json')


if __name__ == '__main__':
    main()
