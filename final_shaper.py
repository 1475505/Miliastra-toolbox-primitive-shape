"""
轮廓描边工具 - Shaper V6
曲率分段 + 路径行走 + 距离场约束 + 多策略间隙检测
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
                 aspect_ratio_limit=2.5, precision=0.5, allowed_types=None):
        self.min_size = min_size
        self.max_size = max_size
        self.spacing_ratio = spacing_ratio
        self.aspect_ratio_limit = aspect_ratio_limit
        self.precision = precision
        self.allowed_types = allowed_types  # list of allowed ShapeType or None

        # 精度相关的派生参数
        self.effective_aspect_limit = aspect_ratio_limit + (1.0 - precision) * 3.0
        self.rect_bonus = (1.0 - precision) * 1.5
        self.min_radius_for_stretch = min_size * (0.5 + precision * 0.5)

        # P0: 曲率分析阈值
        self.curvature_sigma = 5.0           # 曲率高斯平滑 σ
        self.straight_thresh = 0.012         # < 此值视为直线
        self.tight_thresh = 0.06             # ≥ 此值视为急弯

        # P2: 后处理
        self.gap_sample_step = 1.0           # 覆盖检测采样步长 (更密)
        self.gap_dilate_px = 1               # 覆盖膨胀像素 (更精确)
        self.gap_fill_iterations = 5         # 间隙填充最大迭代轮数
        self.expand_growth_factors = (1.15, 1.1, 1.05)  # 扩展尝试倍率 (收紧)

        # V5: 路径行走优化
        self.step_tight_factor = 0.55        # 急弯步长缩放
        self.step_straight_factor = 1.15     # 直线步长缩放


# ═══════════════════════════════════════════════════════
#  P3: 智能前景提取
# ═══════════════════════════════════════════════════════

def extract_mask(img):
    """
    智能提取前景 mask (白色=前景, 黑色=背景)。

    策略优先级:
    1. RGBA 图片 → 用 alpha 通道 (阈值 127)
    2. 无 alpha → 边框采样推断背景色 → 计算每像素到背景色的颜色距离
       → Otsu 自适应分割
    3. Fallback: 灰度 Otsu 反转

    返回: 二值 mask (uint8, 0/255)
    """
    h, w = img.shape[:2]

    # ── 策略 1: Alpha 通道 ──
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
        print(f'  [mask] alpha 通道, 前景={np.sum(mask>0)/(h*w)*100:.1f}%')
        return mask

    # ── 准备灰度和 BGR ──
    if len(img.shape) == 2:
        gray = img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bgr = img[:, :, :3] if img.shape[2] >= 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── 策略 2: 边框采样 + 颜色距离 + Otsu ──
    # 采样图像四周边缘像素作为背景样本
    margin = max(2, min(h, w) // 50)  # 自适应采样宽度
    border_pixels = np.concatenate([
        bgr[:margin, :].reshape(-1, 3),        # 顶部
        bgr[-margin:, :].reshape(-1, 3),       # 底部
        bgr[margin:-margin, :margin].reshape(-1, 3),   # 左侧
        bgr[margin:-margin, -margin:].reshape(-1, 3),  # 右侧
    ])

    # 背景色 = 边框像素的中值 (比均值更鲁棒)
    bg_color = np.median(border_pixels, axis=0).astype(np.float64)
    print(f'  [mask] 边框采样背景色 BGR=({bg_color[0]:.0f},{bg_color[1]:.0f},{bg_color[2]:.0f})')

    # 每像素到背景色的欧氏距离
    diff = bgr.astype(np.float64) - bg_color
    color_dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # 归一化到 0-255 范围
    max_dist = max(color_dist.max(), 1.0)
    dist_u8 = np.clip(color_dist / max_dist * 255, 0, 255).astype(np.uint8)

    # Otsu 自动阈值 (在颜色距离图上)
    otsu_t, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_ratio = np.sum(mask > 0) / (h * w)
    print(f'  [mask] 颜色距离 Otsu 阈值={otsu_t:.0f}, 前景={fg_ratio*100:.1f}%')

    # 如果 Otsu 得到的前景比例合理 (1%~95%), 使用该结果
    if 0.01 < fg_ratio < 0.95:
        return mask

    # ── 策略 3: 灰度 Otsu (fallback) ──
    otsu_t2, mask2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg_ratio2 = np.sum(mask2 > 0) / (h * w)
    print(f'  [mask] 灰度 Otsu fallback 阈值={otsu_t2:.0f}, 前景={fg_ratio2*100:.1f}%')

    if 0.01 < fg_ratio2 < 0.95:
        return mask2

    # ── 最后手段: 固定阈值 ──
    _, mask3 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    print(f'  [mask] 固定阈值 fallback')
    return mask3


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


def tangent_normal_at(pts, idx, dist_map):
    """
    基于距离场的自适应法线方向检测。
    用距离场值比较两侧, 选择更深入内部的方向。
    """
    N = len(pts)
    i0 = (idx - 1) % N
    i1 = (idx + 1) % N
    t = pts[i1] - pts[i0]
    n = np.linalg.norm(t)
    t = t / max(n, 1e-10)

    # 法线: 逆时针轮廓中, 左侧 (-ty, tx) 通常指向内部
    normal = np.array([-t[1], t[0]])

    # 验证法线方向 — 基于距离场比较两侧, 选更深入内部的方向
    px, py = pts[idx]
    h, w = dist_map.shape
    # 自适应探测距离: 取当前点距离场值的 40%, 至少 2px
    ix0, iy0 = int(px), int(py)
    if 0 <= ix0 < w and 0 <= iy0 < h:
        probe = max(2.0, float(dist_map[iy0, ix0]) * 0.4)
    else:
        probe = 2.0
    # 比较法线两侧的距离场值, 选值更大(更深入内部)的方向
    t1x, t1y = px + normal[0] * probe, py + normal[1] * probe
    t2x, t2y = px - normal[0] * probe, py - normal[1] * probe
    v1, v2 = 0.0, 0.0
    ix1, iy1 = int(t1x), int(t1y)
    ix2, iy2 = int(t2x), int(t2y)
    if 0 <= ix1 < w and 0 <= iy1 < h:
        v1 = float(dist_map[iy1, ix1])
    if 0 <= ix2 < w and 0 <= iy2 < h:
        v2 = float(dist_map[iy2, ix2])
    if v2 > v1:
        normal = -normal

    return t, normal


def max_inscribed_radius(px, py, normal, dist_map, rmin, rmax):
    """
    二分查找最大内切圆半径。
    沿内法线方向寻找圆心位置, 使 dist_map[center] ≥ r。
    相比线性扫描, 20 次迭代即可达 ~0.3px 精度。
    """
    h, w = dist_map.shape
    # V5: 允许搜索到比 rmin 更小的半径, 解决窄轮廓处溢出
    actual_lo = max(1.5, rmin * 0.3)
    lo, hi = actual_lo, rmax
    best_r = actual_lo
    best_c = (px + normal[0] * actual_lo, py + normal[1] * actual_lo)

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

    # 最终 clamp — 确保半径不超过中心点到前景边界的实际距离
    if best_c is not None:
        cix, ciy = int(best_c[0]), int(best_c[1])
        if 0 <= cix < w and 0 <= ciy < h:
            best_r = min(best_r, float(dist_map[ciy, cix]))
    best_r = max(best_r, 1.5)  # 绝对最小半径

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


def dist_map_containment_check(poly_shape, dist_map, sample_n=32):
    """
    V5: 加密边界采样 (32点) + 距离场阈值 (≥0.5px)。
    在图元边界上均匀采样, 检查每个点是否在前景内部。
    """
    h, w = dist_map.shape
    try:
        boundary = poly_shape.boundary
        total_len = boundary.length
        if total_len < 1:
            return 1.0
        inside = 0
        for i in range(sample_n):
            pt = boundary.interpolate(i / sample_n * total_len)
            ix, iy = int(pt.x), int(pt.y)
            if 0 <= ix < w and 0 <= iy < h and dist_map[iy, ix] >= 0.5:
                inside += 1
        return inside / sample_n
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════
#  P0-C: 单点最优图元选择
# ═══════════════════════════════════════════════════════

def best_primitive_at(px, py, tangent, normal, dist_map, poly, cfg,
                      kappa, seg_label):
    """
    在轮廓点 (px, py) 处选择得分最高的图元。
    得分 = 图元面积 × 类型加成 (鼓励大图元覆盖更多轮廓)。
    """
    rmin = cfg.min_size / 2
    rmax = cfg.max_size / 2
    best_r, center = max_inscribed_radius(px, py, normal, dist_map, rmin, rmax)
    ang = math.degrees(math.atan2(tangent[1], tangent[0]))

    # 包含性阈值: 精度越高要求越严格 (0.88 ~ 0.98)
    contain_t = 0.88 + cfg.precision * 0.10

    candidates = []

    # ---- 基础候选: 圆 (归类为 ELLIPSE) ----
    if cfg.allowed_types is None or ShapeType.ELLIPSE in cfg.allowed_types:
        cpoly = Point(center).buffer(best_r)
        candidates.append({
            'type': ShapeType.ELLIPSE, 'center': center,
            'size': (best_r, best_r), 'rot': ang,
            'score': math.pi * best_r ** 2,
            'poly': cpoly, 'tr': best_r, 'sr': best_r,
        })

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

        for asp in (1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0, max_stretch):
            if asp > max_stretch:
                continue
            mr = best_r * asp

            # -- 椭圆候选 --
            if cfg.allowed_types is None or ShapeType.ELLIPSE in cfg.allowed_types:
                try:
                    ep = make_ellipse_poly(center[0], center[1], mr, best_r, ang)
                    if ep.is_valid:
                        ia = poly.intersection(ep).area
                        ea = ep.area
                        # Shapely 包含性检查 + 距离场边界验证
                        if ea > 0 and ia >= ea * contain_t:
                            dm_ratio = dist_map_containment_check(ep, dist_map)
                            # 拉伸图元dm_ratio要求适当放宽
                            dm_thresh = max(0.70, 0.88 - asp * 0.03)
                            if dm_ratio >= dm_thresh:
                                # V5: 多维评分 — compactness 惩罚随精度缩放
                                containment = ia / ea
                                compactness = best_r / mr
                                # 低精度→几乎不惩罚拉伸; 高精度→惩罚拉伸
                                cp_weight = cfg.precision  # 0.0~1.0
                                cp_factor = 1.0 - cp_weight * (1.0 - compactness) * 0.5
                                score = (math.pi * mr * best_r
                                         * containment ** 2
                                         * cp_factor)
                                candidates.append({
                                    'type': ShapeType.ELLIPSE, 'center': center,
                                    'size': (mr, best_r), 'rot': ang,
                                    'score': score,
                                    'poly': ep, 'tr': mr, 'sr': best_r,
                                })
                except Exception:
                    pass

            # -- 矩形候选 --
            if cfg.allowed_types is None or ShapeType.RECTANGLE in cfg.allowed_types:
                # 矩形的角天然伸出弧形轮廓, 用更宽松的独立包含阈值
                rect_contain_t = max(0.86, contain_t - (1.0 - cfg.precision) * 0.06)
                try:
                    rw, rh = mr * 2, best_r * 2
                    rp = make_rect_poly(center[0], center[1], rw, rh, ang)
                    if rp.is_valid:
                        ia = poly.intersection(rp).area
                        ra = rp.area
                        # Shapely 包含性检查 (矩形独立阈值) + 距离场边界验证
                        if ra > 0 and ia >= ra * rect_contain_t:
                            dm_ratio = dist_map_containment_check(rp, dist_map)
                            # 矩形dm_ratio要求比椭圆适度放宽
                            dm_thresh = max(0.65, 0.85 - asp * 0.04)
                            if dm_ratio >= dm_thresh:
                                # V5: 矩形用 containment³ 抑制溢出
                                containment = ia / ra
                                compactness = rh / rw
                                cp_weight = cfg.precision
                                cp_factor = 1.0 - cp_weight * (1.0 - compactness) * 0.5
                                score = (rw * rh * (1.0 + rect_bonus)
                                         * containment ** 3
                                         * cp_factor)
                                candidates.append({
                                    'type': ShapeType.RECTANGLE, 'center': center,
                                    'size': (rw, rh), 'rot': ang,
                                    'score': score,
                                    'poly': rp, 'tr': rw / 2, 'sr': rh / 2,
                                })
                except Exception:
                    pass

    if not candidates:
        return None
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

    # ---- 沿轮廓铺设 ----
    elements = []
    cursor = 0.0
    max_iter = int(total_arc / (cfg.min_size * 0.3)) + 300
    it = 0

    while cursor < total_arc and it < max_iter:
        it += 1
        idx = cursor_to_index(cursor, cum_arc, total_arc)
        px, py = pts[idx]
        tangent, normal = tangent_normal_at(pts, idx, dist_map)
        k = kappa[idx]
        lb = labels[idx]

        best = best_primitive_at(px, py, tangent, normal,
                                 dist_map, poly, cfg, k, lb)

        if best is None:
            cursor += cfg.min_size * 0.5
            continue

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

        # 曲率自适应步长 — 直线段用长轴步进, 弯曲段用短轴限制防跳步
        sr = best.get('sr', best['tr'])
        tr = best['tr']
        lookahead_end = min(idx + 20, N - 1)
        k_ahead = 0.03  # 默认: curved
        if lookahead_end > idx + 2:
            k_ahead = float(np.mean(kappa[idx:lookahead_end]))
        if k_ahead < cfg.straight_thresh:
            # 直线段: 长轴已沿切线覆盖前方, 步进可用长轴
            base_step = tr * 2 * cfg.spacing_ratio * cfg.step_straight_factor
        elif k_ahead >= cfg.tight_thresh:
            # 急弯: 保守步进, 基于短轴
            base_step = min(tr, sr * 2.0) * 2 * cfg.spacing_ratio * cfg.step_tight_factor
        else:
            # 弯曲段: 中等限制
            base_step = min(tr, sr * 2.5) * 2 * cfg.spacing_ratio
        cursor += max(base_step, cfg.min_size * 0.4)

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


def render_element_to_mask(elem, img_shape):
    """将单个图元渲染到二值 mask, 用于快速覆盖检测。"""
    h, w = img_shape[:2]
    single = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(elem['center']['x']), int(elem['center']['y'])
    ang = elem['rotation']
    if elem['type'] == ShapeType.ELLIPSE:
        s = elem['size']
        axes = (max(int(s['rx']), 1), max(int(s['ry']), 1))
        cv2.ellipse(single, (cx, cy), axes, ang, 0, 360, 255, -1)
    elif elem['type'] == ShapeType.RECTANGLE:
        rw, rh = elem['size']['width'], elem['size']['height']
        rect = ((cx, cy), (max(rw, 1), max(rh, 1)), ang)
        bpts = np.intp(cv2.boxPoints(rect))
        cv2.drawContours(single, [bpts], 0, 255, -1)
    return single


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
    V5: 多策略间隙检测 + 曲率引导填充。
    策略1: 渲染覆盖掩码检测 (像素级)
    策略2: 图元序列间距检测 (拓扑级, 补充渲染法盲区)
    """
    current = list(elements)

    # V5: 预计算曲率, 用于填充图元的类型选择 (替代固定 kappa=0.03)
    kappa = compute_curvature(contour_pts, cfg.curvature_sigma)
    labels = classify_curvature(kappa, cfg)
    labels = merge_short_runs(labels, min_run=5)
    N_pts = len(contour_pts)

    for iteration in range(cfg.gap_fill_iterations):
        # 策略1: 渲染覆盖检测
        gaps = detect_gaps(current, contour_pts, cum_arc, total_arc,
                           img_shape, cfg)

        # V5 策略2: 图元序列间距检测
        if current:
            sorted_elems = sorted(
                current, key=lambda e: e.get('_cursor', 0))
            for i in range(1, len(sorted_elems)):
                e1, e2 = sorted_elems[i - 1], sorted_elems[i]
                c1, c2 = e1['center'], e2['center']
                d = math.sqrt((c1['x'] - c2['x']) ** 2
                              + (c1['y'] - c2['y']) ** 2)
                tr1 = e1.get('_tr', cfg.min_size / 2)
                tr2 = e2.get('_tr', cfg.min_size / 2)
                expected = min(tr1, tr2) * 2 * cfg.spacing_ratio
                if d > expected * 1.5:
                    s_arc = e1.get('_cursor', 0)
                    e_arc = e2.get('_cursor', 0)
                    g_len = e_arc - s_arc
                    if g_len > cfg.min_size * 0.3:
                        already = any(abs(g[0] - s_arc) < cfg.min_size
                                      for g in gaps)
                        if not already:
                            gaps.append((s_arc, e_arc, g_len))

        if not gaps:
            break

        # 渲染已有图元的覆盖掩码, 用于跳过已被充分覆盖的填充位置
        cov_mask = render_coverage_mask(current, img_shape)

        added = 0
        for g_start, g_end, g_len in gaps:
            if g_len < cfg.min_size * 0.3:
                continue
            n_fillers = max(1, int(g_len / cfg.min_size))
            for fi in range(n_fillers):
                frac = (fi + 0.5) / n_fillers
                fill_arc = g_start + frac * g_len
                idx = cursor_to_index(fill_arc, cum_arc, total_arc)
                px, py = contour_pts[idx]
                tangent, normal = tangent_normal_at(
                    contour_pts, idx, dist_map)

                # V5: 使用真实曲率
                k_val = float(kappa[idx]) if idx < N_pts else 0.03
                lb_val = int(labels[idx]) if idx < N_pts else 1

                best = best_primitive_at(
                    px, py, tangent, normal,
                    dist_map, poly, cfg,
                    kappa=k_val, seg_label=lb_val,
                )
                if best is None:
                    continue
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

                # 跳过已被现有图元大面积覆盖的填充图元
                single = render_element_to_mask(elem, img_shape)
                elem_px = np.sum(single > 0)
                if elem_px > 0:
                    covered_px = np.sum((single > 0) & (cov_mask > 0))
                    if covered_px / elem_px > 0.7:
                        continue
                cov_mask = np.maximum(cov_mask, single)

                current.append(elem)
                added += 1

        if added == 0:
            break

    return current


# ═══════════════════════════════════════════════════════
#  P2-C: 图元扩展
# ═══════════════════════════════════════════════════════

def expand_elements(elements, poly, cfg, dist_map=None):
    """
    V5: 双向扩展 (长轴 + 短轴) + 距离场双重验证。
    在两个轴向上分别尝试扩大, 提高覆盖饱满度。
    """
    contain_t = 0.88 + cfg.precision * 0.10

    for elem in elements:
        if elem.get('_poly') is None:
            continue
        cx = elem['center']['x']
        cy = elem['center']['y']
        ang = elem['rotation']

        if elem['type'] == ShapeType.ELLIPSE:
            rx = elem['size']['rx']
            ry = elem['size']['ry']
            # 长轴扩展
            for g in cfg.expand_growth_factors:
                new_rx = rx * g
                ep = make_ellipse_poly(cx, cy, new_rx, ry, ang)
                try:
                    ia = poly.intersection(ep).area
                    ea = ep.area
                    if ea > 0 and ia >= ea * contain_t:
                        if dist_map is None or \
                                dist_map_containment_check(ep, dist_map) >= 0.92:
                            elem['size']['rx'] = round(new_rx, 2)
                            elem['_poly'] = ep
                            elem['_tr'] = new_rx
                            rx = new_rx
                            break
                except Exception:
                    pass
            # 短轴扩展
            for g in cfg.expand_growth_factors:
                new_ry = ry * g
                ep = make_ellipse_poly(cx, cy, rx, new_ry, ang)
                try:
                    ia = poly.intersection(ep).area
                    ea = ep.area
                    if ea > 0 and ia >= ea * contain_t:
                        if dist_map is None or \
                                dist_map_containment_check(ep, dist_map) >= 0.92:
                            elem['size']['ry'] = round(new_ry, 2)
                            elem['_poly'] = ep
                            ry = new_ry
                            break
                except Exception:
                    pass

        elif elem['type'] == ShapeType.RECTANGLE:
            rw = elem['size']['width']
            rh = elem['size']['height']
            # 长边扩展
            for g in cfg.expand_growth_factors:
                new_w = rw * g
                rp = make_rect_poly(cx, cy, new_w, rh, ang)
                try:
                    ia = poly.intersection(rp).area
                    ra = rp.area
                    if ra > 0 and ia >= ra * contain_t:
                        if dist_map is None or \
                                dist_map_containment_check(rp, dist_map) >= 0.92:
                            elem['size']['width'] = round(new_w, 2)
                            elem['_poly'] = rp
                            elem['_tr'] = new_w / 2
                            rw = new_w
                            break
                except Exception:
                    pass
            # 短边扩展
            for g in cfg.expand_growth_factors:
                new_h = rh * g
                rp = make_rect_poly(cx, cy, rw, new_h, ang)
                try:
                    ia = poly.intersection(rp).area
                    ra = rp.area
                    if ra > 0 and ia >= ra * contain_t:
                        if dist_map is None or \
                                dist_map_containment_check(rp, dist_map) >= 0.92:
                            elem['size']['height'] = round(new_h, 2)
                            elem['_poly'] = rp
                            rh = new_h
                            break
                except Exception:
                    pass

    return elements


def suppress_overlap(elements, img_shape, coverage_thresh=0.85):
    """
    重叠抑制 — 用渲染掩码检测累积覆盖, 移除冗余图元。

    策略: 按面积从大到小排序, 依次渲染到累积掩码。
    对每个较小图元, 检查其像素是否已被掩码大面积覆盖。
    相比逐对 Shapely 交集, 自然处理多图元联合覆盖且更快。
    """
    if len(elements) <= 1:
        return elements

    h, w = img_shape[:2]

    def elem_area(e):
        p = e.get('_poly')
        if p is not None:
            try:
                return p.area
            except Exception:
                pass
        return 0

    # 按面积降序排列
    indexed = sorted(enumerate(elements), key=lambda x: elem_area(x[1]),
                     reverse=True)

    accepted_idx = set()
    cumulative_mask = np.zeros((h, w), dtype=np.uint8)
    removed = 0

    for orig_idx, elem in indexed:
        single = render_element_to_mask(elem, img_shape)
        elem_px = np.sum(single > 0)

        if elem_px < 1:
            accepted_idx.add(orig_idx)
            continue

        # 检查该图元是否已被累积掩码大面积覆盖
        covered_px = np.sum((single > 0) & (cumulative_mask > 0))
        if covered_px / elem_px >= coverage_thresh:
            removed += 1
            continue

        accepted_idx.add(orig_idx)
        cumulative_mask = np.maximum(cumulative_mask, single)

    result = [elements[i] for i in sorted(accepted_idx)]
    if removed > 0:
        print(f'  重叠抑制: 移除 {removed} 个冗余图元')
    return result
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

        if elem['type'] == ShapeType.ELLIPSE:
            fill_c = (0, 230, 255, 255)       # 黄色填充 (BGRA)
            border_c = (0, 180, 220, 255)
            ov_c = (0, 200, 255)               # 黄色描边 (BGR)
            s = elem['size']
            axes = (max(int(s['rx']), 1), max(int(s['ry']), 1))
            cv2.ellipse(canvas, (cx, cy), axes, ang, 0, 360, fill_c, -1)
            cv2.ellipse(canvas, (cx, cy), axes, ang, 0, 360, border_c, 1)
            cv2.ellipse(overlay, (cx, cy), axes, ang, 0, 360, ov_c, 1)

        elif elem['type'] == ShapeType.RECTANGLE:
            fill_c = (255, 150, 50, 255)       # 蓝色填充 (BGRA)
            border_c = (255, 100, 30, 255)
            ov_c = (255, 120, 0)               # 蓝色描边 (BGR)
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
        description='Shaper V6 – 轮廓描边工具')
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
    ap.add_argument('-o', '--output_dir', type=str, default=None,
                    help='输出目录 (默认=输入图片所在目录)')
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

    # ── 输出目录 ──
    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.dirname(os.path.abspath(args.image_path))
    os.makedirs(out_dir, exist_ok=True)

    # ── P3: 智能 Mask 提取 ──
    mask = extract_mask(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 额外的开操作去除噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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

                # P2-C: 图元双向扩展
                elems = expand_elements(elems, poly, cfg, dist_map)

        all_elements.extend(elems)

    # ── P2-E: 全局重叠抑制 ──
    all_elements = suppress_overlap(all_elements, img.shape)

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

    with open(os.path.join(out_dir, 'result_C_data.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # 保存 mask 调试图
    cv2.imwrite(os.path.join(out_dir, 'result_D_mask.png'), mask)

    draw_results(img.shape, output['elements'],
                 os.path.join(out_dir, 'result_A_shapes_only.png'),
                 os.path.join(out_dir, 'result_B_overlay.png'), img)

    print(f'完成! 共 {len(all_elements)} 个图元')
    print(f'输出目录: {out_dir}')
    print('  result_A_shapes_only.png, result_B_overlay.png,')
    print('  result_C_data.json, result_D_mask.png')


if __name__ == '__main__':
    main()
