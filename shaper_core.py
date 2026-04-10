"""
Shaper Core API — 核心逻辑与交互分离
支持两种模式:
  - outline (轮廓描边): 基于曲率路径行走的串珠式描边
  - fill (填充拟合): 基于 Hill Climbing 的图元缩放填充
"""

import cv2
import numpy as np
import base64
import time
import math
from shapely.geometry import Polygon
import final_shaper as fs
import fill_shaper


def process_image(image_bytes, config=None):
    """
    处理图片并返回结构化结果。

    Args:
        image_bytes: 原始图片文件字节
        config: dict, 可选键:
            mode (str): 'outline' 或 'fill', 默认 'fill'
            primitive_size (float): 图元基准大小 (px), 默认 15
            spacing (float): 间距系数, 默认 0.9
            precision (float): 精度 0.0-1.0, 默认 0.3
            num_primitives (int): fill 模式图元数量, 默认 100
            candidates (int): fill 模式每轮候选数, 默认 32
            hill_climb_iter (int): fill 模式爬山迭代数, 默认 64

    Returns:
        dict 包含图元数据、base64 图片等
    """
    if config is None:
        config = {}

    mode = config.get('mode', 'fill')
    if mode == 'fill':
        return process_image_fill(image_bytes, config)
    else:
        return process_image_outline(image_bytes, config)


def process_image_fill(image_bytes, config=None):
    """
    填充拟合模式：用圆形/矩形缩放拟合图片区域。

    Args:
        image_bytes: 原始图片文件字节
        config: dict, 可选键:
            primitive_size (float): 图元基准大小 (px), 默认 15
            num_primitives (int): 图元数量, 默认 100
            candidates (int): 每轮候选数, 默认 32
            hill_climb_iter (int): 爬山迭代数, 默认 64
            primitives (list): 图元配置列表
            origin (dict): 原点配置

    Returns:
        dict 包含图元数据、base64 图片等
    """
    if config is None:
        config = {}

    t0 = time.time()

    # 解码图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码图片")

    h, w = img.shape[:2]

    # 原点处理
    origin_cfg = config.get('origin', {})
    if origin_cfg.get('type') == 'custom':
        img_center = (
            float(origin_cfg.get('x', w / 2.0)),
            float(origin_cfg.get('y', h / 2.0))
        )
    elif origin_cfg.get('type') == 'top_left':
        img_center = (0.0, 0.0)
    else:
        img_center = (w / 2.0, h / 2.0)

    # 图元大小配置
    prim_size = max(3, min(200, config.get('primitive_size', 15)))

    # 允许的图元类型
    allowed_types = None
    type_colors = {}
    if 'primitives' in config and config['primitives']:
        at_set = set()
        for p in config['primitives']:
            s = p.get('shape')
            c = p.get('color')
            if s == 'circle':
                at_set.add(fill_shaper.ShapeType.CIRCLE)
                if c:
                    type_colors[fill_shaper.ShapeType.CIRCLE] = c
            elif s == 'rect':
                at_set.add(fill_shaper.ShapeType.RECT)
                if c:
                    type_colors[fill_shaper.ShapeType.RECT] = c
        if at_set:
            allowed_types = list(at_set)
        else:
            allowed_types = [fill_shaper.ShapeType.CIRCLE]

    # 填充配置
    fill_cfg = fill_shaper.FillConfig(
        num_primitives=int(config.get('num_primitives', 100)),
        candidates=int(config.get('candidates', 32)),
        hill_climb_iter=int(config.get('hill_climb_iter', 64)),
        min_size=max(2, prim_size * 0.3),
        max_size=max(4, prim_size * 3),
        allowed_types=allowed_types,
    )

    # 准备图片
    if len(img.shape) == 2:
        work_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        work_img = img.copy()
    else:
        work_img = img.copy()

    # 运行填充拟合
    results = fill_shaper.fit_primitives(work_img, fill_cfg)

    # 分配颜色覆盖
    for r in results:
        t = r.get('type')
        if t in type_colors:
            hex_color = type_colors[t].lstrip('#')
            r['color'] = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]

    # 转换为兼容元素格式
    primitives = config.get('primitives', [])
    elements = fill_shaper.results_to_elements(
        results, prim_size, img_center, primitives)

    # 渲染预览图
    preview = fill_shaper.render_results(work_img, results)

    elapsed = time.time() - t0

    # 编码图片
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img
    _, buf = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(buf).decode('utf-8')

    # 编码预览图
    _, pbuf = cv2.imencode('.png', preview)
    preview_b64 = base64.b64encode(pbuf).decode('utf-8')

    # 编码 mask (fill 模式用全白 mask)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    _, mbuf = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(mbuf).decode('utf-8')

    return {
        'mode': 'fill',
        'image_center': {'x': img_center[0], 'y': img_center[1]},
        'image_size': {'width': w, 'height': h},
        'config': {
            'mode': 'fill',
            'primitive_size': prim_size,
            'num_primitives': fill_cfg.num_primitives,
            'candidates': fill_cfg.candidates,
            'hill_climb_iter': fill_cfg.hill_climb_iter,
        },
        'elements_count': len(elements),
        'elements': elements,
        'image_base64': img_b64,
        'preview_base64': preview_b64,
        'mask_base64': mask_b64,
        'elapsed_seconds': round(elapsed, 2),
    }


def process_image_outline(image_bytes, config=None):
    """
    轮廓描边模式（原有逻辑）

    Args:
        image_bytes: 原始图片文件字节
        config: dict, 可选键:
            primitive_size (float): 图元基准大小 (px), 默认 15
            spacing (float): 间距系数, 默认 0.9
            precision (float): 精度 0.0-1.0, 默认 0.3

    Returns:
        dict 包含图元数据、base64 图片等
    """
    if config is None:
        config = {}

    t0 = time.time()

    # 解码图片
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码图片")

    h, w = img.shape[:2]
    
    # 原点处理
    origin_cfg = config.get('origin', {})
    if origin_cfg.get('type') == 'custom':
        img_center = (
            float(origin_cfg.get('x', w / 2.0)),
            float(origin_cfg.get('y', h / 2.0))
        )
    elif origin_cfg.get('type') == 'top_left':
        img_center = (0.0, 0.0)
    else:
        img_center = (w / 2.0, h / 2.0)

    # 从 primitive_size 推导 min/max
    prim_size = max(3, min(200, config.get('primitive_size', 15)))
    min_size = max(2, int(prim_size * 0.4))
    max_size = max(min_size + 2, int(prim_size * 2.0))

    # Mask 提取 + 形态学
    mask = fs.extract_mask(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    allowed_types = None
    type_colors = {}
    if 'primitives' in config and config['primitives']:
        at_set = set()
        for p in config['primitives']:
            s = p.get('shape')
            c = p.get('color')
            if s == 'circle':
                at_set.add(fs.ShapeType.ELLIPSE)
                if c:
                    type_colors[fs.ShapeType.ELLIPSE] = c
            elif s == 'rect':
                at_set.add(fs.ShapeType.RECTANGLE)
                if c:
                    type_colors[fs.ShapeType.RECTANGLE] = c
        if at_set:
            allowed_types = list(at_set)
        else:
            # 所有图元类型都被禁用，设置空列表表示不生成任何图元
            allowed_types = []

    cfg = fs.FittingConfig(
        min_size=min_size,
        max_size=max_size,
        spacing_ratio=config.get('spacing', 0.9),
        precision=max(0.0, min(1.0, config.get('precision', 0.3))),
        allowed_types=allowed_types
    )

    # 处理轮廓
    all_elements = []
    img_area = w * h
    min_contour_area = max(100, img_area * 0.00005)

    for ci, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw > w * 0.95 and ch > h * 0.95:
            continue
        margin = 5
        if (x + cw) < margin or x > (w - margin) \
                or (y + ch) < margin or y > (h - margin):
            continue

        elems = fs.fit_beads(ci, contour, mask, dist_map, cfg, img_center)
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) >= 3:
            poly = Polygon(pts).simplify(1.0, preserve_topology=True)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area >= cfg.min_size ** 2:
                cum_arc, total_arc = fs.build_arc_length_index(pts)
                elems = fs.fill_gaps(
                    elems, ci, pts, cum_arc, total_arc,
                    poly, dist_map, img.shape, cfg, img_center)
                elems = fs.expand_elements(elems, poly, cfg, dist_map)
        all_elements.extend(elems)

    all_elements = fs.suppress_overlap(all_elements, img.shape)
    
    # 分配颜色
    for e in all_elements:
        t = e.get('type')
        if t in type_colors:
            e['color'] = type_colors[t]
            
    elapsed = time.time() - t0

    # 编码图片
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img
    _, buf = cv2.imencode('.png', img_bgr)
    img_b64 = base64.b64encode(buf).decode('utf-8')
    _, mbuf = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(mbuf).decode('utf-8')

    elements = []
    origin_u = {'x': img_center[0] / prim_size, 'y': -img_center[1] / prim_size}
    for e in all_elements:
        item = {k: v for k, v in e.items() if not k.startswith('_')}

        cx = float(item['center']['x']) / prim_size
        cy = -float(item['center']['y']) / prim_size

        if 'size' in item:
            for k in item['size']:
                item['size'][k] = round(float(item['size'][k]) / prim_size, 4)

        rot_z = -float(item.get('rotation', 0))
        if item.get('type') == fs.ShapeType.RECTANGLE and 'size' in item:
            rect_h = float(item['size'].get('height', 0))
            theta = math.radians(rot_z)
            cx += (rect_h * 0.5) * math.sin(theta)
            cy += -(rect_h * 0.5) * math.cos(theta)

        item['center']['x'] = round(cx, 4)
        item['center']['y'] = round(cy, 4)

        item['relative_position'] = {
            'x': round(cx - origin_u['x'], 4),
            'y': round(cy - origin_u['y'], 4),
        }

        item['rotation'] = {'x': 0, 'y': 0, 'z': round(rot_z, 4)}

        elements.append(item)

    return {
        'mode': 'outline',
        'image_center': {'x': img_center[0], 'y': img_center[1]},
        'image_size': {'width': w, 'height': h},
        'config': {
            'mode': 'outline',
            'primitive_size': prim_size,
            'spacing': cfg.spacing_ratio,
            'precision': cfg.precision,
        },
        'elements_count': len(elements),
        'elements': elements,
        'image_base64': img_b64,
        'mask_base64': mask_b64,
        'elapsed_seconds': round(elapsed, 2),
    }
