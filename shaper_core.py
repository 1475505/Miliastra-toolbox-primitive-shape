"""
Shaper Core API — 核心逻辑与交互分离
"""

import cv2
import numpy as np
import base64
import time
import math
from shapely.geometry import Polygon
import final_shaper as fs


def process_image(image_bytes, config=None):
    """
    处理图片并返回结构化结果。

    Args:
        image_bytes: 原始图片文件字节
        config: dict, 可选键:
            primitive_size (float): 图元基准大小 (px), 默认 15
                实际输出图元在 size*0.4 ~ size*2 之间自动缩放
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
        'image_center': {'x': img_center[0], 'y': img_center[1]},
        'image_size': {'width': w, 'height': h},
        'config': {
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
