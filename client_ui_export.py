"""拟合结果 -> 客户端「容器节点」控件模板 GIA（纯 Python）。

协议来源：qxqy-lua/knowledge/ui/{gia.proto,01..05}，以及
ugc-tool/src/views/ClientUIAnimationEditor/giaExporter.ts。
客户端 sibling 按底→顶写入，不能套用素材组 GIA 的反序规则。
根资产是 UIControlTemplate(70)，不是服务端 UIControlGroup(21)。
坐标还原为导出图像素，所选原点对齐容器中心，Y 向上、Z 逆时针。
"""

import math
import struct

import lua_export


FIRST_GUID = 1073741825
GAME_VERSION = "7.1.0"
_KINDS = {"rectangle": 0, "ellipse": 1, "triangle": 2}
_ASSETS = {"rectangle": 100001, "ellipse": 100002, "triangle": 100003}
# tag: (Data.field501, Data.field502, Details.field501)
_SLOTS = {73: (63, 83, 65), 77: (67, 90, 69),
          78: (68, 91, 70), 83: (73, 96, 75), 84: (74, 97, 76)}


def _varint(value):
    value = int(value)
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("GIA 整数超出范围")
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def _uint(tag, value):
    return _varint(tag << 3) + _varint(value)


def _blob(tag, value=b""):
    return _varint((tag << 3) | 2) + _varint(len(value)) + value


def _text(tag, value):
    return _blob(tag, value.encode("utf-8"))


def _number(value, label, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须是有限数值") from exc
    if not math.isfinite(number) or abs(number) > 3.4028234663852886e38:
        raise ValueError(f"{label} 必须是有限的 float32 数值")
    if positive and number <= 0:
        raise ValueError(f"{label} 必须大于 0")
    return number


def _vector(values, tags=(501, 502)):
    # 向量本身保留显式空 message，零分量依 proto3 省略。
    return b"".join(_varint((tag << 3) | 5) + struct.pack("<f", _number(value, "布局"))
                    for tag, value in zip(tags, values) if value != 0)


def _identity(guid):
    return _uint(2, 1) + _uint(3, 8) + _uint(4, guid)


def _details(guid, first, kind, value):
    return (value + _uint(501, first) + _uint(502, kind)
            + _uint(503, 1) + _blob(504, _identity(guid)))


def _slot(guid, tag, value=b""):
    first, kind, inner = _SLOTS[tag]
    return (_blob(tag) + _uint(501, first) + _uint(502, kind)
            + _blob(503, _details(guid, inner, kind, _blob(tag + 1, value))))


def _name_component(name):
    return _blob(12, _text(501, name)) + _uint(501, 2) + _uint(502, 15)


def _type_component(guid):
    value = _blob(17) + _uint(501, 7)
    kind = 86
    return (_blob(14, value) + _uint(501, 4) + _uint(502, kind)
            + _blob(503, _details(guid, 5, kind, _blob(14, value))))


def _transform(guid, preset, offset=(0, 0), size=(0, 0), pivot=(.5, .5),
               rotation=(0, 0, 0)):
    rect = b"".join(_blob(tag, _vector(values, axes)) for tag, values, axes in (
        (501, (1, 1, 1), (1, 2, 3)),
        (502, (.5, .5), (501, 502)),
        (503, (.5, .5), (501, 502)),
        (504, offset, (501, 502)), (505, size, (501, 502)),
        (506, pivot, (501, 502)), (508, rotation, (1, 2, 3)),
    ))
    platforms = b"".join(_blob(501, (_uint(501, device) if device else b"") + _blob(502, rect))
                         for device in range(4))
    transform = _blob(12, platforms + _uint(502, preset) + _uint(504, 1)) + _uint(501, 2)
    return (_blob(11, _blob(12) + _uint(501, 2)) + _uint(501, 1) + _uint(502, 12)
            + _blob(503, _details(guid, 4, 12, _blob(13, transform))))


def _image_config(mask=False):
    # 保留原生图片 B 槽默认值；图元关闭遮罩、羽化、进度填充。
    values = {502: 1, 504: 1, 505: 1, 507: 2, 508: 100, 510: 1, 515: 1}
    return ((_uint(501, 1) if mask else b"")
            + b"".join(_uint(tag, value) for tag, value in values.items())
            + _blob(511, _vector((8, 8)))
            + _vector((85, 85), (512, 513)))


def _unit(guid, name, components, children=(), parent=None, template=False):
    info = _blob(11, _uint(501, guid)) + _uint(501, 1) + _uint(502, 5)
    content = _uint(501, guid) + _blob(502, info)
    # 每个新建根/子控件登记独立资产编号，不能按图片类型复用编号。
    # 这里只构造新文件，按创建顺序登记；不用于重编既有 GIA 的编号。
    asset_index = guid - FIRST_GUID + 1
    content += _blob(502, _blob(12, _uint(501, asset_index)) + _uint(501, 2) + _uint(502, 6))
    if template:
        # 新模板没有外部伴生资源，不复制其他模板的关联 GUID。
        content += _blob(502, _blob(14) + _uint(501, 4) + _uint(502, 4))
    if children:
        content += _blob(503, b"".join(_varint(child) for child in children))
    if parent is not None:
        content += _uint(504, parent)
    content += b"".join(_blob(505, component) for component in components)
    return (_blob(1, _identity(guid))
            + b"".join(_blob(2, _identity(child)) for child in children)
            + _text(3, name) + _uint(5, 70 if template else 15)
            + _blob(19, _blob(1, content)))


def _control(guid, name, tag, children=(), parent=None, offset=(0, 0), size=(0, 0),
             pivot=(.5, .5), rotation=(0, 0, 0), color=0xFFFFFFFF, asset=100001, mask=False,
             template=False):
    components = [_name_component(name), _type_component(guid),
                  _transform(guid, 25 if tag == 78 else 23, offset, size, pivot, rotation),
                  _slot(guid, 73)]
    if tag == 83:
        # A.504=Stretch，依据 ugc-tool 原生图片字段映射（不在旧 proto 快照中）。
        components += [_slot(guid, 83, _uint(502, color) + _uint(503, asset) + _uint(504, 1)),
                       _slot(guid, 84, _image_config(mask))]
    else:
        components.append(_slot(guid, 78))
    components.append(_slot(guid, 77))
    return _unit(guid, name, components, children, parent, template=template)


def _packed_color(element):
    r, g, b = lua_export._color_to_rgb(element)
    alpha = element.get("alpha")
    if alpha is None:
        color = element.get("color")
        if isinstance(color, (list, tuple)) and len(color) >= 4:
            alpha = _number(color[3], "颜色 Alpha") / 255
        elif element.get("packed_color") is not None:
            alpha = ((int(element["packed_color"]) >> 24) & 255) / 255
        else:
            alpha = .5  # 与结果页、CSS 和素材组导出的回落一致。
    alpha = max(0, min(1, _number(alpha, "透明度")))
    return (int(math.floor(alpha * 255 + .5)) << 24) | (r << 16) | (g << 8) | b


def build_container_node_gia(result_data, image_name="", origin_x=None, origin_y=None):
    """生成客户端「容器节点」模板；origin 使用图片像素坐标（Y 向下）。"""
    image_size = result_data.get("image_size") or {}
    width = _number(image_size.get("width", 0), "图片宽度", positive=True)
    height = _number(image_size.get("height", 0), "图片高度", positive=True)
    config = result_data.get("config") or {}
    ppu = _number(config.get("pixel_per_unit", 1), "pixel_per_unit", positive=True)
    origin = result_data.get("image_center") or {"x": width / 2, "y": height / 2}
    ox = _number(origin.get("x", width / 2) if origin_x is None else origin_x, "原点 X")
    oy = _number(origin.get("y", height / 2) if origin_y is None else origin_y, "原点 Y")
    name = str(image_name or "").strip() or "shaper_result"
    elements = result_data.get("elements") or []
    mask = result_data.get("mask") or {}
    masked = bool(mask.get("enabled"))
    if masked and mask.get("shape_type", "rectangle") not in ("rectangle", "rect"):
        raise ValueError("容器节点导出目前只支持矩形裁剪")

    root_id = FIRST_GUID
    parent_id = root_id + 1 if masked else root_id
    first_image_id = parent_id + 1
    end_image_id = first_image_id + len(elements)
    if end_image_id - 1 > 2147483647:
        raise ValueError("图元数量超出 GIA 控件编号范围")
    image_ids = list(range(first_image_id, end_image_id))
    root_children = [parent_id] if masked else image_ids
    units = [_control(root_id, name, 78, children=root_children, size=(width, height), template=True)]
    # 默认以所选原点为图案原点。启用裁剪时将图元重定位到裁剪控件下。
    parent_x, parent_y = ox, -oy
    if masked:
        center, size = mask.get("center") or {}, mask.get("size") or {}
        parent_x = _number(center.get("x", 0), "裁剪中心 X") * ppu
        parent_y = _number(center.get("y", 0), "裁剪中心 Y") * ppu
        mask_size = tuple(_number(size.get(key, 0), "裁剪尺寸", positive=True) * ppu
                          for key in ("width", "height"))
        # 透明图片仍可遮罩后代，不引入可见的白色裁剪背景。
        units.append(_control(parent_id, "裁剪区域", 83, children=image_ids, parent=root_id,
                              offset=(parent_x - ox, parent_y + oy), size=mask_size,
                              color=0x00FFFFFF, mask=True))

    for index, (guid, element) in enumerate(zip(image_ids, elements), 1):
        shape = lua_export._normalize_type(element)
        if shape not in _KINDS:
            raise ValueError(f"第 {index} 个图元类型不支持：{shape}")
        geometry = lua_export._element_geometry(element, _KINDS[shape])
        if geometry is None:
            raise ValueError(f"第 {index} 个图元中心无效")
        cx, cy, w, h = geometry
        cx, cy = (_number(value, "图元中心") * ppu for value in (cx, cy))
        w, h = (_number(value, "图元尺寸", positive=True) * ppu for value in (w, h))
        rotation = element.get("rotation") or {}
        if not isinstance(rotation, dict):
            rotation = {"z": rotation}
        angles = tuple(_number(rotation.get(axis, 0), "旋转角度") for axis in ("x", "y", "z"))
        asset = element.get("image_asset_ref", _ASSETS[shape])
        if isinstance(asset, bool) or not isinstance(asset, int) or not 0 < asset <= 0xFFFFFFFF:
            raise ValueError(f"第 {index} 个图元的图片资产 ID 无效")
        units.append(_control(guid, str(index), 83, parent=parent_id,
                              offset=(cx - parent_x, cy - parent_y), size=(w, h),
                              pivot=(.5, 1 / 3 if shape == "triangle" else .5),
                              rotation=angles, color=_packed_color(element), asset=asset))

    # 主资产就是容器节点模板；附件只含它的后代，不添加服务端包装层。
    payload = (_blob(1, units[0]) + b"".join(_blob(2, unit) for unit in units[1:])
               + _text(5, GAME_VERSION))
    return struct.pack(">5I", len(payload) + 20, 1, 0x326, 3, len(payload)) + payload + struct.pack(">I", 0x679)
