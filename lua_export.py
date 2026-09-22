# -*- coding: utf-8 -*-
"""图元拟合结果 -> 千星奇域客户端 Lua 脚本导出。

输入为 shaper_core.process_image 的 result_data（fill / outline 两种模式通用），
输出一份自带使用说明注释的 levelScript.lua 文本：

    结果 JSON -> 原图像素坐标 -> 调色板去重 -> ELEMENTS 记录表 -> Lua 文本

坐标约定（导出到 Lua 后）：
    - 以原图左下角为原点，X 向右、Y 向上（与千星 UI 画布一致）；
    - 单位为原图像素；运行时再乘以 BASE_SCALE / 自适应缩放；
    - 旋转角直接沿用元素 rotation.z（工具链已换算为 Y 轴向上约定，
      在 Unity 风格 UI 中绕 Z 正值即逆时针，无需再次取负）。
"""

import math

DEFAULT_PIXEL_PER_UNIT = 1.0

KIND_RECT = 0
KIND_CIRCLE = 1
KIND_TRIANGLE = 2

KIND_NAMES = {
    KIND_RECT: "实心方块",
    KIND_CIRCLE: "实心圆形(可拉伸成椭圆)",
    KIND_TRIANGLE: "实心三角形",
}

_HEADER_TEMPLATE = """\
-- =========================================================
-- {name} - 千星奇域图元拟合拼图
-- 由 Miliastra-toolbox-primitive-shape 自动导出
-- 模式: {mode} | 图元数: {element_count} (背景 {background_count}) | 原图: {img_w}x{img_h}
-- =========================================================
--
-- 使用说明
-- 1. 准备一个客户端图片控件，设为「仅存为模板」，关闭遮罩/羽化。
-- 2. 必填：把 IMAGE_PREFAB_ID = 0 改为该控件的「控件模板索引ID」。
--    不是图片资产ID；只填这一处，脚本自动切换矩形、圆形与三角形图片。
-- 3. 挂载：将本文件作为客户端脚本挂到专用空客户端容器节点。
--    进入运行预览后在 OnStart 绘制；不要把创建逻辑移到 OnInit。
-- 可选：BASE_SCALE 缩放；OFFSET_X/Y 平移（Y向上）；SKIP_BACKGROUND 跳过背景。
-- 每个图元创建一个图片控件。控件容量与真机显示请在运行预览中确认。
-- =========================================================
"""

_BODY_TOP_TEMPLATE = """\
-- ====================== 可调参数 ==========================

-- 必填：客户端图片控件模板索引ID
local IMAGE_PREFAB_ID = 0
-- 静态图片资产（无需修改）
local IMAGE_ID_BY_KIND = {{ [0] = 100001, [1] = 100002, [2] = 100003 }}

local BASE_SCALE = 1.0 -- 整体缩放倍率
local FIT_TO_CANVAS = true -- 超出画布时自动缩小
local CANVAS_MARGIN = 0.9 -- 自适应时占画布宽高比例的上限
local OFFSET_X = 0 -- 图案中心相对父控件中心的水平偏移
local OFFSET_Y = 0 -- 图案中心相对父控件中心的竖直偏移(Y 向上)
local SKIP_BACKGROUND = false -- true 时跳过下面的背景图元
local ROTATION_BIAS = 0 -- 全局附加旋转角(度)，一般保持 0

-- ==================== 数据(勿手改) ========================

local IMG_WIDTH = {img_w} -- 原图宽度(像素)
local IMG_HEIGHT = {img_h} -- 原图高度(像素)
local BACKGROUND_COUNT = {background_count} -- 开头的背景图元数量

-- 调色板: RGB 0-255
local PALETTE = {{
{palette_body}
}}

-- 图元记录: {{kind, cx, cy, w, h, rotZ, colorIndex, alpha}}
--   kind: 0=矩形 1=椭圆(圆拉伸) 2=三角形
--   cx,cy: 图元中心，原图像素坐标，左下角原点、Y 向上
--   w,h:   矩形/三角形为宽高；椭圆为直径(2*rx, 2*ry)
--   rotZ:  旋转角(度)，绕各自身心；三角形轴心在其质心(0.5, 1/3)
--   colorIndex: PALETTE 下标;  alpha: 0-255
local ELEMENTS = {{
{elements_body}
}}
"""

_BODY_BOTTOM_TEMPLATE = """\
-- ===================== 运行时逻辑 ==========================

local createdControls = {{}}
local warnedMissingPrefab = false

local function GetPrefabId(kind)
    local id = IMAGE_PREFAB_ID
    if type(id) == "number" and id > 0 and id % 1 == 0 then
        return id
    end
    if not warnedMissingPrefab then
        warnedMissingPrefab = true
        printerr("[图元拼图] 请填写 IMAGE_PREFAB_ID：客户端图片控件模板索引ID")
    end
    return nil
end

local function DestroyAll()
    for _, control in ipairs(createdControls) do
        pcall(function()
            game.DestroyClientUIControl(control)
        end)
    end
    createdControls = {{}}
end

local function DrawElement(parent, item, scale)
    local prefabId = GetPrefabId(item[1])
    if prefabId == nil then
        return
    end

    local image = game.InstantiateClientUIControl(prefabId, parent)
    if image == nil then
        return
    end

    -- 三角形质心在底边上方 1/3 高度处，其余形状为中心
    local pivotY = 0.5
    if item[1] == {kind_triangle} then
        pivotY = 1 / 3
    end

    image:SetAnchorMin(0.5, 0.5)
    image:SetAnchorMax(0.5, 0.5)
    image:SetPivot(0.5, pivotY)
    image:SetImage(Enum.ImageSource.StaticReference, IMAGE_ID_BY_KIND[item[1]])
    pcall(function() image.imageType = Enum.ImageType.Stretch end)

    local color = PALETTE[item[7]]
    if color ~= nil then
        image.imageColor = Color.FromRGBA(color[1], color[2], color[3], item[8])
    end

    image:SetSizeDelta(item[4] * scale, item[5] * scale)

    local rotZ = item[6] + ROTATION_BIAS
    if rotZ ~= 0 then
        image:SetLocalRotation(0, 0, rotZ)
    end

    local px = (item[2] - IMG_WIDTH / 2) * scale + OFFSET_X
    local py = (item[3] - IMG_HEIGHT / 2) * scale + OFFSET_Y
    image:SetAnchoredPosition(px, py)

    image:SetAsLastSibling()
    table.insert(createdControls, image)
end

function OnStart()
    DestroyAll()

    local parent = script.object
    if parent == nil then
        printerr("[图元拼图] 取不到脚本宿主控件(script.object)，图元未创建")
        return
    end

    -- 计算最终缩放：BASE_SCALE 与画布自适应取小者
    local scale = BASE_SCALE
    if FIT_TO_CANVAS then
        local canvasWidth, canvasHeight = game.GetUICanvasSize()
        if canvasWidth and canvasHeight and canvasWidth > 0 and canvasHeight > 0 then
            local fitScale = math.min(canvasWidth / IMG_WIDTH, canvasHeight / IMG_HEIGHT) * CANVAS_MARGIN
            if fitScale > 0 and fitScale < scale then
                scale = fitScale
            end
        end
    end

    local startIndex = 1
    if SKIP_BACKGROUND then
        startIndex = BACKGROUND_COUNT + 1
    end

    for index = startIndex, #ELEMENTS do
        DrawElement(parent, ELEMENTS[index], scale)
    end
end

function OnDestroy()
    DestroyAll()
end
"""


def _normalize_type(element):
    raw = str(element.get("type", "")).lower()
    if raw in ("circle", "ellipse"):
        return "ellipse"
    if raw in ("rect", "rectangle"):
        return "rectangle"
    if raw == "triangle":
        return "triangle"
    return raw or "unknown"


def _color_to_rgb(element):
    color = element.get("color")
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        try:
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
        except (TypeError, ValueError):
            pass
    elif isinstance(color, str):
        value = color.strip().lstrip("#")
        if len(value) == 3:
            value = "".join(ch + ch for ch in value)
        if len(value) >= 6:
            try:
                return (
                    int(value[0:2], 16),
                    int(value[2:4], 16),
                    int(value[4:6], 16),
                )
            except ValueError:
                pass
    packed = element.get("packed_color")
    if isinstance(packed, int):
        return (packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF
    return 255, 255, 255


def _alpha_to_255(element):
    try:
        alpha = float(element.get("alpha", 1.0))
    except (TypeError, ValueError):
        alpha = 1.0
    alpha = max(0.0, min(1.0, alpha))
    return int(round(alpha * 255))


def _rotation_z(element):
    rotation = element.get("rotation", 0)
    if isinstance(rotation, dict):
        try:
            return float(rotation.get("z") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(rotation or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _element_geometry(element, kind):
    """返回 (cx, cy, w, h)，单位为原图像素，cy 为左下原点 Y 向上。"""
    center = element.get("center") or {}
    size = element.get("size") or {}
    try:
        cx = float(center.get("x", 0))
        cy = float(center.get("y", 0))
    except (TypeError, ValueError):
        return None

    if kind == KIND_CIRCLE:
        rx = size.get("rx")
        ry = size.get("ry")
        if rx is None:
            rx = float(size.get("width", 0) or 0) / 2.0
        else:
            rx = float(rx)
        if ry is None:
            ry = float(size.get("height", 0) or 0) / 2.0
        else:
            ry = float(ry)
        w, h = abs(rx * 2.0), abs(ry * 2.0)
    else:
        w = abs(float(size.get("width", 0) or 0))
        h = abs(float(size.get("height", 0) or 0))
        if kind == KIND_TRIANGLE and h <= 0 and w > 0:
            h = w * math.sqrt(3.0) / 2.0

    return cx, cy, w, h


class _PaletteBuilder:
    def __init__(self):
        self._index_by_rgb = {}
        self.items = []

    def index_for(self, rgb):
        found = self._index_by_rgb.get(rgb)
        if found is not None:
            return found
        self.items.append(rgb)
        self._index_by_rgb[rgb] = len(self.items)
        return len(self.items)


def build_lua_export_text(result_data, image_name=""):
    """把 process_image 的结果转换为 levelScript.lua 文本。"""
    config = result_data.get("config") or {}
    image_size = result_data.get("image_size") or {}
    img_w = float(image_size.get("width", 0) or 0)
    img_h = float(image_size.get("height", 0) or 0)
    if img_w <= 0 or img_h <= 0:
        raise ValueError("result_data 缺少有效的 image_size")

    try:
        pixel_per_unit = float(config.get("pixel_per_unit") or DEFAULT_PIXEL_PER_UNIT)
    except (TypeError, ValueError):
        pixel_per_unit = DEFAULT_PIXEL_PER_UNIT
    if pixel_per_unit <= 0:
        pixel_per_unit = DEFAULT_PIXEL_PER_UNIT

    mode = str(result_data.get("mode", "fill"))
    safe_name = str(image_name or "").strip() or "shaper_result"
    safe_name = safe_name.replace("\n", " ").replace("\r", " ").replace("--", "﹍")

    palette = _PaletteBuilder()
    records = []
    background_count = 0
    skipped = 0
    used_kinds = set()

    for element in result_data.get("elements", []) or []:
        norm_type = _normalize_type(element)
        kind = {
            "rectangle": KIND_RECT,
            "ellipse": KIND_CIRCLE,
            "triangle": KIND_TRIANGLE,
        }.get(norm_type)
        geometry = None if kind is None else _element_geometry(element, kind)
        if kind is None or geometry is None:
            skipped += 1
            continue

        cx, cy, w, h = geometry
        # 存储坐标是"单元"坐标(除以过 pixel_per_unit)，还原回原图像素
        cx *= pixel_per_unit
        cy *= pixel_per_unit
        w *= pixel_per_unit
        h *= pixel_per_unit

        rgb = _color_to_rgb(element)
        color_index = palette.index_for(rgb)
        alpha255 = _alpha_to_255(element)
        rot_z = round(_rotation_z(element), 2)

        record = {
            "kind": kind,
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "w": round(w, 2),
            "h": round(h, 2),
            "rot_z": rot_z,
            "color_index": color_index,
            "alpha": alpha255,
            "is_background": bool(element.get("is_background")),
        }
        if record["is_background"]:
            background_count += 1
        records.append(record)
        used_kinds.add(kind)

    if skipped:
        records.append({"__comment__": f"另有 {skipped} 个不支持的图元被跳过"})

    # 背景图元保持在最前(与渲染顺序一致)
    records.sort(key=lambda item: 0 if item.get("is_background") else 1)

    palette_body_lines = []
    for index, (r, g, b) in enumerate(palette.items, start=1):
        palette_body_lines.append(f"    [{index}] = {{{r}, {g}, {b}}},")

    element_lines = []
    for record in records:
        if "__comment__" in record:
            element_lines.append(f"    -- {record['__comment__']}")
            continue
        element_lines.append(
            "    {{{kind}, {cx}, {cy}, {w}, {h}, {rot_z}, {color_index}, {alpha}}},".format(
                **record
            )
        )

    header = _HEADER_TEMPLATE.format(
        name=safe_name,
        mode=mode,
        element_count=len(records) - (1 if skipped else 0),
        background_count=background_count,
        img_w=int(round(img_w)),
        img_h=int(round(img_h)),
    )

    body_top = _BODY_TOP_TEMPLATE.format(
        kind_rect=KIND_RECT,
        kind_circle=KIND_CIRCLE,
        kind_triangle=KIND_TRIANGLE,
        img_w=int(round(img_w)),
        img_h=int(round(img_h)),
        background_count=background_count,
        palette_body="\n".join(palette_body_lines) or "    -- 空",
        elements_body="\n".join(element_lines) or "    -- 空",
    )

    body_bottom = _BODY_BOTTOM_TEMPLATE.format(kind_triangle=KIND_TRIANGLE)

    return header + body_top + body_bottom
