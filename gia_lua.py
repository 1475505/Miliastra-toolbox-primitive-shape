"""超限图片素材组 GIA -> 客户端 Lua；不依赖外部工具目录或 protobuf 包。

字段依据：qxqy-lua/knowledge/ui/gia.proto 与图片拼图工具的 image-mode 编码器。
素材组（class=61）的 children 为前景优先；Lua 按逆序创建并显式置顶。
"""
import math
import struct


ASSET_NAMES = {100001: "矩形", 100002: "圆形/椭圆", 100003: "三角形",
               100004: "四角星", 100005: "五角星", 100006: "圆环"}


def _varint(data, pos):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise ValueError("GIA 数据截断")
        byte = data[pos]
        pos += 1
        if shift == 63 and byte > 1:
            raise ValueError("GIA varint 超出范围")
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("GIA varint 无效")


def _fields(data):
    result = {}
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        tag, wire = key >> 3, key & 7
        if not tag:
            raise ValueError("GIA 字段编号无效")
        if wire == 0:
            value, pos = _varint(data, pos)
        elif wire in (1, 2, 5):
            if wire == 2:
                size, pos = _varint(data, pos)
            else:
                size = 8 if wire == 1 else 4
            end = pos + size
            if end > len(data):
                raise ValueError("GIA 字段长度越界")
            value, pos = data[pos:end], end
        else:
            raise ValueError(f"GIA wire 类型不支持：{wire}")
        result.setdefault(tag, []).append((wire, value))
    return result


def _get(fields, tag, wire=0, default=0):
    values = fields.get(tag)
    if not values:
        return default
    actual, value = values[-1]
    if actual != wire:
        raise ValueError(f"GIA 字段 {tag} 类型不匹配")
    return value


def _message(fields, tag):
    return _fields(_get(fields, tag, 2, b""))


def _messages(fields, tag):
    for wire, value in fields.get(tag, []):
        if wire != 2:
            raise ValueError(f"GIA 字段 {tag} 应为消息")
        yield _fields(value)


def _children(content):
    children = []
    for wire, value in content.get(503, []):
        if wire == 0:
            children.append(value)
        elif wire == 2:
            pos = 0
            while pos < len(value):
                guid, pos = _varint(value, pos)
                children.append(guid)
        else:
            raise ValueError("GIA 子控件列表无效")
    return children


def _float(fields, tag, default=0):
    raw = _get(fields, tag, 5, None)
    value = default if raw is None else struct.unpack('<f', raw)[0]
    if not math.isfinite(value):
        raise ValueError("GIA 变换包含非有限数值")
    return value


def _vector(fields, tag, tags=(501, 502), default=(0, 0)):
    vec = _message(fields, tag)
    return [_float(vec, key, fallback) for key, fallback in zip(tags, default)]


def _components(content):
    return {_get(item, 502): item for item in _messages(content, 505)}


def _transform(components):
    component = components.get(12) or components.get(11)
    if not component:
        raise ValueError("素材控件缺少自定义变换")
    transform = _message(_message(component, 503), 13) or _message(component, 11)
    platforms = list(_messages(_message(transform, 12), 501))
    keyboard = next((p for p in platforms if _get(p, 501) == 0), None)
    if keyboard is None:
        raise ValueError("素材控件缺少键鼠布局")
    rect = _message(keyboard, 502)
    scale = _vector(rect, 501, (1, 2, 3), (1, 1, 1))
    rotation = _vector(rect, 508, (1, 2, 3), (0, 0, 0))
    if rotation[0] or rotation[1]:
        raise ValueError("暂不支持含 X/Y 轴旋转的素材，请先转为平面图片")
    # 保留轴心、锚点、缩放（含镜像），不重新拟合或按形状猜测轴心。
    return (_vector(rect, 504) + _vector(rect, 505)
            + _vector(rect, 506) + _vector(rect, 502) + _vector(rect, 503)
            + scale[:2] + [rotation[2]])


def parse_material_gia(data, ignore_mask=False):
    if len(data) < 24:
        raise ValueError("不是有效的 GIA 文件")
    length, schema, magic, kind, size = struct.unpack('>5I', data[:20])
    if (length != len(data) - 4 or size != len(data) - 24 or schema != 1
            or magic != 0x326 or kind != 3 or data[-4:] != b'\0\0\x06\x79'):
        raise ValueError("GIA 文件头、长度或尾标记无效")
    root = _fields(data[20:-4])
    if _get(root, 4) != 0:
        raise ValueError("请上传超限模式素材组；经典模式请先转换为超限模式")
    primary = list(_messages(root, 1))
    if len(primary) != 1 or _get(primary[0], 5) != 61:
        raise ValueError("需要图片素材组 GIA，不支持造物组、脚本或客户端控件模板 GIA")
    primary = primary[0]
    root_content = _message(_message(primary, 19), 1)
    root_guid = _get(root_content, 501)
    units = {}
    for unit in _messages(root, 2):
        content = _message(_message(unit, 19), 1)
        guid = _get(content, 501)
        if guid:
            if guid in units or guid == root_guid:
                raise ValueError("GIA 包含重复控件 GUID")
            units[guid] = content
    components = _components(root_content)
    mask = _message(_message(components.get(56, {}), 503), 47)
    has_mask = bool(_get(mask, 4))
    if has_mask and not ignore_mask:
        raise ValueError("素材组启用了遮罩/裁剪。当前不能等价转换组遮罩；可勾选「忽略组遮罩」绘制全部图片，或在素材组编辑器处理裁剪后重试")
    root_transform = _transform(components) if (12 in components or 11 in components) else None
    children = _children(root_content)
    if not children:
        raise ValueError("素材组没有图片控件")
    if len(children) != len(set(children)):
        raise ValueError("素材组子控件引用重复")
    # 现有 image_template.gia 含已删除图片的旧 GUID 引用；没有资源实体，不能生成控件。
    missing = [guid for guid in children if guid not in units]
    records = []
    for guid in reversed(children):
        if guid not in units:
            continue
        content = units[guid]
        if _get(content, 504) != root_guid or _children(content):
            raise ValueError("当前支持单层图片素材组，请先在素材组编辑器展平嵌套分组")
        parts = _components(content)
        image = _message(_message(parts.get(38, {}), 503), 31)
        asset = _get(image, 2)
        if not asset or 25 in parts:
            raise ValueError(f"控件 {guid} 不是静态图片；请将文本/动态素材转换成图片后重试")
        source = _message(image, 3)
        if _get(source, 501, default=2**64-1) != 2**64-1:
            raise ValueError(f"控件 {guid} 使用动态图片引用，暂不支持")
        color = _get(image, 4) & 0xffffffff
        rgba = [(color >> shift) & 255 for shift in (16, 8, 0, 24)]
        records.append([asset] + _transform(parts) + rgba)
    if not records:
        raise ValueError("素材组没有可读取的图片实体，子控件引用全部缺失")
    if root_transform is None:
        # 素材组根通常只有布局/遮罩组件，没有 RectTransform。图片以组原点为中心。
        # 用保守旋转包围盒建立对称画布，保持所有原始偏移不动。
        half_w = half_h = 1.0
        for row in records:
            if row[7:11] != [0.5, 0.5, 0.5, 0.5]:
                raise ValueError("素材组没有根尺寸，无法解析非居中锚点，请先改为中心锚点")
            _, x, y, w, h, px, py, *rest = row
            sx, sy, angle = row[11:14]
            radians = math.radians(angle)
            for dx in (-px * w, (1-px) * w):
                for dy in (-py * h, (1-py) * h):
                    xx = x + dx*sx*math.cos(radians) - dy*sy*math.sin(radians)
                    yy = y + dx*sx*math.sin(radians) + dy*sy*math.cos(radians)
                    half_w, half_h = max(half_w, abs(xx)), max(half_h, abs(yy))
        root_transform = [0, 0, half_w*2, half_h*2, .5, .5, .5, .5, .5, .5, 1, 1, 0]
    return {"root_transform": root_transform, "records": records, "ignored_mask": has_mask, "missing_refs": missing}


def build_gia_lua(data, name="素材组", ignore_mask=False):
    scene = parse_material_gia(data, ignore_mask=ignore_mask)
    records = scene['records']
    assets = sorted({row[0] for row in records})
    safe_name = str(name).replace('\r', ' ').replace('\n', ' ')
    lines = [f"-- {safe_name}：超限素材组 Lua 绘制脚本", f"-- 图片数量：{len(records)}；静态图片资产：{len(assets)} 种",
             "-- 1. 准备一个客户端图片控件，设为「仅存为模板」。关闭模板遮罩/羽化。",
             "-- 2. 必填：将下面 IMAGE_PREFAB_ID = 0 改为该图片控件的「控件模板索引ID」。",
             "--    不是图片资产ID；只填这一处，脚本会自动切换每个图元的静态图片。",
             "-- 3. 挂载：创建专用空客户端容器节点，将本文件作为客户端脚本挂到该节点。",
             "--    进入运行预览即绘制（OnStart）。脚本会设置该节点尺寸，请不要挂在已有界面的根节点上。",
             "-- 可选：BASE_SCALE 缩放；OFFSET_X/Y 平移（Y向上）。自定义图片需在当前关卡可用。",
             "-- 保留键鼠布局；不转换嵌套组、文本、动态引用及组遮罩。",
             "", "local IMAGE_PREFAB_ID = 0 -- 必填：客户端图片控件模板索引ID",
             "local BASE_SCALE = 1", "local OFFSET_X = 0", "local OFFSET_Y = 0",
              "-- 数据顺序：图片资产,x,y,w,h,pivotX,pivotY,anchorMinX,anchorMinY,anchorMaxX,anchorMaxY,scaleX,scaleY,rotZ,r,g,b,a",
              "local ROOT = {" + ','.join(map(repr, scene['root_transform'])) + "}", "local ELEMENTS = {"]
    lines += ["    {" + ','.join(map(repr, row)) + "}," for row in records]
    lines += ["}", _RUNTIME]
    if scene['ignored_mask']:
        lines.insert(0, "-- 注意：已按用户选项忽略原素材组遮罩；裁剪范围外的图片也会绘制，外观可能不同。")
    if scene['missing_refs']:
        lines.insert(0, f"-- 注意：原 GIA 有 {len(scene['missing_refs'])} 个没有图片实体的悬空引用，无法绘制；其余图片已保留。")
    return '\n'.join(lines)


_RUNTIME = '''
local created = {}
local function Clear()
    for i = #created, 1, -1 do
        game.DestroyClientUIControl(created[i])
    end
    created = {}
end

function OnStart()
    Clear()
    if type(IMAGE_PREFAB_ID) ~= "number" or IMAGE_PREFAB_ID <= 0 or IMAGE_PREFAB_ID % 1 ~= 0 then
        printerr("[GIA绘制] 请填写 IMAGE_PREFAB_ID：客户端图片控件模板索引ID")
        return
    end
    local parent = script.object
    if parent == nil then return end
    parent:SetAnchorMin(0.5, 0.5)
    parent:SetAnchorMax(0.5, 0.5)
    parent:SetPivot(ROOT[5], ROOT[6])
    parent:SetSizeDelta(ROOT[3], ROOT[4])
    parent:SetLocalScale(ROOT[11] * BASE_SCALE, ROOT[12] * BASE_SCALE, 1)
    parent:SetLocalRotation(0, 0, ROOT[13])
    parent:SetAnchoredPosition(OFFSET_X, OFFSET_Y)
    local ok, err = pcall(function()
        for _, item in ipairs(ELEMENTS) do
            local image = game.InstantiateClientUIControl(IMAGE_PREFAB_ID, parent)
            if image == nil then error("图片模板无法实例化，请确认已设为仅存为模板") end
            table.insert(created, image)
            image:SetImage(Enum.ImageSource.StaticReference, item[1])
            -- imageType 仅部分图片支持，保留不支持该属性的静态图片。
            pcall(function() image.imageType = Enum.ImageType.Stretch end)
            image:SetAnchorMin(item[8], item[9])
            image:SetAnchorMax(item[10], item[11])
            image:SetPivot(item[6], item[7])
            image:SetSizeDelta(item[4], item[5])
            image:SetLocalScale(item[12], item[13], 1)
            image:SetLocalRotation(0, 0, item[14])
            image:SetAnchoredPosition(item[2], item[3])
            image.imageColor = Color.FromRGBA(item[15], item[16], item[17], item[18])
            image:SetAsLastSibling()
        end
    end)
    if not ok then
        Clear()
        printerr("[GIA绘制] " .. tostring(err))
    end
end

function OnDestroy()
    Clear()
end
'''
