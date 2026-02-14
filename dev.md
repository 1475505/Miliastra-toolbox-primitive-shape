# Shaper 开发指南

本文档面向开发人员，记录项目架构、代码位置及扩展开发指南。

---

## 1. 项目架构概览

```
Shaper/
├── server.py              # Flask Web 服务器 + 前端 HTML 模板
├── shaper_core.py        # 图片处理核心 API
├── final_shaper.py       # 底层算法实现 (路径行走、拟合)
├── web/
│   ├── upload.js         # 上传页面前端逻辑
│   ├── app.js           # 结果页面交互逻辑
│   ├── style.css        # 全局样式
│   └── (无独立 HTML 文件，嵌入 server.py)
├── gia/
│   └── json_to_gia.py   # JSON → GIA 文件转换
└── win/                  # Windows 打包配置
```

### 数据流

```
用户上传图片
    │
    ▼
server.py (Flask)
    │
    ▼
shaper_core.process_image()
    │
    ├── 提取 Mask / 距离场 (OpenCV)
    ├── 图元拟合 (final_shaper.py)
    └── 返回 JSON (elements + image_base64)
    │
    ▼
结果页 (app.js) / GIA 导出 (json_to_gia.py)
```

---

## 2. 代码位置速查

| 功能 | 文件 | 关键行号 |
|------|------|----------|
| 上传页面 HTML | server.py | `PAGE_UPLOAD` (~行41-164) |
| 图元选择 UI | server.py | ~行81-130 |
| 前端预设配置 | upload.js | `PRESETS` 常量 (~行15-25) |
| 提交逻辑 | upload.js | `mainForm.onsubmit` (~行151-) |
| 图片处理入口 | shaper_core.py | `process_image()` (~行14) |
| 图元类型过滤 | shaper_core.py | ~行69-86 (`allowed_types`) |
| 核心拟合算法 | final_shaper.py | `fit_beads()` (~行350) |
| GIA 生成 | json_to_gia.py | `convert_json_to_gia_bytes()` (~行404) |
| GIA 旋转处理 | json_to_gia.py | `create_decoration_payload()` (~行295) |

---

## 3. 常见开发任务

### 3.1 新增圆形预设 (以岩元素徽章为例)

假设要新增一种圆形预设「星辉石 0.8×0.8」，type_id 为 `20009999`，Z轴旋转 45°。

#### Step 1: 修改前端 UI (server.py)

在 `PAGE_UPLOAD` 模板中找到圆形选择区域，添加新的 radio 选项：

```html
<!-- 约行 84-96 -->
<label class="radio-chip">
  <input type="radio" name="circle_type" value="star_ore">
  <span>星辉石 0.8×0.8</span>
</label>
```

#### Step 2: 修改前端预设配置 (upload.js)

在 `PRESETS.circle` 中添加新预设：

```javascript
// 约行 15-20
var PRESETS = {
  circle: {
    coin: { w: 1, h: 1, type_id: 10005009, color: '#f59e0b' },
    geo_badge: { w: 0.3, h: 0.3, type_id: 20001285, color: '#a855f7', rot_z: 90, rot_y_add: 90 },
    star_ore: { w: 0.8, h: 0.8, type_id: 20009999, color: '#facc15', rot_z: 45 },  // 新增
    custom: { w: 1, h: 1, type_id: 10005009 }
  },
  // ...
};
```

#### Step 3: GIA 旋转处理 (json_to_gia.py)

如果新预设需要特殊旋转，修改 `convert_json_to_gia_bytes` 函数中的处理逻辑：

```python
# 约行 480-530
# 在椭圆类型处理块中
if shape_type == 'ellipse':
    type_id = element_type_id if element_type_id else 10005009
    # 新增预设的特殊处理
    if type_id == 20009999:  # 星辉石
        final_rot_z = rot_z + 45.0  # 额外旋转
    # ... 其他逻辑
```

#### Step 4: 验证

1. 启动服务 `python server.py`
2. 访问 http://localhost:5555
3. 选择新预设，上传图片测试
4. 下载 GIA 文件验证

---

### 3.2 新增矩形预设

流程与圆形相同，区别在于：

1. **UI**: 在矩形区域添加选项 (`rect_type`)
2. **PRESETS**: 在 `PRESETS.rect` 中添加
3. **GIA**: 在 `shape_type == 'rectangle'` 分支处理

---

### 3.3 禁用图元类型

当前支持禁用矩形（圆形必须保留）。

- **UI**: server.py 中 `rect_type` 添加 `disabled` 选项
- **逻辑**: upload.js 提交时 `if (rectType !== 'disabled')` 才添加矩形
- **后端**: shaper_core.py 处理 `allowed_types = []` 表示空列表

---

### 3.4 修改图元颜色

前端颜色选择器位于 UI 中，选择后通过 `primitives_json` 传递给后端：

```javascript
// upload.js 约行 173
circleColorVal = circleColorInput ? circleColorInput.value : '#f59e0b';
```

颜色在 shaper_core.py 中传递给算法：

```python
# shaper_core.py 约行 71-83
if s == 'circle':
    at_set.add(fs.ShapeType.ELLIPSE)
    if c:
        type_colors[fs.ShapeType.ELLIPSE] = c
```

---

## 4. 后端关键逻辑

### 4.1 图元类型过滤

```python
# shaper_core.py
allowed_types = None
if 'primitives' in config and config['primitives']:
    at_set = set()
    for p in config['primitives']:
        s = p.get('shape')
        if s == 'circle':
            at_set.add(fs.ShapeType.ELLIPSE)
        elif s == 'rect':
            at_set.add(fs.ShapeType.RECTANGLE)
    if at_set:
        allowed_types = list(at_set)
    else:
        allowed_types = []  # 全部禁用
```

### 4.2 GIA 旋转处理

```python
# json_to_gia.py
def create_decoration_payload(guid, name, type_id, parent_guid, pos, scale, rot_z=0.0, rot_y=0.0):
    # ...
    vec_rot.write_float(3, rot_z)  # Z轴
    vec_rot.write_float(2, rot_y)  # Y轴
```

---

## 5. 后续迭代思考

### 5.1 功能扩展

| 方向 | 说明 | 复杂度 |
|------|------|--------|
| 更多预设类型 | 如「风之翼」「岩王爷」等游戏道具 | 低 |
| 自定义 type_id | 允许用户输入任意 GIA type_id | 低 |
| 预设分组 | 将常用预设打包成「冒险者套装」等 | 中 |
| 预设导入/导出 | 用户保存自己的预设配置 | 中 |

### 5.2 算法优化

| 方向 | 说明 | 复杂度 |
|------|------|--------|
| 三角形图元 | tech.md 中提到的扩展 | 高 |
| 自定义图元形状 | 用户上传 SVG 定义形状 | 高 |
| 多层图元 | 前景/背景分别拟合 | 中 |
| 动态参数 | 根据图片自动推荐参数 | 中 |

### 5.3 前端交互

| 方向 | 说明 | 复杂度 |
|------|------|--------|
| 实时预览 | 拖动滑块实时更新效果 | 中 |
| 图元编辑 | 在结果页手动增删改图元 | 高 |
| 批量处理 | 一次上传多张图片 | 中 |
| 历史记录 | 保存之前的处理结果 | 中 |

### 5.4 GIA 相关

| 方向 | 说明 | 复杂度 |
|------|------|--------|
| 经典模式 GIA | 除了超限模式，生成经典模式 GIA | 低 |
| 预设旋转模板 | 不同预设对应不同旋转规则，可配置化 | 低 |
| GIA 预览 | 导入 GIA 后在网页预览效果 | 中 |
| 图层管理 | 多个 GIA 叠加导出 | 中 |

### 5.5 潜在问题

| 问题 | 说明 | 建议 |
|------|------|------|
| 大图处理慢 | 2000x2000+ 图片可能超时 | 增加超时提示，或后台处理 |
| 内存占用 | 多用户并发时内存增长 | 定期清理 tasks 字典 |
| 预设 ID 冲突 | 游戏更新后 type_id 变化 | 预留配置接口，定期更新 |
| 移动端适配 | 左栏 400px 在手机上显示问题 | 响应式布局 |

---

## 6. 调试技巧

### 6.1 前端调试

```javascript
// 在 upload.js 的 onsubmit 中添加
console.log('primitives:', JSON.stringify(primitives));
```

### 6.2 后端调试

```python
# 在 shaper_core.py 中添加
import logging
logging.info(f"primitives: {config.get('primitives')}")
```

### 6.3 GIA 调试

```python
# json_to_gia.py 中添加 verbose=True
gia_bytes = mod.convert_json_to_gia_bytes(json_data=json_data, base_gia_path=base_gia_path, verbose=True)
```

---

## 7. 相关文档

- [tech.md](./tech.md) - 核心技术方案与算法详解
- [user_guide.md](./user_guide.md) - 用户使用指南
- [README.md](./README.md) - 项目简介
