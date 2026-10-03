# 技术方案

Shaper 是一个**图片素材组拟合工具**：把一张图片用基础图元（圆形/椭圆、矩形、三角形）在
前景区域内迭代拟合，产出可直接导入千星奇域的**图片素材组 GIA**、**Lua 客户端绘制脚本**，
以及可继续编辑的 **JSON / CSS / SVG / PNG**。

---

## 使用方式

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
python server.py                                        # Web 界面
python server.py --cli --input demo.png --output o.gia  # CLI 直接导出 GIA
python server.py --cli --input demo.png --export lua --output o.lua
```

---

## 依赖

| 库 | 用途 |
|----|------|
| `opencv-python` | 图像读写、Mask 提取、预览渲染 |
| `numpy` | 数值计算、向量运算 |
| `flask` | Web 服务 |

---

## 文件结构

```
├── server.py               # Flask Web 服务器 + 前端 HTML 模板 + CLI
├── shaper_core.py          # 图片处理核心 API（含目标分辨率重定标）
├── fill_shaper.py          # 填充模式引擎（随机优化拟合 + extract_mask）
├── primitive_backend.py    # Go primitive 后端（图元搜索、alpha 软权重）
├── lua_export.py           # 拟合结果 -> Lua 客户端绘制脚本（PALETTE/ELEMENTS）
├── gia_lua.py              # 素材组 GIA -> Lua 客户端绘制脚本（ROOT/ELEMENTS）
├── build_pyc.py            # 编译 .pyc 脚本
├── web/
│   ├── upload.js           # 上传页面前端逻辑
│   ├── local_fit.js        # 本地模式（WASM）客户端
│   ├── app.js              # 结果页面交互逻辑
│   ├── style.css           # 全局样式
│   └── wasm/               # WASM 产物、Go JS 桥、拟合 Worker
├── gia/
│   ├── json_to_gia.py      # JSON -> GIA 文件转换（构建器源码，本地不入库）
│   ├── json_to_gia.pyc     # 上面源码的编译产物（入库）
│   ├── image_template.gia  # GIA 模板
│   ├── convert_to_classic.py   # 超限模式 GIA -> 经典模式 GIA
│   └── convert_to_overlimit.py # 经典模式 GIA -> 超限模式 GIA
├── win/                    # Windows 打包配置
├── wasm/                   # primitive WASM 的 Go 入口
└── demo/                   # 测试图片与结果
```

---
## 填充模式 (Fill Mode)

填充模式是当前唯一的拟合方式，使用 `fill_shaper.py`（配合 `primitive_backend.py` 调用 Go 版 primitive 做图元搜索）实现基于蒙版/软权重的图元拟合。

> 历史说明：早期版本还提供「装饰物拟合（轮廓模式）」——沿轮廓路径行走排列图元（引擎为 `final_shaper.py`）。
> 该模式已整块移除，需要时请从 git 历史查阅对应实现与文档。

填充模式是 V2 新增的核心功能，使用 `fill_shaper.py` 实现基于蒙版/软权重的图元拟合，替代了之前依赖 Go 版 primitive 的方案。

### 核心算法

填充模式采用**迭代随机优化**（类似 Hill Climbing）：

1. **误差图采样**：计算当前画布与目标图片的误差，按误差权重随机选取焦点位置
2. **候选生成**：在焦点附近随机生成多个候选图元（位置、大小、旋转）
3. **颜色求解**：对每个候选，用加权最小二乘法求解最优颜色
4. **评分**：计算 `delta = Σ(new_error² - old_error²) × weight + spill_penalty × 溢出面积`
5. **爬山优化**：对最优候选做 48 轮随机扰动，保留最优改进
6. **画布更新**：将最优图元合成到画布上

### 软权重 (Coverage Weights)

V2 的核心改进是引入了**软权重**机制：

- **蒙版模式**（JPG 输入）：`coverage_weights = mask`（0 或 1 的硬权重）
- **PNG 模式**：`coverage_weights = alpha / 255`（0~1 的软权重）

软权重使得：
- 颜色求解时，半透明区域的像素权重较低，不会过度拟合边缘
- 评分时，图元溢出到权重为 0 的区域会受到 `spill_penalty` 惩罚
- 渲染输出时，背景区域保持透明

### PNG 透明模式

启用 `enable_png_mode` 后：
- 不使用二值蒙版，直接用 alpha 通道作为软权重
- 渲染输出为 RGBA，背景区域 alpha=0
- 图元 alpha 会根据所在位置的 alpha 权重自动调整
- 适合需要保持透明背景的场景（如游戏素材）

### 图元类型

| 类型 | 参数 | 光栅化 |
|------|------|--------|
| Circle | cx, cy, rx, ry, angle | 抗锯齿椭圆 |
| Rect | cx, cy, hw, hh, angle | 抗锯齿旋转矩形 |
| Triangle | cx, cy, size, angle | 抗锯齿等边三角形 |

所有图元的光栅化都使用**亚像素抗锯齿**：边缘像素的 alpha 值按距离平滑过渡，避免锯齿。

### 输出格式

填充模式输出的每个图元包含：

```jsonc
{
  "type": "circle",         // circle | rect | triangle
  "cx": 123.4,              // 中心 x 坐标
  "cy": 56.7,               // 中心 y 坐标
  "rx": 8.5,                // 椭圆 x 半轴 / 矩形半宽
  "ry": 6.2,                // 椭圆 y 半轴 / 矩形半高
  "angle": 15.0,            // 旋转角度（度）
  "color": "#ff8040",       // BGR 转 HEX 颜色
  "alpha": 0.85,            // 拟合 alpha（已缩放）
  "packed_color": 0xD9FF8040  // ARGB 打包颜色
}
```

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_primitives` | 400 | 拟合图元数量 |
| `candidates` | 24 | 每轮候选数量 |
| `hill_climb_iter` | 48 | 爬山优化迭代次数 |
| `min_size` | 自动 | 最小图元尺寸（像素） |
| `max_size` | 自动 | 最大图元尺寸（像素） |
| `min_mask_coverage` | 0.55/0.12 | 最低蒙版覆盖率（蒙版/PNG 模式） |
| `spill_penalty` | 10000/12000 | 溢出惩罚系数 |
| `alpha_range` | (0.15, 1.0) | 图元透明度范围 |
| `allowed_types` | ["circle"] | 允许的图元类型 |

---

## Lua 导出（客户端绘制脚本）

把拟合结果或已有素材组转成可直接挂进游戏的客户端脚本，两条通路共用同一份客户端运行时（`OnStart` 绘制、`OnDestroy` 清理、重复进入先清理再重建）：

| 源 | 生成器 | 版式 | 写入内容 |
|------|--------|------|----------|
| 拟合结果 | `lua_export.py` | `PALETTE` + `ELEMENTS`（8 字段） | 调色板去重后写 `{kind, cx, cy, w, h, rotZ, colorIndex, alpha[, rotX, rotY]}`（X/Y 旋转非零时追加）；矩形 / 圆形 / 三角形对应静态图片 `100001` / `100002` / `100003` |
| 素材组 GIA | `gia_lua.py` | `ROOT` + `ELEMENTS`（18 字段） | 逐图片资产写 18 字段（图片资产、位置、尺寸、pivot、anchor、scale、rotZ、RGBA），X/Y 旋转非零时追加 `rotX, rotY`，保留素材组布局 |

坐标约定：以原图左下角为原点，X 向右、Y 向上，单位为原图像素，运行时再乘 `BASE_SCALE` 或画布自适应缩放。三角形轴心取质心 `(0.5, 1/3)`，其余形状取中心。

脚本头部自带使用说明：`IMAGE_PREFAB_ID` 必填为「仅存为模板」图片控件的**控件模板索引 ID**（不是图片资产 ID），脚本挂到专用空客户端容器节点后在 `OnStart` 绘制；每个图元实例化一个控件，控件容量与真机显示需在运行预览和真机确认。

---

## 感谢

https://github.com/script-1024/genshin-miliastra-file-format
