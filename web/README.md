# Shaper Web 服务

基于 Flask MPA（多页应用）架构的图元拟合 Web 服务（填充模式 + 轮廓模式），提供三栏交互式 UI；可开启浏览器内 WASM 本地模式。

## 文件结构

```
web/
├── upload.js     # 上传页逻辑（图元/参数表单、预设、本地模式开关、GIA 转换工具）
├── local_fit.js  # 本地模式：WASM 分布式拟合、结果组装、/register_result
├── app.js        # 结果页交互脚本（Canvas 渲染、悬浮/选中、导出等）
├── clipboard.js  # 浏览器 / WebView2 共用的剪贴板封装
├── style.css     # 全局样式（三栏布局、面板、按钮、Canvas 区域等）
├── wasm/         # primitive.wasm、wasm_exec.js、fit_worker.js
└── README.md
server.py        # Flask 服务端（位于项目根目录）
shaper_core.py   # 核心 API 封装（位于项目根目录）
lua_export.py    # 拟合结果 → Lua 客户端绘制脚本（位于项目根目录）
gia_lua.py       # 素材组 GIA → Lua 客户端绘制脚本（位于项目根目录）
```

## 快速启动

```bash
# 1. 安装依赖
pip install flask opencv-python-headless numpy scipy shapely

# 2. 启动服务
python server.py

# 3. 浏览器访问
open http://127.0.0.1:5555
```

服务默认监听 **5555** 端口。

## 架构说明

采用 MPA（Multi-Page Application）架构，所有页面导航使用原生浏览器行为，不依赖 XHR/fetch：

| 步骤 | 路由 | 说明 |
|------|------|------|
| 1 | `GET /` | 上传页 — 三栏布局，图片上传、图元定义、参数配置 |
| 2 | `POST /submit` | 提交表单，后端创建任务，302 重定向到状态页 |
| 3 | `GET /status/<tid>` | 状态页 — 带 `<meta http-equiv="refresh">` 每秒轮询 |
| 4 | `GET /result/<tid>` | 结果页 — 服务端注入 `window.RESULT`，前端渲染 Canvas |

## 页面功能

### 上传页 `/`

- **本地模式**：顶部开关；开启后拟合在浏览器内用 WebAssembly 完成（仅填充模式、单图处理），不上传图片到服务端
- **图片上传**：拖放、点击或 Ctrl+V 粘贴，支持预览
- **图元类型**：勾选启用圆形 / 矩形 / 三角形（默认仅圆形）
- **填充参数**：图元数量（40–1200 滑杆，最多 3000）、输出尺寸（按比例缩放 / 指定分辨率 16–4096）、透明度、PNG 模式
- **轮廓参数**：图元大小（3–200）、间距（0.1–1.0）、精度（0–1），并可维护装饰物元件列表
- **GIA 模式转换**：独立标签页，超限 ↔ 经典模式互转，或素材组 GIA 转 Lua 绘制脚本
- **使用说明**：页面内折叠面板，含流程、注意事项与教程链接

### 结果页 `/result/<tid>`

- **Canvas 渲染**：自适应缩放，叠加显示原图、遮罩、图元填充/边框
- **交互操作**：
  - 鼠标悬浮高亮 + 信息悬浮框
  - 点击选中，左侧面板展示图元详细参数
  - 右键点击设置新原点
- **显示控制**：可独立开关原图、遮罩、填充、边框图层
- **导出**：按用途分组——素材组资产（超限 / 经典模式 GIA）、客户端脚本（Lua，可复制）、进一步编辑（JSON / CSS，可复制；SVG / PNG）
- **参数调整**：可在结果页直接修改参数重新处理

## 前端文件说明

### style.css

全局样式，包含：
- 顶栏（`.topbar`）
- 三栏布局（`.app-layout` → `.panel-left` / `.canvas-area` / `.panel-right`）
- 图元卡片（`.prim-card`）、配置行（`.config-row`）
- Canvas 悬浮框（`.tooltip`）、图元信息表格（`.elem-info`）
- 加载动画（`.loading-overlay` / `.spinner`）

### app.js

结果页专用脚本，读取服务端注入的 `window.RESULT` 数据：
- 图片加载与自适应缩放
- Canvas 分层渲染（底图 → 遮罩 → 填充 → 边框 → 原点十字线 → 高亮）
- 旋转感知的椭圆/矩形碰撞检测
- 原点拖拽与坐标转换
- 导出：GIA（超限/经典）、Lua、JSON / CSS / SVG / PNG（JSON、CSS、Lua 支持复制到剪贴板）

## API 参数

通过表单或 `shaper_core.py` 传入：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | string | `fill` | 处理模式：`fill`（填充）/ `outline`（轮廓） |
| `num_primitives` | int | 400 | 填充模式拟合图元数量 |
| `image_scale` | float | 1.0 | 填充模式图片缩放（指定分辨率时按 1.0） |
| `output_alpha` | float | 100 | 填充模式透明度百分比 |
| `detail_scale` | float | 1.0 | 填充模式细节缩放 |
| `mask_threshold` | int | 127 | 填充模式透明 PNG 的 alpha 阈值 |
| `enable_png_mode` | bool | false | 填充模式保留 PNG 透明背景 |
| `primitive_size` | int | 15 | 轮廓模式图元基准大小，派生 min_size = size×0.4, max_size = size×2.0 |
| `precision` | float | 0.3 | 轮廓拟合精度 (0=粗略, 1=精细) |
| `spacing` | float | 0.9 | 轮廓模式图元间距系数 |
| `primitives_json` | string | — | 可选，JSON 格式的图元定义列表 |
