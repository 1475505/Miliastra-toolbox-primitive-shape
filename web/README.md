# Shaper Web 服务

基于 Flask MPA（多页应用）架构的轮廓描边 Web 服务，提供三栏交互式 UI。

## 文件结构

```
web/
├── app.js      # 结果页交互脚本（Canvas 渲染、悬浮/选中、导出等）
├── style.css   # 全局样式（三栏布局、面板、按钮、Canvas 区域等）
└── README.md
server.py        # Flask 服务端（位于项目根目录）
shaper_core.py   # 核心 API 封装（位于项目根目录）
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

- **图片上传**：拖放或点击选择，支持预览
- **图元定义**：可添加多个基础图元（椭圆/矩形），各自指定名称和大小
- **参数配置**：图元大小（3–80）、精度（0–1）、间距（0.5–1）

### 结果页 `/result/<tid>`

- **Canvas 渲染**：自适应缩放，叠加显示原图、遮罩、图元填充/边框
- **交互操作**：
  - 鼠标悬浮高亮 + 信息悬浮框
  - 点击选中，左侧面板展示图元详细参数
  - 右键点击设置新原点
- **显示控制**：可独立开关原图、遮罩、填充、边框图层
- **导出**：JSON 数据导出、PNG 图片导出
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
- JSON / PNG 导出

## API 参数

通过表单或 `shaper_core.py` 传入：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `primitive_size` | int | 15 | 图元基准大小，派生 min_size = size×0.4, max_size = size×2.0 |
| `precision` | float | 0.3 | 轮廓拟合精度 (0=粗略, 1=精细) |
| `spacing` | float | 0.9 | 图元间距系数 |
| `primitives_json` | string | — | 可选，JSON 格式的图元定义列表 |
