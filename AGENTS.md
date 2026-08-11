# 千星奇域图片拟合工具 (Shaper)

图片拟合工具，用基础图元（椭圆/矩形/三角形）沿轮廓或填充区域来拟合图片，导出为 SVG/PNG/CSS/GIA 格式。

## 目录结构

```
├── server.py                 # Flask Web 服务器 + HTML 模板（单文件 ~1300 行）
├── shaper_core.py            # 核心 API 入口（填充+轮廓，含目标分辨率重定标）
├── fill_shaper.py            # 填充模式引擎 — 随机优化拟合（蒙版/软权重）
├── final_shaper.py           # 轮廓模式引擎 — 路径行走拟合（V6）
├── primitive_backend.py      # Go primitive 后端封装（保留兼容）
├── build_pyc.py              # 编译 .pyc 脚本（将 .py 编译部署用）
├── requirements.txt          # Python 依赖
├── Dockerfile                # Docker 部署
├── AGENTS.md                 # 本文件
├── README.md                 # 项目简介
├── tech.md                   # 核心技术方案与算法详解
├── user_guide.md             # 用户使用指南
├── dev.md                    # 开发指南
│
├── wasm/
│   ├── main.go               # Go→WASM 封装（本地模式拟合引擎，复刻后端流程）
│   ├── go.mod                # 独立 module（replace 指向 third_party/primitive）
│   └── test_node.mjs         # Node 冒烟测试
│
├── web/
│   ├── upload.js             # 上传页逻辑 + 预设 + 本地模式 (~1000 行)
│   ├── local_fit.js          # 本地模式客户端（预处理、结果组装、/register_result）
│   ├── app.js                # 结果页 Canvas 交互
│   ├── style.css             # 全局样式
│   ├── wasm/
│   │   ├── primitive.wasm    # WASM 编译产物（GOOS=js GOARCH=wasm，~4.4MB）
│   │   ├── wasm_exec.js      # Go 官方 JS 桥（随 Go 版本分发，勿手改）
│   │   └── fit_worker.js     # Web Worker（加载 WASM、进度上报）
│   └── README.md             # Web 服务说明
│
├── gia/
│   ├── json_to_gia.pyc       # JSON→GIA 转换（仅 .pyc 发布，源码不便开源）
│   ├── convert_to_classic.py # 超限→经典 GIA 转换
│   ├── convert_to_overlimit.py # 经典→超限 GIA 转换
│   ├── template.gia          # GIA 模板文件
│   └── image_template.gia    # 图片 GIA 模板
│
├── win/
│   ├── app_desktop.py        # pywebview 桌面应用入口
│   ├── build_windows.bat     # Windows 打包脚本
│   ├── requirements.txt      # 桌面应用依赖
│   └── shaper.spec           # PyInstaller 配置
│
├── tools/
│   └── primitive.exe         # Go primitive 可执行文件（Windows）
│
├── demo/                     # 测试图片和结果
└── third_party/              # 第三方代码
```

## 技术栈

- **后端**: Python 3.13+, Flask, OpenCV, NumPy, Shapely, SciPy
- **前端**: 原生 JS + CSS（无框架）, Canvas 2D
- **部署**: Flask 内嵌 HTML 模板（MPA 架构，无前后端分离）
- **桌面**: pywebview (Windows WebView2)

## 核心模式

| 模式 | 引擎 | 拟合方式 | 图元类型 |
|------|------|---------|---------|
| 轮廓 (Outline) | `final_shaper.py` | 路径行走（沿轮廓排列） | 椭圆、矩形 |
| 填充 (Fill) | `fill_shaper.py` | 随机优化 + 爬山（区域内分布） | 圆形、椭圆、矩形、三角形 |

## 路由架构

| 路由 | 说明 |
|------|------|
| `GET /` | 上传页（三栏布局） |
| `POST /submit` | 提交处理，302 → 状态页 |
| `POST /register_result` | 寄存本地（WASM）拟合结果，返回 task_id 复用结果页 |
| `GET /status/<tid>` | 轮询处理状态 |
| `GET /result/<tid>` | 结果页（Canvas 渲染 + 导出） |

## 本地模式（WebAssembly）

- 「本地模式」开关在上传页顶部；仅支持填充模式，单图处理后跳转结果页
- 拟合引擎 = `third_party/primitive` 编译的 WASM（与云端同一算法），在 Web Worker 中运行
- 多核：Go js/wasm 单线程（GOMAXPROCS=1），多核 = 多 WASM 实例。**单图并行**采用与云端 `primitive -j N` 相同的 Step 内候选并行（`wasm/main.go` 分步 API：init/state/search/apply/finish；`web/local_fit.js` 的 `runDistributed` 编排，每 Step 各实例搜索候选、主实例应用最优），池大小 min(4, hardwareConcurrency)，实测 4 核 ~3.3x 提速
- 重新构建：`cd wasm && GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o ../web/wasm/primitive.wasm .`（需 Go ≥ 1.25，`wasm_exec.js` 必须与编译版本一致）
- 本地结果经 `web/local_fit.js` 组装成与后端一致的结构后 POST `/register_result`，结果页与导出链路完全复用
- 「输出尺寸」为二选一：按比例缩放（image_scale）或指定分辨率（target_width/height，此时缩放按 1.0）；指定分辨率模式上传图片后自动填充当前图片分辨率

## 开发注意事项

1. 前端 HTML 模板全部嵌入在 `server.py` 中，修改前端需同时改 server.py 和 `web/` 下的 JS/CSS
2. `gia/json_to_gia.py` 仅以 `.pyc` 形式发布，无源码；转换逻辑可通过 `convert_to_classic.py` / `convert_to_overlimit.py` 参考
3. 预设配置（图元 ID、尺寸等）定义在 `web/upload.js` 的 `PRESETS` 常量中
4. 服务默认端口 5555
5. Go primitive 可执行文件需放在 `tools/` 下，缺失时处理会失败
