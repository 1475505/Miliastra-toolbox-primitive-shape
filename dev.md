# 开发手册

面向接手本仓库的开发者：架构、数据流、关键代码位置与常见改动点。

> 行号为 2026-10 现状，改动后请以函数名为准检索。

---

## 1. 架构总览

```
Shaper/
├── server.py             # Flask 服务器 + 内嵌 HTML 模板（上传页 / 结果页）+ CLI
├── shaper_core.py        # 图片处理核心 API（入口、蒙版、目标分辨率重定标）
├── fill_shaper.py        # 填充模式引擎（随机优化拟合 + extract_mask）
├── primitive_backend.py  # Go 版 primitive 子进程封装（图元搜索、PNG alpha 软权重）
├── lua_export.py         # 拟合结果 → Lua 客户端绘制脚本（PALETTE / ELEMENTS）
├── gia_lua.py            # 素材组 GIA → Lua 客户端绘制脚本（ROOT / ELEMENTS）
├── build_pyc.py          # 编译 .pyc（对 gia/json_to_gia.py 额外产出入库的 pyc）
├── web/                  # 前端：upload.js / app.js / local_fit.js / style.css / wasm/
├── gia/                  # GIA 构建与互转
│   ├── json_to_gia.py        # 构建器源码（.gitignore 忽略，不入库）
│   ├── json_to_gia.pyc       # 构建器编译产物（入库，生产环境只需它）
│   ├── image_template.gia    # 图片素材组模板（与 editor-webui 共用同一份）
│   ├── convert_to_classic.py # 超限 → 经典
│   └── convert_to_overlimit.py # 经典 → 超限
├── wasm/                 # primitive WASM 的 Go 入口（main.go）
└── win/                  # Windows 打包（PyInstaller）
```

### 数据流

```
用户上传图片
    │
    ▼
server.py /submit ──► shaper_core.process_image()
    │                     └─ process_image_fill()
    │                          ├─ 前景蒙版 / alpha 软权重
    │                          └─ primitive_backend.fit_image_with_primitive()  (Go 子进程)
    │                                └─ fill_shaper.results_to_elements()
    │                                       └─ element: {type, center, size, rotation, color, alpha, packed_color}
    ▼
结果页 /result/<tid>  (app.js)
    ├─ 超限 GIA   /download_overlimit_gia  → server._convert_result_to_gia_bytes() → gia/json_to_gia.py
    ├─ 经典 GIA   /download_classic_gia    → 同上 + convert_to_classic
    ├─ Lua        /download_lua            → lua_export.py
    └─ JSON / CSS / SVG / PNG              → 纯前端生成（app.js）
```

**本地模式（默认）**：`DEFAULT_FIT_MODE` 默认 `local`，拟合在浏览器内用 WASM 完成，
`web/local_fit.js` 把元素 JSON `POST /register_result` 寄存，之后复用同一条结果页 / 导出链路。
上传页的「本地模式」开关状态记在 `localStorage.shaper.localMode`。

**素材组 GIA 的两条独立通路**：上传页「GIA转换」支持超限 ↔ 经典互转，以及素材组 GIA → Lua 脚本。

---

## 2. 代码位置速查

| 功能 | 文件 | 位置 |
|------|------|------|
| 上传页 HTML | server.py | `PAGE_UPLOAD` |
| 结果页 HTML | server.py | `PAGE_RESULT` |
| 结果元素 → GIA JSON | server.py | `_convert_result_to_gia_bytes()` (~行110) |
| CLI 配置 / 执行 | server.py | `_build_cli_config()` / `_run_cli()` (~行204 / 232) |
| 提交与任务调度 | server.py | `submit()` (~行815) |
| 本地结果寄存 | server.py | `register_result()` (~行992) |
| 处理入口 | shaper_core.py | `process_image()` (~行225) |
| 填充处理 | shaper_core.py | `process_image_fill()` (~行230) |
| 目标分辨率重定标 | shaper_core.py | `_rescale_fill_output()` (~行156) |
| 前景蒙版提取 | fill_shaper.py | `extract_mask()` (~行38) |
| 拟合主循环 | fill_shaper.py | `fit_primitives()` (~行618) |
| 拟合结果 → 元素 | fill_shaper.py | `results_to_elements()` (~行757) |
| ARGB 打包 | fill_shaper.py | `_pack_color()` (~行364) |
| Go 后端封装 | primitive_backend.py | `fit_image_with_primitive()` (~行478) |
| SVG 结果解析 | primitive_backend.py | `parse_primitive_svg()` (~行338) |
| GIA 构建入口 | gia/json_to_gia.py | `convert_json_to_gia_bytes()` (~行1321) |
| 图片模式写节点 | gia/json_to_gia.py | `_convert_image_mode()` (~行1450) |
| alpha / packed_color 归一化 | gia/json_to_gia.py | `_alpha_to_int()` / `_color_to_packed()` (~行1091 / 1114) |
| 前端：CSS 导出 | web/app.js | `buildCssExportText()` (~行541) |
| 前端：统一 config | web/upload.js | `buildUnifiedConfig()` (~行422) |
| 前端：本地拟合流程 | web/upload.js | `processSingleLocal()` (~行477) |
| 前端：WASM 结果 → 元素 | web/local_fit.js | `resultsToElements()` (~行356) |
| 前端：结果寄存 | web/local_fit.js | `registerResult()` (~行723) |

---

## 3. 约定与坑

### 3.1 `packed_color` 与 alpha

`packed_color` 是 ARGB 打包整数：`(alpha << 24) | (r << 16) | (g << 8) | b`。
元素同时携带 `color`（`#rrggbb` 字符串）与 `alpha`（0–1 比例），GIA 构建器以
`_color_to_packed(color, alpha, fallback)` 重算 `packed_color`。

**alpha 的量纲必须靠数值区间判断，不能靠类型**：浏览器 `JSON.stringify(1.0)` 得到的是整数 `1`，
若用 `isinstance(x, float)` 判断量纲，100% 不透明会被误读成 1/255（几乎全透明）。
`_alpha_to_int()` 统一按 [0,1] 视作比例；`server._convert_result_to_gia_bytes()` 也在入口
把 alpha 归一化成 `float` 作为第二道保险。

**缺字段时不要写 `None`**：构建器用 `int(element.get('packed_color', 默认值))` 取值，
Python 在「键存在但值为 `None`」时不会采用默认值，会直接 `int(None)` 抛 `TypeError`。
省略键才能触发 `0x80FFFFFF`（alpha=128）回落——这也与前端画布 / CSS 导出的 `0.5` 回落一致。

### 3.2 元素顺序

`json_to_gia.py::_order_elements_for_image_mode()` 对元素列表做**整体反转**，
契约是：调用方按「底 → 顶」传入（背景在前、最上层在后），构建器负责转成 GIA 的存储顺序。
`shaper_core.process_image_fill()` 就是这么产生的（白底 `insert(0, ...)`）。

### 3.3 GIA 构建器

`gia/json_to_gia.py` 与 `Miliastra-image-editor-webui/backend/vendor/gia/json_to_gia.py` 同源，
是同一份通用构建器（同时支持 `MODE_DECORATION` / `MODE_IMAGE`，本仓库只用 `MODE_IMAGE`）。

仓库设计是**源码本地、pyc 入库**（`.gitignore`：`gia/*` 忽略、只放行 `gia/json_to_gia.pyc`）。
因此改了构建器必须重编 pyc，且**必须在 Python 3.13 下编译**（生产镜像按 pyc 的 magic number 加载）：

```bash
python build_pyc.py     # 需要 Python 3.13
```

生产环境只有 pyc（`import json_to_gia` 会优先 `.py`，没有源码时回落 `.pyc`），
改动后请用「隐藏 `gia/json_to_gia.py` 再跑一遍导出」的方式验证 pyc 单独可用。

### 3.4 透明度相关的三条回落约定

图元缺 `alpha` 时，三处都回落 `0.5`：前端画布渲染、CSS 导出（`app.js`）、GIA 构建器
（`0x80FFFFFF`）。改动其一请同步其余两处。

---

## 4. 常见开发任务

### 4.1 调整拟合参数

- UI / 表单：`server.py` 的 `PAGE_UPLOAD`，提交字段在 `submit()` 里解析成 `cfg`
- 前端本地模式：`web/upload.js::buildUnifiedConfig()`（本地 / 云端共用的唯一 config 来源）
- 算法默认值：`fill_shaper.FillConfig`、`shaper_core.process_image_fill()`

### 4.2 支持新的图元形状

1. `fill_shaper.ShapeType` 增加类型，`DEFAULT_IMAGE_ASSET_REFS` 补映射
2. `primitive_backend.SHAPE_MODE_MAP` 补 Go primitive 的模式号
3. `gia/json_to_gia.py::_convert_image_mode()` 的尺寸分支补 `size` 字段映射
4. 前端 `app.js` 的 `normalizeType()` / 渲染 / CSS 导出、`local_fit.js` 同步

### 4.3 新增导出格式

在 `server.py` 加路由（参考 `download_lua`），结果数据统一从 `tasks[tid]["result"]` 取；
纯前端生成的格式（JSON / CSS / SVG / PNG）参考 `web/app.js`。

### 4.4 调试

```python
# 后端：打开构建器 verbose，打印每个生成的 image 节点
mod.convert_json_to_gia_bytes(json_data=..., base_gia_path=..., mode=mod.MODE_IMAGE, verbose=True)

# 抽取导出 GIA 的子节点 packed_color 做断言（配合 Flask test client）
python - <<'EOF'
import server
c = server.app.test_client()
resp = c.get("/download_overlimit_gia/<tid>?origin_x=0&origin_y=0")
open("/tmp/out.gia", "wb").write(resp.data)
EOF
```

前端在浏览器控制台里可直接看 `window.RESULT`（结果页的元素数组）。

---

## 5. 后续迭代方向

| 方向 | 说明 | 复杂度 |
|------|------|--------|
| 自定义图元形状 | 用户上传 SVG 定义形状 | 高 |
| 多层图元 | 前景 / 背景分别拟合 | 中 |
| 实时预览 | 拖动滑块实时更新效果 | 中 |
| 批量处理 | 一次上传多张图片 | 中 |
| GIA 预览 | 导入 GIA 后在网页预览效果 | 中 |
| 图元编辑 | 在结果页手动增删改图元 | 高 |

已知风险：

- 2000×2000+ 大图拟合耗时较长，需要超时提示或后台处理
- `tasks` 字典按时间清理（`cleanup()`），多用户并发时注意内存
- 生产部署依赖 Python 3.13（pyc 版本），换基础镜像需重新编译 `gia/json_to_gia.pyc`

---

## 6. 相关文档

- [tech.md](./tech.md) - 核心技术方案
- [user_guide.md](./user_guide.md) - 用户使用指南
- [README.md](./README.md) - 项目简介
