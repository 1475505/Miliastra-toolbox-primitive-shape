import json
import os
import sys
import threading
import time
import traceback
import uuid
from urllib.parse import quote

from flask import Flask, Response, redirect, render_template_string, request, send_from_directory


if getattr(sys, "frozen", False):
    if hasattr(sys, "_MEIPASS"):
        BASE_DIR = sys._MEIPASS
    else:
        BASE_DIR = os.path.dirname(sys.executable)
        if os.path.exists(os.path.join(BASE_DIR, "_internal")):
            BASE_DIR = os.path.join(BASE_DIR, "_internal")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import shaper_core


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web"), static_url_path="/web")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

tasks = {}


def cleanup():
    now = time.time()
    for key in [key for key, value in tasks.items() if now - value.get("ts", 0) > 1800]:
        del tasks[key]


def _derive_upload_image_name(filename):
    if not filename:
        return ""
    base_name = os.path.basename(str(filename).strip())
    stem, _ = os.path.splitext(base_name)
    return stem.strip()


def _export_basename(image_name):
    normalized = _derive_upload_image_name(image_name)
    if normalized:
        return normalized
    return (image_name or "").strip() or "shaper_result"


def _attachment_filename(filename):
    safe_ascii = "".join(ch if ch.isascii() and ch not in {'"', "\\"} else "_" for ch in filename) or "download"
    return f"attachment; filename={safe_ascii}; filename*=UTF-8''{quote(filename)}"


PAGE_UPLOAD = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Primitive Shape Fitter</title>
  <link rel="stylesheet" href="/web/style.css">
</head>
<body class="page-upload">
  <header class="topbar">
    <div class="topbar-left">
      <a href="/" style="text-decoration:none;"><h1>图片图元拟合</h1></a>
      <span class="topbar-subtitle">默认填充模式 · 默认仅圆形</span>
      <a href="#" id="outlineLink" class="topbar-link topbar-link-subtle">装饰物</a>
    </div>
    <div class="topbar-right">
      <a href="https://github.com/1475505/Miliastra-toolbox-primitive-shape" target="_blank" class="topbar-link">仓库</a>
      <span class="topbar-status active">就绪</span>
    </div>
  </header>

  <div class="app-layout">
    <aside class="panel panel-left">
      <form id="mainForm" action="/submit" method="POST" enctype="multipart/form-data">
        <input type="hidden" name="mode" id="modeInput" value="fill">
        <input type="hidden" name="primitives_json" id="primJson">

        <section class="panel-section">
          <h3>输入图片</h3>
          <div id="dropZone" class="drop-zone">
            <div class="drop-zone-content">
              <span class="drop-icon">图片</span>
              <p>点击、拖拽或 <strong>Ctrl+V</strong> 粘贴图片</p>
            </div>
            <input type="file" id="fileInput" name="image" accept="image/*" required hidden>
            <img id="prev" class="preview-img" hidden>
            <span id="fname" class="file-name"></span>
            <div id="uploadReady" class="upload-ready" hidden>已选择图片</div>
          </div>
          <p class="hint">支持 PNG / JPG / WEBP。仅在开启 PNG 模式时才会直接使用透明通道；否则会先铺白底，再生成遮罩。</p>
        </section>

        <!-- 图元 / 装饰物配置 -->
        <section class="panel-section" id="shapeTypeSection">
          <h3 id="shapeSectionTitle">图元类型</h3>

          <div id="fillShapeSection">
            <div class="shape-checks">
              <label class="shape-check active">
                <input type="checkbox" name="shape_circle" id="shapeCircle" checked>
                <span class="shape-icon">○</span>
                <span>圆形</span>
              </label>
              <label class="shape-check">
                <input type="checkbox" name="shape_rect" id="shapeRect">
                <span class="shape-icon">□</span>
                <span>矩形</span>
              </label>
              <label class="shape-check" id="triangleCheckLabel">
                <input type="checkbox" name="shape_triangle" id="shapeTriangle">
                <span class="shape-icon">△</span>
                <span>三角形</span>
              </label>
            </div>
            <p class="hint" id="shapeHint">默认只启用圆形；需要时再叠加矩形或三角形。</p>
          </div>

          <div id="primitiveListSection" hidden>
            <div class="outline-intro">
              <strong>装饰物元件列表</strong>
              <span>装饰物模式会优先使用这里的元件参数，生成结果时保留类型 ID、元件类型和旋转设置。</span>
            </div>
            <div class="primitive-toolbar">
              <button type="button" id="addCirclePrimitiveBtn" class="btn-chip">+ 圆形元件</button>
              <button type="button" id="addRectPrimitiveBtn" class="btn-chip">+ 矩形元件</button>
            </div>
            <p class="hint" id="primitiveCountHint">建议至少保留一种元件类型。</p>
            <div id="primitiveList" class="primitive-list"></div>
            <div id="primitiveEmpty" class="primitive-empty" hidden>
              <p>还没有装饰物元件</p>
              <span>先添加一个元件，再选择预设或手动填写参数。</span>
            </div>
          </div>
        </section>

        <div id="fillParams">
          <section class="panel-section">
            <h3>拟合参数</h3>
            <div class="param-item">
              <div class="param-head">
                <span class="param-title">图元数量</span>
                <span id="numPrimsVal" class="val-tag">400</span>
              </div>
              <p class="param-desc">越多越细，但耗时也更高。</p>
              <input type="range" name="num_primitives" id="numPrims" min="40" max="2000" step="10" value="400">
              <input type="number" name="num_primitives_manual" id="numPrimsManual" min="40" max="3000" value="400" class="num-input" style="margin-top:8px;width:100%;">
            </div>

            <div class="param-item">
              <div class="param-head">
                <span class="param-title">图片缩放</span>
                <span id="imageScaleVal" class="val-tag">1.0</span>
              </div>
              <p class="param-desc">1.0 表示导出尺寸与原图分辨率一致。</p>
              <input type="range" name="image_scale" id="imageScale" min="0.5" max="4" step="0.1" value="1.0">
            </div>

            <div class="param-item">
              <div class="param-head">
                <span class="param-title">透明度</span>
                <span id="outputAlphaVal" class="val-tag">100%</span>
              </div>
              <p class="param-desc">100% 表示不透明；完全透明0%。</p>
              <input type="range" name="output_alpha" id="outputAlpha" min="0" max="100" step="5" value="100">
            </div>

            <div class="param-item">
              <div class="param-head">
                <span class="param-title">PNG 模式</span>
                <span class="val-tag">可选</span>
              </div>
              <p class="param-desc">默认关闭。仅对带透明通道的 PNG 生效；开启后直接按透明通道拟合并保留透明背景，关闭时会改为白底加遮罩流程。</p>
              <label style="display:flex;align-items:center;gap:8px;font-size:14px;color:var(--text-main);cursor:pointer;">
                <input type="checkbox" name="enable_png_mode" id="enablePngMode">
                <span>启用 PNG 模式</span>
              </label>
            </div>
          </section>
        </div>

        <div id="outlineParams" hidden>
          <section class="panel-section">
            <h3>轮廓参数</h3>
            <div class="param-item">
              <div class="param-head">
                <span class="param-title">图元大小</span>
                <span id="olPrimSizeVal" class="val-tag">30</span>
              </div>
              <p class="param-desc">轮廓模式下每个图元的基础尺寸。</p>
              <input type="range" name="ol_primitive_size" id="olPrimSize" min="3" max="200" step="1" value="30">
            </div>

            <div class="param-item">
              <div class="param-head">
                <span class="param-title">间距</span>
                <span id="olSpacingVal" class="val-tag">0.9</span>
              </div>
              <p class="param-desc">图元之间的间距比例，越大越密。</p>
              <input type="range" name="ol_spacing" id="olSpacing" min="0.1" max="1.0" step="0.05" value="0.9">
            </div>

            <div class="param-item">
              <div class="param-head">
                <span class="param-title">精度</span>
                <span id="olPrecisionVal" class="val-tag">0.3</span>
              </div>
              <p class="param-desc">拟合精度，越高越精细但更慢。</p>
              <input type="range" name="ol_precision" id="olPrecision" min="0.0" max="1.0" step="0.05" value="0.3">
            </div>
          </section>
        </div>

        <section class="panel-section section-submit">
          <button type="submit" id="btnSubmit" class="btn-primary">开始处理</button>
        </section>
      </form>
    </aside>

    <aside class="panel panel-right">
      <section class="panel-section guide-card">
        <h3>流程</h3>
        <ol class="steps">
          <li>上传图片</li>
          <li>确认图元和算法参数</li>
          <li>生成预览并导出 GIA / JSON / PNG</li>
        </ol>
        <ul class="tips">
          <li>⚠️ 开发中 (demo)，有很多 bug，已知目前对三角形和矩形支持不友好，优化中</li>
          <li>PNG 图片默认会将透明区域与白色背景混合。如需保留透明背景，请在参数中开启「PNG 模式」。</li>
          <li>原本的装饰物模式可能因为新增特性导致不可用</li>
          <li>遮罩待优化</li>
          <li>GIA 暂时只支持超限模式资产.</li>
          <li>用户 QQ 群：1007538100</li>
        </ul>
      </section>
    </aside>
  </div>

  <script src="/web/upload.js?v=23"></script>
</body>
</html>"""


PAGE_STATUS = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="5">
  <link rel="stylesheet" href="/web/style.css">
</head>
<body>
  <div class="loading-overlay">
    <div class="spinner"></div>
    <h2 style="margin-top:20px;font-weight:600;color:var(--text-main)">处理中 ({{ elapsed }}s)</h2>
    <p style="margin-top:8px;font-size:13px;color:var(--text-dim)">任务 ID: {{ task_id }}</p>
  </div>
</body>
</html>"""


PAGE_RESULT = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Primitive Shape Fitter Result</title>
  <link rel="stylesheet" href="/web/style.css">
  <script>
    var RESULT={{ result_json|safe }};
    var TASK_CFG={{ config_json|safe }};
    var TASK_ID="{{ task_id }}";
    var TASK_IMAGE_NAME={{ image_name_json|safe }};
  </script>
</head>
<body>
  <header class="topbar">
    <div class="topbar-left">
      <a href="/" style="text-decoration:none;"><h1>图片图元拟合</h1></a>
      <span class="topbar-subtitle" id="modeLabel">填充拟合</span>
    </div>
    <div class="topbar-right">
      <span id="statusText" class="topbar-status active">完成 · {{ count }} 图元 · {{ elapsed }}s</span>
      <a href="/" class="btn-primary" style="text-decoration:none">新建</a>
    </div>
  </header>

  <div class="app-layout">
    <aside class="panel panel-left">
      <section class="panel-section">
        <h3>导出</h3>
        <button id="btnExportJSON" class="btn-sm">导出 JSON</button>
        <button id="btnExportPNG" class="btn-sm">导出 PNG</button>
        <button id="btnExportGIAOverlimit" class="btn-sm">导出 GIA</button>
      </section>

      <section class="panel-section">
        <h3>原点</h3>
        <p class="hint">右键画布也可以重设原点。</p>
        <div class="config-row">
          <label>X</label>
          <input type="number" id="originX" value="0" step="0.1" class="num-input">
          <label style="margin-left:8px">Y</label>
          <input type="number" id="originY" value="0" step="0.1" class="num-input">
        </div>
        <button id="btnResetOrigin" class="btn-sm">重置为图片中心</button>
      </section>

      <section class="panel-section">
        <h3>统计</h3>
        <div class="elem-info"><table>
          <tr><td>模式</td><td id="statMode">—</td></tr>
          <tr><td>总数</td><td id="statTotal">—</td></tr>
          <tr><td>圆形</td><td id="statEllipse">—</td></tr>
          <tr><td>矩形</td><td id="statRect">—</td></tr>
          <tr><td>三角形</td><td id="statTriangle">—</td></tr>
          <tr><td>图片尺寸</td><td id="statImgSize">—</td></tr>
        </table></div>
      </section>

      <section class="panel-section" id="retrySectionFill">
        <h3>重新处理</h3>
        <form action="/retry/{{ task_id }}" method="POST">
          <div class="config-row">
            <label>图元数量</label>
            <input type="number" name="num_primitives" value="{{ cfg_np }}" min="40" max="3000" class="num-input">
          </div>
          <div class="config-row">
            <label>图片缩放</label>
            <input type="number" name="image_scale" value="{{ cfg_scale }}" min="0.5" max="4" step="0.1" class="num-input">
          </div>
          <div class="config-row">
            <label>透明度</label>
            <input type="number" name="output_alpha" value="{{ cfg_alpha }}" min="0" max="100" step="5" class="num-input">
          </div>
          <button type="submit" class="btn-primary" style="margin-top:8px">重新处理</button>
        </form>
      </section>

      <section class="panel-section" id="retrySectionOutline" hidden>
        <h3>重新处理</h3>
        <form action="/retry/{{ task_id }}" method="POST">
          <div class="config-row">
            <label>图元大小</label>
            <input type="number" name="primitive_size" value="{{ cfg_ol_size }}" min="3" max="200" class="num-input">
          </div>
          <div class="config-row">
            <label>间距</label>
            <input type="number" name="spacing" value="{{ cfg_ol_spacing }}" min="0.1" max="1.0" step="0.05" class="num-input">
          </div>
          <div class="config-row">
            <label>精度</label>
            <input type="number" name="precision" value="{{ cfg_ol_precision }}" min="0.0" max="1.0" step="0.05" class="num-input">
          </div>
          <button type="submit" class="btn-primary" style="margin-top:8px">重新处理</button>
        </form>
      </section>
    </aside>

    <main class="canvas-area">
      <div id="canvasWrap" class="canvas-wrap">
        <canvas id="mainCanvas"></canvas>
        <div id="tooltip" class="tooltip" hidden></div>
      </div>

      <div id="previewCompare" class="preview-compare" hidden>
        <div class="preview-item">
          <span class="preview-label">拟合效果</span>
          <img id="previewImg" class="preview-thumb">
        </div>
        <div class="preview-item">
          <span class="preview-label">原图</span>
          <img id="originalThumb" class="preview-thumb">
        </div>
      </div>

      <div class="canvas-bar">
        <span id="coordsDisplay">坐标: —</span>
        <span id="elemCountDisplay">图元: —</span>
        <label><input type="checkbox" id="showImage"> 原图</label>
        <label><input type="checkbox" id="showMask"> 遮罩</label>
        <label><input type="checkbox" id="showFill" checked> 填充</label>
        <label><input type="checkbox" id="showBorder"> 边框</label>
        <label><input type="checkbox" id="showOrigin" checked> 原点</label>
      </div>
    </main>

    <aside class="panel panel-right">
      <section class="panel-section">
        <h3>选中图元</h3>
        <p class="hint">点击画布中的图元查看详情。</p>
        <div id="infoEmpty" class="empty-panel">
          <p>未选中图元</p>
          <span>点击图元后这里会显示详细信息</span>
        </div>
        <div id="infoPanel" class="elem-info" hidden><table>
          <tr><td>ID</td><td id="infoId">—</td></tr>
          <tr><td>类型</td><td id="infoType">—</td></tr>
          <tr><td>中心</td><td id="infoCenter">—</td></tr>
          <tr><td>相对原点</td><td id="infoRelative">—</td></tr>
          <tr><td>尺寸</td><td id="infoSize">—</td></tr>
          <tr><td>旋转</td><td id="infoRotation">—</td></tr>
        </table></div>
      </section>
    </aside>
  </div>

  <script src="/web/app.js?v=21"></script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE_UPLOAD


@app.route("/web/<path:filename>")
def static_file(filename):
    response = send_from_directory("web", filename)
    if filename.endswith((".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store"
    return response


@app.route("/submit", methods=["POST"])
def submit():
    cleanup()
    if "image" not in request.files:
        return "缺少图片", 400
    upload = request.files["image"]
    blob = upload.read()
    if not blob:
        return "图片为空", 400
    image_name = _derive_upload_image_name(upload.filename)

    mode = request.form.get("mode", "fill")
    cfg = {
        "mode": mode,
        "source_filename": upload.filename or "",
        "source_ext": os.path.splitext(upload.filename or "")[1].lower(),
        "origin": {
            "type": request.form.get("origin_type", "center"),
            "x": request.form.get("origin_x", ""),
            "y": request.form.get("origin_y", ""),
        },
    }
    primitives = []
    try:
        parsed_primitives = json.loads(request.form.get("primitives_json", "[]"))
        if isinstance(parsed_primitives, list):
            primitives = parsed_primitives
    except Exception:
        primitives = []

    if mode == "fill":
        # Support both slider and manual input for num_primitives
        manual_prims = request.form.get("num_primitives_manual", "")
        if manual_prims:
            cfg["num_primitives"] = max(40, min(3000, int(manual_prims)))
        else:
            cfg["num_primitives"] = max(40, min(3000, int(request.form.get("num_primitives", 400))))
        cfg["mask_threshold"] = int(request.form.get("mask_threshold", 127))
        cfg["detail_scale"] = float(request.form.get("detail_scale", 1.0))
        cfg["image_scale"] = float(request.form.get("image_scale", 1.0))
        cfg["output_alpha"] = float(request.form.get("output_alpha", 100)) / 100.0
        cfg["enable_png_mode"] = request.form.get("enable_png_mode") == "on"
        allowed_shapes = []
        if request.form.get("shape_circle") == "on":
            allowed_shapes.append("circle")
        if request.form.get("shape_rect") == "on":
            allowed_shapes.append("rect")
        if request.form.get("shape_triangle") == "on":
            allowed_shapes.append("triangle")
        cfg["allowed_shapes"] = allowed_shapes or ["circle"]
    else:
        cfg["primitive_size"] = float(request.form.get("ol_primitive_size", 30))
        cfg["spacing"] = float(request.form.get("ol_spacing", 0.9))
        cfg["precision"] = float(request.form.get("ol_precision", 0.3))
        if primitives:
            outline_shapes = []
            for primitive in primitives:
                shape = str(primitive.get("shape", "")).strip().lower()
                if shape in ("circle", "rect") and shape not in outline_shapes:
                    outline_shapes.append(shape)
            cfg["allowed_shapes"] = outline_shapes or ["circle"]
        else:
            cfg["allowed_shapes"] = ["circle"]

    if primitives:
        cfg["primitives"] = primitives

    task_id = uuid.uuid4().hex[:8]
    tasks[task_id] = {
        "status": "processing",
        "ts": time.time(),
        "image_bytes": blob,
        "image_name": image_name,
        "config": cfg,
    }

    def worker():
        try:
            result = shaper_core.process_image(blob, cfg)
            tasks[task_id]["result"] = result
            tasks[task_id]["status"] = "done"
        except Exception as exc:
            traceback.print_exc()
            tasks[task_id]["error"] = str(exc)
            tasks[task_id]["status"] = "error"

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f"/status/{task_id}")


@app.route("/retry/<tid>", methods=["POST"])
def retry(tid):
    old_task = tasks.get(tid)
    if not old_task or "image_bytes" not in old_task:
        return redirect("/")

    old_cfg = old_task.get("config", {})
    mode = old_cfg.get("mode", "fill")
    cfg = {"mode": mode}

    if mode == "fill":
        cfg["num_primitives"] = max(40, min(3000, int(request.form.get("num_primitives", old_cfg.get("num_primitives", 400)))))
        cfg["mask_threshold"] = int(request.form.get("mask_threshold", old_cfg.get("mask_threshold", 127)))
        cfg["detail_scale"] = float(request.form.get("detail_scale", old_cfg.get("detail_scale", 1.0)))
        cfg["image_scale"] = float(request.form.get("image_scale", old_cfg.get("image_scale", 1.0)))
        cfg["output_alpha"] = float(request.form.get("output_alpha", int(old_cfg.get("output_alpha", 1.0) * 100))) / 100.0
        cfg["enable_png_mode"] = old_cfg.get("enable_png_mode", False)
        cfg["allowed_shapes"] = old_cfg.get("allowed_shapes", ["circle"])
    else:
        cfg["primitive_size"] = float(request.form.get("primitive_size", old_cfg.get("primitive_size", 30)))
        cfg["spacing"] = float(request.form.get("spacing", old_cfg.get("spacing", 0.9)))
        cfg["precision"] = float(request.form.get("precision", old_cfg.get("precision", 0.3)))

    if "primitives" in old_cfg:
        cfg["primitives"] = old_cfg["primitives"]
    if "origin" in old_cfg:
        cfg["origin"] = old_cfg["origin"]

    new_id = uuid.uuid4().hex[:8]
    tasks[new_id] = {
        "status": "processing",
        "ts": time.time(),
        "image_bytes": old_task["image_bytes"],
        "image_name": old_task.get("image_name", ""),
        "config": cfg,
    }

    def worker():
        try:
            result = shaper_core.process_image(old_task["image_bytes"], cfg)
            tasks[new_id]["result"] = result
            tasks[new_id]["status"] = "done"
        except Exception as exc:
            traceback.print_exc()
            tasks[new_id]["error"] = str(exc)
            tasks[new_id]["status"] = "error"

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f"/status/{new_id}")


@app.route("/status/<tid>")
def status(tid):
    task = tasks.get(tid)
    if not task:
        return redirect("/")
    if task["status"] == "done":
        return redirect(f"/result/{tid}")
    if task["status"] == "error":
        return f"<h2>出错</h2><p>{task.get('error')}</p><a href='/'>返回</a>"
    elapsed = int(time.time() - task["ts"])
    return render_template_string(PAGE_STATUS, task_id=tid, elapsed=elapsed)


@app.route("/result/<tid>")
def result(tid):
    task = tasks.get(tid)
    if not task or "result" not in task:
        return redirect("/")

    result_data = task["result"]
    cfg = task["config"]
    return render_template_string(
        PAGE_RESULT,
        result_json=json.dumps(result_data),
        config_json=json.dumps(cfg),
        task_id=tid,
        image_name_json=json.dumps(task.get("image_name", "")),
        count=result_data["elements_count"],
        elapsed=result_data["elapsed_seconds"],
        cfg_np=cfg.get("num_primitives", 400),
        cfg_scale=cfg.get("image_scale", 1.0),
        cfg_alpha=int(cfg.get("output_alpha", 1.0) * 100),
        cfg_ol_size=cfg.get("primitive_size", 30),
        cfg_ol_spacing=cfg.get("spacing", 0.9),
        cfg_ol_precision=cfg.get("precision", 0.3),
    )


_json_to_gia_mod = None


def _load_json_to_gia():
    global _json_to_gia_mod
    if _json_to_gia_mod is not None:
        return _json_to_gia_mod

    gia_dir = os.path.join(BASE_DIR, "gia")
    if gia_dir not in sys.path:
        sys.path.insert(0, gia_dir)

    import json_to_gia

    _json_to_gia_mod = json_to_gia
    return _json_to_gia_mod


@app.route("/download_overlimit_gia/<tid>")
def download_overlimit_gia(tid):
    task = tasks.get(tid)
    if not task or "result" not in task:
        return "任务不存在", 404

    result_data = task["result"]
    cfg = task.get("config", {})
    pixel_per_unit = float(result_data.get("config", {}).get("pixel_per_unit") or cfg.get("primitive_size") or 1.0)
    origin_default = result_data.get("image_center", {"x": 0, "y": 0})
    try:
        origin_x = float(request.args.get("origin_x", origin_default.get("x", 0)))
        origin_y = float(request.args.get("origin_y", origin_default.get("y", 0)))
    except Exception:
        return "origin 参数无效", 400

    origin_units_x = origin_x / pixel_per_unit
    origin_units_y = -origin_y / pixel_per_unit

    elements = []
    for element in result_data.get("elements", []):
        shape_type = element.get("type")
        if shape_type == "circle":
            shape_type = "ellipse"
        elif shape_type == "rect":
            shape_type = "rectangle"

        rotation = element.get("rotation", {}) or {}
        center = element.get("center", {}) or {}
        exported = {
            "type": shape_type,
            "relative": {
                "x": float(center.get("x", 0)) - origin_units_x,
                "y": float(center.get("y", 0)) - origin_units_y,
            },
            "size": element.get("size", {}),
            "rotation": rotation,
            "color": element.get("color"),
            "alpha": element.get("alpha"),
            "packed_color": element.get("packed_color"),
            "image_asset_ref": element.get("image_asset_ref", 100002),
        }
        if element.get("type_id") is not None:
            exported["type_id"] = int(element["type_id"])
        elif element.get("element_type_id") is not None:
            exported["type_id"] = int(element["element_type_id"])
        if element.get("element_type_id") is not None:
            exported["element_type_id"] = int(element["element_type_id"])
        if rotation.get("y") is not None:
            exported["rot_y_add"] = float(rotation.get("y", 0))
        elements.append(exported)

    mask_cfg = None
    mask_data = result_data.get("mask") or {}
    if mask_data.get("enabled"):
        mask_center = mask_data.get("center") or {}
        mask_size = mask_data.get("size") or {}
        mask_cfg = {
            "enabled": True,
            "shape_type": mask_data.get("shape_type", "rectangle"),
            "center": {
                "x": float(mask_center.get("x", 0)) - origin_units_x,
                "y": float(mask_center.get("y", 0)) - origin_units_y,
            },
            "size": {
                "width": float(mask_size.get("width", 0)),
                "height": float(mask_size.get("height", 0)),
            },
        }

    json_data = {
        "elements": elements,
        "mask": mask_cfg,
        "group_name": _export_basename(task.get("image_name", "")),
    }

    mod = _load_json_to_gia()
    gia_bytes = mod.convert_json_to_gia_bytes(
        json_data=json_data,
        base_gia_path=os.path.join(BASE_DIR, "gia", "image_template.gia"),
        mode=mod.MODE_IMAGE,
    )

    response = Response(gia_bytes, mimetype="application/octet-stream")
    export_name = f"{_export_basename(task.get('image_name', ''))}.gia"
    response.headers["Content-Disposition"] = _attachment_filename(export_name)
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5555"))
    print(f"Shaper http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True)
