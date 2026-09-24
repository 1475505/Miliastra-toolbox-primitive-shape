import argparse
import gzip
import json
import logging
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

import lua_export
import gia_lua
import shaper_core


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


app = Flask(__name__, static_folder=os.path.join(BASE_DIR, "web"), static_url_path="/web")
# 本地模式寄存会携带 源图 + 原图/预览/蒙版 base64，大图请求体可达上百 MB
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024

tasks = {}
WEB_DIR = os.path.join(BASE_DIR, "web")

DEFAULT_IMAGE_ASSET_REFS = {
    "rectangle": 100001,
    "ellipse": 100002,
    "triangle": 100003,
}


def _compute_web_asset_version():
    asset_paths = [
        os.path.join(WEB_DIR, "style.css"),
        os.path.join(WEB_DIR, "upload.js"),
        os.path.join(WEB_DIR, "local_fit.js"),
        os.path.join(WEB_DIR, "app.js"),
        os.path.join(WEB_DIR, "clipboard.js"),
    ]
    mtimes = []
    for path in asset_paths:
        if os.path.exists(path):
            mtimes.append(int(os.path.getmtime(path)))
    return str(max(mtimes) if mtimes else 1)


WEB_ASSET_VERSION = _compute_web_asset_version()

# 默认拟合运行方式（上传页「本地模式」开关初始状态）：local=本地（WASM），cloud=云端
DEFAULT_FIT_MODE = os.environ.get("DEFAULT_FIT_MODE", "local").strip().lower()
if DEFAULT_FIT_MODE not in ("local", "cloud"):
    DEFAULT_FIT_MODE = "local"


def cleanup():
    now = time.time()
    for key in [key for key, value in tasks.items() if now - value.get("ts", 0) > 1800]:
        del tasks[key]


@app.errorhandler(413)
def request_entity_too_large(_error):
    # 返回 JSON 而非 HTML 错误页，前端 registerResult 能给出可读信息
    return {"ok": False, "error": "请求体过大（超过 256MB 上限），请减小图片尺寸后重试"}, 413


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


def _convert_result_to_gia_bytes(result_data, cfg=None, image_name="", origin_x=None, origin_y=None):
    cfg = cfg or {}
    pixel_per_unit = float(result_data.get("config", {}).get("pixel_per_unit") or cfg.get("primitive_size") or 1.0)
    origin_default = result_data.get("image_center", {"x": 0, "y": 0})
    resolved_origin_x = float(origin_default.get("x", 0) if origin_x is None else origin_x)
    resolved_origin_y = float(origin_default.get("y", 0) if origin_y is None else origin_y)

    origin_units_x = resolved_origin_x / pixel_per_unit
    origin_units_y = -resolved_origin_y / pixel_per_unit

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
            "image_asset_ref": element.get("image_asset_ref", DEFAULT_IMAGE_ASSET_REFS.get(shape_type, 100002)),
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
    if mask_data:
        mask_center = mask_data.get("center") or {}
        mask_size = mask_data.get("size") or {}
        mask_cfg = {
            "enabled": bool(mask_data.get("enabled", False)),
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
        "group_name": _export_basename(image_name),
    }

    mod = _load_json_to_gia()
    return mod.convert_json_to_gia_bytes(
        json_data=json_data,
        base_gia_path=os.path.join(BASE_DIR, "gia", "image_template.gia"),
        mode=mod.MODE_IMAGE,
    )


def _parse_cli_shape_list(shape_args):
    allowed = []
    for shape in shape_args or []:
        normalized = str(shape).strip().lower()
        if normalized in ("circle", "rect", "triangle") and normalized not in allowed:
            allowed.append(normalized)
    return allowed or ["circle"]


def _build_cli_config(args, input_path):
    source_ext = os.path.splitext(input_path)[1].lower()
    cfg = {
        "mode": args.mode,
        "source_filename": os.path.basename(input_path),
        "source_ext": source_ext,
        "origin": {
            "type": "custom" if args.origin_x is not None or args.origin_y is not None else "center",
            "x": 0 if args.origin_x is None else float(args.origin_x),
            "y": 0 if args.origin_y is None else float(args.origin_y),
        },
    }

    if args.mode == "fill":
        cfg["num_primitives"] = max(40, min(3000, int(args.num_primitives)))
        cfg["mask_threshold"] = int(args.mask_threshold)
        cfg["detail_scale"] = float(args.detail_scale)
        cfg["image_scale"] = float(args.image_scale)
        cfg["output_alpha"] = float(args.output_alpha) / 100.0
        cfg["enable_png_mode"] = bool(args.enable_png_mode)
        cfg["allowed_shapes"] = _parse_cli_shape_list(args.shape)
    else:
        cfg["primitive_size"] = float(args.primitive_size)
        cfg["spacing"] = float(args.spacing)
        cfg["precision"] = float(args.precision)
        cfg["allowed_shapes"] = [shape for shape in _parse_cli_shape_list(args.shape) if shape in ("circle", "rect")] or ["circle"]
    return cfg


def _default_cli_output_path(input_path):
    base_name = _export_basename(input_path)
    return os.path.join(os.getcwd(), f"{base_name}.gia")


def _run_cli(args):
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input image not found: {input_path}")

    with open(input_path, "rb") as f:
        image_bytes = f.read()

    cfg = _build_cli_config(args, input_path)
    result = shaper_core.process_image(image_bytes, cfg)
    origin_default = result.get("image_center", {"x": 0, "y": 0})
    image_name = args.name if args.name else os.path.basename(input_path)

    if args.export == "lua":
        try:
            lua_text = lua_export.build_lua_export_text(result, image_name=image_name)
        except Exception as exc:
            logger.exception("cli_lua_export_error")
            print(f"导出 Lua 失败: {exc}")
            return 1
        base_name = _export_basename(image_name)
        output_path = os.path.abspath(args.output or os.path.join(os.getcwd(), f"{base_name}.lua"))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lua_text)
        print(f"Generated Lua: {output_path}")
        print(f"Mode: {result.get('mode')}")
        print(f"Elements: {result.get('elements_count')}")
        print(f"Elapsed: {result.get('elapsed_seconds')}s")
        return 0

    gia_bytes = _convert_result_to_gia_bytes(
        result_data=result,
        cfg=cfg,
        image_name=image_name,
        origin_x=origin_default.get("x", 0) if args.origin_x is None else float(args.origin_x),
        origin_y=origin_default.get("y", 0) if args.origin_y is None else float(args.origin_y),
    )

    if args.gia_mode == "classic":
        try:
            mod = _load_convert_to_classic()
            gia_bytes = mod.convert_gia_bytes_to_classic(gia_bytes)
        except Exception as exc:
            logger.exception("classic_gia_convert_error")
            print(f"转换为经典模式失败: {exc}")
            return 1

    output_path = os.path.abspath(args.output or _default_cli_output_path(input_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(gia_bytes)

    print(f"Generated GIA: {output_path}")
    print(f"Mode: {result.get('mode')}")
    print(f"Elements: {result.get('elements_count')}")
    print(f"Elapsed: {result.get('elapsed_seconds')}s")
    return 0


def _create_arg_parser():
    parser = argparse.ArgumentParser(description="Shaper web server / CLI")
    parser.add_argument("--cli", action="store_true", help="run in CLI mode and export .gia directly")
    parser.add_argument("--host", default="0.0.0.0", help="web mode host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5555")), help="web mode port")

    parser.add_argument("--input", help="input image path for CLI mode")
    parser.add_argument("--output", help="output .gia path for CLI mode")
    parser.add_argument("--name", help="character name used as GIA group_name (defaults to input filename stem)")
    parser.add_argument("--mode", choices=["fill", "outline"], default="fill", help="processing mode for CLI mode")
    parser.add_argument("--shape", action="append", help="allowed shape, repeatable: circle / rect / triangle")
    parser.add_argument("--origin-x", type=float, help="custom origin x in pixels for gia export")
    parser.add_argument("--origin-y", type=float, help="custom origin y in pixels for gia export")

    parser.add_argument("--num-primitives", type=int, default=400, help="fill mode primitive count")
    parser.add_argument("--image-scale", type=float, default=1.0, help="fill mode image scale")
    parser.add_argument("--output-alpha", type=float, default=100.0, help="fill mode alpha percent")
    parser.add_argument("--detail-scale", type=float, default=1.0, help="fill mode detail scale")
    parser.add_argument("--mask-threshold", type=int, default=127, help="fill mode alpha threshold")
    parser.add_argument("--enable-png-mode", action="store_true", help="fill mode transparent png output")

    parser.add_argument("--primitive-size", type=float, default=30.0, help="outline mode primitive size")
    parser.add_argument("--spacing", type=float, default=0.9, help="outline mode spacing")
    parser.add_argument("--precision", type=float, default=0.3, help="outline mode precision")
    parser.add_argument("--gia-mode", choices=["overlimit", "classic"], default="overlimit", help="GIA output mode for CLI (default: overlimit)")
    parser.add_argument("--export", choices=["gia", "lua"], default="gia", help="CLI export format: gia (default) or lua client script")
    return parser


PAGE_UPLOAD = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>千星奇域拼图工具</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/web/style.css?v={{ asset_version }}">
</head>
<body class="page-upload" data-default-fit-mode="{{ default_fit_mode }}">
  <header class="topbar">
    <div class="topbar-left">
      <a href="/" class="brand-link"><h1>图片图元拟合</h1></a>
      <span class="topbar-subtitle">默认填充模式 · 默认仅圆形</span>
      <nav class="tool-tabs" aria-label="工具页签">
        <button type="button" id="imageToolTab" class="tool-tab active">图片拟合</button>
        <button type="button" id="classicToolTab" class="tool-tab">GIA转换</button>
      </nav>
      <a href="#" id="outlineLink" class="topbar-link topbar-link-subtle">装饰物</a>
    </div>
    <div class="topbar-right">
      <a href="https://github.com/1475505/Miliastra-toolbox-primitive-shape" target="_blank" class="topbar-link">仓库</a>
      <a href="https://ugc.070077.xyz/" target="_blank" class="topbar-link">知识问答</a>
      <span class="topbar-status active">就绪</span>
    </div>
  </header>

  <div class="app-layout" id="imageToolPage">
    <main class="fit-workspace">
      <form id="mainForm" action="/submit" method="POST" enctype="multipart/form-data">
        <input type="hidden" name="mode" id="modeInput" value="fill">
        <input type="hidden" name="primitives_json" id="primJson">
        <div class="form-column settings-column">
          <div class="settings-scroll">
            <section class="panel-section" id="localModeSection">
              <h3>运行方式</h3>
              <div class="engine-card" id="engineCard">
                <div class="engine-card-head">
                  <label class="switch" for="localModeToggle">
                    <input type="checkbox" id="localModeToggle">
                    <span class="switch-track"><span class="switch-thumb"></span></span>
                  </label>
                  <div class="engine-card-title">
                    <strong>本地模式</strong>
                    <span class="engine-badge" id="engineBadge">云端引擎</span>
                  </div>
                </div>
                <p class="hint" id="engineHint">云端模式：由服务器完成拟合计算。</p>
                <div class="engine-status" id="engineStatus" hidden>
                  <span class="engine-status-dot"></span>
                  <span id="engineStatusText">本地引擎加载中…</span>
                </div>
              </div>
            </section>

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
                    <span id="numPrimsVal" hidden>400</span>
                  </div>
                  <p class="param-desc">越多越细，但耗时也更高。</p>
                  <div class="primitive-count-row">
                    <input type="range" name="num_primitives" id="numPrims" min="40" max="1200" step="10" value="400" aria-label="图元数量">
                    <input type="number" name="num_primitives_manual" id="numPrimsManual" min="40" max="3000" value="400" class="num-input" aria-label="手动输入图元数量">
                  </div>
                </div>

                <div class="param-item">
                  <div class="param-head">
                    <span class="param-title">输出尺寸</span>
                    <span id="outputSizeVal" class="val-tag">缩放 ×1.0</span>
                  </div>
                  <div class="seg-toggle" id="outputSizeToggle">
                    <button type="button" class="seg-btn active" id="segScale" data-mode="scale">按比例缩放</button>
                    <button type="button" class="seg-btn" id="segTarget" data-mode="target">指定分辨率</button>
                  </div>

                  <div id="scalePanel" class="output-size-panel">
                    <div class="scale-row">
                      <input type="range" name="image_scale" id="imageScale" min="0.2" max="4" step="0.1" value="1.0">
                      <span id="imageScaleVal" class="val-tag">1.0</span>
                    </div>
                    <p class="hint">1.0 表示导出尺寸与原图分辨率一致。</p>
                  </div>

                  <div id="targetPanel" class="output-size-panel" hidden>
                    <input type="checkbox" name="enable_target_resolution" id="enableTargetRes" hidden>
                    <div class="target-res-row" id="targetResRow">
                      <input type="number" name="target_width" id="targetWidth" min="16" max="4096" step="1" placeholder="宽" class="num-input">
                      <span class="target-res-x">×</span>
                      <input type="number" name="target_height" id="targetHeight" min="16" max="4096" step="1" placeholder="高" class="num-input">
                      <button type="button" id="targetResLock" class="btn-chip lock-btn active" title="锁定宽高比">等比</button>
                    </div>
                    <p class="hint" id="targetResHint">按原图比例联动；取消「等比」可自由拉伸。指定分辨率时缩放按 1.0 处理。</p>
                  </div>
                </div>

                <div class="param-item">
                  <div class="param-head">
                    <span class="param-title">透明度</span>
                    <span id="outputAlphaVal" class="val-tag">100%</span>
                  </div>
                  <p class="param-desc">100% 不透明，0% 完全透明。</p>
                  <input type="range" name="output_alpha" id="outputAlpha" min="0" max="100" step="5" value="100">
                </div>

                <div class="param-item">
                  <label class="png-mode-toggle">
                    <input type="checkbox" name="enable_png_mode" id="enablePngMode">
                    <span>启用 PNG 模式</span>
                  </label>
                  <p class="param-desc">保留 PNG 的透明背景；关闭时先铺白底再拟合。</p>
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

            <details class="upload-guide">
              <summary>使用说明</summary>
              <div class="guide-content">
              <section>
                <h3>流程</h3>
                <ol class="steps">
                  <li>上传图片</li>
                  <li>确认图元和算法参数</li>
                  <li>预览效果，导出 GIA / Lua / CSS 等格式</li>
                </ol>
              </section>
              <section>
                <ul class="tips">
                  <li><strong>禁止拟合政治、人物、事件、OOC等任何不适合的内容</strong></li>
                  <li><strong>请合理使用本工具生成的gia资产，不要上传到资产中心等，若造成不良影响与工具作者无关，</strong></li>
                  <li>目前对三角形和矩形支持不友好，多类型图形的效果较差</li>
                  <li>PNG 图片默认会将透明区域与白色背景混合。如需保留透明背景，请在参数中开启「PNG 模式」。</li>
                  <li>之前用于拼字的系统：<a href="https://qx-shaper.up.railway.app" target="_blank" rel="noopener noreferrer">qx-shaper.up.railway.app</a></li>
                  <li>使用教程：<a href="https://www.bilibili.com/video/BV1kKDyB9EvY" target="_blank" rel="noopener noreferrer">BV1kKDyB9EvY</a></li>
                  <li>用户 QQ 群：<a href="https://qm.qq.com/cgi-bin/qm/qr?k=1007538100" target="_blank" rel="noopener noreferrer">1007538100</a></li>
                </ul>
              </section>
              </div>
            </details>
          </div>
          <section class="panel-section section-submit">
            <button type="submit" id="btnSubmit" class="btn-primary">开始处理</button>
            <div id="localProgress" class="local-progress" hidden>
              <div class="local-progress-head">
                <span id="localProgressText">本地引擎准备中…</span>
                <span id="localProgressPct" class="val-tag">0%</span>
              </div>
              <div class="local-progress-bar"><span id="localProgressFill"></span></div>
            </div>
          </section>
        </div>
        <div class="form-column source-column">
          <section class="panel-section section-input">
            <h3>输入图片</h3>
            <div id="dropZone" class="drop-zone" role="button" tabindex="0" aria-label="上传图片，点击、拖拽或 Ctrl+V 粘贴">
              <div class="drop-zone-content">
                <span class="drop-icon" aria-hidden="true"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m3 17 5-5 4 4 3-3 6 6"/></svg></span>
                <p>点击、拖拽或 <strong>Ctrl+V</strong> 粘贴图片</p>
              </div>
              <input type="file" id="fileInput" name="image" accept="image/png,image/jpeg,image/webp" required hidden>
              <img id="prev" class="preview-img" hidden>
              <span id="fname" class="file-name"></span>
              <span id="imgSize" class="img-size"></span>
              <div id="uploadReady" class="upload-ready" hidden aria-live="polite">已选择图片</div>
              <div id="uploadWarning" class="upload-warning" hidden aria-live="polite">不建议上传分辨率较大的图片，会很慢</div>
            </div>
            <p class="hint">支持 PNG / JPG / WEBP；透明背景请开启「PNG 模式」。</p>
          </section>
        </div>
      </form>
    </main>
  </div>

  <div class="app-layout" id="classicToolPage" hidden>
    <aside class="panel panel-left">
      <form id="classicGiaForm" action="/convert_gia_mode" method="POST" enctype="multipart/form-data">
        <input type="hidden" name="direction" id="giaDirectionInput" value="overlimit_to_classic">
        <section class="panel-section">
          <h3>转换方向</h3>
          <div class="direction-select" style="display:flex;flex-direction:column;gap:8px;">
            <label style="display:flex;align-items:center;gap:8px;font-size:14px;color:var(--text-main);cursor:pointer;">
              <input type="radio" name="direction_radio" id="dirOverToClassic" value="overlimit_to_classic" checked>
              <span>超限模式转经典模式</span>
            </label>
            <label style="display:flex;align-items:center;gap:8px;font-size:14px;color:var(--text-main);cursor:pointer;">
              <input type="radio" name="direction_radio" id="dirClassicToOver" value="classic_to_overlimit">
              <span>经典模式转超限模式</span>
            </label>
            <label style="display:flex;align-items:center;gap:8px;font-size:14px;color:var(--text-main);cursor:pointer;">
              <input type="radio" name="direction_radio" id="dirGiaToLua" value="gia_to_lua">
              <span>素材组 GIA 转 Lua 绘制脚本（超限模式）</span>
            </label>
          </div>
        </section>

        <section class="panel-section">
          <h3 id="classicUploadTitle">超限模式 GIA</h3>
          <div id="classicDropZone" class="drop-zone">
            <div class="drop-zone-content">
              <span class="drop-icon">GIA</span>
              <p id="classicDropText">点击或拖拽上传超限模式 <strong>.gia</strong></p>
            </div>
            <input type="file" id="classicGiaInput" name="gia" accept=".gia,application/octet-stream" required hidden>
            <span id="classicGiaName" class="file-name"></span>
            <div id="classicGiaReady" class="upload-ready" hidden>已选择 GIA</div>
          </div>
          <p class="hint" id="classicHint">转换会为 GIA 写入经典模式标记，原始文件不会被修改。</p>
          <label id="giaLuaMaskOption" hidden>
            <input type="checkbox" name="ignore_mask" value="1">
            忽略组遮罩，绘制全部图片（裁剪范围外的内容也会显示）
          </label>
        </section>

        <section class="panel-section section-submit">
          <button type="submit" id="btnConvertClassicGia" class="btn-primary">导出经典模式 GIA</button>
          <button type="submit" name="lua_action" value="copy" id="btnCopyGiaLua" class="btn-sm" hidden>复制 Lua</button>
          <p id="giaLuaCopyStatus" class="hint" role="status" aria-live="polite"></p>
        </section>
      </form>
    </aside>

    <main class="canvas-area tool-empty">
      <div class="classic-tool-copy">
        <h2 id="classicToolTitle">超限模式转经典模式</h2>
        <p id="classicToolDesc">上传现有超限模式 GIA，转换后会下载一个带 <code>_classic</code> 后缀的经典模式 GIA。</p>
      </div>
    </main>

    <aside class="panel panel-right">
      <section class="panel-section guide-card">
        <h3>说明</h3>
        <ol class="steps" id="classicSteps">
          <li>上传超限模式 .gia</li>
          <li>写入经典模式标记</li>
          <li>下载新的经典模式 .gia</li>
        </ol>
        <ul class="tips" id="classicModeTips">
          <li>只处理 GIA 文件头部的模式字段，不会重新生成素材内容。</li>
          <li>如果文件已经是目标模式，也会重新导出为目标模式文件。</li>
        </ul>
      </section>
    </aside>
  </div>

  <script src="/web/clipboard.js?v={{ asset_version }}"></script>
  <script src="/web/local_fit.js?v={{ asset_version }}"></script>
  <script src="/web/upload.js?v={{ asset_version }}"></script>
</body>
</html>"""


PAGE_STATUS = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="5">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/web/style.css?v={{ asset_version }}">
</head>
<body class="page-status">
  <div class="loading-overlay">
    <div class="spinner"></div>
    <h2 style="margin-top:20px;font-weight:600;color:var(--md-on-surface)">处理中 ({{ elapsed }}s)</h2>
    <p style="margin-top:8px;font-size:13px;color:var(--md-on-surface-variant);font-family:var(--font-mono)">任务 ID: {{ task_id }}</p>
  </div>
</body>
</html>"""


PAGE_RESULT = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>千星奇域拼图工具 - 结果</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/web/style.css?v={{ asset_version }}">
  <script>
    var RESULT={{ result_json|safe }};
    var TASK_CFG={{ config_json|safe }};
    var TASK_ID="{{ task_id }}";
    var TASK_IMAGE_NAME={{ image_name_json|safe }};
  </script>
</head>
<body class="page-result">
  <header class="topbar">
    <div class="topbar-left">
      <a href="/" class="brand-link"><h1>图片图元拟合</h1></a>
      <span class="topbar-subtitle" id="modeLabel">填充拟合</span>
    </div>
    <div class="topbar-right">
      <span id="statusText" class="topbar-status active">完成 · {{ count }} 图元 · {{ elapsed }}s</span>
      <a href="/" class="btn-tonal">新建</a>
    </div>
  </header>

  <div class="app-layout">
    <aside class="panel panel-left">
      <section class="panel-section">
        <h3>导出</h3>
        <div class="export-name-field">
          <label for="exportFileName">导出文件名</label>
          <input type="text" id="exportFileName" autocomplete="off">
          <p class="hint">默认与原文件名一致，同时用于 GIA 素材组名称。</p>
        </div>
        <a class="editor-link" href="https://qx.070077.xyz/" target="_blank" rel="noopener noreferrer" aria-label="导入 CSS 进一步编辑（在新窗口打开）">导入 CSS 进一步编辑 ↗</a>
        <div class="export-group">
          <p class="export-group-title">素材组资产</p>
          <div class="export-stack">
            <button id="btnExportGIAOverlimit" class="btn-sm">导出超限模式 GIA</button>
            <button id="btnExportGIAClassic" class="btn-sm">导出经典模式 GIA</button>
          </div>
        </div>
        <div class="export-group">
          <p class="export-group-title">客户端脚本</p>
          <div class="export-stack">
            <div class="export-action">
              <button id="btnExportLua" class="btn-sm">导出 Lua</button>
              <button id="btnCopyLua" class="btn-copy" type="button">复制</button>
            </div>
          </div>
        </div>
        <div class="export-group">
          <p class="export-group-title">进一步编辑</p>
          <div class="export-stack">
            <div class="export-action">
              <button id="btnExportJSON" class="btn-sm">导出 JSON</button>
              <button id="btnCopyJSON" class="btn-copy" type="button">复制</button>
            </div>
            <div class="export-action">
              <button id="btnExportCSS" class="btn-sm">导出 CSS</button>
              <button id="btnCopyCSS" class="btn-copy" type="button">复制</button>
            </div>
            <button id="btnExportSVG" class="btn-sm">导出 SVG</button>
            <button id="btnExportPNG" class="btn-sm">导出 PNG</button>
          </div>
          <p class="hint">JSON、CSS、SVG 可用于素材组编辑；PNG 用于保存拟合效果图。</p>
        </div>
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
            <input type="number" name="image_scale" value="{{ cfg_scale }}" min="0.2" max="4" step="0.1" class="num-input">
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

  <script src="/web/clipboard.js?v={{ asset_version }}"></script>
  <script src="/web/local_fit.js?v={{ asset_version }}"></script>
  <script src="/web/app.js?v={{ asset_version }}"></script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(PAGE_UPLOAD, asset_version=WEB_ASSET_VERSION, default_fit_mode=DEFAULT_FIT_MODE)


@app.route("/healthz")
def healthz():
    return {"ok": True, "service": "primitive-shape"}


@app.route("/web/<path:filename>")
def static_file(filename):
    response = send_from_directory(WEB_DIR, filename)
    if filename.endswith((".js", ".css")):
        response.headers["Cache-Control"] = "no-cache, no-store"
    elif filename.endswith(".wasm"):
        response.headers["Content-Type"] = "application/wasm"
        response.headers["Cache-Control"] = "public, max-age=86400"
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
        if request.form.get("enable_target_resolution") == "on":
            try:
                cfg["target_width"] = max(16, min(4096, int(request.form.get("target_width", 0))))
                cfg["target_height"] = max(16, min(4096, int(request.form.get("target_height", 0))))
            except (TypeError, ValueError):
                cfg.pop("target_width", None)
                cfg.pop("target_height", None)
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
    logger.info(
        "task_submit id=%s mode=%s filename=%s bytes=%d cfg=%s",
        task_id,
        mode,
        upload.filename or "",
        len(blob),
        {
            "num_primitives": cfg.get("num_primitives"),
            "detail_scale": cfg.get("detail_scale"),
            "allowed_shapes": cfg.get("allowed_shapes"),
            "primitive_size": cfg.get("primitive_size"),
            "spacing": cfg.get("spacing"),
            "precision": cfg.get("precision"),
            "enable_png_mode": cfg.get("enable_png_mode"),
        },
    )

    def worker():
        started = time.perf_counter()
        try:
            logger.info("task_start id=%s mode=%s", task_id, cfg.get("mode", "fill"))
            result = shaper_core.process_image(blob, cfg)
            tasks[task_id]["result"] = result
            tasks[task_id]["status"] = "done"
            logger.info(
                "task_done id=%s elapsed=%.3fs elements=%s",
                task_id,
                time.perf_counter() - started,
                result.get("elements_count"),
            )
        except Exception as exc:
            traceback.print_exc()
            tasks[task_id]["error"] = str(exc)
            tasks[task_id]["status"] = "error"
            logger.exception("task_error id=%s elapsed=%.3fs", task_id, time.perf_counter() - started)

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f"/status/{task_id}")


@app.route("/retry/<tid>", methods=["POST"])
def retry(tid):
    old_task = tasks.get(tid)
    if not old_task or "image_bytes" not in old_task:
        return redirect("/")
    # 本地任务源图可能尚未后台补传完成，此时无法在服务端重跑
    if old_task.get("local") and not old_task.get("image_ready"):
        return "原图尚未上传完成，暂时无法重新处理，请稍后再试", 409

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
        if old_cfg.get("target_width") and old_cfg.get("target_height"):
            cfg["target_width"] = old_cfg["target_width"]
            cfg["target_height"] = old_cfg["target_height"]
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
    logger.info(
        "task_retry old_id=%s new_id=%s mode=%s cfg=%s",
        tid,
        new_id,
        mode,
        {
            "num_primitives": cfg.get("num_primitives"),
            "detail_scale": cfg.get("detail_scale"),
            "allowed_shapes": cfg.get("allowed_shapes"),
            "primitive_size": cfg.get("primitive_size"),
            "spacing": cfg.get("spacing"),
            "precision": cfg.get("precision"),
        },
    )

    def worker():
        started = time.perf_counter()
        try:
            logger.info("task_start id=%s mode=%s", new_id, cfg.get("mode", "fill"))
            result = shaper_core.process_image(old_task["image_bytes"], cfg)
            tasks[new_id]["result"] = result
            tasks[new_id]["status"] = "done"
            logger.info(
                "task_done id=%s elapsed=%.3fs elements=%s",
                new_id,
                time.perf_counter() - started,
                result.get("elements_count"),
            )
        except Exception as exc:
            traceback.print_exc()
            tasks[new_id]["error"] = str(exc)
            tasks[new_id]["status"] = "error"
            logger.exception("task_error id=%s elapsed=%.3fs", new_id, time.perf_counter() - started)

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f"/status/{new_id}")


@app.route("/register_result", methods=["POST"])
def register_result():
    """寄存本地（WebAssembly）拟合完成的结果，返回 task_id 复用结果页与导出链路。"""
    cleanup()
    raw = request.get_data(cache=False)
    # 弱网优化：客户端可能用 CompressionStream('gzip') 压缩请求体（JSON 实测可降 5 倍）。
    # 按 gzip 魔数识别而非请求头，避免中间层改写头部带来的歧义；旧客户端明文照旧可用。
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            # 截断会抛 EOFError、损坏会抛 OSError/zlib.error，统一按坏请求处理
            logger.warning("task_register_local 请求体解压失败 compressed_bytes=%d", len(raw))
            return {"ok": False, "error": "请求体解压失败"}, 400
    payload = {}
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    result_data = payload.get("result")
    if not isinstance(result_data, dict) or not result_data.get("elements"):
        return {"ok": False, "error": "缺少有效结果"}, 400

    cfg = payload.get("config") or {}
    image_name = _derive_upload_image_name(payload.get("image_name", ""))
    image_bytes = b""
    image_b64 = payload.get("image_base64") or ""
    if image_b64:
        try:
            import base64 as _b64

            image_bytes = _b64.b64decode(image_b64)
        except Exception:
            image_bytes = b""

    task_id = uuid.uuid4().hex[:8]
    # 兼容旧客户端：随请求带上源图 base64 时行为不变（注入底图、图片就绪）
    if image_b64:
        if not result_data.get("image_base64"):
            result_data["image_base64"] = image_b64
    tasks[task_id] = {
        "status": "done",
        "ts": time.time(),
        "image_bytes": image_bytes,
        "image_name": image_name,
        "config": cfg,
        "result": result_data,
        "local": True,
        # 新流程：寄存只传 elements/mask（瞬时），源图由结果页后台补传 /register_image
        "image_ready": bool(image_bytes),
        "image_content_type": "image/png",
    }
    logger.info(
        "task_register_local id=%s elements=%s image_name=%s bytes=%d",
        task_id,
        result_data.get("elements_count"),
        image_name,
        len(image_bytes),
    )
    return {"ok": True, "task_id": task_id}


@app.route("/register_image/<tid>", methods=["POST"])
def register_image(tid):
    """本地模式延迟补传源图（二进制 body）。寄存只传 elements，图片在结果页后台补传。"""
    task = tasks.get(tid)
    if not task or not task.get("local"):
        return {"ok": False, "error": "任务不存在"}, 404
    image_bytes = request.get_data(cache=False)
    if not image_bytes:
        return {"ok": False, "error": "缺少图片数据"}, 400
    if len(image_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return {"ok": False, "error": "图片过大"}, 413
    task["image_bytes"] = image_bytes
    task["image_ready"] = True
    content_type = (request.headers.get("Content-Type") or "").split(";")[0].strip()
    if content_type and content_type.startswith("image/"):
        task["image_content_type"] = content_type
    logger.info("task_register_image id=%s bytes=%d", tid, len(image_bytes))
    return {"ok": True}


@app.route("/task_image/<tid>")
def task_image(tid):
    """按 task_id 取源图（本地模式结果页底图的按需加载）。"""
    task = tasks.get(tid)
    if not task or not task.get("image_bytes"):
        return "图片不存在", 404
    return Response(task["image_bytes"], mimetype=task.get("image_content_type", "image/png"))


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
    return render_template_string(PAGE_STATUS, task_id=tid, elapsed=elapsed, asset_version=WEB_ASSET_VERSION)


@app.route("/result/<tid>")
def result(tid):
    task = tasks.get(tid)
    if not task or "result" not in task:
        return redirect("/")

    result_data = task["result"]
    cfg = task["config"]
    if task.get("local"):
        # 本地任务不把源图 base64 内嵌进 HTML（大图会让结果页达到几十 MB），
        # 改为按需加载 /task_image；未补传完成时由前端走 IndexedDB/后台补传
        render_data = dict(result_data)
        render_data["image_base64"] = None
        render_data["local_task"] = True
        render_data["image_ready"] = bool(task.get("image_ready"))
        render_data["image_url"] = f"/task_image/{tid}" if task.get("image_ready") else None
    else:
        render_data = result_data
    return render_template_string(
        PAGE_RESULT,
        result_json=json.dumps(render_data),
        config_json=json.dumps(cfg),
        task_id=tid,
        image_name_json=json.dumps(task.get("image_name", "")),
        asset_version=WEB_ASSET_VERSION,
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


_convert_to_classic_mod = None


def _load_convert_to_classic():
    global _convert_to_classic_mod
    if _convert_to_classic_mod is not None:
        return _convert_to_classic_mod

    gia_dir = os.path.join(BASE_DIR, "gia")
    if gia_dir not in sys.path:
        sys.path.insert(0, gia_dir)

    import convert_to_classic

    _convert_to_classic_mod = convert_to_classic
    return _convert_to_classic_mod


_convert_to_overlimit_mod = None


def _load_convert_to_overlimit():
    global _convert_to_overlimit_mod
    if _convert_to_overlimit_mod is not None:
        return _convert_to_overlimit_mod

    gia_dir = os.path.join(BASE_DIR, "gia")
    if gia_dir not in sys.path:
        sys.path.insert(0, gia_dir)

    import convert_to_overlimit

    _convert_to_overlimit_mod = convert_to_overlimit
    return _convert_to_overlimit_mod


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

    export_name = request.args.get("export_name", "")
    resolved_image_name = export_name or task.get("image_name", "")

    gia_bytes = _convert_result_to_gia_bytes(
        result_data=result_data,
        cfg=cfg,
        image_name=resolved_image_name,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    response = Response(gia_bytes, mimetype="application/octet-stream")
    download_name = f"{_export_basename(resolved_image_name)}.gia"
    response.headers["Content-Disposition"] = _attachment_filename(download_name)
    return response


@app.route("/download_classic_gia/<tid>")
def download_classic_gia(tid):
    task = tasks.get(tid)
    if not task or "result" not in task:
        return "任务不存在", 404

    result_data = task["result"]
    cfg = task.get("config", {})
    origin_default = result_data.get("image_center", {"x": 0, "y": 0})
    try:
        origin_x = float(request.args.get("origin_x", origin_default.get("x", 0)))
        origin_y = float(request.args.get("origin_y", origin_default.get("y", 0)))
    except Exception:
        return "origin 参数无效", 400

    export_name = request.args.get("export_name", "")
    resolved_image_name = export_name or task.get("image_name", "")

    gia_bytes = _convert_result_to_gia_bytes(
        result_data=result_data,
        cfg=cfg,
        image_name=resolved_image_name,
        origin_x=origin_x,
        origin_y=origin_y,
    )

    try:
        mod = _load_convert_to_classic()
        gia_bytes = mod.convert_gia_bytes_to_classic(gia_bytes)
    except Exception as exc:
        logger.exception("classic_gia_convert_error task_id=%s", tid)
        return f"转换为经典模式失败: {exc}", 400

    response = Response(gia_bytes, mimetype="application/octet-stream")
    download_name = f"{_export_basename(resolved_image_name)}_classic.gia"
    response.headers["Content-Disposition"] = _attachment_filename(download_name)
    return response


@app.route("/download_lua/<tid>")
def download_lua(tid):
    task = tasks.get(tid)
    if not task or "result" not in task:
        return "任务不存在", 404

    result_data = task["result"]
    export_name = request.args.get("export_name", "")
    resolved_image_name = export_name or task.get("image_name", "")

    try:
        lua_text = lua_export.build_lua_export_text(result_data, image_name=resolved_image_name)
    except Exception as exc:
        logger.exception("lua_export_error task_id=%s", tid)
        return f"导出 Lua 失败: {exc}", 400

    response = Response(lua_text, mimetype="text/plain; charset=utf-8")
    download_name = f"{_export_basename(resolved_image_name)}.lua"
    response.headers["Content-Disposition"] = _attachment_filename(download_name)
    return response


@app.route("/convert_gia_mode", methods=["POST"])
def convert_gia_mode():
    upload = request.files.get("gia")
    if not upload:
        return "缺少 GIA 文件", 400

    blob = upload.read()
    if not blob:
        return "GIA 文件为空", 400

    direction = request.form.get("direction", "overlimit_to_classic")
    source_name = _derive_upload_image_name(upload.filename) or "gia_mode"

    try:
        if direction == "gia_to_lua":
            lua_text = gia_lua.build_gia_lua(blob, source_name, ignore_mask=request.form.get("ignore_mask") == "1")
            response = Response(lua_text, mimetype="text/plain; charset=utf-8")
            response.headers["Content-Disposition"] = _attachment_filename(f"{_export_basename(source_name)}.lua")
            return response
        if direction == "classic_to_overlimit":
            mod = _load_convert_to_overlimit()
            result_bytes = mod.convert_gia_bytes_to_overlimit(blob)
            suffix = "_overlimit"
        else:
            mod = _load_convert_to_classic()
            result_bytes = mod.convert_gia_bytes_to_classic(blob)
            suffix = "_classic"
    except Exception as exc:
        logger.exception("gia_mode_convert_error direction=%s filename=%s", direction, upload.filename or "")
        return f"转换失败: {exc}", 400

    response = Response(result_bytes, mimetype="application/octet-stream")
    response.headers["Content-Disposition"] = _attachment_filename(f"{source_name}{suffix}.gia")
    return response


if __name__ == "__main__":
    parser = _create_arg_parser()
    args = parser.parse_args()

    if args.cli:
        if not args.input:
            parser.error("--input is required when using --cli")
        raise SystemExit(_run_cli(args))

    # 拟合会打满 CPU；CPU 较差时建议 `nice -n 5 python server.py` 启动，
    # 配合 primitive_backend 里的子进程降优先级，保证 /status 轮询能及时响应
    print(f"Shaper http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
