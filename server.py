"""
Shaper Web Server — Flask MPA
后端: 表单 POST → 重定向状态页(meta refresh) → 重定向结果页
前端: 三栏布局, 图元定义, 原点控制, 悬浮/选中, 导出
端口: 5555
"""

import os, sys, json, uuid, traceback, threading, time
import importlib.util
from flask import (Flask, request, redirect, send_from_directory,
                   render_template_string, Response)

# Detect base directory for frozen (PyInstaller) or normal execution
if getattr(sys, 'frozen', False):
    if hasattr(sys, '_MEIPASS'):
        BASE_DIR = sys._MEIPASS
    else:
        BASE_DIR = os.path.dirname(sys.executable)
        # PyInstaller 6+ onedir mode puts content in _internal
        if os.path.exists(os.path.join(BASE_DIR, '_internal')):
            BASE_DIR = os.path.join(BASE_DIR, '_internal')
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import shaper_core

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'web'), static_url_path='/web')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

tasks = {}

def cleanup():
    now = time.time()
    for k in [k for k, v in tasks.items() if now - v.get('ts', 0) > 1800]:
        del tasks[k]

# ───────────────────────── 上传页（三栏） ─────────────────────────
PAGE_UPLOAD = r'''<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaper — 轮廓描边工具</title>
<link rel="stylesheet" href="/web/style.css">
</head><body class="page-upload">
<header class="topbar">
  <div class="topbar-left">
    <a href="/" style="text-decoration:none;"><h1>千星奇域拼好模工具</h1></a>
    <span class="topbar-subtitle">轮廓描边与图元拟合</span>
    <a href="https://ugc.070077.xyz/" target="_blank" class="topbar-link">📚 AI知识库-千星奇域工具箱</a>
    <a href="https://github.com/1475505/Miliastra-toolbox-primitive-shape" target="_blank" class="topbar-link">开源地址</a>
  </div>
  <div class="topbar-right">
    <span class="topbar-status">就绪</span>
  </div>
</header>
<div class="app-layout">

  <!-- 左 -->
  <aside class="panel panel-left">
    <form id="mainForm" action="/submit" method="POST" enctype="multipart/form-data">
      <section class="panel-section">
        <h3>1. 选择图片</h3>
        <div id="dropZone" class="drop-zone">
          <p>点击、拖拽或<strong>粘贴(Ctrl+V)</strong>图片到此处</p>
          <input type="file" id="fileInput" name="image" accept="image/*" required hidden>
          <img id="prev" class="preview-img" hidden>
          <span id="fname" class="file-name"></span>
          <div id="uploadReady" class="upload-ready" hidden>
            <span class="icon">✅</span> 已选择图片
          </div>
        </div>
        <p class="hint">支持 PNG / JPG / WEBP 等常见格式 (不支持 HEIC/SVG)，边缘清晰效果更好</p>
      </section>

      <section class="panel-section">
        <h3>2. 图元</h3>
        <p class="hint">圆形必须有，矩形可禁用</p>
        
        <!-- 圆形选择 -->
        <div class="preset-bar">
          <span class="preset-label">圆形：</span>
          <select name="circle_type" id="circleTypeSelect" class="form-select" style="min-width: 150px; margin-right: 12px; margin-top: 4px; margin-bottom: 4px;">
            <option value="coin">冒险币 1.0×1.0</option>
            <option value="electro_badge">雷元素徽章 0.3×0.3</option>
            <option value="pyro_badge">火元素徽章 0.3×0.3</option>
            <option value="dendro_badge">草元素徽章 0.3×0.3</option>
            <option value="cryo_badge">冰元素徽章 0.3×0.3</option>
            <option value="geo_badge" selected>岩元素徽章 0.3×0.3 (默认低负载)</option>
            <option value="hydro_badge">水元素徽章 0.3×0.3</option>
            <option value="anemo_badge">风元素徽章 0.3×0.3</option>
            <option value="custom">自定义</option>
          </select>
        </div>
        <div id="circleCustomFields" class="custom-fields" hidden>
          <input type="number" id="circleW" value="0.3" min="0.1" max="10" step="0.1" title="宽">
          <span class="prim-x">×</span>
          <input type="number" id="circleH" value="0.3" min="0.1" max="10" step="0.1" title="高">
        </div>
        <div class="color-picker-row">
          <span class="preset-label">圆形颜色：</span>
          <input type="color" id="circleColor" value="#eab308">
        </div>
        
        <!-- 矩形选择 -->
        <div class="preset-bar" style="margin-top:12px;">
          <span class="preset-label">矩形：</span>
          <select name="rect_type" id="rectTypeSelect" class="form-select" style="min-width: 150px; margin-right: 12px; margin-top: 4px; margin-bottom: 4px;">
            <option value="wood_box">木质箱子 1.0×1.0</option>
            <option value="wood_pillar">木质柱子 0.5×5.0</option>
            <option value="custom">自定义</option>
            <option value="disabled" selected>禁用</option>
          </select>
        </div>
        <div id="rectCustomFields" class="custom-fields" hidden>
          <input type="number" id="rectW" value="0.5" min="0.1" max="10" step="0.1" title="宽">
          <span class="prim-x">×</span>
          <input type="number" id="rectH" value="5" min="0.1" max="10" step="0.1" title="高">
        </div>
        <div class="color-picker-row">
          <span class="preset-label">矩形颜色：</span>
          <input type="color" id="rectColor" value="#38bdf8">
        </div>
        
        <input type="hidden" name="primitives_json" id="primJson">
      </section>

      <section class="panel-section">
        <h3>3. 处理参数</h3>
        <p class="hint">推荐先用默认值</p>
        <div class="param-item">
          <div class="param-head">
            <span class="param-title">图元像素大小</span>
            <span id="primSizeVal" class="val-tag">30</span>
          </div>
          <p class="param-desc">控制图元最小/最大尺寸的基准</p>
          <input type="range" name="primitive_size" id="primSize" min="3" max="80" step="1" value="30" class="range-input">
        </div>
        <div class="param-item">
          <div class="param-head">
            <span class="param-title">精度</span>
            <span id="precisionVal" class="val-tag">0.3</span>
          </div>
          <p class="param-desc">越高越贴合轮廓，图元数也会更多</p>
          <input type="range" name="precision" id="precision" min="0" max="1" step="0.1" value="0.3" class="range-input">
        </div>
        <div class="param-item">
          <div class="param-head">
            <span class="param-title">间距</span>
            <span id="spacingVal" class="val-tag">0.9</span>
          </div>
          <p class="param-desc">控制图元之间的紧密程度</p>
          <input type="range" name="spacing" id="spacing" min="0.5" max="1" step="0.05" value="0.9" class="range-input">
        </div>
      </section>

      <section class="panel-section">
        <h3>4. 原点</h3>
        <p class="hint">一般使用图像中心</p>
        <select name="origin_type" id="originType" class="form-select">
            <option value="center">图像中心 (默认)</option>
            <option value="top_left">左上角 (0,0)</option>
            <option value="custom">自定义坐标</option>
        </select>
        <div id="customOrigin" class="config-row" hidden style="margin-top:8px">
            <input type="number" name="origin_x" placeholder="X" step="0.1" style="width:48%">
            <input type="number" name="origin_y" placeholder="Y" step="0.1" style="width:48%">
        </div>
      </section>

      <section class="panel-section section-submit">
        <button type="submit" id="btnSubmit" class="btn-primary">开始处理</button>
      </section>
    </form>
  </aside>

  <!-- 右 -->
  <aside class="panel panel-right">
    <section class="panel-section guide-card">
      <h3>✨ 快速上手</h3>
      <p class="hint">只需三步即可看到结果</p>
      <ol class="steps">
        <li>选择图片</li>
        <li>保留或添加图元</li>
        <li>直接开始处理</li>
      </ol>
      <ul class="tips">
        <li>结果页可右键设置原点</li>
        <li>支持导出 JSON / PNG</li>
        <li>📺 <a href="https://www.bilibili.com/video/BV1RccnzKECg" target="_blank" style="color:inherit;">视频示例</a></li>
        <li>💡 建议上传小尺寸图片，节约性能</li>
        <li>👥 用户QQ群：1007538100</li>
      </ul>
    </section>
    
    <div style="margin-top:auto; padding:16px; text-align:center; font-size:12px; color:var(--text-dim); line-height:1.6;">
      <p>该工具仅供个人兴趣使用，与任何组织无关<br>若侵犯权益，可联系开发者删除</p>
    </div>
  </aside>
</div>

<script src="/web/upload.js?v=2"></script>
</body></html>'''

# ───────────────────────── 等待页 (meta refresh) ─────────────────────────
PAGE_STATUS = r'''<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="1">
<link rel="stylesheet" href="/web/style.css">
</head><body>
<div class="loading-overlay">
  <div class="spinner"></div>
  <h2 style="margin-top:20px;font-weight:600;color:var(--text-main)">处理中… ({{ elapsed }}s)</h2>
  <p style="margin-top:8px;font-size:13px;color:var(--text-dim)">任务 ID: {{ task_id }}</p>
</div>
</body></html>'''

# ───────────────────────── 结果页（三栏 + 全交互） ─────────────────────────
PAGE_RESULT = r'''<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaper — 结果</title>
<link rel="stylesheet" href="/web/style.css">
<script>var RESULT={{ result_json|safe }};var TASK_CFG={{ config_json|safe }};var TASK_ID="{{ task_id }}";</script>
</head><body>
<header class="topbar">
  <div class="topbar-left">
    <a href="/" style="text-decoration:none;"><h1>千星奇域拼好模工具</h1></a>
    <span class="topbar-subtitle">轮廓描边与图元拟合</span>
    <a href="https://ugc.070077.xyz/" target="_blank" class="topbar-link">📚 AI知识库-千星奇域工具箱</a>
    <a href="https://github.com/1475505/Miliastra-toolbox-primitive-shape" target="_blank" class="topbar-link">开源地址</a>
  </div>
  <div class="topbar-right">
    <span id="statusText" class="topbar-status active">完成 — {{ count }} 图元 · {{ elapsed }}s</span>
    <a href="/" class="btn-primary" style="text-decoration:none">新建</a>
  </div>
</header>
<div class="app-layout">

  <!-- 左 -->
  <aside class="panel panel-left">
    <section class="panel-section">
      <h3>导出结果</h3>
      <button id="btnExportJSON" class="btn-sm">导出 JSON</button>
      <button id="btnExportPNG"  class="btn-sm">导出 PNG</button>
      <button id="btnExportGIAOverlimit" class="btn-sm">下载超限模式gia</button>
    </section>
    <section class="panel-section">
      <h3>原点与坐标</h3>
      <p class="hint">右键画布设置原点</p>
      <div class="config-row">
        <label>X</label><input type="number" id="originX" value="0" step="0.1" class="num-input">
        <label style="margin-left:8px">Y</label><input type="number" id="originY" value="0" step="0.1" class="num-input">
      </div>
      <button id="btnResetOrigin" class="btn-sm">重置为图片中心</button>
    </section>

    <section class="panel-section">
      <h3>统计概览</h3>
      <div class="elem-info"><table>
        <tr><td>图元总数</td><td id="statTotal">—</td></tr>
        <tr><td>椭圆</td><td id="statEllipse">—</td></tr>
        <tr><td>矩形</td><td id="statRect">—</td></tr>
        <tr><td>图片尺寸</td><td id="statImgSize">—</td></tr>
      </table></div>
    </section>

    <section class="panel-section">
      <h3>调整参数重新处理</h3>
      <p class="hint">仅影响当前结果</p>
      <form action="/retry/{{ task_id }}" method="POST">
        <div class="config-row">
          <label>图元像素大小</label>
          <input type="number" name="primitive_size" value="{{ cfg_ps }}" min="3" max="80" class="num-input">
        </div>
        <div class="config-row">
          <label>间距</label>
          <input type="number" name="spacing" value="{{ cfg_sp }}" step="0.1" class="num-input">
        </div>
        <div class="config-row">
          <label>精度</label>
          <input type="number" name="precision" value="{{ cfg_pr }}" step="0.1" class="num-input">
        </div>
        <button type="submit" class="btn-primary" style="margin-top:8px">重新处理</button>
      </form>
    </section>

  </aside>

  <!-- 中 -->
  <main class="canvas-area">
    <div id="canvasWrap" class="canvas-wrap">
      <canvas id="mainCanvas"></canvas>
      <div id="tooltip" class="tooltip" hidden></div>
    </div>
    <div class="canvas-bar">
      <span id="coordsDisplay">坐标: —</span>
      <span id="elemCountDisplay">图元: —</span>
      <label><input type="checkbox" id="showImage"> 原图</label>
      <label><input type="checkbox" id="showMask"> 遮罩</label>
      <label><input type="checkbox" id="showFill" checked> 填充</label>
      <label><input type="checkbox" id="showBorder" checked> 描边</label>
      <label><input type="checkbox" id="showOrigin" checked> 原点</label>
    </div>
  </main>

  <!-- 右 -->
  <aside class="panel panel-right">
    <section class="panel-section">
      <a href="/" class="btn-primary" style="display:block;text-align:center;text-decoration:none;margin-bottom:0;">⬅ 再来一张</a>
    </section>
    <section class="panel-section">
      <h3>选中图元详情</h3>
      <p class="hint">点击画布中的图元查看详情</p>
      <div id="infoEmpty" class="empty-panel">
        <p>未选中图元</p>
        <span>点击图元显示详细信息</span>
      </div>
      <div id="infoPanel" class="elem-info" hidden><table>
        <tr><td>ID</td><td id="infoId">—</td></tr>
        <tr><td>类型</td><td id="infoType">—</td></tr>
        <tr><td>中心 (绝对)</td><td id="infoCenter">—</td></tr>
        <tr><td>中心 (原点)</td><td id="infoRelative">—</td></tr>
        <tr><td>尺寸</td><td id="infoSize">—</td></tr>
        <tr><td>旋转</td><td id="infoRotation">—</td></tr>
      </table></div>
    </section>

    <div style="margin-top:auto; padding:16px; text-align:center; font-size:12px; color:var(--text-dim); line-height:1.6;">
      <p>该工具仅供个人研究使用，与任何组织无关<br>若侵犯权益，可联系开发者删除</p>
    </div>
  </aside>

</div>
<script src="/web/app.js?v=9"></script>
</body></html>'''

# ───────────────────────── 路由 ─────────────────────────

@app.route('/')
def index():
    return PAGE_UPLOAD

@app.route('/web/<path:filename>')
def static_file(filename):
    resp = send_from_directory('web', filename)
    if filename.endswith(('.js', '.css')):
        resp.headers['Cache-Control'] = 'no-cache, no-store'
    return resp

@app.route('/submit', methods=['POST'])
def submit():
    cleanup()
    if 'image' not in request.files:
        return '缺少图片', 400
    blob = request.files['image'].read()
    if not blob:
        return '图片为空', 400

    cfg = {
        'primitive_size': float(request.form.get('primitive_size', 30)),
        'spacing':        float(request.form.get('spacing', 0.9)),
        'precision':      float(request.form.get('precision', 0.3)),
        'origin': {
            'type': request.form.get('origin_type', 'center'),
            'x': request.form.get('origin_x', ''),
            'y': request.form.get('origin_y', '')
        }
    }
    try:
        prims = json.loads(request.form.get('primitives_json', '[]'))
        if prims:
            cfg['primitives'] = prims
    except:
        pass

    tid = uuid.uuid4().hex[:8]
    tasks[tid] = {'status': 'processing', 'ts': time.time(),
                  'image_bytes': blob, 'config': cfg}

    def worker():
        try:
            res = shaper_core.process_image(blob, cfg)
            tasks[tid]['result'] = res
            tasks[tid]['status'] = 'done'
        except Exception as e:
            traceback.print_exc()
            tasks[tid]['error'] = str(e)
            tasks[tid]['status'] = 'error'

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f'/status/{tid}')

@app.route('/retry/<tid>', methods=['POST'])
def retry(tid):
    old = tasks.get(tid)
    if not old or 'image_bytes' not in old:
        return redirect('/')
    cfg = {
        'primitive_size': float(request.form.get('primitive_size', 30)),
        'spacing':        float(request.form.get('spacing', 0.9)),
        'precision':      float(request.form.get('precision', 0.3)),
    }
    old_cfg = old.get('config', {})
    if 'primitives' in old_cfg:
        cfg['primitives'] = old_cfg['primitives']
    if 'origin' in old_cfg:
        cfg['origin'] = old_cfg['origin']

    new_id = uuid.uuid4().hex[:8]
    tasks[new_id] = {'status': 'processing', 'ts': time.time(),
                     'image_bytes': old['image_bytes'], 'config': cfg}

    def worker():
        try:
            res = shaper_core.process_image(old['image_bytes'], cfg)
            tasks[new_id]['result'] = res
            tasks[new_id]['status'] = 'done'
        except Exception as e:
            traceback.print_exc()
            tasks[new_id]['error'] = str(e)
            tasks[new_id]['status'] = 'error'

    threading.Thread(target=worker, daemon=True).start()
    return redirect(f'/status/{new_id}')

@app.route('/status/<tid>')
def status(tid):
    t = tasks.get(tid)
    if not t:
        return redirect('/')
    if t['status'] == 'done':
        return redirect(f'/result/{tid}')
    if t['status'] == 'error':
        return f'<h2>出错</h2><p>{t.get("error")}</p><a href="/">返回</a>'
    elapsed = int(time.time() - t['ts'])
    return render_template_string(PAGE_STATUS, task_id=tid, elapsed=elapsed)

@app.route('/result/<tid>')
def result(tid):
    t = tasks.get(tid)
    if not t or 'result' not in t:
        return redirect('/')
    res = t['result']
    cfg = t['config']
    return render_template_string(PAGE_RESULT,
        result_json=json.dumps(res),
        config_json=json.dumps(cfg),
        task_id=tid,
        count=res['elements_count'],
        elapsed=res['elapsed_seconds'],
        cfg_ps=cfg.get('primitive_size', 30),
        cfg_sp=cfg.get('spacing', 0.9),
        cfg_pr=cfg.get('precision', 0.3))

_json_to_gia_mod = None

def _load_json_to_gia():
    global _json_to_gia_mod
    if _json_to_gia_mod is not None:
        return _json_to_gia_mod
    
    # Add gia directory to sys.path
    gia_dir = os.path.join(BASE_DIR, 'gia')
    if gia_dir not in sys.path:
        sys.path.insert(0, gia_dir)
    
    try:
        import json_to_gia
        _json_to_gia_mod = json_to_gia
        return _json_to_gia_mod
    except ImportError as e:
        raise RuntimeError(f'无法加载 json_to_gia 模块: {e}')

@app.route('/download_overlimit_gia/<tid>')
def download_overlimit_gia(tid):
    t = tasks.get(tid)
    if not t or 'result' not in t:
        return '任务不存在', 404
    res = t['result']
    cfg = t.get('config', {})
    ps = float(cfg.get('primitive_size', 1) or 1)
    origin_default = res.get('image_center', {'x': 0, 'y': 0})
    try:
        origin_x = float(request.args.get('origin_x', origin_default.get('x', 0)))
        origin_y = float(request.args.get('origin_y', origin_default.get('y', 0)))
    except:
        return 'origin 参数无效', 400

    # 获取primitives配置（包含预设信息）
    primitives = cfg.get('primitives', [])
    prim_map = {}
    for p in primitives:
        shape = p.get('shape')
        if shape:
            prim_map[shape] = p

    ox = origin_x / ps
    oy = -origin_y / ps
    elements = []
    for e in res.get('elements', []):
        c = e.get('center', {}) or {}
        rel = {'x': float(c.get('x', 0)) - ox, 'y': float(c.get('y', 0)) - oy}
        
        elem_data = {'type': e.get('type'), 'center': rel, 'size': e.get('size', {}), 'rotation': e.get('rotation', {})}
        
        # 根据元素类型获取对应的预设信息
        elem_type = e.get('type')
        if elem_type == 'ellipse':
            preset = prim_map.get('circle', {})
        elif elem_type == 'rectangle':
            preset = prim_map.get('rect', {})
        else:
            preset = {}
        
        # 添加预设信息到元素数据
        if preset.get('type_id'):
            elem_data['type_id'] = preset['type_id']
        if preset.get('rot_z'):
            elem_data['rot_z'] = preset['rot_z']
        if preset.get('rot_y_add'):
            elem_data['rot_y_add'] = preset['rot_y_add']
        
        elements.append(elem_data)

    json_data = {'elements': elements}
    base_gia_path = os.path.join(BASE_DIR, 'gia', 'template.gia')
    mod = _load_json_to_gia()
    gia_bytes = mod.convert_json_to_gia_bytes(json_data=json_data, base_gia_path=base_gia_path)

    resp = Response(gia_bytes, mimetype='application/octet-stream')
    resp.headers['Content-Disposition'] = f'attachment; filename="overlimit_{tid}.gia"'
    return resp

if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5555'))
    print(f'🎨 Shaper  http://localhost:{port}')
    app.run(host='0.0.0.0', port=port, threaded=True)
