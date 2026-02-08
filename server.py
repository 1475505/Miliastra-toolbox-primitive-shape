"""
Shaper Web Server — Flask MPA
后端: 表单 POST → 重定向状态页(meta refresh) → 重定向结果页
前端: 三栏布局, 图元定义, 原点控制, 悬浮/选中, 导出
端口: 5555
"""

import os, sys, json, uuid, traceback, threading, time
from flask import (Flask, request, redirect, send_from_directory,
                   render_template_string, Response)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shaper_core

app = Flask(__name__, static_folder='web', static_url_path='/web')
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
</head><body>
<header class="topbar">
  <h1>🎨 Shaper</h1>
  <span class="topbar-subtitle">轮廓描边工具</span>
  <span class="topbar-status">就绪</span>
</header>
<div class="app-layout">

  <!-- 左 -->
  <aside class="panel panel-left">
    <form id="mainForm" action="/submit" method="POST" enctype="multipart/form-data">
      <section class="panel-section">
        <h3>📁 输入图片</h3>
        <div class="drop-zone" onclick="document.getElementById('fileInput').click()">
          <p>拖放图片或 <span class="file-label">选择文件</span></p>
          <input type="file" id="fileInput" name="image" accept="image/*" required hidden
                 onchange="var f=this.files[0];if(f){document.getElementById('fname').textContent=f.name;document.getElementById('prev').src=URL.createObjectURL(f);document.getElementById('prev').hidden=false;}">
          <img id="prev" class="preview-img" hidden>
          <span id="fname" class="file-name"></span>
        </div>
      </section>

      <section class="panel-section">
        <h3>🧩 图元定义</h3>
        <p class="hint">定义用于拟合的基础图元</p>
        <div id="primList"></div>
        <button type="button" id="btnAddPrim" class="btn-sm" style="margin-top:6px">+ 添加图元</button>
        <input type="hidden" name="primitives_json" id="primJson">
      </section>

      <section class="panel-section">
        <h3>⚙️ 参数</h3>
        <div class="config-row">
          <label>图元大小 <span id="primSizeVal" class="val-tag">15</span></label>
          <input type="range" name="primitive_size" id="primSize" min="3" max="80" step="1" value="15" class="range-input">
        </div>
        <div class="config-row">
          <label>精度 <span id="precisionVal" class="val-tag">0.3</span></label>
          <input type="range" name="precision" id="precision" min="0" max="1" step="0.1" value="0.3" class="range-input">
        </div>
        <div class="config-row">
          <label>间距 <span id="spacingVal" class="val-tag">0.9</span></label>
          <input type="range" name="spacing" id="spacing" min="0.5" max="1" step="0.05" value="0.9" class="range-input">
        </div>
      </section>

      <section class="panel-section">
        <button type="submit" class="btn-primary">▶ 开始处理</button>
      </section>
    </form>
  </aside>

  <!-- 中 -->
  <main class="canvas-area">
    <div class="canvas-wrap">
      <div class="empty-hint"><p>👈 上传图片并点击处理</p></div>
    </div>
    <div class="canvas-bar"><span>坐标: —</span></div>
  </main>

  <!-- 右 -->
  <aside class="panel panel-right">
    <section class="panel-section">
      <h3>📍 使用说明</h3>
      <p class="hint" style="line-height:1.6">
        1. 左侧上传图片<br>
        2. 配置图元与参数<br>
        3. 点击「开始处理」<br>
        4. 等待自动跳转到结果页<br>
        5. 在结果页中交互查看<br>
        6. 右键画布设置原点<br>
        7. 导出 JSON / PNG
      </p>
    </section>
  </aside>
</div>

<script>
// 图元卡片
var primList = document.getElementById('primList');
function addPrim(shape, w, h, color) {
  shape=shape||'circle'; w=w||1; h=h||1; color=color||'#ffcc00';
  var d=document.createElement('div'); d.className='prim-card';
  d.innerHTML='<select data-f="shape"><option value="circle"'+(shape==='circle'?' selected':'')+'>圆形</option><option value="rect"'+(shape==='rect'?' selected':'')+'>矩形</option></select>'
    +'<input type="number" data-f="w" value="'+w+'" min="1" max="10" title="宽">'
    +'<span class="prim-x">×</span>'
    +'<input type="number" data-f="h" value="'+h+'" min="1" max="10" title="高">'
    +'<input type="color" data-f="color" value="'+color+'">'
    +'<button type="button" class="btn-del" onclick="this.parentNode.remove()">✕</button>';
  primList.appendChild(d);
}
document.getElementById('btnAddPrim').onclick=function(){addPrim()};
addPrim();

// 滑块标签
['primSize','precision','spacing'].forEach(function(id){
  var el=document.getElementById(id),tag=document.getElementById(id+'Val');
  if(el&&tag) el.oninput=function(){tag.textContent=el.value};
});

// 提交前序列化图元
document.getElementById('mainForm').onsubmit=function(){
  var arr=[];
  primList.querySelectorAll('.prim-card').forEach(function(c){
    arr.push({shape:c.querySelector('[data-f=shape]').value,
      w:parseInt(c.querySelector('[data-f=w]').value)||1,
      h:parseInt(c.querySelector('[data-f=h]').value)||1,
      color:c.querySelector('[data-f=color]').value});
  });
  document.getElementById('primJson').value=JSON.stringify(arr);
};
</script>
</body></html>'''

# ───────────────────────── 等待页 (meta refresh) ─────────────────────────
PAGE_STATUS = r'''<!DOCTYPE html><html><head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="1">
<link rel="stylesheet" href="/web/style.css">
<style>
body{display:flex;justify-content:center;align-items:center;height:100vh;flex-direction:column;background:#1e1e1e;color:#ccc}
.spinner{border:4px solid #333;border-top:4px solid #3b82f6;border-radius:50%;width:50px;height:50px;animation:spin 1s linear infinite;margin-bottom:20px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head><body>
<div class="spinner"></div>
<h2>处理中… ({{ elapsed }}s)</h2>
<p style="margin-top:12px;font-size:13px;opacity:.5">完成后自动跳转</p>
</body></html>'''

# ───────────────────────── 结果页（三栏 + 全交互） ─────────────────────────
PAGE_RESULT = r'''<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaper — 结果</title>
<link rel="stylesheet" href="/web/style.css">
<script>var RESULT={{ result_json|safe }};var TASK_CFG={{ config_json|safe }};</script>
</head><body>
<header class="topbar">
  <h1>🎨 Shaper</h1>
  <span class="topbar-subtitle">轮廓描边工具</span>
  <span id="statusText" class="topbar-status active">完成 — {{ count }} 图元 · {{ elapsed }}s</span>
  <a href="/" class="btn-sm" style="text-decoration:none;margin-left:12px;width:auto;padding:4px 12px">🏠 新建</a>
</header>
<div class="app-layout">

  <!-- 左 -->
  <aside class="panel panel-left">
    <section class="panel-section">
      <h3>📍 原点</h3>
      <p class="hint">右键画布设置原点</p>
      <div class="config-row">
        <label>X</label><input type="number" id="originX" value="0" step="0.1" class="num-input">
        <label style="margin-left:8px">Y</label><input type="number" id="originY" value="0" step="0.1" class="num-input">
      </div>
      <button id="btnResetOrigin" class="btn-sm">重置为图片中心</button>
    </section>

    <section class="panel-section">
      <h3>📊 统计</h3>
      <div class="elem-info"><table>
        <tr><td>图元总数</td><td id="statTotal">—</td></tr>
        <tr><td>椭圆</td><td id="statEllipse">—</td></tr>
        <tr><td>矩形</td><td id="statRect">—</td></tr>
        <tr><td>图片尺寸</td><td id="statImgSize">—</td></tr>
      </table></div>
    </section>

    <section class="panel-section">
      <h3>⚙️ 参数重试</h3>
      <form action="/retry/{{ task_id }}" method="POST">
        <div class="config-row">
          <label>图元大小</label>
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
        <button type="submit" class="btn-primary" style="margin-top:8px">🔄 重新处理</button>
      </form>
    </section>

    <section class="panel-section">
      <h3>💾 导出</h3>
      <button id="btnExportJSON" class="btn-sm">导出 JSON</button>
      <button id="btnExportPNG"  class="btn-sm">导出 PNG</button>
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
      <label><input type="checkbox" id="showImage" checked> 原图</label>
      <label><input type="checkbox" id="showMask"> Mask</label>
      <label><input type="checkbox" id="showFill" checked> 填充</label>
      <label><input type="checkbox" id="showBorder" checked> 描边</label>
    </div>
  </main>

  <!-- 右 -->
  <aside class="panel panel-right">
    <section class="panel-section">
      <h3>🔍 图元详情</h3>
      <p class="hint">悬停或点击画布上的图元</p>
      <div class="elem-info"><table>
        <tr><td>ID</td><td id="infoId">—</td></tr>
        <tr><td>类型</td><td id="infoType">—</td></tr>
        <tr><td>中心 (绝对)</td><td id="infoCenter">—</td></tr>
        <tr><td>中心 (原点)</td><td id="infoRelative">—</td></tr>
        <tr><td>尺寸</td><td id="infoSize">—</td></tr>
        <tr><td>旋转</td><td id="infoRotation">—</td></tr>
      </table></div>
    </section>
  </aside>

</div>
<script src="/web/app.js?v=8"></script>
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
        'primitive_size': float(request.form.get('primitive_size', 15)),
        'spacing':        float(request.form.get('spacing', 0.9)),
        'precision':      float(request.form.get('precision', 0.3)),
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
        'primitive_size': float(request.form.get('primitive_size', 15)),
        'spacing':        float(request.form.get('spacing', 0.9)),
        'precision':      float(request.form.get('precision', 0.3)),
    }
    old_cfg = old.get('config', {})
    if 'primitives' in old_cfg:
        cfg['primitives'] = old_cfg['primitives']

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
        cfg_ps=cfg.get('primitive_size', 15),
        cfg_sp=cfg.get('spacing', 0.9),
        cfg_pr=cfg.get('precision', 0.3))

if __name__ == '__main__':
    print('🎨 Shaper  http://localhost:5555')
    app.run(host='0.0.0.0', port=5555, threaded=True)
