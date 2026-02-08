/* Shaper v8 — 结果页交互 (数据由服务端注入 window.RESULT) */
(() => {
"use strict";
const $ = id => document.getElementById(id);
const data = window.RESULT;
if (!data) return;

/* ── 状态 ── */
const W = data.image_size.width, H = data.image_size.height;
let baseImg = null, maskImg = null;
let origin = { x: data.image_center.x, y: data.image_center.y };
let scale = 1, hovered = null, selected = null;

const canvas = $('mainCanvas'), ctx = canvas.getContext('2d');

/* ── 加载图片 ── */
let loaded = 0;
function onLoad() { if (++loaded >= 2) { initUI(); render(); } }
baseImg = new Image(); baseImg.onload = onLoad;
baseImg.src = 'data:image/png;base64,' + data.image_base64;
maskImg = new Image(); maskImg.onload = onLoad;
maskImg.src = 'data:image/png;base64,' + data.mask_base64;

/* ── 初始化 UI ── */
function initUI() {
  // 原点
  $('originX').value = origin.x;
  $('originY').value = origin.y;

  // 统计
  var nE = 0, nR = 0;
  data.elements.forEach(e => e.type === 'ellipse' ? nE++ : nR++);
  $('statTotal').textContent   = data.elements.length;
  $('statEllipse').textContent = nE;
  $('statRect').textContent    = nR;
  $('statImgSize').textContent = W + '×' + H;
  $('elemCountDisplay').textContent = '图元: ' + data.elements.length;
}

/* ── 渲染 ── */
function render() {
  var wrap = $('canvasWrap');
  scale = Math.min(1, (wrap.clientWidth - 40) / W, (wrap.clientHeight - 40) / H);
  canvas.width  = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  // 底图
  if ($('showImage').checked) ctx.drawImage(baseImg, 0, 0, W, H);
  else { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H); }

  // Mask
  if ($('showMask').checked && maskImg) {
    ctx.globalAlpha = 0.3; ctx.drawImage(maskImg, 0, 0, W, H); ctx.globalAlpha = 1;
  }

  // 图元
  data.elements.forEach((el, i) => drawElem(el, i === hovered || i === selected));

  // 原点十字
  ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5 / scale;
  var L = 12 / scale;
  ctx.beginPath();
  ctx.moveTo(origin.x - L, origin.y); ctx.lineTo(origin.x + L, origin.y);
  ctx.moveTo(origin.x, origin.y - L); ctx.lineTo(origin.x, origin.y + L);
  ctx.stroke();
  ctx.fillStyle = '#ef4444'; ctx.font = (10 / scale) + 'px sans-serif';
  ctx.fillText('原点', origin.x + L + 2, origin.y - 3);

  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawElem(e, hl) {
  ctx.save();
  ctx.translate(e.center.x, e.center.y);
  ctx.rotate((e.rotation || 0) * Math.PI / 180);
  ctx.beginPath();
  if (e.type === 'ellipse') ctx.ellipse(0, 0, e.size.rx, e.size.ry, 0, 0, Math.PI * 2);
  else { var hw = e.size.width / 2, hh = e.size.height / 2; ctx.rect(-hw, -hh, hw * 2, hh * 2); }
  if ($('showFill').checked) {
    ctx.fillStyle = hl ? 'rgba(59,130,246,0.5)' : 'rgba(255,204,0,0.35)'; ctx.fill();
  }
  if ($('showBorder').checked) {
    ctx.strokeStyle = hl ? '#2563eb' : 'rgba(255,170,0,0.7)';
    ctx.lineWidth = hl ? 1.5 : 0.8; ctx.stroke();
  }
  ctx.restore();
}

/* ── 显示控制 ── */
['showImage','showMask','showFill','showBorder'].forEach(id => $(id).addEventListener('change', render));
window.addEventListener('resize', render);

/* ── 鼠标交互 ── */
canvas.addEventListener('mousemove', function(ev) {
  var p = px(ev);
  $('coordsDisplay').textContent = '坐标: (' + p.x.toFixed(1) + ', ' + p.y.toFixed(1) + ')';
  var idx = hitTest(p.x, p.y);
  if (idx !== hovered) { hovered = idx; render(); }
  // tooltip
  var tip = $('tooltip');
  if (idx !== null) {
    var el = data.elements[idx];
    var rx = (el.center.x - origin.x).toFixed(1), ry = (el.center.y - origin.y).toFixed(1);
    var sz = el.type === 'ellipse' ? 'rx=' + el.size.rx + ' ry=' + el.size.ry
                                    : el.size.width + '×' + el.size.height;
    tip.innerHTML = '<b>#' + idx + '</b> ' + el.type + '<br>'
      + '中心: (' + el.center.x + ', ' + el.center.y + ')<br>'
      + '相对原点: (' + rx + ', ' + ry + ')<br>'
      + '尺寸: ' + sz + '<br>'
      + '旋转: ' + (el.rotation||0).toFixed(1) + '°';
    tip.hidden = false;
    var r = $('canvasWrap').getBoundingClientRect();
    tip.style.left = (ev.clientX - r.left + 14) + 'px';
    tip.style.top  = (ev.clientY - r.top + 14)  + 'px';
  } else tip.hidden = true;
});

canvas.addEventListener('mouseleave', function() {
  $('tooltip').hidden = true;
  if (hovered !== null) { hovered = null; render(); }
});

canvas.addEventListener('click', function(ev) {
  var p = px(ev), idx = hitTest(p.x, p.y);
  selected = idx; render();
  if (idx !== null) showDetail(data.elements[idx], idx);
});

canvas.addEventListener('contextmenu', function(ev) {
  ev.preventDefault();
  var p = px(ev);
  origin = { x: p.x, y: p.y };
  $('originX').value = p.x.toFixed(1);
  $('originY').value = p.y.toFixed(1);
  render();
});

/* ── 碰撞检测 ── */
function px(ev) {
  var r = canvas.getBoundingClientRect();
  return { x: (ev.clientX - r.left) * W / canvas.width,
           y: (ev.clientY - r.top)  * H / canvas.height };
}

function hitTest(px, py) {
  for (var i = data.elements.length - 1; i >= 0; i--) {
    var e = data.elements[i];
    var rot = -(e.rotation||0) * Math.PI / 180;
    var dx0 = px - e.center.x, dy0 = py - e.center.y;
    var dx = dx0*Math.cos(rot) - dy0*Math.sin(rot);
    var dy = dx0*Math.sin(rot) + dy0*Math.cos(rot);
    if (e.type === 'ellipse') {
      if (dx*dx/(e.size.rx*e.size.rx) + dy*dy/(e.size.ry*e.size.ry) <= 1) return i;
    } else {
      if (Math.abs(dx) <= e.size.width/2 && Math.abs(dy) <= e.size.height/2) return i;
    }
  }
  return null;
}

/* ── 详情面板 ── */
function showDetail(el, idx) {
  var rx = (el.center.x - origin.x).toFixed(2);
  var ry = (el.center.y - origin.y).toFixed(2);
  $('infoId').textContent       = el.id || idx;
  $('infoType').textContent     = el.type;
  $('infoCenter').textContent   = '(' + el.center.x + ', ' + el.center.y + ')';
  $('infoRelative').textContent = '(' + rx + ', ' + ry + ')';
  $('infoSize').textContent     = el.type === 'ellipse'
    ? 'rx=' + el.size.rx + '  ry=' + el.size.ry
    : el.size.width + ' × ' + el.size.height;
  $('infoRotation').textContent = (el.rotation||0).toFixed(1) + '°';
}

/* ── 原点控制 ── */
$('originX').addEventListener('change', function() { origin.x = +this.value; render(); });
$('originY').addEventListener('change', function() { origin.y = +this.value; render(); });
$('btnResetOrigin').addEventListener('click', function() {
  origin = { x: data.image_center.x, y: data.image_center.y };
  $('originX').value = origin.x; $('originY').value = origin.y; render();
});

/* ── 导出 ── */
$('btnExportJSON').addEventListener('click', function() {
  var out = data.elements.map(function(e, i) {
    return { id: e.id||i, type: e.type, center: e.center,
      relative: { x: +(e.center.x-origin.x).toFixed(2), y: +(e.center.y-origin.y).toFixed(2) },
      size: e.size, rotation: e.rotation };
  });
  dl(new Blob([JSON.stringify({ origin:origin, image_size:data.image_size,
    config:window.TASK_CFG, elements:out }, null, 2)], {type:'application/json'}), 'shaper_result.json');
});
$('btnExportPNG').addEventListener('click', function() {
  canvas.toBlob(function(b) { dl(b, 'shaper_result.png'); }, 'image/png');
});
function dl(blob, name) {
  var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}
})();
