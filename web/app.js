/* Shaper v8 — 结果页交互 (数据由服务端注入 window.RESULT) */
(() => {
"use strict";
const $ = id => document.getElementById(id);
const data = window.RESULT;
if (!data) return;
const ps = data.config.primitive_size || 1;

/* ── 状态 ── */
const W = data.image_size.width, H = data.image_size.height;
let baseImg = null, maskImg = null;
let origin = { x: data.image_center.x, y: data.image_center.y };
let scale = 1, hovered = null, selected = null;

const canvas = $('mainCanvas'), ctx = canvas.getContext('2d');

/* ── 加载图片 ── */
let loaded = 0;
let expected = 1;
function onAsset() { if (++loaded >= expected) { initUI(); render(); } }
baseImg = new Image(); baseImg.onload = onAsset; baseImg.onerror = onAsset;
if (data.image_base64) {
  baseImg.src = 'data:image/png;base64,' + data.image_base64;
} else {
  onAsset();
}
if (data.mask_base64) {
  expected = 2;
  maskImg = new Image(); maskImg.onload = onAsset; maskImg.onerror = onAsset;
  maskImg.src = 'data:image/png;base64,' + data.mask_base64;
}

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
  clearDetail();
}

/* ── 渲染 ── */
function render() {
  var wrap = $('canvasWrap');
  scale = Math.min(1, (wrap.clientWidth - 40) / W, (wrap.clientHeight - 40) / H);
  canvas.width  = Math.round(W * scale);
  canvas.height = Math.round(H * scale);
  ctx.setTransform(scale, 0, 0, scale, 0, 0);

  // 底图
  if ($('showImage').checked && baseImg && baseImg.complete) ctx.drawImage(baseImg, 0, 0, W, H);
  else { ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H); }

  // Mask
  if ($('showMask').checked && maskImg) {
    ctx.globalAlpha = 0.3; ctx.drawImage(maskImg, 0, 0, W, H); ctx.globalAlpha = 1;
  }

  // 图元
  // 渲染逻辑：如果已选中 (selected !== null)，则优先高亮选中项，不显示 hover 高亮
  // 如果未选中，则显示 hover 高亮
  data.elements.forEach((el, i) => {
    var isSelected = (i === selected);
    // 只有在没有选中项时，才显示 hover 高亮；或者 hover 的就是选中项（虽然此时样式一样）
    // 为了消除晃动，当 selected 有值时，我们忽略 hovered 的高亮，或者仅当 hovered == selected 时才高亮
    // 用户说“选中图元还是会晃动”，可能是指 tooltip 或者是 hover 样式冲突
    // 策略：如果 selected != null，则只高亮 selected。hovered 仅用于 tooltip (如果需要)
    // 但用户说“选中图元后不要移动”，可能指 tooltip 不要跟手？
    
    // 这里我们修改逻辑：
    // 1. 选中态 (selected) 优先级最高，且样式固定
    // 2. 悬浮态 (hovered) 仅在没有选中项，或者悬浮项 != 选中项时显示？
    // 通常逻辑是：选中一项高亮；悬浮另一项也高亮（辅助）。
    // 但用户觉得“晃动”，可能是指在选中项上移动鼠标时，hovered 状态反复触发导致重绘或样式跳变。
    // 我们统一：如果 i === selected，则 isHigh = true。
    // 如果 i === hovered 且 selected === null，则 isHigh = true。
    // 也就是说，选中时，鼠标在上面移动不会改变样式（因为已经是高亮了）。
    
    var highlight = isSelected || (selected === null && i === hovered);
    drawElem(el, highlight, isSelected); 
  });

  // 原点十字
  if ($('showOrigin').checked) {
    ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 1.5 / scale;
    var L = 12 / scale;
    ctx.beginPath();
    ctx.moveTo(origin.x - L, origin.y); ctx.lineTo(origin.x + L, origin.y);
    ctx.moveTo(origin.x, origin.y - L); ctx.lineTo(origin.x, origin.y + L);
    ctx.stroke();
    ctx.fillStyle = '#ef4444'; ctx.font = (10 / scale) + 'px sans-serif';
    ctx.fillText('原点', origin.x + L + 2, origin.y - 3);
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawElem(e, hl, isSel) {
  ctx.save();
  ctx.translate(e.center.x * ps, -e.center.y * ps);
  var rot = e.rotation ? (e.rotation.z || 0) : 0;
  ctx.rotate(-rot * Math.PI / 180);
  ctx.beginPath();
  if (e.type === 'ellipse') ctx.ellipse(0, 0, e.size.rx * ps, e.size.ry * ps, 0, 0, Math.PI * 2);
  else { var hw = (e.size.width * ps) / 2, hh = (e.size.height * ps); ctx.rect(-hw, -hh, hw * 2, hh); }
  
  if ($('showFill').checked) {
    if (isSel) {
      ctx.fillStyle = 'rgba(59,130,246,0.6)'; // 选中态颜色加深
    } else if (hl) {
      ctx.fillStyle = 'rgba(59,130,246,0.4)'; // 悬浮态
    } else if (e.color) {
      ctx.fillStyle = hexToRgba(e.color, 0.45);
    } else {
      ctx.fillStyle = 'rgba(255,204,0,0.35)';
    }
    ctx.fill();
  }
  
  if ($('showBorder').checked) {
    if (isSel) {
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2.0; // 选中态边框加粗
    } else if (hl) {
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 1.5;
    } else if (e.color) {
      ctx.strokeStyle = e.color;
      ctx.lineWidth = 0.8;
    } else {
      ctx.strokeStyle = 'rgba(255,170,0,0.7)';
      ctx.lineWidth = 0.8;
    }
    ctx.stroke();
  }
  ctx.restore();
}

function hexToRgba(hex, alpha) {
    var c;
    if(/^#([A-Fa-f0-9]{3}){1,2}$/.test(hex)){
        c= hex.substring(1).split('');
        if(c.length== 3){
            c= [c[0], c[0], c[1], c[1], c[2], c[2]];
        }
        c= '0x'+c.join('');
        return 'rgba('+[(c>>16)&255, (c>>8)&255, c&255].join(',')+','+alpha+')';
    }
    return hex; // fallback
}

/* ── 显示控制 ── */
['showImage','showMask','showFill','showBorder','showOrigin'].forEach(id => $(id).addEventListener('change', render));
window.addEventListener('resize', render);

/* ── 鼠标交互 ── */
canvas.addEventListener('mousemove', function(ev) {
  var p = px(ev);
  var ux = p.x / ps;
  var uy = -p.y / ps;
  $('coordsDisplay').textContent = '坐标: (' + ux.toFixed(2) + ', ' + uy.toFixed(2) + ')';
  
  // 如果已经选中了某个图元，就不再更新 hovered，避免样式跳变
  // 但这样会导致用户无法感知其他图元。
  // 根据用户反馈“选中图元还是会晃动”，我们假设用户是指选中项本身的样式不稳定
  // 前面的 render 逻辑已经确保选中项样式最高优先级。
  // 此外，tooltip 也是晃动的来源。
  
  var idx = hitTest(p.x, p.y);
  if (idx !== hovered) { hovered = idx; render(); }
  
  // Tooltip 逻辑优化：
  // 1. 如果已选中某项，且鼠标还在该项上，则不显示 tooltip（或显示静态的）？
  // 用户说“选中图元后不要移动”，可能指 tooltip 不要跟随鼠标。
  // 策略：当 selected !== null 且 hovered === selected 时，隐藏 tooltip（因为右侧已经有详情了）
  // 或者固定 tooltip 位置。隐藏是最简单的，因为右侧面板已经展开。
  
  var tip = $('tooltip');
  if (idx !== null && idx !== selected) {
    // 仅当 hover 的不是当前选中项时显示 tooltip
    // 或者当没有选中项时显示
    // 这样选中项就不会有 tooltip 跟随晃动
    var el = data.elements[idx];
    var ox = origin.x / ps, oy = -origin.y / ps;
    var rx = (el.center.x - ox).toFixed(2), ry = (el.center.y - oy).toFixed(2);
    var sz = el.type === 'ellipse' ? 'rx=' + el.size.rx.toFixed(2) + ' ry=' + el.size.ry.toFixed(2)
                                    : el.size.width.toFixed(2) + '×' + el.size.height.toFixed(2);
    var rot = el.rotation ? (el.rotation.z || 0) : 0;
    
    tip.innerHTML = '<b>#' + idx + '</b> ' + el.type + '<br>'
      + '中心: (' + el.center.x.toFixed(2) + ', ' + el.center.y.toFixed(2) + ')<br>'
      + '相对原点: (' + rx + ', ' + ry + ')<br>'
      + '尺寸: ' + sz + '<br>'
      + '旋转: ' + rot.toFixed(1) + '°';
    tip.hidden = false;
    var r = $('canvasWrap').getBoundingClientRect();
    tip.style.left = (ev.clientX - r.left + 14) + 'px';
    tip.style.top  = (ev.clientY - r.top + 14)  + 'px';
  } else {
    tip.hidden = true;
  }
});

canvas.addEventListener('mouseleave', function() {
  $('tooltip').hidden = true;
  if (hovered !== null) { hovered = null; render(); }
});

canvas.addEventListener('click', function(ev) {
  var p = px(ev), idx = hitTest(p.x, p.y);
  selected = idx;
  hovered = null;
  render();
  if (idx !== null) showDetail(data.elements[idx], idx);
  else clearDetail();
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
  return { x: (ev.clientX - r.left) * W / r.width,
           y: (ev.clientY - r.top)  * H / r.height };
}

function hitTest(px, py) {
  for (var i = data.elements.length - 1; i >= 0; i--) {
    var e = data.elements[i];
    var rot = ((e.rotation ? (e.rotation.z || 0) : 0)) * Math.PI / 180;
    var cx = e.center.x * ps, cy = -e.center.y * ps;
    var dx0 = px - cx, dy0 = py - cy;
    var dx = dx0*Math.cos(rot) - dy0*Math.sin(rot);
    var dy = dx0*Math.sin(rot) + dy0*Math.cos(rot);
    if (e.type === 'ellipse') {
      var rx = e.size.rx * ps, ry = e.size.ry * ps;
      if (dx*dx/(rx*rx) + dy*dy/(ry*ry) <= 1) return i;
    } else {
      var w = e.size.width * ps, h = e.size.height * ps;
      if (Math.abs(dx) <= w/2 && dy <= 0 && dy >= -h) return i;
    }
  }
  return null;
}

/* ── 详情面板 ── */
function showDetail(el, idx) {
  // 强制确保元素存在再操作
  var pEmpty = $('infoEmpty');
  var pPanel = $('infoPanel');
  if (pEmpty) pEmpty.hidden = true;
  if (pPanel) pPanel.hidden = false;
  
  // 更新数据
  $('infoId').textContent       = el.id || idx;
  $('infoType').textContent     = el.type;
  $('infoCenter').textContent   = '(' + el.center.x.toFixed(2) + ', ' + el.center.y.toFixed(2) + ')';
  
  var relX = el.center.x - origin.x / ps;
  var relY = el.center.y + origin.y / ps;
  $('infoRelative').textContent = '(' + relX.toFixed(2) + ', ' + relY.toFixed(2) + ')';
  
  $('infoSize').textContent     = el.type === 'ellipse'
    ? 'rx=' + el.size.rx.toFixed(2) + '  ry=' + el.size.ry.toFixed(2)
    : el.size.width.toFixed(2) + ' × ' + el.size.height.toFixed(2);
    
  var rot = el.rotation ? (el.rotation.z || 0) : 0;
  $('infoRotation').textContent = rot.toFixed(1) + '°';
}

function clearDetail() {
  var pEmpty = $('infoEmpty');
  var pPanel = $('infoPanel');
  if (pEmpty) pEmpty.hidden = false;
  if (pPanel) pPanel.hidden = true;
  
  $('infoId').textContent = '—';
  $('infoType').textContent = '—';
  $('infoCenter').textContent = '—';
  $('infoRelative').textContent = '—';
  $('infoSize').textContent = '—';
  $('infoRotation').textContent = '—';
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
  var ox = origin.x / ps, oy = -origin.y / ps;
  var out = data.elements.map(function(e, i) {
    var relX = +(e.center.x - ox).toFixed(4);
    var relY = +(e.center.y - oy).toFixed(4);
    return { id: e.id||i, type: e.type, center: e.center,
      relative: { x: relX, y: relY },
      size: e.size, rotation: e.rotation };
  });
  dl(new Blob([JSON.stringify({ origin:{ x: ox, y: oy }, image_size:data.image_size,
    config:window.TASK_CFG, elements:out }, null, 2)], {type:'application/json'}), 'shaper_result.json');
});
$('btnExportPNG').addEventListener('click', function() {
  canvas.toBlob(function(b) { dl(b, 'shaper_result.png'); }, 'image/png');
});
var btnGIA = $('btnExportGIAOverlimit');
if (btnGIA) btnGIA.addEventListener('click', function() {
  var tid = window.TASK_ID || '';
  var qs = new URLSearchParams({ origin_x: origin.x, origin_y: origin.y }).toString();
  fetch('/download_overlimit_gia/' + encodeURIComponent(tid) + '?' + qs)
    .then(function(resp) {
      if (!resp.ok) return resp.text().then(function(t) { throw new Error(t || ('HTTP ' + resp.status)); });
      return resp.blob();
    })
    .then(function(blob) { dl(blob, '超限模式_' + (tid || 'result') + '.gia'); })
    .catch(function(err) { alert('下载失败: ' + (err && err.message ? err.message : err)); });
});
function dl(blob, name) {
  var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = name; a.click(); URL.revokeObjectURL(a.href);
}
})();
