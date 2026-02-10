(function() {
  'use strict';

  // Elements
  var dropZone = document.getElementById('dropZone');
  var fileInput = document.getElementById('fileInput');
  var fname = document.getElementById('fname');
  var prev = document.getElementById('prev');
  var uploadReady = document.getElementById('uploadReady');
  var btnSubmit = document.getElementById('btnSubmit');
  var mainForm = document.getElementById('mainForm');
  var primList = document.getElementById('primList');
  var primJson = document.getElementById('primJson');

  // File Handling Logic
  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    // Update fileInput files for form submission
    var dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;

    fname.textContent = file.name;
    prev.src = URL.createObjectURL(file);
    prev.hidden = false;
    
    uploadReady.innerHTML = '<span class="icon">✅</span> 已选择图片'; 
    uploadReady.hidden = false;
    
    dropZone.classList.add('ready');
    btnSubmit.classList.add('ready');
    btnSubmit.innerHTML = '开始处理 &rarr;';
  }

  // File Input Handling
  if (dropZone && fileInput) {
    dropZone.onclick = function() { fileInput.click(); };

    fileInput.onchange = function() {
      if (this.files[0]) handleFile(this.files[0]);
    };

    // Drag & Drop
    dropZone.addEventListener('dragover', function(e) {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', function(e) {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', function(e) {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('drag-over');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    // Paste
    document.addEventListener('paste', function(e) {
      var items = (e.clipboardData || e.originalEvent.clipboardData).items;
      for (var i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          var file = items[i].getAsFile();
          handleFile(file);
          break;
        }
      }
    });
  }

  // Primitive Management
  function syncPrimButtons() {
    if (!primList) return;
    var hasCircle = !!primList.querySelector('[data-shape="circle"]');
    var hasRect = !!primList.querySelector('[data-shape="rect"]');
    var btnCircle = document.getElementById('btnAddCircle');
    var btnRect = document.getElementById('btnAddRect');
    if (btnCircle) btnCircle.disabled = hasCircle;
    if (btnRect) btnRect.disabled = hasRect;
  }

  function addPrim(shape, w, h, color) {
    if (!primList) return;
    shape = shape || 'circle'; w = w || 1; h = h || 1; color = color || '#ffcc00';
    var existing = primList.querySelector('[data-shape="' + shape + '"]');
    if (existing) {
      existing.querySelector('[data-f=w]').value = w;
      existing.querySelector('[data-f=h]').value = h;
      existing.querySelector('[data-f=color]').value = color;
      return;
    }
    var d = document.createElement('div');
    d.className = 'prim-card';
    d.setAttribute('data-shape', shape);
    var label = shape === 'circle' ? '圆形' : '矩形';
    d.innerHTML = '<span class="prim-tag">' + label + '</span>'
      + '<input type="number" data-f="w" value="' + w + '" min="1" max="10" title="宽">'
      + '<span class="prim-x">×</span>'
      + '<input type="number" data-f="h" value="' + h + '" min="1" max="10" title="高">'
      + '<input type="color" data-f="color" value="' + color + '">'
      + '<button type="button" class="btn-del">✕</button>';
    d.querySelector('.btn-del').onclick = function() { d.remove(); syncPrimButtons(); };
    primList.appendChild(d);
    syncPrimButtons();
  }

  // Bind Buttons
  var btnAddCircle = document.getElementById('btnAddCircle');
  var btnAddRect = document.getElementById('btnAddRect');
  var presetCoin = document.getElementById('presetCoin');
  var presetRect = document.getElementById('presetRect');

  if (btnAddCircle) btnAddCircle.onclick = function() { addPrim('circle', 1, 1, '#f59e0b'); };
  if (btnAddRect) btnAddRect.onclick = function() { addPrim('rect', 1, 10, '#38bdf8'); };
  if (presetCoin) presetCoin.onclick = function() { addPrim('circle', 1, 1, '#f59e0b'); };
  if (presetRect) presetRect.onclick = function() { addPrim('rect', 1, 10, '#38bdf8'); };

  // Init Defaults
  if (primList) {
    addPrim('circle', 1, 1, '#f59e0b');
    addPrim('rect', 1, 10, '#38bdf8');
  }

  // Sliders
  ['primSize', 'precision', 'spacing'].forEach(function(id) {
    var el = document.getElementById(id), tag = document.getElementById(id + 'Val');
    if (el && tag) el.oninput = function() { tag.textContent = el.value; };
  });

  // Origin Type Toggle
  var originType = document.getElementById('originType');
  var customOrigin = document.getElementById('customOrigin');
  if (originType && customOrigin) {
    originType.onchange = function() {
        customOrigin.hidden = (this.value !== 'custom');
    };
  }

  // Submit
  if (mainForm) {
    mainForm.onsubmit = function() {
      var arr = [];
      primList.querySelectorAll('.prim-card').forEach(function(c) {
        arr.push({
          shape: c.getAttribute('data-shape'),
          w: parseInt(c.querySelector('[data-f=w]').value) || 1,
          h: parseInt(c.querySelector('[data-f=h]').value) || 1,
          color: c.querySelector('[data-f=color]').value
        });
      });
      if (primJson) primJson.value = JSON.stringify(arr);
    };
  }

})();
