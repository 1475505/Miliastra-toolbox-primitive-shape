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
  var primJson = document.getElementById('primJson');
  var modeInput = document.getElementById('modeInput');

  // Mode switch elements
  var fillParams = document.getElementById('fillParams');
  var outlineParams = document.getElementById('outlineParams');
  var modeFill = document.getElementById('modeFill');
  var modeOutline = document.getElementById('modeOutline');

  // 预设配置
  var PRESETS = {
    circle: {
      coin: { w: 1, h: 1, type_id: 10005009, color: '#f59e0b' },
      electro_badge: { w: 0.3, h: 0.3, type_id: 20001281, color: '#c084fc', rot_z: 90, rot_y_add: 90 },
      pyro_badge: { w: 0.3, h: 0.3, type_id: 20001282, color: '#ef4444', rot_z: 90, rot_y_add: 90 },
      dendro_badge: { w: 0.3, h: 0.3, type_id: 20001283, color: '#22c55e', rot_z: 90, rot_y_add: 90 },
      cryo_badge: { w: 0.3, h: 0.3, type_id: 20001284, color: '#38bdf8', rot_z: 90, rot_y_add: 90 },
      geo_badge: { w: 0.3, h: 0.3, type_id: 20001285, color: '#eab308', rot_z: 90, rot_y_add: 90 },
      hydro_badge: { w: 0.3, h: 0.3, type_id: 20001286, color: '#3b82f6', rot_z: 90, rot_y_add: 90 },
      anemo_badge: { w: 0.3, h: 0.3, type_id: 20001287, color: '#10b981', rot_z: 90, rot_y_add: 90 },
      custom: { w: 1, h: 1, type_id: 10005009 }
    },
    rect: {
      wood_box: { w: 1.0, h: 1.0, type_id: 20001224, color: '#fcd34d' },
      wood_pillar: { w: 0.5, h: 5, type_id: 20002129, color: '#38bdf8' },
      custom: { w: 0.5, h: 5, type_id: 20002129 }
    }
  };

  // ── Mode Switch ──
  var modeRadios = document.querySelectorAll('input[name="mode_radio"]');
  modeRadios.forEach(function(radio) {
    radio.addEventListener('change', function() {
      var mode = this.value;
      modeInput.value = mode;
      if (mode === 'fill') {
        fillParams.hidden = false;
        outlineParams.hidden = true;
        modeFill.classList.add('active');
        modeOutline.classList.remove('active');
      } else {
        fillParams.hidden = true;
        outlineParams.hidden = false;
        modeFill.classList.remove('active');
        modeOutline.classList.add('active');
      }
    });
  });

  // ── File Handling ──
  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
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

  if (dropZone && fileInput) {
    dropZone.onclick = function() { fileInput.click(); };

    fileInput.onchange = function() {
      if (this.files[0]) handleFile(this.files[0]);
    };

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

  // ── Preset select + custom fields + color ──
  function setupPresetSelect(selectId, customFieldsId, colorId, presetGroup) {
    var sel = document.getElementById(selectId);
    var custom = document.getElementById(customFieldsId);
    var color = document.getElementById(colorId);
    if (!sel) return;
    sel.addEventListener('change', function() {
      if (custom) custom.hidden = (this.value !== 'custom');
      if (this.value !== 'custom' && color && PRESETS[presetGroup]) {
        var p = PRESETS[presetGroup][this.value];
        if (p && p.color) color.value = p.color;
      }
    });
  }

  // Fill mode presets
  setupPresetSelect('circleTypeSelect', 'circleCustomFields', 'circleColor', 'circle');
  setupPresetSelect('rectTypeSelect', 'rectCustomFields', 'rectColor', 'rect');
  // Outline mode presets
  setupPresetSelect('olCircleTypeSelect', 'olCircleCustomFields', 'olCircleColor', 'circle');
  setupPresetSelect('olRectTypeSelect', 'olRectCustomFields', 'olRectColor', 'rect');

  // ── Sliders ──
  // Fill mode sliders
  ['numPrims', 'primSize'].forEach(function(id) {
    var el = document.getElementById(id), tag = document.getElementById(id + 'Val');
    if (el && tag) el.oninput = function() { tag.textContent = el.value; };
  });
  // Outline mode sliders
  ['olPrimSize', 'olPrecision', 'olSpacing'].forEach(function(id) {
    var el = document.getElementById(id), tag = document.getElementById(id + 'Val');
    if (el && tag) el.oninput = function() { tag.textContent = el.value; };
  });

  // ── Origin Type Toggle ──
  function setupOriginToggle(typeId, customId) {
    var typeEl = document.getElementById(typeId);
    var customEl = document.getElementById(customId);
    if (typeEl && customEl) {
      typeEl.onchange = function() { customEl.hidden = (this.value !== 'custom'); };
    }
  }
  setupOriginToggle('originType', 'customOrigin');
  setupOriginToggle('olOriginType', 'olCustomOrigin');

  // ── Build primitives from selects ──
  function buildPrimitives(circleSelectId, rectSelectId, circleCustomId, rectCustomId, circleColorId, rectColorId) {
    var circleSelect = document.getElementById(circleSelectId);
    var rectSelect = document.getElementById(rectSelectId);
    var circleCustomFields = document.getElementById(circleCustomId);
    var rectCustomFields = document.getElementById(rectCustomId);
    var circleColorInput = document.getElementById(circleColorId);
    var rectColorInput = document.getElementById(rectColorId);

    var circleType = circleSelect ? circleSelect.value : 'geo_badge';
    var circleW = 0.3, circleH = 0.3, circleColorVal = '#eab308';

    if (circleType === 'custom' && circleCustomFields && !circleCustomFields.hidden) {
      var cwInput = document.getElementById('circleW');
      var chInput = document.getElementById('circleH');
      circleW = parseFloat(cwInput && cwInput.value) || 1;
      circleH = parseFloat(chInput && chInput.value) || 1;
    } else {
      var cp = PRESETS.circle[circleType];
      circleW = cp.w;
      circleH = cp.h;
    }
    circleColorVal = circleColorInput ? circleColorInput.value : '#eab308';

    var rectType = rectSelect ? rectSelect.value : 'disabled';
    var rectW = 0.5, rectH = 5, rectColorVal = '#38bdf8';

    if (rectType !== 'disabled') {
      if (rectType === 'custom' && rectCustomFields && !rectCustomFields.hidden) {
        var rwInput = document.getElementById('rectW');
        var rhInput = document.getElementById('rectH');
        rectW = parseFloat(rwInput && rwInput.value) || 0.5;
        rectH = parseFloat(rhInput && rhInput.value) || 5;
      } else {
        var rp = PRESETS.rect[rectType];
        rectW = rp.w;
        rectH = rp.h;
      }
      rectColorVal = rectColorInput ? rectColorInput.value : '#38bdf8';
    }

    var primitives = [];
    var circlePreset = PRESETS.circle[circleType];
    primitives.push({
      shape: 'circle',
      preset_type: circleType,
      w: circleW,
      h: circleH,
      color: circleColorVal,
      type_id: circlePreset.type_id,
      rot_z: circlePreset.rot_z || 0,
      rot_y_add: circlePreset.rot_y_add || 0
    });

    if (rectType !== 'disabled') {
      var rectPreset = PRESETS.rect[rectType];
      primitives.push({
        shape: 'rect',
        preset_type: rectType,
        w: rectW,
        h: rectH,
        color: rectColorVal,
        type_id: rectPreset.type_id
      });
    }

    return primitives;
  }

  // ── Submit ──
  if (mainForm) {
    mainForm.onsubmit = function() {
      var mode = modeInput ? modeInput.value : 'fill';

      var primitives;
      if (mode === 'fill') {
        primitives = buildPrimitives('circleTypeSelect', 'rectTypeSelect', 'circleCustomFields', 'rectCustomFields', 'circleColor', 'rectColor');
      } else {
        primitives = buildPrimitives('olCircleTypeSelect', 'olRectTypeSelect', 'olCircleCustomFields', 'olRectCustomFields', 'olCircleColor', 'olRectColor');
      }

      if (primJson) primJson.value = JSON.stringify(primitives);
    };
  }

})();
