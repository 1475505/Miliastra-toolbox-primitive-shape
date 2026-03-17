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

  // 互斥选择逻辑
  function setupRadioToggle(circleSelectId, rectSelectId) {
    // 圆形选择
    var circleSelect = document.getElementById(circleSelectId);
    var circleCustomFields = document.getElementById('circleCustomFields');
    var circleColor = document.getElementById('circleColor');
    
    if (circleSelect) {
      circleSelect.addEventListener('change', function() {
        if (circleCustomFields) {
          circleCustomFields.hidden = (this.value !== 'custom');
        }
        // 当选择预设时，更新颜色
        if (this.value !== 'custom' && circleColor) {
          var preset = PRESETS.circle[this.value];
          if (preset && preset.color) {
            circleColor.value = preset.color;
          }
        }
      });
    }

    // 矩形选择
    var rectSelect = document.getElementById(rectSelectId);
    var rectCustomFields = document.getElementById('rectCustomFields');
    var rectColor = document.getElementById('rectColor');
    
    if (rectSelect) {
      rectSelect.addEventListener('change', function() {
        if (rectCustomFields) {
          rectCustomFields.hidden = (this.value !== 'custom');
        }
        // 当选择预设时，更新颜色
        if (this.value !== 'custom' && rectColor) {
          var preset = PRESETS.rect[this.value];
          if (preset && preset.color) {
            rectColor.value = preset.color;
          }
        }
      });
    }
  }

  setupRadioToggle('circleTypeSelect', 'rectTypeSelect');

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
      // 获取圆形设置
      var circleType = 'geo_badge'; // default
      var circleSelect = document.getElementById('circleTypeSelect');
      if (circleSelect) {
        circleType = circleSelect.value;
      }
      
      var circleW = 1, circleH = 1, circleColorVal = '#eab308';
      var circleCustomFields = document.getElementById('circleCustomFields');
      var circleWInput = document.getElementById('circleW');
      var circleHInput = document.getElementById('circleH');
      var circleColorInput = document.getElementById('circleColor');
      
      if (circleType === 'custom' && circleCustomFields && !circleCustomFields.hidden) {
        circleW = parseFloat(circleWInput && circleWInput.value) || 1;
        circleH = parseFloat(circleHInput && circleHInput.value) || 1;
      } else if (circleType !== 'disabled') {
        var preset = PRESETS.circle[circleType];
        circleW = preset.w;
        circleH = preset.h;
      }
      circleColorVal = circleColorInput ? circleColorInput.value : '#f59e0b';

      var rectType = 'disabled';
      var rectSelect = document.getElementById('rectTypeSelect');
      if (rectSelect) {
        rectType = rectSelect.value;
      } else {
        var rectRadios = document.getElementsByName('rect_type');
        rectRadios.forEach(function(r) { if (r.checked) rectType = r.value; });
      }
      
      var rectW = 0.5, rectH = 5, rectColorVal = '#38bdf8';
      var rectCustomFields = document.getElementById('rectCustomFields');
      var rectWInput = document.getElementById('rectW');
      var rectHInput = document.getElementById('rectH');
      var rectColorInput = document.getElementById('rectColor');
      
      if (rectType === 'custom' && rectCustomFields && !rectCustomFields.hidden) {
        rectW = parseFloat(rectWInput && rectWInput.value) || 0.5;
        rectH = parseFloat(rectHInput && rectHInput.value) || 5;
      } else if (rectType !== 'disabled') {
        var preset = PRESETS.rect[rectType];
        rectW = preset.w;
        rectH = preset.h;
      }
      rectColorVal = rectColorInput ? rectColorInput.value : '#38bdf8';

      // 构建primitives数据
      var primitives = [];
      
      // 添加圆形
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
      
      // 添加矩形（仅在未禁用时）
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

      if (primJson) primJson.value = JSON.stringify(primitives);
    };
  }

})();
