# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec — 打包 Shaper 为 Windows 单文件应用 (One-File)
用法: pyinstaller shaper.spec
输出: dist/Shaper.exe
"""

import sys
import os

block_cipher = None
BASE = os.path.abspath('..')

a = Analysis(
    ['app_desktop.py'],
    pathex=[BASE],
    binaries=[],
    datas=[
        # Web 前端资源
        (os.path.join(BASE, 'web'), 'web'),
        # Python 核心模块
        (os.path.join(BASE, 'shaper_core.py'), '.'),
        (os.path.join(BASE, 'final_shaper.py'), '.'),
        (os.path.join(BASE, 'server.py'), '.'),
    ],
    hiddenimports=[
        'flask',
        'flask.json',
        'jinja2',
        'jinja2.ext',
        'markupsafe',
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.debug',
        'cv2',
        'numpy',
        'scipy',
        'scipy.optimize',
        'scipy.spatial',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
        'webview',
        'shaper_core',
        'final_shaper',
        'server',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter', 'unittest', 'test',
        'pip', 'ensurepip',
        'matplotlib', 'PIL',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Shaper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
