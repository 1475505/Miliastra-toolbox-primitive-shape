# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path


block_cipher = None

spec_dir = Path(SPECPATH).resolve()
project_dir = spec_dir.parent
win_dir = spec_dir
gia_dir = project_dir / "gia"
web_dir = project_dir / "web"

a = Analysis(
    [str(win_dir / "app_desktop.py")],
    pathex=[str(project_dir), str(gia_dir)],
    binaries=[],
    datas=[
        (str(web_dir), "web"),
        (str(gia_dir / "image_template.gia"), "gia"),
    ],
    hiddenimports=[
        "server",
        "shaper_core",
        "fill_shaper",
        "final_shaper",
        "json_to_gia",
        "scipy.special.cython_special",
        "shapely",
        "shapely.geometry",
        "shapely.algorithms",
        "shapely.coords",
        "cv2",
        "numpy",
        "flask",
        "webview",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "cefpython3",
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
    name="Shaper",
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
