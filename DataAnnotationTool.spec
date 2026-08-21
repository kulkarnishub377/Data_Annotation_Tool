# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None
BASE_DIR = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(BASE_DIR, 'index.html'), '.'),
    (os.path.join(BASE_DIR, 'static'), 'static'),
    (os.path.join(BASE_DIR, 'scripts'), 'scripts'),
    (os.path.join(BASE_DIR, 'VERSION'), '.'),
]

hidden_imports = [
    'webview',
    'clr',
    'clr_loader',
    'sqlite3',
    'jinja2',
    'werkzeug',
    'flask',
    'PIL',
    'PIL.Image',
    'cv2',
    'numpy',
    'dotenv',
    'configparser',
    'ultralytics',
    'torch',
]

a = Analysis(
    ['app.py'],
    pathex=[BASE_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DataAnnotationStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(BASE_DIR, 'static', 'assets', 'app_icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DataAnnotationStudio',
)
