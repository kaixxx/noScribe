# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

datas = [('../img/noScribeLogo.png', 'img/'), ('../img/graphic_sw.png', 'img/'), ('../LICENSE.txt', '.'), ('../prompt.yml', '.'), ('../prompt_nd.yml', '.'), ('../pyannote', 'pyannote/'), ('../README.md', '.'), ('../trans', 'trans/')]
binaries = []
hiddenimports = ['PIL._tkinter_finder']
datas += collect_data_files('faster_whisper')
datas += collect_data_files('lightning_fabric')
tmp_ret = collect_all('pyannote')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('speechbrain')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add model files to `datas` if available. This way, it is possible to test the
# pyinstaller in the CI without downloading the big models all the time.
if (Path("./models") / "precise" / "model.bin").exists():
    datas.append(('../models/precise', 'models/precise/'))
    datas.append(('../models/fast', 'models/fast/'))
else:
    # Add a models dir with an empty file to the bundle to avoid errors in
    # noScribe. Adding empty dirs doesn't work with pyinstaller.
    (Path("./models") / "empty").touch()
    datas.append(('../models/empty', 'models/empty'))


a = Analysis(
    ['../noScribe/__main__.py'],
    pathex=['..'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='noScribe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['../img/noScribeLogo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)
