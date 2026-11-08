# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

# The repository, including the shared collector, is mounted at /noScribe.
sys.path.insert(0, '/noScribe/pyinstaller')
from collect_licenses import collect_license_files

datas = [('/noScribe/img/noScribeLogo.png', 'img/'), ('/noScribe/img/graphic_sw.png', 'img/'), ('/noScribe/LICENSE.txt', '.'), ('/noScribe/prompts/prompt.yml', 'prompts/'), ('/noScribe/prompts/prompt_nd.yml', 'prompts/'), ('/noScribe/pyannote', 'pyannote/'), ('/noScribe/README.md', '.'), ('/noScribe/trans', 'trans/')]
binaries = []
# noScribe.main is reached through the package's lazy __getattr__, which
# PyInstaller's static analysis cannot follow -- declare it explicitly.
hiddenimports = ['PIL._tkinter_finder', 'noScribe.main']
datas += collect_data_files('faster_whisper')
datas += collect_data_files('noScribe.plugins', includes=['**/backend.json'])
datas += collect_data_files('lightning_fabric')
tmp_ret = collect_all('pyannote')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

datas.append(('/noScribe/models/precise', 'models/precise/'))
datas.append(('/noScribe/models/fast', 'models/fast/'))


a = Analysis(
    ['/noScribe/noScribe/__main__.py'],
    pathex=['..'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The inference server is deployed from source and is not part of the
    # desktop application bundle.
    excludes=['speechbrain', 'noScribe.server', 'uvicorn'],
    noarchive=False,
    optimize=0,
)
license_datas = collect_license_files(
    a.pure + a.binaries + a.datas + a.scripts,
    workpath,
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
    icon=['/noScribe/img/noScribeLogo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    license_datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)
