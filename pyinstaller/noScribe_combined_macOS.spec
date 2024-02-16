# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_all

# noScribe

noScribe_a = Analysis(
    ['/path/to/noScribe/noScribe.py'],
    pathex=[],
    binaries=[('/path/to/noScribe/ffmpeg', '.')],
    datas=[('/path/to/noScribe/trans', 'trans/'), ('/path/to/noScribe/graphic_sw.png', '.'), ('/path/to/noScribe/models/faster-whisper-small', 'models/faster-whisper-small/'), ('/path/to/noScribe/models/faster-whisper-large-v2', 'models/faster-whisper-large-v2/'), ('/path/to/noScribe/prompt.yml', '.'), ('/path/to/noScribe/LICENSE.txt', '.'), ('/path/to/noScribe/README.md', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
noScribe_pyz = PYZ(noScribe_a.pure)

noScribe_exe = EXE(
    noScribe_pyz,
    noScribe_a.scripts,
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
    icon=['/path/to/noScribe/noScribeLogo.ico'],
)

# diarize

diarize_datas = [('/path/to/noScribe/models/pyannote_config.yaml', 'models/.'), ('/path/to/noScribe/models/pytorch_model.bin', 'models/.'), ('/path/to/noScribe/models/torch', 'models/torch/')]
diarize_binaries = []
diarize_hiddenimports = []
diarize_datas += collect_data_files('lightning_fabric')
diarize_tmp_ret = collect_all('pyannote')
diarize_datas += diarize_tmp_ret[0]; diarize_binaries += diarize_tmp_ret[1]; diarize_hiddenimports += diarize_tmp_ret[2]
diarize_tmp_ret = collect_all('speechbrain')
diarize_datas += diarize_tmp_ret[0]; diarize_binaries += diarize_tmp_ret[1]; diarize_hiddenimports += diarize_tmp_ret[2]


diarize_a = Analysis(
    ['/path/to/noScribe/diarize.py'],
    pathex=[],
    binaries=diarize_binaries,
    datas=diarize_datas,
    hiddenimports=diarize_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
diarize_pyz = PYZ(diarize_a.pure)

diarize_exe = EXE(
    diarize_pyz,
    diarize_a.scripts,
    [],
    exclude_binaries=True,
    name='diarize',
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
)

# combine

coll = COLLECT(
    noScribe_exe,
    noScribe_a.binaries,
    noScribe_a.datas,
    diarize_exe,
    diarize_a.binaries,
    diarize_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)

app = BUNDLE(
    coll,
    name='noScribe.app',
    icon='/path/to/noScribe/noScribeLogo.ico',
    bundle_identifier=None,
    info_plist={"CFBundleShortVersionString":"0.4.4"},
)
