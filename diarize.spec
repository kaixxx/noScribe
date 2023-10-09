# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/pyannote_config.yaml', 'models/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/pytorch_model.bin', 'models/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/huggingface', 'models/huggingface/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/torch', 'models/torch/')]
binaries = []
hiddenimports = []
datas += collect_data_files('lightning')
datas += collect_data_files('gradio')
datas += collect_data_files('lightning_fabric')
datas += collect_data_files('librosa')
datas += collect_data_files('pyannote')
datas += copy_metadata('filelock')
datas += copy_metadata('tqdm')
datas += copy_metadata('regex')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('numpy')
datas += copy_metadata('tokenizers')
datas += copy_metadata('pyannote.audio')
datas += copy_metadata('pyannote.core')
datas += copy_metadata('pyannote.database')
datas += copy_metadata('pyannote.metrics')
datas += copy_metadata('pyannote.pipeline')
binaries += collect_dynamic_libs('pyannote')
hiddenimports += collect_submodules('pyannote')
tmp_ret = collect_all('speechbrain')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


block_cipher = None


a = Analysis(
    ['C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/diarize.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='diarize',
)
