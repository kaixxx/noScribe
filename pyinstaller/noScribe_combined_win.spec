# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# noScribe:

noScribe_datas = [('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/faster-whisper-large-v2', './models/faster-whisper-large-v2/'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/faster-whisper-small', './models/faster-whisper-small/'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribeEdit', 'noScribeEdit/'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/trans', 'trans/'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/graphic_sw.png', '.'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/LICENSE.txt', '.'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribeLogo.ico', '.'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/prompt.yml', '.'), 
('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/README.md', '.')]
noScribe_datas += collect_data_files('customtkinter')
noScribe_datas += copy_metadata('AdvancedHTMLParser')
noScribe_datas += collect_data_files('faster_whisper')

noScribe_a = Analysis(
    ['C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribe.py'],
    pathex=[],
    binaries=[('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/ffmpeg.exe', '.')],
    datas=noScribe_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

noScribe_pyz = PYZ(noScribe_a.pure, noScribe_a.zipped_data, cipher=block_cipher)

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
    icon=['C:\\Users\\kai\\Documents\\Programmierung\\2023_WhisperTranscribe\\noScribe\\noScribeLogo.ico'],
)

# diarize:

diarize_datas = [('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/pyannote_config.yaml', 'models/'), 
    ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/pytorch_model.bin', 'models/'), 
    ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/torch', 'models/torch/')]
diarize_binaries = []
diarize_hiddenimports = []
diarize_datas += collect_data_files('lightning')
diarize_datas += collect_data_files('gradio')
diarize_datas += collect_data_files('lightning_fabric')
diarize_datas += collect_data_files('librosa')
diarize_datas += collect_data_files('pyannote')
diarize_datas += copy_metadata('filelock')
diarize_datas += copy_metadata('tqdm')
diarize_datas += copy_metadata('regex')
diarize_datas += copy_metadata('requests')
diarize_datas += copy_metadata('packaging')
diarize_datas += copy_metadata('numpy')
diarize_datas += copy_metadata('tokenizers')
diarize_datas += copy_metadata('pyannote.audio')
diarize_datas += copy_metadata('pyannote.core')
diarize_datas += copy_metadata('pyannote.database')
diarize_datas += copy_metadata('pyannote.metrics')
diarize_datas += copy_metadata('pyannote.pipeline')
diarize_binaries += collect_dynamic_libs('pyannote')
diarize_hiddenimports += collect_submodules('pyannote')
tmp_ret = collect_all('speechbrain')
diarize_datas += tmp_ret[0]; diarize_binaries += tmp_ret[1]; diarize_hiddenimports += tmp_ret[2]

diarize_a = Analysis(
    ['C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/diarize.py'],
    pathex=[],
    binaries=diarize_binaries,
    datas=diarize_datas,
    hiddenimports=diarize_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

diarize_pyz = PYZ(diarize_a.pure, diarize_a.zipped_data, cipher=block_cipher)

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

# bring it all together:

coll = COLLECT(
    noScribe_exe,
    noScribe_a.binaries,
    noScribe_a.zipfiles,
    noScribe_a.datas,
    diarize_exe,
    diarize_a.binaries,
    diarize_a.zipfiles,
    diarize_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)
