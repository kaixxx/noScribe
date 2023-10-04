# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/faster-whisper-large-v2', './models/faster-whisper-large-v2/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/models/faster-whisper-small', './models/faster-whisper-small/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribeEdit', 'noScribeEdit/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/trans', 'trans/'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/graphic_sw.png', '.'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/LICENSE.txt', '.'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribeLogo.ico', '.'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/prompt.yml', '.'), ('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/README.md', '.')]
datas += collect_data_files('customtkinter')
datas += copy_metadata('AdvancedHTMLParser')


block_cipher = None


a = Analysis(
    ['C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/noScribe.py'],
    pathex=[],
    binaries=[('C:/Users/kai/Documents/Programmierung/2023_WhisperTranscribe/noScribe/ffmpeg.exe', '.')],
    datas=datas,
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
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
    icon=['C:\\Users\\kai\\Documents\\Programmierung\\2023_WhisperTranscribe\\noScribe\\noScribeLogo.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)
