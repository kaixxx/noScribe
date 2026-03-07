# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

# noScribe:

noScribe_datas = [] 
noScribe_binaries = []
noScribe_hiddenimports = []

noScribe_datas += [
('../models/precise/', './models/precise/'), 
('../models/fast/', './models/fast/'), 
('../noScribeEdit/', './noScribeEdit/'), 
('../trans/', './trans/'), 
('../graphic_sw.png', '.'), 
('../LICENSE.txt', '.'), 
('../noScribeLogo.ico', '.'), 
('../prompt.yml', '.'),
('../prompt_nd.yml', '.'), 
('../README.md', '.')]
noScribe_datas += collect_data_files('customtkinter')
noScribe_datas += copy_metadata('AdvancedHTMLParser')
noScribe_datas += collect_data_files('faster_whisper')

# for pyannote:
noScribe_datas += [('../pyannote/', './pyannote/')]
noScribe_datas += collect_data_files('lightning')
noScribe_datas += collect_data_files('gradio')
noScribe_datas += collect_data_files('lightning_fabric')
noScribe_datas += collect_data_files('librosa')
noScribe_datas += collect_data_files('pyannote')
noScribe_datas += copy_metadata('filelock')
noScribe_datas += copy_metadata('tqdm')
# noScribe_datas += copy_metadata('regex')
noScribe_datas += copy_metadata('requests')
noScribe_datas += copy_metadata('packaging')
noScribe_datas += copy_metadata('numpy')
noScribe_datas += copy_metadata('scipy')
noScribe_datas += copy_metadata('tokenizers')
noScribe_datas += copy_metadata('pyannote.audio')
noScribe_datas += copy_metadata('pyannote.core')
noScribe_datas += copy_metadata('pyannote.database')
noScribe_datas += copy_metadata('pyannote.metrics')
noScribe_datas += copy_metadata('pyannote.pipeline')
noScribe_binaries += collect_dynamic_libs('pyannote')
noScribe_hiddenimports += collect_submodules('pyannote')
noScribe_hiddenimports += collect_submodules('scipy')
# noScribe_hiddenimports += ['scipy._lib.array_api_compat.numpy.fft']
tmp_ret = collect_all('speechbrain')
noScribe_datas += tmp_ret[0]; noScribe_binaries += tmp_ret[1]; noScribe_hiddenimports += tmp_ret[2]

noScribe_a = Analysis(
    ['../noScribe.py'],
    pathex=[],
    binaries=noScribe_binaries,
    datas=noScribe_datas,
    hiddenimports=noScribe_hiddenimports,   # <-- use them
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
    upx=False,
    # console=False,
    hide_console="hide-late",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['../noScribeLogo.ico'],
)

# assemble the dist folder with all needed DLLs, datas, etc.
noScribe_coll = COLLECT(
    noScribe_exe,
    noScribe_a.binaries,
    noScribe_a.zipfiles,
    noScribe_a.datas,
    strip=False,
    upx=False,
    name='noScribe'
)
