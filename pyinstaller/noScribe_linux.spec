# -*- mode: python ; coding: utf-8 -*-

############################################
# Run from /noScribe/pyinstaller subdir!
############################################

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

block_cipher = None

noScribe_datas = [
    ('../models/precise/', './models/precise/'),
    ('../models/fast/', './models/fast/'),
    # ('../noScribeEdit/', './noScribeEdit/'),
    ('../trans/', './trans/'),
    ('../graphic_sw.png', '.'),
    ('../LICENSE.txt', '.'),
    ('../noScribeLogo.png', '.'),
    ('../prompt.yml', '.'),
    ('../prompt_nd.yml', '.'),
    ('../README.md', '.'),
]
noScribe_datas += collect_data_files('customtkinter')
noScribe_datas += copy_metadata('AdvancedHTMLParser')
noScribe_datas += collect_data_files('faster_whisper')

# pyannote integration
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

noScribe_binaries = [('../ffmpeg-linux-x86_64', '.')]
noScribe_binaries += collect_dynamic_libs('pyannote')

noScribe_hiddenimports = ['PIL._tkinter_finder']
noScribe_hiddenimports += ['PIL.ImageTk']
noScribe_hiddenimports += ['PIL._imagingtk']
noScribe_hiddenimports += collect_submodules('pyannote')
noScribe_hiddenimports += collect_submodules('scipy')
# noScribe_hiddenimports += ['scipy._lib.array_api_compat.numpy.fft']

tmp_ret = collect_all('speechbrain')
noScribe_datas += tmp_ret[0]
noScribe_binaries += tmp_ret[1]
noScribe_hiddenimports += tmp_ret[2]

noScribe_a = Analysis(
    ['../noScribe.py'],
    pathex=[],
    binaries=noScribe_binaries,
    datas=noScribe_datas,
    hiddenimports=noScribe_hiddenimports,
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
    icon=['../noScribeLogo.ico'],
)

coll = COLLECT(
    noScribe_exe,
    noScribe_a.binaries,
    noScribe_a.zipfiles,
    noScribe_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='noScribe',
)

app = BUNDLE(
    coll,
    name='noScribe.app',
    icon='../noScribeLogo.ico',
    bundle_identifier=None,
    info_plist={"CFBundleShortVersionString":"0.7"},
)

