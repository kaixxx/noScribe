from PyInstaller.utils.hooks import collect_all
import site

# noScribe

noScribe_a = Analysis(
    ['noScribe.py'],
    pathex=[],
    binaries=[],
    datas=[('trans', 'trans/'), ('noScribeLogo.png', '.'), ('graphic_sw.png', '.'), ('ffmpeg', '.'), ('models/faster-whisper-small', 'models/faster-whisper-small/'), ('models/faster-whisper-large-v2', 'models/faster-whisper-large-v2/'), ('prompt.yml', '.'), ('LICENSE.txt', '.'), ('README.md', '.')],
    hiddenimports=['PIL', 'PIL._imagingtk', 'PIL._tkinter_finder'],
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
    icon=['noScribeLogo.ico'],
)

# diarize

site_packages = site.getsitepackages()
diarize_datas = [('models/pyannote_config.yaml', 'models/.'), (f'{site_packages[0]}/lightning', 'lightning/'), (f'{site_packages[0]}/lightning_fabric', 'lightning_fabric'), (f'{site_packages[0]}/torchaudio', 'torchaudio'), (f'{site_packages[0]}/pyannote', 'pyannote/'), (f'{site_packages[0]}/pytorch_metric_learning', 'pytorch_metric_learning/'), (f'{site_packages[0]}/sklearn', 'sklearn/'), ('models/pytorch_model.bin', 'models/.'), (f'{site_packages[0]}/asteroid_filterbanks', 'asteroid_filterbanks/'), (f'{site_packages[0]}/pytorch_lightning', 'pytorch_lightning/'), ('models/torch', 'models/torch/')]
diarize_binaries = []
diarize_hiddenimports = []
diarize_tmp_ret = collect_all('speechbrain')
diarize_datas += diarize_tmp_ret[0]; diarize_binaries += diarize_tmp_ret[1]; diarize_hiddenimports += diarize_tmp_ret[2]

diarize_a = Analysis(
    ['diarize.py'],
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

# final

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
    icon='noScribeLogo.ico',
    bundle_identifier='org.noScribe.noScribe',
)
