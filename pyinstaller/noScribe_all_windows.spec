# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

if "__file__" in globals():
    BASE = Path(__file__).resolve().parent.parent
else:
    BASE = Path.cwd().resolve()


def common_datas():
    datas = []
    for rel in ["noScribeEdit", "trans", "models"]:
        if (BASE / rel).exists():
            datas.append((str(BASE / rel), rel))
    for rel in ["prompt.yml", "graphic_sw.png", "noScribeLogo.ico"]:
        if (BASE / rel).exists():
            datas.append((str(BASE / rel), "."))
    for pattern in ["README*", "LICENSE*"]:
        for path in BASE.glob(pattern):
            datas.append((str(path), "."))
    datas += collect_data_files("customtkinter")
    datas += copy_metadata("AdvancedHTMLParser")
    if sys.platform.startswith(("win", "linux")):
        datas += collect_data_files("faster_whisper")
        datas += copy_metadata("faster_whisper")
    return datas


def platform_binaries():
    binaries = []
    ffmpeg = BASE / "ffmpeg.exe"
    if ffmpeg.exists():
        binaries.append((str(ffmpeg), "ffmpeg.exe"))
    return binaries


block_cipher = None

a = Analysis(
    [str(BASE / "pyinstaller" / "entry" / "gui_entry.py"),
     str(BASE / "pyinstaller" / "entry" / "cli_entry.py")],
    pathex=[str(BASE)],
    binaries=platform_binaries(),
    datas=common_datas(),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

gui = EXE(
    pyz,
    a.scripts[0],
    [],
    exclude_binaries=True,
    name="noScribe",
    console=False,
    icon=str(BASE / "noScribeLogo.ico"),
)

cli = EXE(
    pyz,
    a.scripts[1],
    [],
    exclude_binaries=True,
    name="noScribe-cli",
    console=True,
    icon=str(BASE / "noScribeLogo.ico"),
)

coll_gui = COLLECT(
    gui,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="noScribe",
)

coll_cli = COLLECT(
    cli,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="noScribe-cli",
)
