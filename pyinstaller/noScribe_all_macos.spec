# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

# Resolve repository root regardless of current working directory
BASE = Path.cwd().resolve()
while not (BASE / ".git").exists() and BASE != BASE.parent:
    BASE = BASE.parent


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
    return datas


def platform_binaries():
    binaries = []
    ffmpeg = BASE / "ffmpeg" / "ffmpeg"
    if ffmpeg.exists():
        binaries.append((str(ffmpeg), "ffmpeg"))
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
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

gui = EXE(
    pyz,
    [a.scripts[0]],
    [],
    exclude_binaries=True,
    name="noScribe",
    console=False,
    icon=str(BASE / "noScribeLogo.ico"),
)

cli = EXE(
    pyz,
    [a.scripts[1]],
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

app = BUNDLE(coll_gui, name="noScribe.app", icon=str(BASE / "noScribeLogo.ico"))

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
