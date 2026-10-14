# -*- mode: python ; coding: utf-8 -*-
"""
Сборка:  pyinstaller desktop.spec
Режим:   --onedir (папка с exe). --onefile с torch нецелесообразен (2+ ГБ).
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    copy_metadata,
)

# ── данные библиотек ──
datas = []
datas += collect_data_files("streamlit")
datas += collect_data_files("transformers")
datas += collect_data_files("pywebview")

# ── метаданные (.dist-info) - без них падает importlib.metadata ──
for pkg in (
    "streamlit", "transformers", "torch", "plotly", "pandas",
    "numpy", "scikit-learn", "pywebview", "altair", "protobuf",
    "click", "requests", "rich", "tenacity", "tornado", "watchdog",
    "cachetools", "blinker", "gitpython", "pyarrow", "pydeck",
    "toml", "typing-extensions", "pillow",
):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass  # пакет не установлен - пропускаем

# ── скрытые импорты ──
hiddenimports = (
    collect_submodules("streamlit")
    + collect_submodules("transformers")
    + collect_submodules("scipy")
    + [
        # sklearn иногда прячет C-расширения
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._partition_nodes",
        "sklearn.tree._utils",
        # pywebview на Windows
        "webview.platforms.edgechromium",
        "webview.platforms.winforms",
    ]
)

# ── динамические библиотеки torch (.dll / .so) ──
binaries = collect_dynamic_libs("torch")

# ── анализ ──
a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # выкидываем то, что не нужно, чтобы уменьшить размер
        "tkinter", "matplotlib",
        "IPython", "jupyter", "notebook",
        "pytest", "sphinx",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,       # ← True = режим --onedir
    name="MoodApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                   # UPX с torch часто ломает .dll
    console=True,               # False = без чёрного окна консоли
    icon=None,                   # сюда можно вписать "icon.ico"
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="MoodApp",
)