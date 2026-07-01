# -*- mode: python ; coding: utf-8 -*-
# PyInstaller onedir build of the Flask backend, to ship as a Tauri sidecar so the
# desktop app runs on machines without the Python/PyTorch dev setup.
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

# Packages that lack good auto-detection (dynamic imports, data files, native bins).
# torch / onnxruntime / numpy / scipy are handled by PyInstaller's built-in hooks.
for pkg in (
    "audio_separator", "static_ffmpeg",
    "rotary_embedding_torch", "einops", "ml_collections", "beartype",
    "librosa", "soundfile", "soxr", "samplerate", "resampy",
    "demucs", "julius", "lameenc", "diffq", "omegaconf", "dora_search",
    "pydub", "hyperpyyaml", "ruamel.yaml",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print(f"[spec] skip {pkg}: {e}")

datas += [("index.html", ".")]  # served at "/"

# audio-separator loads these by name at runtime
hiddenimports += [
    "audio_separator.separator.architectures.mdxc_separator",
    "audio_separator.separator.architectures.mdx_separator",
    "audio_separator.separator.architectures.vr_separator",
    "audio_separator.separator.architectures.demucs_separator",
]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "pandas", "IPython", "jupyter", "notebook",
              "PyQt5", "PyQt6", "PySide6", "pytest", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)
# console=False: no console window pops up behind the desktop app. Startup/model
# errors still surface via the backend's /status endpoint (app.py). Flip to True
# to debug a frozen build that never reaches Flask (e.g. an import error).
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="vr-backend", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="vr-backend")
