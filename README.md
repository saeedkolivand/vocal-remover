# Vocal Remover

Local, GPU-accelerated vocal / instrumental separation. Drop a song → get a clean
`vocals.flac` + `instrumental.flac` back. Everything runs on your machine —
nothing is uploaded.

- **Engine:** Mel-Band RoFormer (Kim FT) via [audio-separator], on PyTorch CUDA
- **Output:** lossless FLAC
- **UI:** browser or a native desktop app (Tauri)

## Run it (dev)

Needs the Python env (`.venv` with PyTorch CUDA) and Node + pnpm.

```sh
pnpm dev          # browser — Flask UI on http://127.0.0.1:8000
pnpm dev:desktop  # native desktop window (Tauri)
pnpm test         # end-to-end self-check
```

Try a different model with `MODEL=<checkpoint>.ckpt pnpm dev` (audio-separator
downloads it on first use; it caches in `./models`).

## Layout

| Path | What |
|------|------|
| `app.py` | Flask backend — `/separate`, `/history`, `/download`, serves the UI |
| `index.html` | the web UI (drag-drop, players, history) |
| `apps/` | pnpm + turbo workspaces (`backend`, `desktop` wrappers) |
| `src-tauri/` | Tauri desktop shell (Rust) — spawns the backend, shows the UI |
| `models/` | the model checkpoint (auto-downloaded, or bundled for distribution) |
| `vr-backend.spec` | PyInstaller recipe for the standalone backend |
| `selfcheck.py` | end-to-end smoke test |

## Distribute (standalone — no Python/dev setup on the target)

The app can ship self-contained (bundled Python + PyTorch + CUDA + model) for
machines with an **NVIDIA GPU + driver**:

```sh
pyinstaller vr-backend.spec --distpath sidecar          # freeze the backend (~4.8 GB)
cargo build --release --manifest-path src-tauri/Cargo.toml
# then place next to the shell:  app.exe + backend/ (the sidecar) + models/
```

A ready-to-run portable build lives in **`release-portable/vocal-remover/`** —
copy that folder to any NVIDIA machine and run `app.exe`. No install, no Python.
First launch copies the model into `%LOCALAPPDATA%\VocalRemover` once, then it's
instant.

[audio-separator]: https://github.com/nomadkaraoke/python-audio-separator
