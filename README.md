# Vocal Remover

Local, GPU-accelerated vocal / instrumental separation. Drop a song → get a clean
`vocals.flac` + `instrumental.flac` back. Everything runs on your machine —
nothing is uploaded.

- **Engine:** Mel-Band RoFormer (Kim FT) via [audio-separator]
- **Output:** lossless FLAC
- **UI:** browser or a native desktop app (Tauri)

**Platforms:** NVIDIA GPU (CUDA) is the fast path. The engine auto-detects the
device, so it also runs on macOS (Apple Silicon MPS / CPU) and CPU-only machines
— just slower. Only the optional CUDA torch wheel in `requirements.txt` is
NVIDIA-specific; on macOS install plain torch.

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

## The backend: local Python vs. bundled sidecar

`src-tauri/src/lib.rs` spawns the Flask backend one of two ways, auto-detected at
launch:

1. **Sidecar** — if a frozen backend sits next to the app (`backend/vr-backend[.exe]`),
   run it. This is a [PyInstaller] onedir build (`vr-backend.spec`) carrying its own
   Python + PyTorch (+ CUDA) — so the target machine needs **no Python/dev setup**.
2. **Local Python** — otherwise, run `app.py` with the project's `.venv` (dev) or
   the system `python`. This is what the CI installers use — they ship the app
   shell only, so the machine must have Python + `requirements.txt` installed.

### Does it download the model at launch?

The backend calls `separator.load_model()` at startup. Where the model comes from:

- **Bundled** (full self-contained build): the model ships as a Tauri resource,
  gets mirrored into a writable cache on first launch → **no download**.
- **Not bundled** (dev, or the CI shell installers): `app.py` finds no local model
  and [audio-separator] **downloads it on first run** (~200 MB), cached in `./models`
  (or `MODEL_DIR`). One-time; later launches are instant.

The model is gitignored, so the CI builds deliberately don't bundle it (shell only)
— on those, the first launch of the local Python backend triggers that download.

**Progress UI:** the model loads in a background thread, so Flask serves right away.
Until it's ready, `GET /` returns a progress page that polls `GET /status`
(`{ready, phase, downloaded_mb, error}`) and reloads into the app once loaded — so
the first-run download shows live MB instead of a frozen window. `/separate` returns
`503` while loading.

## Distribute (standalone — no Python/dev setup on the target)

For the fully self-contained build (bundled Python + PyTorch + CUDA + model),
freeze the sidecar and bundle it. Best on machines with an **NVIDIA GPU + driver**:

```sh
pyinstaller vr-backend.spec --distpath sidecar          # freeze the backend (~4.8 GB)
cargo build --release --manifest-path src-tauri/Cargo.toml
# then place next to the shell:  app.exe + backend/ (the sidecar) + models/
```

A ready-to-run portable build lives in **`release-portable/vocal-remover/`** —
copy that folder to any NVIDIA machine and run `app.exe`. No install, no Python.
First launch copies the model into `%LOCALAPPDATA%\VocalRemover` once, then it's
instant.

> The signed `release.yml` installers are **shells** (no sidecar/model) — the
> self-contained sidecar bundle is not wired into CI yet (it's multi-GB and
> per-OS). Build it locally as above, or extend `release.yml` to run PyInstaller.

[PyInstaller]: https://pyinstaller.org

## Releases & auto-update (CI)

Two GitHub Actions workflows:

| Workflow | Trigger | Does |
|----------|---------|------|
| `.github/workflows/build.yml` | PRs, non-`main` pushes | Compile-check on macOS + Windows, uploads unsigned installers as artifacts |
| `.github/workflows/release.yml` | push to `main` | [semantic-release] cuts a version from the commits, then builds **signed** installers + updater artifacts on both OSes and attaches them (incl. `latest.json`) to the GitHub Release |

**Versioning is commit-driven** ([Conventional Commits]): `fix:` → patch,
`feat:` → minor, `feat!:`/`BREAKING CHANGE:` → major. Commits that aren't
`fix`/`feat` (e.g. `chore:`, `docs:`) don't cut a release. `scripts/bump-version.mjs`
syncs the new version into `package.json`, `src-tauri/tauri.conf.json`, and
`Cargo.toml`.

**Auto-update:** on launch the desktop app checks the release feed and, if a newer
signed build exists, downloads + installs it and relaunches (`src-tauri/src/lib.rs`).
Signing uses a minisign keypair — the public key is in `tauri.conf.json`; the
private key + password live in the `TAURI_SIGNING_PRIVATE_KEY` /
`TAURI_SIGNING_PRIVATE_KEY_PASSWORD` repo secrets. **Back up the private key** —
if it's lost, existing installs can no longer verify updates.

[semantic-release]: https://semantic-release.gitbook.io
[Conventional Commits]: https://www.conventionalcommits.org
[audio-separator]: https://github.com/nomadkaraoke/python-audio-separator
