# Deployment & code signing

How the app ships, and what's left to make "download one file, double-click, done" true
on every platform. The pipeline is described in `README.md`; this file covers the
**installability gaps** and the signing that closes them.

## What ships today

Pushing Conventional Commits to `main` runs `release.yml`: semantic-release cuts a
version, then each OS runner freezes the Flask backend with PyInstaller (**Windows:
DirectML torch** for any-GPU acceleration via `dml_hybrid.py`; **macOS: CPU/MPS torch**)
and `tauri-action` attaches self-contained installers + `latest.json` to the GitHub
Release. The desktop app auto-updates from that feed on launch.

Self-contained means the user needs **no Python, no ffmpeg, no runtimes**:

- Python + torch + onnxruntime + libsndfile are frozen into the sidecar.
- **Windows GPU acceleration** works on any DirectX 12 GPU (NVIDIA/AMD/Intel) via DirectML —
  no CUDA install, no vendor drivers. `dml_hybrid.py` runs the model's complex STFT on CPU and
  the transformer on the GPU (~5× faster than CPU); it auto-falls back to CPU if no GPU.
- **ffmpeg/ffprobe are baked in** — CI runs `static_ffmpeg.add_paths()` before PyInstaller
  so `collect_all` bundles the binaries (they are otherwise downloaded at the user's first
  launch into the install dir, which fails on read-only installs).
- WebView2 uses `embedBootstrapper` (installer carries the bootstrapper; no install-time
  download on the common case).
- Windows ships **NSIS only** (per-user, no UAC). The MSI (per-machine, UAC on install
  *and* on every silent auto-update) was dropped from `bundle.targets`.

**First run still needs internet once:** the ~871 MB model is not bundled (GitHub's 2 GB
asset limit). It downloads to a per-user cache with a determinate progress bar, is reused
across updates, and the app runs fully offline afterward. A truncated/interrupted download
is now deleted and re-fetched instead of bricking the app, and the loading page has a
**Try again** button.

## Gap 1 — macOS Gatekeeper (blocks 100% of macOS users) — NOT YET DONE

The macOS build is unsigned/un-notarized, so a downloaded `.dmg` fails Gatekeeper with
*"'vocal-remover' is damaged and can't be opened."* The only workaround is a Terminal
command, which is not "zero prerequisite". Closing this needs a **paid Apple Developer
account ($99/yr)**. Steps:

1. **Enroll** in the Apple Developer Program; create a **Developer ID Application**
   certificate; export it as a base64 `.p12`.
2. **Add repo secrets:** `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`,
   `APPLE_SIGNING_IDENTITY` (e.g. `Developer ID Application: Your Name (TEAMID)`),
   `APPLE_ID`, `APPLE_PASSWORD` (an app-specific password), `APPLE_TEAM_ID`.
3. **Pass them to `tauri-action`** in `release.yml` (empty on machines without the secrets
   = current unsigned behavior, so this is safe to add now):

   ```yaml
   - uses: tauri-apps/tauri-action@v0
     env:
       GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
       TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
       TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}
       APPLE_CERTIFICATE: ${{ secrets.APPLE_CERTIFICATE }}
       APPLE_CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
       APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
       APPLE_ID: ${{ secrets.APPLE_ID }}
       APPLE_PASSWORD: ${{ secrets.APPLE_PASSWORD }}
       APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
   ```

4. **Reference the entitlements** (`src-tauri/Entitlements.plist` already exists) so the
   hardened runtime doesn't kill the frozen Python. Add to `tauri.conf.json` under
   `bundle.macOS`:

   ```json
   "macOS": {
     "minimumSystemVersion": "12.0",
     "entitlements": "Entitlements.plist"
   }
   ```

5. **Deep-sign the sidecar before `tauri-action`** — the PyInstaller tree is hundreds of
   unsigned Mach-O dylibs; Tauri signs the app wrapper, not resource files, so notarization
   rejects them otherwise. Add a macOS-only CI step after the sidecar build:

   ```yaml
   - name: Deep-sign backend sidecar (macOS)
     if: runner.os == 'macOS' && env.APPLE_SIGNING_IDENTITY != ''
     env:
       APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
     run: |
       find sidecar/vr-backend -type f \( -name "*.so" -o -name "*.dylib" -o -perm +111 \) \
         -exec codesign --force --timestamp --options runtime \
         --entitlements src-tauri/Entitlements.plist -s "$APPLE_SIGNING_IDENTITY" {} \;
   ```

**Intel Macs:** `macos-latest` is arm64, so there is no Intel artifact. Either document
"Apple Silicon only" or add `macos-13` to the `bundle` matrix in `release.yml` (needs its
own per-arch sidecar; a universal Tauri binary alone won't cover the single-arch sidecar).

## Gap 2 — Windows SmartScreen (a speed bump, not a wall) — NOT YET DONE

Unsigned installers trip SmartScreen's "unknown publisher" warning (users click *More
info → Run anyway*). Needs a purchased **OV/EV cert** or **Azure Trusted Signing**. Then:

1. Add the cert (e.g. base64 `.pfx` in `WIN_CERT_BASE64` + `WIN_CERT_PASSWORD` secrets, or
   Azure Trusted Signing credentials).
2. **Sign the sidecar exe** — like macOS, `vr-backend.exe` ships as a *resource*, which
   Tauri's bundler does not sign; an unsigned PyInstaller bootloader is a common AV
   false-positive. Sign `sidecar/vr-backend/vr-backend.exe` with `signtool` before the
   build, guarded by `if: runner.os == 'Windows' && env.WIN_CERT_BASE64 != ''`.
3. **Sign the installer** via Tauri's `bundle.windows.signCommand` (inject it into
   `tauri.conf.json` in a guarded CI step so builds without the secret still work), e.g.
   `signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /f cert.pfx /p $PW %1`.

Also surface backend spawn failure — `lib.rs` already shows a boot-failed message if the
backend doesn't answer, so an AV-quarantined exe no longer hangs on the splash for 2 min.

## Optional — Linux

No Linux artifact is built. To add one: put `ubuntu-22.04` in the `release.yml` `bundle`
matrix, `apt-get install` Tauri's Linux deps (`libwebkit2gtk-4.1-dev`, `libappindicator3-dev`,
`librsvg2-dev`, `patchelf`), add `appimage`/`deb` to `bundle.targets`, and confirm the
PyInstaller sidecar freezes cleanly on Linux. Skipped by default — no evident demand and
the sidecar freeze is unverified on Linux.

## Reminders

- **Back up the updater key.** The minisign private key (`TAURI_SIGNING_PRIVATE_KEY`
  secret, backup at `C:\Users\Saeed\vocal-remover-updater-PRIVATE.key`) is irreplaceable —
  lose it and existing installs can't verify updates and must reinstall manually.
- **MSI → NSIS transition.** Existing MSI-installed users won't get auto-updates (the feed
  no longer has an `msi` entry); they reinstall once via the NSIS `setup.exe`. NSIS was the
  default download anyway.
