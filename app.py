"""Local vocal remover — Flask + Mel-Band RoFormer (audio-separator) on GPU.

Upload a song -> separates into vocals + instrumental and serves them back as
lossless FLAC. Everything runs locally; nothing leaves the machine.

Engine: Mel-Band RoFormer "Kim FT" (see MODEL below). Runs on PyTorch CUDA.
"""
import os, sys, json, time, shutil, pathlib, threading, uuid

# PyInstaller windowed build (console=False) sets sys.stdout/stderr to None, so any
# library that prints — static_ffmpeg's first-run downloader, audio-separator's
# logging — crashes on .write before Flask even starts. Point them at a log file so
# the writes succeed (and we get a packaged-run log). ponytail: only when there's no console.
if sys.stdout is None or sys.stderr is None:
    import tempfile
    _log = open(os.path.join(tempfile.gettempdir(), "vocal-remover-backend.log"), "a", buffering=1)
    sys.stdout = sys.stdout or _log
    sys.stderr = sys.stderr or _log

import static_ffmpeg
static_ffmpeg.add_paths()  # ffmpeg on PATH for decode + flac encode

from flask import Flask, request, jsonify, send_file, send_from_directory
from audio_separator.separator import Separator

FROZEN = getattr(sys, "frozen", False)               # True in the PyInstaller-built app
ROOT = pathlib.Path(__file__).parent
BASE = pathlib.Path(getattr(sys, "_MEIPASS", ROOT))  # bundled read-only assets (index.html)
# Writable data dir. Dev = repo; packaged = user-writable (the install dir is read-only).
DATA = (pathlib.Path(os.environ.get("VR_DATA")
        or pathlib.Path(os.environ.get("LOCALAPPDATA", pathlib.Path.home())) / "VocalRemover")
        if FROZEN else ROOT)
OUT = DATA / "output"; OUT.mkdir(parents=True, exist_ok=True)
UPLOAD = DATA / "uploads"; UPLOAD.mkdir(parents=True, exist_ok=True)


def _models_dir():
    """Where the model lives. Must be WRITABLE (audio-separator manages this dir).

    Dev: ./models (or MODEL_DIR).  Packaged: MODEL_DIR points at the read-only bundled
    copy (Tauri resource) — mirror it into a writable dir once so nothing tries to
    write into Program Files.
    """
    if not FROZEN:
        d = pathlib.Path(os.environ.get("MODEL_DIR", ROOT / "models"))
        d.mkdir(parents=True, exist_ok=True)
        return d
    d = DATA / "models"; d.mkdir(parents=True, exist_ok=True)
    src = os.environ.get("MODEL_DIR")
    if src and pathlib.Path(src).exists():
        for f in pathlib.Path(src).iterdir():
            if f.is_file() and not (d / f.name).exists():
                shutil.copy2(f, d / f.name)
    return d


MODELS = _models_dir()

# 2-stem (vocals / instrumental). Mel-Band RoFormer "Kim FT" — cleaner vocals with
# less bleed. Override with the MODEL env var to try others; audio-separator
# auto-downloads any model on first load (incl. old ones, if you ever want them back).
#   Alternatives: mel_band_roformer_kim_ft2_unwa.ckpt     (newer FT)
#                 bs_roformer_vocals_revive_v2_unwa.ckpt  (fuller vocals)
MODEL = os.environ.get("MODEL", "mel_band_roformer_kim_ft_unwa.ckpt")
TOTAL_MB = 871  # ~size of the Kim FT checkpoint — lets the first-run bar show a real %


def _accel():
    """Which compute backend the model will use: cuda / mps / directml / cpu.
    DirectML (any Windows GPU — NVIDIA/AMD/Intel) needs use_directml=True on the Separator
    plus the dml_hybrid patch (DirectML can't run the model's complex STFT). Drives the UI
    copy and the separation overlap."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            return "directml"
    except Exception:
        pass
    return "cpu"


ACCEL = _accel()
DEVICE = "gpu" if ACCEL in ("cuda", "mps", "directml") else "cpu"  # for the UI

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB upload cap

# The model loads (and, on first run, downloads ~900 MB) in a background thread so
# Flask serves immediately and the UI can show progress via /status instead of a
# frozen window. split() and /separate wait on MODEL_READY.
separator = None
MODEL_READY = threading.Event()
status = {"ready": False, "phase": "starting", "downloaded_mb": 0.0,
          "total_mb": TOTAL_MB, "device": DEVICE, "error": None}


def _watch_download():
    """Best-effort first-run download progress: sum the model file(s) growing in MODELS.
    ponytail: byte count off disk — robust across audio-separator versions; no % since we
    don't know the total. Add a total (from download_checks.json) if a real bar is wanted."""
    stem = MODEL.rsplit(".", 1)[0]
    while status["phase"] == "downloading":
        try:
            mb = sum(f.stat().st_size for f in MODELS.iterdir()
                     if f.is_file() and stem in f.name) / 1048576
            status["downloaded_mb"] = round(mb, 1)
        except OSError:
            pass
        time.sleep(0.5)


def _make_separator(use_dml):
    sep = Separator(
        output_format="flac",          # lossless — avoids the second lossy pass MP3 adds
        output_dir=str(OUT),
        model_file_dir=str(MODELS),    # stable cache, not /tmp
        use_directml=use_dml,          # route torch to the DirectML device on Windows GPUs
        # `overlap` is the chunk STEP in seconds: lower = more overlap = fewer seam artifacts
        # (and slower). ~2s ≈ 66% overlap, which native CUDA/MPS eat; on DirectML (CPU-side
        # STFT per chunk) and CPU we step wider to stay usable.
        mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                     "batch_size": 1, "overlap": 2 if ACCEL in ("cuda", "mps") else 4,
                     "pitch_shift": 0},
    )
    sep.load_model(model_filename=MODEL)
    return sep


def _dml_selftest_ok(sep):
    """One tiny forward on the GPU. DirectML op coverage is uniform across GPU vendors, but
    we can only test some hardware — if the GPU path doesn't actually run here, the caller
    falls back to CPU so the app is never worse than the plain CPU build."""
    try:
        import torch
        with torch.no_grad():
            sep.model_instance.model_run(torch.randn(1, 2, 44100, device=sep.torch_device))
        return True
    except Exception as e:
        print(f"DirectML self-test failed ({e}); falling back to CPU", flush=True)
        return False


def _load_model():
    global separator
    try:
        cached = (MODELS / MODEL).exists()
        status["phase"] = "loading" if cached else "downloading"
        if not cached:
            threading.Thread(target=_watch_download, daemon=True).start()
        use_dml = ACCEL == "directml"
        if use_dml:
            # DirectML fatally aborts on the model's complex STFT ops; the patch runs those
            # on CPU and the transformer on the GPU (~5x faster than CPU on any DX12 GPU).
            import dml_hybrid
            dml_hybrid.enable()
        sep = _make_separator(use_dml)
        if use_dml and not _dml_selftest_ok(sep):
            import dml_hybrid
            dml_hybrid.disable()
            status["device"] = "cpu"        # keep the UI copy honest after fallback
            sep = _make_separator(False)
        separator = sep
        status.update(phase="ready", ready=True)
    except Exception as e:
        # A killed/interrupted first-run download leaves a truncated .ckpt that torch can't
        # deserialize — and since "file exists" counts as cached, every later launch would
        # re-hit it and the app would be bricked forever. Delete it so a retry (or relaunch)
        # re-downloads cleanly. ponytail: nuke-on-failure; worst case is one wasted re-download
        # if a genuine (non-corruption) load bug ever trips this.
        try:
            (MODELS / MODEL).unlink(missing_ok=True)
        except OSError:
            pass
        status.update(phase="error", error=str(e))
    finally:
        MODEL_READY.set()  # unblock waiters whether we succeeded or errored


def start_load():
    """(Re)start model loading from a clean status — used at boot and by POST /retry."""
    status.update(ready=False, phase="starting", downloaded_mb=0.0, error=None)
    MODEL_READY.clear()
    threading.Thread(target=_load_model, daemon=True).start()


start_load()

# ponytail: one GPU, so serialize jobs with a lock — also guards the shared
# separator.output_dir we set per job. Add a real queue only if this ever
# becomes multi-user (it won't — it's a local tool).
gpu_lock = threading.Lock()


def split(in_path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    """Separate; move the produced stems into out_dir. Returns {stem: filename}.

    Routes by the stem name inside each produced filename (not fixed names), so
    changing MODEL or output_format can't mislabel or silently drop a stem. The
    gpu_lock serializes jobs (one GPU) and guards the shared separator output_dir.
    """
    MODEL_READY.wait()  # first job may arrive before the model finishes loading
    if separator is None:
        raise RuntimeError(status["error"] or "model failed to load")
    with gpu_lock:
        produced = separator.separate(str(in_path))
    # Resolve to real paths (separate() returns names relative to output_dir).
    paths = [p if (p := pathlib.Path(f)).is_absolute() else OUT / p for f in produced]
    paths = [p for p in paths if p.exists()]
    # 2-stem models: one file is vocals; the complement is labelled differently per
    # model ("Instrumental", "Other", "No Vocals", ...), so treat "not vocals" as it.
    vocals = next((p for p in paths if "vocal" in p.name.lower()), None)
    instrumental = next((p for p in paths if p is not vocals), None)
    moved = {}
    for stem, p in (("vocals", vocals), ("instrumental", instrumental)):
        if p:
            dest = out_dir / f"{stem}{p.suffix}"
            p.replace(dest)
            moved[stem] = dest.name
    return moved


LOADING_HTML = """<!doctype html><meta charset=utf-8>
<title>Vocal Remover</title>
<style>
  *{box-sizing:border-box;margin:0}html,body{height:100%}
  body{background:oklch(0.15 0.012 235);color:oklch(0.97 0.005 235);
    font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    display:grid;place-items:center;height:100vh;overflow:hidden}
  .aurora{position:fixed;inset:-30vmax;z-index:0;pointer-events:none;
    background:radial-gradient(40vmax 40vmax at 35% 35%,oklch(0.72 0.19 35/0.20),transparent 60%),
      radial-gradient(42vmax 42vmax at 68% 65%,oklch(0.80 0.13 205/0.18),transparent 62%);
    filter:blur(40px);animation:drift 34s linear infinite}
  @keyframes drift{to{transform:rotate(360deg)}}
  .wrap{position:relative;z-index:1;text-align:center;width:min(90vw,380px)}
  .dot{width:18px;height:18px;border-radius:50%;margin:0 auto 1.6rem;background:oklch(0.82 0.17 52);
    box-shadow:0 0 18px oklch(0.82 0.17 52),0 0 44px oklch(0.72 0.19 35);
    animation:pulse 1.6s cubic-bezier(0.16,1,0.3,1) infinite}
  @keyframes pulse{0%,100%{transform:scale(.7);opacity:.5}50%{transform:scale(1);opacity:1}}
  h1{font-size:1.5rem;font-weight:700;letter-spacing:-0.02em}
  p{margin-top:.5rem;color:oklch(0.74 0.012 235);font-size:.9rem}
  .bar{margin-top:1.2rem;height:4px;border-radius:99px;background:oklch(0.30 0.02 235);overflow:hidden}
  .bar>i{display:block;height:100%;width:35%;border-radius:99px;background:oklch(0.82 0.17 52);
    animation:slide 1.4s ease-in-out infinite}
  .bar>i.det{animation:none;margin-left:0;transition:width .4s ease}
  @keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}
  .btn{margin-top:1.4rem;display:none;font:inherit;font-weight:600;cursor:pointer;
    color:oklch(0.15 0.012 235);background:oklch(0.82 0.17 52);border:none;border-radius:10px;padding:.6rem 1.4rem}
  .btn:hover{filter:brightness(1.06)}
  @media (prefers-reduced-motion:reduce){.aurora,.dot,.bar>i{animation:none}}
</style>
<div class=aurora></div>
<div class=wrap>
  <div class=dot></div>
  <h1 id=t>Starting Vocal Remover…</h1>
  <p id=m>Warming up the engine.</p>
  <div class=bar><i id=bi></i></div>
  <button id=retry class=btn onclick="doRetry()">Try again</button>
</div>
<script>
const t=document.getElementById('t'),m=document.getElementById('m'),
      bi=document.getElementById('bi'),retry=document.getElementById('retry');
function friendly(err){
  return /Connection|Max retries|Failed to establish|getaddrinfo|Temporary failure|NewConnectionError/i.test(err||'')
    ? 'No internet connection — the one-time ~900 MB model download needs one.'
    : (err||'The backend could not load the model.');
}
async function doRetry(){
  retry.style.display='none';
  try{ await fetch('/retry',{method:'POST'}); }catch(e){}
  tick();
}
async function tick(){
  try{
    const s=await (await fetch('/status',{cache:'no-store'})).json();
    if(s.ready){location.reload();return}
    if(s.phase==='downloading'){
      t.textContent='Downloading AI model…';
      const pct=s.total_mb?Math.min(99,Math.round(s.downloaded_mb/s.total_mb*100)):0;
      m.textContent='One-time download — '+(s.downloaded_mb||0)+' of '+(s.total_mb||'?')+' MB'+(pct?' ('+pct+'%)':'');
      bi.classList.add('det'); bi.style.width=pct+'%';
    }
    else if(s.phase==='loading'){t.textContent='Loading model…';m.textContent='Almost there.';
      bi.classList.remove('det'); bi.style.width='';}
    else if(s.phase==='error'){t.textContent='Couldn’t start';
      m.textContent=friendly(s.error); bi.classList.add('det'); bi.style.width='0%';
      retry.style.display='inline-block'; return;}
    else{bi.classList.remove('det'); bi.style.width='';}
  }catch(e){/* backend still booting */}
  setTimeout(tick,600);
}
tick();
</script>
"""


@app.get("/")
def index():
    # Until the model is ready, serve a same-origin progress page that polls /status
    # and reloads into the app once ready (no CORS/cross-scheme fetch from the shell).
    if not status["ready"]:
        return LOADING_HTML
    return send_file(BASE / "index.html")


@app.get("/status")
def status_route():
    return jsonify(status)


@app.post("/retry")
def retry_route():
    # Re-attempt a failed load (e.g. the first-run download dropped). No-op unless errored.
    if status["phase"] == "error":
        start_load()
    return jsonify(status)


@app.post("/separate")
def separate():
    if not status["ready"]:
        return jsonify(error="model still loading", status=status), 503
    f = request.files.get("audio")
    if not f or not f.filename:
        return jsonify(error="no audio file"), 400

    job = uuid.uuid4().hex[:12]
    job_dir = OUT / job; job_dir.mkdir(parents=True, exist_ok=True)
    in_path = UPLOAD / f"{job}_{pathlib.Path(f.filename).name}"
    f.save(in_path)
    try:
        stems = split(in_path, job_dir)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        in_path.unlink(missing_ok=True)

    if "vocals" not in stems or "instrumental" not in stems:
        return jsonify(error=f"model produced {sorted(stems) or 'no'} stems, expected vocals+instrumental"), 500

    name = pathlib.Path(f.filename).stem
    created = int(time.time())
    (job_dir / "meta.json").write_text(json.dumps({"name": name, "created": created}))  # for /history

    return jsonify(
        name=name, created=created,
        vocals=f"/file/{job}/{stems['vocals']}",
        instrumental=f"/file/{job}/{stems['instrumental']}",
    )


@app.get("/file/<job>/<name>")
def file(job, name):
    return send_from_directory(OUT / job, name)  # inline — for the in-page audio players


@app.get("/download/<job>/<name>")
def download(job, name):
    # as_attachment sets Content-Disposition, so browsers save it AND Tauri's WebView2
    # native downloader kicks in (the <a download> attribute alone doesn't trigger it).
    return send_from_directory(OUT / job, name, as_attachment=True)


@app.get("/history")
def history():
    """Past jobs, newest first — scanned from output/ (+ per-job meta.json)."""
    items = []
    for d in OUT.iterdir():
        if not d.is_dir():
            continue
        stems = {p.stem.lower(): p.name for p in d.iterdir() if p.stem.lower() in ("vocals", "instrumental")}
        if "vocals" not in stems or "instrumental" not in stems:
            continue
        meta = {}
        try:
            meta = json.loads((d / "meta.json").read_text())
        except Exception:
            pass
        items.append({
            "job": d.name,
            "name": meta.get("name") or d.name,
            "created": meta.get("created") or int(d.stat().st_mtime),
            "vocals": f"/file/{d.name}/{stems['vocals']}",
            "instrumental": f"/file/{d.name}/{stems['instrumental']}",
        })
    items.sort(key=lambda x: x["created"], reverse=True)
    return jsonify(items)


@app.delete("/history/<job>")
def history_delete(job):
    """Delete a past result to reclaim disk (stems accumulate forever otherwise).
    Resolve + parent-check so a crafted job id can't escape the output dir."""
    d = (OUT / job).resolve()
    if d.parent != OUT.resolve() or not d.is_dir():
        return jsonify(error="not found"), 404
    shutil.rmtree(d, ignore_errors=True)
    return jsonify(ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
