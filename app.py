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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB upload cap

# The model loads (and, on first run, downloads ~200 MB+) in a background thread so
# Flask serves immediately and the UI can show progress via /status instead of a
# frozen window. split() and /separate wait on MODEL_READY.
separator = None
MODEL_READY = threading.Event()
status = {"ready": False, "phase": "starting", "downloaded_mb": 0.0, "error": None}


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


def _load_model():
    global separator
    try:
        cached = (MODELS / MODEL).exists()
        status["phase"] = "loading" if cached else "downloading"
        if not cached:
            threading.Thread(target=_watch_download, daemon=True).start()
        sep = Separator(
            output_format="flac",          # lossless — avoids the second lossy pass MP3 adds
            output_dir=str(OUT),
            model_file_dir=str(MODELS),    # stable cache, not /tmp
            # Device priority (audio-separator): CUDA > MPS > DirectML > CPU. This flag
            # only matters when torch_directml is installed and there's no CUDA/MPS —
            # i.e. the packaged Windows build, where it runs the model on ANY GPU
            # (incl. NVIDIA) via DirectX. Harmless in the CUDA dev env (CUDA wins).
            use_directml=True,
            # `overlap` is the chunk STEP in seconds: lower = more overlap = fewer seam
            # artifacts (and slower). ~2s step on ~6s chunks ≈ 66% overlap; the 4090 eats it.
            mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                         "batch_size": 1, "overlap": 2, "pitch_shift": 0},
        )
        sep.load_model(model_filename=MODEL)
        separator = sep
        status.update(phase="ready", ready=True)
    except Exception as e:
        status.update(phase="error", error=str(e))
    finally:
        MODEL_READY.set()  # unblock waiters whether we succeeded or errored


threading.Thread(target=_load_model, daemon=True).start()

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
  @keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}
  @media (prefers-reduced-motion:reduce){.aurora,.dot,.bar>i{animation:none}}
</style>
<div class=aurora></div>
<div class=wrap>
  <div class=dot></div>
  <h1 id=t>Starting Vocal Remover…</h1>
  <p id=m>Warming up the engine.</p>
  <div class=bar><i></i></div>
</div>
<script>
const t=document.getElementById('t'),m=document.getElementById('m');
async function tick(){
  try{
    const s=await (await fetch('/status',{cache:'no-store'})).json();
    if(s.ready){location.reload();return}
    if(s.phase==='downloading'){t.textContent='Downloading AI model…';
      m.textContent='One-time download'+(s.downloaded_mb?' — '+s.downloaded_mb+' MB':'')+'. This can take a minute.';}
    else if(s.phase==='loading'){t.textContent='Loading model…';m.textContent='Almost there.';}
    else if(s.phase==='error'){t.textContent='Failed to start';
      m.textContent=s.error||'The backend could not load the model.';return;}
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
