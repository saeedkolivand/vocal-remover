"""Local vocal remover — Flask + Mel-Band RoFormer (audio-separator) on GPU.

Upload a song -> separates into vocals + instrumental and serves them back as
lossless FLAC. Everything runs locally; nothing leaves the machine.

Engine: Mel-Band RoFormer "Kim FT" (see MODEL below). Runs on PyTorch CUDA.
"""
import os, sys, json, time, shutil, pathlib, threading, uuid

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

print(f"Loading {MODEL} (downloads ~200 MB on first run)...")
separator = Separator(
    output_format="flac",          # lossless — avoids the second lossy pass MP3 adds
    output_dir=str(OUT),
    model_file_dir=str(MODELS),    # stable cache, not /tmp
    # `overlap` is the chunk STEP in seconds: lower = more overlap = fewer seam
    # artifacts (and slower). ~2s step on ~6s chunks ≈ 66% overlap; the 4090 eats it.
    mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                 "batch_size": 1, "overlap": 2, "pitch_shift": 0},
)
separator.load_model(model_filename=MODEL)
print("Model ready.")

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


@app.get("/")
def index():
    return send_file(BASE / "index.html")


@app.post("/separate")
def separate():
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
