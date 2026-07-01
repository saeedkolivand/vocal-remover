"""End-to-end smoke test: generate a 5s stereo tone, run it through the real
separation pipeline, assert both stems come out non-empty. Verifies torch+CUDA,
ffmpeg, audio-separator, the configured MODEL, and our split() all work together."""
import pathlib, math, struct, wave, tempfile

from app import split  # imports app.py -> loads the model once

def make_tone(path, secs=5, sr=44100, hz=220):
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        frames = bytearray()
        for i in range(secs * sr):
            v = int(0.3 * 32767 * math.sin(2 * math.pi * hz * i / sr))
            frames += struct.pack("<hh", v, v)
        w.writeframes(frames)

tmp = pathlib.Path(tempfile.mkdtemp())
make_tone(tmp / "tone.wav")
stems = split(tmp / "tone.wav", tmp)

assert set(stems) == {"vocals", "instrumental"}, f"expected vocals+instrumental, got {stems}"
for stem, name in stems.items():
    p = tmp / name
    assert p.exists() and p.stat().st_size > 0, f"missing/empty {name}"
    print(f"  OK  {name}  ({p.stat().st_size//1024} KB)")

print("PASS — full pipeline works end to end.")
