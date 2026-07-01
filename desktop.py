"""Cross-platform desktop app (Windows / macOS / Linux).

Reuses the exact Flask app + UI. Runs Flask on a background thread and shows it
in a native OS webview window (Edge WebView2 on Windows, WKWebView on macOS,
WebKitGTK on Linux) — no Chromium bundled, no second language.

Run:  .venv/Scripts/python desktop.py   (Windows)
      ./.venv/bin/python desktop.py      (macOS/Linux)
"""
import threading, time, urllib.request

import webview
from app import app  # importing loads the Demucs model once

PORT = 8000


def serve():
    app.run(host="127.0.0.1", port=PORT, threaded=True)


def wait_ready(timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


if __name__ == "__main__":
    threading.Thread(target=serve, daemon=True).start()
    wait_ready()
    webview.create_window("Vocal Remover", f"http://127.0.0.1:{PORT}",
                          width=1120, height=840, min_size=(820, 620))
    webview.start()
