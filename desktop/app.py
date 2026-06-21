"""ToothPrint Desktop — a native window over the local certification server.

Cross-platform (Linux / Windows / macOS): starts the FastAPI backend in-process on a
free localhost port, then opens it in a native webview window. If no webview backend
or display is available (e.g. a headless box), it falls back to the default browser.

    python -m desktop.app          # from the repo root, with ".[api,io]" installed
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve(port: int) -> None:
    import uvicorn
    from api.main import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def _wait(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.3).close()
            return True
        except OSError:
            time.sleep(0.1)
    return False


def main() -> None:
    port = _free_port()
    threading.Thread(target=_serve, args=(port,), daemon=True).start()
    if not _wait(port):
        print("ToothPrint server failed to start", file=sys.stderr)
        sys.exit(1)
    url = f"http://127.0.0.1:{port}/studio"
    try:
        import webview                       # native window when a GUI is available
        webview.create_window("ToothPrint Studio", url, width=1200, height=840,
                              min_size=(960, 680))
        webview.start()
    except Exception:                          # headless / no webview -> browser
        print(f"ToothPrint Studio is running at {url}")
        print("Open it in your browser. Press Ctrl+C to quit.")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
