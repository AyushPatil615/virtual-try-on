"""
DCI-VTON Web Launcher
=====================
Starts the FastAPI server and opens a public ngrok tunnel.

Just run:
    python web_app/start.py
"""

import sys, os, time, threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 8000
HOST = "0.0.0.0"

# ── ngrok auth token (already configured) ────────────────────────────
NGROK_TOKEN = "3BwVPtg5judoT7yZK6vY6MqF1Ez_51rsmHyLbB5MoRP7PATWZ"


def start_tunnel(port: int):
    """Open ngrok tunnel and print the public URL."""
    time.sleep(3)  # Wait for uvicorn to be ready
    try:
        from pyngrok import ngrok, conf

        # Set auth token
        conf.get_default().auth_token = NGROK_TOKEN

        print("\n  Opening public tunnel via ngrok...")
        tunnel = ngrok.connect(port, "http")
        url = tunnel.public_url
        # Upgrade to https
        if url.startswith("http://"):
            url = "https://" + url[7:]

        print("\n" + "=" * 65)
        print(f"  >>> PUBLIC URL: {url}")
        print("=" * 65)
        print("  Open this link on ANY device - phone, tablet, laptop.")
        print("  Works from anywhere in the world.")
        print("  Keep this window OPEN while presenting.")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n  [Tunnel error] {e}")
        print(f"  Local only: http://localhost:{port}\n")


def main():
    print("\n  DCI-VTON Virtual Try-On  |  Web Server")
    print(f"  Starting on http://localhost:{PORT} ...")
    print()

    # Start tunnel in background
    t = threading.Thread(target=start_tunnel, args=(PORT,), daemon=True)
    t.start()

    # Start uvicorn (blocks until Ctrl+C)
    import uvicorn
    uvicorn.run(
        "web_app.server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="warning",
        app_dir=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
