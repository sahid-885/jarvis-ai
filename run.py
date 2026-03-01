"""
RUN SCRIPT - Start the J.A.R.V.I.S server
=========================================

PURPOSE:
  Single entry point to start the backend. Run this once per user/machine;
  the server then handles all chat and realtime requests for that instance.

WHAT IT DOES:
  - Imports the FastAPI app from app.main.
  - Runs it with uvicorn on host 0.0.0.0 (accept connections from any interface) and port 8000.
  - reload=True means any change to Python files will restart the server (handy for development).

  #.\.venv\Scripts\python.exe run.py
USAGE:
    python run.py

  Then open http://localhost:8000 in the browser, or use the API from another app.

NOTE:
  Before running, set GROQ_API_KEY (and optionally TAVILY_API_KEY for realtime search) in .env.

─────────────────────────────────────────────────────────────────────────────
WHAT IS UVICORN?
─────────────────────────────────────────────────────────────────────────────
Uvicorn is an ASGI (Asynchronous Server Gateway Interface) server for Python.

In simple terms:
  - FastAPI defines *what* happens when a request arrives (your route handlers).
  - Uvicorn is the *engine* that actually listens on a network port, accepts
    incoming HTTP connections, and hands each request to FastAPI for processing.

Think of it like this:
  - FastAPI = the chef who knows every recipe.
  - Uvicorn = the restaurant that seats customers and delivers their orders to the chef.

ASGI (vs. WSGI):
  - WSGI (older standard): handles one request at a time per worker. Used by
    Flask, Django (traditional).
  - ASGI (newer standard): supports async/await, WebSockets, and can handle
    many concurrent connections efficiently. FastAPI is built for ASGI.

─────────────────────────────────────────────────────────────────────────────
THE if __name__ == "__main__" GUARD
─────────────────────────────────────────────────────────────────────────────
This is a standard Python idiom. When you run a file directly:
    python run.py
Python sets the special variable __name__ to "__main__" for that file.

When a file is *imported* by another module:
    import run      # (hypothetical)
__name__ is set to "run" (the module name), NOT "__main__".

The guard ensures that uvicorn.run() is ONLY called when you execute this
file directly, not when it happens to be imported. Without this guard, simply
importing this module would start the server — which is almost never what
you want.
"""

# ─── uvicorn: the ASGI server that runs our FastAPI application ─────────────
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv("/etc/secrets/.env")

print("🚀 Starting server...") 

# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
# Only run uvicorn when this file is executed directly (python run.py),
# not when it is imported by another module.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Using port: {port}")

    uvicorn.run(
        # ── "app.main:app" ──────────────────────────────────────────────────
        # This is a string import path, not a direct Python import. It tells
        # uvicorn: "look in the module app.main and find the variable called app."
        # Using a string (instead of importing the object directly) is REQUIRED
        # when reload=True, because uvicorn needs to re-import the module on each
        # reload — it can't do that if you already hold a reference to the object.
        "app.main:app",

        # ── host="0.0.0.0" ─────────────────────────────────────────────────
        # Listen on ALL network interfaces. This means:
        #   - localhost / 127.0.0.1: your own machine (always works).
        #   - Your LAN IP (e.g. 192.168.1.42): other devices on your Wi-Fi can
        #     reach the server (useful for testing on a phone or another PC).
        # If you set host="127.0.0.1", only your own machine could connect.
        host="0.0.0.0",

        # ── port=8000 ──────────────────────────────────────────────────────
        # The TCP port to listen on. 8000 is a common default for development
        # servers. After starting, the API is at http://localhost:8000.
        # If port 8000 is already in use, change this to another number (e.g. 8001).
        port=port,

        # ── reload=True ────────────────────────────────────────────────────
        # Enables auto-reload: uvicorn watches all .py files in the project and
        # automatically restarts the server whenever a file changes. This is
        # extremely convenient during development — you edit code, save, and the
        # server picks up changes without you manually stopping and restarting it.
        # In production, set reload=False (or omit it) because file-watching adds
        # overhead and you don't want unreviewed changes going live.
        reload=False
    )
