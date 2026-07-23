"""
CRAFTAI — Application Launcher
--------------------------------
Usage:
    python main.py            # start API + UI
    python main.py --api-only # start FastAPI only (no Streamlit)

Prerequisites (do these once before running):
    pip install -r requirements.txt
    cp .env.example .env       # then add your GROQ_API_KEY
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Windows consoles often default to cp1252, which can't encode the box-drawing
# and status characters (╔ ✓ ⚠ ✗) used below — force UTF-8 so the banner and
# status lines don't crash the launcher before anything else even starts.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Interpreter resolution ────────────────────────────────────────────────────
# Do NOT trust sys.executable to launch subprocesses. On Windows a
# virtualenv-created `venv\Scripts\python.exe` is a launcher stub that delegates
# to the base interpreter, so sys.executable inside a running script can report
# the *system* Python — spawning uvicorn/streamlit with that bypasses the venv's
# packages (torch, sentence-transformers, langchain-groq) and the app hangs /
# fails mid-pipeline. Resolve the project venv interpreter explicitly instead.

def _find_venv_dir() -> Path | None:
    for d in (ROOT / "venv", ROOT / ".venv"):
        if (d / "Scripts" / "python.exe").exists() or (d / "bin" / "python").exists():
            return d
    return None


_VENV_DIR = _find_venv_dir()


def _venv_python() -> str:
    """Absolute path to the project venv's Python; falls back to sys.executable."""
    if _VENV_DIR:
        win = _VENV_DIR / "Scripts" / "python.exe"
        posix = _VENV_DIR / "bin" / "python"
        return str(win if win.exists() else posix)
    return sys.executable


def _venv_env(base: dict) -> dict:
    """Env that mimics `activate` so subprocesses (incl. uvicorn --reload
    workers) resolve to the venv regardless of how they respawn."""
    env = dict(base)
    if _VENV_DIR:
        scripts = _VENV_DIR / ("Scripts" if os.name == "nt" else "bin")
        env["VIRTUAL_ENV"] = str(_VENV_DIR)
        env["PATH"] = str(scripts) + os.pathsep + env.get("PATH", "")
        env.pop("PYTHONHOME", None)
    return env


# ── ANSI colours ─────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def info(msg: str)  -> None: print(_c(f"  [CRAFTAI] {msg}", "96"))
def ok(msg: str)    -> None: print(_c(f"  ✓  {msg}", "92"))
def warn(msg: str)  -> None: print(_c(f"  ⚠  {msg}", "93"))
def err(msg: str)   -> None: print(_c(f"  ✗  {msg}", "91"), file=sys.stderr)


def banner() -> None:
    print(_c("""
╔══════════════════════════════════════════════════════════════╗
║   CRAFTAI — Intelligent Marketing Content Supply Chain       ║
║   LangGraph · ChromaDB · FastAPI · Streamlit · MLflow        ║
╚══════════════════════════════════════════════════════════════╝
""", "95"))


# ── Check .env exists ─────────────────────────────────────────────────────────

def check_env() -> None:
    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"

    if not env_path.exists():
        if example_path.exists():
            import shutil
            shutil.copy(example_path, env_path)
            warn(".env was missing — copied from .env.example.")
            warn("Open .env and set your GROQ_API_KEY, then re-run.")
        else:
            warn(".env file not found. Create it with at least: GROQ_API_KEY=your_key")

    # Load .env into os.environ
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        warn("GROQ_API_KEY is not set — LLM calls will fail.")
        warn("Add it to .env:  GROQ_API_KEY=gsk_...")
    else:
        ok("GROQ_API_KEY loaded.")


# ── Seed DB if needed ─────────────────────────────────────────────────────────

def seed_if_needed() -> None:
    db_path = Path(os.environ.get("SQLITE_DB_PATH", str(ROOT / "data" / "craftai.db")))
    if db_path.exists() and db_path.stat().st_size > 0:
        ok("Knowledge base already seeded.")
        return

    info("First run — seeding ChromaDB + SQLite (may take ~30 s)…")
    try:
        subprocess.check_call(
            [_venv_python(), str(ROOT / "scripts" / "init_chroma.py")],
            cwd=str(ROOT),
            env=_venv_env({**os.environ, "PYTHONPATH": str(ROOT)}),
        )
        subprocess.check_call(
            [_venv_python(), str(ROOT / "scripts" / "generate_guidelines.py")],
            cwd=str(ROOT),
            env=_venv_env({**os.environ, "PYTHONPATH": str(ROOT)}),
        )
        ok("Knowledge base seeded.")
    except subprocess.CalledProcessError as exc:
        warn(f"Seeding failed ({exc}). App will start but RAG context will be empty.")


# ── Process management ────────────────────────────────────────────────────────

_procs: list[subprocess.Popen] = []


def _stream(proc: subprocess.Popen, label: str, colour: str) -> None:
    """Read stdout from proc and print with a coloured label prefix."""
    def _read() -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            print(_c(f"[{label}]", colour) + " " + line.rstrip())
    threading.Thread(target=_read, daemon=True).start()


def _shutdown(sig=None, frame=None) -> None:
    print()
    info("Shutting down…")
    for p in _procs:
        if p.poll() is None:
            p.terminate()
    time.sleep(1)
    for p in _procs:
        if p.poll() is None:
            p.kill()
    ok("Stopped. Goodbye!")
    sys.exit(0)


def start(api_only: bool = False) -> None:
    env = _venv_env({**os.environ, "PYTHONPATH": str(ROOT)})

    py = _venv_python()
    if _VENV_DIR:
        ok(f"Using venv interpreter: {py}")
    else:
        warn(f"No project venv found — using {py}. Packages may be missing.")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    # --reload is OFF by default: uvicorn's reloader respawns its worker via its
    # own interpreter resolution, which on a venv-stub Windows setup lands the
    # worker back on the *system* Python (missing torch/sentence-transformers),
    # hanging the pipeline at retrieval. Without --reload the process we launch
    # here (explicit venv Python) IS the server. Opt back in with CRAFTAI_RELOAD=1.
    uvicorn_cmd = [
        py, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ]
    if os.environ.get("CRAFTAI_RELOAD") == "1":
        uvicorn_cmd.append("--reload")
        warn("CRAFTAI_RELOAD=1 — auto-reload on; worker may use the wrong interpreter on venv-stub setups.")

    info("Starting FastAPI  →  http://localhost:8000   (docs: /docs)")
    api = subprocess.Popen(
        uvicorn_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _procs.append(api)
    _stream(api, "API ", "94")      # blue

    time.sleep(2)                   # let the API bind before Streamlit opens

    # ── Streamlit ─────────────────────────────────────────────────────────────
    if not api_only:
        info("Starting Streamlit  →  http://localhost:8501")
        ui = subprocess.Popen(
            [
                py, "-m", "streamlit",
                "run", str(ROOT / "ui" / "app.py"),
                "--server.port", "8501",
                "--server.address", "0.0.0.0",
                "--server.headless", "true",
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _procs.append(ui)
        _stream(ui, "UI  ", "92")   # green

    print()
    print(_c("  ┌──────────────────────────────────────────────────┐", "95"))
    print(_c("  │  CRAFTAI is running!                             │", "95"))
    print(_c("  │                                                  │", "95"))
    print(_c("  │  API  →  http://localhost:8000                   │", "95"))
    print(_c("  │  Docs →  http://localhost:8000/docs              │", "95"))
    if not api_only:
        print(_c("  │  UI   →  http://localhost:8501                   │", "95"))
    print(_c("  │                                                  │", "95"))
    print(_c("  │  Press Ctrl-C to stop                            │", "95"))
    print(_c("  └──────────────────────────────────────────────────┘", "95"))
    print()


def watch() -> None:
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    while True:
        time.sleep(2)
        dead = [p for p in _procs if p.poll() is not None]
        if dead:
            for p in dead:
                label = "API" if _procs.index(p) == 0 else "UI"
                err(f"{label} process exited unexpectedly (code {p.returncode}).")
            _shutdown()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CRAFTAI launcher")
    parser.add_argument(
        "--api-only", action="store_true",
        help="Start the FastAPI backend only (no Streamlit UI).",
    )
    parser.add_argument(
        "--skip-seed", action="store_true",
        help="Skip the ChromaDB/SQLite seeding check.",
    )
    args = parser.parse_args()

    banner()
    check_env()

    if not args.skip_seed:
        seed_if_needed()

    start(api_only=args.api_only)
    watch()


if __name__ == "__main__":
    main()
