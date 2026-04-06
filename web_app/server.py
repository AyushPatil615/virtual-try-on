"""
DCI-VTON Web Server
===================
Wraps the existing 3-phase pipeline in a FastAPI REST API.

Run with:
    conda activate dci-vton
    python web_app/start.py
"""

import os, sys, io, uuid, threading, queue, json, shutil, asyncio, time
from pathlib import Path
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress tqdm / invalid handle noise before any ML imports
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── FastAPI ───────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Import pipeline functions from run_app (safe — __main__ guard) ───
import importlib.util as _ilu

def _load_run_app():
    spec = _ilu.spec_from_file_location("run_app", str(PROJECT_ROOT / "run_app.py"))
    mod  = _ilu.module_from_spec(spec)
    # Prevent exec of tkinter/GUI at import time by stubbing ImageTk if needed
    spec.loader.exec_module(mod)
    return mod

_run_app = _load_run_app()
phase1_preprocess = _run_app.phase1_preprocess
phase2_warp       = _run_app.phase2_warp
phase3_inference  = _run_app.phase3_inference

# ── Directories ───────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
JOBS_DIR   = PROJECT_ROOT / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="DCI-VTON Virtual Try-On", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (index.html, assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Job registry ──────────────────────────────────────────────────────
# jobs[job_id] = {
#   "status":      "queued" | "running" | "done" | "error",
#   "log_list":    [str, ...],         # all log lines so far
#   "log_queue":   Queue,              # new lines for SSE streaming
#   "result_path": str | None,
#   "error":       str | None,
# }
jobs: dict = {}

# Only one inference at a time (GPU constraint)
_gpu_lock = threading.Lock()
_job_queue: queue.Queue = queue.Queue()


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "active_jobs": sum(1 for j in jobs.values() if j["status"] == "running")}


@app.post("/api/run")
async def run_tryon(
    person: UploadFile = File(..., description="Person photo"),
    cloth:  UploadFile = File(..., description="Garment photo"),
):
    """Upload person + cloth images, start pipeline, return job_id."""
    job_id  = str(uuid.uuid4())[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploads
    person_ext = Path(person.filename).suffix if person.filename else ".jpg"
    cloth_ext  = Path(cloth.filename).suffix  if cloth.filename  else ".jpg"
    if person_ext.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        person_ext = ".jpg"
    if cloth_ext.lower()  not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        cloth_ext  = ".jpg"

    person_path = job_dir / f"person{person_ext}"
    cloth_path  = job_dir / f"cloth{cloth_ext}"

    person_path.write_bytes(await person.read())
    cloth_path.write_bytes(await cloth.read())

    jobs[job_id] = {
        "status":      "queued",
        "log_list":    [],
        "log_queue":   queue.Queue(),
        "result_path": None,
        "error":       None,
    }

    # Spawn background worker thread
    t = threading.Thread(
        target=_worker,
        args=(job_id, str(person_path), str(cloth_path)),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {
        "status":     j["status"],
        "error":      j["error"],
        "has_result": j["result_path"] is not None,
    }


@app.get("/api/stream/{job_id}")
async def stream_logs(job_id: str):
    """Server-Sent Events endpoint — streams log lines as they arrive."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    async def generator():
        # Replay already-recorded lines first (for reconnect)
        for line in list(job["log_list"]):
            yield f"data: {json.dumps({'log': line})}\n\n"

        # Stream new lines until job finishes
        while True:
            try:
                line = job["log_queue"].get(timeout=0.4)
                yield f"data: {json.dumps({'log': line})}\n\n"
                if line in ("__DONE__", "__ERROR__"):
                    break
            except queue.Empty:
                if job["status"] in ("done", "error"):
                    break
                # Keep-alive ping
                yield f"data: {json.dumps({'ping': True})}\n\n"
                await asyncio.sleep(0.1)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """Return the output PNG image when the job is done."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done" or not j["result_path"]:
        raise HTTPException(404, "Result not ready yet")
    rp = Path(j["result_path"])
    if not rp.exists():
        raise HTTPException(404, "Result file missing")
    return FileResponse(str(rp), media_type="image/png",
                        filename=f"tryon_{job_id}.png")


# ── Background worker ─────────────────────────────────────────────────

def _worker(job_id: str, person_path: str, cloth_path: str):
    """Runs in a daemon thread. Serialises GPU access via _gpu_lock."""
    job = jobs[job_id]

    def log(msg: str):
        job["log_list"].append(msg)
        job["log_queue"].put(msg)

    # Redirect stdout/stderr to avoid handles errors on console-less launch
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
    except Exception:
        pass

    try:
        with _gpu_lock:
            job["status"] = "running"
            log("🔒 GPU acquired — starting pipeline")

            # ── Phase 1 ────────────────────────────────────────────
            log("━" * 55)
            log("PHASE 1 — PREPROCESSING")
            log("━" * 55)
            phase1_preprocess(person_path, cloth_path, log)

            # Free GPU between phases
            try:
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                log("  ✓ GPU cleared after Phase 1")
            except Exception:
                pass

            # ── Phase 2 ────────────────────────────────────────────
            phase2_warp(log)

            # ── Phase 3 ────────────────────────────────────────────
            result_path = phase3_inference(log)

        if result_path and Path(result_path).exists():
            # Copy result to job-specific location for safe download
            job_result = JOBS_DIR / job_id / "result.png"
            shutil.copy2(result_path, job_result)
            job["result_path"] = str(job_result)
            job["status"]      = "done"
            log(f"\n✅ Done!  result saved → {job_result}")
            log("__DONE__")
        else:
            raise RuntimeError("Pipeline completed but no output image found.")

    except Exception as exc:
        import traceback
        job["status"] = "error"
        job["error"]  = str(exc)
        log(f"\n❌ ERROR: {exc}")
        log(traceback.format_exc())
        log("__ERROR__")
    finally:
        try:
            sys.stdout = _original_stdout
            sys.stderr = _original_stderr
        except Exception:
            pass
