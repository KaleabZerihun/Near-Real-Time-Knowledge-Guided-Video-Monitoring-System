from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from dotenv import load_dotenv

# Ensure `src.*` imports work when running from repo root or backend/
THIS_DIR = Path(__file__).resolve().parent  # .../backend
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.db.init_db import init_db
from src.frame_selector.config import FrameSelectorConfig
from src.api.routes import router as api_router

# Try to import PipelineRunner; if ML deps are missing, keep API running
try:
    from src.pipeline.runner import PipelineRunner
except ModuleNotFoundError as e:
    PipelineRunner = None  # type: ignore
    print(f"[WARN] PipelineRunner disabled (missing dependency): {e}")

app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring (Sprint 1)")

runner = None  # PipelineRunner | None (kept untyped for when PipelineRunner is None)


@app.on_event("startup")
def startup() -> None:
    global runner

    # Repo root is one level above backend/
    repo_root = THIS_DIR.parent
    thesis_root = repo_root / "THESIS"

    # Load env vars from repo root (.env)
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    # Stable default DB location: backend/data/app.db
    default_db_path = (THIS_DIR / "data" / "app.db").resolve()
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path.as_posix()}")

    # Ensure DB + tables exist
    init_db(database_url)

    # If ML deps missing, skip runner but keep API/Dashboard alive
    if PipelineRunner is None:
        print("[WARN] Skipping PipelineRunner startup; API will run without pipeline.")
        runner = None
        return

    cfg = FrameSelectorConfig(
        source=0, #which webcam
        source_id="webcam0", 
        target_fps=8.0, #how many fps
        resize_hw=(224, 224), #resize frames for vad model input
        clip_len=16, # how batches are formed
        stride=8, 
        frame_ring_maxlen=256, # buffer size
        max_batches=8, #queue size
    )

    runner = PipelineRunner(cfg=cfg, thesis_root=str(thesis_root))
    runner.start()


@app.on_event("shutdown")
def shutdown() -> None:
    global runner
    if runner:
        runner.stop()
        runner = None
    os._exit(0)


@app.get("/status")
def status():
    """Simple status endpoint so the dashboard can decide whether to call pipeline endpoints."""
    return {"pipeline_enabled": runner is not None}


@app.get("/favicon.ico")
def favicon():
    """Avoid 404 spam in logs."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    # Dashboard loads even if pipeline is disabled
    return HTMLResponse(
        """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Sprint 1 Live Dashboard</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 18px; }
      .row { display: flex; gap: 18px; align-items: flex-start; }
      .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; }
      img { width: 720px; max-width: 100%; border-radius: 10px; border: 1px solid #eee; }
      pre { background: #0b1020; color: #e7e7e7; padding: 12px; border-radius: 10px; overflow: auto; width: 520px; max-width: 100%; }
      .muted { color: #666; font-size: 13px; }
      .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 10px; margin-top: 10px; }
      code { background: #f6f6f6; padding: 1px 4px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <h2>Sprint 1 — Live Webcam + VAD (FastAPI)</h2>
    <div class="muted">
      Video: <code>/video/mjpeg</code> | Live VAD stream: <code>/pipeline/stream</code> | Metrics: <code>/frame-selector/metrics</code>
    </div>

    <div class="warn" id="pipelineWarn" style="display:none;">
      <b>Pipeline is disabled.</b>
      This API is running, but ML dependencies are missing, so <code>/video/mjpeg</code> and <code>/pipeline/stream</code> will not be called.
    </div>

    <div class="row" style="margin-top:12px;">
      <div class="card">
        <h3>Live Video</h3>
        <div id="videoWrap">
          <img id="videoImg" alt="Live video stream" />
        </div>
      </div>

      <div class="card">
        <h3>Live VAD Output</h3>
        <pre id="out">{ "status": "checking_pipeline_status" }</pre>
      </div>
    </div>

    <script>
      const pre = document.getElementById("out");
      const warn = document.getElementById("pipelineWarn");
      const img = document.getElementById("videoImg");

      fetch("/status")
        .then(r => r.json())
        .then(s => {
          if (!s.pipeline_enabled) {
            warn.style.display = "block";
            pre.textContent = JSON.stringify({ status: "pipeline_disabled" }, null, 2);
            // Do NOT request /video/mjpeg or /pipeline/stream when disabled
            img.style.display = "none";
            return;
          }

          // Pipeline is enabled: now it's safe to call endpoints
          warn.style.display = "none";
          img.style.display = "block";
          img.src = "/video/mjpeg";

          const es = new EventSource("/pipeline/stream");
          es.onmessage = (e) => {
            try {
              const obj = JSON.parse(e.data);
              pre.textContent = JSON.stringify(obj, null, 2);
            } catch {
              pre.textContent = e.data;
            }
          };
          es.onerror = () => {
            pre.textContent = JSON.stringify({ error: "SSE disconnected — refresh page" }, null, 2);
            try { es.close(); } catch {}
          };
        })
        .catch(() => {
          warn.style.display = "block";
          pre.textContent = JSON.stringify({ error: "status check failed" }, null, 2);
          img.style.display = "none";
        });
    </script>
  </body>
</html>
        """
    )


# Give API router access to the runner via app.state
@app.middleware("http")
async def attach_runner(request, call_next):
    request.app.state.runner = runner
    return await call_next(request)


app.include_router(api_router)
