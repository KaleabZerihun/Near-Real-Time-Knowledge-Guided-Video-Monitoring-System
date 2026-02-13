from __future__ import annotations
import os
import sys
import threading
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

# Ensure `src.*` imports work when running from repo root or backend/
THIS_DIR = Path(__file__).resolve().parent  
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from src.api.routes import router as api_router
from src.db.init_db import init_db
from src.db.retention import run_retention_once
from src.frame_selector.config import FrameSelectorConfig

# Try to import PipelineRunner; if ML deps are missing, keep API running
try:
    from src.pipeline.runner import PipelineRunner
except ModuleNotFoundError as e:
    PipelineRunner = None  
    print(f"[WARN] PipelineRunner disabled (missing dependency): {e}")
app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring")

runner = None  

_stop_retention = threading.Event()
_retention_thread: threading.Thread | None = None

def _retention_loop() -> None:
    while not _stop_retention.is_set():
        try:
            result = run_retention_once(
                keep_events_days=30,  # archive >30d to events_archive
                keep_orphans_days=7,  # delete unmatched inputs after 7d
                do_vacuum=False,
            )
            print("[retention]", result)
        except Exception as e:
            print("[retention] error:", e)

        _stop_retention.wait(30 * 60)  # every 30 minutes

@app.on_event("startup")
def startup() -> None:
    # Startup order (important):
    #  1) load .env
    #  2) init DB/schema
    #  3) start pipeline runner (if available)
    #  4) start retention thread
    global runner, _retention_thread

    repo_root = THIS_DIR.parent
    thesis_root = repo_root / "THESIS"

    # Load env vars from repo root (.env)
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    # Ensure DB + tables exist 
    try:
        init_db()
    except TypeError:
        default_db_path = (THIS_DIR / "data" / "app.db").resolve()
        database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path.as_posix()}")
        init_db(database_url)

    # Start pipeline runner 
    if PipelineRunner is None:
        print("[WARN] Skipping PipelineRunner startup; API will run without pipeline.")
        runner = None
    else:
        cfg = FrameSelectorConfig(
            source=0,  # webcam index
            source_id="webcam0",
            target_fps=8.0,
            resize_hw=(224, 224),
            clip_len=16,
            stride=8,
            frame_ring_maxlen=256,
            max_batches=8,
        )
        runner = PipelineRunner(cfg=cfg, thesis_root=str(thesis_root))
        runner.start()

    # Start retention background thread
    _stop_retention.clear()
    _retention_thread = threading.Thread(target=_retention_loop, daemon=True)
    _retention_thread.start()

@app.on_event("shutdown")
def shutdown() -> None:
    global runner

    # Stop retention loop
    _stop_retention.set()

    # Stop pipeline runner
    if runner:
        runner.stop()
        runner = None

@app.get("/status")
def status():
    # Simple status endpoint so the dashboard can decide whether to call pipeline endpoints.
    return {"pipeline_enabled": runner is not None}

@app.get("/favicon.ico")
def favicon():
    # Avoid 404 spam in logs.
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

      #toasts { position: fixed; right: 18px; top: 18px; display: flex; flex-direction: column; gap: 8px; z-index: 9999; pointer-events: none; }
      .toast { min-width: 260px; max-width: 360px; pointer-events: auto; background: #fff; color: #111; padding: 10px 12px; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); transform: translateY(-8px) scale(0.98); opacity: 0; transition: all 0.25s ease; font-size: 14px; }
      .toast.visible { transform: translateY(0) scale(1); opacity: 1; }
      .toast.info { border-left: 4px solid grey; }
      .toast.warning { border-left: 4px solid orange; }
      .toast.critical { border-left: 4px solid red; }
      .toast.meta { font-size: 11px; color: dark grey; margin-top: 6px; }
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
            img.style.display = "none";
            return;
          }

          warn.style.display = "none";
          img.style.display = "block";
          img.src = "/video/mjpeg";

          const toastContainer = document.createElement('div');
          toastContainer.id = 'toasts';
          document.body.appendChild(toastContainer);

          function showToast(alert) {
            const el = document.createElement('div');
            el.className = 'toast ' + (alert.severity || 'info');
            el.innerHTML = `<strong>${alert.source.toUpperCase()}</strong>: ${alert.message}<div class="meta">${new Date(alert.ts * 1000).toLocaleTimeString()}</div>`;
            toastContainer.appendChild(el);

            requestAnimationFrame(() => el.classList.add('visible'));

            const dismiss = () => { el.classList.remove('visible'); setTimeout(() => el.remove(), 300); };
            el.addEventListener('click', dismiss);
            setTimeout(dismiss, 6000);
          }

          const es = new EventSource("/pipeline/stream");
          es.onmessage = (e) => {
            try {
              const obj = JSON.parse(e.data);
              pre.textContent = JSON.stringify(obj, null, 2);
              if (obj.alerts && obj.alerts.length) {
                obj.alerts.forEach(a => showToast(a));
              }
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