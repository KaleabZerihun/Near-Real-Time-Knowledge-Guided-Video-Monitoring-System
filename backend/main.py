from __future__ import annotations

import os
import sys
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Make sure imports work when running from backend/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from src.frame_selector.config import FrameSelectorConfig
from src.pipeline.runner import PipelineRunner
from src.api.routes import router as api_router

app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring (Sprint 1)")

runner: PipelineRunner | None = None


@app.on_event("startup")
def startup() -> None:
    global runner

    # repo root is one level above backend/
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    thesis_root = os.path.join(repo_root, "THESIS")

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

    runner = PipelineRunner(cfg=cfg, thesis_root=thesis_root)
    runner.start()


@app.on_event("shutdown")
def shutdown() -> None:
    global runner
    if runner:
        runner.stop()
        runner = None
    os._exit(0)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    # Simple “Sprint 1” dashboard (video + live JSON)
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
    </style>
  </head>
  <body>
    <h2>Sprint 1 — Live Webcam + VAD (FastAPI)</h2>
    <div class="muted">
      Video: <code>/video/mjpeg</code> | Live VAD stream: <code>/pipeline/stream</code> | Metrics: <code>/frame-selector/metrics</code>
    </div>

    <div class="row" style="margin-top:12px;">
      <div class="card">
        <h3>Live Video</h3>
        <img src="/video/mjpeg" />
      </div>

      <div class="card">
        <h3>Live VAD Output</h3>
        <pre id="out">{ "status": "waiting_for_first_result" }</pre>
      </div>
    </div>

    <script>
      const pre = document.getElementById("out");
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
        pre.textContent = "{ \\"error\\": \\"SSE disconnected — refresh page\\" }";
      };
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
