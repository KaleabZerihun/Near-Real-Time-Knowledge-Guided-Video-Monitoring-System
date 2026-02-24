from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, FileResponse
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

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

app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring")

repo_root = THIS_DIR.parent
img_dir = repo_root / "images"
if img_dir.exists():
    app.mount("/images", StaticFiles(directory=str(img_dir)), name="images")
else:
    print(f"[WARN] images directory not found at {img_dir}")

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
        source=0,  # which webcam
        source_id="webcam0",
        target_fps=8.0,        # how many fps
        resize_hw=(224, 224),  # resize frames for vad model input
        clip_len=16,           # how batches are formed
        stride=8,
        frame_ring_maxlen=256, # buffer size
        max_batches=8,         # queue size
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
    icon_path = img_dir / "CameraEye.png"
    if icon_path.exists():
        return FileResponse(str(icon_path), media_type="image/png")
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
    <title>Dashboard</title>
    <link rel="icon" type="image/png" href="/favicon.ico" />
    <style>
      body { font-family: Arial, sans-serif; margin: 18px; }
      .row { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
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

      canvas { width: 720px; max-width: 100%; height: 240px; border: 1px solid #eee; border-radius: 10px; }
      .event-item { padding: 10px; border-bottom: 1px solid #f0f0f0; background: #fafafa; margin-bottom: 4px; border-radius: 6px; font-size: 13px; }
      .event-item:last-child { border-bottom: none; }
      .event-time { color: #666; font-size: 11px; }
      .event-label { font-weight: 600; color: #1f77b4; }
      .event-confidence { color: #d62728; font-weight: 600; }
      .event-clip { color: #999; font-size: 11px; }
    </style>
  </head>
  <body>
    <div class="muted">
      Video: <code>/video/mjpeg</code> | Live VAD stream: <code>/pipeline/stream</code> |
      Metrics: <code>/frame-selector/metrics</code> | Overtime: <code>/pipeline/history</code>
    </div>

    <div class="warn" id="pipelineWarn" style="display:none;">
      <b>Pipeline is disabled.</b>
      This API is running, but ML dependencies are missing, so <code>/video/mjpeg</code> and <code>/pipeline/stream</code> will not be called.
    </div>

    <div class="row" style="margin-top:12px;">
      <div class="card">
        <h3>Live Video</h3>
        <div id="vadBadge" style="margin: 8px 0 10px 0; padding: 8px 10px; border: 1px solid #eee; border-radius: 10px; background: #fafafa; font-weight: 700;">
             VAD: waiting...
        </div>
        <div id="videoWrap">
          <img id="videoImg" alt="Live video stream" />
        </div>
      </div>

      <div class="card">
        <h3>Live VAD Output</h3>
        <pre id="out">{ "status": "checking_pipeline_status" }</pre>
      </div>

      <div class="card">
        <h3>Overtime Performance VAD Only (Confidence vs Time)</h3>

        <!-- haaaaaaaaaaaaaaaaaaaaaaa -->
        <div class="muted" id="avgLine" style="margin-bottom:8px;">
          Rolling average: waiting...
        </div>
        <!-- jaaaaaaaaaaaaaaaaaaaaaaaaaaaa -->

        <div class="muted" style="margin-bottom:8px;">X-axis = time, Y-axis = VAD confidence</div>
        <div class="muted" style="margin-bottom:8px;">blue = confidence vs time, green = rolling average</div>
        <canvas id="perfChart" width="720" height="240"></canvas>
      </div>

      <div class="card">
        <h3>Recent Events</h3>
        <div class="muted" style="margin-bottom: 8px;">Last 5 Alerts</div>
        <div id="eventsContainer" style="max-height: 400px; overflow-y: auto;">
          <div style="color: black; text-align: center; padding: 20px;">No events yet...</div>
        </div>
      </div>
    </div>

    <script>
      const pre = document.getElementById("out");
      const warn = document.getElementById("pipelineWarn");
      const img = document.getElementById("videoImg");
      const badge = document.getElementById("vadBadge");

      // haaaaaaaaaaaaaaaaaaaaaaa
      const avgText = document.getElementById("avgLine");
      const AVG_WINDOW = 10; // average of last 10 VAD outputs
      // jaaaaaaaaaaaaaaaaaaaaaaaaaaaa

      const toastContainer = document.createElement('div');
      toastContainer.id = 'toasts';
      document.body.appendChild(toastContainer);

      function showToast(alert) {
        const el = document.createElement('div');
        el.className = 'toast ' + (alert.severity || 'info');
        el.innerHTML = `<strong>${(alert.source || "SRC").toUpperCase()}</strong>: ${alert.message}<div class="meta">${new Date(alert.ts * 1000).toLocaleTimeString()}</div>`;
        toastContainer.appendChild(el);

        requestAnimationFrame(() => el.classList.add('visible'));

        const dismiss = () => { el.classList.remove('visible'); setTimeout(() => el.remove(), 300); };
        el.addEventListener('click', dismiss);
        setTimeout(dismiss, 6000);
      }

      // ---------------- Overtime graph (canvas) ----------------
      const canvas = document.getElementById("perfChart");
      const ctx = canvas ? canvas.getContext("2d") : null;

      // store points: {t: unixSeconds, confidence: 0..1}
      const points = [];
      const MAX_POINTS = 300;

      function pushPoint(t, confidence) {
        if (!Number.isFinite(t) || !Number.isFinite(confidence)) return;
        points.push({ t, confidence });
        while (points.length > MAX_POINTS) points.shift();
      }

      // haaaaaaaaaaaaaaaaaaaaaaa
      // Compute rolling average for the last N points (N = AVG_WINDOW)
      function rollingAvgLastN(n) {
        const k = Math.max(1, Math.min(points.length, n));
        if (points.length === 0) return null;

        let sum = 0;
        for (let i = points.length - k; i < points.length; i++) {
          sum += points[i].confidence;
        }
        return sum / k;
      }

      // Build an array of rolling-average points aligned with `points`
      function buildRollingAvgSeries(n) {
        const series = [];
        const k = Math.max(1, n);

        for (let i = 0; i < points.length; i++) {
          const start = Math.max(0, i - k + 1);
          let sum = 0;
          let count = 0;
          for (let j = start; j <= i; j++) {
            sum += points[j].confidence;
            count += 1;
          }
          series.push({ t: points[i].t, confidence: sum / count });
        }
        return series;
      }
      // jaaaaaaaaaaaaaaaaaaaaaaaaaaaa

      // Store recent logger events/alerts
      const loggerEvents = [];
      const MAX_LOGGER_EVENTS = 5;

      function pushLoggerEvent(alert) {
        if (!alert || !alert.ts || !alert.message) return;
        loggerEvents.unshift({
          id: alert.id || Math.random(),
          ts: alert.ts,
          source: alert.source || "unknown",
          severity: alert.severity || "info",
          message: alert.message,
        });
        while (loggerEvents.length > MAX_LOGGER_EVENTS) loggerEvents.pop();
        updateEventsDisplay();
      }

      function updateEventsDisplay() {
        const container = document.getElementById("eventsContainer");
        if (loggerEvents.length === 0) {
          container.innerHTML = '<div style="color: black; text-align: center; padding: 20px;">No events yet...</div>';
          return;
        }

        container.innerHTML = "";
        loggerEvents.forEach(evt => {
          const time = new Date(evt.ts * 1000).toLocaleTimeString();
          const severityColor = evt.severity === "critical" ? "#d62728" : evt.severity === "warning" ? "#ff7f0e" : "#1f77b4";
          const item = document.createElement("div");
          item.className = "event-item";
          item.innerHTML = `
            <div><span class="event-label" style="color: ${severityColor}">${evt.severity.toUpperCase()}</span> <span style="color: #666; font-size: 12px;">${evt.source}</span></div>
            <div style="margin: 4px 0;">${evt.message}</div>
            <div class="event-time">${time}</div>
          `;
          container.appendChild(item);
        });
      }

      function drawChart() {
        if (!ctx || !canvas) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const padL = 52, padR = 12, padT = 10, padB = 26;
        const w = canvas.width - padL - padR;
        const h = canvas.height - padT - padB;

        ctx.strokeStyle = "#cccccc";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, padT + h);
        ctx.lineTo(padL + w, padT + h);
        ctx.stroke();

        ctx.fillStyle = "#333333";
        ctx.font = "12px Arial";
        const yTicks = [0, 0.5, 1.0];
        yTicks.forEach(v => {
          const y = padT + (1 - v) * h;
          ctx.strokeStyle = "#eeeeee";
          ctx.beginPath();
          ctx.moveTo(padL, y);
          ctx.lineTo(padL + w, y);
          ctx.stroke();

          ctx.fillStyle = "#333333";
          ctx.fillText(v.toFixed(1), 10, y + 4);
        });

        if (points.length < 2) {
          ctx.fillStyle = "#666";
          ctx.fillText("Waiting for VAD confidence...", padL + 10, padT + 18);
          return;
        }

        const tMin = points[0].t;
        const tMax = points[points.length - 1].t;
        const tSpan = Math.max(1e-6, tMax - tMin);

        const leftLabel = new Date(tMin * 1000).toLocaleTimeString();
        const rightLabel = new Date(tMax * 1000).toLocaleTimeString();
        ctx.fillStyle = "#333333";
        ctx.fillText(leftLabel, padL, padT + h + 18);
        ctx.fillText(rightLabel, padL + w - ctx.measureText(rightLabel).width, padT + h + 18);

        // Raw confidence line
        ctx.strokeStyle = "#1f77b4";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < points.length; i++) {
          const p = points[i];
          const x = padL + ((p.t - tMin) / tSpan) * w;
          const y = padT + (1 - Math.max(0, Math.min(1, p.confidence))) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // haaaaaaaaaaaaaaaaaaaaaaa
        // Rolling average line (smoother)
        const avgSeries = buildRollingAvgSeries(AVG_WINDOW);
        ctx.strokeStyle = "#2ca02c";
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < avgSeries.length; i++) {
          const p = avgSeries[i];
          const x = padL + ((p.t - tMin) / tSpan) * w;
          const y = padT + (1 - Math.max(0, Math.min(1, p.confidence))) * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Update average text
        const avgNow = rollingAvgLastN(AVG_WINDOW);
        if (avgText) {
          if (avgNow === null) avgText.textContent = "Rolling average: waiting...";
          else avgText.textContent = `Rolling average (last ${AVG_WINDOW}): ${avgNow.toFixed(3)}`;
        }
        // jaaaaaaaaaaaaaaaaaaaaaaaaaaaa

        // latest dot (raw)
        const last = points[points.length - 1];
        const xLast = padL + ((last.t - tMin) / tSpan) * w;
        const yLast = padT + (1 - Math.max(0, Math.min(1, last.confidence))) * h;
        ctx.fillStyle = "#d62728";
        ctx.beginPath();
        ctx.arc(xLast, yLast, 4, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = "#111";
        ctx.fillText(`latest: ${last.confidence.toFixed(3)}`, padL + 10, padT + 18);
      }

      fetch("/status")
        .then(r => r.json())
        .then(async (s) => {
          if (!s.pipeline_enabled) {
            if (badge) badge.textContent = "VAD: pipeline disabled";
            if (avgText) avgText.textContent = "Rolling average: pipeline disabled";
            warn.style.display = "block";
            pre.textContent = JSON.stringify({ status: "pipeline_disabled" }, null, 2);
            img.style.display = "none";
            drawChart();
            return;
          }

          warn.style.display = "none";
          img.style.display = "block";
          img.src = "/video/mjpeg";

          // Load initial overtime points
          try {
            const histRes = await fetch("/pipeline/history?limit=300");
            const hist = await histRes.json();
            if (hist.points && Array.isArray(hist.points)) {
              hist.points.forEach(p => pushPoint(p.t, p.confidence));
              drawChart();
            } else {
              drawChart();
            }
          } catch (e) {
            drawChart();
          }

          updateEventsDisplay();

          const es = new EventSource("/pipeline/stream");

          es.onopen = () => {
            if (badge) badge.textContent = "VAD: SSE connected...";
          };

          es.onmessage = (e) => {
            try {
              const obj = JSON.parse(e.data);
              pre.textContent = JSON.stringify(obj, null, 2);

              if (badge && obj && obj.vad) {
                const label = String(obj.vad.label || "").toUpperCase();
                const conf = (typeof obj.vad.confidence === "number") ? obj.vad.confidence.toFixed(3) : "N/A";
                badge.textContent = `VAD: ${label} | confidence: ${conf}`;
              }

              // Update graph in real-time from SSE payload
              if (obj && obj.updated_at && obj.vad && typeof obj.vad.confidence === "number") {
                pushPoint(obj.updated_at, obj.vad.confidence);
                drawChart();
              }

              if (obj.alerts && obj.alerts.length) {
                obj.alerts.forEach(a => {
                  showToast(a);
                  pushLoggerEvent(a);
                });
              }
            } catch {
              pre.textContent = e.data;
            }
          };

          es.onerror = () => {
            if (badge) badge.textContent = "VAD: SSE disconnected — refresh page";
            if (avgText) avgText.textContent = "Rolling average: SSE disconnected";
            pre.textContent = JSON.stringify({ error: "SSE disconnected — refresh page" }, null, 2);
            try { es.close(); } catch {}
          };
        })
        .catch(() => {
          warn.style.display = "block";
          pre.textContent = JSON.stringify({ error: "status check failed" }, null, 2);
          img.style.display = "none";
          if (avgText) avgText.textContent = "Rolling average: status check failed";
          drawChart();
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