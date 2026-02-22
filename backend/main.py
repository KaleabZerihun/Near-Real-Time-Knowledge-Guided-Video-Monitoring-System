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

app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring")

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

    try:
        runner = PipelineRunner(cfg=cfg, thesis_root=str(thesis_root))
        runner.start()
    except Exception as e:
        print(f"[WARN] PipelineRunner failed to start (camera unavailable?): {e}")
        runner = None


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
    <title>Dashboard</title>
    <style>
      * { box-sizing: border-box; }
      body { font-family: Arial, sans-serif; margin: 18px; background: #f8f9fa; color: #222; }
      h2 { margin: 0 0 4px 0; font-size: 15px; color: #444; text-transform: uppercase; letter-spacing: .04em; }
      h3 { margin: 0 0 10px 0; font-size: 14px; color: #444; }
      .row { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; margin-top: 14px; }
      .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 14px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
      img { width: 720px; max-width: 100%; border-radius: 10px; border: 1px solid #eee; }
      pre { background: #0b1020; color: #e7e7e7; padding: 12px; border-radius: 10px; overflow: auto; width: 520px; max-width: 100%; font-size: 12px; }
      .muted { color: #888; font-size: 12px; }
      .warn { background: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 10px; margin-top: 10px; }
      code { background: #f0f0f0; padding: 1px 5px; border-radius: 5px; font-size: 12px; }
      canvas { width: 720px; max-width: 100%; height: 240px; border: 1px solid #eee; border-radius: 10px; }

      /* ---- Metrics grid ---- */
      .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; min-width: 340px; }
      .metric-cell { background: #f4f6fb; border-radius: 8px; padding: 10px 12px; }
      .metric-label { font-size: 11px; color: #888; margin-bottom: 3px; text-transform: uppercase; letter-spacing: .04em; }
      .metric-value { font-size: 22px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
      .metric-sub { font-size: 11px; color: #aaa; margin-top: 2px; }
      .metric-cell.accent-green .metric-value { color: #1e7e34; }
      .metric-cell.accent-red   .metric-value { color: #c0392b; }
      .metric-cell.accent-blue  .metric-value { color: #1565c0; }
      #metricsStatus { font-size: 12px; color: #999; margin-top: 8px; }

      /* ---- Toasts ---- */
      #toasts { position: fixed; right: 18px; top: 18px; display: flex; flex-direction: column; gap: 8px; z-index: 9999; pointer-events: none; }
      .toast { min-width: 260px; max-width: 360px; pointer-events: auto; background: #fff; color: #111; padding: 10px 12px; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.18); transform: translateY(-8px) scale(0.98); opacity: 0; transition: all 0.25s ease; font-size: 14px; }
      .toast.visible { transform: translateY(0) scale(1); opacity: 1; }
      .toast.info { border-left: 4px solid #888; }
      .toast.warning { border-left: 4px solid orange; }
      .toast.critical { border-left: 4px solid red; }
    </style>
  </head>
  <body>
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
      <div>
        <h2 style="font-size:18px;color:#1a1a2e;margin:0;">Near Real-Time Video Monitoring</h2>
        <div class="muted" style="margin-top:3px;">
          <code>/video/mjpeg</code> &nbsp;|&nbsp; <code>/pipeline/stream</code> &nbsp;|&nbsp;
          <code>/metrics/summary</code> &nbsp;|&nbsp; <code>/docs</code>
        </div>
      </div>
      <div id="pipelineBadge" style="padding:5px 12px;border-radius:20px;font-size:12px;font-weight:700;background:#eee;color:#555;">
        checking...
      </div>
    </div>

    <div class="warn" id="pipelineWarn" style="display:none;margin-top:10px;">
      <b>Pipeline is disabled.</b>
      The API is running, but the webcam is unavailable so live video and VAD are off.
    </div>

    <!-- Row 1: Video + VAD JSON -->
    <div class="row">
      <div class="card">
        <h3>Live Video</h3>
        <div id="vadBadge" style="margin: 6px 0 10px 0; padding: 8px 12px; border: 1px solid #eee; border-radius: 8px; background: #f4f6fb; font-weight: 700; font-size:14px;">
          VAD: waiting...
        </div>
        <div id="videoWrap">
          <img id="videoImg" alt="Live video stream" style="display:none;" />
        </div>
      </div>

      <div class="card">
        <h3>Live VAD Output</h3>
        <pre id="out">{ "status": "checking_pipeline_status" }</pre>
      </div>
    </div>

    <!-- Row 2: Confidence chart + Performance Metrics -->
    <div class="row">
      <div class="card">
        <h3>Confidence Over Time</h3>
        <div class="muted" style="margin-bottom:8px;">X = time &nbsp;|&nbsp; Y = VAD confidence (0–1)</div>
        <canvas id="perfChart" width="720" height="240"></canvas>
      </div>

      <div class="card">
        <h3>Performance Metrics <span class="muted" style="font-weight:400;">(last 300 clips)</span></h3>
        <div class="metrics-grid">
          <!-- Timing -->
          <div class="metric-cell accent-blue">
            <div class="metric-label">VAD Inference</div>
            <div class="metric-value" id="m-vad-mean">—</div>
            <div class="metric-sub" id="m-vad-p95">p95: — ms</div>
          </div>
          <div class="metric-cell accent-blue">
            <div class="metric-label">E2E Latency</div>
            <div class="metric-value" id="m-e2e-mean">—</div>
            <div class="metric-sub" id="m-e2e-p95">p95: — ms</div>
          </div>
          <div class="metric-cell">
            <div class="metric-label">Throughput</div>
            <div class="metric-value" id="m-throughput">—</div>
            <div class="metric-sub">clips / sec</div>
          </div>
          <!-- Detection -->
          <div class="metric-cell accent-red">
            <div class="metric-label">Anomaly Rate</div>
            <div class="metric-value" id="m-anomaly-rate">—</div>
            <div class="metric-sub" id="m-clips">0 clips</div>
          </div>
          <div class="metric-cell">
            <div class="metric-label">Mean Confidence</div>
            <div class="metric-value" id="m-mean-conf">—</div>
            <div class="metric-sub">VAD score avg</div>
          </div>
          <div class="metric-cell accent-green">
            <div class="metric-label">Capture FPS</div>
            <div class="metric-value" id="m-capture-fps">—</div>
            <div class="metric-sub" id="m-sel-fps">selected: —</div>
          </div>
          <!-- Drop counters -->
          <div class="metric-cell">
            <div class="metric-label">Dropped Frames</div>
            <div class="metric-value" id="m-dropped-frames">—</div>
            <div class="metric-sub">cumulative</div>
          </div>
          <div class="metric-cell">
            <div class="metric-label">Dropped Batches</div>
            <div class="metric-value" id="m-dropped-batches">—</div>
            <div class="metric-sub">cumulative</div>
          </div>
          <div class="metric-cell">
            <div class="metric-label">VAD Max ms</div>
            <div class="metric-value" id="m-vad-max">—</div>
            <div class="metric-sub">worst case</div>
          </div>
        </div>
        <div id="metricsStatus">Waiting for pipeline data...</div>
      </div>
    </div>

    <script>
      const pre    = document.getElementById("out");
      const warn   = document.getElementById("pipelineWarn");
      const img    = document.getElementById("videoImg");
      const badge  = document.getElementById("vadBadge");
      const pbadge = document.getElementById("pipelineBadge");

      // ── Toasts ────────────────────────────────────────────────
      const toastContainer = document.createElement('div');
      toastContainer.id = 'toasts';
      document.body.appendChild(toastContainer);

      function showToast(alert) {
        const el = document.createElement('div');
        el.className = 'toast ' + (alert.severity || 'info');
        el.innerHTML = `<strong>${(alert.source || "SRC").toUpperCase()}</strong>: ${alert.message}
          <div class="muted">${new Date(alert.ts * 1000).toLocaleTimeString()}</div>`;
        toastContainer.appendChild(el);
        requestAnimationFrame(() => el.classList.add('visible'));
        const dismiss = () => { el.classList.remove('visible'); setTimeout(() => el.remove(), 300); };
        el.addEventListener('click', dismiss);
        setTimeout(dismiss, 6000);
      }

      // ── Confidence chart ──────────────────────────────────────
      const canvas = document.getElementById("perfChart");
      const ctx    = canvas ? canvas.getContext("2d") : null;
      const points = [];
      const MAX_POINTS = 300;

      function pushPoint(t, confidence) {
        if (!Number.isFinite(t) || !Number.isFinite(confidence)) return;
        points.push({ t, confidence });
        while (points.length > MAX_POINTS) points.shift();
      }

      function drawChart() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const padL = 52, padR = 12, padT = 10, padB = 26;
        const w = canvas.width - padL - padR;
        const h = canvas.height - padT - padB;

        ctx.strokeStyle = "#cccccc"; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + h); ctx.lineTo(padL + w, padT + h);
        ctx.stroke();

        ctx.font = "12px Arial";
        [0, 0.5, 1.0].forEach(v => {
          const y = padT + (1 - v) * h;
          ctx.strokeStyle = "#eeeeee"; ctx.beginPath();
          ctx.moveTo(padL, y); ctx.lineTo(padL + w, y); ctx.stroke();
          ctx.fillStyle = "#333"; ctx.fillText(v.toFixed(1), 10, y + 4);
        });

        if (points.length < 2) {
          ctx.fillStyle = "#aaa";
          ctx.fillText("Waiting for VAD confidence...", padL + 10, padT + 18);
          return;
        }

        const tMin = points[0].t, tMax = points[points.length - 1].t;
        const tSpan = Math.max(1e-6, tMax - tMin);
        const leftLabel  = new Date(tMin * 1000).toLocaleTimeString();
        const rightLabel = new Date(tMax * 1000).toLocaleTimeString();
        ctx.fillStyle = "#333";
        ctx.fillText(leftLabel, padL, padT + h + 18);
        ctx.fillText(rightLabel, padL + w - ctx.measureText(rightLabel).width, padT + h + 18);

        ctx.strokeStyle = "#1f77b4"; ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
          const x = padL + ((p.t - tMin) / tSpan) * w;
          const y = padT + (1 - Math.max(0, Math.min(1, p.confidence))) * h;
          i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        const last  = points[points.length - 1];
        const xLast = padL + ((last.t - tMin) / tSpan) * w;
        const yLast = padT + (1 - Math.max(0, Math.min(1, last.confidence))) * h;
        ctx.fillStyle = "#d62728"; ctx.beginPath();
        ctx.arc(xLast, yLast, 4, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#111";
        ctx.fillText(`latest: ${last.confidence.toFixed(3)}`, padL + 10, padT + 18);
      }

      // ── Performance metrics panel ─────────────────────────────
      function setText(id, val) {
        const el = document.getElementById(id);
        if (el) el.textContent = val;
      }

      function updateMetrics() {
        fetch("/metrics/summary?last_n=300")
          .then(r => r.ok ? r.json() : null)
          .then(d => {
            if (!d || d.status === "no_data_yet" || d.error) {
              setText("metricsStatus", d?.status === "no_data_yet"
                ? "No clips processed yet — start the pipeline with a webcam."
                : (d?.error || "Metrics unavailable"));
              return;
            }

            const ms = v => v != null ? v.toFixed(1) + " ms" : "—";
            const pct = v => v != null ? (v * 100).toFixed(1) + "%" : "—";
            const fps = v => v != null ? v.toFixed(1) : "—";
            const num = v => v != null ? String(v) : "—";

            setText("m-vad-mean",        ms(d.mean_vad_ms));
            setText("m-vad-p95",         "p95: " + ms(d.p95_vad_ms));
            setText("m-vad-max",         ms(d.max_vad_ms));
            setText("m-e2e-mean",        ms(d.mean_e2e_ms));
            setText("m-e2e-p95",         "p95: " + ms(d.p95_e2e_ms));
            setText("m-throughput",      d.throughput_clips_per_sec != null
                                            ? d.throughput_clips_per_sec.toFixed(2) : "—");
            setText("m-anomaly-rate",    pct(d.anomaly_rate));
            setText("m-clips",           (d.window_clips || 0) + " clips");
            setText("m-mean-conf",       d.mean_confidence != null
                                            ? d.mean_confidence.toFixed(3) : "—");
            setText("m-capture-fps",     fps(d.capture_fps));
            setText("m-sel-fps",         "selected: " + fps(d.selected_fps));
            setText("m-dropped-frames",  num(d.dropped_frames_total));
            setText("m-dropped-batches", num(d.dropped_batches_total));

            const since = d.since ? new Date(d.since * 1000).toLocaleTimeString() : "—";
            setText("metricsStatus", `Updated ${new Date().toLocaleTimeString()} · window since ${since}`);
          })
          .catch(() => setText("metricsStatus", "Failed to fetch metrics"));
      }

      // ── Pipeline status + SSE ─────────────────────────────────
      fetch("/status")
        .then(r => r.json())
        .then(async (s) => {
          if (!s.pipeline_enabled) {
            pbadge.textContent  = "Pipeline OFF";
            pbadge.style.background = "#fde8e8";
            pbadge.style.color      = "#c0392b";
            badge.textContent = "VAD: pipeline disabled";
            warn.style.display = "block";
            pre.textContent = JSON.stringify({ status: "pipeline_disabled" }, null, 2);
            drawChart();
            setText("metricsStatus", "Pipeline is disabled — no live metrics available.");
            return;
          }

          pbadge.textContent       = "Pipeline ON";
          pbadge.style.background  = "#e8f5e9";
          pbadge.style.color       = "#1e7e34";
          warn.style.display = "none";
          img.style.display  = "block";
          img.src = "/video/mjpeg";

          // Seed confidence chart from history
          try {
            const hist = await fetch("/pipeline/history?limit=300").then(r => r.json());
            if (hist.points) hist.points.forEach(p => pushPoint(p.t, p.confidence));
            drawChart();
          } catch { drawChart(); }

          // Start polling metrics every 3 s
          updateMetrics();
          setInterval(updateMetrics, 3000);

          // SSE for live VAD
          const es = new EventSource("/pipeline/stream");
          es.onopen = () => { badge.textContent = "VAD: SSE connected..."; };
          es.onmessage = (e) => {
            try {
              const obj = JSON.parse(e.data);
              pre.textContent = JSON.stringify(obj, null, 2);
              if (obj?.vad) {
                const lbl  = String(obj.vad.label || "").toUpperCase();
                const conf = typeof obj.vad.confidence === "number"
                              ? obj.vad.confidence.toFixed(3) : "N/A";
                badge.textContent = `VAD: ${lbl} | confidence: ${conf}`;
              }
              if (obj?.updated_at && typeof obj.vad?.confidence === "number") {
                pushPoint(obj.updated_at, obj.vad.confidence);
                drawChart();
              }
              if (obj?.alerts?.length) obj.alerts.forEach(showToast);
            } catch { pre.textContent = e.data; }
          };
          es.onerror = () => {
            badge.textContent = "VAD: SSE disconnected — refresh page";
            pre.textContent = JSON.stringify({ error: "SSE disconnected" }, null, 2);
            try { es.close(); } catch {}
          };
        })
        .catch(() => {
          warn.style.display = "block";
          pre.textContent = JSON.stringify({ error: "status check failed" }, null, 2);
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
