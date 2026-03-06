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


# Give API router access to the runner via app.state
@app.middleware("http")
async def attach_runner(request, call_next):
    request.app.state.runner = runner
    return await call_next(request)


app.include_router(api_router)