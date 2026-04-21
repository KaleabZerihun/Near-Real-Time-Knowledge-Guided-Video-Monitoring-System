from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
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

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_resize_hw(value: str) -> tuple[int, int]:
    try:
        parts = [int(x.strip()) for x in value.split(",") if x.strip()]
        if len(parts) == 2:
            return (parts[0], parts[1])
    except ValueError:
        pass
    return (224, 224)


app = FastAPI(title="Near Real-Time Knowledge-Guided Video Monitoring")

# Configure CORS for Amplify frontend and other allowed origins
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
allowed_origins = [origin.strip() for origin in allowed_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo_root = THIS_DIR.parent
img_dir = repo_root / "images"
if img_dir.exists():
    app.mount("/images", StaticFiles(directory=str(img_dir)), name="images")
else:
    print(f"[WARN] images directory not found at {img_dir}")

runner = None  # PipelineRunner | None (kept untyped for when PipelineRunner is None)
video_source_status = os.getenv("VIDEO_SOURCE", "0")


@app.on_event("startup")
def startup() -> None:
    global runner, video_source_status

    # Repo root is one level above backend/
    repo_root = THIS_DIR.parent
    rtvad_root = repo_root / "RT-VAD"

    # Load env vars from repo root (.env)
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    # Stable default DB location: backend/data/app.db
    default_db_path = (THIS_DIR / "data" / "app.db").resolve()
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{default_db_path.as_posix()}")

    # Ensure DB + tables exist when using sqlite
    init_db(database_url)

    if not _env_bool("RUN_PIPELINE", True):
        print("[INFO] RUN_PIPELINE=false; skipping PipelineRunner startup.")
        runner = None
        return

    if PipelineRunner is None:
        print("[WARN] RUN_PIPELINE=true but PipelineRunner dependencies are unavailable. Pipeline remains disabled.")
        runner = None
        return

    # Video source can be a webcam index or a remote source path / file path.
    video_source = os.getenv("VIDEO_SOURCE", "0")
    video_source_status = video_source
    try:
        source = int(video_source)
    except ValueError:
        source = video_source

    cfg = FrameSelectorConfig(
        source=source,
        source_id=os.getenv("VIDEO_SOURCE_ID", "webcam0"),
        select_every=_env_int("SELECT_EVERY", 2),
        resize_hw=_parse_resize_hw(os.getenv("RESIZE_HW", "224,224")),
        clip_len=_env_int("CLIP_LEN", 16),
        stride=_env_int("STRIDE", 8),
        frame_ring_maxlen=_env_int("FRAME_RING_MAXLEN", 256),
        max_batches=_env_int("MAX_BATCHES", 8),
    )

    try:
        runner = PipelineRunner(cfg=cfg, rtvad_root=str(rtvad_root))
        runner.start()
    except Exception as e:
        print(f"[WARN] PipelineRunner failed to start: {e}")
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
    return {
        "pipeline_enabled": runner is not None,
        "video_source": video_source_status,
    }


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