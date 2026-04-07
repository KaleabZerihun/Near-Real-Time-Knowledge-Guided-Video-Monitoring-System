"""
Run:
    streamlit run flashback_eval.py
"""

import json
import time
import statistics
import warnings
import tempfile
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from scipy.ndimage import gaussian_filter1d
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from textencoder import (
    load_imagebind,
    build_custom_anomaly_embedding,
)

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())  # OPT: use all CPU cores

# CONFIG
DEFAULT_VIDEO_PATH = "./video.mp4"
DEFAULT_EMBEDDINGS_DIR = "./embeddings/stage1"
DEFAULT_CUSTOM_MEMORY_PATH = "./custom_anomaly_memory.json"
CHECKPOINT_PATH = "./.checkpoints/imagebind_huge.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
T_SEGMENT = 1.0
T_OVERLAP = 0.0
T_SAMPLE = 2
INPUT_RES = (224, 224)
THRESHOLD = 0.5
GAUSS_SIGMA = 0.5
GAUSS_WIDTH = 100
CUSTOM_ANOMALY_CATEGORY = "user_defined_anomaly"

# ---------------- UI STYLE ---------------- #

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #e8eaf6);
    color: #222;
}
.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #1a237e;
    margin-bottom: 0;
}
.sub-caption {
    font-size: 18px;
    color: #3949ab;
    margin-bottom: 30px;
}
div.stButton > button:first-child {
    background-color: #1565c0;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}
div.stButton > button:hover {
    background-color: #0d47a1;
    color: white;
}
.metric-card {
    background-color: #ffffffcc;
    border-radius: 12px;
    padding: 10px 20px;
    margin: 10px 0;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Flashback - Video Anomaly Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-caption'>Zero-Shot Real-Time Detection using ImageBind & Pseudo-Scene Memory</div>", unsafe_allow_html=True)

# ---------------- SESSION STATE ---------------- #

for key, default in [
    ("running", False),
    ("last_score", 0.0),
    ("last_caption", "N/A"),
    ("last_category", "N/A"),
    ("last_frame", None),
    ("frame_scores", []),
    ("segment_summaries", []),
    ("webcam_buffer", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def start_detection():
    st.session_state.running = True
    st.session_state.webcam_buffer = []


def stop_detection():
    st.session_state.running = False


# ---------------- CUSTOM MEMORY ---------------- #

def ensure_custom_memory_file(path: str):
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"custom_anomalies": []}, f, ensure_ascii=False, indent=2)


def load_custom_memory(path: str):
    ensure_custom_memory_file(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("custom_anomalies", [])


def save_custom_memory(path: str, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"custom_anomalies": data}, f, ensure_ascii=False, indent=2)


@st.cache_resource
def load_model():
    model = imagebind_model.imagebind_huge(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    # OPT: half() only helps on CUDA; skip on CPU
    if DEVICE == "cuda":
        model.half()
    return model


@st.cache_resource
def load_text_encoder():
    device = torch.device(DEVICE)
    return load_imagebind(device)


# OPT: cache combined memory on device — recomputed only when custom memory file changes
@st.cache_resource(show_spinner="Loading memory...")
def load_combined_memory(embeddings_dir: str, custom_memory_path: str, _cache_key: float):
    """
    _cache_key: pass mtime of custom_memory_path so cache invalidates when file changes.
    """
    ZN = torch.tensor(
        np.load(Path(embeddings_dir) / "embeddings_normal.npy"),
        dtype=torch.float32
    )
    ZA = torch.tensor(
        np.load(Path(embeddings_dir) / "embeddings_anomalous.npy"),
        dtype=torch.float32
    )
    Z = torch.cat([ZN, ZA], dim=0)
    Y = torch.zeros(len(Z), dtype=torch.float32)
    Y[len(ZN):] = 1.0

    with open(Path(embeddings_dir) / "memory_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    caps = meta["captions_normal"] + meta["captions_anomalous"]
    cats = meta["categories_normal"] + meta["categories_anomalous"]
    NN, NA = len(ZN), len(ZA)

    # merge custom memory
    items = load_custom_memory(custom_memory_path)
    NC = len(items)
    if items:
        embs = [np.array(x["embedding"], dtype=np.float32) for x in items]
        Zc = torch.tensor(np.stack(embs), dtype=torch.float32)
        Yc = torch.ones(len(Zc), dtype=torch.float32)
        Z = torch.cat([Z, Zc], dim=0)
        Y = torch.cat([Y, Yc], dim=0)
        caps += [x["text"] for x in items]
        cats += [CUSTOM_ANOMALY_CATEGORY] * NC

    # OPT: move to device once; normalize once
    Z = F.normalize(Z, dim=-1).to(DEVICE)
    Y = Y.to(DEVICE)

    return Z, Y, caps, cats, NN, NA, NC


def get_custom_memory_mtime() -> float:
    p = Path(DEFAULT_CUSTOM_MEMORY_PATH)
    return p.stat().st_mtime if p.exists() else 0.0


def add_custom_anomaly(text: str, path: str):
    text = text.strip()
    if text == "":
        return False, "Event text is empty."

    items = load_custom_memory(path)
    if any(x["text"].strip().lower() == text.lower() for x in items):
        return False, "This custom anomaly already exists."

    model, mod_type, data = load_text_encoder()
    emb, _ = build_custom_anomaly_embedding(
        text=text,
        model=model,
        ModalityType=mod_type,
        imagebind_data=data,
        device=torch.device(DEVICE),
        alpha=0.95,
        apply_prompt_template=False,
    )

    items.append({
        "id": str(uuid.uuid4()),
        "text": text,
        "category": CUSTOM_ANOMALY_CATEGORY,
        "label": 1,
        "embedding": emb.tolist(),
        "created_at": time.time(),
    })

    save_custom_memory(path, items)
    return True, "Custom anomaly added."


def delete_event(event_id: str, path: str):
    items = load_custom_memory(path)
    new_items = [x for x in items if x["id"] != event_id]
    save_custom_memory(path, new_items)


# ---------------- VIDEO UTILS ---------------- #

def sample_frames(cap, start_frame: int, end_frame: int, n: int = T_SAMPLE):
    """
    OPT: sequential read instead of repeated cap.set() seeks.
    Only seeks once to start_frame, then reads forward.
    """
    total = end_frame - start_frame
    if total <= 0:
        return None

    indices = set(np.linspace(0, total - 1, n, dtype=int))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            return None
        if i in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return np.stack(frames) if frames else None


def preprocess(frames: np.ndarray):
    resized = np.stack([cv2.resize(f, INPUT_RES) for f in frames])
    t = torch.from_numpy(resized).float() / 255.0
    t = t.permute(0, 3, 1, 2).unsqueeze(0)
    if DEVICE == "cuda":
        t = t.half()
    return t


@torch.inference_mode()
def encode_segment(model, frames):
    emb = model({ModalityType.VISION: frames.to(DEVICE)})[ModalityType.VISION]
    return F.normalize(emb, dim=-1)


def retrieve(vs, Z, Y, caps, cats):
    # OPT: Z and Y are already on DEVICE and normalized — no repeated .to() or normalize here
    vs = vs.to(DEVICE).float()

    sim = torch.matmul(vs, Z.T)
    vals, idx = torch.topk(sim, min(TOP_K, Z.shape[0]), dim=-1)

    ws = F.softmax(vals, dim=-1)
    score = torch.sum(ws[0] * Y[idx[0]]).item()

    best = idx[0][0].item()
    return score, caps[best], cats[best]


def smooth_scores(scores: list, total_frames: int, ranges: list) -> np.ndarray:
    s_sum = np.zeros(total_frames, dtype=np.float64)
    count = np.zeros(total_frames, dtype=np.float64)

    for A_s, (fs, fe) in zip(scores, ranges):
        s_sum[fs:fe] += A_s
        count[fs:fe] += 1.0

    count = np.where(count == 0, 1.0, count)
    pt = s_sum / count
    truncate = (GAUSS_WIDTH / 2.0) / GAUSS_SIGMA
    return gaussian_filter1d(pt, sigma=GAUSS_SIGMA, truncate=truncate).astype(np.float32)


# ---------------- UI ---------------- #

st.sidebar.markdown("## Settings")
research_mode = st.sidebar.checkbox("Research Mode", False)

st.markdown("### Custom Anomaly Memory")

event_text = st.text_input(
    "Add custom anomalous event",
    placeholder="e.g. person climbing scaffold without safety harness"
)

c1, c2 = st.columns(2)

with c1:
    if st.button("Add Custom Event"):
        ok, msg = add_custom_anomaly(event_text, DEFAULT_CUSTOM_MEMORY_PATH)
        if ok:
            st.session_state["msg"] = "New event added and encoded. Ready to start detection."
            # OPT: clear memory cache so next run picks up new entry
            load_combined_memory.clear()
            st.rerun()
        else:
            st.warning(msg)

with c2:
    if st.button("Reload Memory"):
        load_combined_memory.clear()
        st.session_state["msg"] = "Custom memory reloaded."
        st.rerun()

if "msg" in st.session_state:
    st.success(st.session_state["msg"])
    del st.session_state["msg"]

items = load_custom_memory(DEFAULT_CUSTOM_MEMORY_PATH)

if items:
    st.write("#### Active Custom Events")
    for item in items:
        r1, r2 = st.columns([6, 1])
        r1.write(item["text"])
        if r2.button("Delete", key=item["id"]):
            delete_event(item["id"], DEFAULT_CUSTOM_MEMORY_PATH)
            load_combined_memory.clear()
            st.session_state["msg"] = "Event deleted."
            st.rerun()
else:
    st.info("No custom events.")

# ---------------- VIDEO SOURCE ---------------- #

st.markdown("### Video Source")

source = st.radio("Choose input", ["Webcam", "Video file"], horizontal=True)

video_path = None
uploaded = None

if source == "Video file":
    video_path = st.text_input("Video path", DEFAULT_VIDEO_PATH)
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

col1, col2 = st.columns(2)
col1.button("Start Detection", on_click=start_detection)
col2.button("Stop Detection", on_click=stop_detection)

# ---------------- DETECTION ---------------- #

if st.session_state.running:
    model = load_model()

    # OPT: single cached load — memory stays on device across segments
    mtime = get_custom_memory_mtime()
    Z, Y, caps, cats, NN, NA, NC = load_combined_memory(
        DEFAULT_EMBEDDINGS_DIR, DEFAULT_CUSTOM_MEMORY_PATH, mtime
    )

    st.success(f"Memory loaded — {NN} normal + {NA} anomalous | custom {NC}")

    tmp_path = None

    if source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if uploaded:
            suffix = Path(uploaded.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            tmp.close()
            tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp.name)
        else:
            cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open video source. Check path or webcam permissions.")
        st.session_state.running = False
        st.stop()

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source != "Webcam" else 0
    frames_per_seg = max(1, int(round(fps_vid * T_SEGMENT)))
    frames_per_stride = max(1, int(round(fps_vid * (T_SEGMENT - T_OVERLAP))))

    frame_window = st.image(np.zeros((720, 1280, 3), dtype=np.uint8), use_container_width=True)

    # OPT: single placeholder for metric card — updated in-place, no new elements per segment
    metric_placeholder = st.empty()

    st.info("Press **Stop Detection** to end.")

    prog = st.progress(0) if total_frames > 0 else None

    segment_scores = []
    segment_ranges = []
    seg_summaries = []
    timings = []
    seg_idx = 0
    f_start = 0
    start_time = time.time()

    while st.session_state.running:
        if source != "Webcam":
            f_end = min(f_start + frames_per_seg, total_frames)
            if f_start >= total_frames:
                st.warning("Reached end of video.")
                break

            frames = sample_frames(cap, f_start, f_end, n=T_SAMPLE)
            if frames is None:
                f_start += frames_per_stride
                continue

            mid_frame = frames[len(frames) // 2]
            display = cv2.resize(mid_frame, (1280, 720))
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("Frame grab failed.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.webcam_buffer.append(rgb)

            if len(st.session_state.webcam_buffer) < T_SAMPLE:
                continue

            frames = np.stack(st.session_state.webcam_buffer[-T_SAMPLE:])
            st.session_state.webcam_buffer = []
            display = cv2.resize(rgb, (1280, 720))
            f_end = f_start + T_SAMPLE

        tensor = preprocess(frames)

        t0 = time.time()
        vs = encode_segment(model, tensor)
        t1 = time.time()
        timings.append((t1 - t0) * 1000)

        score, cap_text, cat = retrieve(vs, Z, Y, caps, cats)
        caption = cap_text.replace("Normal:", "").replace("Anomalous:", "").strip()

        # ---------- overlay ---------- #
        overlay = display.copy()

        panel_x1, panel_y1 = 15, 15
        panel_x2, panel_y2 = 1000, 260
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (255, 255, 255), -1)

        label = "NORMAL" if score < THRESHOLD else "ANOMALOUS"
        color = (0, 200, 0) if score < THRESHOLD else (220, 30, 30)

        cv2.putText(overlay, f"Score: {score:.3f} [{label}]", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(overlay, f"Category: {cat}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)

        max_chars = 55
        lines = [caption[i:i + max_chars] for i in range(0, len(caption), max_chars)]

        y = 160
        cv2.putText(overlay, "Caption:", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
        for i, line in enumerate(lines[:3]):
            cv2.putText(overlay, line, (150, y + i * 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)

        # OPT: update image in-place (no new Streamlit element)
        frame_window.image(overlay, use_container_width=True)

        # OPT: update metric card in-place via placeholder
        metric_placeholder.markdown(
            f"<div class='metric-card'>"
            f"<h3>Anomaly Score: <code>{score:.3f}</code> — <b>{label}</b></h3>"
            f"<p><b>Category:</b> {cat}</p>"
            f"<p><b>Caption:</b> {caption}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # OPT: single copy for session state
        st.session_state.last_frame = overlay
        st.session_state.last_score = score
        st.session_state.last_caption = caption
        st.session_state.last_category = cat

        segment_scores.append(score)
        segment_ranges.append((f_start, f_end))
        seg_summaries.append({
            "segment": seg_idx,
            "frame_start": f_start,
            "frame_end": f_end,
            "anomaly_score": round(score, 6),
            "label": "anomalous" if score >= THRESHOLD else "normal",
            "category": cat,
            "caption": cap_text,
        })
        st.session_state.frame_scores = segment_scores
        st.session_state.segment_summaries = seg_summaries

        seg_idx += 1
        elapsed = time.time() - start_time

        if seg_idx % 5 == 0:
            avg_ms = statistics.mean(timings) if timings else 0
            fps_disp = (seg_idx * frames_per_seg) / elapsed if elapsed > 0 else 0
            st.caption(
                f"Seg {seg_idx} | {fps_disp:.1f} fps | "
                f"Avg inference {avg_ms:.1f} ms | "
                f"Tprocess {avg_ms/1000:.3f}s (limit: {T_SEGMENT - T_OVERLAP:.1f}s)"
            )

        if prog is not None and total_frames > 0:
            prog.progress(min(f_end / total_frames, 1.0))

        f_start += frames_per_stride

    cap.release()

    if tmp_path is not None:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    st.success("Detection stopped.")


if not st.session_state.running and st.session_state.last_frame is not None:
    st.markdown("### Last Detected Frame")
    st.image(
        st.session_state.last_frame,
        caption=(
            f"Score: {st.session_state.last_score:.3f} | "
            f"Category: {st.session_state.last_category} | "
            f"Caption: {st.session_state.last_caption}"
        ),
        use_container_width=True,
    )