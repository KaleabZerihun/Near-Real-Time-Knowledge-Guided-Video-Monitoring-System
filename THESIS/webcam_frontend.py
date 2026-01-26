import cv2
import torch
import numpy as np
import time
import torch.nn.functional as F
import streamlit as st
import statistics
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# CONFIG
TEXT_EMB_PATH = "/Users/girisha/Desktop/thesis_local/thesis/src/memory/flashback_text_embeddings_SAP.npy"
CAPTIONS_PATH = "/Users/girisha/Desktop/thesis_local/thesis/src/memory/flashback_captions.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10

#css
st.markdown(
    """
    <style>
        /* Page Background */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa, #e8eaf6);
            color: #222;
        }
        /* Title */
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
        /* Buttons */
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
        /* Metrics area */
        .metric-card {
            background-color: #ffffffcc;
            border-radius: 12px;
            padding: 10px 20px;
            margin: 10px 0;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main-title'>Flashback Live - Video Anomaly Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-caption'>Zero-Shot Detection using ImageBind & Flashback Memory</div>", unsafe_allow_html=True)

#state variables
if "running" not in st.session_state:
    st.session_state.running = False
if "last_score" not in st.session_state:
    st.session_state.last_score = 0.0
if "last_caption" not in st.session_state:
    st.session_state.last_caption = "N/A"
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

def start_detection():
    st.session_state.running = True

def stop_detection():
    st.session_state.running = False

col1, col2 = st.columns(2)
col1.button("Start Live Detection", on_click=start_detection)
col2.button("Stop Detection", on_click=stop_detection)

@st.cache_resource
def load_model_and_memory():
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval().to(DEVICE)
    if DEVICE == "cuda":
        model.half()

    text_emb = torch.tensor(np.load(TEXT_EMB_PATH), dtype=torch.float32).to(DEVICE)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    if DEVICE == "cuda":
        text_emb = text_emb.half()

    n_total = text_emb.shape[0]
    half = n_total // 2
    y_labels = torch.zeros(n_total, device=DEVICE)
    y_labels[half:] = 1

    with open(CAPTIONS_PATH) as f:
        captions = [c.strip() for c in f.readlines()]

    return model, text_emb, y_labels, captions

if st.session_state.running:
    model, text_emb, y_labels, captions = load_model_and_memory()
    st.success("Model and memory loaded successfully!")

    def compute_anomaly_score(video_emb):
        scores = torch.matmul(video_emb, text_emb.T)
        topk_vals, topk_idx = torch.topk(scores, k=TOP_K, dim=-1)
        weights = F.softmax(topk_vals, dim=-1)
        selected_labels = y_labels[topk_idx[0]]
        A_s = torch.sum(weights[0] * selected_labels)
        top_caption = captions[topk_idx[0][0].item()] if captions else "Unknown"
        return A_s.item(), top_caption

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam. Please allow camera permissions.")
        st.stop()

    frame_window = st.image(np.zeros((720, 1280, 3), dtype=np.uint8), use_container_width=True)
    score_placeholder = st.empty()
    caption_placeholder = st.empty()
    st.info("Press **Stop Detection** to end demo.")

    start_time = time.time()
    frame_count = 0
    timings = []

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Frame grab failed.")
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        display_frame = cv2.resize(frame, (1280, 720))  
        model_input = cv2.resize(frame, (224, 224))     
        rgb = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(DEVICE, dtype=torch.half if DEVICE == "cuda" else torch.float32)

        t0 = time.time()
        with torch.inference_mode():
            video_emb = model({ModalityType.VISION: frame_tensor})[ModalityType.VISION]
        t1 = time.time()
        timings.append((t1 - t0) * 1000)
        video_emb = F.normalize(video_emb, p=2, dim=-1)
        anomaly_score, top_caption = compute_anomaly_score(video_emb)
        caption_display = top_caption.replace("Normal:", "").replace("Anomalous:", "").strip()

        color = (0, 255, 0) if anomaly_score < 0.5 else (255, 0, 0)
        overlay = display_frame.copy()
        cv2.putText(overlay, f"Anomaly: {anomaly_score:.3f}", (25, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
        cv2.putText(overlay, f"Caption: {caption_display[:70]}", (25, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        frame_window.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.container():
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown(f"### **Anomaly Score:** `{anomaly_score:.3f}`")
            st.markdown(f"**Caption:** {caption_display}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.last_score = anomaly_score
        st.session_state.last_caption = caption_display
        st.session_state.last_frame = overlay.copy()

        if frame_count % 10 == 0:
            avg_ms = statistics.mean(timings)
            fps = frame_count / (time.time() - start_time)
            print(f"[Frame {frame_count}] Avg Inference: {avg_ms:.1f} ms | Visible FPS: {fps:.2f}", flush=True)
            st.caption(f"{fps:.2f} FPS")

    cap.release()
    st.success("Webcam stopped successfully.")

if not st.session_state.running and st.session_state.last_frame is not None:
    st.image(
        cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB),
        caption=(
            f"Last Frame | Score: {st.session_state.last_score:.3f} "
            f"| Caption: {st.session_state.last_caption}"
        ),
        use_container_width=True
    )
    st.subheader("Final Summary")
    st.metric(label="Final Anomaly Score", value=f"{st.session_state.last_score:.3f}")
    st.write(f"**Predicted Caption:** {st.session_state.last_caption}")
    if st.session_state.last_score > 0.5:
        st.error("Classified as **ANOMALOUS**")
    else:
        st.success("Classified as **NORMAL**")
