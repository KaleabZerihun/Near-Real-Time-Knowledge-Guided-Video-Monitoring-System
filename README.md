# Near Real-Time Video Anomaly Detection System

Near Real-Time Video Anomaly Detection System monitors live webcam video and detects unusual events using a video anomaly detection model. The system processes video frames, analyzes activity, logs detected events, and displays the results through a web dashboard.

The system also allows users to type their own custom anomaly or event through a textbox. After the user submits the custom anomaly, the system watches the live video stream and attempts to detect that specific type of activity.

The goal of this project is to create a monitoring system that can help users identify abnormal activity in near real time. The dashboard shows live video, anomaly alerts, confidence scores, event logs, and performance graphs over time.

## Project Purpose

This project was built to explore how artificial intelligence and computer vision can be used in a near real-time monitoring system.

Many video monitoring systems require a person to manually watch footage for suspicious or unusual activity. This project helps reduce that problem by using an anomaly detection model to analyze live video and flag unusual activity automatically.

## Features

- Live webcam video processing
- Near real-time anomaly detection
- Frame selection and buffering
- Video anomaly detection model integration
- Custom anomaly input through a textbox
- Real-time anomaly alerts
- Event logging system
- SQLite database for storing logs
- Live web dashboard
- VAD result graph
- Over-time performance graphs
- Confidence score display
- Backend API for video and anomaly results
- MJPEG video streaming

## How the System Works

The system follows this general pipeline:


Webcam Stream
    ↓
Frame Selector / Buffer
    ↓
Video Anomaly Detection Model
    ↓
Custom Anomaly Matching / Event Monitoring
    ↓
Logger / Database
    ↓
Web Dashboard  


## To run this project:

- Memory embeddings are generated in `RT-VAD/embeddings/stage1/` using the scripts in `RT-VAD/`.
- If you change prompts, add new prompts, or modify `RT-VAD/flashback_batch.json`, regenerate memory via RT-VAD.

Memory generation flow:
1. Edit prompt batches in `RT-VAD/flashback_batch.json`.
2. Run `python RT-VAD/merge_prompts.py` to produce `RT-VAD/memory.json`.
3. Run `python RT-VAD/textencoder.py` to generate `RT-VAD/embeddings/stage1/`.
4. Run `streamlit run RT-VAD/flashback_eval.py` to view anomaly scores, frame visualization, and predictions.

If you do not change prompts or `flashback_batch.json`, you can skip directly to evaluation.

Setup:

- cd Near-Real-Time-Knowledge-Guided-Video-Monitoring-System
- python -m venv .venv
- .\.venv\Scripts\Activate.ps1
- python -m pip install --upgrade pip
- pip install -r backend/requirements.txt

Run backend:

- cd backend
- uvicorn main:app --reload --port 8000

Run frontend:

- cd frontend
- docker compose up --build
- open http://localhost:5173/
