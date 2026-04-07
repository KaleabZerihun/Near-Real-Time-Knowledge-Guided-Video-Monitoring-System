To run this project:

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
