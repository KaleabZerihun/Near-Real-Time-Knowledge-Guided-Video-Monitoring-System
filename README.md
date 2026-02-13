TO run this project:
 make sure the file shared by the sponosr(flashback_text_embedings_SAP.npy) is in the Thesis/src/memory folder…it will not be there so you have to download it and put it there.
 then after you do a gitll pull
-cd Near-Real-Time-Knowledge-Guided-Video-Monitoring-System
- py -3.11 -m venv .venv
- .\.venv\Scripts\Activate.ps1

# macOS/Linux
# python3.11 -m venv .venv
# source .venv/bin/activate

- python -m pip install --upgrade pip
- pip install -r backend/requirements.txt

- cd backend
- uvicorn main:app --reload --port 8000
