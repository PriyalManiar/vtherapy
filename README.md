# VTherapy

Hand physiotherapy app: chatbot, video exercise recognition, tracking, and appointment booking.

## Run with Docker

### 1. Build and run

```bash
docker build -t vtherapy .
docker run -p 5001:5001 vtherapy
```

Open **http://localhost:5001** in your browser.

### 2. Run with persistent data (DB, uploads)

Using docker-compose:

```bash
docker compose up --build
```

Or with plain docker:

```bash
docker run -p 5001:5001 \
  -v vtherapy_instance:/app/instance \
  -v vtherapy_cam_inputs:/app/cam_inputs \
  -v vtherapy_processed:/app/processed_inputs \
  vtherapy
```

### 3. Optional: Mail env vars for appointment form

```bash
docker run -p 5001:5001 \
  -e MAIL_USERNAME=your@email.com \
  -e MAIL_PASSWORD=yourpassword \
  vtherapy
```

## Run without Docker (development)

Requires Python 3.8+, ffmpeg, and pip.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
python -m scripts.download_hand_landmarker   # if models/hand_landmarker.task missing
python -m scripts.chatbot_train              # if models/data.pth missing
python run.py
```

Open **http://127.0.0.1:5001**.

## Project structure

```
vtherapy/
├── run.py                 # App entry point
├── Dockerfile
├── config/
│   └── intents.json       # Chatbot intents
├── instance/
│   └── virtuotherapy.db   # SQLite DB (created on first run)
├── models/                # ML models
│   ├── data.pth           # Chatbot model (from chatbot_train)
│   ├── hand_landmarker.task  # MediaPipe hand model (download if missing)
│   └── bestonemoreresnetlstm.pth  # Exercise classifier
├── sample_exercises/      # Reference videos for accuracy comparison
│   ├── spreadfingers.mp4
│   └── makefist.mp4
├── cam_inputs/            # Uploaded camera recordings
├── processed_inputs/      # Processed frames/landmarks
├── scripts/
├── static/
├── templates/
└── requirements.txt
```

## Requirements

- Python 3.8+ (or Docker)
- ffmpeg (for video conversion)
- Optional: `MAIL_USERNAME`, `MAIL_PASSWORD` env vars for appointment emails
