# VTherapy: DL-based Hand Rehabilitation System Post Injury

## Code Authors: Priyal Maniar & Gargi Vedpathak
## Published Work: [IEEE](https://ieeexplore.ieee.org/document/10842730)

A deep learning-based virtual platform designed to track hand exercises and provide real-time feedback. Get your questions answered via a personalized "PhysioBot." The system uses MediaPipe for hand landmarking and a custom ResNet-LSTM architecture to classify exercise accuracy.

## Key Features
* **Real-time Exercise Tracking:** Leverages MediaPipe for high-fidelity hand skeletal tracking.
* **Accuracy Classification:** Custom ResNet-LSTM model to evaluate finger spreading and fist-making precision.
* **PhysioBot:** An NLTK-powered chatbot that guides users through their rehabilitation journey.
* **Dockerized Infrastructure:** Consistent deployment across all environments including specialized support for FFmpeg and computer vision libraries.

## Tech Stack
* **Core:** Python 3.11, Flask
* **ML/CV:** PyTorch, MediaPipe, OpenCV (Headless)
* **NLP:** NLTK
* **Infrastructure:** Docker, Docker Compose, Git LFS

---

## Project Structure
```text
vtherapy/
├── run.py                 # App entry point
├── Dockerfile
├── docker-compose.yml     # Docker orchestration
├── config/
│   └── intents.json       # Chatbot intents
├── instance/
│   └── virtuotherapy.db   # SQLite DB (created on first run)
├── models/                # ML models
│   ├── data.pth           # Chatbot model (from chatbot_train)
│   ├── hand_landmarker.task  # MediaPipe hand model
│   └── bestonemoreresnetlstm.pth  # Exercise classifier (LFS)
├── sample_exercises/      # Reference videos for accuracy comparison
├── cam_inputs/            # Uploaded camera recordings
├── processed_inputs/      # Processed frames/landmarks
├── scripts/               # Logic for CV, NLP, and Data
├── static/                # CSS, JS, and Video assets
├── templates/             # HTML Frontend
└── requirements.txt

---
```
##  Installation & Setup

### 1. Prerequisites
* [Git LFS](https://git-lfs.com/) installed on your local machine.
* **Optional:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) for containerized execution.

### 2. Clone and Initialize
```bash
git clone [https://github.com/PriyalManiar/vtherapy.git](https://github.com/PriyalManiar/vtherapy.git)
cd vtherapy
git lfs pull
```

### 3. Run with Docker (Recommended)
```bash
# This builds the image and starts the Flask server on port 5001
docker-compose up --build
```
The app will be accessible at `http://localhost:5001`.

### 4. Run without Docker (Development)
Requires Python 3.8+, ffmpeg, and pip.
```bash
# Install dependencies
pip install -r requirements.txt

# Setup models and NLP data
python -c "import nltk; nltk.download('punkt')"
python -m scripts.download_hand_landmarker   # if hand_landmarker.task missing
python -m scripts.chatbot_train              # if data.pth missing

# Launch
python run.py
```
Open `http://127.0.0.1:5001`.

---

##  Development & Testing
To verify the MediaPipe integration inside the container, run:
```bash
docker exec -it vtherapy-vtherapy-1 python -c "import mediapipe as mp; print('MediaPipe Version:', mp.__version__)"
```
---

