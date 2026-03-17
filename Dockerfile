FROM python:3.11-slim

# Install ffmpeg for video conversion
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data for chatbot
RUN python -c "import nltk; nltk.download('punkt', quiet=True)"

# Copy application
COPY . .

# Download hand landmarker model if not present (at build time)
RUN python -m scripts.download_hand_landmarker || true

# Train chatbot if data.pth missing (optional, can fail if intents invalid)
RUN python -m scripts.chatbot_train 2>/dev/null || true

# Bind to 0.0.0.0 for Docker networking
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5001

EXPOSE 5001

CMD ["python", "run.py"]
