"""
Download MediaPipe hand_landmarker.task if missing.
Used at app startup and can be run standalone: python -m scripts.download_hand_landmarker
"""
import os
import urllib.request

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
FILENAME = "hand_landmarker.task"


def ensure_hand_landmarker(base_dir: str) -> str | None:
    """
    If models/hand_landmarker.task does not exist, download it.
    base_dir should be the project root (where models/ folder lives).
    Returns the path to the file, or None if download failed.
    """
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, FILENAME)

    if os.path.isfile(path):
        return path

    print(f"Downloading {FILENAME} from MediaPipe...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
        print(f"Saved to {path}")
        return path
    except Exception as e:
        print(f"Download failed: {e}")
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass
        return None


if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = ensure_hand_landmarker(PROJECT_ROOT)
    exit(0 if result else 1)
