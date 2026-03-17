from flask import Flask, request, jsonify, render_template
from flask import redirect, url_for
from flask_mailman import Mail, EmailMessage
import os
import cv2
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from .classifier_model import ResNeXtLSTMBinary
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import subprocess

# Hand landmark connections for drawing (MediaPipe 21-point hand model)
HAND_CONNECTIONS = (
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),  # palm
    (1, 2), (2, 3), (3, 4),   # thumb
    (5, 6), (6, 7), (7, 8),   # index
    (9, 10), (10, 11), (11, 12),  # middle
    (13, 14), (14, 15), (15, 16),  # ring
    (17, 18), (18, 19), (19, 20),  # pinky
)
from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy
from datetime import timedelta

# Project root (parent of scripts/) for db, templates, static, models, uploads, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'static'),
    static_url_path='/static',
)

_instance_dir = os.path.join(PROJECT_ROOT, 'instance')
os.makedirs(_instance_dir, exist_ok=True)
_db_path = os.path.join(_instance_dir, 'virtuotherapy.db').replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{_db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from .chat import get_response

# Mail configuration (used for booking form emails).
# Credentials are read from environment variables to avoid hardcoding secrets.
mail = Mail()
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', '587'))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
app.config['MAIL_USE_SSL'] = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
mail.init_app(app)

def calculate_streak():
    exercises = Exercise.query.order_by(Exercise.timestamp.asc()).all()
    streak = 0
    prev_date = None

    for exercise in exercises:
        exercise_date = exercise.timestamp.date()
        if prev_date is None or exercise_date == prev_date + timedelta(days=1):
            streak += 1
        elif exercise_date != prev_date:
            streak = 1
        prev_date = exercise_date

    return streak

@app.context_processor
def inject_streak():
    streak = calculate_streak()  
    return dict(streak=streak)

@app.route('/')
def index():
    # Start the app directly on the home page (no login gate).
    return redirect(url_for('home'))


@app.route("/predict",methods = ["POST"])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer":response}
    return jsonify(message)


class Exercise(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exercise_type = db.Column(db.String(50), nullable=False)
    count = db.Column(db.Integer, nullable=False, default=1)
    accuracy = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f"<Exercise {self.exercise_type}>"


def create_unique_folder(base_folder):

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    path = os.path.join(base_folder, timestamp)
    os.makedirs(path, exist_ok=True)
    return path


@app.route('/home')  
def home():
    # Render the actual home page of the app
    return render_template('start.html')

@app.route('/vtherapy') 
def vtherapy():
    return render_template('vtherapy.html') 

@app.route('/track')
def track():
    # Fetch all exercise data from the database
    exercises_data = Exercise.query.order_by(Exercise.timestamp.asc()).all()

    # Prepare the data for the progress charts
    data = [{
        'timestamp': exercise.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'type': exercise.exercise_type,
        'accuracy': exercise.accuracy
    } for exercise in exercises_data]

    today_start = datetime.combine(date.today(), datetime.min.time())
    today_end = datetime.combine(date.today(), datetime.max.time())

    # Get today's count for 'palmfingers' and 'palmfist'
    palmfingers_count = db.session.query(db.func.count(Exercise.id)).filter(
        Exercise.exercise_type == 'palmfingers',
        Exercise.timestamp >= today_start,
        Exercise.timestamp <= today_end
    ).scalar() or 0

    palmfist_count = db.session.query(db.func.count(Exercise.id)).filter(
        Exercise.exercise_type == 'palmfist',
        Exercise.timestamp >= today_start,
        Exercise.timestamp <= today_end
    ).scalar() or 0

    return render_template('track.html', exercises_data=data, palmfingers_count=palmfingers_count, palmfist_count=palmfist_count)

@app.route('/book')
def book():
    return render_template('book.html')

@app.route('/success')
def success():
    return render_template('success.html')  # A template that shows a success message


@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        contact_number = request.form.get('contactNumber')
        preferred_date = request.form.get('preferredDate')
        time_slot = request.form.get('timeSlot')
        additional_info = request.form.get('additionalInfo')

        email_body = f"""
Name: {name}
Age: {age}
Contact Number: {contact_number}
Preferred Date: {preferred_date}
Time Slot: {time_slot}
Additional Information: {additional_info}
""".strip()

        # If mail creds are not configured, skip sending and just show success.
        if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
            print("MAIL_USERNAME/MAIL_PASSWORD not set. Skipping email send.")
            return redirect(url_for('success'))

        email = EmailMessage(
            subject='New Appointment Request',
            body=email_body,
            to=[
                os.environ.get('APPOINTMENT_EMAIL_TO_1', 'gargi.vedpathak237@nmims.edu.in'),
                os.environ.get('APPOINTMENT_EMAIL_TO_2', 'priyal.maniar134@nmims.edu.in'),
            ],
        )

        try:
            email.send()
            return redirect(url_for('success'))
        except Exception as e:
            print(f"An error occurred while sending the email: {e}")
            return redirect(url_for('success'))
    return redirect(url_for('book'))


CAM_INPUTS_FOLDER = os.path.join(PROJECT_ROOT, 'cam_inputs')
PROCESSED_INPUTS_FOLDER = os.path.join(PROJECT_ROOT, 'processed_inputs')
SAMPLE_EXERCISES_FOLDER = os.path.join(PROJECT_ROOT, 'sample_exercises')
NUM_FRAMES = 10
SAMPLE_VIDEO_MAP = {'palmfingers': 'spreadfingers.mp4', 'palmfist': 'makefist.mp4'}

for folder in (CAM_INPUTS_FOLDER, PROCESSED_INPUTS_FOLDER, os.path.join(PROJECT_ROOT, 'models'), SAMPLE_EXERCISES_FOLDER):
    os.makedirs(folder, exist_ok=True)

from .download_hand_landmarker import ensure_hand_landmarker
ensure_hand_landmarker(PROJECT_ROOT)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        upload_run_folder = create_unique_folder(CAM_INPUTS_FOLDER)

        video_file = request.files.get('file')
        if not video_file:
            return jsonify({'error': 'No video file provided'}), 400

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        webm_filename = f"capture_{timestamp}.webm"
        mp4_filename = f"capture_{timestamp}.mp4"
        webm_video_path = os.path.join(upload_run_folder, webm_filename)
        mp4_video_path = os.path.join(upload_run_folder, mp4_filename)

        video_file.save(webm_video_path)

        if os.path.getsize(webm_video_path) < 1000:
            return jsonify({'error': 'Recording too short. Please record for at least 2–3 seconds with your hand visible.'}), 400

        convert_webm_to_mp4(webm_video_path, mp4_video_path)

        result = process_video(mp4_video_path)
        if result is None:
            return jsonify({'error': 'Video processing failed. Ensure models/hand_landmarker.task and models/bestonemoreresnetlstm.pth exist, and your hand is clearly visible in frame.'}), 503

        prediction, accuracy = result
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"upload_video error: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


    latest_exercise = Exercise.query.filter_by(exercise_type=prediction).order_by(Exercise.timestamp.desc()).first()
    count = (latest_exercise.count + 1) if latest_exercise else 1

    new_exercise = Exercise(exercise_type=prediction, count=count, accuracy=float(accuracy))
    db.session.add(new_exercise)
    db.session.commit()

    return jsonify({
        'prediction': prediction,
        'accuracy': accuracy,
        'count': count
    })

@app.route('/reset_count', methods=['POST'])
def reset_count():
    data = request.json
    exercise_type = data.get('exercise_type')
    if not exercise_type:
        return jsonify({'error': 'Missing exercise type'}), 400

    # Update the counter for the specified exercise type to 0
    Exercise.query.filter_by(exercise_type=exercise_type).update({'count': 0})
    db.session.commit()

    return jsonify({'message': f'Counter reset for {exercise_type}.'})


def estimate_hand_landmarks_in_video(video_path, hand_landmarker_path=None):
    if hand_landmarker_path is None:
        hand_landmarker_path = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    if not os.path.isfile(hand_landmarker_path):
        return []
    with open(hand_landmarker_path, 'rb') as f:
        model_data = f.read()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []
    all_hand_landmarks = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks])
                all_hand_landmarks.append(landmarks)

    cap.release()
    return all_hand_landmarks

def convert_webm_to_mp4(input_path, output_path):
    command = ['ffmpeg', '-y', '-i', input_path, '-crf', '18', '-preset', 'fast', output_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='replace')[:500]}")

def video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_path = os.path.join(output_folder, f"frame_{idx:04d}.png")
        cv2.imwrite(frame_path, frame)

    cap.release()


MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (0, 0, 0)


def _draw_hand_landmarks_cv2(image, hand_landmarks, handedness_name):
    """Draw landmarks and connections with cv2 (no mediapipe.solutions)."""
    height, width, _ = image.shape
    pts = []
    for lm in hand_landmarks:
        x, y = int(lm.x * width), int(lm.y * height)
        pts.append((x, y))
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    for (i, j) in HAND_CONNECTIONS:
        if i < len(pts) and j < len(pts):
            cv2.line(image, pts[i], pts[j], (0, 255, 0), 1)
    if pts and handedness_name:
        text_x = min(p[0] for p in pts)
        text_y = min(p[1] for p in pts) - MARGIN
        cv2.putText(image, handedness_name, (text_x, max(0, text_y)),
                    cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness_name = handedness_list[idx][0].category_name if handedness_list else ""
        _draw_hand_landmarks_cv2(annotated_image, hand_landmarks, handedness_name)
    return annotated_image

def process_frames_and_extract_landmarks(folder_path, output_folder, model_path):
    with open(model_path, 'rb') as file:
        model_data = file.read()

    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
    print(f"Found {len(frame_files)} frames to process for landmarks.")
        
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        image = mp.Image.create_from_file(frame_path)
        detection_result = detector.detect(image)
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
        filename_without_extension = os.path.splitext(frame_file)[0]

        output_path = os.path.join(output_folder, f"{filename_without_extension}.png")
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"Landmarks annotated and saved: {output_path}")


_exercise_model = None

def _get_exercise_model():
    global _exercise_model
    if _exercise_model is not None:
        return _exercise_model
    model_save_path = os.path.join(PROJECT_ROOT, 'models', 'bestonemoreresnetlstm.pth')
    if not os.path.isfile(model_save_path):
        return None
    try:
        base_model = models.resnext50_32x4d(weights=None)
    except TypeError:
        base_model = models.resnext50_32x4d(pretrained=False)
    _exercise_model = ResNeXtLSTMBinary(base_model, hidden_dim=512, lstm_layers=2, bidirectional=False, dropout_prob=0.3)
    _exercise_model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))
    _exercise_model.eval()
    return _exercise_model

def predict_exercise(frames_folder):
    model = _get_exercise_model()
    if model is None:
        return None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not frames:
        return None
    images = [transform(Image.open(frame)) for frame in frames]
    input_tensor = torch.stack(images).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    class_labels = ['palmfingers', 'palmfist']
    return class_labels[predicted.item()]

key_landmarks_indices = [4, 8, 12, 16, 20]  

def calculate_temporal_similarity(user_landmarks_list, sample_landmarks_list):
    similarities = []
    prev_user_landmarks = None
    prev_sample_landmarks = None

    for user_landmarks, sample_landmarks in zip(user_landmarks_list, sample_landmarks_list):
        
        user_key_landmarks = user_landmarks[key_landmarks_indices, :]
        sample_key_landmarks = sample_landmarks[key_landmarks_indices, :]

        
        norm_factor_user = np.linalg.norm(user_landmarks[0] - user_landmarks[12])
        norm_factor_sample = np.linalg.norm(sample_landmarks[0] - sample_landmarks[12])
        norm_factor = (norm_factor_user + norm_factor_sample) / 2  
        
        if prev_user_landmarks is not None and prev_sample_landmarks is not None:
            user_velocity = np.linalg.norm(user_key_landmarks - prev_user_landmarks[key_landmarks_indices, :]) / norm_factor
            sample_velocity = np.linalg.norm(sample_key_landmarks - prev_sample_landmarks[key_landmarks_indices, :]) / norm_factor
            velocity_similarity = 1 - min(abs(user_velocity - sample_velocity), 1)  
        else:
            velocity_similarity = 1  

        
        distance = np.linalg.norm((user_key_landmarks - sample_key_landmarks) / norm_factor)
        positional_similarity = 1 / (1 + distance)

        
        frame_similarity = (positional_similarity + velocity_similarity) / 2
        similarities.append(frame_similarity)

        
        prev_user_landmarks = user_landmarks
        prev_sample_landmarks = sample_landmarks

    
    overall_similarity = np.mean(similarities) if similarities else 0
    return overall_similarity * 100  

def process_video(video_path):
    hand_landmarker_path = os.path.join(PROJECT_ROOT, 'models', 'hand_landmarker.task')
    if not os.path.isfile(hand_landmarker_path):
        return None
    try:
        frames_folder = create_unique_folder(os.path.join(PROCESSED_INPUTS_FOLDER, 'frames'))
        landmarks_folder = create_unique_folder(os.path.join(PROCESSED_INPUTS_FOLDER, 'landmarks'))
        video_to_frames(video_path, frames_folder)
        process_frames_and_extract_landmarks(frames_folder, landmarks_folder, hand_landmarker_path)
        prediction = predict_exercise(landmarks_folder)
        if prediction is None:
            return None
        sample_filename = SAMPLE_VIDEO_MAP.get(prediction, f'{prediction}me.mp4')
        sample_video_path = os.path.join(SAMPLE_EXERCISES_FOLDER, sample_filename)
        if not os.path.isfile(sample_video_path):
            return prediction, 0.0
        user_landmarks_list = estimate_hand_landmarks_in_video(video_path)
        sample_landmarks_list = estimate_hand_landmarks_in_video(sample_video_path)
        min_length = min(len(user_landmarks_list), len(sample_landmarks_list))
        if min_length == 0:
            return prediction, 0.0
        user_landmarks_list = user_landmarks_list[:min_length]
        sample_landmarks_list = sample_landmarks_list[:min_length]
        accuracy = calculate_temporal_similarity(user_landmarks_list, sample_landmarks_list)
        return prediction, accuracy
    except Exception as e:
        print(f"process_video error: {e}")
        return None

if __name__ == "__main__":
    print("Starting Flask app...")
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5001)
