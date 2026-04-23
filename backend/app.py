# ========================= IMPORTS ========================= #
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import librosa
import numpy as np
import io
import os
import logging
import tempfile
import time
import threading
import functools
import random
import string

# ========================= CONFIG ========================= #
class Config:
    DEBUG = True
    MAX_AUDIO_SIZE = 10 * 1024 * 1024
    TEMP_DIR = "temp_files"
    LOG_LEVEL = logging.INFO

config = Config()

# ========================= LOGGER ========================= #
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger("AI_BACKEND")

# ========================= APP INIT ========================= #
app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

if not os.path.exists(config.TEMP_DIR):
    os.makedirs(config.TEMP_DIR)

# ========================= UTILITIES ========================= #
def generate_id(length=12):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def safe_execute(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return jsonify({"error": "Internal failure"}), 500
    return wrapper

def validate_json(keys):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = request.json
            if not data:
                return jsonify({"error": "Missing JSON"}), 400
            for key in keys:
                if key not in data:
                    return jsonify({"error": f"Missing key: {key}"}), 400
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ========================= AI SERVICES ========================= #
class ChatbotService:
    def __init__(self):
        self.name = "CoreChatbotEngine"

    def process(self, text):
        # Simulated AI logic
        return f"[AI]: Processed -> {text[::-1]}"

class SentimentService:
    def analyze(self, text):
        score = random.random()
        sentiment = "positive" if score > 0.6 else "negative" if score < 0.4 else "neutral"
        return {"sentiment": sentiment, "confidence": score}

class AudioService:
    def decode_audio(self, audio_base64):
        if "," in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        return base64.b64decode(audio_base64)

    def load_audio(self, audio_bytes):
        return librosa.load(io.BytesIO(audio_bytes), sr=None)

    def analyze_emotion(self, audio_array):
        val = np.mean(np.abs(audio_array))
        if val > 0.5:
            return {"emotion": "angry", "score": val}
        elif val > 0.2:
            return {"emotion": "happy", "score": val}
        return {"emotion": "neutral", "score": val}

    def transcribe(self, audio_bytes):
        return "simulated transcription"

# ========================= INSTANCES ========================= #
chatbot_service = ChatbotService()
sentiment_service = SentimentService()
audio_service = AudioService()

# ========================= MIDDLEWARE ========================= #
@app.before_request
def before():
    request.start_time = time.time()

@app.after_request
def after(response):
    duration = time.time() - request.start_time
    logger.info(f"{request.path} completed in {duration:.4f}s")
    return response

# ========================= ROUTES ========================= #
@app.route("/")
def index():
    return app.send_static_file("index.html")

# ---------------- TEXT CHAT ---------------- #
@app.route("/api/chat", methods=["POST"])
@safe_execute
@validate_json(["message"])
def chat():
    data = request.json
    message = data["message"]

    response = chatbot_service.process(message)
    sentiment = sentiment_service.analyze(message)

    return jsonify({
        "id": generate_id(),
        "response": response,
        "sentiment": sentiment["sentiment"],
        "confidence": sentiment["confidence"]
    })

# ---------------- AUDIO CHAT ---------------- #
@app.route("/api/audio-chat", methods=["POST"])
@safe_execute
@validate_json(["audio"])
def audio_chat():
    data = request.json
    audio_base64 = data["audio"]

    if len(audio_base64) > config.MAX_AUDIO_SIZE:
        return jsonify({"error": "Audio too large"}), 400

    audio_bytes = audio_service.decode_audio(audio_base64)
    audio_array, sr = audio_service.load_audio(audio_bytes)

    sentiment = audio_service.analyze_emotion(audio_array)
    text = audio_service.transcribe(audio_bytes)

    response = chatbot_service.process(text)

    return jsonify({
        "id": generate_id(),
        "response": response,
        "transcription": text,
        "emotion": sentiment["emotion"],
        "confidence": sentiment["score"]
    })

# ========================= THREADING TASK ========================= #
def background_cleanup():
    while True:
        time.sleep(60)
        try:
            for file in os.listdir(config.TEMP_DIR):
                path = os.path.join(config.TEMP_DIR, file)
                if os.path.isfile(path):
                    os.remove(path)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

# ========================= ADVANCED HELPERS ========================= #
class Pipeline:
    def __init__(self):
        self.steps = []

    def add(self, func):
        self.steps.append(func)
        return self

    def run(self, data):
        for step in self.steps:
            data = step(data)
        return data

def noise_filter(audio):
    return audio * 0.95

def normalize(audio):
    return audio / np.max(np.abs(audio) + 1e-6)

audio_pipeline = Pipeline().add(noise_filter).add(normalize)

# ========================= EXTRA ROUTES ========================= #
@app.route("/api/debug", methods=["GET"])
def debug():
    return jsonify({
        "status": "running",
        "services": ["chatbot", "sentiment", "audio"],
        "uptime": time.time()
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"health": "ok"})

# ========================= ========================= #
def complex_transform(x):
    for _ in range(50):
        x = np.tanh(x) + np.sin(x)
    return x

def deep_layer_process(data):
    arr = np.array(data)
    arr = complex_transform(arr)
    return np.mean(arr)

# ========================= MODEL ========================= #
class model:
    def predict(self, x):
        return deep_layer_process(x)

model = model()

@app.route("/api/model", methods=["POST"])
@safe_execute
def model_route():
    data = request.json.get("data", [1,2,3])
    result = model.predict(data)
    return jsonify({"prediction": float(result)})

# ========================= SECURITY LAYER ========================= #
def simple_auth(token):
    return token == "secure123"

@app.route("/api/secure", methods=["POST"])
def secure():
    token = request.headers.get("Authorization", "")
    if not simple_auth(token):
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"message": "Secure access granted"})


# ========================= FINAL RUN ========================= #
if __name__ == "__main__":
    app.run(debug=config.DEBUG)
