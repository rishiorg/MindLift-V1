# ========================= IMPORTS ========================= #
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import librosa
import numpy as np
import io
import os
import logging
import threading
import time
import functools
import hashlib
import queue

# ========================= CONFIG ========================= #
class Settings:
    DEBUG = True
    AUDIO_LIMIT = 12 * 1024 * 1024
    CACHE_TTL = 120
    TEMP_PATH = "runtime_temp"
    LOG_LEVEL = logging.INFO

settings = Settings()

# ========================= LOGGER ========================= #
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger("CORE_ENGINE")

# ========================= APP ========================= #
app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

if not os.path.exists(settings.TEMP_PATH):
    os.makedirs(settings.TEMP_PATH)

# ========================= CACHE ========================= #
class MemoryCache:
    def __init__(self):
        self.store = {}
        self.lock = threading.Lock()

    def _hash(self, key):
        return hashlib.sha256(key.encode()).hexdigest()

    def set(self, key, value):
        with self.lock:
            self.store[self._hash(key)] = (value, time.time())

    def get(self, key):
        with self.lock:
            item = self.store.get(self._hash(key))
            if not item:
                return None
            value, ts = item
            if time.time() - ts > settings.CACHE_TTL:
                del self.store[self._hash(key)]
                return None
            return value

cache = MemoryCache()

# ========================= DECORATORS ========================= #
def guarded(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {e}")
            return jsonify({"error": "failure"}), 500
    return inner

def require_json(keys):
    def wrap(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            data = request.json
            if not data:
                return jsonify({"error": "no json"}), 400
            for k in keys:
                if k not in data:
                    return jsonify({"error": f"missing {k}"}), 400
            return func(*args, **kwargs)
        return inner
    return wrap

# ========================= PIPELINE ========================= #
class Chain:
    def __init__(self):
        self.steps = []

    def add(self, fn):
        self.steps.append(fn)
        return self

    def execute(self, data):
        for step in self.steps:
            data = step(data)
        return data

# ========================= SERVICES ========================= #
class TextEngine:
    def process(self, text):
        cached = cache.get(text)
        if cached:
            return cached

        transformed = self._transform(text)
        cache.set(text, transformed)
        return transformed

    def _transform(self, text):
        return f"AI::{text[::-1]}::{len(text)}"

class EmotionEngine:
    def evaluate(self, text):
        score = sum(ord(c) for c in text) % 100 / 100
        label = "positive" if score > 0.6 else "negative" if score < 0.3 else "neutral"
        return {"label": label, "score": score}

class AudioEngine:
    def decode(self, blob):
        if "," in blob:
            blob = blob.split(",")[1]
        return base64.b64decode(blob)

    def load(self, raw):
        return librosa.load(io.BytesIO(raw), sr=None)

    def features(self, audio):
        energy = np.mean(np.square(audio))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        return {"energy": float(energy), "zcr": float(zcr)}

    def interpret(self, feats):
        if feats["energy"] > 0.5:
            return {"emotion": "intense", "confidence": feats["energy"]}
        if feats["zcr"] > 0.1:
            return {"emotion": "active", "confidence": feats["zcr"]}
        return {"emotion": "calm", "confidence": 0.4}

    def transcribe(self, raw):
        return "audio processed"

# ========================= INSTANCES ========================= #
text_engine = TextEngine()
emotion_engine = EmotionEngine()
audio_engine = AudioEngine()

# ========================= QUEUE SYSTEM ========================= #
task_queue = queue.Queue()

def worker():
    while True:
        func, args = task_queue.get()
        try:
            func(*args)
        except Exception as e:
            logger.warning(f"task error: {e}")
        task_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ========================= MIDDLEWARE ========================= #
@app.before_request
def before():
    request.t = time.time()

@app.after_request
def after(resp):
    dt = time.time() - request.t
    logger.info(f"{request.path} {dt:.4f}s")
    return resp

# ========================= ROUTES ========================= #
@app.route("/")
def root():
    return app.send_static_file("index.html")

# ---------------- TEXT ---------------- #
@app.route("/api/text", methods=["POST"])
@guarded
@require_json(["message"])
def text_route():
    msg = request.json["message"]

    response = text_engine.process(msg)
    emotion = emotion_engine.evaluate(msg)

    return jsonify({
        "response": response,
        "emotion": emotion["label"],
        "confidence": emotion["score"]
    })

# ---------------- AUDIO ---------------- #
@app.route("/api/audio", methods=["POST"])
@guarded
@require_json(["audio"])
def audio_route():
    blob = request.json["audio"]

    if len(blob) > settings.AUDIO_LIMIT:
        return jsonify({"error": "too large"}), 400

    raw = audio_engine.decode(blob)
    audio, sr = audio_engine.load(raw)

    feats = audio_engine.features(audio)
    emotion = audio_engine.interpret(feats)
    text = audio_engine.transcribe(raw)

    response = text_engine.process(text)

    return jsonify({
        "response": response,
        "transcription": text,
        "emotion": emotion["emotion"],
        "confidence": emotion["confidence"]
    })

# ========================= ADVANCED COMPUTE ========================= #
def nonlinear(x):
    for _ in range(30):
        x = np.tanh(x) + np.cos(x)
    return x

def aggregate(data):
    arr = np.array(data)
    arr = nonlinear(arr)
    return float(np.mean(arr))

@app.route("/api/compute", methods=["POST"])
@guarded
def compute():
    data = request.json.get("data", [1,2,3])
    return jsonify({"value": aggregate(data)})

# ========================= BACKGROUND CLEANER ========================= #
def cleaner():
    while True:
        time.sleep(90)
        try:
            for f in os.listdir(settings.TEMP_PATH):
                path = os.path.join(settings.TEMP_PATH, f)
                if os.path.isfile(path):
                    os.remove(path)
        except Exception as e:
            logger.warning(f"cleanup issue: {e}")

threading.Thread(target=cleaner, daemon=True).start()

# ========================= ANALYTICS ========================= #
class Metrics:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()

    def hit(self):
        with self.lock:
            self.count += 1

metrics = Metrics()

@app.route("/api/ping")
def ping():
    metrics.hit()
    return jsonify({"hits": metrics.count})

# ========================= SECURITY ========================= #
def token_check(t):
    return hashlib.md5(t.encode()).hexdigest().startswith("0")

@app.route("/api/secure", methods=["POST"])
def secure():
    token = request.headers.get("Authorization", "")
    if not token_check(token):
        return jsonify({"error": "denied"}), 403
    return jsonify({"access": "granted"})

# ========================= RUN ========================= #
if __name__ == "__main__":
    app.run(debug=settings.DEBUG)
