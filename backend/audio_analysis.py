# ========================= IMPORTS ========================= #
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import base64, io, os, time, threading, logging, hashlib, queue, functools
import numpy as np
import librosa
from collections import OrderedDict

# ========================= CONFIG ========================= #
class Env:
    DEBUG = True
    RATE_LIMIT = 40
    WINDOW = 60
    CACHE_SIZE = 200
    TTL = 180
    TEMP = "rt"
    WORKERS = 3

env = Env()

# ========================= LOGGER ========================= #
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("SYS")

# ========================= APP ========================= #
app = Flask(__name__)
CORS(app)

if not os.path.exists(env.TEMP):
    os.makedirs(env.TEMP)

# ========================= STATE ========================= #
class Registry:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def inc(self, k):
        with self.lock:
            self.data[k] = self.data.get(k, 0) + 1

    def get(self):
        return self.data

registry = Registry()

# ========================= RATE LIMIT ========================= #
class Limiter:
    def __init__(self):
        self.bucket = {}

    def allow(self, key):
        now = time.time()
        hits = self.bucket.get(key, [])
        hits = [h for h in hits if now - h < env.WINDOW]

        if len(hits) >= env.RATE_LIMIT:
            return False

        hits.append(now)
        self.bucket[key] = hits
        return True

limiter = Limiter()

# ========================= CACHE ========================= #
class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.store = OrderedDict()

    def get(self, k):
        if k not in self.store:
            return None
        v, t = self.store.pop(k)
        if time.time() - t > env.TTL:
            return None
        self.store[k] = (v, t)
        return v

    def set(self, k, v):
        if k in self.store:
            self.store.pop(k)
        elif len(self.store) >= self.cap:
            self.store.popitem(last=False)
        self.store[k] = (v, time.time())

cache = LRU(env.CACHE_SIZE)

# ========================= EVENT BUS ========================= #
class Bus:
    def __init__(self):
        self.subs = {}

    def subscribe(self, event, fn):
        self.subs.setdefault(event, []).append(fn)

    def emit(self, event, data):
        for fn in self.subs.get(event, []):
            try:
                fn(data)
            except:
                pass

bus = Bus()

# ========================= EXECUTOR ========================= #
class Executor:
    def __init__(self):
        self.q = queue.Queue()
        for _ in range(env.WORKERS):
            threading.Thread(target=self._run, daemon=True).start()

    def submit(self, fn, *args):
        self.q.put((fn, args))

    def _run(self):
        while True:
            fn, args = self.q.get()
            try:
                fn(*args)
            except Exception as e:
                log.warning(f"task: {e}")
            self.q.task_done()

executor = Executor()

# ========================= CIRCUIT BREAKER ========================= #
class Circuit:
    def __init__(self):
        self.fail = 0
        self.state = "CLOSED"

    def call(self, fn, *a):
        if self.state == "OPEN":
            raise Exception("blocked")

        try:
            r = fn(*a)
            self.fail = 0
            return r
        except:
            self.fail += 1
            if self.fail > 3:
                self.state = "OPEN"
            raise

circuit = Circuit()

# ========================= PIPELINES ========================= #
class Flow:
    def __init__(self):
        self.nodes = []

    def add(self, fn):
        self.nodes.append(fn)
        return self

    def run(self, data):
        for n in self.nodes:
            data = n(data)
        return data

# ========================= ENGINES ========================= #
class Core:
    def transform(self, text):
        key = hashlib.md5(text.encode()).hexdigest()
        c = cache.get(key)
        if c:
            return c

        res = "".join(chr((ord(x)+3) % 127) for x in text)
        cache.set(key, res)
        return res

class Signal:
    def decode(self, blob):
        if "," in blob:
            blob = blob.split(",")[1]
        return base64.b64decode(blob)

    def load(self, raw):
        return librosa.load(io.BytesIO(raw), sr=None)

    def analyze(self, x):
        return {
            "rms": float(np.sqrt(np.mean(x**2))),
            "peak": float(np.max(np.abs(x)))
        }

class Affect:
    def infer(self, val):
        if val["peak"] > 0.7:
            return ("high", val["peak"])
        if val["rms"] > 0.2:
            return ("medium", val["rms"])
        return ("low", 0.3)

core = Core()
signal = Signal()
affect = Affect()

# ========================= STREAM ========================= #
def stream_gen(text):
    for i in range(0, len(text), 5):
        chunk = text[i:i+5]
        yield chunk
        time.sleep(0.05)

# ========================= DECORATORS ========================= #
def protect(fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        ip = request.remote_addr
        if not limiter.allow(ip):
            return jsonify({"error": "rate"}), 429
        try:
            return fn(*a, **k)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return wrap

# ========================= ROUTES ========================= #
@app.route("/api/text", methods=["POST"])
@protect
def text():
    data = request.json or {}
    msg = data.get("message","")

    registry.inc("text")

    result = circuit.call(core.transform, msg)

    bus.emit("text_event", result)

    return jsonify({"output": result})

@app.route("/api/audio", methods=["POST"])
@protect
def audio():
    data = request.json or {}
    blob = data.get("audio","")

    raw = signal.decode(blob)
    arr, sr = signal.load(raw)

    feats = signal.analyze(arr)
    label, conf = affect.infer(feats)

    txt = f"{label}:{conf}"
    out = core.transform(txt)

    registry.inc("audio")

    return jsonify({
        "output": out,
        "emotion": label,
        "confidence": conf
    })

@app.route("/api/stream", methods=["POST"])
def stream():
    msg = request.json.get("message","")
    result = core.transform(msg)

    return Response(stream_gen(result), mimetype="text/plain")

@app.route("/api/state")
def state():
    return jsonify(registry.get())

# ========================= EVENTS ========================= #
def log_event(data):
    log.info(f"event: {data}")

bus.subscribe("text_event", log_event)

# ========================= CLEANER ========================= #
def sweep():
    while True:
        time.sleep(120)
        try:
            for f in os.listdir(env.TEMP):
                os.remove(os.path.join(env.TEMP, f))
        except:
            pass

threading.Thread(target=sweep, daemon=True).start()

# ========================= RUN ========================= #
if __name__ == "__main__":
    app.run(debug=env.DEBUG)
