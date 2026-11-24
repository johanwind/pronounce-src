"""End-to-end test against a running reader.py (PRONOUNCE_DATA should point to a temp dir)."""
import io, json, sys, time, urllib.parse, urllib.request
import numpy as np, soundfile as sf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=200) as r:
        return r.read(), r.headers.get("Content-Type")

def post(path, body, headers=None):
    req = urllib.request.Request(BASE + path, data=body, headers=headers or {}, method="POST")
    with urllib.request.urlopen(req, timeout=200) as r:
        return json.load(r)

def tts(text):
    b, ct = get("/api/tts?text=" + urllib.parse.quote(text))
    assert ct == "audio/wav", ct
    a, sr = sf.read(io.BytesIO(b), dtype="float32")
    return a, sr

def analyse(text, audio, sr):
    t0 = time.time()
    d = post(f"/api/analyse?rate={sr}", audio.astype("<f4").tobytes(),
             {"X-Text": urllib.parse.quote(text)})
    return d, time.time() - t0

html, ct = get("/")
assert b"Read Aloud" in html and ct.startswith("text/html")
st = json.loads(get("/api/state")[0]); assert st["history"] == [], st
p = post("/api/paragraph", b'{"kind":"random"}')["text"]
print("paragraph:", p[:70], "...")

# 1. clean reading by the model voice, with 1s silence padding and quiet noise
a, sr = tts(p)
pad = np.zeros(sr, np.float32)
noisy = np.concatenate([pad, a, pad]) + np.random.default_rng(0).normal(0, 0.003, len(a) + 2 * sr).astype(np.float32)
d, dt = analyse(p, noisy, sr)
s = d["stats"]
print(f"clean: clarity={s['clarity']} clear={s['words_clear']}/{s['words_scored']} wpm={s['wpm']} hes={len(s['hesitations'])} ({dt:.1f}s)")
assert s["clarity"] >= 90, s
assert s["words_clear"] >= s["words_scored"] - 2
w0 = next(w for w in d["words"] if w["start"] is not None)
assert 0.9 < w0["start"] < 1.7, w0  # offset includes leading silence
assert d["previous"] is None

# 2. same text again -> previous is set
d2, _ = analyse(p, a, sr)
assert d2["previous"] and d2["previous"]["stats"]["clarity"] == s["clarity"], d2["previous"]

# 3. planted errors via custom text
good = "I think the weather will be very warm this weekend, so we should visit the village by the river."
bad = "I sink the wetter will be wery warm this weekend, so we should visit the willage by the river."
assert post("/api/paragraph", json.dumps({"kind": "custom", "text": good}).encode())["text"] == good
ab, sr = tts(bad)
d3, _ = analyse(good, ab, sr)
flag = [w["word"] for w in d3["words"] if w["score"] is not None and w["score"] < 0.5]
print("planted errors flagged:", flag)
print("focus:", [(f["area"], f["evidence"]) for f in d3["focus"]])
# TTS "sink"/"wetter" stay close to think/weather for the recogniser; since the 2026-11-19
# real-speech calibration (tests/en_speechocean.py) only the clear v->w swaps reliably flag
assert {"very", "village"} <= set(flag), flag
# th is no longer reliably flagged on TTS "sink"/"wetter" since the real-speech calibration
assert any("v and w" == f["area"] for f in d3["focus"]), d3["focus"]

# 4. silence / too-short recordings give a friendly error, not a crash
d4, _ = analyse(good, np.zeros(sr * 3, np.float32), sr)
assert "error" in d4, d4
d5, _ = analyse(p, a[: sr // 2], sr)
assert "error" in d5, d5.get("stats")
# 6. stopped halfway: flagged as incomplete
half = a[: len(a) // 2]
d6, _ = analyse(p, half, sr)
print("half reading: maybe_incomplete =", d6.get("stats", {}).get("maybe_incomplete"), d6.get("error"))
assert d6.get("error") or d6["stats"]["maybe_incomplete"], d6.get("stats")

# 7. coach endpoint returns (ok or a clean error)
c = json.loads(get(f"/api/coach?id={d3['id']}&wait=100")[0])
print("coach:", {k: (v if k != "focus" else len(v)) for k, v in c.items()})
assert "ok" in c

# 8. next paragraph: from coach or bank, different from the last one
n = post("/api/paragraph", b'{"kind":"next"}')["text"]
print("next:", n[:80])
st = json.loads(get("/api/state")[0])
assert len(st["history"]) >= 4, len(st["history"])
print("state areas:", [(a["area"], a["mean"]) for a in st["areas"][:4]])
assert all(a["mean"] > 0.6 for a in st["areas"] if a["n"] >= 5), "garbage readings leaked into stats"
print("ALL OK")
