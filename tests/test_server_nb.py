"""End-to-end test of the Norwegian reader (reader.py --lang nb) against a running server.
Start it with PRONOUNCE_DATA pointing to a temp dir so your history is untouched:

    PRONOUNCE_DATA=/tmp/x .venv/bin/python reader.py --lang nb --port 8796
    .venv/bin/python tests/test_server_nb.py http://localhost:8796

The planted errors are synthetic (Piper reads the wrong text); real-speech numbers come
from tests/nb_fleurs.py.
"""
import io, json, sys, time, urllib.parse, urllib.request
import numpy as np, soundfile as sf

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8766"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=300) as r:
        return r.read(), r.headers.get("Content-Type")


def post(path, body, headers=None):
    req = urllib.request.Request(BASE + path, data=body, headers=headers or {}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def tts(text):
    b, ct = get("/api/tts?text=" + urllib.parse.quote(text))
    assert ct == "audio/wav", ct
    return sf.read(io.BytesIO(b), dtype="float32")


def analyse(text, audio, sr):
    t0 = time.time()
    d = post(f"/api/analyse?rate={sr}", audio.astype("<f4").tobytes(),
             {"X-Text": urllib.parse.quote(text)})
    return d, time.time() - t0


st = json.loads(get("/api/state")[0])
assert st["lang"] == "nb", st
n_hist = len(st["history"])
p = post("/api/paragraph", b'{"kind":"random"}')["text"]
print("paragraph:", p[:70], "...")

# 1. the model voice reading the paragraph: (nearly) everything clear
a, sr = tts(p)
pad = np.zeros(sr, np.float32)
noisy = np.concatenate([pad, a, pad]) + np.random.default_rng(0).normal(0, 0.003, len(a) + 2 * sr).astype(np.float32)
d, dt = analyse(p, noisy, sr)
s = d["stats"]
print(f"clean: clarity={s['clarity']} clear={s['words_clear']}/{s['words_scored']} wpm={s['wpm']} ({dt:.1f}s)")
print("  flagged:", [(w["word"], w.get("heard_as")) for w in d["words"] if w["score"] is not None and w["score"] < 0.5])
assert s["clarity"] >= 90, s
assert s["words_clear"] >= s["words_scored"] - 2, s
w0 = next(w for w in d["words"] if w["start"] is not None)
lead = np.argmax(np.abs(a) > 0.02) / sr  # the voice's own silence before the first word (Google: ~0.9 s)
assert 0.9 + lead < w0["start"] < 1.7 + lead, (lead, w0)  # offset includes the 1 s padding

# 2. planted letter swaps (ø->e, u->y) and one misread word (torget -> markedet)
good = "Først låste hun huset med en stor nøkkel, og så gikk hun til torget i byen."
bad = "Ferst låste hun hyset med en stor nekkel, og så gikk hun til markedet i byen."
assert post("/api/paragraph", json.dumps({"kind": "custom", "text": good}).encode())["text"] == good
ab, sr = tts(bad)
d2, _ = analyse(good, ab, sr)
flag = {w["word"]: w for w in d2["words"] if w["score"] is not None and w["score"] < 0.5}
print("planted:", [(w, f.get("heard_as"), "slip" if f.get("slip") else [b["phone"] + "→" + b["heard"] for b in f["bad"]])
                   for w, f in flag.items()])
# Piper samples noise, so a borderline swap can vary between server runs, and the recogniser
# leans towards real words (Piper's "sjøpte" is usually heard as kjøpte, tests/nb_planted_tts.py);
# these three were caught with both Piper voices there
said = {"Først": "ferst", "huset": "hyset", "nøkkel": "nekkel"}
hit = [w for w in said if w in flag]
assert len(hit) >= 2, flag
for w in hit:
    assert flag[w].get("heard_as") and not flag[w].get("slip") and flag[w]["bad"], flag[w]
assert flag["torget"].get("slip") and flag["torget"]["bad"] == [], flag.get("torget")
assert "byen" not in flag and "låste" not in flag and "stor" not in flag
print("focus:", [(f["area"], f["evidence"]) for f in d2["focus"]])
# a misreading doesn't count against any sound area (its letters are unscored)
assert all(p["norm"] is None for p in flag["torget"]["phones"]), flag["torget"]

# 3. silence / too short -> friendly error
d3, _ = analyse(good, np.zeros(sr * 3, np.float32), sr)
assert "error" in d3, d3
d4, _ = analyse(p, a[: sr // 2], sr)
assert "error" in d4, d4.get("stats")

# 4. coach endpoint answers (ok or a clean error)
c = json.loads(get(f"/api/coach?id={d2['id']}&wait=120")[0])
print("coach:", {k: (v if k != "focus" else [f.get("area") for f in v]) for k, v in c.items() if k != "next_paragraphs"})
assert "ok" in c

st = json.loads(get("/api/state")[0])
assert len(st["history"]) >= n_hist + 2, len(st["history"])
print("ALL OK")
