"""Formant analysis of the user's ʊ drill recordings vs the model voices.

Re-scores each saved drill recording to get the vowel segment, then measures
F1/F2/F3 (Praat Burg) at 25/50/75% of the vowel and its duration. Writes
data/drill_formants.json.

    reserve --mem 4G --cpus 4 -- .venv/bin/python tests/analyse_drill_formants.py
"""
import json, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import parselmouth

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scoring import Scorer
from drill import Drill, PROTO_VOICES

s = Scorer(device="cpu", log=lambda *a: None)
d = Drill(s, log=lambda *a: None)
d.bank.ready.wait()


def formants(audio, rate, seg, ceiling):
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=rate)
    f = snd.to_formant_burg(time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling)
    out = []
    a, b = seg
    for frac in (0.25, 0.5, 0.75):
        t = a + frac * (b - a)
        out.append([f.get_value_at_time(i, t) for i in (1, 2, 3)])
    return out


def measure(audio, rate, seg):
    return {str(c): formants(audio, rate, seg, c) for c in (5000, 5500)}


out = {"user": [], "model": []}
rows = [json.loads(l) for l in open(ROOT / "data/drill.jsonl")]
for r in rows:
    if r.get("target") != "ʊ" or "rec" not in r:
        continue
    p = ROOT / "data/drill_recordings" / f"{r['rec']}.flac"
    if not p.exists():
        continue
    a, rate = sf.read(p, dtype="float32")
    res = d.attempt(r["word"], "ʊ", a, rate)
    if "error" in res:
        continue
    out["user"].append(dict(word=r["word"], rec=r["rec"], time=r["time"], score=res["score"],
                            logged=r["score"], heard=res["heard"], ms=res["ms"], seg=res["seg"],
                            f=measure(a, rate, res["seg"])))

json.dump(out, open(ROOT / "data/drill_formants.json", "w"))
# model voices: the drilled words plus uː / oʊ / ʌ comparison words
for word, target in [("book", "ʊ"), ("good", "ʊ"), ("look", "ʊ"), ("put", "ʊ"),
                     ("boot", "uː"), ("food", "uː"), ("boat", "oʊ"), ("goad", "oʊ"),
                     ("buck", "ʌ"), ("cut", "ʌ")]:
    for v in PROTO_VOICES:
        wav = s.tts_audio(word + ".", v)
        res = d.attempt(word, target, wav, 16000)
        if "error" in res:
            continue
        out["model"].append(dict(word=word, target=target, voice=v, score=res["score"], ms=res["ms"],
                                 seg=res["seg"], f=measure(wav, 16000, res["seg"])))
    print(word, "done", flush=True)

json.dump(out, open(ROOT / "data/drill_formants.json", "w"))
print(len(out["user"]), "user", len(out["model"]), "model")
