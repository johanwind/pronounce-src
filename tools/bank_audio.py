"""Stage 4 of the English reading bank: the model voice for every bank passage and every focus-area
example phrase, as MP3, plus each passage's word timings (so one word can be played from the
passage audio). The hosted app serves these instead of running a TTS model, which keeps it small
enough for a 3 GB AWS Lambda.  Output: audio_<lang>/ (index.json + <hash>.mp3).

    reserve --mem 6G --cpus 4 -- .venv/bin/python tools/bank_audio.py
    PRONOUNCE_LANG=nb reserve --mem 8G --cpus 4 -- .venv/bin/python tools/bank_audio.py
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / f"audio_{os.environ.get('PRONOUNCE_LANG', 'en')}"


def key(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def main():
    import paragraphs
    from scoring import PLAY_SPEED, PLAY_VOICE, Scorer
    from stats import AREAS

    OUT.mkdir(exist_ok=True)
    index = json.loads((OUT / "index.json").read_text()) if (OUT / "index.json").exists() else {}
    s = Scorer()
    phrases = [ex for _, ex, _ in AREAS.values() if ex]
    texts = [(t, True) for t in paragraphs.BANK] + [(p, False) for p in phrases]
    t0 = time.time()
    for i, (text, timings) in enumerate(texts):
        k = key(text)
        if k in index and (OUT / f"{k}.mp3").exists():
            continue
        a = s.tts_audio(text, PLAY_VOICE, speed=PLAY_SPEED, rate=24000)  # nb: slowed to a careful pace
        sf.write(OUT / f"{k}.mp3", a, 24000, format="MP3")
        entry = dict(file=f"{k}.mp3", sec=round(len(a) / 24000, 2))
        if timings:  # word start/end in the audio, from aligning the voice to its own text
            a16 = resample_poly(a, 2, 3).astype(np.float32)
            wp = [(w, alts[0]) for w, alts, _ in s.word_prons(text)]
            res = s.align(a16, wp) or []
            words = [[None, None] for _ in wp]
            for r in res:
                if r["phone"] == "|":
                    continue
                w = words[r["word"]]
                w[0] = r["start"] if w[0] is None else min(w[0], r["start"])
                w[1] = r["end"] if w[1] is None else max(w[1], r["end"])
            entry["words"] = [[round(x, 3) if x is not None else None for x in w] for w in words]
        index[k] = entry
        if i % 50 == 0:
            (OUT / "index.json").write_text(json.dumps(index))
            print(f"{i}/{len(texts)}  {(time.time() - t0) / (i + 1):.2f} s each", flush=True)
    (OUT / "index.json").write_text(json.dumps(index))
    size = sum(p.stat().st_size for p in OUT.glob("*.mp3"))
    print(f"{len(index)} clips, {size / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
