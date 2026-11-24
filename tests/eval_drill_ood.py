"""How well does the drill reject wrong vowels from voices/languages it has never seen?

For each ʊ target word: correct readings by unseen English voices should score high;
English near-misses (Luke/luck/lick...) and foreign vowels (Spanish [u], French [y],
[ø]) should score low. Prints mean scores and the accept rate at score >= 0.7.

    .venv/bin/python tests/eval_drill_ood.py
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import Scorer
from drill import Drill

s = Scorer(log=lambda *a: None)
d = Drill(s, log=print)
d.bank.ready.wait()

from kokoro import KPipeline
pipes = {"a": s.tts}
def say(text, voice):
    lang = voice[0]
    if lang not in pipes:
        pipes[lang] = KPipeline(lang_code=lang, model=s.tts.model)
    with s.lock:
        chunks = [a.numpy() for _, _, a in pipes[lang](text, voice=voice)]
    from scipy.signal import resample_poly
    return resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)

UNSEEN_EN = ["am_puck", "am_onyx", "am_echo", "bf_emma", "bm_george"]
CASES = {
    # target: (english near-misses, foreign: [(text, voice)])
    "look": (["Luke", "luck", "lick", "lock", "lake"], [("luc", "em_alex"), ("luc", "ef_dora"), ("Luc", "ff_siwis"), ("leuk", "ff_siwis"), ("luca", "im_nicola")]),
    "good": (["gooed", "god", "gid", "goad", "gud"], [("gud", "em_alex"), ("gude", "ff_siwis"), ("gueux", "ff_siwis"), ("gu", "pm_alex")]),
    "put": (["poot", "pit", "pot", "putt", "pate"], [("put", "em_alex"), ("pute", "ff_siwis"), ("peu", "ff_siwis"), ("putto", "im_nicola")]),
    "book": (["boot", "beak", "buck", "bake", "bike"], [("buk", "em_alex"), ("buc", "ff_siwis"), ("bouc", "ff_siwis")]),
}
RIGHT_EXTRA = ["food", "moon", "school", "think", "sit", "big", "cat", "bad", "seat", "green", "cup", "love", "bed", "boat"]
# synthesize everything once
items = []  # (kind, word, label, audio)
for word in RIGHT_EXTRA:
    d.reference(word)
    for v in UNSEEN_EN:
        items.append(("other", word, v, say(word + ".", v)))
for word, (misses, foreign) in CASES.items():
    d.reference(word)
    for v in UNSEEN_EN:
        items.append(("right", word, v, say(word + ".", v)))
    for m in misses:
        for v in UNSEEN_EN[:3]:
            items.append(("near", word, f"{m}/{v}", say(m + ".", v)))
    for text, v in foreign:
        items.append(("foreign", word, f"{text}/{v}", say(text + ".", v)))

def evaluate(tau_scale, ref_norm, len_penalty, verbose=False, context=True, mix=False):
    d.tau_scale, d.ref_norm, d.len_penalty, d.context, d.mix = tau_scale, ref_norm, len_penalty, context, mix
    d.cache.clear()  # model-voice scores depend on the mode (variant prototypes stay cached)

    res = {"right": [], "near": [], "foreign": [], "other": []}
    detail = []
    for kind, word, label, a in items:
        r = d.attempt(word, None, a, 16000)
        res[kind].append(r["score"])
        detail.append((kind, word, label, round(r["score"], 2), r["heard"], r["ms"], r["ref_ms"]))
    acc = {k: np.mean(np.array(v) >= 0.7) for k, v in res.items()}
    mean = {k: np.mean(v) for k, v in res.items()}
    print(f"ctx={context!s:5} mix={mix!s:5} other-targets accept {acc['other']:.0%} | tau x{tau_scale:<4} ref_norm={ref_norm!s:5} len={len_penalty!s:5} | accept: right {acc['right']:.0%}  "
          f"near {acc['near']:.0%}  foreign {acc['foreign']:.0%} | mean: right {mean['right']:.2f} near {mean['near']:.2f} foreign {mean['foreign']:.2f}")
    if verbose:
        print("  rejected correct:", [x for x in detail if x[0] in ("right", "other") and x[3] < 0.7])
        print("  accepted wrong:", [x for x in detail if x[0] != "right" and x[3] >= 0.7])

for ctx, mix in ((False, False), (True, False), (True, True)):
    evaluate(1.0, True, True, verbose=True, context=ctx, mix=mix)
for ts in ():
    for rn in (True, False):
        for lp in (False, True):
            evaluate(ts, rn, lp)
cfg = [float(x) if i == 0 else x == "True" for i, x in enumerate(sys.argv[1:4])] if len(sys.argv) > 3 else None
if cfg:
    evaluate(*cfg, verbose=True)
