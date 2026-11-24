"""Sanity checks for scoring.py using synthetic speech.

- other TTS voices reading the paragraph should score high everywhere
- a reading of an altered text (planted mispronunciations) should be flagged at
  exactly those words
- a different paragraph should score low
- compare the reference-score distribution with the old practice.py method
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import Scorer  # noqa: E402

P = ("I think the weather will be very warm this weekend, so we should visit the "
     "village by the river. My brother thought it would rain, but the forecast "
     "says there is a thirty percent chance at worst.")
# planted errors: think->sink, very->wery, village->willage, thought->taught,
# thirty->dirty, weather->wedder
P_BAD = ("I sink the wetter will be wery warm this weekend, so we should visit the "
         "willage by the river. My brother taught it would rain, but the forecast "
         "says there is a dirty percent chance at worst.")
PLANTED = {1: "think", 3: "weather", 6: "very", 15: "village", 21: "thought", 32: "thirty"}
OTHER = ("Fresh bread and strong coffee are the best way to start a cold morning "
         "before a long day of meetings in the city center downtown.")


def summary(name, res):
    words = res["words"]
    flagged = [(i, w["word"], w["score"]) for i, w in enumerate(words)
               if w["score"] is not None and w["score"] < 0.5]
    st = res["stats"]
    print(f"{name:28s} clarity={st['clarity']:3d}  clear={st['words_clear']}/{st['words_scored']}"
          f"  wpm={st['wpm']}  flagged={[(i, w, round(s, 2)) for i, w, s in flagged]}")
    return flagged


def old_method_ref(s, text, audio):
    """practice.py's scoring: g2p_en + ARPABET map, exp(mean log p) over the span."""
    from g2p_en import G2p
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from practice import text_to_tokens
    toks, labels = text_to_tokens(text, G2p(), s.vocab)
    em = s.emissions(audio)
    ali, sc = torchaudio.functional.forced_align(
        em[None], torch.tensor([toks], dtype=torch.int32, device=em.device), blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], sc[0])
    return labels, [math.exp(x.score) for x in spans]


def main():
    s = Scorer()
    t0 = time.time()
    ref = s.reference(P)
    print(f"reference took {time.time()-t0:.1f}s; phones={len(ref['scores'])}")
    rs = np.array(ref["scores"])
    print(f"NEW ref scores: median={np.median(rs):.2f}  <0.25 (unreliable): {(rs<0.25).mean():.0%}"
          f"  <0.5: {(rs<0.5).mean():.0%}")
    flat = [p for _, ps in ref["wp"] for p in ps]
    low = [(flat[i], round(rs[i], 2)) for i in np.argsort(rs)[:10]]
    print("  lowest:", low)

    a = s.tts_audio(P, "am_michael")
    labels, old = old_method_ref(s, P, a)
    old = np.array(old)
    print(f"OLD ref scores (am_michael): median={np.median(old):.2f}  <0.2: {(old<0.2).mean():.0%}"
          f"  <0.5: {(old<0.5).mean():.0%}")

    ok = True
    neg = []  # word scores from clean readings
    pos = []  # planted-error word scores
    for v in ["am_adam", "am_liam", "af_bella", "bf_emma", "bm_george"]:
        t0 = time.time()
        res = s.analyse(P, s.tts_audio(P, v))
        fl = summary(f"clean {v} ({time.time()-t0:.1f}s)", res)
        neg += [w["score"] for w in res["words"] if w["score"] is not None]
        if v.startswith("a") and len(fl) > 2:
            ok = False
    for v in ["am_adam", "af_bella", "am_liam", "bf_emma"]:
        res = s.analyse(P, s.tts_audio(P_BAD, v))
        fl = summary(f"planted errors {v}", res)
        pos += [res["words"][i]["score"] for i in PLANTED]
        idx = {i for i, _, _ in fl}
        hit = [PLANTED[i] for i in PLANTED if i in idx]
        miss = [PLANTED[i] for i in PLANTED if i not in idx]
        false = [w for i, w, _ in fl if i not in PLANTED]
        print(f"   caught {len(hit)}/{len(PLANTED)} {hit}  missed {miss}  false alarms {false}")
        for i in PLANTED:
            w = res["words"][i]
            print(f"     {w['word']:9s} {w['score']:.2f}  bad={w['bad']}")
    pos_a, neg_a = np.array(pos), np.array(neg)
    auc = ((pos_a[:, None] < neg_a[None]).mean() + 0.5 * (pos_a[:, None] == neg_a[None]).mean())
    print(f"SEPARATION AUC (planted vs clean words) = {auc:.3f};  clean words <0.5: {(neg_a<0.5).mean():.1%};"
          f"  planted <0.5: {(pos_a<0.5).mean():.0%}")
    res = s.analyse(P, s.tts_audio(OTHER, "am_adam"))
    summary("wrong paragraph", res)
    print("PASS" if ok else "FAIL: clean US voices flagged too much")


if __name__ == "__main__":
    main()
