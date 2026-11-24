"""Norwegian scoring on REAL native speech (FLEURS nb_no dev, CC-BY), through the app's
own Scorer.analyse.

Clean words: native readings scored against their own transcript should pass.
Swapped words: the same audio scored against a text where one word is replaced by a
real minimal-pair word (ny->ni, søt->set, bok->båk ...) that the NB Uttale lexicon says
sounds different: the scorer should flag it, since the speaker did not say that.
Caveat: the swaps are drawn from the same kinds of confusion the word scorer checks
(lang_nb.CONFUSIONS), so the catch rate is optimistic; the clean flag rate is the fair number.

Also scores the readings saved in data/nb (if any), listing what gets flagged there.

    .venv/bin/python tests/nb_fleurs.py      (needs ~/data/fleurs_nb; N_UTT=20 for a quick run)

Result: see README.
"""
import csv
import json
import os
import random
import sys
from pathlib import Path

os.environ["PRONOUNCE_LANG"] = "nb"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
from scoring import Scorer, tokenize
import lang_nb

FLEURS = Path.home() / "data/fleurs_nb"
USER = Path(__file__).resolve().parent.parent / "data/nb"
N_UTT = int(os.environ.get("N_UTT", 200))
SWAPS = [("kj", "sj"), ("sj", "kj"), ("y", "i"), ("i", "y"), ("ø", "e"), ("e", "ø"), ("æ", "e"),
         ("e", "æ"), ("o", "å"), ("å", "o"), ("u", "o"), ("o", "u"), ("y", "ø"), ("ø", "y")]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def make_cases(lex):
    rows = list(csv.reader(open(FLEURS / "dev.tsv"), delimiter="\t"))
    seen, cases = set(), []
    rng = random.Random(0)
    for r in rows:
        text = r[2]
        if r[0] in seen or any(c.isdigit() for c in text) or not (FLEURS / "dev" / r[1]).exists():
            continue
        words = tokenize(text)[0]
        swaps = {}
        for i, w in enumerate(words):
            lw = w.lower()
            if lw in lang_nb.FUNCTION_WORDS or len(lw) < 3 or not lex.lookup(lw):
                continue
            for a, b in SWAPS:
                if a in lw:
                    cand = lw.replace(a, b, 1)
                    if lex._get(cand) and lex._get(cand)[0] != lex.lookup(lw)[0]:
                        swaps[i] = cand
                        break
        if not swaps:
            continue
        seen.add(r[0])
        cases.append((r[1], text, dict(rng.sample(sorted(swaps.items()), min(2, len(swaps))))))
        if len(cases) >= N_UTT:
            break
    return cases


def key(w):
    """Sort key: the raw ratio where there is one (finer than the clipped score)."""
    r = w.get("ratio")
    return w["score"] if r is None else 0.5 + 0.5 * r / lang_nb.LR_SCALE


s = Scorer(device=os.environ.get("DEVICE", "cuda"), log=lambda *a: None)
cases = make_cases(lang_nb.Lexicon())
print(len(cases), "utterances;", sum(len(c[2]) for c in cases), "swaps, e.g.",
      [(tokenize(t)[0][i], w) for _, t, sw in cases[:4] for i, w in sw.items()], flush=True)
pos, neg, clean_clarity, worst, n_words = [], [], [], [], 0
for wav, text, swaps in cases:
    a, sr = sf.read(FLEURS / "dev" / wav, dtype="float32")
    words = tokenize(text)[0]
    r = s.analyse(text, a, sr)
    if "error" in r:
        print("skip", r["error"])
        continue
    clean_clarity.append(r["stats"]["clarity"])
    for w in r["words"]:
        n_words += w["word"].lower() not in lang_nb.FUNCTION_WORDS
        if w["score"] is not None:
            neg.append(key(w))
            if w["score"] < 0.5:
                worst.append((w["word"], w.get("heard_as")))
    expected = " ".join(swaps.get(i, w) for i, w in enumerate(words)) + "."
    r2 = s.analyse(expected, a, sr)
    if "error" in r2:
        continue
    for i in swaps:
        if r2["words"][i]["score"] is not None:
            pos.append(key(r2["words"][i]))
pos, neg = np.array(pos), np.array(neg)
print(f"AUC {auc(pos, neg):.3f}; clean native words flagged {np.mean(neg < 0.5):.1%}, swapped words caught "
      f"{np.mean(pos < 0.5):.0%} (n swapped {len(pos)}, clean {len(neg)} of {n_words} content words scored); "
      f"native clarity median {np.median(clean_clarity):.0f}%")
print("clean score quantiles 5/10/25/50%:", np.round(np.quantile(neg, [0.05, 0.1, 0.25, 0.5]), 2))
print("flagged clean words (word, sounded like):", worst[:40])

if (USER / "history.jsonl").exists():
    n = flagged = 0
    out = []
    for line in open(USER / "history.jsonl"):
        d = json.loads(line)
        f = USER / "recordings" / f"{d['id']}.flac"
        if not f.exists():
            continue
        a, sr = sf.read(f, dtype="float32")
        r = s.analyse(d["text"], a.mean(1) if a.ndim > 1 else a, sr)
        if "error" in r:
            continue
        for w in r["words"]:
            if w["score"] is not None:
                n += 1
                if w["score"] < 0.5:
                    flagged += 1
                    out.append((w["word"], w.get("heard_as"), "slip" if w.get("slip") else round(w["score"], 2)))
    print(f"saved readings (data/nb): {flagged}/{n} words flagged:", out)
