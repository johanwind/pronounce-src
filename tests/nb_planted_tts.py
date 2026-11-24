"""Norwegian: planted NON-WORD errors read by a Piper voice, scored against the right text.

tests/nb_fleurs.py can only swap in real words (the audio is a native reading), but learner
errors mostly make non-words ("sjøpte" for "kjøpte"), and the Norwegian recogniser has learnt
the vocabulary: it tends to hear the real word anyway. This measures that bias. Synthetic
speech, so treat the numbers as relative (between models / settings), not as real-learner rates.

    .venv/bin/python tests/nb_planted_tts.py          (PHONEME_MODEL=... to compare recognisers)
"""
import os
import random
import sys
from pathlib import Path

os.environ["PRONOUNCE_LANG"] = "nb"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import lang_nb
import paragraphs_nb
from scoring import Scorer, tokenize

VOICE = os.environ.get("VOICE", "talesyntese")
# learner-typical swaps that clearly change the sound; only applied where the result is not a
# word in the lexicon. Not o->å (bodde/bådde sound alike) or unstressed e->i (dagen/dagin).
PLANT = [("kj", "sj"), ("sj", "s"), ("y", "i"), ("ø", "e"), ("æ", "e"), ("u", "y"), ("rt", "t"), ("rn", "n")]


def plant(text, lex, rng, n=3):
    words = tokenize(text)[0]
    cands = []
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in lang_nb.FUNCTION_WORDS or len(lw) < 3 or not lex.lookup(lw):
            continue
        for a, b in PLANT:
            if a in lw:
                nw = lw.replace(a, b, 1)
                if not lex._get(nw):
                    cands.append((i, nw))
                    break
    chosen = dict(rng.sample(cands, min(n, len(cands))))
    return " ".join(chosen.get(i, w) for i, w in enumerate(words)) + ".", chosen


s = Scorer(device=os.environ.get("DEVICE", "cuda"), log=lambda *a: None)
lex, rng = lang_nb.Lexicon(), random.Random(0)
caught, clean_flags, n_clean, examples = [], 0, 0, []
for text in paragraphs_nb.BANK:
    bad, chosen = plant(text, lex, rng)
    a = s.tts_audio(bad, VOICE)
    r = s.analyse(text, a)
    if "error" in r:
        print("skip", r["error"])
        continue
    for i, w in enumerate(r["words"]):
        if w["score"] is None:
            continue
        if i in chosen:
            caught.append(w["score"] < 0.5)
            examples.append((w["word"], chosen[i], w.get("heard_as"), round(w["score"], 2)))
        else:
            n_clean += 1
            clean_flags += w["score"] < 0.5
print(f"model {os.environ.get('PHONEME_MODEL', lang_nb.RECOGNIZER)}, voice {VOICE}: planted non-word errors caught "
      f"{np.mean(caught):.0%} of {len(caught)}; other words flagged {clean_flags / n_clean:.1%} of {n_clean}")
print("planted (word, said, heard_as, score):", examples)
