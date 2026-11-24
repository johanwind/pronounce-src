"""Greek scoring on REAL native speech (FLEURS el_gr dev, CC-BY), through the app's Scorer.

    PRONOUNCE_LANG=el .venv/bin/python tests/el_fleurs.py [hear|voices|swap]   (needs ~/data/fleurs_el)

hear:   phone error rate of the recogniser on native speech vs espeak's phones (allophones merged):
        can it hear Greek at all?
voices: the same for each TTS voice reading the same sentences: which voice is clearest?
swap:   Scorer.analyse on native audio. Clean words should pass; words whose expected phones are
        changed to a contrasting sound (θ->t, ð->d, γ->g, χ->k, ρ->l, ι->ε ...) that the speaker
        did not say should be flagged. Reports AUC, flag rates and the calibration threshold.
"""
import csv
import difflib
import os
import random
import sys
from pathlib import Path

os.environ["PRONOUNCE_LANG"] = "el"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scoring import Scorer, tokenize, SAMPLE_RATE
import lang_el

FLEURS = Path.home() / "data/fleurs_el"
N_UTT = int(os.environ.get("N_UTT", 80))
SWAP = {"θ": "t", "ð": "d", "ɣ": "ɡ", "x": "k", "ç": "k", "r": "l", "l": "r", "z": "s", "v": "b",
        "f": "p", "ts": "t", "i": "e", "e": "i", "o": "u", "u": "o", "a": "o", "m": "n", "n": "m",
        "p": "b", "t": "d", "k": "ɡ", "s": "θ", "ʝ": "ɡ", "b": "v", "d": "ð"}
MERGE = {"ʝ": "j", "ç": "x", "c": "k", "ɟ": "ɡ", "ɾ": "r", "ɲ": "n", "ʎ": "l", "ŋ": "n",
         "ɐ": "a", "ɑ": "a", "ɛ": "e", "ɔ": "o", "ʊ": "u", "ɪ": "i", "iː": "i", "uː": "u"}  # vowel labels


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def cases():
    out, seen = [], set()
    for r in csv.reader(open(FLEURS / "dev.tsv"), delimiter="\t"):
        if r[0] in seen or any(c.isdigit() for c in r[2]) or not (FLEURS / "dev" / r[1]).exists():
            continue
        seen.add(r[0])
        out.append((r[1], r[2]))
    return out[:N_UTT]


def load(wav):
    a, sr = sf.read(FLEURS / "dev" / wav, dtype="float32")
    return resample_poly(a, SAMPLE_RATE, sr).astype(np.float32) if sr != SAMPLE_RATE else a


def decode(s, a):
    ids, out, prev = s.emissions(a).argmax(-1).tolist(), [], None
    for i in ids:
        if i != prev and i != s.blank:
            out.append(s.id2phone[i])
        prev = i
    return out


def per(s, text, a):
    exp = [MERGE.get(p, p) for _, ps in s.words_and_phones(text) for p in ps]
    got = [MERGE.get(p, p) for p in decode(s, a)]
    m = difflib.SequenceMatcher(None, exp, got, autojunk=False)
    err = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in m.get_opcodes() if op != "equal")
    subs = [(" ".join(exp[i1:i2]), " ".join(got[j1:j2])) for op, i1, i2, j1, j2 in m.get_opcodes() if op == "replace"]
    return err, len(exp), subs


def hear(s, cs):
    e = n = 0
    from collections import Counter
    c = Counter()
    for wav, text in cs:
        de, dn, subs = per(s, text, load(wav))
        e, n = e + de, n + dn
        c.update(subs)
    print(f"native speech: phone error rate {e / n:.1%} ({n} phones); most common substitutions {c.most_common(12)}", flush=True)


def voices(s, cs):
    from collections import Counter
    for v in ["joy", "rapunzelina", "rapunzelina_low", "mms"]:
        e = n = 0
        c = Counter()
        for _, text in cs[:40]:
            de, dn, subs = per(s, text, s.tts_audio(text, v))
            e, n = e + de, n + dn
            c.update(subs)
        print(f"voice {v:16} phone error rate {e / n:.1%}; {c.most_common(8)}", flush=True)


def swap(s, cs):
    rng = random.Random(0)
    pos, neg, clar, worst = [], [], [], []
    for wav, text in cs:
        a = load(wav)
        r = s.analyse(text, a, SAMPLE_RATE)
        if "error" in r:
            continue
        clar.append(r["stats"]["clarity"])
        for w in r["words"]:
            if w["score"] is not None:
                neg.append(w["score"])
                if w["score"] < 0.5:
                    worst.append(w["word"])
        # same audio, expected phones of up to 2 scored words changed to a contrasting sound
        ref = s.reference(text)
        idx = [i for i, (_, _, tr) in enumerate(ref["prons"]) if tr and any(p in SWAP for p in ref["wp"][i][1])]
        chosen = rng.sample(idx, min(2, len(idx)))
        wp = list(ref["wp"])
        for i in chosen:
            phs = list(wp[i][1])
            j = rng.choice([k for k, p in enumerate(phs) if p in SWAP])
            phs[j] = SWAP[phs[j]]
            wp[i] = (wp[i][0], phs)
        s.ref_cache[text] = dict(ref, wp=wp)  # ref scores stay those of the correct phones
        r2 = s.analyse(text, a, SAMPLE_RATE)
        s.ref_cache[text] = ref
        if "error" in r2:
            continue
        pos += [r2["words"][i]["score"] for i in chosen if r2["words"][i]["score"] is not None]
    pos, neg = np.array(pos), np.array(neg)
    print(f"AUC {auc(pos, neg):.3f}; at the app's 0.5 threshold: clean native words flagged {np.mean(neg < 0.5):.0%}, "
          f"swapped words caught {np.mean(pos < 0.5):.0%}; at 10% clean flags catches {np.mean(pos < np.quantile(neg, 0.1)):.0%} "
          f"(n swapped {len(pos)}, clean {len(neg)}); native clarity median {np.median(clar):.0f}%", flush=True)
    import scoring
    q = float(np.quantile(neg, 0.1))
    raw = 2 * q * scoring.CALIB_T if q < 0.5 else scoring.CALIB_T + (q - 0.5) * 2 * (1 - scoring.CALIB_T)
    print(f"for 10% clean flags: shown score {q:.2f} = raw {raw:.3f} (set CALIB_T to that)")
    from collections import Counter
    print("most flagged clean words:", Counter(worst).most_common(12))
    return pos, neg


if __name__ == "__main__":
    s = Scorer(device=os.environ.get("DEVICE", "cuda"), log=lambda *a: None)
    cs = cases()
    print(len(cs), "utterances", flush=True)
    for mode in sys.argv[1:] or ["hear", "voices", "swap"]:
        {"hear": hear, "voices": voices, "swap": swap}[mode](s, cs)
