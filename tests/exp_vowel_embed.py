"""Experiment: can wav2vec2 hidden states of a vowel segment identify the vowel across
speakers? Leave-one-voice-out nearest-centroid accuracy, per layer."""
import sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import Scorer, trim_silence, SAMPLE_RATE

WORDS = {
 "ʊ": ["book", "good", "put", "look", "foot", "wood", "cook", "hood", "full", "push"],
 "uː": ["food", "boot", "pool", "moon", "suit", "shoot", "cool", "true", "room", "fool"],
 "ɪ": ["bit", "pit", "sit", "kid", "fill", "hid", "lick", "big", "ship", "fish"],
 "iː": ["beat", "seat", "feel", "keep", "heed", "sheep", "leave", "meet", "feed", "peel"],
 "ʌ": ["but", "cup", "luck", "hut", "bud", "sun", "cut", "buck", "dust", "fun"],
 "ɑː": ["hot", "pot", "cot", "rock", "lot", "top", "stop", "box", "doll", "shop"],
 "ɔː": ["thought", "caught", "law", "saw", "talk", "ball", "walk", "dawn", "fought", "hall"],
 "oʊ": ["boat", "go", "coat", "road", "home", "note", "bone", "goal", "soap", "pole"],
 "ɛ": ["bed", "pet", "set", "head", "men", "said", "bet", "red", "neck", "well"],
 "æ": ["bat", "cat", "bad", "man", "hat", "back", "sad", "map", "cap", "fan"],
}
VOICES = ["am_michael", "am_adam", "am_eric", "am_liam", "af_heart", "af_bella", "af_nicole", "af_sarah"]
s = Scorer(log=lambda *a: None)
LAYERS = [4, 6, 8, 10, 12, 14]

def feats(audio, word, vowel):
    wp = s.words_and_phones(word)
    phones = [p for _, ps in wp for p in ps]
    if vowel not in phones:
        return None
    a, _ = trim_silence(audio)
    res = s.align(a, wp)
    if not res:
        return None
    t = res[phones.index(vowel)]
    lo, hi = int(t["seg"][0] / 0.02), int(t["seg"][1] / 0.02)
    with s.lock, torch.no_grad():
        x = s.fe(a, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(s.device)
        out = s.model(x, output_hidden_states=True)
    hs = out.hidden_states
    hi = max(hi, lo + 1)
    return {L: hs[L][0, lo:hi].mean(0).float().cpu().numpy() for L in LAYERS}

t0 = time.time()
data = []  # (vowel, voice, feats)
for v, words in WORDS.items():
    for w in words:
        for vo in VOICES:
            f = feats(s.tts_audio(w + ".", vo), w, v)
            if f is not None:
                data.append((v, vo, w, f))
print(f"{len(data)} samples in {time.time()-t0:.0f}s")
vow = list(WORDS)
def norm(X): return X / np.linalg.norm(X, axis=1, keepdims=True)
variants = {f"L{L}": norm(np.stack([d[3][L] for d in data])) for L in LAYERS}
variants["L6-12 mean"] = norm(np.mean([np.stack([norm(np.stack([d[3][L] for d in data]))]) for L in (6, 8, 10, 12)], 0)[0])
variants["L8+L12 concat"] = norm(np.concatenate([variants["L8"], variants["L12"]], 1))
for L, X in variants.items():
    y = np.array([d[0] for d in data]); vo = np.array([d[1] for d in data])
    correct = 0; conf = {}
    for test in VOICES:
        tr, te = vo != test, vo == test
        cents = np.stack([X[tr & (y == v)].mean(0) for v in vow]); cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        pred = np.array(vow)[(X[te] @ cents.T).argmax(1)]
        correct += (pred == y[te]).sum()
        for a, b in zip(y[te], pred):
            if a != b: conf[(a, b)] = conf.get((a, b), 0) + 1
    per = {v: 1 - sum(n for (a, b), n in conf.items() if a == v) / (y == v).sum() for v in vow}
    print(f"layer {L}: LOVO accuracy {correct/len(y):.1%}  per-vowel " + " ".join(f"{v}={a:.0%}" for v, a in per.items())
          + f"  ʊ confusions: {[(b, n) for (a, b), n in conf.items() if a == 'ʊ']}")
np.save("/tmp/claude-1000/vowel_feats_meta.npy", np.array([(d[0], d[1], d[2]) for d in data]))
