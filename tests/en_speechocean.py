"""English scoring on REAL learner speech: speechocean762 (Apache-2.0, Mandarin-L1 adults
and children reading short sentences, each phone scored 0-2 by 5 experts).

A word counts as mispronounced if any phone's mean expert score is < 1 (most experts
heard it wrong), clean if every phone is >= 1.8. We run the app's Scorer.analyse and
report AUC (word score separates the two), flag rates at the app's 0.5 threshold, and
the correlation of word scores with the experts' 0-10 word accuracy.

    .venv/bin/python tests/en_speechocean.py [w2v|w2v_int8|zipa ...]      (N_UTT=600 by default)
w2v_int8: the recogniser int8-quantised on CPU (tools/quantize.py; INT8_FILE, default data/en_int8.pt),
for the references too, as the hosted app would run it.
Needs ~/data/speechocean762/test.parquet (hf mispeech/speechocean762).
"""
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scoring import Scorer, tokenize

N_UTT = int(os.environ.get("N_UTT", 600))
ADULTS = os.environ.get("ADULTS") == "1"


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] < neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def load():
    t = pq.read_table(Path.home() / "data/speechocean762/test.parquet")
    rows = t.to_pylist()
    if ADULTS:
        rows = [r for r in rows if r["age"] >= 16]
    return rows[:N_UTT]


def evaluate(s, label, rows):
    bad, clean, corr_ours, corr_human, utt_ours, utt_human = [], [], [], [], [], []
    for r in rows:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        text = r["text"].lower()
        words = tokenize(text)[0]
        if len(words) != len(r["words"]):
            continue
        res = s.analyse(text, a, sr)
        if "error" in res:
            continue
        if res["stats"].get("clarity") is not None:
            utt_ours.append(res["stats"]["clarity"])
            utt_human.append(r["accuracy"])
        for w, h in zip(res["words"], r["words"]):
            if w["score"] is None:
                continue
            pa = h["phones-accuracy"]
            corr_ours.append(w["score"])
            corr_human.append(h["accuracy"])
            if min(pa) < 1.0:
                bad.append(w["score"])
            elif min(pa) >= 1.8:
                clean.append(w["score"])
    bad, clean = np.array(bad), np.array(clean)
    print(f"{label:5} AUC {auc(bad, clean):.3f} | at 0.5: clean flagged {np.mean(clean < 0.5):.0%}, mispronounced flagged "
          f"{np.mean(bad < 0.5):.0%} | at 10% clean flags catches {np.mean(bad < np.quantile(clean, 0.1)):.0%} | "
          f"word r={np.corrcoef(corr_ours, corr_human)[0, 1]:.2f} utt r={np.corrcoef(utt_ours, utt_human)[0, 1]:.2f} "
          f"(n bad {len(bad)}, clean {len(clean)})", flush=True)
    for q in (0.1, 0.15, 0.2):
        thr = np.quantile(clean, q)
        print(f"      threshold {thr:.2f} flags {q:.0%} of clean words, catches {np.mean(bad < thr):.0%} of mispronounced", flush=True)
    return clean, bad


def cache_emissions(s, name):
    """Memoise the recogniser per audio clip on disk, so re-runs only redo the scoring."""
    import hashlib
    d = Path.home() / "data/speechocean762/cache" / name
    d.mkdir(parents=True, exist_ok=True)
    inner = s.emissions

    def emissions(audio):
        a = np.ascontiguousarray(np.asarray(audio, np.float32))
        f = d / (hashlib.sha1(a.tobytes()).hexdigest()[:16] + ".npy")
        if f.exists():
            return __import__("torch").from_numpy(np.load(f))
        em = inner(a).cpu()
        np.save(f, em.numpy())
        return em
    s.emissions = emissions


rows = load()
print(len(rows), "utterances", "(adults only)" if ADULTS else "", flush=True)
s = Scorer(log=lambda *a: None)
for backend in sys.argv[1:] or ["w2v"]:
    if backend == "zipa":
        from scoring import use_zipa
        use_zipa(s)
    if backend == "w2v_int8":
        import torch
        s.model = torch.load(os.environ.get("INT8_FILE", "data/en_int8.pt"), weights_only=False).eval()
        s.device = "cpu"
    cache_emissions(s, backend)
    s.ref_cache = {}
    clean, bad = evaluate(s, backend, rows)
    np.savez(Path.home() / f"data/speechocean762/scores_{backend}{'_adults' if ADULTS else ''}.npz", clean=clean, bad=bad)
