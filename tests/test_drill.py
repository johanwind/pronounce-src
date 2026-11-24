"""Drill sanity: correct readings score high; wrong vowels land in the right place on
the vowel chart and get the right direction hint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scoring import Scorer
from drill import Drill

s = Scorer(log=lambda *a: None)
d = Drill(s, log=print); d.bank.ready.wait()
cases = [  # (target word, what is actually said, expectation)
    ("book", "book", "good"), ("good", "good", "good"), ("put", "put", "good"), ("look", "look", "good"),
    ("book", "bick", "front"), ("put", "pit", "front"), ("look", "Luke", "uu"), ("full", "fool", "uu"),
    ("book", "buck", "open"), ("food", "food", "good"), ("food", "feed", "front"), ("good", "gid", "front"), ("book", "boot", "uu"),
    ("think", "think", "good"),
]
fails = 0
for word, said, exp in cases:
    ref = d.reference(word)
    a = s.tts_audio(said + ".", "am_liam")  # a voice not used as reference
    r = d.attempt(word, None, a, 16000)
    pt = r["point"]; c = ref["center"]
    ok = {"good": r["score"] >= 0.7,
          "front": r["score"] < 0.5 and pt and pt[0] < c[0] - 0.12,
          "uu": r["score"] < 0.5 and pt and pt[0] > c[0] and pt[1] < c[1],
          "open": r["score"] < 0.5 and pt and pt[1] > c[1] + 0.12,
          "bad": r["score"] < 0.5}[exp]
    fails += not ok
    print(f"{'OK ' if ok else 'BAD'} {word:6s} said {said:6s} target={ref['target']} score={r['score']:.2f} "
          f"point={pt and tuple(round(x,2) for x in pt)} center={c and tuple(round(x,2) for x in c)} "
          f"refscore={ref['ref_score']:.2f} heard={sorted(r['dist'].items(), key=lambda kv:-kv[1])[:3]} hint={r['hint']}")
print("refs:", {w: [(v['voice'], round(v['score'],2), v['point'] and tuple(round(x,2) for x in v['point'][:2])) for v in d.reference(w)['voices']] for w in ["book", "put"]})
print("FAILS:", fails)
