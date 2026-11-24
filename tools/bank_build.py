"""Stage 3 of the English reading bank: pick the final passages and precompute everything the
app would otherwise compute per paragraph at run time (reference-voice scores, duration,
sound-area densities), so a CPU-only server needs no TTS or reference alignment per text.
Output: bank_<lang>.json (resumable: passages already in it are kept).

    reserve --mem 6G --cpus 4 -- .venv/bin/python tools/bank_build.py [--per-book 60]
    PRONOUNCE_LANG=nb .venv/bin/python tools/bank_build.py   # paragraphs_nb.BANK (el: paragraphs_el.BANK)
"""
import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
JUDGED = ROOT / "data" / "bank" / "judged.jsonl"
LANG = os.environ.get("PRONOUNCE_LANG", "en")
OUT = ROOT / f"bank_{LANG}.json"


def fix_quotes(t):
    if t.count("“") > t.count("”"):
        t += "”"
    return t


def select(per_book):
    if LANG in ("nb", "el"):  # the hand-written passages
        import importlib
        return [dict(text=t) for t in importlib.import_module(f"paragraphs_{LANG}").BANK]
    rows = [json.loads(l) for l in open(JUDGED)]
    good = [r for r in rows if r["fun"] >= 4 and r["standalone"] and not r["offensive"]]
    seen, by_book = set(), defaultdict(list)
    for r in good:
        if r["text"] not in seen:
            seen.add(r["text"])
            by_book[r["title"]].append(r)
    random.seed(2)
    out = []
    for title, rs in by_book.items():
        random.shuffle(rs)
        rs.sort(key=lambda r: -r["fun"])  # stable: random order within each fun level
        out += rs[:per_book]
    random.shuffle(out)
    return [dict(text=fix_quotes(r["text"]), author=r["author"], title=r["title"], setup=r["setup"],
                 fun=r["fun"]) for r in out]


def rescore(part, parts):
    """Same passages, new reference scores: for a different recogniser (the int8 hosted model).
    Run the parts in parallel, then: tools/bank_build.py --merge PARTS."""
    from scoring import Scorer
    bank = json.load(open(OUT))[part::parts]
    s = Scorer()
    if os.environ.get("PHONEME_MODEL_FILE"):
        s.device = "cpu"  # int8 runs on CPU only; the reference voices still use the GPU
    out_path = OUT.with_suffix(f".part{part}.json")
    # resumable: passages already rescored in out_path are kept (it is saved every 25)
    done = {b["text"]: b for b in json.load(open(out_path))} if out_path.exists() else {}
    t0, n_new = time.time(), 0
    for i, b in enumerate(bank):
        if b["text"] in done:
            bank[i] = done[b["text"]]
            continue
        ref = s.reference(b["text"])
        b["ref"].update(scores=[round(x, 4) for x in ref["scores"]], duration=round(ref["duration"], 3))
        s.ref_cache.clear()
        done[b["text"]] = b
        n_new += 1
        if n_new % 25 == 0:
            json.dump(list(done.values()), open(out_path, "w"), ensure_ascii=False)
            print(f"part {part}: {len(done)}/{len(bank)}  {(time.time() - t0) / n_new:.2f} s each", flush=True)
    json.dump(bank, open(out_path, "w"), ensure_ascii=False)
    print(f"part {part}: wrote {len(bank)} to {out_path}")


def merge_parts(parts):
    """Put rescored parts back in the bank's original order."""
    bank = json.load(open(OUT))
    new = {}
    for k in range(parts):
        for b in json.load(open(OUT.with_suffix(f".part{k}.json"))):
            new[b["text"]] = b
    assert len(new) == len(bank), (len(new), len(bank))
    json.dump([new[b["text"]] for b in bank], open(OUT, "w"), ensure_ascii=False)
    print(f"merged {len(new)} rescored passages into {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-book", type=int, default=60)
    ap.add_argument("--rescore", nargs=2, type=int, metavar=("PART", "PARTS"),
                    help="recompute the references of passages already in the bank (after changing the "
                         "recogniser, e.g. PHONEME_MODEL_FILE), part PART of PARTS; writes bank_<lang>.partN.json")
    args = ap.parse_args()
    if args.rescore:
        return rescore(*args.rescore)
    import coach
    from scoring import Scorer
    from stats import AREAS

    bank = select(args.per_book)
    done = {b["text"]: b for b in json.load(open(OUT))} if OUT.exists() else {}
    print(f"{len(bank)} passages, {len(done)} already built", flush=True)
    s = Scorer()
    if os.environ.get("PHONEME_MODEL_FILE"):
        s.device = "cpu"  # int8 runs on CPU only (as in rescore)
    out, t0 = [], time.time()
    for i, b in enumerate(bank):
        if b["text"] in done:
            out.append(done[b["text"]])
            continue
        ref = s.reference(b["text"])
        if not ref["scores"] or ref["duration"] is None:
            print("no reference, skipped:", b["text"][:60])
            continue
        b["areas"] = {a: round(coach.area_density(s, b["text"], {a}), 4) for a in AREAS}
        b["ref"] = dict(wp=ref["wp"], prons=ref["prons"], scores=[round(x, 4) for x in ref["scores"]],
                        duration=round(ref["duration"], 3))
        if LANG == "nb":  # word scoring also needs these (Scorer.reference)
            b["ref"].update(extra=ref["extra"], silent=ref["silent"],
                            ratio=[None if x is None else round(x, 4) for x in ref["ratio"]] if ref["ratio"] else None)
        out.append(b)
        s.ref_cache.clear()  # emissions are large; they are not needed again
        if len(out) % 25 == 0:
            json.dump(out, open(OUT, "w"), ensure_ascii=False)
            print(f"{len(out)}/{len(bank)}  {(time.time() - t0) / (i + 1):.2f} s each", flush=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False)
    print(f"wrote {len(out)} passages to {OUT}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--merge"]:
        merge_parts(int(sys.argv[2]))
    else:
        main()
