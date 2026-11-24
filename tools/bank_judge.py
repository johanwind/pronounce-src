"""Stage 2 of the English reading bank: Claude (subscription CLI, not API credits) rates a
balanced sample of candidates and writes a one-line setup for each.
Output: data/bank/judged.jsonl (appends; rerunning skips passages already judged).

    .venv/bin/python tools/bank_judge.py [--per-book 100] [--batch 50] [--jobs 6]
"""
import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAND = ROOT / "data" / "bank" / "candidates.jsonl"
OUT = ROOT / "data" / "bank" / "judged.jsonl"

SYSTEM = """You curate passages for an English pronunciation app: a learner reads one passage aloud \
(about 45 words, ~20 seconds) and gets feedback. The passages are excerpts from public-domain books.

For each passage return:
- fun (1-5): would an adult learner ENJOY reading this aloud? Wit, a vivid scene, a surprising \
idea, tension, a good line of dialogue score high. Dull description, plot logistics, lists, \
dense archaic prose score low.
- standalone (true/false): does it make sense on its own, given only your one-line setup? False if \
it hinges on unexplained earlier events, starts mid-argument, or ends mid-thought.
- offensive (true/false): racial or ethnic slurs or stereotypes, demeaning content, graphic violence. \
Period attitudes that would make a modern reader wince count as true.
- setup: one short line (max 12 words) that orients the reader before reading, in present tense, \
e.g. "Holmes questions a client about her sister's strange death." No spoilers of the passage itself.
Be strict: only about a third of passages should get fun >= 4."""

SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "fun": {"type": "integer"},
                       "standalone": {"type": "boolean"}, "offensive": {"type": "boolean"},
                       "setup": {"type": "string"}},
        "required": ["id", "fun", "standalone", "offensive", "setup"]}}},
    "required": ["items"],
}


def ask(batch, model):
    prompt = "\n\n".join(f"[{c['id']}] ({c['author']}, {c['title']})\n{c['text']}" for c in batch)
    prompt += "\n\nReturn one item per passage id."
    cmd = ["claude", "-p", "--model", model, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--setting-sources", "", "--system-prompt", SYSTEM,
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600, cwd="/tmp", env=env)
    d = json.loads(p.stdout)
    if d.get("is_error"):
        raise RuntimeError(str(d.get("result"))[:300])
    out = d.get("structured_output") or json.loads(d["result"])
    return out["items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-book", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--model", default="sonnet")
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(CAND)]
    for i, c in enumerate(cands):
        c["id"] = i
    done = {json.loads(l)["text"] for l in open(OUT)} if OUT.exists() else set()
    by_book = defaultdict(list)
    for c in cands:
        by_book[c["title"]].append(c)
    random.seed(1)
    pick = []
    for title, cs in by_book.items():
        pick += random.sample(cs, min(args.per_book, len(cs)))
    pick = [c for c in pick if c["text"] not in done]
    random.shuffle(pick)  # mix books inside each batch so ratings are comparable
    batches = [pick[i:i + args.batch] for i in range(0, len(pick), args.batch)]
    print(f"{len(pick)} passages in {len(batches)} batches")
    byid = {c["id"]: c for c in pick}
    with ThreadPoolExecutor(args.jobs) as ex, open(OUT, "a") as f:
        futs = {ex.submit(ask, b, args.model): b for b in batches}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                items = fut.result()
            except Exception as e:  # noqa: BLE001
                print("batch failed:", e)
                continue
            for it in items:
                c = byid.get(it["id"])
                if c:
                    f.write(json.dumps({**c, **{k: it[k] for k in ("fun", "standalone", "offensive", "setup")}},
                                       ensure_ascii=False) + "\n")
            f.flush()
            print(f"batch {n}/{len(batches)} done", flush=True)


if __name__ == "__main__":
    main()
