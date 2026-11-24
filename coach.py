"""Coaching: pick the 3 weakest areas and the next paragraph.

Haiku (via the `claude` CLI, i.e. the subscription login) writes the tips and a fresh
paragraph that drills the focus sounds. Without it, everything still works from the
local statistics: the areas come from stats.top_focus and the next paragraph is the
bank passage densest in those sounds.
"""

import json
import os
import shutil
import subprocess

from stats import AREAS, FINAL, LANG, label, phone_areas

CLI_MODEL = "haiku"

SYSTEM = """You are a warm, precise English pronunciation coach (American English target).
You get machine measurements of a learner reading a paragraph aloud. The scorer compares
each sound with clear synthetic speech: 1.0 = as clear as the reference, below 0.5 = the
model heard something else. "confusions" like "θ→s" mean expected θ, heard s.
The measurements are noisy: trust patterns across several words, not single slips, and
ignore areas with very few samples. Pick the THREE most useful things to work on next.
For each give: a short name, the evidence (which words), one concrete physical tip
(tongue/lips/jaw/voicing), and a mini drill (a few words or a short phrase).
Then write TWO next paragraphs for them to read, on different topics: natural, interesting prose
of 55-75 words each, no digits, no tongue-twister nonsense, dense in the three focus sounds."""

if LANG == "nb":
    from lang_nb import COACH_SYSTEM as SYSTEM
elif LANG == "el":
    from lang_el import COACH_SYSTEM as SYSTEM

SCHEMA = {
    "type": "object",
    "properties": {
        "focus": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "area": {"type": "string"},
                    "evidence": {"type": "string"},
                    "tip": {"type": "string"},
                    "drill": {"type": "string"},
                },
                "required": ["area", "evidence", "tip", "drill"],
            },
        },
        "summary": {"type": "string", "description": "one or two encouraging sentences on this reading"},
        "next_paragraphs": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}},
    },
    "required": ["focus", "summary", "next_paragraphs"],
}


def build_report(result, table, history):
    words = result["words"]
    worst = sorted((w for w in words if w["score"] is not None and w["score"] < 0.5),
                   key=lambda w: w["score"])[:12]
    recent = [dict(clarity=h["stats"].get("clarity"), wpm=h["stats"].get("wpm"),
                   focus=[label(a) for a in h.get("focus_areas") or []])
              for h in history[-6:]]
    return dict(
        stats={k: v for k, v in result["stats"].items() if k != "hesitations"},
        hesitations=result["stats"].get("hesitations", [])[:6],
        areas=[{k: (label(t[k]) if k == "area" else t[k]) for k in ("area", "mean", "n", "weak", "words", "confusions")}
               for t in table[:10]],
        worst_words=[dict(word=w["word"], bad=w["bad"],
                          **({"misread_as" if w.get("slip") else "sounded_like": w["heard_as"]} if w.get("heard_as") else {}))
                     for w in worst],
        recent_attempts=recent,
    )


def ask_haiku(report, timeout=90):
    """Ask Haiku through the claude CLI, i.e. the Claude subscription login (not API credits)."""
    prompt = ("Measurements (JSON):\n" + json.dumps(report, ensure_ascii=False)
              + "\n\nReply with the JSON object only.")
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found (the coach uses your Claude subscription login)")
    return _cli(prompt, timeout)


def _cli(prompt, timeout):
    cmd = ["claude", "-p", "--model", CLI_MODEL, "--tools", "", "--strict-mcp-config",
           "--no-session-persistence", "--setting-sources", "", "--system-prompt", SYSTEM,
           "--output-format", "json", "--json-schema", json.dumps(SCHEMA)]
    # without the API key the CLI uses the subscription login, not API credits
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                       cwd="/tmp", env=env)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError((p.stderr or p.stdout)[-300:])
    if d.get("is_error"):
        raise RuntimeError(str(d.get("result"))[:300])
    out = d.get("structured_output")
    if out is None:
        out = json.loads(d["result"])
    return out


# ---------- offline fallback ----------
def drill_phone(row):
    """Which sound the drill page should open for a focus area."""
    if LANG != "en" or os.environ.get("PRONOUNCE_HOSTED") == "1":
        return None  # the drill is English-only, and not on the hosted version
    from drill import DRILL_SETS
    for c in row.get("confusions", []):  # "ʊ→ɪ": the sound that actually went wrong
        ph = c.split("→")[0]
        if ph in DRILL_SETS:
            return ph
    for ph in sorted(AREAS[row["area"]][0]):
        if ph in DRILL_SETS:
            return ph
    return None


EVIDENCE = {
    "en": dict(heard="heard", earlier="in earlier readings",
               lowest="No clear errors, but your lowest area at {pct}%. Could be crisper."),
    "nb": dict(heard="hørt", earlier="i tidligere lesinger",
               lowest="Ingen tydelige feil, men dette er ditt svakeste område ({pct}%). Kan bli skarpere."),
    "el": dict(heard="ακούστηκε", earlier="σε προηγούμενες αναγνώσεις",
               lowest="Κανένα σαφές λάθος, αλλά αυτός είναι ο πιο αδύναμος τομέας σου ({pct}%). Μπορεί να γίνει πιο καθαρός."),
}


def local_focus(focus_rows):
    ev = EVIDENCE.get(LANG, EVIDENCE["en"])
    out = []
    for t in focus_rows:
        if not t["weak"] and t["mean"] >= 0.9:
            continue  # a filler area that is already clear is not worth a card (the page says so)
        conf = f" ({ev['heard']} {', '.join(t['confusions'])})" if t["confusions"] else ""
        if t["weak"]:
            evidence = (", ".join(t["words"]) or ev["earlier"]) + conf
        else:
            evidence = ev["lowest"].format(pct=round(100 * t["mean"]))
        out.append(dict(
            area=label(t["area"]),
            drill_phone=drill_phone(t),
            evidence=evidence,
            tip=t["how"] or "",
            drill=t["examples"],
        ))
    return out


def area_density(scorer, text, areas):
    """Fraction of phones in text that belong to the given areas."""
    import paragraphs
    pre = paragraphs.META.get(text, {}).get("areas")
    if pre:  # precomputed for the bank (tools/bank_build.py)
        return sum(pre.get(a, 0.0) for a in areas)
    wp = scorer.words_and_phones(text)
    phones = [p for _, ps in wp for p in ps]
    if not phones:
        return 0.0
    hit = 0
    for _, ps in wp:
        pa = phone_areas(ps)
        for i, p in enumerate(ps):
            if pa[i] in areas:
                hit += 1
            elif FINAL in areas and i == len(ps) - 1 and len(ps) > 1:
                hit += 1
    return hit / len(phones)


def pick_bank_paragraph(scorer, bank, areas, recent_texts, read_texts=()):
    """Fallback when the coach has no paragraph ready: a random pick among the paragraphs
    densest in the focus areas, favouring ones read less often (always taking the single
    densest one cycled through the same few paragraphs)."""
    import random
    from collections import Counter
    candidates = [t for t in bank if t not in recent_texts] or bank
    if areas:
        candidates = sorted(candidates, key=lambda t: area_density(scorer, t, set(areas)), reverse=True)
        candidates = candidates[:max(6, len(candidates) // 3)]
    reads = Counter(read_texts)
    return random.choices(candidates, [1 / (1 + reads[t]) ** 2 for t in candidates])[0]


def valid_area(name):
    return name in AREAS
