"""Turn per-phone scores into learner-facing focus areas, and keep history."""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

LANG = os.environ.get("PRONOUNCE_LANG", "en")
_default_data = Path(__file__).resolve().parent / "data"
DATA = Path(os.environ.get("PRONOUNCE_DATA", _default_data / LANG if LANG != "en" else _default_data))
HISTORY = DATA / "history.jsonl"

# area name -> (phones, example words, one-line description)
AREAS = {
    "th sounds": ({"θ", "ð"}, "think, this, weather", "tongue tip lightly between the teeth, air flows"),
    "v and w": ({"v", "w"}, "very, wet, village", "v: top teeth on lower lip; w: rounded lips, no teeth"),
    "z (buzzing s)": ({"z", "ʒ"}, "is, easy, vision", "voiced: the throat should buzz"),
    "sh, ch, j": ({"ʃ", "tʃ", "dʒ"}, "she, cheap, job", "lips pushed forward; ch/j start with a stop"),
    "American r": ({"ɹ", "ɚ", "ɜː", "ɪɹ", "ɛɹ", "ɑːɹ", "ɔːɹ", "ʊɹ"}, "red, bird, water", "tongue bunched or curled back, never tapped or rolled"),
    "l": ({"l", "əl"}, "light, full, little", "tongue tip on the ridge; dark l at word end"),
    "ee vs i (seat/sit)": ({"iː", "i", "ɪ"}, "seat, sit, been", "iː is tense and smiling, ɪ is short and relaxed"),
    "æ (cat)": ({"æ"}, "cat, back, man", "jaw open wide, between 'e' and 'a'"),
    "open vowels (cup, cot, caught)": ({"ʌ", "ɑː", "ɔː", "ɐ"}, "cup, hot, talk", "jaw drop; ʌ is short and central"),
    "e (bed)": ({"ɛ"}, "bed, said, men", "mid-open, short"),
    "u sounds (book, food)": ({"ʊ", "uː", "u"}, "book, food, should", "ʊ short and lax, uː long with rounded lips"),
    "diphthongs (go, day, my, now, boy)": ({"oʊ", "eɪ", "aɪ", "aʊ", "ɔɪ"}, "go, day, my, now", "glide all the way to the second vowel"),
    "weak vowels (schwa)": ({"ə", "ᵻ"}, "about, the, problem", "unstressed syllables are short and neutral"),
    "p, t, k": ({"p", "t", "k"}, "pen, time, cold", "aspirated at the start of stressed syllables"),
    "b, d, g": ({"b", "d", "ɡ", "ɾ"}, "big, dog, better", "voiced; American t/d between vowels become a quick tap"),
    "s": ({"s"}, "see, city, bus", "a clean hiss, tongue behind the teeth"),
    "f and h": ({"f", "h"}, "fine, have, half", ""),
    "m, n, ng": ({"m", "n", "ŋ"}, "man, sing, thinking", "ng: back of tongue, no g after it"),
    "y sound": ({"j"}, "yes, you, few", ""),
}
VOWELS = {"ə", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɐ", "ɑː", "ɔː", "ʊ", "u", "uː", "ɚ", "ɜː",
          "oʊ", "eɪ", "aɪ", "aʊ", "ɔɪ", "ɪɹ", "ɛɹ", "ɑːɹ", "ɔːɹ", "ʊɹ", "əl"}
FINAL = "word endings (final consonants)"
AREAS[FINAL] = (set(), "worked, hands, lived", "finish every word; don't drop or devoice the last sound")
if LANG == "nb":
    import lang_nb
    AREAS = dict(lang_nb.AREAS)
    VOWELS = lang_nb.VOWELS
    AREAS[FINAL] = (set(), *lang_nb.FINAL_EXAMPLES)
elif LANG == "el":
    import lang_el
    AREAS = dict(lang_el.AREAS)
    VOWELS = lang_el.VOWELS
    AREAS[FINAL] = (set(), *lang_el.FINAL_EXAMPLES)
PHONE2AREA = {p: a for a, (ps, _, _) in AREAS.items() for p in ps}
# what the page shows for an area; the keys above stay the same in saved history and the bank
LABELS = ({**lang_nb.AREA_LABELS, FINAL: lang_nb.FINAL_LABEL} if LANG == "nb" else
          {**lang_el.AREA_LABELS, FINAL: lang_el.FINAL_LABEL} if LANG == "el" else {})


def label(area):
    return LABELS.get(area, area)

WEAK = 0.5


def phone_areas(phones):
    """Focus area per unit of a word: by phone (English) or by letter pattern (Norwegian)."""
    if LANG == "nb":
        return lang_nb.letter_areas(phones)
    return [PHONE2AREA.get(p) for p in phones]


def attempt_observations(result):
    """[(area, norm, word, phone, heard)] for every scored phone."""
    obs = []
    for w in result["words"]:
        phs = w["phones"]
        last_cons = max((i for i, p in enumerate(phs) if p["phone"] not in VOWELS), default=None)
        areas = phone_areas([p["phone"] for p in phs])
        for i, p in enumerate(phs):
            if p["norm"] is None:
                continue
            area = areas[i]
            if area:
                obs.append((area, p["norm"], w["word"], p["phone"], p["heard"]))
            if i == last_cons and i == len(phs) - 1 and len(phs) > 1:
                obs.append((FINAL, p["norm"], w["word"], p["phone"], p["heard"]))
    return obs


def area_table(results, decay=0.7):
    """Aggregate several attempts (most recent first, weighted by decay**age)."""
    agg = defaultdict(lambda: dict(wsum=0.0, score=0.0, n=0, weak=0, words=defaultdict(int),
                                   confusions=defaultdict(int)))
    for age, res in enumerate(results):
        wt = decay ** age
        for area, norm, word, phone, heard in attempt_observations(res):
            a = agg[area]
            a["wsum"] += wt
            a["score"] += wt * norm
            a["n"] += 1
            if norm < WEAK:
                a["weak"] += 1
                if age == 0:
                    a["words"][word.lower()] += 1
                a["confusions"][f"{phone}→{heard}"] += 1
    table = []
    for area, a in agg.items():
        mean = a["score"] / a["wsum"]
        table.append(dict(
            area=area, mean=round(mean, 3), n=a["n"], weak=a["weak"],
            examples=AREAS[area][1], how=AREAS[area][2],
            words=[w for w, _ in sorted(a["words"].items(), key=lambda kv: -kv[1])][:6],
            confusions=[c for c, _ in sorted(a["confusions"].items(), key=lambda kv: -kv[1])][:3],
        ))
    # rank by how much it costs you: deficit weighted by how often the sound occurs
    for t in table:
        t["priority"] = round((1 - t["mean"]) * min(t["n"], 20) ** 0.5, 3)
    table.sort(key=lambda t: -t["priority"])
    return table


def top_focus(table, k=3, min_n=3):
    """Areas with actual errors first (by priority), then fill up to k with the
    lowest-scoring remaining areas so there are always three things to work on."""
    eligible = [t for t in table if t["n"] >= min_n]
    weak, covered = [], set()
    for t in (t for t in eligible if t["weak"] > 0):
        # skip an area whose problem words are all explained by a higher-priority one
        # (e.g. "with" said "wit" is a th problem, not also a word-ending problem)
        if t["words"] and set(t["words"]) <= covered:
            continue
        weak.append(t)
        covered |= set(t["words"])
    rest = sorted((t for t in eligible if t["weak"] == 0), key=lambda t: t["mean"])
    return (weak + rest)[:k]


# ---------- history ----------
def save_attempt(entry):
    entry = {"time": time.time(), **entry}
    DATA.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load_history(limit=200):
    if not HISTORY.exists():
        return []
    out = []
    for line in HISTORY.read_text().splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
