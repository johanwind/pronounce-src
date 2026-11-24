"""Norwegian (Bokmål, Urban East Norwegian) language pack.

Multilingual phone recognisers hear Norwegian through other languages' categories (ʈ as t,
æ as a/ɛ, kj as sj, y as i), so on real native speech they barely separate minimal pairs.
The National Library's Norwegian wav2vec2 (trained on Norwegian speech with orthographic
targets) does much better, and Norwegian spelling encodes the contrasts learners miss
(y/i, ø/e, æ/e, o/å, kj/sj).

Scoring unit: words. Each word competes against confusable spellings (CONFUSIONS) and the
recogniser's own transcript of it; see scoring.Scorer.word_ratios. Per-letter posteriors
(the earlier method) were too noisy. The NB Uttale lexicon (Språkbanken, CC0) marks silent
letters, accepted variants and homophones, so those don't count as errors:
    python lang_nb.py build   # one-off: CSV -> data/nb_lexicon.sqlite
Evaluation: tests/nb_fleurs.py (real speech), tests/nb_planted_tts.py (non-word errors).
"""

import os
import re
import sqlite3
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LEX_CSV = Path(os.environ.get("NB_UTTALE_CSV", Path.home() / "data/nb_uttale/nb_uttale_leksika/e_spoken_pronunciation_lexicon.csv"))
LEX_DB = ROOT / "data" / "nb_lexicon.sqlite"
VOICE_DIR = ROOT / "data" / "nb_voices"

# voice name -> (piper model, speaker id)
VOICES = {
    **{f"nvcc_{k}": ("no_NO-nvcc-medium", i) for k, i in
       {"KNN": 0, "KSV": 1, "MMN": 2, "KON": 3, "MNN": 4, "MSV": 5, "MON": 6, "MNV": 7, "KMN": 8, "KNV": 9}.items()},
    "talesyntese": ("no_NO-talesyntese-medium", None),
    "vox": ("nb_NO-vox-medium", None),
}
# references: the two clearest voices. The multi-speaker nvcc model mumbles consonants
# (the Norwegian recogniser hears "huset" as k-y...), and adding it as a reference made
# real-speech results worse (tests/nb_fleurs.py: AUC 0.85 with these two, 0.76 with nvcc_MON).
REF_VOICES = os.environ.get("NB_REF_VOICES", "vox,talesyntese").split(",")
# The 1B v2 model (Dec 2024) gives the fewest false flags with word scoring (tests/nb_fleurs.py);
# 0.4 s per 20 s of audio and 4.3 GB on a laptop 4070. Per-letter posteriors were better with
# the old 300m v1 model, but word scoring beats both.
RECOGNIZER = "NbAiLab/nb-wav2vec2-1b-bokmaal-v2"
# Piper voices read ~330 words/min; references are scored at that speed (calibrated so),
# but playback is slowed to a careful pace (~165 words/min measured).
PLAY_SPEED = 0.6
# Playback voice: Google Cloud Chirp3-HD (male, natural; scores 100 with no flags as a reader).
# The references stay Piper, so scores don't depend on it. Falls back to Piper vox offline.
PLAY_VOICE = os.environ.get("NB_PLAY_VOICE", "google:nb-NO-Chirp3-HD-Iapetus")
GOOGLE_RATE = 0.85 / PLAY_SPEED  # Google speakingRate 0.85 reads at the pace of Piper at 0.6

# NoFAbet phone (stress digit stripped) -> recognizer tokens
NOFA = {
    "AX": ["ə"], "S": ["s"], "R": ["r"], "T": ["t"], "N": ["n"], "L": ["l"], "AH": ["ɑ"], "K": ["k"],
    "IH": ["ɪ"], "M": ["m"], "V": ["ʋ"], "D": ["d"], "P": ["p"], "EH": ["ɛ"], "F": ["f"], "B": ["b"],
    "OAH": ["ɔ"], "AA": ["ɑː"], "G": ["ɡ"], "NG": ["ŋ"], "EE": ["eː"], "NX": ["n̩"], "OH": ["ʊ"],
    "UH": ["ʉ"], "II": ["iː"], "RS": ["ʂ"], "OO": ["uː"], "H": ["h"], "UU": ["ʉ"], "SJ": ["ʃ"],
    "OA": ["oː"], "J": ["j"], "OE": ["øː"], "AEJ": ["æi"], "YH": ["y"], "RT": ["ʈ"], "OEH": ["œ"],
    "RN": ["ɳ"], "YY": ["yː"], "AEH": ["æ"], "RNX": ["ɳ"], "AE": ["æː"], "KJ": ["ç"], "LX": ["l̩"],
    "AEW": ["æ", "ʉ"], "RD": ["ɖ"], "OEJ": ["øi"], "RL": ["ɭ"], "AJ": ["aɪ"], "RLX": ["ɭ"],
    "OAJ": ["ɔɪ"], "OU": ["oʊ"],
}



# short function words: reduced or merged in connected speech, so their alignment is
# fragile. They are aligned (so timing stays right) but not scored.
FUNCTION_WORDS = set("""er var det den de dem en ei et og i på å med han hun ham henne når om at som
til for av så har jeg meg deg seg vi du dere oss man men kan skal vil ble blir ikke her der nå da
fra opp ut inn mot ved hos""".split())

VOWELS = set("aeiouyæøå")  # letters (used to find a word's final consonant)

# Word scoring (scoring.Scorer.word_ratios): each word competes against these confusable
# spellings, applied at every position, plus whatever the recogniser itself transcribes over
# the word. A word is flagged when a competitor explains the audio better than its spelling.
CONFUSIONS = [
    ("y", "i"), ("i", "y"), ("y", "ø"), ("ø", "y"), ("ø", "e"), ("e", "ø"), ("ø", "o"), ("u", "o"),
    ("o", "u"), ("u", "y"), ("y", "u"), ("o", "å"), ("å", "o"), ("å", "a"), ("æ", "e"), ("e", "æ"),
    ("æ", "a"), ("a", "æ"), ("e", "i"), ("i", "e"), ("a", "e"), ("kj", "sj"), ("sj", "kj"),
    ("kj", "k"), ("sj", "s"), ("skj", "sk"), ("tj", "t"), ("rt", "t"), ("rd", "d"), ("rn", "n"),
    ("rs", "s"), ("rl", "l"), ("ei", "e"), ("øy", "ø"), ("au", "o"),
]
# word score = 0.5 + 0.5 * ratio / LR_SCALE, clipped to [0, 1]; ratio is the log-likelihood
# margin (nats per frame) of the spelling over its best competitor, so 0 is the flag line.
LR_SCALE = float(os.environ.get("NB_LR_SCALE", 0.3))
# a competitor much unlike the word ("store" for "samme") is a reading slip, not pronunciation
SLIP_SIMILARITY = 0.5
# extra margin (nats per frame) the recogniser's own transcript of a word needs to win
FREE_MARGIN = float(os.environ.get("NB_FREE_MARGIN", 0.5))
# words the model voices themselves lose (ratio below this) are not scored: the recogniser
# can't tell the spelling from a competitor even in clean speech
REF_MIN_RATIO = float(os.environ.get("NB_REF_MIN", 0.0))


def alternatives(lets, marks):
    """Confusable spellings of a word (letters), skipping changes the lexicon accepts (votter
    said with å, lunsj with ø) and changes to silent letters. marks: letter_marks output."""
    w, out = "".join(lets), set()
    for a, b in CONFUSIONS:
        start = 0
        while (k := w.find(a, start)) >= 0:
            start = k + 1
            if any(marks[j][0] for j in range(k, k + len(a))):
                continue
            if len(a) == 1 and len(b) == 1 and b in marks[k][1]:
                continue
            out.add(w[:k] + b + w[k + len(a):])
    out.discard(w)
    return sorted(out)


def letter_diff(lets, alt):
    """{letter index: what the competitor has there} ('' = dropped). An insertion is put on the
    letter before it (skrev -> skreve: v -> 've')."""
    from difflib import SequenceMatcher
    w, out = "".join(lets), {}
    for op, i1, i2, j1, j2 in SequenceMatcher(None, w, alt, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            for k in range(i1, i2):
                out[k] = alt[j1:j2] if k == i1 else ""
        elif op == "insert":
            k = max(i1 - 1, 0)
            out[k] = (w[k] + alt[j1:j2]) if i1 > 0 else (alt[j1:j2] + w[0])
    return out


def letters(word, vocab):
    """The word's letters that the recogniser knows (drops apostrophes, hyphens; é -> e)."""
    w = word.lower().replace("é", "e").replace("è", "e").replace("ê", "e").replace("ó", "o").replace("ò", "o").replace("ô", "o")
    return [c for c in w if c in vocab and c != "|"]


# area -> (unused, example words, how). Areas are found from letter patterns (letter_areas).
# The keys are stored in history and the bank (bank_nb.json): the page shows AREA_LABELS instead.
AREAS = {
    "retroflex rt, rd, rn, rl, rs": (set(), "kort, gård, barn, ærlig, norsk",
                                     "r + t/d/n/l/s smelter sammen til én lyd: tungespissen bøyd bakover, ingen egen r"),
    "kj sound (kj, tj, ki, ky)": (set(), "kjøpe, tjue, kino, kyss",
                                  "midten av tungen opp mot den harde ganen, et mykt sus, leppene spredt"),
    "sj / skj sound": (set(), "sju, skjorte, ski, sky", "tungen lenger bak enn for s, leppene litt runde"),
    "y, ø and u (front rounded)": (set(), "by, syv, søt, høst, hus",
                                   "rund leppene godt mens tungen ligger fremme som for i eller e"),
    "o and å": (set(), "bok, ost, båt, åtte", "bokstaven o uttales ofte u (bok = buːk); å er en o-lyd med mindre runding"),
    "æ": (set(), "være, lære, æren", "åpen kjeve, mellom e og a"),
    "a": (set(), "mat, takk, dag", "langt bak og åpen, aldri som i engelsk «cat»"),
    "e and i": (set(), "se, penn, vin, finne", "e: stram og lang, eller kort og åpen; i: spredte lepper"),
    "diphthongs ei, øy, au": (set(), "nei, øy, hei, høy, sau", "start åpent og gli oppover"),
    "r (tapped)": (set(), "rød, fire, bra", "et raskt slag med tungespissen, aldri engelsk r"),
    "l": (set(), "lys, sol, hel", "lys l, tungespissen mot tannkjøttet bak overtennene"),
    "ng": (set(), "lang, tenke, sang", "bakre del av tungen, ingen g etter"),
    "v": (set(), "vann, hav, svar", "myk v, underleppen så vidt borti tennene"),
    "j (j, gj, hj)": (set(), "ja, gjøre, hjem", "gj, hj og g foran i/y/ei uttales j"),
    "p, t, k": (set(), "pen, tak, kald", "et lite pust etter lyden i starten av trykksterke stavelser"),
    "b, d, g": (set(), "bil, dag, god", "fullt stemt"),
    "s": (set(), "sol, is, hus", "en ren s-lyd"),
    "f and h": (set(), "fem, hus, hav", ""),
    "m, n": (set(), "mor, ni, vannet", ""),
}
AREA_LABELS = {
    "retroflex rt, rd, rn, rl, rs": "rt, rd, rn, rl, rs (retroflekser)",
    "kj sound (kj, tj, ki, ky)": "kj-lyden (kj, tj, ki, ky)",
    "sj / skj sound": "sj-lyden (sj, skj, ski, sky)",
    "y, ø and u (front rounded)": "y, ø og u (runde fremre vokaler)",
    "o and å": "o og å",
    "e and i": "e og i",
    "diphthongs ei, øy, au": "diftongene ei, øy, au",
    "r (tapped)": "r (tungespiss-r)",
    "f and h": "f og h",
}
_SINGLE = {**{c: "y, ø and u (front rounded)" for c in "yøu"}, **{c: "o and å" for c in "oå"}, "æ": "æ",
           "a": "a", "e": "e and i", "i": "e and i", "r": "r (tapped)", "l": "l", "v": "v", "w": "v",
           "j": "j (j, gj, hj)", "p": "p, t, k", "t": "p, t, k", "k": "p, t, k", "b": "b, d, g",
           "d": "b, d, g", "g": "b, d, g", "s": "s", "z": "s", "f": "f and h", "h": "f and h",
           "m": "m, n", "n": "m, n"}


def letter_areas(ls):
    """Focus area for each letter of a word (multi-letter patterns first)."""
    w = "".join(ls)
    out = [None] * len(w)

    def mark(i, n, area):
        for k in range(i, min(i + n, len(w))):
            if out[k] is None:
                out[k] = area
    for i in range(len(w)):
        rest, nxt = w[i:], w[i + 1:i + 2]
        if rest.startswith(("skj", "sj")) or (rest.startswith("sk") and w[i + 2:i + 3] in ("i", "y")):
            mark(i, 3 if rest.startswith("skj") else 2, "sj / skj sound")
        elif rest.startswith(("kj", "tj")) or (w[i] == "k" and nxt in ("i", "y") and (i == 0 or w[i - 1] != "c")):
            mark(i, 2 if rest.startswith(("kj", "tj")) else 1, "kj sound (kj, tj, ki, ky)")
        elif w[i] == "r" and nxt and nxt in "tdnls":
            mark(i, 2, "retroflex rt, rd, rn, rl, rs")
        elif rest.startswith(("ei", "øy", "au", "ai", "oy")):
            mark(i, 2, "diphthongs ei, øy, au")
        elif rest.startswith(("gj", "hj")):
            mark(i, 2, "j (j, gj, hj)")
        elif rest.startswith("ng"):
            mark(i, 2, "ng")
    for i, c in enumerate(w):
        if out[i] is None:
            out[i] = _SINGLE.get(c)
    return out


FINAL_EXAMPLES = ("gikk, hest, sterkt", "fullfør hvert ord; ikke slipp den siste konsonanten")
FINAL_LABEL = "ordslutt (siste konsonant)"

COACH_SYSTEM = """You are a warm, precise Norwegian pronunciation coach (Bokmål, Urban East Norwegian / Oslo standard).
You get machine measurements of a learner reading a Norwegian paragraph aloud. A Norwegian speech
recogniser scores each WORD by whether its spelling explains the audio better than confusable
spellings (y/i, ø/e, kj/sj, o/å, æ/e, rt/t ...) and than what the recogniser itself transcribed:
1.0 = clearly the right word, below 0.5 = another spelling fit better. "sounded_like" gives that
spelling (lyset sounded like "liset": y came out as i); "misread_as" means a different word was
read (a reading slip, not a pronunciation problem: mention it briefly at most). In letter
"confusions" like "y→i", where the letter y was it heard an i sound.
Remember Norwegian spelling: o is often u (bok), rt/rd/rn/rl/rs are retroflex, kj/tj/k+i/y is the
kj sound, sj/skj/sk+i/y the sj sound, d/g/t are silent in some words (god, og, det).
Trust patterns across several words more than single slips, and ignore areas with very few samples. It does not measure tonal accent. Pick the THREE most
useful things to work on next. For each give: a short name, the evidence (which words), one concrete
physical tip (tongue/lips/jaw/voicing), and a mini drill (a few Norwegian words or a short phrase).
Write everything in simple, clear Bokmål (the learner reads the page in Norwegian), and use the
area names as given.
Then write TWO next paragraphs for them to read, on different topics: natural, interesting Bokmål
prose of 50-70 words each, no digits, no tongue-twister nonsense, dense in the three focus sounds. Use correct standard Bokmål
spelling, never Danish or Swedish forms (igjen not igen, ikke not ikkje, hvordan not hur)."""


# ---------- lexicon ----------
def build_lexicon(csv_path=LEX_CSV, db_path=LEX_DB):
    import csv
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE lex (word TEXT, pron TEXT, n INTEGER, PRIMARY KEY (word, pron))")
    counts = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            phones = []
            ok = True
            for p in row["nofabet_transcription"].split():
                base = re.sub(r"\d", "", p)
                if base == "_":
                    continue
                if base not in NOFA:
                    ok = False
                    break
                phones += NOFA[base]
            if ok and phones:
                key = (row["wordform"].lower(), " ".join(phones))
                counts[key] = counts.get(key, 0) + 1
    con.executemany("INSERT INTO lex VALUES (?, ?, ?)", [(w, p, n) for (w, p), n in counts.items()])
    con.commit()
    con.close()
    tmp.replace(db_path)
    return len(counts)


# Which letters can spell each lexicon sound (length marks and syllabic marks stripped).
SOUND_LETTERS = {
    "ə": "ea", "s": "szcx", "r": "r", "t": "td", "n": "n", "l": "l", "ɑ": "a", "k": "kcgqx", "ɪ": "iey",
    "m": "m", "ʋ": "vwf", "d": "d", "p": "pb", "ɛ": "eæa", "f": "fv", "b": "b", "ɔ": "oå", "ɡ": "g",
    "ŋ": "ng", "e": "e", "ʊ": "ou", "ʉ": "uy", "i": "iy", "ʂ": "rs", "u": "ou", "h": "h", "ʃ": "skjh",
    "o": "åo", "j": "jgiy", "ø": "ø", "æi": "ei", "y": "y", "ʈ": "rt", "œ": "øo", "ɳ": "rn", "æ": "æea",
    "ç": "kjt", "ɖ": "rd", "øi": "øy", "ɭ": "rl", "aɪ": "ai", "ɔɪ": "oyi", "oʊ": "ou", "z": "zs",
    "w": "wu", "x": "x",
}
SOUND_LETTER = {"œ": "ø", "ø": "ø", "ʉ": "u", "y": "y", "i": "i", "ɪ": "i", "e": "e", "ɛ": "e",
                "æ": "æ", "ɑ": "a", "ɔ": "å", "o": "å", "u": "o", "ʊ": "o", "ə": "e"}


def _base(ph):
    return ph.replace("ː", "").replace("\u0329", "")


def letter_marks(lets, prons):
    """Align the word's letters to each lexicon pronunciation. Returns, per letter,
    (silent, alternatives): silent if no sound is spelled by it in ANY pronunciation (the d
    in "tidlig", the h in "hvitt", the t in "lyset"); alternatives are the vowel letters a
    listed pronunciation uses instead (lunsj: u -> ø)."""
    silent = [False] * len(lets)
    alts = [set() for _ in lets]
    for pron in prons[:4]:
        ph = [_base(p) for p in pron if _base(p)]
        n, m = len(lets), len(ph)
        INF = 1e9
        # cost[i][j][s]: i letters, j sounds used; s=1 if letter i-1 spelled sound j-1 (can be joined)
        cost = [[[INF, INF] for _ in range(m + 1)] for _ in range(n + 1)]
        back = [[[None, None] for _ in range(m + 1)] for _ in range(n + 1)]
        cost[0][0][0] = 0
        for i in range(n + 1):
            for j in range(m + 1):
                for s in (0, 1):
                    c = cost[i][j][s]
                    if c >= INF:
                        continue
                    if i < n and j < m:  # letter i spells sound j (substitution if not compatible)
                        ok = lets[i] in SOUND_LETTERS.get(ph[j], "")
                        nc = c + (0 if ok else 1.5 if lets[i] in VOWELS and ph[j] in SOUND_LETTER else 2)
                        if nc < cost[i + 1][j + 1][1]:
                            cost[i + 1][j + 1][1], back[i + 1][j + 1][1] = nc, (i, j, s, "match")
                    if i < n and j > 0 and s == 1:  # letter i joins the previous letter's sound
                        ok = lets[i] in SOUND_LETTERS.get(ph[j - 1], "") or lets[i] == lets[i - 1]
                        nc = c + (0 if ok else 1.5)
                        if nc < cost[i + 1][j][1]:
                            cost[i + 1][j][1], back[i + 1][j][1] = nc, (i, j, s, "join")
                    if i < n:  # silent letter
                        if c + 1 < cost[i + 1][j][0]:
                            cost[i + 1][j][0], back[i + 1][j][0] = c + 1, (i, j, s, "silent")
                    if j < m:  # sound with no letter
                        if c + 1 < cost[i][j + 1][0]:
                            cost[i][j + 1][0], back[i][j + 1][0] = c + 1, (i, j, s, "insert")
        s = 0 if cost[n][m][0] <= cost[n][m][1] else 1
        i, j = n, m
        while back[i][j][s]:
            pi, pj, ps, op = back[i][j][s]
            if op == "silent":
                silent[pi] = True
            elif op == "match" and lets[pi] not in SOUND_LETTERS.get(ph[pj], "") and ph[pj] in SOUND_LETTER:
                alts[pi].add(SOUND_LETTER[ph[pj]])
            i, j, s = pi, pj, ps
    return [(s, sorted(a - {l})) for s, a, l in zip(silent, alts, lets)]


class Lexicon:
    def __init__(self, db_path=LEX_DB):
        if not db_path.exists():
            print("building Norwegian lexicon (one-off, ~1 min)...", flush=True)
            build_lexicon(db_path=db_path)
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        self.memo = {}

    def _get(self, w):
        with self.lock:
            rows = self.con.execute("SELECT pron FROM lex WHERE word = ? ORDER BY n DESC, length(pron)", (w,)).fetchall()
        return [r[0].split() for r in rows]

    def lookup(self, word):
        """All pronunciations (lists of tokens), most frequent first; [] if unknown.
        Unknown compounds are split (hyphen, or longest known head + known tail) and
        the parts' main pronunciations joined."""
        w = word.lower().replace("’", "'")
        if w in self.memo:
            return self.memo[w]
        self.memo[w] = []  # guards recursion
        self.memo[w] = out = self._lookup(w)
        return out

    def _lookup(self, w):
        alts = self._get(w)
        if alts:
            return alts
        if "-" in w:
            parts = [self.lookup(p) for p in w.split("-") if p]
            return [[ph for p in parts for ph in p[0]]] if parts and all(parts) else []
        # two known parts (head may carry a linking s or e), most balanced split first
        best = None
        for i in range(3, len(w) - 2):
            tail = self._get(w[i:])
            head = tail and self._head(w[:i])
            if head and (best is None or min(i, len(w) - i) > best[0]):
                best = (min(i, len(w) - i), head + tail[0])
        if best:
            return [best[1]]
        # more parts: known tail, recursively split head, longest tail first
        for i in range(3, len(w) - 2):
            tail = self._get(w[i:])
            head = tail and self.lookup(w[:i])
            if head:
                return [head[0] + tail[0]]
        return []

    def _head(self, h):
        alts = self._get(h)
        if alts:
            return alts[0]
        if len(h) > 3 and h[-1] in "se":
            alts = self._get(h[:-1])
            if alts:
                return alts[0] + (["s"] if h[-1] == "s" else ["ə"])
        return None


# ---------- TTS ----------
VOICE_REPOS = {
    "no_NO-nvcc-medium": ("rhasspy/piper-voices", "no/no_NO/nvcc/medium/"),
    "no_NO-talesyntese-medium": ("rhasspy/piper-voices", "no/no_NO/talesyntese/medium/"),
    "nb_NO-vox-medium": ("rasmusjs/nb_NO-vox-medium", ""),
}


def fetch_voice(model, repos=None, voice_dir=None):
    """Download a Piper voice (onnx + json) into data/nb_voices (or voice_dir)."""
    import shutil
    from huggingface_hub import hf_hub_download
    repos, voice_dir = repos or VOICE_REPOS, voice_dir or VOICE_DIR
    repo, prefix = repos[model]
    voice_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".onnx", ".onnx.json"):
        shutil.copy(hf_hub_download(repo, prefix + model + ext), voice_dir / (model + ext))


_google = {}


def google_tts(text, name, rate=1.0):
    """Google Cloud Text-to-Speech with the gcloud login (offline tools and local mode only;
    the hosted server plays pre-made clips). -> (float32 audio, 24000)"""
    import base64
    import json
    import subprocess
    import time
    import urllib.request
    if _google.get("until", 0) < time.time():
        run = lambda *a: subprocess.run(["gcloud", *a], capture_output=True, text=True, timeout=30,
                                        check=True).stdout.strip()
        _google.update(token=run("auth", "print-access-token"), until=time.time() + 1800,
                       project=_google.get("project") or run("config", "get-value", "project"))
    body = dict(input=dict(text=text), voice=dict(languageCode=name[:5], name=name),
                audioConfig=dict(audioEncoding="LINEAR16", sampleRateHertz=24000,
                                 speakingRate=min(2.0, max(0.25, rate))))
    req = urllib.request.Request(
        "https://texttospeech.googleapis.com/v1/text:synthesize", json.dumps(body).encode(),
        {"Authorization": f"Bearer {_google['token']}", "x-goog-user-project": _google["project"],
         "Content-Type": "application/json"})
    wav = base64.b64decode(json.load(urllib.request.urlopen(req, timeout=60))["audioContent"])
    import io
    import soundfile as sf
    a, sr = sf.read(io.BytesIO(wav), dtype="float32")  # LINEAR16 comes as a WAV file
    return a, sr


class PiperTTS:
    """Piper voices by name. Defaults to the Norwegian ones; lang_el passes its own."""

    def __init__(self, voices=None, repos=None, voice_dir=None):
        self.voices, self.repos, self.dir = voices or VOICES, repos or VOICE_REPOS, voice_dir or VOICE_DIR
        self.models = {}
        self.lock = threading.Lock()

    def _voice(self, name):
        model, spk = self.voices[name]
        if model not in self.models:
            from piper import PiperVoice
            path = self.dir / f"{model}.onnx"
            if not path.exists():
                fetch_voice(model, self.repos, self.dir)
            self.models[model] = PiperVoice.load(str(path))
        return self.models[model], spk

    def audio(self, text, voice, speed=1.0):
        """(float32 audio, sample rate)"""
        if voice.startswith("google:"):
            try:
                return google_tts(text, voice[7:], speed * GOOGLE_RATE)
            except Exception as e:  # offline / not logged in: the old Piper playback voice
                print("Google TTS failed, using Piper vox:", e, file=sys.stderr)
                voice = "vox"
        from piper import SynthesisConfig
        with self.lock:
            v, spk = self._voice(voice)
            cfg = SynthesisConfig(speaker_id=spk, length_scale=1.0 / speed)
            chunks = [c.audio_float_array for c in v.synthesize(text, cfg)]
        a = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(v.config.sample_rate, np.float32)
        return a, v.config.sample_rate


if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    print(build_lexicon(), "word/pronunciation pairs")
