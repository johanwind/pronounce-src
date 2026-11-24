"""Greek (Modern Standard Greek) language pack.

Greek uses the same pipeline as English: espeak-ng ("el") turns the text into phones and the
multilingual phone recogniser (facebook/wav2vec2-lv-60-espeak-cv-ft) scores them. That works
here, unlike Norwegian, because Greek spelling maps regularly to a small set of sounds the
recogniser knows well (θ ð ɣ x ç ʝ are common across its training languages), and there is
one standard spoken form. References are Greek TTS voices (Piper, Meta MMS).
"""
import os
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
VOICE_DIR = ROOT / "data" / "el_voices"

# Piper voices: name -> (model, speaker id); "mms" is Meta's MMS-TTS Greek (VITS)
VOICES = {
    "joy": ("el_GR-joy-medium", None),
    "rapunzelina": ("el_GR-rapunzelina-medium", None),
    "rapunzelina_low": ("el_GR-rapunzelina-low", None),
}
VOICE_REPOS = {
    "el_GR-joy-medium": ("rhasspy/piper-voices", "el/el_GR/joy/medium/"),
    "el_GR-rapunzelina-medium": ("rhasspy/piper-voices", "el/el_GR/rapunzelina/medium/"),
    "el_GR-rapunzelina-low": ("rhasspy/piper-voices", "el/el_GR/rapunzelina/low/"),
}
MMS_REPO = "facebook/mms-tts-ell"
# MMS alone scores as well as MMS + Piper joy (AUC 0.855 vs 0.861, tests/el_fleurs.py) and keeps
# the install small; Piper voices still work if piper-tts is installed (EL_REF_VOICES=mms,joy).
REF_VOICES = os.environ.get("EL_REF_VOICES", "mms,joy").split(",")
# playback (the "hear it" voice): Google's Chirp3-HD Iapetus, the same male voice as Norwegian, at
# a careful 0.85 rate. Offline tools and local mode only (gcloud login); the hosted server plays
# pre-made clips (tools/bank_audio.py). Falls back to MMS. References stay MMS (+ Piper joy).
PLAY_VOICE = os.environ.get("EL_PLAY_VOICE", "google:el-GR-Chirp3-HD-Iapetus")
PLAY_SPEED = 1.0
GOOGLE_RATE = 0.85

# allophones and spelling-level variants a native speaker uses interchangeably
VARIANTS = {
    "ʝ": {"j"}, "j": {"ʝ"},          # γ before e/i: espeak writes j
    "ç": {"x"}, "x": {"ç"},          # χ before e/i is palatal
    "c": {"k"}, "k": {"c"},          # κ before e/i is palatal
    "ɟ": {"ɡ"}, "ɡ": {"ɟ"},
    "r": {"ɾ"}, "ɾ": {"r"},          # ρ is a tap
    "ɲ": {"n"}, "ʎ": {"l"},          # νι, λι before a vowel
    "ŋ": {"n"},
    # Greek's five vowels are mid/plain; the recogniser labels them with neighbouring symbols
    "a": {"ɐ", "ɑ", "ä"}, "e": {"ɛ", "e̞"}, "o": {"ɔ", "ʊ", "o̞"}, "u": {"ʊ", "uː"}, "i": {"ɪ", "iː"},
}

# articles, clitic pronouns, particles, short prepositions: reduced in connected speech
FUNCTION_WORDS = set("""ο η το οι τα τον την τη του της των τους τις ένας μία μια ένα έναν
και κι να θα με σε σ για από απ που πού ότι δεν δε μη μην στο στη στην στον στα στις στους
μου σου του μας σας τους τον τη τα τις ή αν ως ενώ αλλά ούτε είναι είμαι ήταν""".split())

VOWELS = {"a", "e", "i", "o", "u"}

# area name -> (phones, example words, how). The keys are stored in history and the bank
# (bank_el.json); the page shows AREA_LABELS where a key has English in it.
AREAS = {
    "θ (θέλω)": ({"θ"}, "θέλω, Αθήνα, θάλασσα", "η άκρη της γλώσσας ανάμεσα στα δόντια, χωρίς φωνή, ο αέρας περνάει· όχι τ ή φ"),
    "δ (δρόμος)": ({"ð"}, "δρόμος, παιδί, δέκα", "η γλώσσα ανάμεσα στα δόντια, με φωνή, όπως στο αγγλικό «this»· όχι ντ"),
    "γ (γάλα, γεια)": ({"ɣ", "ʝ", "j"}, "γάλα, νερό γλυκό, γεια", "τριβόμενος ήχος με φωνή πίσω στο στόμα (γα, γο, γου) ή σαν δυνατό γι (γε, γι)· ποτέ γκ"),
    "χ (χαρά, χέρι)": ({"x", "ç"}, "χαρά, όχι, χέρι", "βαθιά στο στόμα πριν από α, ο, ου· πιο μπροστά, όπως στο γερμανικό «ich», πριν από ε, ι"),
    "ρ (tapped r)": ({"r", "ɾ"}, "ρόδι, καρέκλα, μέρα", "ένα γρήγορο χτύπημα της άκρης της γλώσσας, όχι από τον λαιμό και όχι σαν το αγγλικό r"),
    "π, τ, κ (no puff of air)": ({"p", "t", "k", "c"}, "πατάτα, τώρα, καλά", "χωρίς φύσημα αέρα μετά τον ήχο, όπως μετά το s στο αγγλικό «spin»"),
    "μπ, ντ, γκ (b, d, g)": ({"b", "d", "ɡ", "ɟ"}, "μπορώ, ντομάτα, γκρι", "ηχηρά σύμφωνα που γράφονται με δύο γράμματα· μέσα στη λέξη συχνά με ένα ελαφρύ μ, ν"),
    "σ and ζ": ({"s", "z"}, "σπίτι, ζάχαρη, κόσμος", "το σ είναι καθαρό σ (ζ πριν από ηχηρό σύμφωνο: κόσμος)· το ζ βουίζει"),
    "τσ, τζ": ({"ts", "dz"}, "τσάι, τζάκι, κορίτσι", "ένας ήχος: τ και σ μαζί, γρήγορα"),
    "λι, νι (ʎ, ɲ)": ({"ʎ", "ɲ"}, "ήλιος, χωριό, νιάτα", "λ και ν πριν από ι + φωνήεν γίνονται ένας ήχος στον ουρανίσκο, όπως το ιταλικό gli, gn"),
    "α, ε, ο": ({"a", "e", "o"}, "μέρα, εδώ, όλα", "καθαρά φωνήεντα χωρίς γλίστρημα, ίδιο μήκος με τόνο ή χωρίς"),
    "ι, ου": ({"i", "u"}, "ήλιος, ουρανός, μου", "το ι (και η, υ, ει, οι) είναι πάντα καθαρό ι· το ου με στρογγυλά χείλη"),
    "φ, β": ({"f", "v"}, "φως, βουνό, αυτό", "το κάτω χείλος στα πάνω δόντια· αυ, ευ γίνονται αφ, εφ πριν από άηχο σύμφωνο, αβ, εβ αλλού"),
    "λ": ({"l"}, "λάδι, καλά, όλοι", "ελαφρύ λ, η άκρη της γλώσσας πίσω από τα πάνω δόντια"),
    "μ, ν": ({"m", "n", "ŋ"}, "μαμά, νερό, άγγελος", ""),
}
AREA_LABELS = {
    "ρ (tapped r)": "ρ (χτυπητό)",
    "π, τ, κ (no puff of air)": "π, τ, κ (χωρίς φύσημα)",
    "μπ, ντ, γκ (b, d, g)": "μπ, ντ, γκ",
    "σ and ζ": "σ και ζ",
    "λι, νι (ʎ, ɲ)": "λι, νι",
}
FINAL_EXAMPLES = ("καλός, παιδιών, είναι", "πες καθαρά το τελικό ς και ν")
FINAL_LABEL = "τέλος της λέξης (τελικά σύμφωνα)"

COACH_SYSTEM = """You are a warm, precise Greek pronunciation coach (Modern Standard Greek target).
You get machine measurements of a learner reading a Greek paragraph aloud. The scorer compares
each sound with clear synthetic Greek speech: 1.0 = as clear as the reference, below 0.5 = the
model heard something else. "confusions" like "θ→t" mean expected θ, heard t.
The measurements are noisy: trust patterns across several words, not single slips, and
ignore areas with very few samples. Stress (the written accent) is NOT measured, so don't
comment on it unless asked. Pick the THREE most useful things to work on next.
For each give: a short name, the evidence (which words), one concrete physical tip
(tongue/lips/jaw/voicing), and a mini drill (a few Greek words or a short phrase).
Write everything (area names, evidence, tips, drills) in simple, clear Greek: the learner reads
the page in Greek.
Then write TWO next paragraphs for them to read, on different topics: natural, interesting
Modern Greek prose of 45-65 words each, with correct monotonic accents, no digits, no
tongue-twister nonsense, everyday vocabulary, dense in the three focus sounds."""


SPLIT = {"aɪ": ["a", "i"], "eɪ": ["e", "i"], "ɔɪ": ["o", "i"], "oɪ": ["o", "i"], "aʊ": ["a", "u"]}


def fix_phones(phones):
    """espeak keeps written double consonants (θάλασσα -> s s); Greek says them once. It
    also writes English diphthongs for vowel pairs (τσάι -> aɪ), which Greek says as two."""
    out = []
    for p in [x for p in phones for x in SPLIT.get(p, [p])]:
        if out and p == out[-1] and p not in VOWELS:
            continue
        out.append(p)
    return out


class TTS:
    """Greek voices: Piper models, plus "mms" (Meta MMS-TTS, transformers VITS)."""

    def __init__(self):
        self.piper = None
        self.mms = None
        self.lock = threading.Lock()

    def audio(self, text, voice, speed=1.0):
        """Speak text in clause-sized chunks. Both VITS voice families clip the last sound of
        their input ("δρόμου" came out as "δρομ"), and MMS ignores punctuation entirely (no
        pauses between sentences). So each chunk gets a throwaway " -" pad, which makes the
        voice finish the word, and chunks are joined with pauses: 0.35 s after . ! ? ; and
        0.15 s after , :."""
        import re
        if voice.startswith("google:"):  # reads punctuation and finishes words: the whole text at once
            import lang_nb
            try:
                return lang_nb.google_tts(text, voice[7:], speed * GOOGLE_RATE)
            except Exception as e:  # offline / not logged in
                print("Google TTS failed, using MMS:", e, file=sys.stderr)
                voice = "mms"
        parts = [m.group(1).strip() for m in re.finditer(r"([^.!?;,:·]+)([.!?;,:·]*)", text)]
        stops = [m.group(2) for m in re.finditer(r"([^.!?;,:·]+)([.!?;,:·]*)", text)]
        out, sr = [], None
        for chunk, stop in zip(parts, stops):
            if not re.search(r"\w", chunk):
                continue
            a, sr = self._chunk(chunk + " -", voice, speed)
            a = _trim_end(a, sr)
            pause = 0.35 if any(c in stop for c in ".!?;") else 0.15 if stop else 0.0
            out += [a, np.zeros(int(sr * pause / speed), np.float32)]
        if not out:
            return self._chunk(text, voice, speed)
        return np.concatenate(out), sr

    def _chunk(self, text, voice, speed):
        if voice != "mms":
            if self.piper is None:
                import lang_nb  # Piper wrapper; optional for Greek
                self.piper = lang_nb.PiperTTS(VOICES, VOICE_REPOS, VOICE_DIR)
            return self.piper.audio(text, voice, speed)
        import torch
        with self.lock:
            if self.mms is None:
                from transformers import AutoTokenizer, VitsModel
                self.mms = (AutoTokenizer.from_pretrained(MMS_REPO), VitsModel.from_pretrained(MMS_REPO).eval())
            tok, model = self.mms
            model.config.speaking_rate = speed
            torch.manual_seed(0)  # VITS samples durations/noise: keep references reproducible
            with torch.no_grad():
                a = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
        return a.astype(np.float32), model.config.sampling_rate


def _trim_end(a, sr, db=-40):
    """Drop trailing silence (the pad's pause) but keep 60 ms after the last loud frame."""
    f = int(sr * 0.02)
    n = len(a) // f
    if n == 0:
        return a
    e = 10 * np.log10((a[: n * f].reshape(n, f) ** 2).mean(1) + 1e-10)
    loud = np.where(e > e.max() + db)[0]
    return a[: min(len(a), (loud[-1] + 4) * f)] if len(loud) else a
