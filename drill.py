"""Single-sound drill: say one word, get feedback on one target sound.

Feedback is relative to several TTS voices saying the same word, so you learn the
sound category rather than one speaker's voice. For vowels, the recogniser's belief
over vowels is turned into a point on the vowel chart (front/back x close/open), which
shows which direction to move even when the score says "wrong".
"""

import io
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import torch

from scoring import FRAME_SEC, PHONEME_MODEL_ID, SAMPLE_RATE, trim_silence

DRILL_VOICES = ["am_michael", "af_heart", "am_adam", "af_bella"]

# IPA vowel chart positions: x = 0 front .. 1 back, y = 0 close (tongue high) .. 1 open
VOWEL_POS = {
    "iː": (0.0, 0.0), "i": (0.03, 0.03), "y": (0.03, 0.0), "yː": (0.03, 0.0),
    "ɪ": (0.18, 0.15), "ᵻ": (0.45, 0.12), "ɨ": (0.5, 0.0), "ʉ": (0.55, 0.0),
    "e": (0.05, 0.33), "eː": (0.05, 0.33), "eɪ": (0.07, 0.3), "ø": (0.08, 0.33),
    "ɛ": (0.12, 0.6), "æ": (0.2, 0.82), "a": (0.25, 1.0), "aɪ": (0.3, 0.95), "aʊ": (0.3, 0.95),
    "ə": (0.5, 0.5), "ɐ": (0.5, 0.7), "ɜː": (0.5, 0.55), "ɜ": (0.5, 0.55), "ɚ": (0.52, 0.5),
    "ʌ": (0.75, 0.62), "ɑː": (1.0, 1.0), "ɑ": (0.95, 1.0), "ɒ": (1.0, 0.95),
    "ɔː": (1.0, 0.68), "ɔ": (1.0, 0.68), "ɔɪ": (1.0, 0.68),
    "o": (1.0, 0.33), "oː": (1.0, 0.33), "oʊ": (0.95, 0.38), "əʊ": (0.7, 0.45),
    "ʊ": (0.78, 0.17), "u": (0.98, 0.02), "uː": (1.0, 0.0),
}
VOWEL_EXAMPLES = {
    "iː": "see", "ɪ": "sit", "eɪ": "day", "ɛ": "bed", "æ": "cat", "ə": "about",
    "ɜː": "bird", "ʌ": "cup", "ɑː": "father", "ɔː": "thought", "oʊ": "go", "ʊ": "book", "uː": "food",
}

# Words for drilling, per target phone. Pairs show the contrast with the sound it's
# most often confused with.
DRILL_SETS = {
    "ʊ": dict(words=["book", "good", "put", "look", "foot", "could", "wood", "cook", "full", "push", "sugar", "woman"],
              pairs=[("full", "fool"), ("pull", "pool"), ("look", "Luke"), ("put", "pit"), ("book", "buck"), ("could", "cud")]),
    "uː": dict(words=["food", "blue", "move", "two", "moon", "school", "true", "soup", "room", "choose"],
               pairs=[("fool", "full"), ("pool", "pull"), ("food", "feed"), ("moon", "moan")]),
    "θ": dict(words=["think", "three", "thank", "both", "path", "month", "health", "nothing", "author"],
              pairs=[("think", "sink"), ("three", "tree"), ("thank", "tank"), ("path", "pass"), ("both", "boat")]),
    "ð": dict(words=["this", "the", "they", "mother", "weather", "breathe", "other", "smooth"],
              pairs=[("they", "day"), ("then", "den"), ("breathe", "breeze"), ("other", "udder")]),
    "z": dict(words=["zoo", "easy", "music", "lazy", "his", "says", "rose", "visit", "busy"],
              pairs=[("zoo", "sue"), ("rise", "rice"), ("his", "hiss"), ("buzz", "bus"), ("eyes", "ice")]),
    "v": dict(words=["very", "voice", "village", "seven", "over", "live", "love", "give"],
              pairs=[("vest", "west"), ("vine", "wine"), ("very", "wary"), ("veil", "whale")]),
    "w": dict(words=["we", "water", "west", "wine", "away", "quick", "one", "window"],
              pairs=[("west", "vest"), ("wine", "vine"), ("wet", "vet")]),
    "æ": dict(words=["cat", "bad", "man", "back", "hand", "happy", "apple", "black"],
              pairs=[("bad", "bed"), ("man", "men"), ("cat", "cut"), ("bat", "bet")]),
    "ɪ": dict(words=["sit", "bit", "live", "fish", "will", "big", "this", "city"],
              pairs=[("sit", "seat"), ("live", "leave"), ("ship", "sheep"), ("fill", "feel")]),
    "iː": dict(words=["see", "seat", "leave", "sheep", "feel", "green", "need", "people"],
               pairs=[("seat", "sit"), ("leave", "live"), ("sheep", "ship"), ("feel", "fill")]),
    "ʌ": dict(words=["cup", "but", "love", "come", "run", "money", "under", "mother"],
              pairs=[("cup", "cop"), ("luck", "look"), ("buck", "book"), ("hut", "hot")]),
    "ɹ": dict(words=["red", "right", "very", "sorry", "around", "three", "grow", "train"],
              pairs=[("right", "light"), ("red", "led"), ("pray", "play"), ("grass", "glass")]),
    "dʒ": dict(words=["job", "just", "jump", "age", "bridge", "large", "page"],
               pairs=[("jeep", "cheap"), ("joke", "choke"), ("ridge", "rich")]),
    "ʃ": dict(words=["she", "shop", "wish", "nation", "sugar", "fish", "push"],
              pairs=[("she", "see"), ("ship", "sip"), ("shoe", "sue")]),
}


def vowel_point(dist):
    """Expected chart position under the recogniser's belief over vowels.
    Returns (x, y, vowel_mass) or None if the window was mostly not a vowel."""
    xs = ys = m = 0.0
    for ph, p in dist.items():
        pos = VOWEL_POS.get(ph)
        if pos:
            xs += p * pos[0]
            ys += p * pos[1]
            m += p
    if m < 0.25:
        return None
    return (xs / m, ys / m, m)


def direction_hint(target, you, ref_center):
    """Plain-language direction from your point toward the model voices' point."""
    if not you or not ref_center:
        return None
    dx, dy = ref_center[0] - you[0], ref_center[1] - you[1]
    tips = []
    if dx > 0.12:
        tips.append("pull the tongue further back (and let the lips round a little)")
    elif dx < -0.12:
        tips.append("bring the tongue further forward (lips less rounded)")
    # height is noisier between voices than front/back, so demand a bigger gap
    if dy > 0.2:
        tips.append("open the jaw a bit more, tongue lower")
    elif dy < -0.2:
        tips.append("close the jaw a bit, tongue higher")
    return "; ".join(tips) or None


# ---------- vowel prototypes from the model's internal features ----------
# The recogniser's *labels* are unreliable for some English vowels (it gives ʊ in a
# clean "book" only ~5%), but its middle-layer features separate vowels well across
# speakers: nearest-prototype accuracy over 10 vowels with 8 held-out TTS voices was
# 86% (ʊ 84%) using the mean of layers 6-12 (tests/exp_vowel_embed.py).
EMBED_LAYERS = (6, 8, 10, 12)
PROTO_VOICES = ["am_michael", "am_adam", "am_eric", "am_liam", "af_heart", "af_bella", "af_nicole", "af_sarah"]
PROTO_WORDS = {
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
# Non-English vowels learners often produce instead. They are never drill targets;
# they give a wrong vowel somewhere to go. Without them a pure, tense [u] (Spanish
# "luc", or a Norwegian-style u) had no class of its own and landed on ʊ.
# label -> (kokoro lang, espeak lang, phone, words, voices)
FOREIGN = {
    "u": [("e", "es", "u", ["luz", "sur", "uno", "cura", "duro", "muro", "puro", "nube", "luna", "tuvo"], ["ef_dora", "em_alex", "em_santa"]),
          ("i", "it", "u", ["lupo", "tutto", "luce", "punto", "muro", "uva", "fumo", "lungo"], ["if_sara", "im_nicola"]),
          ("p", "pt-br", "u", ["luz", "rua", "tudo", "muro", "uva", "suco"], ["pf_dora", "pm_alex"])],
    "y": [("f", "fr-fr", "y", ["lune", "rue", "tu", "mur", "sur", "pur", "plus", "jupe", "vue", "lutte", "buche", "pull"], ["ff_siwis"])],
    "ø": [("f", "fr-fr", "ø", ["peu", "deux", "feu", "bleu", "jeu", "nœud", "eux", "ceux", "veut", "creux"], ["ff_siwis"])],
}
FOREIGN_EXAMPLES = {"u": "Spanish luz", "y": "French tu", "ø": "French peu"}
BANK_VERSION = 2
BANK_FILE = Path(os.environ.get("PRONOUNCE_DATA", Path(__file__).resolve().parent / "data")) / "vowel_bank.npz"


def segment_embedding(scorer, audio16, seg):
    """Mean of L2-normalised layer features over the segment (seconds)."""
    lo = int(seg[0] / FRAME_SEC)
    hi = max(int(seg[1] / FRAME_SEC), lo + 1)
    with scorer.lock, torch.no_grad():
        x = scorer.fe(audio16, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(scorer.device)
        hs = scorer.model(x, output_hidden_states=True).hidden_states
    vs = []
    for L in EMBED_LAYERS:
        v = hs[L][0, lo:hi].mean(0).float().cpu().numpy()
        vs.append(v / np.linalg.norm(v))
    v = np.mean(vs, 0)
    return v / np.linalg.norm(v)


class VowelBank:
    def __init__(self, scorer, log=print):
        self.s = scorer
        self.log = log
        self.ready = threading.Event()
        self.vowels, self.protos, self.tau = None, None, 0.02

    def load_or_build(self):
        try:
            if BANK_FILE.exists():
                d = np.load(BANK_FILE, allow_pickle=False)
                if (str(d["model"]) == PHONEME_MODEL_ID and "version" in d.files
                        and int(d["version"]) == BANK_VERSION):
                    self.vowels, self.protos, self.tau = list(d["vowels"]), d["protos"], float(d["tau"])
                    return
            self.build()
        finally:
            self.ready.set()

    def build(self):
        t0 = time.time()
        self.log("building vowel prototypes (one-off, about a minute)")
        X, y, who = [], [], []
        for v, words in PROTO_WORDS.items():
            for w in words:
                wp = self.s.words_and_phones(w)
                phones = [p for _, ps in wp for p in ps]
                if v not in phones:
                    continue
                for vo in PROTO_VOICES:
                    a, _ = trim_silence(self.s.tts_audio(w + ".", vo))
                    res = self.s.align(a, wp)
                    if not res:
                        continue
                    X.append(segment_embedding(self.s, a, res[phones.index(v)]["seg"]))
                    y.append(v)
                    who.append(vo)
        X, y, who = np.array(X), np.array(y), np.array(who)
        FX, fy = self._foreign_samples()
        vowels = list(PROTO_WORDS) + list(FOREIGN)
        allX, ally = np.concatenate([X, FX]), np.concatenate([y, fy])
        protos = np.stack([allX[ally == v].mean(0) for v in vowels])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        # temperature: maximise held-out-voice likelihood (English voices as folds;
        # foreign classes stay in every fold as distractors)
        sims = []
        for vo in PROTO_VOICES:
            tr = who != vo
            c = np.stack([X[tr & (y == v)].mean(0) for v in PROTO_WORDS] + [FX[fy == v].mean(0) for v in FOREIGN])
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            sims.append((X[~tr] @ c.T, np.array([vowels.index(t) for t in y[~tr]])))
        best = None
        for tau in np.geomspace(0.003, 0.2, 40):
            ll = 0.0
            for S, idx in sims:
                z = S / tau
                z -= z.max(1, keepdims=True)
                ll += (z[np.arange(len(idx)), idx] - np.log(np.exp(z).sum(1))).sum()
            if best is None or ll > best[0]:
                best = (ll, tau)
        self.vowels, self.protos, self.tau = vowels, protos, float(best[1])
        BANK_FILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(BANK_FILE, model=PHONEME_MODEL_ID, version=BANK_VERSION, vowels=np.array(vowels),
                 protos=protos, tau=self.tau)
        self.log(f"vowel prototypes ready ({len(X)}+{len(FX)} samples, tau={self.tau:.3f}, {time.time()-t0:.0f}s)")

    def _foreign_samples(self):
        from kokoro import KPipeline
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        from scipy.signal import resample_poly
        X, y = [], []
        pipes = {}
        for label, groups in FOREIGN.items():
            for lang, elang, phone, words, voices in groups:
                if lang not in pipes:
                    pipes[lang] = KPipeline(lang_code=lang, model=self.s.tts.model)
                be = EspeakBackend(elang, preserve_punctuation=False, with_stress=False,
                                   language_switch="remove-flags")
                for w in words:
                    ipa = be.phonemize([w], separator=Separator(phone=" ", word="|"), strip=True)[0]
                    phones = [t for ph in ipa.replace("|", " ").split() for t in self.s._to_vocab(ph)]
                    if phone not in phones:
                        continue
                    wp = [(w, phones)]
                    for vo in voices:
                        with self.s.lock:
                            chunks = [a.numpy() for _, _, a in pipes[lang](w + ".", voice=vo)]
                        if not chunks:
                            continue
                        a = resample_poly(np.concatenate(chunks), 2, 3).astype(np.float32)
                        a, _ = trim_silence(a)
                        res = self.s.align(a, wp)
                        if not res:
                            continue
                        X.append(segment_embedding(self.s, a, res[phones.index(phone)]["seg"]))
                        y.append(label)
        self.log(f"foreign vowel samples: { {v: y.count(v) for v in FOREIGN} }")
        return np.array(X), np.array(y)

    def has(self, vowel):
        self.ready.wait()
        return vowel in self.vowels

    def classify(self, emb, tau_scale=1.0):
        z = self.protos @ emb / (self.tau * tau_scale)
        z = np.exp(z - z.max())
        p = z / z.sum()
        return {v: round(float(x), 3) for v, x in zip(self.vowels, p)}


# espeak vowel -> kokoro/misaki phoneme symbol (for synthesising a word with a swapped vowel)
MISAKI = {"ʊ": "ʊ", "uː": "u", "ɪ": "ɪ", "iː": "i", "ʌ": "ʌ", "ɑː": "ɑ", "ɔː": "ɔ", "oʊ": "O", "ɛ": "ɛ", "æ": "æ"}
CTX_VOICES = PROTO_VOICES

SHORT_VOWELS = {"ʊ", "ɪ", "ɛ", "æ", "ʌ"}
LONG_RATIO = 1.5


class Drill:
    def __init__(self, scorer, log=print):
        self.s = scorer
        self.bank = VowelBank(scorer, log)
        threading.Thread(target=self.bank.load_or_build, daemon=True).start()
        self.cache = {}
        self.lock = threading.Lock()
        # scoring knobs (chosen with tests/eval_drill_ood.py)
        self.tau_scale = 1.0      # >1 softens the prototype probabilities
        self.ref_norm = True      # divide by the model voices' own score
        self.len_penalty = True   # penalise drawn-out short vowels
        self.context = True       # compare against the same word with the vowel swapped
        self.mix = False          # average same-word and global prototypes
        self.ctx_cache = {}

    def _target_index(self, phones, target):
        if target and target in phones:
            return phones.index(target)
        for i, p in enumerate(phones):  # default: first vowel
            if p in VOWEL_POS:
                return i
        return 0

    def _judge(self, audio16, res, ti, use_bank, ctx=None):
        """Distribution over what the target sounded like, and the target's score."""
        t = res[ti]
        if not use_bank:
            return t, t["dist"]
        emb = segment_embedding(self.s, audio16, t["seg"])
        glob = self.bank.classify(emb, self.tau_scale)
        if not ctx:
            return t, glob
        z = ctx["protos"] @ emb / ctx["tau"]
        z = np.exp(z - z.max())
        p = z / z.sum()
        if self.mix:
            # average with the global prototypes: same-word variants are sharp but
            # occasionally ambiguous (the TTS renders two swapped vowels alike)
            p = 0.5 * p + 0.5 * np.array([glob.get(v, 0.0) for v in ctx["vowels"]])
            p = p / p.sum()
        # foreign vowels (no English variant can represent them) come from the global bank
        foreign = sum(glob.get(f, 0.0) for f in FOREIGN)
        dist = {v: round(float(x) * (1 - foreign), 3) for v, x in zip(ctx["vowels"], p)}
        for f in FOREIGN:
            if glob.get(f, 0) > 0.01:
                dist[f] = round(glob[f], 3)
        return t, dist

    def context_protos(self, word, target, wp, ti):
        """Prototypes for this word with each candidate vowel swapped in, spoken by
        several voices: same consonants, same position, only the vowel differs."""
        key = (word.lower(), target)
        if key in self.ctx_cache:
            return self.ctx_cache[key]
        if target not in MISAKI:
            return None
        ps, _ = self.s.tts.g2p(word)
        sym = MISAKI[target]
        idx = ps.find(sym)
        if idx < 0:
            return None
        from kokoro import KPipeline
        from scipy.signal import resample_poly
        vowels = list(MISAKI)
        flat = [p for _, phs in wp for p in phs]
        embs = {v: [] for v in vowels}
        wavs = {}
        for v in vowels:
            vps = ps[:idx] + MISAKI[v] + ps[idx + len(sym):]
            vflat = flat[:ti] + [v] + flat[ti + 1:]
            for vo in CTX_VOICES:
                with self.s.lock:
                    out = KPipeline.infer(self.s.tts.model, vps, self.s.tts.load_voice(vo), 1.0)
                a = resample_poly(out.audio.cpu().numpy(), 2, 3).astype(np.float32)
                a16, off = trim_silence(a)
                res = self.s.align(a16, [(word, vflat)])
                if not res:
                    continue
                embs[v].append(segment_embedding(self.s, a16, res[ti]["seg"]))
                if vo in DRILL_VOICES[:2]:
                    buf = io.BytesIO()
                    sf.write(buf, a, SAMPLE_RATE, format="WAV", subtype="PCM_16")
                    wavs[(v, vo)] = buf.getvalue()
        vowels = [v for v in vowels if len(embs[v]) >= 2]
        protos = np.stack([np.mean(embs[v], 0) for v in vowels])
        protos /= np.linalg.norm(protos, axis=1, keepdims=True)
        # temperature by leave-one-voice-out likelihood on these variants
        best = None
        n = min(len(embs[v]) for v in vowels)
        for tau in np.geomspace(0.002, 0.1, 30):
            ll = 0.0
            for k in range(n):
                c = np.stack([np.mean([e for j, e in enumerate(embs[v]) if j != k], 0) for v in vowels])
                c /= np.linalg.norm(c, axis=1, keepdims=True)
                X = np.stack([embs[v][k] for v in vowels])
                zz = X @ c.T / tau
                zz -= zz.max(1, keepdims=True)
                ll += (np.diag(zz) - np.log(np.exp(zz).sum(1))).sum()
            if best is None or ll > best[0]:
                best = (ll, tau)
        ctx = dict(vowels=vowels, protos=protos, tau=float(best[1]), wavs=wavs)
        self.ctx_cache[key] = ctx
        return ctx

    def reference(self, word, target=None):
        key = (word.lower(), target)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        wp = self.s.words_and_phones(word)
        phones = [p for _, ps in wp for p in ps]
        if not phones:
            raise ValueError(f"Couldn't find any sounds in '{word}'.")
        ti = self._target_index(phones, target)
        use_bank = self.bank.has(phones[ti])
        ctx = self.context_protos(word, phones[ti], wp, ti) if use_bank else None
        voices = []
        for v in DRILL_VOICES:
            # a trailing period makes the TTS say a single word naturally
            a = self.s.tts_audio(word.strip().rstrip(".") + ".", v)
            a16, off = trim_silence(a)
            res = self.s.align(a16, wp)
            if not res:
                continue
            t, dist_g = self._judge(a16, res, ti, use_bank, None)
            dist_c = self._judge(a16, res, ti, use_bank, ctx)[1] if ctx else dist_g
            dist = dist_c if self.context else dist_g
            buf = io.BytesIO()
            sf.write(buf, a, SAMPLE_RATE, format="WAV", subtype="PCM_16")
            voices.append(dict(
                voice=v, wav=buf.getvalue(),
                score=dist.get(phones[ti], 0.0) if use_bank else t["score"],
                score_g=dist_g.get(phones[ti], 0.0) if use_bank else t["score"],
                score_c=dist_c.get(phones[ti], 0.0) if use_bank else t["score"],
                dist=dist, point=vowel_point(dist),
                seg=(t["seg"][0] + off, t["seg"][1] + off),
            ))
        ref_score = float(np.mean([x["score"] for x in voices])) if voices else 1.0
        ref_score_g = float(np.mean([x["score_g"] for x in voices])) if voices else 1.0
        ref_score_c = float(np.mean([x["score_c"] for x in voices])) if voices else 1.0
        pts = [x["point"] for x in voices if x["point"]]
        center = (float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))) if pts else None
        ref_ms = float(np.median([1000 * (x["seg"][1] - x["seg"][0]) for x in voices])) if voices else None
        ref = dict(word=word, phones=phones, target_index=ti, target=phones[ti], use_bank=use_bank, ctx=ctx, ref_score_g=ref_score_g, ref_score_c=ref_score_c,
                   voices=voices, ref_score=ref_score, center=center, wp=wp, ref_ms=ref_ms)
        with self.lock:
            self.cache[key] = ref
        return ref

    def public_reference(self, ref):
        return dict(
            word=ref["word"], phones=ref["phones"], target=ref["target"], target_index=ref["target_index"],
            ref_score=round(ref["ref_score"], 3), center=ref["center"],
            ref_ms=ref["ref_ms"] and round(ref["ref_ms"]),
            method=("variants" if ref["ctx"] else "features") if ref["use_bank"] else "labels",
            variants=sorted({v for v, _ in ref["ctx"]["wavs"]}) if ref["ctx"] else [],
            is_vowel=ref["target"] in VOWEL_POS,
            voices=[dict(voice=v["voice"], score=round(v["score"], 3), dist=v["dist"],
                         point=v["point"] and v["point"][:2], seg=v["seg"]) for v in ref["voices"]],
        )

    def variant_wav(self, word, target, vowel, voice=None):
        """The drilled word spoken with another vowel swapped in (to hear the contrast)."""
        ctx = self.reference(word, target)["ctx"]
        if not ctx:
            raise KeyError("no variants for this word")
        for (v, vo), wav in ctx["wavs"].items():
            if v == vowel and (voice is None or vo == voice):
                return wav
        raise KeyError(vowel)

    def ref_wav(self, word, target, voice):
        for v in self.reference(word, target)["voices"]:
            if v["voice"] == voice:
                return v["wav"]
        raise KeyError(voice)

    def attempt(self, word, target, audio, rate):
        ref = self.reference(word, target)
        if rate != SAMPLE_RATE:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(rate), SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
        trimmed, off = trim_silence(audio)
        if len(trimmed) < SAMPLE_RATE * 0.15 or np.abs(trimmed).max() < 1e-3:
            return dict(error="Didn't hear anything.")
        res = self.s.align(trimmed, ref["wp"])
        if not res:
            return dict(error="Too short.")
        ti = ref["target_index"]
        tgt = ref["target"]
        t, dist = self._judge(trimmed, res, ti, ref["use_bank"], ref["ctx"] if self.context else None)
        if ref["use_bank"]:
            # probability of the target among the vowel prototypes, relative to how
            # clearly the model voices themselves come out (they are not 100% either)
            raw = dist.get(tgt, 0.0)
            rs = ref["ref_score_c"] if (self.context and ref["ctx"]) else ref["ref_score_g"]
            norm = min(1.0, raw / max(rs, 0.3)) if self.ref_norm else raw
        else:
            raw = t["score"]
            norm = min(1.0, raw / max(ref["ref_score"], 0.3))
            if t["heard_p"] > max(0.3, 1.2 * raw) and ref["ref_score"] >= 0.6:
                norm = min(norm, 0.45)
        ms = 1000 * (t["seg"][1] - t["seg"][0])
        too_long = bool(ref["ref_ms"] and tgt in SHORT_VOWELS and ms > LONG_RATIO * ref["ref_ms"])
        if self.len_penalty and too_long:
            # a short (lax) vowel held long sounds like its long (tense) neighbour
            norm = min(norm, 0.6)
        heard = max((k for k in dist if k != tgt), key=lambda k: dist[k], default=None)
        point = vowel_point(dist)
        others = [dict(phone=r["phone"], ok=r["score"] >= 0.3, heard=r["heard"])
                  for i, r in enumerate(res) if i != ti]
        return dict(
            target=tgt, score=round(norm, 3), raw=round(raw, 3), dist=dist, heard=heard,
            point=point and point[:2],
            hint=direction_hint(tgt, point, ref["center"]) if norm < 0.85 else None,
            seg=(t["seg"][0] + off, t["seg"][1] + off),
            ms=round(1000 * (t["seg"][1] - t["seg"][0])), ref_ms=ref["ref_ms"] and round(ref["ref_ms"]),
            word_span=(max(0.0, res[0]["start"] + off - 0.08), res[-1]["end"] + off + 0.12),
            others=others,
        )
