"""Paragraph-level pronunciation scoring.

Pipeline:
  text --espeak--> per-word IPA phones (the same inventory the CTC model was trained on)
  audio --wav2vec2 phoneme CTC--> frame log-posteriors
  forced alignment --> a time window per expected phone
  per phone: share of non-blank posterior mass in its window that goes to the
  expected phone or an accepted variant ("soft GOP"), plus the strongest competitor.
  Scores are normalised against the same paragraph read by TTS voices, so phones the
  model can't recognise well even in clean speech don't count against you.
"""

import json
import math
import os
from pathlib import Path
import re
import threading

import numpy as np
import torch
import torchaudio
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
TTS_RATE = 24000
FRAME_SEC = 0.02  # wav2vec2 stride
PHONEME_MODEL_ID = os.environ.get("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
REF_VOICES = ["am_michael", "af_heart"]
PLAY_VOICE = "am_michael"
PLAY_SPEED = 1.0  # playback speed of the model voice relative to the reference voices

# Realisations a native US speaker might produce for each espeak target.
# Accepting these keeps the scorer from punishing normal reductions.
VARIANTS = {
    "ə": {"ɐ", "ʌ", "ɪ", "ᵻ", "ɚ", "əl"},
    "ɐ": {"ə", "ʌ"},
    "ʌ": {"ɐ", "ə"},
    "ɪ": {"ᵻ", "i", "ə", "iː"},
    "ᵻ": {"ɪ", "ə", "i"},
    "i": {"iː", "ɪ"},
    "iː": {"i"},
    "ɚ": {"ɜː", "ə", "ər1", "ɹ"},
    "ɜː": {"ɚ", "ɜ"},
    "ɑː": {"ɑ", "ɔː", "ɒ", "a"},
    "ɔː": {"ɑː", "ɔ", "ɒ", "ɑ"},
    "oʊ": {"o", "oː", "əʊ"},
    "uː": {"u"},
    "u": {"uː", "ʊ"},
    "ʊ": {"u", "ə"},  # weak forms: should/could/would
    "eɪ": {"e", "eː"},
    "ɛ": {"e"},
    "æ": {"a", "ɛ"},
    "aɪ": {"a"},
    "t": {"ɾ", "ʔ"},
    "d": {"ɾ"},
    "ɾ": {"t", "d"},
    "ɹ": {"r", "ɻ"},
    "l": {"ɫ", "əl"},
    "əl": {"l", "ɫ"},
    "ɪɹ": {"iə", "ɪ", "iː"},
    "ɛɹ": {"eə", "ɛ"},
    "ɑːɹ": {"ɑː", "ɑ"},
    "ɔːɹ": {"ɔː", "oːɹ"},
    "ʊɹ": {"ʊə", "ɔːɹ"},
    "n": {"n̩"},
}


# (previous phone, target) -> extra accepted realisations
CONTEXT_VARIANTS = {
    ("n", "t"): {"n"},   # nasal flap: "went out" -> wen-out, "winter" -> winner
    ("n", "d"): {"n"},
}

LANG = os.environ.get("PRONOUNCE_LANG", "en")
NOTHING = {"nb": "ingenting", "el": "τίποτα"}.get(LANG, "nothing")  # "heard" for a dropped sound, shown on the page
MODEL_FILE = os.environ.get("PHONEME_MODEL_FILE")  # optional: torch.save'd model to use instead
if LANG == "nb":
    # Norwegian scores words with the National Library's Norwegian wav2vec2 (see lang_nb)
    import lang_nb
    PHONEME_MODEL_ID = os.environ.get("PHONEME_MODEL", lang_nb.RECOGNIZER)
    REF_VOICES, PLAY_VOICE, PLAY_SPEED = lang_nb.REF_VOICES, lang_nb.PLAY_VOICE, lang_nb.PLAY_SPEED
    VARIANTS, CONTEXT_VARIANTS = {}, {}
elif LANG == "el":
    # Greek: same phone recogniser and espeak as English, Greek voices (see lang_el)
    import lang_el
    REF_VOICES, PLAY_VOICE, PLAY_SPEED = lang_el.REF_VOICES, lang_el.PLAY_VOICE, lang_el.PLAY_SPEED
    VARIANTS, CONTEXT_VARIANTS = lang_el.VARIANTS, {}


def _espeak(lang="en-us"):
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    data = Path(espeakng_loader.get_data_path())
    if len(str(data)) > 150:
        # espeak-ng has a fixed-size buffer for its data path; past ~160 chars it silently
        # falls back to its build machine's path and fails. Use a short copy instead.
        short = Path.home() / ".cache" / "pronounce-espeak-ng-data"
        if not (short / "phontab").exists():
            import shutil
            shutil.copytree(data, short, dirs_exist_ok=True)
        data = short
    EspeakWrapper.set_data_path(str(data))
    from phonemizer.backend import EspeakBackend
    return EspeakBackend(lang, preserve_punctuation=False, with_stress=False,
                         language_switch="remove-flags")


WORD_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*")  # letters/digits incl. æøå, é (same as the page's)


def tokenize(text):
    """Words, and for each word whether punctuation follows it (a natural pause)."""
    text = text.replace("’", "'")
    words, punct = [], []
    ms = list(WORD_RE.finditer(text))
    for i, m in enumerate(ms):
        words.append(m.group())
        tail = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)]
        punct.append(bool(re.search(r"[.,;:!?()\"—–«»]", tail)))
    return words, punct


class Scorer:
    def __init__(self, device=None, log=print, tts=True):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        from huggingface_hub import hf_hub_download
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"loading phoneme model on {self.device}")
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL_ID)
        if MODEL_FILE:  # a whole saved model, e.g. int8-quantised for a small server (tools/quantize.py)
            self.model = torch.load(MODEL_FILE, weights_only=False).eval()
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL_ID).to(self.device).eval()
        with open(hf_hub_download(PHONEME_MODEL_ID, "vocab.json")) as f:
            self.vocab = json.load(f)
        self.id2phone = {v: k for k, v in self.vocab.items()}
        self.blank = self.vocab["<pad>"] if "<pad>" in self.vocab else self.vocab["[PAD]"]
        log("loading TTS + espeak")
        if LANG == "nb":
            self.tts = lang_nb.PiperTTS() if tts else None
            # pronunciation lexicon: tells which letters are silent or have accepted variants
            self.lex = lang_nb.Lexicon() if lang_nb.LEX_DB.exists() else None
            self.espeak = None  # letters need no phonemiser
        elif LANG == "el":
            self.tts = lang_el.TTS() if tts else None
            self.espeak = _espeak("el")
        else:
            # tts=False: no model voice (hosted: references and audio are precomputed for the bank)
            if tts:
                from kokoro import KPipeline
                self.tts = KPipeline(lang_code="a")
            else:
                self.tts = None
            self.espeak = _espeak()
        self.lock = threading.Lock()  # GPU / model calls are not re-entrant
        self.espeak_lock = threading.Lock()  # phonemizer's espeak wrapper is not thread-safe
        self.ref_cache = {}
        if LANG in ("en", "nb", "el"):  # the bank's references are precomputed (tools/bank_build.py)
            import paragraphs
            for t, b in paragraphs.META.items():
                if "ref" in b:
                    r = b["ref"]
                    self.ref_cache[t] = dict(wp=[tuple(x) for x in r["wp"]], prons=[tuple(x) for x in r["prons"]],
                                             ems=[], extra=r.get("extra"), silent=r.get("silent"),
                                             ratio=r.get("ratio"), scores=r["scores"], duration=r["duration"])
        self.variant_ids = {
            p: [self.vocab[v] for v in vs if v in self.vocab] for p, vs in VARIANTS.items()
        }

    # ---------- text -> phones ----------
    def words_and_phones(self, text):
        """Returns list of (word, [phones]) in reading order. Words espeak can't
        phonemise into vocab tokens get an empty phone list and are skipped in scoring.

        The whole text is phonemised at once so function words get their connected-
        speech (weak) forms: "and" -> ənd, "the" before a vowel -> ði. If espeak's word
        count doesn't match ours (numbers, abbreviations), fall back to per-word."""
        if LANG == "nb":
            return [(w, [p for p in alts[0] if p != "|"]) for w, alts, _ in self.word_prons(text)]
        if LANG == "el":
            return [(w, p) for w, (p,), _ in self.word_prons(text)]
        return self._espeak_words(text)

    def letter_prons(self, text):
        """Norwegian: each word's letters, then the word separator '|' (aligned, not scored)."""
        words = tokenize(text)[0]
        out = []
        for i, w in enumerate(words):
            letters = lang_nb.letters(w, self.vocab)
            if i < len(words) - 1:
                letters = letters + ["|"]
            out.append((w, [letters], bool(letters) and w.lower() not in lang_nb.FUNCTION_WORDS))
        return out

    def letter_info(self, wp):
        """Norwegian, per aligned token of wp: (silent, extra accepted token ids). Silent letters
        (tidlig, hvitt, lyset) are aligned but not scored; listed vowel variants (lunsj with ø)
        count as the expected letter. Without the lexicon nothing is marked."""
        out = []
        for w, phs in wp:
            lets = [p for p in phs if p != "|"]
            marks = lang_nb.letter_marks(lets, self.lex.lookup(w)) if self.lex and lets else [(False, [])] * len(lets)
            it = iter(marks)
            for p in phs:
                s, alts = (False, []) if p == "|" else next(it)
                out.append((s, [self.vocab[a] for a in alts if a in self.vocab]))
        return out

    def word_prons(self, text):
        """[(word, [phone sequence], scored)]. English: espeak phones. Norwegian: letters;
        short function words are aligned but not scored."""
        if LANG == "nb":
            return self.letter_prons(text)
        if LANG == "el":
            return [(w, [lang_el.fix_phones(p)], w.lower() not in lang_el.FUNCTION_WORDS)
                    for w, p in self._espeak_words(text)]
        return [(w, [p], True) for w, p in self._espeak_words(text)]

    def _espeak_words(self, text, separately=False):
        with self.espeak_lock:
            return self._espeak_words_unlocked(text, separately)

    def _espeak_words_unlocked(self, text, separately=False):
        words = tokenize(text)[0]
        from phonemizer.separator import Separator
        sep = Separator(phone=" ", word="|", syllable="")
        whole = self.espeak.phonemize([" ".join(words)], separator=sep, strip=True)[0]
        per_word = [w.strip() for w in whole.split("|") if w.strip()]
        if separately or len(per_word) != len(words):
            per_word = self.espeak.phonemize(words, separator=sep, strip=True)
        out = []
        for w, p in zip(words, per_word):
            phones = []
            for ph in p.replace("|", " ").split():
                phones.extend(self._to_vocab(ph))
            out.append((w, phones))
        return out

    def _to_vocab(self, ph):
        if ph in self.vocab:
            return [ph]
        # split unknown multi-char tokens greedily into known ones
        res, i = [], 0
        while i < len(ph):
            for j in range(len(ph), i, -1):
                if ph[i:j] in self.vocab:
                    res.append(ph[i:j])
                    i = j
                    break
            else:
                i += 1  # drop unknown char (stress / length marks)
        return res

    # ---------- audio -> scores ----------
    def emissions(self, audio):
        with self.lock, torch.no_grad():
            x = self.fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values
            logits = self.model(x.to(self.device)).logits
            return torch.log_softmax(logits.float(), dim=-1)[0]

    # ---------- emissions in fixed pieces (hosted: computed while the reader is still reading)
    def emissions_piece(self, audio, k):
        """Emissions of piece k (frames [k*PIECE, (k+1)*PIECE)) of a recording, computed from
        the piece plus CONTEXT frames of audio on each side. Fixed piece boundaries make the
        result independent of how the audio arrived."""
        f0, f1 = k * PIECE, min((k + 1) * PIECE, n_frames(len(audio)))
        a = max(0, f0 - CONTEXT) * HOP
        b = min(len(audio), (f1 + CONTEXT) * HOP + WIN - HOP)
        em = self.emissions(audio[a:b])
        return em[f0 - a // HOP: f1 - a // HOP]

    def emissions_pieces(self, audio, done=None, final=True):
        """Pieces of the recording so far: those whose right context has arrived, or if final
        everything. done: {k: emissions} already computed (reused, extended in place).
        At the end, all pieces still missing are run as one window (from CONTEXT before the
        first of them to the end of the recording): the reader is waiting, and one window
        costs less than several overlapping ones. done[k] then holds frames k*PIECE to the end."""
        done = {} if done is None else done
        n = len(audio)
        k = 0
        while k * PIECE < n_frames(n):
            if k not in done:
                if final:
                    f0 = k * PIECE
                    a = max(0, f0 - CONTEXT) * HOP
                    done[k] = self.emissions(audio[a:])[f0 - a // HOP: n_frames(n) - a // HOP]
                    break
                if n < ((k + 1) * PIECE + CONTEXT) * HOP + WIN - HOP:
                    break  # its right context hasn't arrived yet
                done[k] = self.emissions_piece(audio, k)
            k += 1
        return done

    def align(self, audio, wp, em=None, extra=None):
        """Force-align audio (or precomputed emissions) to the phone sequence. Returns per-phone dicts.
        extra: optional per-token lists of additional accepted token ids."""
        flat = [(wi, ph) for wi, (_, phs) in enumerate(wp) for ph in phs]
        if not flat:
            return []
        if em is None:
            em = self.emissions(audio)
        T = em.shape[0]
        tokens = torch.tensor([[self.vocab[p] for _, p in flat]], dtype=torch.int32, device=em.device)
        if T < tokens.shape[1]:
            return None  # recording far too short for the text
        with self.lock:
            ali, _ = torchaudio.functional.forced_align(em[None], tokens, blank=self.blank)
        ali = ali[0].cpu().numpy()
        # spans of each target token: index of target advances on each non-blank change
        spans, prev, k = [], self.blank, -1
        for t, a in enumerate(ali):
            if a != self.blank and (a != prev or (k >= 0 and spans[k][1] < t - 1)):
                # new token (repeated identical tokens are separated by a blank)
                k += 1
                spans.append([t, t + 1])
            elif a != self.blank:
                spans[k][1] = t + 1
            prev = a
        assert len(spans) == len(flat), (len(spans), len(flat))
        post = em.exp().cpu().numpy()
        nb = 1.0 - post[:, self.blank]
        results = []
        for i, ((wi, ph), (s, e)) in enumerate(zip(flat, spans)):
            # window: extend halfway into neighbouring blank gaps, max 3 frames each side
            lo = s if i == 0 else max(s - 3, (spans[i - 1][1] + s) // 2)
            hi = e if i == len(spans) - 1 else min(e + 3, (e + spans[i + 1][0] + 1) // 2)
            w = post[lo:hi].copy()
            weight = nb[lo:hi]
            w[:, self.blank] = 0
            q = (w * (weight / np.maximum(weight.sum(), 1e-6))[:, None]).sum(0)
            q = q / max(q.sum(), 1e-9)
            tid = self.vocab[ph]
            acc_ids = [tid] + self.variant_ids.get(ph, []) + (extra[i] if extra else [])
            if i > 0 and (flat[i - 1][1], ph) in CONTEXT_VARIANTS:
                acc_ids += [self.vocab[v] for v in CONTEXT_VARIANTS[(flat[i - 1][1], ph)]]
            score = float(q[acc_ids].sum())
            exact = float(q[tid])
            q2 = q.copy()
            q2[acc_ids] = 0
            comp = int(q2.argmax())
            top = np.argsort(q)[::-1][:6]
            # audible segment: from the previous phone's spike to the next one's (for playback)
            seg_lo = spans[i - 1][1] if i > 0 else max(0, s - 8)
            seg_hi = spans[i + 1][0] if i + 1 < len(spans) else min(len(post), e + 8)
            results.append(dict(word=wi, phone=ph, start=s * FRAME_SEC, end=e * FRAME_SEC,
                                score=score, exact=exact,
                                heard=self.id2phone[comp], heard_p=float(q2[comp]),
                                dist={self.id2phone[int(j)]: round(float(q[j]), 3) for j in top if q[j] > 0.01},
                                q=q,
                                seg=(seg_lo * FRAME_SEC, seg_hi * FRAME_SEC)))
        return results

    def word_ratios(self, em, wp, scored, ali):
        """Norwegian word scores. For each scored word: the CTC log-likelihood of the whole
        utterance with the word as spelled, minus the best competitor (lang_nb.alternatives,
        or what the recogniser transcribes over the word's frames), other words fixed, per
        frame of the word. Negative = a competitor explains the audio better.
        Returns per word None or dict(ratio, alt, slip, diff={letter index: competitor's})."""
        from difflib import SequenceMatcher
        seqs = [[self.vocab[p] for p in phs] for _, phs in wp]
        greedy = em.argmax(-1).tolist()
        spans = {}
        for r in ali:
            if r["phone"] != "|":
                lo, hi = spans.get(r["word"], (r["start"], r["end"]))
                spans[r["word"]] = (min(lo, r["start"]), max(hi, r["end"]))
        out = []
        for i, (w, phs) in enumerate(wp):
            lets = [p for p in phs if p != "|"]
            if not scored[i] or not lets or i not in spans:
                out.append(None)
                continue
            marks = lang_nb.letter_marks(lets, self.lex.lookup(w)) if self.lex else [(False, [])] * len(lets)
            alts = set(lang_nb.alternatives(lets, marks))
            f0, f1 = int(spans[i][0] / FRAME_SEC), int(round(spans[i][1] / FRAME_SEC))
            heard, prev = [], None
            for t in greedy[max(f0 - 2, 0):f1 + 3]:
                if t != prev and t != self.blank:
                    heard.append(self.id2phone[t])
                prev = t
            free = "".join(heard).strip("|").replace("|", "")
            if not (free and free != "".join(lets) and all(c in self.vocab for c in free)):
                free = None
            elif all(marks[k][0] for k in lang_nb.letter_diff(lets, free)):
                free = None  # only differs in silent letters (været said as "være")
            if free in alts:
                free = None  # already a listed confusion
            alts = sorted(a for a in alts | {free} - {None} if not self._homophone(w, a))
            if not alts:
                out.append(dict(ratio=None, alt=None, slip=False, diff={}))  # nothing to confuse it with
                continue
            tail = ["|"] if phs[-1:] == ["|"] else []
            cands = [seqs[i]] + [[self.vocab[c] for c in list(a) + tail] for a in alts]
            before = [t for sq in seqs[:i] for t in sq]
            after = [t for sq in seqs[i + 1:] for t in sq]
            targets = [before + c + after for c in cands]
            L = max(map(len, targets))
            tg = torch.tensor([t + [0] * (L - len(t)) for t in targets], device=em.device)
            with self.lock:
                nll = torch.nn.functional.ctc_loss(
                    em[:, None, :].expand(-1, len(cands), -1), tg,
                    torch.full((len(cands),), em.shape[0], dtype=torch.long),
                    torch.tensor([len(t) for t in targets]), blank=self.blank, reduction="none").cpu().numpy()
            margin = (nll[1:] - nll[0]) / max(f1 - f0, 1)
            # the recogniser's own transcript must win by more: endings blur in connected speech
            # (forskning heard as "forskninge"), and it was not chosen as a likely confusion
            adj = margin + np.array([lang_nb.FREE_MARGIN if a == free else 0.0 for a in alts])
            k = int(np.argmin(adj))
            alt = alts[k]
            slip = SequenceMatcher(None, "".join(lets), alt, autojunk=False).ratio() < lang_nb.SLIP_SIMILARITY
            out.append(dict(ratio=float(adj[k]), alt=alt, slip=slip, diff=lang_nb.letter_diff(lets, alt),
                            conf=min((float(m) for a, m in zip(alts, margin) if a != free), default=None),
                            free=float(margin[alts.index(free)]) if free in alts else None, free_alt=free))
        return out

    def _homophone(self, word, alt):
        """True if the lexicon gives the two spellings a common pronunciation (hvitt/vitt)."""
        if not self.lex:
            return False
        pa = self.lex.lookup(alt) if alt in self.lex.memo or self.lex._get(alt) else []
        if not pa:
            return False
        pw = {" ".join(_strip_len(p)) for p in self.lex.lookup(word)}
        return any(" ".join(_strip_len(p)) in pw for p in pa)

    def nb_word_scores(self, em, wp, scored, ali, ref):
        """Norwegian: word_ratios mapped to scores (0.5 = flag line). Words that even the
        model voices don't win against a competitor are not scored."""
        out = []
        ref_ratio = ref.get("ratio") or [None] * len(wp)
        for wv, rr in zip(self.word_ratios(em, wp, scored, ali), ref_ratio):
            if wv is None or (rr is not None and rr < lang_nb.REF_MIN_RATIO):
                out.append(None)
                continue
            r = wv["ratio"]
            score = 1.0 if r is None else float(np.clip(0.5 + 0.5 * r / lang_nb.LR_SCALE, 0.0, 1.0))
            out.append({**wv, "score": score})
        return out

    def prewarm(self, log=print):
        """Hosted (AWS Lambda): read the weight files once, sequentially, and run one tiny
        inference. The weights are memory-mapped, and on Lambda's lazily loaded image the first
        reading otherwise pages them in piece by piece (measured: 118 s instead of ~7 s)."""
        import glob
        import time
        from huggingface_hub.constants import HF_HUB_CACHE
        t = time.time()
        repo = "models--" + PHONEME_MODEL_ID.replace("/", "--")
        files = ([MODEL_FILE] if MODEL_FILE else []) or \
            glob.glob(f"{HF_HUB_CACHE}/{repo}/snapshots/*/model.safetensors") or \
            glob.glob(f"{HF_HUB_CACHE}/{repo}/snapshots/*/pytorch_model.bin")
        n = 0
        for f in files:
            with open(f, "rb") as fh:
                while chunk := fh.read(1 << 24):
                    n += len(chunk)
        self.emissions(np.zeros(SAMPLE_RATE, np.float32))
        log(f"prewarmed {n / 1e9:.2f} GB of weights in {time.time() - t:.1f} s")

    # ---------- TTS reference ----------
    def tts_audio(self, text, voice, speed=1.0, rate=SAMPLE_RATE):
        if LANG != "en":
            a, sr = self.tts.audio(text, voice, speed)
            g = math.gcd(sr, rate)
            return resample_poly(a, rate // g, sr // g).astype(np.float32)
        with self.lock:
            chunks = [a.numpy() for _, _, a in self.tts(text, voice=voice, speed=speed)]
        a = np.concatenate(chunks) if chunks else np.zeros(TTS_RATE, np.float32)
        if rate != TTS_RATE:
            a = resample_poly(a, rate, TTS_RATE)
        return a.astype(np.float32)

    def reference(self, text):
        """Per-phone reference scores (median over TTS voices) + reference duration."""
        if text in self.ref_cache:
            return self.ref_cache[text]
        prons = self.word_prons(text)
        wp = [(w, alts[0]) for w, alts, _ in prons]
        info = self.letter_info(wp) if LANG == "nb" else None
        extra = [x for _, x in info] if info else None
        ems, durs, ratios = [], [], []
        for v in REF_VOICES:
            em = self.emissions(trim_silence(self.tts_audio(text, v))[0])
            r = self.align(None, wp, em=em, extra=extra)
            if r:
                if LANG == "nb":
                    ratios.append([x and x["ratio"] for x in self.word_ratios(em, wp, [t for _, _, t in prons], r)])
                ems.append(em.cpu())
                durs.append(r[-1]["end"] - r[0]["start"])
        # Norwegian: per word, the model voices' best ratio (None = nothing to compete with)
        ref_ratio = [max((x[i] for x in ratios if x[i] is not None), default=None)
                     for i in range(len(wp))] if ratios else None
        ref = dict(wp=wp, prons=prons, ems=ems, extra=extra, ratio=ref_ratio,
                   silent=[s for s, _ in info] if info else None, scores=self.ref_scores(ems, wp, extra),
                   duration=float(np.median(durs)) if durs else None)
        self.ref_cache[text] = ref
        return ref

    def ref_scores(self, ems, wp, extra=None):
        """Median per-phone score of the reference voices for this phone sequence."""
        runs = [[x["score"] for x in r] for r in (self.align(None, wp, em=em, extra=extra) for em in ems) if r]
        return np.median(np.array(runs), 0).tolist() if runs else []

    # ---------- full analysis ----------
    def analyse(self, text, audio, rate=SAMPLE_RATE, pieces=None):
        """pieces: {k: emissions} from emissions_pieces (hosted): the model runs on fixed pieces
        of the whole recording (mostly while it was being made) instead of once on all of it."""
        if rate != SAMPLE_RATE:
            g = math.gcd(int(rate), SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, int(rate) // g).astype(np.float32)
        ref = self.reference(text)
        wp = ref["wp"]
        punct = tokenize(text)[1]
        trimmed, offset = trim_silence(audio)
        if len(trimmed) < SAMPLE_RATE * 0.5 or np.abs(trimmed).max() < 1e-3:
            return dict(error="Didn't hear anything. Check the microphone.", code="silent")
        if ref["duration"] and len(trimmed) / SAMPLE_RATE < 0.35 * ref["duration"]:
            return dict(error="That recording is much shorter than the paragraph. "
                              "Read the whole text, then press Space.", code="short")
        if pieces is None:
            em = self.emissions(trimmed)
        else:
            pieces = self.emissions_pieces(audio, pieces)
            em_all = torch.cat([pieces[k] for k in sorted(pieces)])
            f0 = int(round(offset * SAMPLE_RATE)) // HOP  # trim_silence cuts at frame boundaries
            em = em_all[f0: f0 + n_frames(len(trimmed))]
        res = self.align(None, wp, em=em, extra=ref.get("extra"))
        if not res:
            return dict(error="Recording too short for this paragraph.", code="short")
        trusted = [t for _, _, t in ref["prons"]]
        nb_words = self.nb_word_scores(em, wp, trusted, res, ref) if LANG == "nb" else None
        phones = []
        silent = ref.get("silent") or [False] * len(res)
        letter_no = {}  # Norwegian: index of each letter within its word
        for r, rs, sil in zip(res, ref["scores"], silent):
            reliable = rs >= 0.25 and trusted[r["word"]] and r["phone"] != "|" and not sil
            # normalise: matching the TTS voices' clarity counts as 1.0
            norm = min(1.0, r["score"] / max(rs, 0.5)) if reliable else None
            if norm is not None:
                norm = calibrate(norm)
            if r["heard"] == "|":  # heard a word boundary: the sound was dropped
                r["heard"] = NOTHING
            if nb_words is not None and r["phone"] != "|":
                # letters carry the word's score where its best competitor differs, else 1.0
                k = letter_no[r["word"]] = letter_no.get(r["word"], -1) + 1
                wv = nb_words[r["word"]]
                if wv is None or wv["slip"] or sil:
                    norm = None
                elif k in wv["diff"]:
                    norm, r["heard"] = wv["score"], wv["diff"][k] or NOTHING
                else:
                    norm, r["heard"] = 1.0, r["phone"]
            r = {k: v for k, v in r.items() if k != "q"}
            phones.append({**r, "start": r["start"] + offset, "end": r["end"] + offset,
                           "ref": rs, "norm": norm})
        # geminates across a word boundary ("want to", "big girl") are released once
        from stats import VOWELS
        for a, b in zip(phones, phones[1:]):
            if (a["word"] != b["word"] and a["phone"] == b["phone"] and a["phone"] not in VOWELS
                    and a["norm"] is not None and b["norm"] is not None):
                a["norm"] = b["norm"] = max(a["norm"], b["norm"])
        words = []
        for wi, (w, phs) in enumerate(wp):
            mine = [p for p in phones if p["word"] == wi and p["phone"] != "|"]
            scored = [p["norm"] for p in mine if p["norm"] is not None]
            if nb_words is not None:  # Norwegian: the word's own score, not its worst letter
                wv = nb_words[wi]
                scored = [wv["score"]] if wv else []
            words.append(dict(
                word=w,
                punct=punct[wi],
                phones=[dict(phone=p["phone"], norm=None if p["norm"] is None else round(p["norm"], 2),
                             heard=p["heard"]) for p in mine],
                score=float(min(scored)) if scored else None,
                mean=float(np.mean(scored)) if scored else None,
                start=mine[0]["start"] if mine else None,
                end=mine[-1]["end"] if mine else None,
                bad=[dict(phone=p["phone"], heard=p["heard"], norm=round(p["norm"], 2))
                     for p in mine if p["norm"] is not None and p["norm"] < 0.5],
            ))
            if nb_words is not None and nb_words[wi]:
                words[-1]["ratio"] = None if nb_words[wi]["ratio"] is None else round(nb_words[wi]["ratio"], 3)
            if nb_words is not None and nb_words[wi] and nb_words[wi]["alt"]:
                # what it sounded like (only said when it won), and whether that was a misreading
                if nb_words[wi]["score"] < 0.5:
                    words[-1]["heard_as"] = nb_words[wi]["alt"]
                    words[-1]["slip"] = nb_words[wi]["slip"]
        coverage = check_coverage(res, len(trimmed) / SAMPLE_RATE)
        # stopping early makes alignment stretch the paragraph over too little audio
        if ref["duration"] and len(trimmed) / SAMPLE_RATE < 0.6 * ref["duration"]:
            coverage["maybe_incomplete"] = True
        return dict(words=words, phones=phones, audio_sec=len(audio) / SAMPLE_RATE,
                    stats={**fluency_stats(words, ref["duration"] and ref["duration"] / PLAY_SPEED), **coverage})


# Raw score at which a sound is flagged; calibrate() maps it to 0.5 so the rest of the app
# (word colours, bad lists, stats) keeps using 0.5. Chosen on REAL speech:
#   nb: not used; Norwegian scores whole words (Scorer.nb_word_scores, lang_nb.LR_SCALE)
#   en: phone GOP flagging ~15% of words experts call correct in adult learner speech
#       (tests/en_speechocean.py, speechocean762); 0.5 uncalibrated flagged 37%. The old
#       "dominant competitor caps at 0.45" rule was dropped: it only added false flags.
#   el: phone GOP flagging ~10% of real native words (tests/el_fleurs.py, FLEURS el_gr)
CALIB_T = float(os.environ.get("CALIB_T", {"nb": 0.55, "el": 0.016}.get(LANG, 0.03)))


def calibrate(raw, t=CALIB_T):
    """Map the raw score so the flag threshold sits at 0.5 (piecewise linear)."""
    return 0.5 * raw / t if raw < t else 0.5 + 0.5 * (raw - t) / (1 - t)


def _strip_len(pron):
    """Pronunciation without length and syllabic marks (homophone check)."""
    return [p.replace("ː", "").replace("\u0329", "") for p in pron]


HOP, WIN = 320, 400            # wav2vec2: one frame per 320 samples (20 ms), 400-sample receptive field
PIECE = int(os.environ.get("PIECE_FRAMES", 200))     # 4 s pieces
CONTEXT = int(os.environ.get("CONTEXT_FRAMES", 100))  # 2 s of audio either side (1 s: more words flip)


def n_frames(n_samples):
    return max(0, (n_samples - WIN) // HOP + 1)


def trim_silence(audio, thresh_db=-40):
    """Cut leading/trailing silence. Returns (audio, offset_seconds)."""
    frame = 320
    n = len(audio) // frame
    if n == 0:
        return audio, 0.0
    e = (audio[: n * frame].reshape(n, frame) ** 2).mean(1)
    db = 10 * np.log10(e + 1e-10)
    loud = np.where(db > db.max() + thresh_db)[0]
    if len(loud) == 0:
        return audio, 0.0
    s = max(0, (loud[0] - 10) * frame)
    t = min(len(audio), (loud[-1] + 10) * frame)
    return audio[s:t], s / SAMPLE_RATE


def check_coverage(res, total_sec):
    """Forced alignment always places every phone somewhere. If the reader stopped
    early, the tail gets squeezed into a few frames at the end; detect that."""
    tail = [r for r in res if r["end"] > total_sec - 0.3]
    squeezed = len(tail) > 12
    return dict(maybe_incomplete=squeezed)


def fluency_stats(words, ref_dur):
    timed = [w for w in words if w["start"] is not None]
    if len(timed) < 2:
        return {}
    dur = timed[-1]["end"] - timed[0]["start"]
    hesitations, pauses = [], 0
    for a, b in zip(timed, timed[1:]):
        gap = b["start"] - a["end"]
        if a["punct"]:
            pauses += gap > 0.15
        elif gap > 0.45:
            hesitations.append(dict(after=a["word"], before=b["word"], sec=round(gap, 2)))
    scored = [w["mean"] for w in words if w["mean"] is not None]
    clear = [w for w in words if w["score"] is not None and w["score"] >= 0.5]
    n_scored = sum(1 for w in words if w["score"] is not None)
    return dict(
        seconds=round(dur, 1),
        wpm=round(60 * len(words) / max(dur, 1e-3)),
        ref_wpm=round(60 * len(words) / ref_dur) if ref_dur else None,
        clarity=round(100 * float(np.mean(scored))) if scored else None,
        words_clear=len(clear),
        words_scored=n_scored,
        hesitations=hesitations,
    )



# ---------- optional recognizer: ZIPA (Zipformer CR-CTC over IPA characters) ----------
ZIPA_REPO = "anyspeech/zipa-large-crctc-500k"
# espeak symbols ZIPA spells differently (its tokens are single IPA characters)
ZIPA_MAP = {"ɡ": "g", "ɚ": "ə˞", "ɝ": "ɜ˞", "ᵻ": "ɨ", "ɐ": "ɐ", "r": "r", "ɬ": "ɬ"}


def use_zipa(scorer, repo=ZIPA_REPO):
    """Swap the recognizer for ZIPA (experiment; see tests/en_speechocean.py). Multi-
    character phones are split into ZIPA's characters (oʊ -> o ʊ, iː -> i ː); accepted
    variants then only apply to single-character phones."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from lhotse.features.kaldi.extractors import Fbank, FbankConfig
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(hf_hub_download(repo, "model.onnx"), providers=providers)
    vocab = {}
    for line in open(hf_hub_download(repo, "tokens.txt"), encoding="utf-8"):
        tok, i = line.split()
        vocab[tok] = int(i)
    fbank = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False))

    def emissions(audio):
        with scorer.lock:
            f = fbank.extract_batch([torch.from_numpy(np.asarray(audio, np.float32))], sampling_rate=SAMPLE_RATE)[0]
            lp = sess.run(None, {"x": f[None].numpy(), "x_lens": np.array([f.shape[0]], np.int64)})[0][0]
        return torch.log_softmax(torch.from_numpy(lp).float(), dim=-1)

    def to_vocab(ph):
        ph = "".join(ZIPA_MAP.get(c, c) for c in ph)
        return [c for c in ph if c in vocab]

    scorer.emissions = emissions
    scorer._to_vocab = to_vocab
    scorer.vocab = vocab
    scorer.id2phone = {v: k for k, v in vocab.items()}
    scorer.blank = vocab["<blk>"]
    scorer.variant_ids = {p: [vocab[c] for v in vs for c in to_vocab(v)][:8] for p, vs in VARIANTS.items() if len(to_vocab(p)) == 1}
    scorer.ref_cache = {}
    return scorer
