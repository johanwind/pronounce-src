#!/usr/bin/env python3
"""
Pronunciation practice tool.
Shows a word/phrase, records you saying it, gives per-phoneme confidence scores
using CTC forced alignment (not phoneme recognition).
"""

import json
import math
import random
import threading
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from g2p_en import G2p

SAMPLE_RATE = 16000
TTS_RATE = 24000
VOICE = "am_michael"
REF_VOICES = ["am_michael", "am_liam", "am_adam"]
TTS_SPEED = 1.0
PHONEME_MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"

# Score thresholds (probability 0-1)
GOOD_THRESH = 0.5
OK_THRESH = 0.2
REF_MIN = 0.01  # ignore phonemes where even TTS scores below this

# ARPABET → IPA mapping for the phoneme model's vocabulary
ARPA2IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɚ', 'EY': 'eɪ',
    'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'iː', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p',
    'R': 'ɹ', 'S': 's', 'SH': 'ʃ', 'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'uː',
    'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
}

CATEGORIES = {
    "short vowels": [
        # ɪ (pit)
        "pit", "big", "did", "win",
        # ɛ (pet)
        "pet", "bed", "red", "yes",
        # æ (pat)
        "bad", "map", "land", "ran",
        # ʌ (cup)
        "but", "love", "come", "run",
        # ʊ (foot)
        "book", "good", "look", "could",
        # ɔ (thought)
        "dog", "law", "long", "gone",
    ],
    "æ": [
        # æ — best ref scores
        "rat", "bat", "gas", "mat", "sand", "land", "back", "bag",
    ],
    "ʌ": [
        # ʌ — best ref scores
        "blood", "love", "come", "bug", "run", "gun", "but", "rub",
    ],
    "ʊ": [
        # ʊ — best ref scores (only look/book are reliable)
        "look", "book",
    ], 
    "ɪ": [
        # ɪ — best ref scores
        "tin", "bit", "pin", "lip", "him", "win", "dig", "rim",
    ],
    "long vowels": [
        # iː (fleece)
        "need", "keep", "leave", "green",
        # eɪ (face)
        "face", "make", "rain", "late",
        # ɑ (palm)
        "hot", "stop", "rock", "drop",
        # oʊ (goat)
        "go", "home", "road", "bone",
        # uː (goose)
        "food", "blue", "move", "moon",
        # ɚ (nurse)
        "bird", "turn", "word", "learn",
    ],
    "diphthongs": [
        # aɪ (price)
        "my", "time", "night", "like",
        # aʊ (mouth)
        "out", "house", "loud", "down",
        # ɔɪ (choice) — model scores this poorly, but ratio scoring handles it
        "join", "voice", "point", "noise",
    ],
    "stops": [
        "pat bat", "tap dab", "cap gap",
        "pet bet", "ten den", "kit get",
        "rip rib", "seat seed", "back bag",
    ],
    "fricatives": [
        # f/v
        "fan van", "safe save", "life live",
        # θ/ð
        "think this", "math bathe", "breath breathe",
        # s/z
        "sip zip", "bus buzz", "rice rise",
        # ʃ/ʒ
        "wish", "vision", "measure",
    ],
    "other consonants": [
        # tʃ/dʒ (affricates)
        "catch", "jump", "judge", "edge",
        # m/n/ŋ (nasals)
        "sum sun sung", "sin sing", "ram ran rang",
        # l/ɹ (liquids)
        "lap rap", "led red", "long wrong",
        # w/j (glides)
        "wet yet", "wine vine", "west vest",
    ],
    "minimal pairs": [
        # vowel contrasts
        "sit seat", "pet pat", "bed bad",
        "men man", "put food", "cup cot",
        # consonant contrasts
        "think fink", "three tree", "this dis",
        "sip ship", "cap gap",
        "lap rap", "light right",
    ],
    "sentences": [
        "the quick brown fox jumps over the lazy dog",
        "she sells seashells by the seashore",
        "how are you doing today",
        "nice to meet you",
        "would you like some water",
        "the weather is beautiful today",
        "i would have thought so",
        "please play with my purple puzzle",
        "fresh french fries taste fine",
        "the weather is better when wearing white",
        "many people have a family and a job",
        "can you give the big black bag back",
    ],
}



def load():
    print("Loading phoneme model... ", end="", flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL_ID)
    phon_model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    phon_model = phon_model.to(device).eval()

    vocab_path = hf_hub_download(PHONEME_MODEL_ID, "vocab.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    id2phone = {v: k for k, v in vocab.items()}
    print(f"done ({device})")

    print("Loading TTS model... ", end="", flush=True)
    tts = KPipeline(lang_code="a")
    print("done")

    print("Loading G2P... ", end="", flush=True)
    g2p = G2p()
    print("done")

    return fe, phon_model, device, vocab, id2phone, tts, g2p


def text_to_tokens(text, g2p, vocab):
    """Convert text to (token_ids, ipa_labels) using g2p and ARPABET→IPA mapping."""
    arpa = g2p(text)
    tokens, labels = [], []
    for p in arpa:
        if p == ' ':
            continue
        base = p.rstrip('012')
        stress = p[-1] if p[-1] in '012' else None
        if base not in ARPA2IPA:
            continue
        ipa = ARPA2IPA[base]
        # Unstressed AH → schwa
        if base == 'AH' and stress == '0':
            ipa = 'ə'
        if ipa in vocab:
            tokens.append(vocab[ipa])
            labels.append(ipa)
    return tokens, labels


def get_emissions(audio, fe, model, device):
    """Run audio through the CTC model, return log-softmax emissions."""
    inputs = fe(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.log_softmax(logits, dim=-1)


def align_scores(audio, tokens, fe, model, device, id2phone=None):
    """Force-align audio against expected token sequence.
    Returns (per-phoneme probabilities, recognized phonemes or None)."""
    emission = get_emissions(audio, fe, model, device)
    targets = torch.tensor([tokens], dtype=torch.int32).to(device)
    input_lengths = torch.tensor([emission.shape[1]]).to(device)
    target_lengths = torch.tensor([len(tokens)]).to(device)

    aligned_tokens, scores = torchaudio.functional.forced_align(
        emission, targets, input_lengths, target_lengths, blank=0
    )
    spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0])
    probs = [math.exp(s.score) for s in spans]

    # Also do recognition (argmax) to show what the user actually said
    recognized = None
    if id2phone is not None:
        ids = torch.argmax(emission[0], dim=-1).tolist()
        phones, prev = [], None
        pad_id = 0  # <pad> is always 0
        for i in ids:
            if i != pad_id and i != prev:
                phones.append(id2phone.get(i, "?"))
            prev = i
        recognized = phones

    return probs, recognized


def record():
    print("  Recording... press Enter to stop", flush=True)
    chunks = []

    def callback(indata, frames, time, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input()

    return np.concatenate(chunks).flatten() if chunks else np.zeros(SAMPLE_RATE, dtype="float32")


def play_recording(audio):
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def _tts_chunks(text, tts, speed=TTS_SPEED):
    chunks = [a.numpy() for _, _, a in tts(text, voice=VOICE, speed=speed)]
    return np.concatenate(chunks) if chunks else np.zeros(TTS_RATE, dtype=np.float32)


def play_tts(text, tts):
    sd.play(_tts_chunks(text, tts, speed=TTS_SPEED), TTS_RATE)
    sd.wait()


def tts_audio_16k(text, tts):
    """Return TTS audio resampled to 16kHz for alignment."""
    tts_text = text if " " in text else text.rstrip(".") + "."
    return resample_poly(_tts_chunks(tts_text, tts), SAMPLE_RATE, TTS_RATE).astype(np.float32)


def evaluate(labels, user_scores, ref_scores, recognized=None):
    """Display per-phoneme scores and return (perfect, phoneme_ratios).
    phoneme_ratios is a list of (label, ratio) for scored phonemes."""
    print()

    # Build display columns
    col_w = 5
    ph_row, ref_row, usr_row, mark_row = [], [], [], []

    good_count = 0
    scored_count = 0
    phoneme_ratios = []

    for i, label in enumerate(labels):
        w = max(len(label) + 1, col_w)
        ph_row.append(label.center(w))

        rs = ref_scores[i] if i < len(ref_scores) else 0
        us = user_scores[i] if i < len(user_scores) else 0

        # If reference itself scores poorly, this phoneme is unreliable
        if rs < REF_MIN:
            ref_row.append("—".center(w))
            usr_row.append("—".center(w))
            mark_row.append(" ".center(w))
            continue

        scored_count += 1
        ref_row.append(f"{rs:.2f}".center(w))
        usr_row.append(f"{us:.2f}".center(w))

        # Score relative to reference: ratio of user/ref
        ratio = us / rs if rs > 0 else 0
        phoneme_ratios.append((label, ratio))
        if ratio >= 0.6:
            mark_row.append("✓".center(w))
            good_count += 1
        elif ratio >= 0.3:
            mark_row.append("~".center(w))
        else:
            mark_row.append("✗".center(w))

    print(f"  Phoneme: {''.join(ph_row)}")
    print(f"  Ref:     {''.join(ref_row)}")
    print(f"  You:     {''.join(usr_row)}")
    print(f"           {''.join(mark_row)}")

    if recognized:
        print(f"  Heard:   {' '.join(recognized)}")

    if scored_count == 0:
        print("  (no scoreable phonemes)")
        return True, []

    pct = 100 * good_count // scored_count
    print(f"\n  Score: {good_count}/{scored_count} phonemes good ({pct}%)")
    return good_count == scored_count, phoneme_ratios


def pick_category():
    """Show category menu and return selected items."""
    names = list(CATEGORIES.keys())
    print("\n=== Pronunciation Practice ===")
    print("\n  Categories:")
    for i, name in enumerate(names, 1):
        count = len(CATEGORIES[name])
        print(f"    {i}. {name} ({count} items)")
    print(f"    {len(names) + 1}. all")

    while True:
        try:
            choice = input("\n  Pick a category (number): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not choice.isdigit():
            continue
        n = int(choice)
        if 1 <= n <= len(names):
            items = CATEGORIES[names[n - 1]].copy()
            print(f"\n  → {names[n - 1]} ({len(items)} items)")
            return items
        if n == len(names) + 1:
            items = [item for cat in CATEGORIES.values() for item in cat]
            print(f"\n  → all ({len(items)} items)")
            return items


def main():
    fe, phon_model, device, vocab, id2phone, tts, g2p = load()

    items = pick_category()
    if items is None:
        return
    random.shuffle(items)
    idx = 0

    print("\n  Enter=record  t=hear it  g=ref scores  s=skip  q=quit\n")

    last_audio = None
    ref_cache = {}  # target -> (tokens, labels, ref_scores)
    # Stats: per-item attempts and best score, per-phoneme ratios
    item_stats = {}   # target -> {"attempts": int, "best_pct": int, "perfect": bool}
    phoneme_stats = {}  # phoneme -> [ratio, ratio, ...]

    def get_ref(target):
        """Get tokens, labels, and TTS baseline scores (median across voices)."""
        if target not in ref_cache:
            tokens, labels = text_to_tokens(target, g2p, vocab)
            if not tokens:
                ref_cache[target] = ([], [], [])
            else:
                all_scores = []
                for voice in REF_VOICES:
                    tts_text = target if " " in target else target.rstrip(".") + "."
                    chunks = [a.numpy() for _, _, a in tts(tts_text, voice=voice, speed=1.0)]
                    if chunks:
                        audio = resample_poly(
                            np.concatenate(chunks), SAMPLE_RATE, TTS_RATE
                        ).astype(np.float32)
                        scores, _ = align_scores(audio, tokens, fe, phon_model, device)
                        all_scores.append(scores)
                if all_scores:
                    median_scores = []
                    for i in range(len(tokens)):
                        per_voice = [run[i] for run in all_scores]
                        med = float(np.median(per_voice))
                        # If median is too low but a voice scored well, use that
                        if med < REF_MIN:
                            med = max(per_voice)
                        median_scores.append(med)
                else:
                    median_scores = [0.0] * len(tokens)
                ref_cache[target] = (tokens, labels, median_scores)
        return ref_cache[target]

    def show_ref(target):
        tokens, labels, ref_scores = get_ref(target)
        if not labels:
            print("  (no phonemes for this text)")
            return
        print(f"  Reference phonemes: {' '.join(labels)}")
        print(f"  Reference scores:   {' '.join(f'{s:.2f}' for s in ref_scores)}")

    retrying = False
    custom_queue = []
    target = None
    while True:
        if not retrying:
            if custom_queue:
                target = custom_queue.pop(0)
                is_custom = True
            else:
                target = items[idx % len(items)]
                is_custom = False

        if not retrying:
            print(f"\n{'─' * 50}")
            print(f"  Say: {target.upper()}")
            print("  Enter=record  t=hear it  g=ref scores  c=custom  s=skip  q=quit")

            advance = False
            quit_flag = False
            while True:
                try:
                    cmd = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    quit_flag = True
                    break
                if cmd == "q":
                    quit_flag = True
                    break
                if cmd == "s":
                    if not is_custom:
                        idx += 1
                    advance = True
                    break
                if cmd == "t":
                    play_tts(target, tts)
                    continue
                if cmd == "g":
                    show_ref(target)
                    continue
                if cmd == "c":
                    try:
                        phrase = input("  Phrase: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        continue
                    if phrase:
                        custom_queue.insert(0, target)
                        custom_queue.insert(0, phrase)
                        advance = True
                    break
                if cmd == "":
                    break  # Enter → record
                continue  # unknown command → ignore

            if quit_flag:
                break
            if advance:
                continue

        retrying = False

        # Compute reference in background while user records
        ref_result = [None]
        ref_thread = threading.Thread(
            target=lambda: ref_result.__setitem__(0, get_ref(target)),
            daemon=True
        )
        ref_thread.start()

        last_audio = record()
        ref_thread.join()
        tokens, labels, ref_scores = ref_result[0]

        if not tokens:
            print("  (no phonemes for this text)")
            idx += 1
            continue

        user_scores, recognized = align_scores(last_audio, tokens, fe, phon_model, device, id2phone)

        print(f"\n  Target: {target.upper()}")
        perfect, phoneme_ratios = evaluate(labels, user_scores, ref_scores, recognized)

        # Track stats
        if target not in item_stats:
            item_stats[target] = {"attempts": 0, "best_pct": 0, "perfect": False}
        item_stats[target]["attempts"] += 1
        if phoneme_ratios:
            pct = int(100 * sum(1 for _, r in phoneme_ratios if r >= 0.6) / len(phoneme_ratios))
            item_stats[target]["best_pct"] = max(item_stats[target]["best_pct"], pct)
        if perfect:
            item_stats[target]["perfect"] = True
        for ph, ratio in phoneme_ratios:
            phoneme_stats.setdefault(ph, []).append(min(ratio, 1.5))

        if perfect:
            print("\n  Enter=next  p=play back  t=hear it  g=ref scores  c=custom  q=quit")
        else:
            print("\n  Enter=retry  n=next  p=play back  t=hear it  g=ref scores  c=custom  q=quit")
        while True:
            try:
                cmd = input("  > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                cmd = "q"

            if cmd == "q":
                idx = len(items)
                break
            if cmd == "p" and last_audio is not None:
                play_recording(last_audio)
                continue
            if cmd == "t":
                play_tts(target, tts)
                continue
            if cmd == "g":
                show_ref(target)
                continue
            if cmd == "c":
                try:
                    phrase = input("  Phrase: ").strip()
                except (EOFError, KeyboardInterrupt):
                    continue
                if not phrase:
                    continue
                custom_queue.insert(0, target)  # come back to current word
                custom_queue.insert(0, phrase)  # custom first
                break
            if cmd == "n":
                if not custom_queue:
                    idx += 1
                break
            if cmd == "r" or (cmd == "" and not perfect):
                retrying = True
                break
            if cmd == "" and perfect:
                if not custom_queue:
                    idx += 1
                break
            continue  # unknown command → ignore

        if idx >= len(items) and not custom_queue:
            break

    # Print session summary
    if not item_stats:
        return

    print(f"\n{'═' * 50}")
    print("  SESSION SUMMARY")
    print(f"{'═' * 50}")

    # Items that needed multiple attempts or never got perfect
    struggled = [(t, s) for t, s in item_stats.items() if not s["perfect"]]
    nailed = [(t, s) for t, s in item_stats.items() if s["perfect"]]

    if nailed:
        print(f"\n  Nailed ({len(nailed)}):")
        for target, s in nailed:
            att = s["attempts"]
            suffix = f" ({att} attempts)" if att > 1 else ""
            print(f"    ✓ {target}{suffix}")

    if struggled:
        print(f"\n  Needs work ({len(struggled)}):")
        for target, s in struggled:
            print(f"    ✗ {target} — best {s['best_pct']}% ({s['attempts']} attempts)")

    # Phoneme breakdown: average ratio per phoneme, sorted worst first
    if phoneme_stats:
        ph_avg = {}
        for ph, ratios in phoneme_stats.items():
            ph_avg[ph] = sum(ratios) / len(ratios)

        weak = [(ph, avg) for ph, avg in ph_avg.items() if avg < 0.6]
        weak.sort(key=lambda x: x[1])

        if weak:
            print(f"\n  Weakest phonemes:")
            for ph, avg in weak:
                n = len(phoneme_stats[ph])
                bar = "█" * int(avg * 10) + "░" * (10 - int(avg * 10))
                print(f"    {ph:4s} {bar} {avg:.0%}  ({n} samples)")

        strong = [(ph, avg) for ph, avg in ph_avg.items() if avg >= 0.6]
        strong.sort(key=lambda x: -x[1])
        if strong:
            print(f"\n  Strongest phonemes:")
            for ph, avg in strong[:10]:
                n = len(phoneme_stats[ph])
                capped = min(avg, 1.0)
                bar = "█" * int(capped * 10) + "░" * (10 - int(capped * 10))
                print(f"    {ph:4s} {bar} {min(avg, 1.5):.0%}  ({n} samples)")

    print()


if __name__ == "__main__":
    main()
