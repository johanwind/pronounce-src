# Read Aloud: paragraph pronunciation trainer

Read a short paragraph aloud and see which words came out unclear. You get your speed (words per minute), a clarity score, and three sound areas to focus on. The next paragraph is chosen, or written by Haiku, to drill those sounds.

```bash
cd ~/python/pronounce && .venv/bin/python reader.py
```

Then open http://localhost:8765 in a browser and allow the microphone. Startup takes about 15 s while the models load (≈2 GB RAM, GPU if available).

## Install on another computer

Copy `dist/pronounce.tar.gz` (code only, under 100 KB; build it with the commands at the end of this section), then on the other machine (Linux or macOS, Python 3.12, about 6 GB of disk):

```bash
tar xzf pronounce.tar.gz && cd pronounce
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python reader.py --lang el     # Greek on http://localhost:8767
```

`uv` can be swapped for `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`. On first start it downloads the recognizer (1.3 GB) and the voices (about 200 MB). A GPU is optional. The coach needs the `claude` CLI logged in to a Claude subscription; without it, focus areas and paragraphs come from local statistics. English additionally needs `pip install kokoro misaki[en]`; Norwegian needs nothing extra (its lexicon is optional).

Build the package: `mkdir -p dist/pronounce && cp *.py README.md requirements.txt dist/pronounce/ && cp -r static dist/pronounce/ && tar czf dist/pronounce.tar.gz -C dist pronounce`

## Using it

- **Space** starts and stops reading. **Esc** discards a recording, stops sound and closes popups. Press **?** (or the ? button) for all shortcuts: **H** hear the paragraph, **P** play back your reading, **←/→** step through unclear words, **M**/**Y**/**B** model voice / your version / both for the selected word, **D** drill the selected word's weakest sound, **A** read again, **N** next targeted paragraph, **Shift+N** random paragraph, **T** own text.
- **▶ Hear it** plays the paragraph in a US model voice, and **▶ My reading** plays back your last recording. While something is playing its button becomes **■ Stop** (Esc also stops it). Playback stops automatically when you start reading.
- After reading, unclear words are highlighted: amber means a bit off, red with a wavy underline means the model heard a different sound. `‖` marks a mid-sentence hesitation. Click a word to see its sounds, what was heard instead, and to play **the model voice vs your own recording** of just that word.
- **Focus on these** shows the three weakest sound areas, computed from this reading plus your last few readings (areas that are already clear are left out). If Haiku is reachable, its tips and a short drill replace the stats-based ones a few seconds later.
- **Next paragraph →** is written by Haiku for your focus areas, or otherwise picked from the built-in bank as the passage densest in those sounds. **Own text** lets you paste anything.
- The progress chart shows clarity and speed across readings. Your history and recordings are kept in `data/`.

## Sound drill (`/drill`)

Drill one sound with one word at a time. Open it from the "Sound drill →" link, from a focus card's "Drill … →" link, or with **D** on a selected word.

- **Hear (H)** plays the word from one of four model voices in rotation, so you learn the sound rather than one voice. **Sound only (V)** cuts just the target sound out of the model's word and plays it twice.
- **Space** records. Recording stops by itself when you go quiet. You then hear your word followed by the model's (turn this off with the checkbox). **Y** / **U** / **B** play your word, your sound only, or your word then the model's.
- **Feedback**: a score for the target sound, bars showing what it sounded like (e.g. 21% uː *food*), your vowel length next to the model voices', and a **vowel map**. The map shows where your sound landed (front↔back, tongue high↔jaw open) relative to the model voices, with a trail of your last tries. A hint suggests a direction to move.
- Words and contrast pairs (full vs fool, put vs pit…) exist for ʊ, uː, θ, ð, z, v, w, æ, ɪ, iː, ʌ, r, dʒ, ʃ, and you can type any word.

For the vowels ʊ uː ɪ iː ʌ ɑː ɔː oʊ ɛ æ, the drill doesn't use the recognizer's labels. Those are unreliable for some English vowels: a clean TTS "book" gets only ~5% ʊ. Instead it compares the recognizer's internal features (layers 6–12) of your vowel against **the same word with each vowel swapped in**, e.g. "look" as l-ʊ-k, l-uː-k, l-ɪ-k, l-ʌ-k… Kokoro synthesizes these from phonemes in 8 voices; it takes 5–7 s the first time a word is used, then it's cached, and the page prepares the next words in the background. A global bank of vowel prototypes adds non-English vowels (continental [u], French [y] and [ø], built from Spanish/Italian/Portuguese/French voices), so a Norwegian-style u has somewhere to go instead of counting as ʊ. A short vowel held more than 1.5× as long as the model voices' is capped at 60%.

Click ▶ next to a bar to hear the word spoken with that vowel, i.e. what you sounded like.

Evaluation (`tests/eval_drill_ood.py`, voices and languages never used to build the prototypes): for ʊ words, 90% of correct readings are accepted (score ≥ 0.7), while 0% of English near-misses (Luke, boot, luck, lick…) and 0% of foreign vowels are. For other target vowels, only about two thirds of correct readings by unseen voices score ≥ 0.7, so treat those scores as rougher. Drill attempts and their audio are saved in `data/drill.jsonl` and `data/drill_recordings/`.

## Coach (Haiku)

The coach calls the `claude` CLI with `--model haiku`, so it runs on your Claude subscription login (`claude auth login`), never on API credits; `ANTHROPIC_API_KEY` is ignored. Without a login, everything still works from local statistics.

## How scoring works (`scoring.py`)

1. The text is converted to American English IPA with espeak-ng, which is the same phone set the recognizer was trained on. The whole paragraph is processed at once, so weak forms ("and" → ənd) come out right.
2. `facebook/wav2vec2-lv-60-espeak-cv-ft` (CTC phoneme recognizer) gives frame-level phone probabilities. Forced alignment then places every expected phone in time.
3. For each phone, the score is the share of the model's (non-silence) belief in that time window that goes to the expected phone or an accepted native variant (flap t, reduced vowels, cot/caught…). The most likely other phone is reported as "heard".
4. Each phone's score is divided by what two TTS voices reading the same paragraph get at that position. Sounds the recognizer struggles with even in clean speech therefore don't count against you, and positions where even the TTS voices score low are skipped.

Set `PHONEME_MODEL=facebook/wav2vec2-xlsr-53-espeak-cv-ft` to use the previous recognizer.
5. Calibration: the normalised score is mapped so that a raw 0.03 becomes 0.5, the flag threshold. The value was chosen on real learner speech (below). Override it with `CALIB_T=...`.

**Validation on real learner speech** (`tests/en_speechocean.py`, speechocean762: Mandarin-L1 adults reading short sentences, every phone scored by 5 experts; 1,220 adult utterances). A word counts as mispronounced when most experts marked one of its phones wrong, and as correct when nearly all marked every phone right. AUC 0.91, word-score correlation with the experts' 0–10 accuracy r = 0.46, utterance clarity r = 0.67.

| setting (2026-11-19) | correct words flagged | mispronounced words flagged |
|---|---|---|
| old: raw 0.5 + "dominant competitor" cap | 37% | 96% |
| now: calibrated, raw 0.03, no cap | 17% | 82% |

Raw scores are bimodal: for 42% of mispronounced words some phone gets essentially zero probability, so the separation sits at the low end. The old substitution cap was dropped because it added 12 points of false flags for 10 points of catches, which is worse than simply moving the threshold. ZIPA (`anyspeech/zipa-large-crctc-500k`, `scoring.use_zipa`) was tried as the recognizer here too: it was clearly worse (AUC 0.77, catching 13% of errors vs 54% at 10% false flags on 600 utterances), so the app keeps wav2vec2. Caveats: the speakers are Mandarin-L1, not Norwegian-L1, the sentences are short, and the threshold was fitted on this same data.

Tests: `tests/test_scoring.py` checks scoring on synthetic speech with planted mispronunciations ("sink" for "think" etc.). Since the calibration, TTS "sink" and "wetter" no longer reliably flag (TTS keeps some θ/ð in them), while v→w swaps do. Real-speech numbers come first; the synthetic tests are only smoke tests. `tests/test_server.py URL` runs an end-to-end test against a running server; start that server with `PRONOUNCE_DATA=/tmp/x` so it doesn't touch your history. `tests/test_browser_mic.py URL` drives headless Chrome with a fake microphone playing `tests/fixtures/planted.wav`, which tests the real capture path.

## Norwegian (Bokmål)

```bash
cd ~/python/pronounce && .venv/bin/python reader.py --lang nb
```

Then open http://localhost:8766. It has its own history in `data/nb/`, 24 Norwegian paragraphs (`paragraphs_nb.py`), Norwegian focus areas (retroflex rt/rd/rn/rl/rs, kj, sj/skj, y/ø/u, o/å, æ, ei/øy/au, tapped r…) and a Norwegian coach prompt. The whole page is in Norwegian: labels (`STRINGS.nb` in `static/reader.html`), area names (`lang_nb.AREA_LABELS`; the English keys stay in history and the bank), tips and the coach's text. Same keys as English; the sound drill is English-only.

Scoring is different from English:

- **The recognizer spells.** Multilingual phone recognizers don't know Norwegian well: they hear ʈ as t, æ as a/ɛ, kj as sj and y as i, and on real native speech they barely separate minimal pairs (AUC ≈ 0.6–0.74). The National Library's Norwegian `NbAiLab/nb-wav2vec2-1b-bokmaal-v2` was trained on Norwegian speech with ordinary spelling as the target, and Norwegian spelling encodes the contrasts learners miss (y/i, ø/e, æ/e, o/å, kj/sj). It is a 4 GB download on first start, and takes 0.4 s per 20 s of audio and 4.3 GB of GPU memory on the laptop 4070 (`PHONEME_MODEL` overrides it).
- **Each word competes with its confusions.** For every word, the scorer asks whether its spelling explains the audio better than confusable spellings (y→i, ø→e, kj→sj, rt→t… in `lang_nb.CONFUSIONS`) and than what the recognizer itself transcribes over that stretch. The score is the log-likelihood margin of the spelling over its best competitor (CTC over the whole utterance, other words fixed, per frame): 0 maps to 0.5, the flag line. A flagged word says what it **sounded like** ("huset" sounded like "hyset"), and the differing letters are marked (u heard y). If the competitor is a different word altogether ("store" for "samme"), the word is shown as **read as** that word: a reading slip, which doesn't count against any sound area. The recognizer's own transcript needs a 0.5 margin to win, because endings blur in connected speech ("forskning" heard as "forskninge").
- The NB Uttale lexicon (`data/nb_lexicon.sqlite`) keeps accepted pronunciations from being flagged: confusions it accepts aren't tried ("lunsj" with ø, "votter" with å), homophones don't compete ("hvitt"/"vitt"), and changes to silent letters are ignored ("været" said as "være"). Short function words (er, det, og, med…) are aligned but not scored. Words that even the Piper voices (`vox`, `talesyntese`) lose to a competitor are not scored (about 4%), since the recognizer can't tell them apart.
- **Why not per-letter scores** (the method until 2026-11-22): letter posteriors around CTC spikes are noisy (samme came out as "m heard as o"). The method flagged 12% of native FLEURS words and 18% of a native speaker's readings, and the newer, better recognizers scored *worse* with it (AUC 0.78 vs 0.86 for the old `nb-wav2vec2-300m-bokmaal`).
- **Validation on real native speech** (`tests/nb_fleurs.py`, Google FLEURS nb dev, 75 utterances): 3.9% of native words are flagged (was 12%) and the median clarity is 98%. When a word in the text is swapped for a real minimal pair the speaker didn't say ("tre"→"trø", "sin"→"syn", "før"→"fer"), 96% of those words are flagged (was 76%). That number is optimistic, because the swaps are the same kinds of confusion the scorer tries. On a native speaker's 8 saved readings, 6 of 241 words are flagged (was 43), including two real misreadings ("store" for "samme", "dag" for "helg").
- **Weak spot: non-words.** The recognizer has learnt Norwegian vocabulary and pulls non-words towards real words: Piper saying "sjøpte" is transcribed "kjøpte". `tests/nb_planted_tts.py` plants clear non-word errors (kj→sj, y→i, ø→e, u→y, rt→t…) in the paragraphs and has Piper read them. The 1b-v2 model catches about 50% of them, while the old 300m v1 catches about 65% but flags 5.6% of native FLEURS words and 27 of the native readings' 236. The 1b model was chosen for its far fewer false flags on real speech. The synthetic numbers are only a relative guide. There is no real learner recording yet.
- **Not measured:** tonal accent (tone 1/2), vowel length, and whether rt/rn/… are truly retroflex (the letters are the same either way). Listen to the model and to yourself (B) for those.
- Voices: the playback voice (what **Hear it** and **M** play) is Google Cloud TTS `nb-NO-Chirp3-HD-Iapetus` at speaking rate 0.85 (`lang_nb.PLAY_VOICE`, made with the gcloud login; the old Piper clips are in `data/bank/audio_nb.piper`). The references the scorer compares with are still the Piper voices `vox` and `talesyntese`; they follow espeak's Norwegian front end, so the retroflexes may sound like r + t in places. The multi-speaker `nvcc` Piper voice mumbles consonants and is not used.

Files: `lang_nb.py` (voices, confusions, areas, coach prompt, letter patterns), `paragraphs_nb.py`; word scoring is `Scorer.word_ratios` in `scoring.py`. The NB Uttale lexicon (Språkbanken, CC0, East Norwegian) marks silent letters, accepted variants and homophones (`lang_nb.letter_marks`), and `tests/nb_fleurs.py` uses it to pick minimal pairs. Tests: `tests/nb_fleurs.py` (real speech), `tests/nb_planted_tts.py` (non-word errors, synthetic), `tests/test_server_nb.py URL` (end to end against a running `--lang nb` server). Build it with `.venv/bin/python lang_nb.py build` after unpacking `nb_uttale_leksika.zip` to `~/data/nb_uttale`. Piper voices download into `data/nb_voices/` on first use.

The old word-drill tool is still in `practice.py`.

## Greek (Modern Standard Greek)

Run `.venv/bin/python reader.py --lang el` and open http://localhost:8767. History is kept separately in `data/el/`. The page, area names, tips and coach are in Greek. The sound drill is English-only.

- **Scoring** uses the same pipeline as English: espeak-ng (`el`) turns the text into phones, and the multilingual phone recognizer (`wav2vec2-lv-60-espeak-cv-ft`) scores them. This works here, unlike Norwegian, because Greek spelling maps regularly onto five plain vowels and consonants the recognizer knows well (θ ð ɣ x ç ʝ all occur in its training languages), and there is one standard spoken form. Allophones are accepted (ʝ/j, ç/x, c/k, tap/trill r, and the recognizer's labels for Greek's mid vowels: ɐ for a, ʊ for o…). Written double consonants are said once, and "άι" counts as two vowels. Articles, clitics and short particles are aligned but not scored.
- **Voices** were measured by how well the recognizer hears them on 40 FLEURS sentences (phone error rate): Meta MMS-TTS `facebook/mms-tts-ell` 12.5%, Piper joy 19.7% (it says θ like f), rapunzelina 29.9%, rapunzelina-low 40.6%. Native speakers come out at 24%. References are MMS + joy. Playback (**Hear it**) is Google Cloud TTS `el-GR-Chirp3-HD-Iapetus` (the same male voice as Norwegian) at speaking rate 0.85, through the gcloud login (`lang_el.PLAY_VOICE`, falls back to MMS offline); the recognizer hears it at least as clearly as MMS (2 vs 7 flagged words on 5 passages).
- **Validation on real native speech** (`tests/el_fleurs.py`, FLEURS el_gr dev, 80 utterances). The test changes a sound the speaker didn't say in the expected phones (θ→t, ð→d, ɣ→g, χ→k, ρ→l, ι→ε…): AUC 0.86, versus 0.84 for Norwegian and 0.91 for English learner speech. The threshold (raw 0.016 → 0.5) flags 10% of clean native words and catches about 60% of the changed sounds. Native clarity median is 91%. The threshold was fitted on the same data. The int8 copy of the recognizer used by the hosted server (`data/en_int8.pt`, the same file as English) scores the same: AUC 0.860, 11% clean flagged at 0.5, 62% vs 61% caught.
- **Limits:** stress (the written accent) is not measured. Fast native speech often raises unstressed o→u and e→i, which the recognizer reports as is.

Files: `lang_el.py` (voices, variants, areas, coach prompt), `paragraphs_el.py` (45 paragraphs; `bank_el.json` holds their precomputed references for the hosted server, `audio_el/` the playback clips). Voices download into `data/el_voices/` and the HF cache on first use; FLEURS Greek goes in `~/data/fleurs_el` (`dev.tsv` plus the unpacked `dev/` audio from `google/fleurs` `data/el_gr`).

## Hosted version (johanwind.github.io/pronounce)

English at https://johanwind.github.io/pronounce/, Norwegian at https://johanwind.github.io/pronounce/nb/ and Greek at https://johanwind.github.io/pronounce/el/. The page is the same `static/reader.html`, copied by `tools/make_pages.py en|nb|el` into `dist/pages/` (the `johanwind/pronounce` repository) with that language's server address and `window.PAGE_LANG`. Each language has its own server: AWS Lambda functions `pronounce`, `pronounce-nb` and `pronounce-el` (eu-north-1), built from `dist/space[_nb|_el]/` by `tools/make_space.py en|nb|el` (`PRONOUNCE_HOSTED=1`: no coach, no drill, no own texts, model voices precomputed by `tools/bank_audio.py`). Recordings are scored in memory; each browser's history lives in its local storage (`history` for English, `history_nb` / `history_el` for Norwegian / Greek, since the pages share one origin) and is sent back to a server that restarted.

Try it locally like the server runs: `PRONOUNCE_HOSTED=1 .venv/bin/python reader.py --lang nb --port 8800`.
