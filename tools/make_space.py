"""Assemble the hosted app (Docker image for AWS Lambda or a Hugging Face Space) in dist/space[_nb]/.
Only the files the hosted app imports are copied.

    .venv/bin/python tools/make_space.py [en|nb|el]
    docker build -t pronounce-space dist/space
    docker run --rm --cpus 2 -p 7860:7860 pronounce-space    # then open http://localhost:7860
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANG = sys.argv[1] if len(sys.argv) > 1 else "en"
OUT = ROOT / "dist" / ("space" if LANG == "en" else f"space_{LANG}")
FILES = ["reader.py", "scoring.py", "coach.py", "stats.py", "paragraphs.py", f"bank_{LANG}.json",
         "static/reader.html"]
if LANG == "nb":
    FILES += ["lang_nb.py", "paragraphs_nb.py", "data/nb_lexicon.sqlite"]
if LANG == "el":
    FILES += ["lang_el.py", "paragraphs_el.py"]
# int8-quantised recognisers (tools/quantize.py): ~2.4x faster on Lambda's CPU, same accuracy
# (en: speechocean762 AUC 0.860 vs 0.861; nb: no flag changes on real readings); the Norwegian
# 1B model only fits a 3 GB Lambda this way. The bank's references are computed with the same file.
# Greek uses the English recogniser, so the same int8 file.
MODEL_FILE = {"en": ROOT / "data" / "en_int8.pt", "nb": ROOT / "data" / "nb1b_int8.pt",
              "el": ROOT / "data" / "en_int8.pt"}.get(LANG)

REQUIREMENTS = """\
# CPU-only torch comes from the PyTorch index (see Dockerfile); versions as tested locally
transformers==5.23.0
huggingface-hub==2.4.0
phonemizer-fork==3.3.2
espeakng-loader==0.2.4
soundfile==0.14.0
scipy==1.18.1
numpy==2.5.5
"""

DOCKERFILE = """\
FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH HF_HOME=/home/user/.cache/huggingface \\
    PRONOUNCE_LANG={lang} PRONOUNCE_HOSTED=1 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1{model_env}
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch==2.15.0 torchaudio==2.11.1 \\
        --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
# models and voices go in a layer of their own, before the app code: an update to the code
# then rebuilds and uploads only the few-MB layers below, not the models
COPY --chown=user fetch_models.py .
RUN python fetch_models.py {model_id} {weights} && chmod -R a+rX /home/user/.cache && chmod a+rx /home/user \\
    && find /home/user/.local ! -perm -o+r -exec chmod a+rX {} +
# the passages' model voice (tools/bank_audio.py): changes only when the bank does
COPY --chown=user audio_{lang}/ audio_{lang}/
{copy_model}COPY --chown=user *.py bank_{lang}.json ./
COPY --chown=user static/ static/
# check that everything the app needs is already in the image, and score one reading
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python warm.py
# runtime: models are in the image; only /tmp may be writable (AWS Lambda)
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 XDG_CACHE_HOME=/tmp/cache \
    AWS_LWA_PORT=7860 AWS_LWA_READINESS_CHECK_PATH=/ AWS_LWA_ASYNC_INIT=true \\
    ALLOWED_ORIGINS=https://johanwind.github.io
# AWS Lambda Web Adapter: runs this HTTP server unchanged on Lambda (inactive elsewhere)
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
EXPOSE 7860
CMD ["python", "reader.py"]
"""

FETCH = """\
\"\"\"Build step: download the recogniser (kept in sync with scoring.py: PHONEME_MODEL_ID). The hosted
app has no model voice: references and the voice's audio are precomputed for the bank.\"\"\"
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

ID, WEIGHTS = sys.argv[1], sys.argv[2] == "1"
Wav2Vec2FeatureExtractor.from_pretrained(ID)
AutoConfig.from_pretrained(ID)
hf_hub_download(ID, "vocab.json")
if WEIGHTS:  # else the image carries a quantised copy (PHONEME_MODEL_FILE)
    Wav2Vec2ForCTC.from_pretrained(ID)
"""

WARM = """\
\"\"\"Build step: check the image has everything the hosted app needs: score the model voice
reading the first bank passage, in pieces as the hosted server does.\"\"\"
import json, os, time
import soundfile as sf
from scipy.signal import resample_poly
import paragraphs
from scoring import Scorer
from reader import AUDIO, AUDIO_DIR, audio_key

s = Scorer(tts=False)
text = paragraphs.BANK[0]
a, sr = sf.read(AUDIO_DIR / AUDIO[audio_key(text)]["file"], dtype="float32")
a = resample_poly(a, 16000, sr).astype("float32")
t = time.time()
res = s.analyse(text, a, 16000, pieces={})
print("warm-up reading scored in", round(time.time() - t, 1), "s:", res["stats"].get("clarity"), "% clarity")
assert res["stats"].get("clarity", 0) > 50, res
"""

README = """\
---
title: Read Aloud
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Read a passage aloud, see which English sounds were unclear
---

# Read Aloud: English pronunciation practice

Read a short passage aloud. The app shows which words and sounds came out unclear, sums them up as three focus areas, and picks the next passage to practise those sounds.

- **How it scores:** a phoneme recogniser ([facebook/wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)) is force-aligned to the expected sounds (espeak-ng, US English), and each sound's clarity is compared with how clear the same sound is in two synthetic voices ([Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)). It measures how clearly each sound comes across, not accent.
- **Passages:** excerpts from public-domain books (Jerome K. Jerome, Saki, P. G. Wodehouse, Stephen Leacock, G. K. Chesterton, O. Henry, Mark Twain, Arthur Conan Doyle, Edgar Allan Poe, Charlotte Perkins Gilman, Aesop), chosen for being fun to read aloud.
- **Privacy:** recordings are scored in memory and discarded; nothing is written to disk. Your progress lives in your own browser (local storage).
- **Speed:** this runs on a CPU (AWS Lambda), so scoring takes a few seconds, and after a quiet spell the server loads its model while you read the first passage.
"""


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / f"audio_{LANG}", OUT / f"audio_{LANG}")
    if MODEL_FILE:
        shutil.copy2(MODEL_FILE, OUT / "model_int8.pt")
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    model_id = {"en": "facebook/wav2vec2-lv-60-espeak-cv-ft", "nb": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
                "el": "facebook/wav2vec2-lv-60-espeak-cv-ft"}[LANG]
    dockerfile = DOCKERFILE.replace("{lang}", LANG).replace("{model_id}", model_id) \
        .replace("{weights}", "0" if MODEL_FILE else "1") \
        .replace("{model_env}", " \\\n    PHONEME_MODEL_FILE=/home/user/app/model_int8.pt" if MODEL_FILE else "") \
        .replace("{copy_model}", "COPY --chown=user model_int8.pt ./\n" if MODEL_FILE else "")
    if LANG == "nb":
        dockerfile = dockerfile.replace("COPY --chown=user static/ static/",
                                        "COPY --chown=user static/ static/\nCOPY --chown=user data/ data/")
    for name, text in [("requirements.txt", REQUIREMENTS), ("Dockerfile", dockerfile), ("warm.py", WARM), ("fetch_models.py", FETCH),
                       ("README.md", README)]:
        (OUT / name).write_text(text)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
