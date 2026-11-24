"""Save an int8-quantised copy of a recogniser for a small CPU server (the Norwegian 1B model:
3.9 GB of fp32 weights does not fit a 3 GB AWS Lambda; int8 is ~1.2 GB and faster on CPU).
Dynamic quantisation: Linear-layer weights are stored as int8, activations stay float.
The hosted app loads it with PHONEME_MODEL_FILE (scoring.MODEL_FILE); references for its bank
must be computed with the same file (tools/bank_build.py with PHONEME_MODEL_FILE set).

    PRONOUNCE_LANG=nb reserve --mem 7G --cpus 4 -- .venv/bin/python tools/quantize.py data/nb1b_int8.pt
"""
import sys
import time

import torch
from transformers import Wav2Vec2ForCTC

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from scoring import PHONEME_MODEL_ID  # noqa: E402


def quantize(m):
    """int8 Linear layers. Weight-norm parametrisations (the positional convolution) are baked
    into plain weights first: torch.save cannot pickle a parametrised module."""
    from torch.nn.utils import parametrize
    for mod in m.modules():
        if parametrize.is_parametrized(mod):
            for name in list(mod.parametrizations):
                parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
        mod._load_state_dict_pre_hooks.clear()  # weight_norm's loading hook (a local function)
    # inplace: each Linear is swapped as it is converted, so fp32 and int8 copies never coexist
    return torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)


def main():
    out = sys.argv[1]
    t = time.time()
    m = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL_ID).eval()
    q = quantize(m)
    del m
    torch.save(q, out)
    print(f"{PHONEME_MODEL_ID} -> {out} (int8) in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
