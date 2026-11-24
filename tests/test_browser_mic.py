"""Exercise the real microphone path (getUserMedia -> AudioWorklet -> upload -> results)
in headless Chrome with a fake microphone that plays tests/fixtures/planted.wav.

    .venv/bin/python tests/test_browser_mic.py http://localhost:8799
    MIC_WAV=... MIC_TEXT=... MIC_FLAG=a,b MIC_CLEAN=c,d .venv/bin/python tests/test_browser_mic.py URL
"""
import os
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import soundfile as sf
import websocket

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
WAV = Path(os.environ.get("MIC_WAV", Path(__file__).resolve().parent / "fixtures" / "planted.wav"))
TEXT = os.environ.get("MIC_TEXT", "I think the weather will be very warm this weekend, so we should visit the village by the river.")
FLAG = set(os.environ.get("MIC_FLAG", "think,very,this").split(","))
CLEAN = set(os.environ.get("MIC_CLEAN", "weekend,river").split(","))
PORT = 9339

dur = sf.info(WAV).duration
prof = tempfile.mkdtemp(prefix="chrome-mic-", dir=Path.home() / ".cache")
chrome = subprocess.Popen([
    "google-chrome", "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
    f"--use-file-for-fake-audio-capture={WAV}%noloop", "--autoplay-policy=no-user-gesture-required",
    "--no-first-run", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
            page = next(t for t in targets if t["type"] == "page")
            break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
    n = [0]

    def call(method, **params):
        n[0] += 1
        ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == n[0]:
                return m

    def js(expr):
        r = call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r.get("result", {}):
            raise RuntimeError(r["result"]["exceptionDetails"])
        return r["result"]["result"].get("value")

    call("Page.navigate", url=BASE)
    time.sleep(2)
    out = js(f"""(async () => {{
      await loadParagraph("custom", {json.dumps(TEXT)});
      await startRec();
      if (!S.rec) return {{error: document.getElementById("msg").innerText}};
      await new Promise(r => setTimeout(r, {int((dur + 0.8) * 1000)}));
      const n = S.chunks.reduce((a, c) => a + c.length, 0);
      stopRec();
      for (let i = 0; i < 200 && (S.busy || !S.result); i++) await new Promise(r => setTimeout(r, 100));
      if (!S.result) return {{error: document.getElementById("msg").innerText, samples: n}};
      return {{samples: n, rate: S.rate, stats: S.result.stats,
              flagged: S.result.words.filter(w => w.score !== null && w.score < 0.5).map(w => w.word),
              tiles: document.getElementById("tiles").innerText.replace(/\\n/g, " | ")}};
    }})()""")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    assert "error" not in out, out
    assert out["samples"] > out["rate"] * dur * 0.9, "captured too little audio"
    assert FLAG & set(out["flagged"]), out["flagged"]
    assert not CLEAN & set(out["flagged"]), out["flagged"]
    print("MIC PATH OK")
finally:
    chrome.terminate()
