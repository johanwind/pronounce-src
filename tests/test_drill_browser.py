"""Drill page through the real mic path: headless Chrome, fake microphone playing a WAV.
Checks auto-stop on silence, scoring, the vowel map and the playback buttons.

    .venv/bin/python tests/test_drill_browser.py http://localhost:8799
"""
import json, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
import websocket

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8765"
FIX = Path(__file__).resolve().parent / "fixtures"
PORT = 9341

def run(wav, path):
    prof = tempfile.mkdtemp(prefix="chrome-drill-", dir=Path.home() / ".cache")
    p = subprocess.Popen(["google-chrome", "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={prof}",
        "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav}%noloop", "--autoplay-policy=no-user-gesture-required",
        "--no-first-run", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                page = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")) if t["type"] == "page"); break
            except Exception: time.sleep(0.2)
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=120)
        n = [0]
        def call(m, **kw):
            n[0] += 1; ws.send(json.dumps({"id": n[0], "method": m, "params": kw}))
            while True:
                r = json.loads(ws.recv())
                if r.get("id") == n[0]: return r
        call("Page.navigate", url=BASE + path); time.sleep(2.5)
        r = call("Runtime.evaluate", awaitPromise=True, returnByValue=True, expression="""(async () => {
          for (let i = 0; i < 300 && !(D.ref && D.ref.voices); i++) await new Promise(r => setTimeout(r, 100));
          const t0 = performance.now();
          await startRec();
          if (!D.rec) return {error: document.getElementById('msg').innerText};
          for (let i = 0; i < 80 && D.rec; i++) await new Promise(r => setTimeout(r, 100));
          const stoppedAfter = (performance.now() - t0) / 1000, autoStopped = !D.rec;
          for (let i = 0; i < 100 && (D.busy || !D.last); i++) await new Promise(r => setTimeout(r, 100));
          if (!D.last) return {error: document.getElementById('msg').innerText, autoStopped};
          return {word: D.word, target: D.last.target, score: D.last.score, heard: D.last.heard, point: D.last.point,
                  stoppedAfter, autoStopped, verdict: document.getElementById('verdict').innerText,
                  mapDots: document.querySelectorAll('#map circle').length, meEnabled: !document.getElementById('meBtn').disabled,
                  hint: document.getElementById('hint').innerText, ipa: document.getElementById('ipa').innerText};
        })()""")
        return r["result"]["result"].get("value")
    finally:
        p.terminate()

res_good = run(FIX / "drill_book_good.wav", "/drill?target=%CA%8A&word=book")
print("book said right:", json.dumps(res_good, ensure_ascii=False))
res_bad = run(FIX / "drill_book_bick.wav", "/drill?target=%CA%8A&word=book")
print("book said 'bick':", json.dumps(res_bad, ensure_ascii=False))
for r in (res_good, res_bad):
    assert r and "error" not in r, r
    assert r["autoStopped"] and r["stoppedAfter"] < 3.5, r
    assert r["mapDots"] >= 5 and r["meEnabled"], r
assert res_good["score"] > res_bad["score"] + 0.3, (res_good["score"], res_bad["score"])
assert res_bad["point"][0] < 0.5, "a front vowel should land on the front half of the map"
print("DRILL MIC PATH OK")
