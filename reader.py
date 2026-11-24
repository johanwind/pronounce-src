#!/usr/bin/env python3
"""Read-aloud pronunciation trainer: read a paragraph, see which words and sounds
came out unclear, get three focus areas and a next paragraph that drills them.

    .venv/bin/python reader.py            # then open http://localhost:8765
    .venv/bin/python reader.py --lang nb  # Norwegian, http://localhost:8766, data in data/nb
    PRONOUNCE_HOSTED=1 reader.py          # public server (AWS Lambda, see tools/make_space.py): one
                                          # state per browser, nothing stored on disk, no coach or drill
"""

import os
import sys

# the language must be known before the scoring/stats modules load their tables
if "--lang" in sys.argv[:-1]:
    os.environ["PRONOUNCE_LANG"] = sys.argv[sys.argv.index("--lang") + 1]
LANG = os.environ.setdefault("PRONOUNCE_LANG", "en")
HOSTED = os.environ.get("PRONOUNCE_HOSTED") == "1"

import argparse
import io
import json
import random
import re
import threading
import time
import traceback
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import soundfile as sf

import importlib

import coach
import paragraphs
import stats

drill = importlib.import_module("drill") if LANG == "en" and not HOSTED else None  # English-only
BANK = importlib.import_module({"nb": "paragraphs_nb", "el": "paragraphs_el"}.get(LANG, "paragraphs")).BANK
COACH = not HOSTED  # the coach runs on the local Claude subscription login
MAX_USERS, MAX_HISTORY = 500, 60  # hosted: per-browser state kept in memory
MAX_STREAMS = 20                    # hosted: recordings being received in pieces
# hosted: other sites whose copy of the page may call this server (e.g. GitHub Pages)
ALLOWED_ORIGINS = {o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o}
AUDIO_DIR = Path(__file__).resolve().parent / f"audio_{LANG}"  # tools/bank_audio.py
AUDIO = (json.loads((AUDIO_DIR / "index.json").read_text())
         if HOSTED and (AUDIO_DIR / "index.json").exists() else {})


def audio_key(text):
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:16]

ROOT = Path(__file__).resolve().parent
REC_DIR = stats.DATA / "recordings"


def page_html():
    """The reader page, told which language it is for (the GitHub Pages copy gets the same line,
    tools/make_pages.py), so its labels are right from the first paint."""
    html = (ROOT / "static" / "reader.html").read_text()
    return html.replace("<script>", f'<script>window.PAGE_LANG = "{LANG}";</script>\n<script>', 1).encode()


def area_rows(table, keys):
    """Focus-area rows for the page, with the area's display name (Norwegian names for nb)."""
    return [{k: (stats.label(t[k]) if k == "area" else t[k]) for k in keys} for t in table]


class Shared:
    """Models and caches shared by every user."""
    def __init__(self, scorer):
        # scorer: a Scorer, or a function that builds one. A function runs in a background thread
        # (hosted): the server answers at once, and only requests that need the model wait for it.
        self._scorer, self._error = None, None
        self.ready = threading.Event()
        if callable(scorer) and not hasattr(scorer, "analyse"):
            threading.Thread(target=self._load, args=(scorer,), daemon=True).start()
        else:
            self._scorer = scorer
            self.ready.set()
        self.pool = ThreadPoolExecutor(1)        # GPU work (references)
        self.coach_pool = ThreadPoolExecutor(2)  # Haiku calls
        self.refs = {}
        self.lock = threading.Lock()
        self.analyse_lock = threading.Lock()     # one scoring at a time (a small CPU server)
        self.tts_cache = {}
        self.drill = drill.Drill(self.scorer) if drill else None
        self.users = {}                          # hosted: browser id -> App, oldest first
        self.streams = {}                        # hosted: (user, recording id) -> audio so far + its pieces

    def _load(self, build):
        t = time.time()
        try:
            self._scorer = build()
            print(f"scorer ready {time.time() - t:.1f} s after start", flush=True)
        except BaseException as e:  # noqa: BLE001 - reported to every request that needs it
            traceback.print_exc()
            self._error = e
        self.ready.set()

    @property
    def scorer(self):
        self.ready.wait()
        if self._error:
            raise RuntimeError(f"the scorer failed to load: {self._error}")
        return self._scorer

    # ----- references are computed in the background as soon as a text is shown
    def prepare(self, text):
        with self.lock:
            if text not in self.refs:
                self.refs[text] = self.pool.submit(self._reference, text)
                if len(self.refs) > 300:  # custom texts; bank references stay in the scorer cache
                    self.refs.pop(next(iter(self.refs)))
            return self.refs[text]

    def _reference(self, text):
        try:
            return self.scorer.reference(text)
        finally:
            free_gpu()

    def tts(self, text, speed):
        key = (text, speed)
        if key not in self.tts_cache:
            from scoring import PLAY_VOICE, PLAY_SPEED
            a = self.scorer.tts_audio(text, PLAY_VOICE, speed=speed * PLAY_SPEED, rate=24000)
            buf = io.BytesIO()
            sf.write(buf, a, 24000, format="WAV", subtype="PCM_16")
            self.tts_cache[key] = buf.getvalue()
            if len(self.tts_cache) > 200:
                self.tts_cache.pop(next(iter(self.tts_cache)))
        return self.tts_cache[key]

    # ----- hosted: the page sends the recording in pieces while it is being made
    def stream_add(self, key, offset, samples):
        """Append 16 kHz samples at offset; run the model on every piece that is complete.
        Returns the number of samples held, or None if out of step (the page then sends the
        whole recording at the end)."""
        with self.lock:
            st = self.streams.get(key)
            if offset == 0:
                st = self.streams[key] = dict(audio=np.zeros(0, np.float32), pieces={})
                while len(self.streams) > MAX_STREAMS:
                    self.streams.pop(next(iter(self.streams)))
            if st is None or len(st["audio"]) != offset:
                self.streams.pop(key, None)
                return None
            st["audio"] = np.concatenate([st["audio"], samples])
        self.scorer.emissions_pieces(st["audio"], st["pieces"], final=False)
        return len(st["audio"])

    def stream_take(self, key, offset):
        with self.lock:
            st = self.streams.pop(key, None)
        return st if st is not None and len(st["audio"]) == offset else None

    def user(self, uid):
        """Local: the one user, history on disk. Hosted: one in-memory state per browser."""
        if not HOSTED:
            uid = "local"
        with self.lock:
            app = self.users.pop(uid, None)
            if app is None:
                app = App(self, [] if HOSTED else stats.load_history(), persist=not HOSTED)
            self.users[uid] = app  # most recently used last
            while len(self.users) > MAX_USERS:
                self.users.pop(next(iter(self.users)))
            return app


class _LazyScorer:
    """Stands in for the scorer; waits for the model only if it is actually used."""
    def __init__(self, shared):
        self._shared = shared

    def __getattr__(self, name):
        return getattr(self._shared.scorer, name)


class App:
    """One user's reading session."""
    def __init__(self, shared, history, persist=True):
        self.shared = shared
        self.persist = persist   # write history and recordings to disk (local use)
        self.lock = threading.Lock()
        self.attempts = {}       # id -> dict(report, focus_rows)
        self.coach_jobs = {}     # attempt id -> Future
        self.ready = []          # coach-written paragraphs, newest last, ready for "next"
        self.current = None      # paragraph on screen
        self.history = history
        self.drill = shared.drill
        self.asked_restore = False  # hosted: asked the page once for its saved history

    @property
    def scorer(self):  # waits while the model is still loading
        return self.shared.scorer

    def prepare(self, text):
        return self.shared.prepare(text)

    def tts(self, text, speed):
        return self.shared.tts(text, speed)

    def recent_texts(self, n=8):
        return [h["text"] for h in self.history[-n:]]

    def paragraph(self, kind, custom=None):
        if kind == "custom" and HOSTED:
            kind = "random"  # the hosted version has no model voice to score a new text against
        if kind == "custom" and custom and custom.strip():
            text = " ".join(custom.split())
        elif kind == "next":
            with self.lock:
                self.ready = [t for t in self.ready if t != self.current]
            if not self.ready and self.coach_jobs:
                # nothing queued yet (first reading): wait for the coach that is writing
                try:
                    self.coach_jobs[next(reversed(self.coach_jobs))].result(timeout=15)
                except Exception:  # noqa: BLE001 - timeout or coach error: use the bank
                    pass
            with self.lock:
                text = self.ready.pop() if self.ready else None
            if not text:
                focus = self.current_focus_names()
                # the bank's sound densities are precomputed, so this needs no model (lazy)
                text = coach.pick_bank_paragraph(_LazyScorer(self.shared), BANK, focus,
                                                 self.recent_texts(min(30, len(BANK) // 2)) + [self.current],
                                                 [h["text"] for h in self.history])
        elif kind == "same" and self.history:
            text = self.history[-1]["text"]
        else:
            recent = self.recent_texts() + [self.current]
            text = random.choice([t for t in BANK if t not in recent] or BANK)
        self.current = text
        self.prepare(text)
        m = paragraphs.META.get(text) or {}
        clip = AUDIO.get(audio_key(text), {})
        return dict(text=text, setup=m.get("setup"), source=m.get("author") and f"{m['author']}, {m['title']}",
                    word_times=clip.get("words"))  # hosted: play one word from the passage audio

    def current_focus_names(self):
        if not self.history:
            return []
        table = stats.area_table(self.normal_results()[::-1][:5])
        return [t["area"] for t in stats.top_focus(table)]

    def normal_results(self):
        # old local histories have a few readings in since-removed modes (pen in mouth...): skip them
        return [h for h in self.history if h.get("mode", "normal") == "normal" and h.get("words")
                and not h["stats"].get("maybe_incomplete")]

    def analyse(self, text, audio, rate, pieces=None):
        self.prepare(text).result()
        with self.shared.analyse_lock:
            res = self.scorer.analyse(text, audio, rate, pieces=pieces)
        if "error" in res:
            return res
        aid = uuid.uuid4().hex[:10]
        if self.persist:
            REC_DIR.mkdir(parents=True, exist_ok=True)
            sf.write(REC_DIR / f"{aid}.flac", audio, int(rate))
        # focus: this attempt plus the last few readings
        compact = dict(words=res["words"])
        prior = self.normal_results()[::-1][:4]
        table = stats.area_table([compact] + prior)
        focus_rows = stats.top_focus(table)
        prev_same = next((h for h in reversed(self.history) if h["text"] == text), None)
        entry = dict(id=aid, text=text, stats=res["stats"],
                     words=[{k: w[k] for k in ("word", "score", "phones", "punct", "heard_as", "slip") if k in w}
                            for w in res["words"]],
                     focus_areas=[t["area"] for t in focus_rows])
        entry = stats.save_attempt(entry) if self.persist else {"time": time.time(), **entry}
        self.history.append(entry)
        del self.history[:-200]
        if COACH:
            report = coach.build_report(res, table, self.history[:-1])
            self.attempts[aid] = dict(report=report, focus_rows=focus_rows)
            self.coach_jobs[aid] = self.shared.coach_pool.submit(self._coach, aid)
        return dict(
            id=aid, words=res["words"], stats=res["stats"], coach=COACH,
            entry=entry if HOSTED else None,  # the page keeps a copy (servers restart)
            focus=coach.local_focus(focus_rows),
            previous=prev_same and dict(stats=prev_same["stats"]),
        )

    def _coach(self, aid):
        a = self.attempts[aid]
        try:
            out = coach.ask_haiku(a["report"])
        except Exception as e:  # noqa: BLE001 - shown to the user, local fallback stays
            print("coach unavailable:", str(e)[:300], flush=True)
            return dict(ok=False, error=str(e)[:300])
        good = []
        for p in out.pop("next_paragraphs", None) or [out.get("next_paragraph", "")]:
            p = " ".join(str(p).split())
            if 30 <= len(p.split()) <= 120 and not any(c.isdigit() for c in p):
                good.append(p)
        with self.lock:
            # newest last; keep a few so "next" never has to wait for the coach
            self.ready = (self.ready + good)[-3:]
        for p in good:
            self.prepare(p)
        out["next_paragraph"] = good[0] if good else None
        return dict(ok=True, **out)

    def prefill(self):
        """At startup, have the coach write paragraphs from your last reading, so even the
        first "next" is instant."""
        last = next((h for h in reversed(self.normal_results())), None)
        if not last:
            return
        words = [dict(w, bad=[dict(phone=p["phone"], heard=p["heard"], norm=p["norm"])
                              for p in w.get("phones", []) if p.get("norm") is not None and p["norm"] < 0.5])
                 for w in last["words"]]
        table = stats.area_table(self.normal_results()[::-1][:5])
        aid = "startup"
        self.attempts[aid] = dict(report=coach.build_report(dict(words=words, stats=last["stats"]),
                                                            table, self.history[:-1]),
                                  focus_rows=stats.top_focus(table))
        self.coach_jobs[aid] = self.shared.coach_pool.submit(self._coach, aid)

    def restore(self, entries):
        """Hosted: the page sends back its saved copy of the history after a server restart.
        Only readings of this language's passages count (the hosted page has no own texts)."""
        if self.history or not isinstance(entries, list):
            return len(self.history)
        keep = []
        bank = set(BANK)
        for e in entries:
            try:
                if str(e["text"]) not in bank or e.get("mode", "normal") != "normal":
                    continue
                keep.append(dict(time=float(e["time"]), text=str(e["text"]),
                                 stats=dict(e["stats"]), focus_areas=list(e.get("focus_areas", []))[:5],
                                 words=[dict(word=str(w["word"]), score=w.get("score"), punct=bool(w.get("punct")),
                                             phones=[dict(phone=str(p["phone"]), norm=p.get("norm"),
                                                          heard=str(p.get("heard", "")))
                                                     for p in w.get("phones", [])])
                                        for w in e["words"]]))
            except (KeyError, TypeError, ValueError):
                continue
        self.history[:] = keep[-MAX_HISTORY:]
        return len(self.history)

    def coach_result(self, aid, wait=60):
        job = self.coach_jobs.get(aid)
        if not job:
            return dict(ok=False, error="unknown attempt")
        try:
            return job.result(timeout=wait)
        except TimeoutError:
            return dict(ok=False, pending=True)

    def weak_phones(self, n=8):
        """Per-phone mean score over recent normal readings, worst first."""
        agg = {}
        for h in self.normal_results()[-15:]:
            for w in h["words"]:
                for p in w["phones"]:
                    if p["norm"] is not None:
                        agg.setdefault(p["phone"], []).append(p["norm"])
        rows = [dict(phone=ph, mean=round(float(np.mean(v)), 2), n=len(v)) for ph, v in agg.items() if len(v) >= 4]
        rows.sort(key=lambda r: r["mean"])
        return rows[:n]

    def drill_sets(self):
        weak = [r for r in self.weak_phones(20) if r["phone"] in drill.DRILL_SETS]
        return dict(sets={k: dict(v, example=v["words"][0]) for k, v in drill.DRILL_SETS.items()},
                    weak=weak[:6], examples=drill.VOWEL_EXAMPLES, positions=drill.VOWEL_POS)

    def drill_attempt(self, word, target, audio, rate):
        out = self.drill.attempt(word, target, audio, rate)
        if "error" not in out:
            rec = uuid.uuid4().hex[:10]
            (stats.DATA / "drill_recordings").mkdir(parents=True, exist_ok=True)
            sf.write(stats.DATA / "drill_recordings" / f"{rec}.flac", audio, int(rate))
            with open(stats.DATA / "drill.jsonl", "a") as f:
                f.write(json.dumps(dict(time=time.time(), rec=rec, word=word, target=out["target"], score=out["score"],
                                        heard=out["heard"], point=out["point"], ms=out["ms"]), ensure_ascii=False) + "\n")
        return out

    def state(self):
        hist = [dict(time=h["time"], clarity=h["stats"].get("clarity"), wpm=h["stats"].get("wpm"))
                for h in self.history[-40:] if h.get("mode", "normal") == "normal"]
        table = stats.area_table(self.normal_results()[::-1][:5]) if self.history else []
        return dict(history=hist, lang=LANG, drill=self.drill is not None, hosted=HOSTED,
                    custom=not HOSTED, areas=area_rows(table, ("area", "mean", "n", "weak")))


def free_gpu():
    """Return cached GPU memory so other jobs (and other sessions) can use it."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


UID_RE = re.compile(r"[0-9a-f]{8,32}")


def make_handler(shared):
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, body, ctype="application/json", cache="no-store"):
            if not isinstance(body, (bytes, bytearray)):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            free_gpu()

        def _cors(self):
            origin = self.headers.get("Origin")
            if HOSTED and origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def do_OPTIONS(self):  # CORS preflight from an allowed copy of the page
            if not (HOSTED and self.headers.get("Origin") in ALLOWED_ORIGINS):
                return self._send(403, dict(error="forbidden"))
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "X-User, X-Text, X-Backup, Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _foreign(self):
            """True for requests a web page on another site could have made: a wrong Host
            (DNS rebinding) or a cross-site Origin. Only this app's own page may use the API."""
            if HOSTED:  # behind the host's proxy: the Host is the public name
                origin = self.headers.get("Origin")
                if origin in ALLOWED_ORIGINS:
                    return False
                if self.path.startswith("/api/tts"):  # read-only audio; <audio> sends no Origin
                    return False
                host = self.headers.get("Host", "")
                return ((origin is not None and urllib.parse.urlparse(origin).netloc != host)
                        or self.headers.get("Sec-Fetch-Site") == "cross-site")
            port = self.server.server_address[1]
            ok = {f"{h}:{port}" for h in ("localhost", "127.0.0.1", "[::1]")}
            origin = self.headers.get("Origin")
            return (self.headers.get("Host") not in ok
                    or (origin is not None and origin.removeprefix("http://") not in ok)
                    or self.headers.get("Sec-Fetch-Site") == "cross-site")

        def _body(self, limit=8 << 20):
            n = int(self.headers.get("Content-Length", 0))
            if n > limit:
                raise ValueError("request too large")
            return self.rfile.read(n) if n else b""

        def _needs_restore(self, app, path):
            """Hosted: this server copy has no history for a browser that has a saved one
            (restart, or a second copy of the app): the page re-sends it, then retries."""
            if not HOSTED or app.history or app.asked_restore or path in ("/api/restore", "/api/tts", "/api/warm"):
                return False
            try:
                have = int(self.headers.get("X-Backup") or 0)
            except ValueError:
                return False
            app.asked_restore = have > 0
            return have > 0

        def _user(self):
            uid = (self.headers.get("X-User") or "").lower()
            return shared.user(uid if UID_RE.fullmatch(uid) else "anonymous")

        def do_GET(self):
            if self._foreign():
                return self._send(403, dict(error="forbidden"))
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            try:
                app = self._user() if u.path.startswith("/api/") else None
                if app and self._needs_restore(app, u.path):
                    return self._send(409, dict(restore=True))
                if u.path == "/":
                    return self._send(200, page_html(), "text/html; charset=utf-8")
                if u.path == "/api/warm":  # the page keeps asking until the model is loaded
                    ready = shared.ready.wait(min(10.0, max(0.0, float(q.get("wait", 3)))))
                    return self._send(200, dict(ready=ready, error=str(shared._error) if shared._error else None))
                if u.path == "/api/state":
                    return self._send(200, app.state())
                if u.path == "/api/tts":
                    if HOSTED:  # precomputed model voice; the page slows it down itself
                        clip = AUDIO.get(audio_key(q.get("text", "")))
                        if not clip:
                            return self._send(404, {"error": "no audio for this text"})
                        return self._send(200, (AUDIO_DIR / clip["file"]).read_bytes(), "audio/mpeg",
                                          cache="public, max-age=86400")
                    speed = min(1.5, max(0.5, float(q.get("speed", 1))))
                    return self._send(200, shared.tts(q["text"][:600 if HOSTED else 2000], speed), "audio/wav")
                if u.path.startswith(("/drill", "/api/drill")) and shared.drill is None:
                    return self._send(404, {"error": "the sound drill is English-only"})
                if u.path == "/drill":
                    return self._send(200, (ROOT / "static" / "drill.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/drill/sets":
                    return self._send(200, app.drill_sets())
                if u.path == "/api/drill/ref":
                    ref = app.drill.reference(q["word"][:40], q.get("target") or None)
                    return self._send(200, app.drill.public_reference(ref))
                if u.path == "/api/drill/variant":
                    return self._send(200, app.drill.variant_wav(q["word"][:40], q.get("target") or None, q["vowel"],
                                                                 q.get("voice")), "audio/wav")
                if u.path == "/api/drill/wav":
                    return self._send(200, app.drill.ref_wav(q["word"][:40], q.get("target") or None, q["voice"]), "audio/wav")
                if u.path == "/api/coach":
                    return self._send(200, app.coach_result(q["id"], float(q.get("wait", 60))))
                self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": str(e)})

        def do_POST(self):
            if self._foreign():
                return self._send(403, dict(error="forbidden"))
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            try:
                app = self._user()
                if self._needs_restore(app, u.path):
                    return self._send(409, dict(restore=True))
                if u.path == "/api/paragraph":
                    d = json.loads(self._body(1 << 16) or b"{}")
                    custom = d.get("text")
                    if HOSTED and custom:  # a new text costs a reference computation on CPU
                        custom = " ".join(str(custom).split()[:120])
                    return self._send(200, app.paragraph(d.get("kind", "random"), custom))
                if u.path == "/api/restore":
                    d = json.loads(self._body(4 << 20) or b"{}")
                    return self._send(200, dict(restored=app.restore(d.get("history"))))
                if u.path == "/api/piece":  # hosted: part of a recording still being made
                    if not HOSTED:
                        return self._send(404, {"error": "not found"})
                    samples = np.frombuffer(self._body(4 << 20), dtype="<f4").astype(np.float32)
                    held = shared.stream_add((app, q.get("rid", "")[:32]), int(q.get("offset", -1)), samples)
                    return self._send(200, dict(held=held))
                if u.path == "/api/analyse":
                    text = urllib.parse.unquote(self.headers["X-Text"])
                    audio = np.frombuffer(self._body(), dtype="<f4").astype(np.float32)
                    rate = float(q.get("rate", 16000))
                    pieces = None
                    if HOSTED and rate == 16000:  # the model runs on fixed pieces (see Scorer.emissions_pieces)
                        pieces = {}
                        if q.get("rid"):  # the page sent the start already: this is the rest
                            st = shared.stream_take((app, q["rid"][:32]), int(q.get("offset", -1)))
                            if st is None:
                                return self._send(409, dict(resend=True))
                            audio, pieces = np.concatenate([st["audio"], audio]), st["pieces"]
                    t0 = time.time()
                    out = app.analyse(text, audio, rate, pieces)
                    print(f"analysed {len(audio)/rate:.1f}s audio in {time.time()-t0:.1f}s", flush=True)
                    return self._send(200, out)
                if u.path == "/api/drill/attempt" and shared.drill is None:
                    return self._send(404, {"error": "the sound drill is English-only"})
                if u.path == "/api/drill/attempt":
                    audio = np.frombuffer(self._body(), dtype="<f4").astype(np.float32)
                    out = app.drill_attempt(q["word"][:40], q.get("target") or None, audio, float(q.get("rate", 16000)))
                    return self._send(200, out)
                self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error": str(e)})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 0))
                    or (7860 if HOSTED else {"nb": 8766, "el": 8767}.get(LANG, 8765)))
    ap.add_argument("--lang", choices=["en", "nb", "el"], default=LANG)
    ap.add_argument("--host", default="0.0.0.0" if HOSTED else "127.0.0.1")
    args = ap.parse_args()
    from scoring import Scorer
    if HOSTED:
        # answer at once and load the model in the background (a cold start on AWS Lambda);
        # see the page's warm-up loop
        def build():
            s = Scorer(tts=False)
            s.prewarm()
            return s
        shared = Shared(build)
    else:
        shared = Shared(Scorer())
        shared.prepare(BANK[0])
    if COACH:
        shared.user("local").prefill()
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(shared))
    print(f"ready: http://localhost:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
