"""Tests for the browser UI and the 'save audio' switch.  Run:  python -m unittest test_webui -v"""
import asyncio
import os
import shutil
import struct
import tempfile
import time
import unittest
from pathlib import Path

_tmp = tempfile.TemporaryDirectory()
os.environ["RAKI_DATA_DIR"] = str(Path(_tmp.name) / "data")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app as stream_app
import manage_users
import webui

REC = Path(_tmp.name) / "recordings"
webui.RECORDINGS_DIR = REC
stream_app.RECORDINGS_DIR = REC


def wav_bytes(seconds: float) -> bytes:
    data = b"\x00\x00" * int(16000 * seconds)
    return b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", len(data)) + data


def make_session(session_id: str, chunks: dict[int, tuple[str | None, str | None]], settle: bool = True) -> Path:
    """chunks: number -> (wav marker or None, transcript line text or None)."""
    d = REC / session_id
    d.mkdir(parents=True, exist_ok=True)
    lines = []
    for n, (wav, text) in chunks.items():
        if wav:
            p = d / f"chunk_{n:04d}.wav"
            p.write_bytes(wav_bytes(8))
            if settle:
                os.utime(p, (time.time() - 60, time.time() - 60))
        if text is not None:
            lines.append(text)
    (d / "transcript.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if settle:
        old = time.time() - 600
        for p in d.iterdir():
            os.utime(p, (old, old))
    return d


class WebUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webui.save_users({"alice": webui.hash_password("correct horse battery")})
        cls.app = FastAPI()
        cls.app.include_router(webui.router)

    def setUp(self):
        shutil.rmtree(REC, ignore_errors=True)
        webui._failures.clear()
        webui._store_settings({"save_audio": True})
        self.client = TestClient(self.app, base_url="https://testserver")

    def login(self, client=None):
        c = client or self.client
        r = c.post("/samsungpro/api/login", json={"username": "alice", "password": "correct horse battery"})
        self.assertEqual(r.status_code, 200, r.text)
        return c

    # auth
    def test_page_is_public_but_api_is_not(self):
        self.assertEqual(self.client.get("/samsungpro/").status_code, 200)
        self.assertEqual(self.client.get("/samsungpro").status_code, 200)
        for path in ("me", "sessions", "settings", "sessions/20260101-000000-abcdef", "sessions/20260101-000000-abcdef/audio/chunk_0001.wav"):
            self.assertEqual(self.client.get(f"/samsungpro/api/{path}").status_code, 401, path)
        self.assertEqual(self.client.post("/samsungpro/api/settings", json={"save_audio": False}).status_code, 401)

    def test_login_ok_and_cookie_flags(self):
        r = self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "correct horse battery"})
        self.assertEqual(r.status_code, 200)
        cookie = r.headers["set-cookie"].lower()
        for flag in ("httponly", "secure", "samesite=strict", "path=/samsungpro"):
            self.assertIn(flag, cookie)
        self.assertEqual(self.client.get("/samsungpro/api/me").json(), {"username": "alice"})

    def test_wrong_password_and_unknown_user_look_the_same(self):
        a = self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "nope nope nope"})
        b = self.client.post("/samsungpro/api/login", json={"username": "ghost", "password": "nope nope nope"})
        self.assertEqual((a.status_code, a.json()), (b.status_code, b.json()))
        self.assertEqual(a.status_code, 401)

    def test_login_throttle_per_ip(self):
        for _ in range(webui.MAX_FAILURES):
            self.assertEqual(self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "bad bad bad"}).status_code, 401)
        r = self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "correct horse battery"})
        self.assertEqual(r.status_code, 429)  # even the right password is refused while throttled
        other = self.client.post("/samsungpro/api/login", headers={"x-real-ip": "203.0.113.9"}, json={"username": "alice", "password": "correct horse battery"})
        self.assertEqual(other.status_code, 200)

    def test_tampered_and_expired_tokens_are_rejected(self):
        token = webui.make_token("alice")
        payload, sig = token.rsplit(".", 1)
        for bad in (payload + "." + "0" * 64, payload[:-2] + "xx." + sig, "garbage", ""):
            self.client.cookies.set(webui.COOKIE_NAME, bad, path="/samsungpro")
            self.assertEqual(self.client.get("/samsungpro/api/me").status_code, 401, bad)
        real = webui.SESSION_TTL_SECONDS
        webui.SESSION_TTL_SECONDS = -10
        try:
            expired = webui.make_token("alice")
        finally:
            webui.SESSION_TTL_SECONDS = real
        self.assertIsNone(webui.read_token(expired))

    def test_removed_user_loses_access(self):
        token = webui.make_token("alice")
        users = webui.load_users()
        webui.save_users({})
        try:
            self.assertIsNone(webui.read_token(token))
        finally:
            webui.save_users(users)

    def test_logout_clears_cookie(self):
        self.login()
        self.client.post("/samsungpro/api/logout")
        self.assertEqual(self.client.get("/samsungpro/api/me").status_code, 401)

    # recordings
    def test_sessions_list_and_detail(self):
        make_session("20260101-101010-aaaaaa", {
            1: ("w", "[chunk 0001 8.0s] hello there"),
            2: ("w", "[chunk 0002 8.0s] "),            # silence
            3: ("w", None),                             # wav settled, not transcribed yet
        })
        make_session("20260101-111111-bbbbbb", {1: (None, "[chunk 0001 8.0s] text only"), 2: (None, "[chunk 0002 3.5s] tail")})
        make_session("20260101-121212-cccccc", {1: ("w", "[chunk 0001] old format line")})
        self.login()
        listing = self.client.get("/samsungpro/api/sessions").json()["sessions"]
        ids = [s["id"] for s in listing]
        self.assertEqual(ids[:3], ["20260101-121212-cccccc", "20260101-111111-bbbbbb", "20260101-101010-aaaaaa"])  # newest first
        by_id = {s["id"]: s for s in listing}
        self.assertTrue(by_id["20260101-101010-aaaaaa"]["has_audio"])
        self.assertFalse(by_id["20260101-111111-bbbbbb"]["has_audio"])
        self.assertEqual(by_id["20260101-111111-bbbbbb"]["seconds"], 11.5)
        self.assertFalse(by_id["20260101-101010-aaaaaa"]["live"])

        d = self.client.get("/samsungpro/api/sessions/20260101-101010-aaaaaa").json()
        self.assertEqual([c["text"] for c in d["chunks"]], ["hello there", "", None])
        self.assertEqual([c["offset"] for c in d["chunks"]], [0.0, 8.0, 16.0])
        self.assertTrue(all(c["has_audio"] for c in d["chunks"]))

        d = self.client.get("/samsungpro/api/sessions/20260101-111111-bbbbbb").json()
        self.assertEqual([(c["has_audio"], c["file"], c["offset"], c["seconds"]) for c in d["chunks"]],
                         [(False, None, 0.0, 8.0), (False, None, 8.0, 3.5)])

        d = self.client.get("/samsungpro/api/sessions/20260101-121212-cccccc").json()
        self.assertEqual(d["chunks"][0]["seconds"], 8.0)  # from the wav size

    def test_unsettled_chunk_is_hidden_and_session_is_live(self):
        make_session("20260102-101010-dddddd", {1: ("w", "[chunk 0001 8.0s] done"), 2: ("w", None)}, settle=False)
        self.login()
        d = self.client.get("/samsungpro/api/sessions/20260102-101010-dddddd").json()
        self.assertEqual(d["chunk_count"], 1)   # chunk 2 may still be being written
        self.assertTrue(d["live"])
        self.assertEqual(d["chunks"][0]["text"], "done")

    def test_audio_served_with_range_and_paths_are_confined(self):
        make_session("20260103-101010-eeeeee", {1: ("w", "[chunk 0001 8.0s] hi")})
        (REC / "secret.txt").write_text("top secret")
        self.login()
        base = "/samsungpro/api/sessions/20260103-101010-eeeeee/audio/"
        r = self.client.get(base + "chunk_0001.wav")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "audio/wav"))
        self.assertEqual(r.content[:4], b"RIFF")
        part = self.client.get(base + "chunk_0001.wav", headers={"Range": "bytes=0-99"})
        self.assertEqual((part.status_code, len(part.content)), (206, 100))
        for bad in ("chunk_0002.wav", "..%2Fsecret.txt", "transcript.txt", "chunk_1.wav"):
            self.assertEqual(self.client.get(base + bad).status_code, 404, bad)
        for bad_id in ("..", "%2e%2e", "nonexistent", "20260103-101010-ffffff"):
            self.assertEqual(self.client.get(f"/samsungpro/api/sessions/{bad_id}").status_code, 404, bad_id)

    # save-audio switch
    def test_settings_roundtrip_and_default_on(self):
        self.login()
        self.assertEqual(self.client.get("/samsungpro/api/settings").json(), {"save_audio": True})
        self.assertEqual(self.client.post("/samsungpro/api/settings", json={"save_audio": False}).json(), {"save_audio": False})
        self.assertFalse(webui.save_audio_enabled())
        self.assertEqual(self.client.post("/samsungpro/api/settings", json={"save_audio": "maybe"}).status_code, 422)
        self.client.post("/samsungpro/api/settings", json={"save_audio": True})
        self.assertTrue(webui.save_audio_enabled())

    def test_missing_or_corrupt_settings_file_means_save(self):
        webui.SETTINGS_FILE.unlink(missing_ok=True)
        self.assertTrue(webui.save_audio_enabled())
        webui.SETTINGS_FILE.write_text("{not json")
        self.assertTrue(webui.save_audio_enabled())


class DeviceAndDeleteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webui.save_users({"alice": webui.hash_password("correct horse battery")})
        cls.app = FastAPI()
        cls.app.include_router(webui.router)

    def setUp(self):
        shutil.rmtree(REC, ignore_errors=True)
        webui._failures.clear()
        webui.ACTIVE_SESSIONS.clear()
        self.client = TestClient(self.app, base_url="https://testserver")
        r = self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "correct horse battery"})
        self.assertEqual(r.status_code, 200)

    def session(self, sid, device_id=None, name="Pixel 7"):
        d = make_session(sid, {1: ("w", "[chunk 0001 20.0s] namaste"), 2: ("w", "[chunk 0002 20.0s] dhanyavaad")})
        if device_id:
            webui.write_session_meta(d, device_id, name)
            old = time.time() - 600
            for p in d.iterdir():
                os.utime(p, (old, old))
        return d

    # identity
    def test_device_from_headers(self):
        self.assertEqual(webui.device_from_headers({"x-device-id": "abc123", "x-device-name": "Samsung SM-A52"}), ("abc123", "Samsung SM-A52"))
        self.assertEqual(webui.device_from_headers({"x-device-id": "abc123"}), ("abc123", "Device abc123"))
        # a bad id is ignored, the IP is hashed and never stored as is
        dev_id, name = webui.device_from_headers({"x-device-id": "../../etc", "x-real-ip": "203.0.113.9"})
        self.assertTrue(dev_id.startswith("ip-") and "203" not in dev_id and name == "Unknown device")
        self.assertEqual(webui.device_from_headers({}), (webui.UNKNOWN_DEVICE_ID, "Unknown device"))
        self.assertEqual(webui.device_from_headers({"x-device-id": "ok", "x-device-name": "A\x00B\n" + "x" * 200})[1][:3], "AB" + "x")

    def test_stream_session_records_device_and_is_active_until_closed(self):
        s = stream_app.StreamSession("dev1", "Pixel 7")
        self.assertIn(s.session_id, webui.ACTIVE_SESSIONS)
        self.assertEqual(webui.read_session_meta(s.dir), ("dev1", "Pixel 7"))
        self.assertEqual(stream_app.CHUNK_SECONDS, 20)
        self.assertEqual(len(s.add_audio(b"\x00" * (16000 * 2 * 19))), 0)
        self.assertEqual(len(s.add_audio(b"\x00" * (16000 * 2 * 1))), 1)  # a chunk is 20 s

    def test_devices_group_sessions_and_old_ones_are_unknown(self):
        self.session("20260101-100000-aaaaaa", "dev1", "Pixel 7")
        self.session("20260102-100000-bbbbbb", "dev1", "Pixel 7 (renamed)")
        self.session("20260103-100000-cccccc", "dev2", "Galaxy A52")
        self.session("20251231-100000-dddddd")  # no meta.json
        devs = {d["id"]: d for d in self.client.get("/samsungpro/api/devices").json()["devices"]}
        self.assertEqual(set(devs), {"dev1", "dev2", "unknown"})
        self.assertEqual(devs["dev1"]["sessions"], 2)
        self.assertEqual(devs["dev1"]["name"], "Pixel 7 (renamed)")  # latest name wins
        self.assertEqual(devs["dev1"]["seconds"], 80.0)
        self.assertEqual(devs["unknown"]["name"], webui.LEGACY_DEVICE_NAME)
        self.assertFalse(any(d["online"] for d in devs.values()))
        by_id = {s["id"]: s for s in self.client.get("/samsungpro/api/sessions").json()["sessions"]}
        self.assertEqual(by_id["20260103-100000-cccccc"]["device_name"], "Galaxy A52")
        self.assertEqual(self.client.get("/samsungpro/api/sessions/20260103-100000-cccccc").json()["device_id"], "dev2")

    def test_open_connection_is_live_and_device_online_even_when_quiet(self):
        self.session("20260101-100000-aaaaaa", "dev1")
        webui.ACTIVE_SESSIONS.add("20260101-100000-aaaaaa")  # files are old: a chunk is still filling up
        self.assertTrue(self.client.get("/samsungpro/api/sessions/20260101-100000-aaaaaa").json()["live"])
        self.assertTrue(self.client.get("/samsungpro/api/devices").json()["devices"][0]["online"])

    # delete
    def test_delete_endpoints_need_login(self):
        c = TestClient(self.app, base_url="https://testserver")
        self.assertEqual(c.delete("/samsungpro/api/sessions/20260101-100000-aaaaaa").status_code, 401)
        self.assertEqual(c.delete("/samsungpro/api/sessions/20260101-100000-aaaaaa/audio").status_code, 401)
        self.assertEqual(c.delete("/samsungpro/api/devices/dev1/sessions").status_code, 401)

    def test_delete_session(self):
        d = self.session("20260101-100000-aaaaaa", "dev1")
        other = self.session("20260102-100000-bbbbbb", "dev1")
        r = self.client.delete("/samsungpro/api/sessions/20260101-100000-aaaaaa")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(d.exists())
        self.assertTrue(other.exists())
        self.assertEqual(self.client.delete("/samsungpro/api/sessions/20260101-100000-aaaaaa").status_code, 404)

    def test_delete_audio_only_keeps_text(self):
        d = self.session("20260101-100000-aaaaaa", "dev1")
        r = self.client.delete("/samsungpro/api/sessions/20260101-100000-aaaaaa/audio")
        self.assertEqual(r.json()["audio_files_deleted"], 2)
        self.assertEqual(list(d.glob("*.wav")), [])
        detail = self.client.get("/samsungpro/api/sessions/20260101-100000-aaaaaa").json()
        self.assertEqual([c["text"] for c in detail["chunks"]], ["namaste", "dhanyavaad"])
        self.assertFalse(detail["has_audio"])

    def test_live_recording_cannot_be_deleted(self):
        d = self.session("20260101-100000-aaaaaa", "dev1")
        webui.ACTIVE_SESSIONS.add(d.name)
        self.assertEqual(self.client.delete(f"/samsungpro/api/sessions/{d.name}").status_code, 409)
        self.assertEqual(self.client.delete(f"/samsungpro/api/sessions/{d.name}/audio").status_code, 409)
        self.assertTrue(d.exists() and list(d.glob("*.wav")))

    def test_delete_device_removes_its_finished_sessions_only(self):
        a = self.session("20260101-100000-aaaaaa", "dev1")
        live = self.session("20260102-100000-bbbbbb", "dev1")
        keep = self.session("20260103-100000-cccccc", "dev2")
        webui.ACTIVE_SESSIONS.add(live.name)
        r = self.client.delete("/samsungpro/api/devices/dev1/sessions")
        self.assertEqual(r.json(), {"deleted": 1, "skipped_live": 1})
        self.assertFalse(a.exists())
        self.assertTrue(live.exists() and keep.exists())
        self.assertEqual(self.client.delete("/samsungpro/api/devices/..%2Fx/sessions").status_code, 404)

    def test_bad_session_ids_cannot_delete_outside_recordings(self):
        victim = REC.parent / "victim"
        victim.mkdir(exist_ok=True)
        for sid in ("..", "%2e%2e", "victim", "20260101-100000-zzzzzz"):
            self.assertEqual(self.client.delete(f"/samsungpro/api/sessions/{sid}").status_code, 404, sid)
        self.assertTrue(victim.exists())
        shutil.rmtree(victim)


class HoursCombineCleanupTests(unittest.TestCase):
    """Hour-wise view, one combined WAV per period, and clean-up with filters."""

    @classmethod
    def setUpClass(cls):
        webui.save_users({"alice": webui.hash_password("correct horse battery")})
        cls.app = FastAPI()
        cls.app.include_router(webui.router)

    def setUp(self):
        shutil.rmtree(REC, ignore_errors=True)
        webui._failures.clear()
        webui.ACTIVE_SESSIONS.clear()
        self.client = TestClient(self.app, base_url="https://testserver")
        self.assertEqual(self.client.post("/samsungpro/api/login", json={"username": "alice", "password": "correct horse battery"}).status_code, 200)
        # two devices: A has 2 chunks at 10:00 and 1 at 10:30 and 11:05; B has 1 at 10:10 (each chunk 8 s of silence)
        self.make("20260301-100000-aaaaaa", "A", "Phone A", ["a1", "a2"])
        self.make("20260301-103000-bbbbbb", "A", "Phone A", ["a3"])
        self.make("20260301-110500-cccccc", "A", "Phone A", ["a4"])
        self.make("20260301-101000-dddddd", "B", "Phone B", ["b1"])

    def make(self, sid, dev, name, texts):
        d = make_session(sid, {i: ("w", f"[chunk {i:04d} 8.0s] {t}") for i, t in enumerate(texts, 1)})
        webui.write_session_meta(d, dev, name)
        old = time.time() - 900
        for p in d.iterdir():
            os.utime(p, (old, old))
        return d

    def at(self, hhmm, day="20260301"):
        return time.mktime(time.strptime(f"{day} {hhmm}", "%Y%m%d %H%M"))

    def get(self, path, **params):
        return self.client.get("/samsungpro/api/" + path, params=params)

    def test_hours_bucket_in_the_callers_time_zone(self):
        local = (time.mktime(time.gmtime(0)) * -1) // 60  # this machine's offset east of UTC, in minutes
        rows = self.get("hours", tz=int(local))
        self.assertEqual(rows.status_code, 200)
        hours = rows.json()["hours"]
        self.assertEqual([(h["start"], h["chunks"], h["audio_chunks"]) for h in hours],
                         [(self.at("1100"), 1, 1), (self.at("1000"), 4, 4)])  # newest first; 10:00 holds a1 a2 b1 a3
        self.assertEqual(hours[1]["devices"], ["Phone A", "Phone B"])
        self.assertTrue(hours[0]["has_audio"])
        only_b = self.get("hours", device="B", tz=int(local)).json()["hours"]
        self.assertEqual([(h["start"], h["chunks"]) for h in only_b], [(self.at("1000"), 1)])
        # half-hour time zone (India): the same chunks fall into other buckets
        shifted = self.get("hours", tz=int(local) + 330).json()["hours"]
        self.assertEqual({h["start"] % 3600 for h in shifted}, {(self.at("1000") - 330 * 60) % 3600})

    def test_range_timeline_and_audio_offsets(self):
        r = self.get("range", start=self.at("1000"), end=self.at("1100")).json()
        self.assertEqual([c["text"] for c in r["chunks"]], ["a1", "a2", "b1", "a3"])  # by time: 10:00:00 10:00:08 10:10 10:30
        self.assertEqual([c["audio_offset"] for c in r["chunks"]], [0, 8, 16, 24])
        self.assertEqual(r["audio_seconds"], 32.0)
        self.assertEqual(self.get("range", start=5, end=5).status_code, 400)
        self.assertEqual(self.get("range", start=0, end=10 ** 7).status_code, 400)
        self.assertEqual(self.get("range", start=0, end=10, device="../x").status_code, 400)

    def test_combined_wav_is_one_valid_seekable_file(self):
        start, end = self.at("1000"), self.at("1100")
        r = self.get("range/audio.wav", start=start, end=end)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers["accept-ranges"], "bytes")
        body = r.content
        pcm = 4 * 8 * 32000
        self.assertEqual(len(body), 44 + pcm)
        self.assertEqual(int(r.headers["content-length"]), len(body))
        self.assertEqual(body[:4], b"RIFF")
        self.assertEqual(struct.unpack("<I", body[40:44])[0], pcm)
        self.assertEqual(struct.unpack("<I", body[4:8])[0], 36 + pcm)
        import wave, io
        with wave.open(io.BytesIO(body)) as w:
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (1, 16000, 4 * 8 * 16000))
        part = self.client.get("/samsungpro/api/range/audio.wav", params={"start": start, "end": end}, headers={"Range": "bytes=100000-100999"})
        self.assertEqual(part.status_code, 206)
        self.assertEqual(part.content, body[100000:101000])
        self.assertEqual(part.headers["content-range"], f"bytes 100000-100999/{len(body)}")
        tail = self.client.get("/samsungpro/api/range/audio.wav", params={"start": start, "end": end}, headers={"Range": "bytes=-10"})
        self.assertEqual(tail.content, body[-10:])
        head = self.client.get("/samsungpro/api/range/audio.wav", params={"start": start, "end": end}, headers={"Range": "bytes=0-9"})
        self.assertEqual(head.content, body[:10])
        self.assertEqual(self.client.get("/samsungpro/api/range/audio.wav", params={"start": start, "end": end}, headers={"Range": f"bytes={len(body)}-"}).status_code, 416)
        only_b = self.get("range/audio.wav", start=start, end=end, device="B")
        self.assertEqual(len(only_b.content), 44 + 8 * 32000)
        self.assertEqual(self.get("range/audio.wav", start=self.at("0900"), end=self.at("0950")).status_code, 404)
        self.assertIn("attachment", self.get("range/audio.wav", start=start, end=end, download="true").headers["content-disposition"])

    def test_combined_wav_survives_a_vanished_file(self):
        start, end = self.at("1000"), self.at("1100")
        orig = webui._stream_virtual_file
        victim = REC / "20260301-101000-dddddd" / "chunk_0001.wav"
        real_stat = Path.stat
        r = None
        # the file is listed (stat ok), then removed before it is read: it must read as silence, same length
        def gone(*a, **k):
            victim.unlink(missing_ok=True)
            return orig(*a, **k)
        webui._stream_virtual_file = gone
        try:
            r = self.get("range/audio.wav", start=start, end=end)
        finally:
            webui._stream_virtual_file = orig
        self.assertEqual(len(r.content), 44 + 4 * 8 * 32000)

    def test_new_endpoints_need_login(self):
        c = TestClient(self.app, base_url="https://testserver")
        self.assertEqual(c.get("/samsungpro/api/hours").status_code, 401)
        self.assertEqual(c.get("/samsungpro/api/range", params={"start": 0, "end": 10}).status_code, 401)
        self.assertEqual(c.get("/samsungpro/api/range/audio.wav", params={"start": 0, "end": 10}).status_code, 401)
        self.assertEqual(c.post("/samsungpro/api/cleanup", json={}).status_code, 401)

    def cleanup(self, **body):
        return self.client.post("/samsungpro/api/cleanup", json=body)

    def wavs(self):
        return sorted(p.parent.name + "/" + p.name for p in REC.glob("*/chunk_*.wav"))

    def test_cleanup_previews_by_default_and_deletes_nothing(self):
        r = self.cleanup(what="audio").json()
        self.assertEqual((r["dry_run"], r["recordings"], r["audio_files"]), (True, 4, 5))
        self.assertGreater(r["bytes"], 5 * 256000)
        self.assertEqual(len(self.wavs()), 5)

    def test_cleanup_audio_by_period_and_device_keeps_text(self):
        r = self.cleanup(what="audio", start=self.at("1000"), end=self.at("1100"), device="A", dry_run=False).json()
        self.assertEqual(r["audio_files"], 3)  # a1 a2 a3; a4 is at 11:05, b1 is device B
        self.assertEqual(self.wavs(), ["20260301-101000-dddddd/chunk_0001.wav", "20260301-110500-cccccc/chunk_0001.wav"])
        detail = self.get("sessions/20260301-100000-aaaaaa").json()
        self.assertEqual([c["text"] for c in detail["chunks"]], ["a1", "a2"])  # text and offsets stay
        self.assertEqual([c["offset"] for c in detail["chunks"]], [0, 8])

    def test_cleanup_audio_can_cut_inside_one_session(self):
        r = self.cleanup(what="audio", start=self.at("1000") + 5, end=self.at("1000") + 60, dry_run=False).json()  # only a2 (starts at +8 s)
        self.assertEqual(r["audio_files"], 1)
        self.assertIn("20260301-100000-aaaaaa/chunk_0001.wav", self.wavs())
        self.assertNotIn("20260301-100000-aaaaaa/chunk_0002.wav", self.wavs())

    def test_cleanup_whole_recordings_by_start_time(self):
        r = self.cleanup(what="all", end=self.at("1020"), dry_run=False).json()  # started before 10:20: aaaaaa and dddddd
        self.assertEqual((r["recordings"], r["audio_files"]), (2, 3))
        self.assertEqual(sorted(p.name for p in REC.iterdir()), ["20260301-103000-bbbbbb", "20260301-110500-cccccc"])

    def test_cleanup_never_touches_live_recordings(self):
        webui.ACTIVE_SESSIONS.add("20260301-100000-aaaaaa")
        for what in ("audio", "all"):
            r = self.cleanup(what=what, device="A", dry_run=False).json()
            self.assertEqual(r["skipped_live"], 1)
        self.assertTrue((REC / "20260301-100000-aaaaaa" / "chunk_0001.wav").exists())
        self.assertTrue((REC / "20260301-100000-aaaaaa" / "transcript.txt").exists())

    def test_cleanup_rejects_bad_input(self):
        self.assertEqual(self.cleanup(device="../x").status_code, 400)
        self.assertEqual(self.cleanup(start=10, end=5).status_code, 400)
        self.assertEqual(self.cleanup(what="everything").status_code, 422)
        self.assertEqual(len(self.wavs()), 5)


class StreamServerSwitchTests(unittest.TestCase):
    """The stream server transcribes every chunk, and keeps the WAV only while the switch is on."""

    def run_chunk(self, save: bool) -> Path:
        webui._store_settings({"save_audio": save})
        stream_app.transcribe_chunk = lambda path: "hello world"
        session = stream_app.StreamSession()
        asyncio.run(session.flush_chunk(b"\x00\x00" * 16000 * 3))  # 3 s
        return session.dir

    def test_switch_on_keeps_audio(self):
        d = self.run_chunk(True)
        self.assertTrue((d / "chunk_0001.wav").exists())
        self.assertEqual((d / "transcript.txt").read_text(), "[chunk 0001 3.0s] hello world\n")

    def test_switch_off_keeps_text_only(self):
        d = self.run_chunk(False)
        self.assertFalse((d / "chunk_0001.wav").exists())
        self.assertEqual((d / "transcript.txt").read_text(), "[chunk 0001 3.0s] hello world\n")


class LanguageTests(unittest.TestCase):
    def transcribe_with(self, env_value):
        """Return the language argument the stream server passes to Whisper."""
        import importlib
        seen = {}

        class FakeModel:
            def transcribe(self, path, language=None):
                seen["language"] = language
                return [], None

        old = os.environ.get("WHISPER_LANGUAGE")
        if env_value is None:
            os.environ.pop("WHISPER_LANGUAGE", None)
        else:
            os.environ["WHISPER_LANGUAGE"] = env_value
        try:
            mod = importlib.reload(stream_app)
            mod.whisper_model = FakeModel()
            mod.transcribe_chunk(Path("x.wav"))
        finally:
            if old is None:
                os.environ.pop("WHISPER_LANGUAGE", None)
            else:
                os.environ["WHISPER_LANGUAGE"] = old
            importlib.reload(stream_app)
            stream_app.RECORDINGS_DIR = REC
        return seen["language"]

    def test_defaults_to_hindi(self):
        self.assertEqual(self.transcribe_with(None), "hi")

    def test_can_be_overridden_or_auto(self):
        self.assertEqual(self.transcribe_with("en"), "en")
        self.assertIsNone(self.transcribe_with("auto"))


class ManageUsersTests(unittest.TestCase):
    def test_password_hash_verifies(self):
        h = webui.hash_password("s3cret-pass")
        self.assertTrue(webui.verify_password("s3cret-pass", h))
        self.assertFalse(webui.verify_password("S3cret-pass", h))
        self.assertFalse(webui.verify_password("x", "not-a-hash"))
        self.assertNotEqual(h, webui.hash_password("s3cret-pass"))  # salted

    def test_users_file_is_private(self):
        webui.save_users({"bob": webui.hash_password("longenough1")})
        self.assertEqual(oct(webui.USERS_FILE.stat().st_mode & 0o777), "0o600")
        self.assertNotIn("longenough1", webui.USERS_FILE.read_text())


if __name__ == "__main__":
    unittest.main()
