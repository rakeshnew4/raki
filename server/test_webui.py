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
