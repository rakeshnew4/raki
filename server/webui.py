"""Browser UI for the recordings: login, list sessions, read the speech-to-text, play the audio.

Mounted under /samsungpro (the public path behind nginx). Everything except the HTML page and
POST /api/login needs a signed session cookie. Users live in DATA_DIR/users.json (see manage_users.py).
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
RECORDINGS_DIR = BASE_DIR / "recordings"
DATA_DIR = Path(os.environ.get("RAKI_DATA_DIR", BASE_DIR / "data"))
USERS_FILE = DATA_DIR / "users.json"
SECRET_FILE = DATA_DIR / "secret.key"
INDEX_HTML = BASE_DIR / "static" / "index.html"

COOKIE_NAME = "raki_session"
SESSION_TTL_SECONDS = 12 * 3600
SAMPLE_RATE, SAMPLE_WIDTH = 16000, 2  # PCM16 mono, same as app.py
CHUNK_SECONDS = 8  # same as app.py; used only for old transcripts that did not record the length
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH
WAV_HEADER_BYTES = 44
LIVE_WINDOW_SECONDS = 20
CHUNK_SETTLE_SECONDS = 1.5  # a chunk younger than this may still be being written

SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
CHUNK_FILE_RE = re.compile(r"^chunk_(\d{4})\.wav$")
# "[chunk 0001] text" (older sessions) or "[chunk 0001 8.0s] text"
TRANSCRIPT_LINE_RE = re.compile(r"^\[chunk (\d{4})(?: ([\d.]+)s)?\] ?(.*)$")
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {"save_audio": True}

router = APIRouter(prefix="/samsungpro")


# ── passwords and users ─────────────────────────────────────────────────────

def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32)
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


# A throw-away hash so an unknown user costs the same time as a wrong password.
_DUMMY_HASH = hash_password("not-a-real-password")


def load_users() -> dict[str, str]:
    try:
        return json.loads(USERS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_users(users: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = USERS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, indent=2))
    os.chmod(tmp, 0o600)
    tmp.replace(USERS_FILE)


# ── signed session cookie ───────────────────────────────────────────────────

def _secret() -> bytes:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SECRET_FILE.exists():
        SECRET_FILE.write_bytes(secrets.token_bytes(32))
        os.chmod(SECRET_FILE, 0o600)
    return SECRET_FILE.read_bytes()


def _sign(payload: str) -> str:
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def make_token(username: str) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"u": username, "exp": int(time.time()) + SESSION_TTL_SECONDS}).encode()).decode()
    return f"{payload}.{_sign(payload)}"


def read_token(token: str | None) -> str | None:
    """Return the username if the token is genuine, unexpired and the user still exists."""
    if not token or "." not in token:
        return None
    payload, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(_sign(payload), signature):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(payload.encode()))
    except ValueError:
        return None
    if data.get("exp", 0) < time.time() or data.get("u") not in load_users():
        return None
    return data["u"]


def current_user(request: Request) -> str:
    user = read_token(request.cookies.get(COOKIE_NAME))
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


# ── login throttle (per client IP, in memory) ───────────────────────────────

MAX_FAILURES, FAILURE_WINDOW_SECONDS = 5, 300
_failures: dict[str, list[float]] = {}


def client_ip(request: Request) -> str:
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "?")


def _recent_failures(ip: str) -> list[float]:
    cutoff = time.time() - FAILURE_WINDOW_SECONDS
    kept = [t for t in _failures.get(ip, []) if t > cutoff]
    if kept:
        _failures[ip] = kept
    else:
        _failures.pop(ip, None)
    return kept


# ── settings (switchable from the UI) ───────────────────────────────────────

def load_settings() -> dict:
    try:
        stored = json.loads(SETTINGS_FILE.read_text())
    except (FileNotFoundError, ValueError):
        stored = {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in stored.items() if k in DEFAULT_SETTINGS}}


def save_audio_enabled() -> bool:
    """Read for every chunk by the stream server, so a change in the UI applies at once."""
    return bool(load_settings()["save_audio"])


def _store_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SETTINGS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(settings))
    tmp.replace(SETTINGS_FILE)


# ── recordings ──────────────────────────────────────────────────────────────

def _read_transcript(session_dir: Path) -> dict[int, tuple[float | None, str]]:
    """chunk number -> (seconds if recorded, text)."""
    entries: dict[int, tuple[float | None, str]] = {}
    try:
        lines = (session_dir / "transcript.txt").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return entries
    for line in lines:
        m = TRANSCRIPT_LINE_RE.match(line)
        if m:
            entries[int(m.group(1))] = (float(m.group(2)) if m.group(2) else None, m.group(3).strip())
    return entries


def _session_dir(session_id: str) -> Path:
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=404, detail="No such session")
    path = RECORDINGS_DIR / session_id
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="No such session")
    return path


def _settled_chunks(session_dir: Path) -> list[Path]:
    now = time.time()
    chunks = [p for p in session_dir.iterdir() if CHUNK_FILE_RE.match(p.name) and now - p.stat().st_mtime > CHUNK_SETTLE_SECONDS]
    return sorted(chunks)


def _seconds(path: Path) -> float:
    return max(0.0, (path.stat().st_size - WAV_HEADER_BYTES) / BYTES_PER_SECOND)


def _started_at(session_id: str) -> float:
    """Session ids are made with time.strftime (server-local time): YYYYmmdd-HHMMSS-xxxxxx."""
    return time.mktime(time.strptime(session_id[:15], "%Y%m%d-%H%M%S"))


def _chunks(session_dir: Path) -> list[dict]:
    """Every chunk of a session, from its transcript and/or its saved WAV files, with running offsets.

    text is None while a chunk is still being transcribed; has_audio is False when audio saving was off.
    """
    entries = _read_transcript(session_dir)
    wavs = {int(CHUNK_FILE_RE.match(p.name).group(1)): p for p in _settled_chunks(session_dir)}
    offset, chunks = 0.0, []
    for number in sorted(set(entries) | set(wavs)):
        recorded_seconds, text = entries.get(number, (None, None))
        wav = wavs.get(number)
        seconds = recorded_seconds if recorded_seconds is not None else (_seconds(wav) if wav else CHUNK_SECONDS)
        chunks.append({"n": number, "file": wav.name if wav else None, "has_audio": wav is not None,
                       "offset": round(offset, 1), "seconds": round(seconds, 1), "text": text})
        offset += seconds
    return chunks


def _summary(session_dir: Path) -> dict:
    chunks = _chunks(session_dir)
    last_activity = max([p.stat().st_mtime for p in session_dir.iterdir()] or [session_dir.stat().st_mtime])
    return {
        "id": session_dir.name,
        "started_at": _started_at(session_dir.name),
        "chunks": len(chunks),
        "seconds": round(sum(c["seconds"] for c in chunks), 1),
        "has_audio": any(c["has_audio"] for c in chunks),
        "live": time.time() - last_activity < LIVE_WINDOW_SECONDS,
    }


# ── routes ──────────────────────────────────────────────────────────────────

class LoginBody(BaseModel):
    username: str
    password: str


@router.get("", include_in_schema=False)
@router.get("/", include_in_schema=False)
def index():
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@router.post("/api/login")
def login(body: LoginBody, request: Request, response: Response):
    ip = client_ip(request)
    if len(_recent_failures(ip)) >= MAX_FAILURES:
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")
    stored = load_users().get(body.username)
    ok = verify_password(body.password, stored or _DUMMY_HASH) and stored is not None
    if not ok:
        _failures.setdefault(ip, []).append(time.time())
        raise HTTPException(status_code=401, detail="Wrong username or password")
    _failures.pop(ip, None)
    response.set_cookie(COOKIE_NAME, make_token(body.username), max_age=SESSION_TTL_SECONDS,
                        httponly=True, secure=True, samesite="strict", path="/samsungpro")
    return {"username": body.username}


@router.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE_NAME, path="/samsungpro")
    return {"ok": True}


@router.get("/api/me")
def me(request: Request):
    return {"username": current_user(request)}


class SettingsBody(BaseModel):
    save_audio: bool


@router.get("/api/settings")
def get_settings(request: Request):
    current_user(request)
    return load_settings()


@router.post("/api/settings")
def update_settings(body: SettingsBody, request: Request):
    current_user(request)
    _store_settings({"save_audio": body.save_audio})
    return load_settings()


@router.get("/api/sessions")
def list_sessions(request: Request):
    current_user(request)
    if not RECORDINGS_DIR.is_dir():
        return {"sessions": []}
    dirs = [d for d in RECORDINGS_DIR.iterdir() if d.is_dir() and SESSION_ID_RE.match(d.name)]
    sessions = [_summary(d) for d in sorted(dirs, reverse=True)]
    return {"sessions": [s for s in sessions if s["chunks"] or s["live"]]}


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, request: Request):
    current_user(request)
    session_dir = _session_dir(session_id)
    chunks = _chunks(session_dir)
    return {**_summary(session_dir), "chunks": chunks, "chunk_count": len(chunks)}


@router.get("/api/sessions/{session_id}/audio/{filename}")
def get_audio(session_id: str, filename: str, request: Request):
    current_user(request)
    session_dir = _session_dir(session_id)
    if not CHUNK_FILE_RE.match(filename):
        raise HTTPException(status_code=404, detail="No such chunk")
    path = session_dir / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No such chunk")
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=3600"})
