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
import shutil
import struct
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
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
LIVE_WINDOW_SECONDS = 45  # fallback when a session is not in ACTIVE_SESSIONS; longer than a chunk (nothing is written while one fills up)
CHUNK_SETTLE_SECONDS = 1.5  # a chunk younger than this may still be being written

SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
CHUNK_FILE_RE = re.compile(r"^chunk_(\d{4})\.wav$")
# "[chunk 0001] text" (older sessions) or "[chunk 0001 8.0s] text"
TRANSCRIPT_LINE_RE = re.compile(r"^\[chunk (\d{4})(?: ([\d.]+)s)?\] ?(.*)$")
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
UNKNOWN_DEVICE_ID = "unknown"
LEGACY_DEVICE_NAME = "Earlier recordings (device not recorded)"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {"save_audio": True}

router = APIRouter(prefix="/samsungpro")

# Sessions whose WebSocket is open right now (the stream server adds and removes them).
ACTIVE_SESSIONS: set[str] = set()


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


# ── devices ─────────────────────────────────────────────────────────────────

def _clean_name(value: str | None) -> str:
    return "".join(ch for ch in (value or "") if ch.isprintable()).strip()[:60]


def device_from_headers(headers) -> tuple[str, str]:
    """(device_id, device_name) of a connecting phone: X-Device-Id / X-Device-Name from the app.

    An older app build sends neither: its devices are told apart by a short hash of the client IP (not the IP itself).
    """
    device_id = (headers.get("x-device-id") or "").strip()
    name = _clean_name(headers.get("x-device-name"))
    if DEVICE_ID_RE.match(device_id):
        return device_id, name or f"Device {device_id[:6]}"
    ip = headers.get("x-real-ip") or ""
    if ip:
        return "ip-" + hashlib.sha256(ip.encode()).hexdigest()[:8], name or "Unknown device"
    return UNKNOWN_DEVICE_ID, name or "Unknown device"


def write_session_meta(session_dir: Path, device_id: str, device_name: str) -> None:
    (session_dir / "meta.json").write_text(json.dumps({"device_id": device_id, "device_name": device_name}), encoding="utf-8")


def read_session_meta(session_dir: Path) -> tuple[str, str]:
    try:
        meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
        return str(meta["device_id"]), str(meta.get("device_name") or "Unknown device")
    except (FileNotFoundError, ValueError, KeyError):
        return UNKNOWN_DEVICE_ID, LEGACY_DEVICE_NAME


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

    text is None for chunks without a transcript (all new ones: nothing is transcribed any more); has_audio is False when audio saving was off.
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


def _is_live(session_dir: Path) -> bool:
    if session_dir.name in ACTIVE_SESSIONS:
        return True
    last_activity = max([p.stat().st_mtime for p in session_dir.iterdir()] or [session_dir.stat().st_mtime])
    return time.time() - last_activity < LIVE_WINDOW_SECONDS


def _summary(session_dir: Path) -> dict:
    chunks = _chunks(session_dir)
    device_id, device_name = read_session_meta(session_dir)
    return {
        "id": session_dir.name,
        "device_id": device_id,
        "device_name": device_name,
        "started_at": _started_at(session_dir.name),
        "chunks": len(chunks),
        "seconds": round(sum(c["seconds"] for c in chunks), 1),
        "has_audio": any(c["has_audio"] for c in chunks),
        "live": _is_live(session_dir),
    }


def _all_session_dirs() -> list[Path]:
    if not RECORDINGS_DIR.is_dir():
        return []
    return sorted((d for d in RECORDINGS_DIR.iterdir() if d.is_dir() and SESSION_ID_RE.match(d.name)), reverse=True)


def _all_summaries() -> list[dict]:
    return [s for s in (_summary(d) for d in _all_session_dirs()) if s["chunks"] or s["live"]]


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
    return {"sessions": _all_summaries()}


@router.get("/api/devices")
def list_devices(request: Request):
    """One row per connected device, newest recording first; online = it has a live session."""
    current_user(request)
    devices: dict[str, dict] = {}
    for s in _all_summaries():  # newest first, so the first name seen is the latest one
        d = devices.setdefault(s["device_id"], {"id": s["device_id"], "name": s["device_name"], "sessions": 0,
                                                "seconds": 0.0, "last_seen": s["started_at"], "online": False})
        d["sessions"] += 1
        d["seconds"] = round(d["seconds"] + s["seconds"], 1)
        d["online"] = d["online"] or s["live"]
    return {"devices": list(devices.values())}


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


# ── deleting ────────────────────────────────────────────────────────────────

def _delete_audio_files(session_dir: Path) -> int:
    removed = 0
    for p in session_dir.iterdir():
        if CHUNK_FILE_RE.match(p.name):
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def _refuse_if_live(session_dir: Path) -> None:
    if _is_live(session_dir):
        raise HTTPException(status_code=409, detail="This recording is still live. Stop it on the phone first.")


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    """Remove the whole recording: audio and text."""
    current_user(request)
    session_dir = _session_dir(session_id)
    _refuse_if_live(session_dir)
    shutil.rmtree(session_dir)
    return {"deleted": session_id}


@router.delete("/api/sessions/{session_id}/audio")
def delete_session_audio(session_id: str, request: Request):
    """Remove only the audio of a recording; its text stays."""
    current_user(request)
    session_dir = _session_dir(session_id)
    _refuse_if_live(session_dir)
    return {"session": session_id, "audio_files_deleted": _delete_audio_files(session_dir)}


@router.delete("/api/devices/{device_id}/sessions")
def delete_device_sessions(device_id: str, request: Request):
    """Remove every finished recording of one device (a live one is left alone)."""
    current_user(request)
    if not DEVICE_ID_RE.match(device_id):
        raise HTTPException(status_code=404, detail="No such device")
    deleted, skipped = 0, 0
    for d in _all_session_dirs():
        if read_session_meta(d)[0] != device_id:
            continue
        if _is_live(d):
            skipped += 1
        else:
            shutil.rmtree(d)
            deleted += 1
    return {"deleted": deleted, "skipped_live": skipped}


# ── time ranges: hour by hour, combined audio ───────────────────────────────

HOUR_SECONDS = 3600
MAX_RANGE_SECONDS = 48 * HOUR_SECONDS  # one combined file / one clean-up window at most this long when both ends are given
STREAM_PIECE_BYTES = 64 * 1024


def _check_device(device: str) -> str:
    if device and not DEVICE_ID_RE.match(device):
        raise HTTPException(status_code=400, detail="Bad device")
    return device


def _range_chunks(start: float, end: float, device: str = "") -> list[dict]:
    """Settled chunks whose start time (session start + offset) is in [start, end), oldest first, any session."""
    found = []
    for d in _all_session_dirs():
        session_start = _started_at(d.name)
        if session_start >= end:
            continue
        device_id, device_name = read_session_meta(d)
        if device and device_id != device:
            continue
        for c in _chunks(d):
            when = session_start + c["offset"]
            if start <= when < end:
                found.append({**c, "time": when, "session": d.name, "device_id": device_id, "device_name": device_name,
                              "path": d / c["file"] if c["file"] else None})
    found.sort(key=lambda c: (c["time"], c["session"], c["n"]))
    return found


@router.get("/api/hours")
def list_hours(request: Request, device: str = "", tz: int = 0):
    """Hour buckets (in the caller's time zone, tz = minutes east of UTC) that hold recordings, newest first."""
    current_user(request)
    _check_device(device)
    tz = max(-14 * 60, min(14 * 60, tz))
    buckets: dict[int, dict] = {}
    for c in _range_chunks(0, float("inf"), device):
        key = int((c["time"] + tz * 60) // HOUR_SECONDS)
        b = buckets.setdefault(key, {"start": key * HOUR_SECONDS - tz * 60, "end": (key + 1) * HOUR_SECONDS - tz * 60,
                                     "chunks": 0, "seconds": 0.0, "audio_chunks": 0, "devices": set()})
        b["chunks"] += 1
        b["seconds"] += c["seconds"]
        b["audio_chunks"] += 1 if c["has_audio"] else 0
        b["devices"].add(c["device_name"])
    rows = [{**b, "seconds": round(b["seconds"], 1), "has_audio": b["audio_chunks"] > 0, "devices": sorted(b["devices"])}
            for b in buckets.values()]
    return {"hours": sorted(rows, key=lambda r: -r["start"])}


def _valid_range(start: float, end: float) -> None:
    if not end > start:
        raise HTTPException(status_code=400, detail="End must be after start")
    if end - start > MAX_RANGE_SECONDS:
        raise HTTPException(status_code=400, detail="Range is too long (48 hours at most)")


@router.get("/api/range")
def get_range(request: Request, start: float, end: float, device: str = ""):
    """The chunks of a time range as one timeline; audio_offset is where the chunk starts in the combined audio."""
    current_user(request)
    _check_device(device)
    _valid_range(start, end)
    rows, audio_at = [], 0.0
    for c in _range_chunks(start, end, device):
        row = {"time": c["time"], "session": c["session"], "n": c["n"], "device_name": c["device_name"],
               "seconds": c["seconds"], "text": c["text"], "has_audio": c["has_audio"], "audio_offset": None}
        if c["path"]:
            row["audio_offset"] = round(audio_at, 2)
            audio_at += _seconds(c["path"])
        rows.append(row)
    return {"start": start, "end": end, "chunks": rows, "audio_seconds": round(audio_at, 1)}


def _wav_header(data_bytes: int) -> bytes:
    return (b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, BYTES_PER_SECOND, SAMPLE_WIDTH, 16)
            + b"data" + struct.pack("<I", data_bytes))


def _parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """A single 'bytes=a-b' range -> (first, last) inclusive; None when absent; ValueError when not satisfiable."""
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, _, last = header[6:].partition("-")
    if first == "":
        n = int(last)
        if n <= 0:
            raise ValueError
        return max(0, size - n), size - 1
    a = int(first)
    b = min(int(last), size - 1) if last else size - 1
    if a >= size or b < a:
        raise ValueError
    return a, b


def _stream_virtual_file(header: bytes, parts: list[tuple[Path, int]], first: int, last: int):
    """Bytes first..last (inclusive) of header + the PCM payload of each WAV in turn (a file that vanished reads as silence)."""
    position = 0
    pieces = [(None, len(header))] + [(p, n) for p, n in parts]
    for path, length in pieces:
        if position > last:
            return
        part_end = position + length
        if part_end > first:
            skip = max(0, first - position)
            take = min(length, last + 1 - position) - skip
            if path is None:
                yield header[skip:skip + take]
            else:
                try:
                    with open(path, "rb") as f:
                        f.seek(WAV_HEADER_BYTES + skip)
                        left = take
                        while left > 0:
                            data = f.read(min(STREAM_PIECE_BYTES, left))
                            if not data:
                                yield b"\x00" * left
                                break
                            left -= len(data)
                            yield data
                except FileNotFoundError:
                    yield b"\x00" * take
        position = part_end


@router.get("/api/range/audio.wav")
def get_range_audio(request: Request, start: float, end: float, device: str = "", download: bool = False):
    """All the audio of a time range joined into one WAV (seekable). Chunks are joined in time order, with no gap filling."""
    current_user(request)
    _check_device(device)
    _valid_range(start, end)
    parts = []
    for c in _range_chunks(start, end, device):
        if c["path"]:
            try:
                parts.append((c["path"], max(0, c["path"].stat().st_size - WAV_HEADER_BYTES)))
            except FileNotFoundError:
                pass
    if not parts:
        raise HTTPException(status_code=404, detail="No audio in this period")
    header = _wav_header(sum(n for _, n in parts))
    size = len(header) + sum(n for _, n in parts)
    try:
        byte_range = _parse_range(request.headers.get("range"), size)
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    first, last = byte_range or (0, size - 1)
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(last - first + 1), "Cache-Control": "no-store"}
    if byte_range:
        headers["Content-Range"] = f"bytes {first}-{last}/{size}"
    if download:
        stamp = time.strftime("%Y%m%d-%H%M", time.localtime(start))
        headers["Content-Disposition"] = f'attachment; filename="raki-{stamp}-{int((end - start) // 60)}min.wav"'
    return StreamingResponse(_stream_virtual_file(header, parts, first, last), status_code=206 if byte_range else 200,
                             media_type="audio/wav", headers=headers)


# ── clean-up with filters ───────────────────────────────────────────────────

class CleanupBody(BaseModel):
    device: str = ""
    start: float | None = None  # epoch seconds, None = from the beginning
    end: float | None = None  # epoch seconds, None = up to now (and later)
    what: Literal["audio", "all"] = "audio"  # audio: the WAV chunks whose time is in the period, text stays; all: whole recordings that started in it
    dry_run: bool = True  # preview first; nothing is deleted unless this is false


@router.post("/api/cleanup")
def cleanup(body: CleanupBody, request: Request):
    """Delete audio (or whole recordings) by device and period. Live recordings are never touched."""
    current_user(request)
    _check_device(body.device)
    start = body.start if body.start is not None else float("-inf")
    end = body.end if body.end is not None else float("inf")
    if not end > start:
        raise HTTPException(status_code=400, detail="End must be after start")
    recordings = audio_files = freed = skipped_live = 0
    for d in _all_session_dirs():
        if body.device and read_session_meta(d)[0] != body.device:
            continue
        session_start = _started_at(d.name)
        if body.what == "all":
            hit = start <= session_start < end
        else:
            hit = session_start < end  # chunks later in a long session can still fall inside the period
        if not hit:
            continue
        if _is_live(d):
            skipped_live += 1
            continue
        if body.what == "all":
            files = [p for p in d.iterdir() if p.is_file()]
            recordings += 1
            audio_files += sum(1 for p in files if CHUNK_FILE_RE.match(p.name))
            freed += sum(p.stat().st_size for p in files)
            if not body.dry_run:
                shutil.rmtree(d)
            continue
        removed_here = 0
        for c in _chunks(d):
            if c["has_audio"] and start <= session_start + c["offset"] < end:
                path = d / c["file"]
                removed_here += 1
                freed += path.stat().st_size
                if not body.dry_run:
                    path.unlink(missing_ok=True)
        if removed_here:
            recordings += 1
            audio_files += removed_here
    return {"dry_run": body.dry_run, "recordings": recordings, "audio_files": audio_files,
            "bytes": freed, "skipped_live": skipped_live}
