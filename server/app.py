import asyncio
import contextlib
import logging
import os
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import webui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio-stream-server")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # PCM16
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "8"))  # audio is cut, saved in pieces of this length
CHUNK_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * CHUNK_SECONDS

RECORDINGS_DIR = Path(__file__).parent / "recordings"

app = FastAPI()
app.include_router(webui.router)  # browser UI under /samsungpro

# Only writes WAV files: no speech-to-text runs on this server.
write_executor = ThreadPoolExecutor(max_workers=1)


@app.get("/")
def health():
    return {"status": "ok"}


def write_wav_chunk(path: Path, pcm_bytes: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


class StreamSession:
    def __init__(self, device_id: str = webui.UNKNOWN_DEVICE_ID, device_name: str = ""):
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = RECORDINGS_DIR / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buffer = bytearray()
        self.chunk_index = 0
        webui.write_session_meta(self.dir, device_id, device_name)
        webui.ACTIVE_SESSIONS.add(self.session_id)
        log.info("New session %s -> %s", self.session_id, self.dir)

    def add_audio(self, data: bytes) -> list[bytes]:
        """Append incoming audio; return any complete chunk(s) ready to flush."""
        self.buffer.extend(data)
        ready = []
        while len(self.buffer) >= CHUNK_BYTES:
            ready.append(bytes(self.buffer[:CHUNK_BYTES]))
            del self.buffer[:CHUNK_BYTES]
        return ready

    def take_remainder(self) -> bytes | None:
        if self.buffer:
            remainder = bytes(self.buffer)
            self.buffer.clear()
            return remainder
        return None

    async def flush_chunk(self, pcm_bytes: bytes) -> None:
        # "Save audio" switch in the web UI: when it is off nothing is stored (there is no transcript to keep).
        if not webui.save_audio_enabled():
            return
        self.chunk_index += 1
        chunk_path = self.dir / f"chunk_{self.chunk_index:04d}.wav"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(write_executor, write_wav_chunk, chunk_path, pcm_bytes)
        log.info("[%s] chunk %d saved (%.1fs)", self.session_id, self.chunk_index, len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS))


@app.websocket("/ws/audio")
@app.websocket("/samsungpro/get_audio")  # public path behind nginx on www.vkpremium.art
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    device_id, device_name = webui.device_from_headers(websocket.headers)
    session = StreamSession(device_id, device_name)
    try:
        while True:
            data = await websocket.receive_bytes()
            for chunk in session.add_audio(data):
                await session.flush_chunk(chunk)
    except WebSocketDisconnect:
        log.info("[%s] client disconnected", session.session_id)
    finally:
        remainder = session.take_remainder()
        if remainder:
            with contextlib.suppress(Exception):
                await session.flush_chunk(remainder)
        webui.ACTIVE_SESSIONS.discard(session.session_id)
        log.info("[%s] session closed", session.session_id)
