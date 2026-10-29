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

from faster_whisper import WhisperModel

import webui

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("audio-stream-server")

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # PCM16
CHUNK_SECONDS = 8
CHUNK_BYTES = SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * CHUNK_SECONDS

RECORDINGS_DIR = Path(__file__).parent / "recordings"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "hi")

app = FastAPI()
app.include_router(webui.router)  # browser UI under /samsungpro

# Loaded once at startup and shared by all connections.
whisper_model: WhisperModel | None = None
transcribe_executor = ThreadPoolExecutor(max_workers=2)


@app.on_event("startup")
def load_model():
    global whisper_model
    log.info("Loading faster-whisper model '%s' (%s/%s)...", WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE)
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    log.info("Model loaded.")


@app.get("/")
def health():
    return {"status": "ok"}


def write_wav_chunk(path: Path, pcm_bytes: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH_BYTES)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)


def transcribe_chunk(path: Path) -> str:
    segments, _info = whisper_model.transcribe(str(path), language=WHISPER_LANGUAGE)
    return " ".join(segment.text.strip() for segment in segments).strip()


class StreamSession:
    def __init__(self):
        self.session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = RECORDINGS_DIR / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buffer = bytearray()
        self.chunk_index = 0
        self.transcript_path = self.dir / "transcript.txt"
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
        self.chunk_index += 1
        chunk_path = self.dir / f"chunk_{self.chunk_index:04d}.wav"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(transcribe_executor, write_wav_chunk, chunk_path, pcm_bytes)
        text = await loop.run_in_executor(transcribe_executor, transcribe_chunk, chunk_path)

        log.info("[%s] chunk %d: %s", self.session_id, self.chunk_index, text or "(silence)")
        seconds = len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS)
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(f"[chunk {self.chunk_index:04d} {seconds:.1f}s] {text}\n")
        # "Save audio" switch in the web UI: the audio is always transcribed, but only kept when it is on.
        if not webui.save_audio_enabled():
            chunk_path.unlink(missing_ok=True)


@app.websocket("/ws/audio")
@app.websocket("/samsungpro/get_audio")  # public path behind nginx on www.vkpremium.art
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    session = StreamSession()
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
        log.info("[%s] session closed", session.session_id)
