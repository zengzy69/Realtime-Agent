"""Doubao (豆包) streaming speech recognition adapter.

Volcengine's SAUC WebSocket protocol frames every message as a 4-byte header,
a big-endian int32 sequence number, a uint32 payload size, and a gzip payload.
The first client frame carries the JSON session config; audio frames follow
with increasing sequence numbers, and the final audio frame negates its
sequence number.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import struct
import uuid
import zlib
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any, cast

import websockets
from loguru import logger
from websockets.asyncio.client import ClientConnection

from nanobot.audio.transcription_registry import (
    REALTIME_PCM_SAMPLE_RATE,
    TranscriptionProviderError,
    TranscriptTextHandler,
)

_DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"
_DEFAULT_RESOURCE_ID = "volc.seedasr.sauc.duration"
_OPEN_TIMEOUT_S = 10.0
_FINAL_RESULT_TIMEOUT_S = 30.0
_FILE_CHUNK_BYTES = 32_000

_FULL_CLIENT_REQUEST = 0b0001
_AUDIO_ONLY_REQUEST = 0b0010
_FULL_SERVER_RESPONSE = 0b1001
_SERVER_ERROR_RESPONSE = 0b1111
_FLAG_SEQUENCE = 0b0001
_FLAG_LAST = 0b0010
_FLAG_EVENT = 0b0100
_SERIALIZATION_NONE = 0b0000
_SERIALIZATION_JSON = 0b0001
_COMPRESSION_GZIP = 0b0001

_PCM_FORMAT: dict[str, Any] = {
    "format": "pcm",
    "codec": "raw",
    "rate": REALTIME_PCM_SAMPLE_RATE,
    "bits": 16,
    "channel": 1,
}
_OGG_OPUS_FORMAT: dict[str, Any] = {"format": "ogg", "codec": "opus"}
_FILE_FORMATS: dict[str, dict[str, Any]] = {
    ".wav": {"format": "wav"},
    ".mp3": {"format": "mp3"},
    ".ogg": _OGG_OPUS_FORMAT,
    ".oga": _OGG_OPUS_FORMAT,
    ".opus": _OGG_OPUS_FORMAT,
    ".m4a": {"format": "m4a"},
    ".aac": {"format": "aac"},
    ".amr": {"format": "amr"},
    ".spx": {"format": "spx"},
    ".pcm": _PCM_FORMAT,
}


def _client_frame(message_type: int, flags: int, sequence: int, payload: bytes, serialization: int) -> bytes:
    body = gzip.compress(payload)
    header = bytes([0x11, (message_type << 4) | flags, (serialization << 4) | _COMPRESSION_GZIP, 0x00])
    return header + struct.pack(">iI", sequence, len(body)) + body


def _decode_payload(payload: bytes, compression: int) -> bytes:
    return gzip.decompress(payload) if payload and compression == _COMPRESSION_GZIP else payload


def _parse_server_frame(raw: bytes | str) -> tuple[bool, dict[str, Any]]:
    """Return ``(is_last, payload)`` for one server frame; raise on error frames."""
    if not isinstance(raw, bytes) or len(raw) < 4:
        raise TranscriptionProviderError("Doubao ASR sent a malformed frame")
    try:
        return _parse_server_frame_bytes(raw)
    except (struct.error, zlib.error, OSError, ValueError) as exc:
        raise TranscriptionProviderError(f"Doubao ASR sent a malformed frame: {exc}") from exc


def _parse_server_frame_bytes(raw: bytes) -> tuple[bool, dict[str, Any]]:
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    serialization = raw[2] >> 4
    compression = raw[2] & 0x0F
    body = raw[(raw[0] & 0x0F) * 4:]
    if flags & _FLAG_SEQUENCE:
        body = body[4:]
    if flags & _FLAG_EVENT:
        body = body[4:]

    if message_type == _SERVER_ERROR_RESPONSE:
        code, size = struct.unpack(">iI", body[:8])
        detail = _decode_payload(body[8:8 + size], compression).decode("utf-8", "replace")
        raise TranscriptionProviderError(f"Doubao ASR error {code}: {detail}")
    if message_type != _FULL_SERVER_RESPONSE:
        raise TranscriptionProviderError(f"Doubao ASR sent unexpected message type {message_type}")

    (size,) = struct.unpack(">I", body[:4])
    payload = _decode_payload(body[4:4 + size], compression)
    data = cast(object, json.loads(payload)) if payload and serialization == _SERIALIZATION_JSON else None
    return bool(flags & _FLAG_LAST), cast(dict[str, Any], data) if isinstance(data, dict) else {}


def _result_text(payload: dict[str, Any]) -> str:
    result = cast(object, payload.get("result"))
    if not isinstance(result, dict):
        return ""
    text = cast(dict[str, Any], result).get("text")
    return text if isinstance(text, str) else ""


def _utterance_complete(payload: dict[str, Any]) -> bool:
    """True when Doubao VAD has closed this utterance (``definite``)."""
    result = cast(object, payload.get("result"))
    if not isinstance(result, dict):
        return False
    data = cast(dict[str, Any], result)
    if data.get("definite") is True:
        return True
    raw_utterances = data.get("utterances")
    if not isinstance(raw_utterances, list) or not raw_utterances:
        return False
    last = cast(object, raw_utterances[-1])
    if not isinstance(last, dict):
        return False
    return cast(dict[str, Any], last).get("definite") is True


def _session_config(audio: dict[str, Any], *, endpoint: bool) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model_name": "bigmodel",
        "enable_itn": True,
        "enable_punc": True,
        "enable_nonstream": True,
        "result_type": "full",
    }
    if endpoint:
        request["end_window_size"] = 800
        request["force_to_speech_time"] = 1000
    return {
        "user": {"uid": "nanobot"},
        "audio": audio,
        "request": request,
    }


async def _send_audio(
    ws: ClientConnection,
    chunks: AsyncIterable[bytes],
    audio_done: asyncio.Event,
) -> None:
    sequence = 2
    async for chunk in chunks:
        if not chunk:
            continue
        await ws.send(_client_frame(_AUDIO_ONLY_REQUEST, _FLAG_SEQUENCE, sequence, chunk, _SERIALIZATION_NONE))
        sequence += 1
    await ws.send(
        _client_frame(
            _AUDIO_ONLY_REQUEST,
            _FLAG_SEQUENCE | _FLAG_LAST,
            -sequence,
            b"",
            _SERIALIZATION_NONE,
        )
    )
    audio_done.set()


async def _receive_text(
    ws: ClientConnection,
    audio_done: asyncio.Event,
    on_partial: TranscriptTextHandler | None,
) -> str:
    text = ""
    while True:
        timeout = _FINAL_RESULT_TIMEOUT_S if audio_done.is_set() else None
        is_last, payload = _parse_server_frame(await asyncio.wait_for(ws.recv(), timeout))
        latest = _result_text(payload)
        if latest != text:
            text = latest
            if on_partial is not None and not is_last and not _utterance_complete(payload):
                await on_partial(text)
        if is_last or _utterance_complete(payload):
            return text.strip()


async def _file_chunks(data: bytes) -> AsyncIterator[bytes]:
    for start in range(0, len(data), _FILE_CHUNK_BYTES):
        yield data[start:start + _FILE_CHUNK_BYTES]


class DoubaoTranscriptionProvider:
    """Voice transcription via Doubao streaming ASR; ``model`` is the Volcengine resource ID."""

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        language: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key
        self.url = api_base or _DEFAULT_URL
        self.resource_id = model or _DEFAULT_RESOURCE_ID

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            logger.warning("Doubao ASR API key not configured for transcription")
            return ""
        path = Path(file_path)
        audio = _FILE_FORMATS.get(path.suffix.lower())
        if audio is None:
            logger.warning("Doubao ASR does not accept {} audio", path.suffix or "extensionless")
            return ""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Audio file not readable: {} ({})", file_path, exc)
            return ""
        try:
            return await self._recognize(audio, _file_chunks(data), None, endpoint=False)
        except TranscriptionProviderError as exc:
            logger.error("Doubao ASR transcription failed: {}", exc)
            return ""

    async def transcribe_realtime(
        self,
        pcm_chunks: AsyncIterable[bytes],
        on_partial: TranscriptTextHandler,
    ) -> str:
        return await self._recognize(_PCM_FORMAT, pcm_chunks, on_partial, endpoint=True)

    async def _recognize(
        self,
        audio: dict[str, Any],
        chunks: AsyncIterable[bytes],
        on_partial: TranscriptTextHandler | None,
        *,
        endpoint: bool,
    ) -> str:
        headers = {
            "X-Api-Key": self.api_key or "",
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        try:
            # Direct connection: websockets would otherwise route through system SOCKS
            # proxies, which need the optional python-socks dependency.
            async with websockets.connect(
                self.url,
                additional_headers=headers,
                proxy=None,
                open_timeout=_OPEN_TIMEOUT_S,
            ) as ws:
                config = json.dumps(_session_config(audio, endpoint=endpoint)).encode()
                await ws.send(_client_frame(_FULL_CLIENT_REQUEST, _FLAG_SEQUENCE, 1, config, _SERIALIZATION_JSON))
                _parse_server_frame(await asyncio.wait_for(ws.recv(), _FINAL_RESULT_TIMEOUT_S))
                return await self._stream(ws, chunks, on_partial)
        except websockets.InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", "replace") if exc.response.body else ""
            raise TranscriptionProviderError(
                f"Doubao ASR rejected the connection ({exc.response.status_code}): {body}"
            ) from exc
        except (OSError, TimeoutError, websockets.WebSocketException) as exc:
            raise TranscriptionProviderError(f"Doubao ASR request failed: {exc}") from exc

    @staticmethod
    async def _stream(
        ws: ClientConnection,
        chunks: AsyncIterable[bytes],
        on_partial: TranscriptTextHandler | None,
    ) -> str:
        audio_done = asyncio.Event()
        sender = asyncio.create_task(_send_audio(ws, chunks, audio_done))
        receiver = asyncio.create_task(_receive_text(ws, audio_done, on_partial))
        try:
            await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            if sender.done() and sender.exception() is not None:
                receiver.cancel()
                await sender
            return await receiver
        finally:
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)
