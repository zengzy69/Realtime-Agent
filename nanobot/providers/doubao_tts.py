"""Doubao (豆包) bidirectional streaming speech synthesis adapter.

Volcengine's TTS WebSocket protocol frames every message as a 4-byte header,
an int32 event number, a length-prefixed session ID (absent on connection
events), and a length-prefixed payload. One connection carries StartConnection,
StartSession, TaskRequest, and FinishSession; the server answers with
audio-only frames and finishes with SessionFinished or SessionFailed.
"""

from __future__ import annotations

import asyncio
import json
import struct
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

TTS_SAMPLE_RATE = 24_000

_DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
_DEFAULT_RESOURCE_ID = "seed-tts-2.0"
_OPEN_TIMEOUT_S = 10.0
_RECEIVE_TIMEOUT_S = 20.0

_FULL_CLIENT_REQUEST = 0b0001
_FULL_SERVER_RESPONSE = 0b1001
_AUDIO_ONLY_RESPONSE = 0b1011
_ERROR_RESPONSE = 0b1111
_FLAG_POSITIVE_SEQUENCE = 0b0001
_FLAG_NEGATIVE_SEQUENCE = 0b0011
_FLAG_EVENT = 0b0100

_START_CONNECTION = 1
_FINISH_CONNECTION = 2
_CONNECTION_STARTED = 50
_CONNECTION_FAILED = 51
_CONNECTION_FINISHED = 52
_START_SESSION = 100
_FINISH_SESSION = 102
_SESSION_STARTED = 150
_SESSION_FINISHED = 152
_SESSION_FAILED = 153
_TASK_REQUEST = 200
_CONNECTION_EVENTS = frozenset({_START_CONNECTION, _FINISH_CONNECTION, _CONNECTION_STARTED,
                                _CONNECTION_FAILED, _CONNECTION_FINISHED})

AudioChunkHandler = Callable[[bytes], Awaitable[None]]


class SpeechSynthesisError(Exception):
    """Doubao TTS rejected or failed a synthesis request."""


def _client_frame(event: int, payload: bytes = b"{}", session_id: str | None = None) -> bytes:
    frame = bytes([0x11, (_FULL_CLIENT_REQUEST << 4) | _FLAG_EVENT, 0x10, 0x00]) + struct.pack(">i", event)
    if session_id is not None:
        encoded = session_id.encode()
        frame += struct.pack(">I", len(encoded)) + encoded
    return frame + struct.pack(">I", len(payload)) + payload


def _parse_server_frame(raw: bytes | str) -> tuple[int, int | None, bytes]:
    """Return ``(message_type, event, payload)``; raise on error frames."""
    if not isinstance(raw, bytes) or len(raw) < 4:
        raise SpeechSynthesisError("Doubao TTS sent a malformed frame")
    try:
        return _parse_server_frame_bytes(raw)
    except (struct.error, ValueError) as exc:
        raise SpeechSynthesisError(f"Doubao TTS sent a malformed frame: {exc}") from exc


def _parse_server_frame_bytes(raw: bytes) -> tuple[int, int | None, bytes]:
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    offset = (raw[0] & 0x0F) * 4
    code: int | None = None
    if message_type == _ERROR_RESPONSE:
        (code,) = struct.unpack_from(">I", raw, offset)
        offset += 4
    elif flags in (_FLAG_POSITIVE_SEQUENCE, _FLAG_NEGATIVE_SEQUENCE):
        offset += 4
    event: int | None = None
    if flags == _FLAG_EVENT:
        (event,) = struct.unpack_from(">i", raw, offset)
        offset += 4
        if event not in _CONNECTION_EVENTS:
            (size,) = struct.unpack_from(">I", raw, offset)
            offset += 4 + size
        if event in (_CONNECTION_STARTED, _CONNECTION_FAILED, _CONNECTION_FINISHED):
            (size,) = struct.unpack_from(">I", raw, offset)
            offset += 4 + size
    (size,) = struct.unpack_from(">I", raw, offset)
    payload = raw[offset + 4:offset + 4 + size]
    if code is not None:
        raise SpeechSynthesisError(f"Doubao TTS error {code}: {payload.decode('utf-8', 'replace')}")
    if event in (_CONNECTION_FAILED, _SESSION_FAILED):
        raise SpeechSynthesisError(f"Doubao TTS failed: {payload.decode('utf-8', 'replace')}")
    return message_type, event, payload


async def _expect_event(ws: ClientConnection, expected: int) -> None:
    _, event, _ = _parse_server_frame(await asyncio.wait_for(ws.recv(), _RECEIVE_TIMEOUT_S))
    if event != expected:
        raise SpeechSynthesisError(f"Doubao TTS sent event {event}, expected {expected}")


class DoubaoTTSProvider:
    """Speech synthesis via Doubao streaming TTS; audio is mono PCM16 at ``TTS_SAMPLE_RATE``."""

    def __init__(
        self,
        api_key: str,
        *,
        voice: str,
        api_base: str | None = None,
        resource_id: str | None = None,
    ):
        self.api_key = api_key
        self.voice = voice
        self.url = api_base or _DEFAULT_URL
        self.resource_id = resource_id or _DEFAULT_RESOURCE_ID

    async def synthesize(self, text: str, on_audio: AudioChunkHandler) -> None:
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
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
                await self._run_session(ws, text, on_audio)
                await ws.send(_client_frame(_FINISH_CONNECTION))
        except websockets.InvalidStatus as exc:
            body = exc.response.body.decode("utf-8", "replace") if exc.response.body else ""
            raise SpeechSynthesisError(
                f"Doubao TTS rejected the connection ({exc.response.status_code}): {body}"
            ) from exc
        except (OSError, TimeoutError, websockets.WebSocketException) as exc:
            raise SpeechSynthesisError(f"Doubao TTS request failed: {exc}") from exc

    async def _run_session(self, ws: ClientConnection, text: str, on_audio: AudioChunkHandler) -> None:
        await ws.send(_client_frame(_START_CONNECTION))
        await _expect_event(ws, _CONNECTION_STARTED)

        session_id = str(uuid.uuid4())
        params: dict[str, Any] = {
            "speaker": self.voice,
            "audio_params": {"format": "pcm", "sample_rate": TTS_SAMPLE_RATE},
            "additions": json.dumps({"disable_markdown_filter": True}),
        }
        start = {"event": _START_SESSION, "req_params": params}
        await ws.send(_client_frame(_START_SESSION, json.dumps(start).encode(), session_id))
        await _expect_event(ws, _SESSION_STARTED)

        task = {"event": _TASK_REQUEST, "req_params": {**params, "text": text}}
        await ws.send(_client_frame(_TASK_REQUEST, json.dumps(task, ensure_ascii=False).encode(), session_id))
        await ws.send(_client_frame(_FINISH_SESSION, b"{}", session_id))

        # Frames may split a 16-bit sample; carry the odd byte so every chunk holds whole samples.
        carry = b""
        while True:
            message_type, event, payload = _parse_server_frame(
                await asyncio.wait_for(ws.recv(), _RECEIVE_TIMEOUT_S)
            )
            if message_type == _AUDIO_ONLY_RESPONSE and payload:
                audio = carry + payload
                whole = len(audio) & ~1
                carry = audio[whole:]
                if whole:
                    await on_audio(audio[:whole])
            elif event == _SESSION_FINISHED:
                return
