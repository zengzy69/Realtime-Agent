"""WebUI transcription envelope handling.

The WebSocket channel owns transport and subscription fan-out. This module owns
the WebUI-specific audio transcription actions carried over that socket.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nanobot.audio.transcription import (
    TranscriptionIngressError,
    resolve_transcription_config,
    transcribe_audio_data_url,
    transcribe_realtime_pcm,
)
from nanobot.audio.transcription_registry import REALTIME_PCM_SAMPLE_RATE, TranscriptDeltaHandler
from nanobot.config.loader import load_config

_MAX_REQUEST_ID_LENGTH = 80
_PCM_BYTES_PER_SECOND = REALTIME_PCM_SAMPLE_RATE * 2

TranscriptionEventSender = Callable[[str, dict[str, Any]], Awaitable[None]]


def _request_id(envelope: dict[str, Any]) -> str | None:
    request_id = envelope.get("request_id")
    if isinstance(request_id, str) and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH:
        return request_id
    return None


async def webui_transcription_event(
    envelope: dict[str, Any],
    *,
    config_path: Path | None = None,
    send_event: TranscriptionEventSender | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return the final WS event name and payload for one WebUI transcription request.

    When ``send_event`` is given, streaming providers also emit
    ``transcription_delta`` events before the final result.
    """
    request_id = _request_id(envelope)
    if request_id is None:
        return "transcription_error", {"detail": "invalid_request"}

    on_delta: TranscriptDeltaHandler | None = None
    if send_event is not None:
        emit = send_event

        async def forward_delta(delta: str) -> None:
            await emit("transcription_delta", {"request_id": request_id, "delta": delta})

        on_delta = forward_delta

    try:
        text = await transcribe_audio_data_url(
            envelope.get("data_url"),
            resolve_transcription_config(load_config(config_path)),
            duration_ms=envelope.get("duration_ms"),
            on_delta=on_delta,
        )
    except TranscriptionIngressError as exc:
        return "transcription_error", {"detail": exc.detail, **exc.extra, "request_id": request_id}
    return "transcription_result", {"request_id": request_id, "text": text}


@dataclass
class _RealtimeSession:
    audio: asyncio.Queue[bytes | None]
    remaining_bytes: int
    task: asyncio.Task[None] | None = None
    closed: bool = False

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.audio.put_nowait(None)


class RealtimeTranscriptionSessions:
    """Realtime transcription sessions fed by WebUI PCM frames, at most one per connection.

    ``transcription_stream_start`` opens a session, ``transcription_stream_audio``
    carries base64 PCM16 at 16 kHz, and ``transcription_stream_stop`` ends the
    audio. The session reports ``transcription_partial`` events with the full
    text so far, then one ``transcription_result`` or ``transcription_error``.
    """

    def __init__(self) -> None:
        self._sessions: dict[object, tuple[str, _RealtimeSession]] = {}

    async def start(
        self,
        owner: object,
        envelope: dict[str, Any],
        send_event: TranscriptionEventSender,
        *,
        config_path: Path | None = None,
    ) -> None:
        request_id = _request_id(envelope)
        if request_id is None:
            await send_event("transcription_error", {"detail": "invalid_request"})
            return
        self.discard(owner)
        config = resolve_transcription_config(load_config(config_path))
        session = _RealtimeSession(
            audio=asyncio.Queue(),
            remaining_bytes=config.max_duration_sec * _PCM_BYTES_PER_SECOND,
        )

        async def pcm_chunks() -> AsyncIterator[bytes]:
            while (chunk := await session.audio.get()) is not None:
                yield chunk

        async def forward_partial(text: str) -> None:
            await send_event("transcription_partial", {"request_id": request_id, "text": text})

        async def run() -> None:
            try:
                text = await transcribe_realtime_pcm(pcm_chunks(), config, on_partial=forward_partial)
            except TranscriptionIngressError as exc:
                await send_event(
                    "transcription_error",
                    {"detail": exc.detail, **exc.extra, "request_id": request_id},
                )
            else:
                await send_event("transcription_result", {"request_id": request_id, "text": text})
            finally:
                if self._sessions.get(owner, (None, None))[1] is session:
                    del self._sessions[owner]

        self._sessions[owner] = (request_id, session)
        session.task = asyncio.create_task(run())

    def push_audio(self, owner: object, envelope: dict[str, Any]) -> None:
        session = self._session(owner, envelope)
        audio = envelope.get("audio")
        if session is None or session.closed or not isinstance(audio, str):
            return
        try:
            pcm = base64.b64decode(audio, validate=True)
        except binascii.Error:
            return
        pcm = pcm[:session.remaining_bytes]
        session.remaining_bytes -= len(pcm)
        if pcm:
            session.audio.put_nowait(pcm)
        if session.remaining_bytes <= 0:
            session.close()

    def stop(self, owner: object, envelope: dict[str, Any]) -> None:
        session = self._session(owner, envelope)
        if session is not None:
            session.close()

    def discard(self, owner: object) -> None:
        entry = self._sessions.pop(owner, None)
        if entry is not None and entry[1].task is not None:
            entry[1].task.cancel()

    def _session(self, owner: object, envelope: dict[str, Any]) -> _RealtimeSession | None:
        entry = self._sessions.get(owner)
        if entry is None or entry[0] != _request_id(envelope):
            return None
        return entry[1]
