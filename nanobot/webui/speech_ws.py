"""WebUI spoken-reply envelope handling.

The WebSocket channel owns transport. This module owns the WebUI speech
actions carried over that socket: ``tts_start`` sends a finished assistant
reply, the gateway picks the spoken excerpt, and audio streams back while
the command loop keeps serving other envelopes.
"""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from nanobot.audio.speech import (
    SpeechIngressError,
    resolve_speech_config,
    speech_excerpt_for,
    synthesize_speech,
)
from nanobot.config.loader import load_config
from nanobot.providers.doubao_tts import TTS_SAMPLE_RATE

_MAX_REQUEST_ID_LENGTH = 80
_MAX_REPLY_CHARS = 20_000

SpeechEventSender = Callable[[str, dict[str, Any]], Awaitable[None]]


def _request_id(envelope: dict[str, Any]) -> str | None:
    request_id = envelope.get("request_id")
    if isinstance(request_id, str) and 0 < len(request_id) <= _MAX_REQUEST_ID_LENGTH:
        return request_id
    return None


class SpeechSynthesisSessions:
    """Spoken-reply streams, at most one per connection.

    A session reports ``tts_started`` with the spoken excerpt, ``tts_audio``
    frames of base64 mono PCM16, then ``tts_end`` or ``tts_error``. A new
    ``tts_start`` or a matching ``tts_stop`` cancels the running session.
    """

    def __init__(self) -> None:
        self._sessions: dict[object, tuple[str, asyncio.Task[None]]] = {}

    async def start(
        self,
        owner: object,
        envelope: dict[str, Any],
        send_event: SpeechEventSender,
        *,
        config_path: Path | None = None,
    ) -> None:
        request_id = _request_id(envelope)
        reply = envelope.get("text")
        if request_id is None or not isinstance(reply, str):
            await send_event("tts_error", {"detail": "invalid_request"})
            return
        self.discard(owner)
        config = resolve_speech_config(load_config(config_path))

        async def forward_audio(chunk: bytes) -> None:
            await send_event(
                "tts_audio",
                {"request_id": request_id, "audio": base64.b64encode(chunk).decode("ascii")},
            )

        async def run() -> None:
            try:
                excerpt = speech_excerpt_for(reply[:_MAX_REPLY_CHARS], config)
                await send_event("tts_started", {
                    "request_id": request_id,
                    "text": excerpt.text,
                    "truncated": excerpt.truncated,
                    "sample_rate": TTS_SAMPLE_RATE,
                })
                await synthesize_speech(excerpt.text, config, forward_audio)
            except SpeechIngressError as exc:
                await send_event("tts_error", {"request_id": request_id, "detail": exc.detail})
            else:
                await send_event("tts_end", {"request_id": request_id})
            finally:
                entry = self._sessions.get(owner)
                if entry is not None and entry[1] is asyncio.current_task():
                    del self._sessions[owner]

        self._sessions[owner] = (request_id, asyncio.create_task(run()))

    def stop(self, owner: object, envelope: dict[str, Any]) -> None:
        entry = self._sessions.get(owner)
        if entry is not None and entry[0] == _request_id(envelope):
            self.discard(owner)

    def discard(self, owner: object) -> None:
        entry = self._sessions.pop(owner, None)
        if entry is not None:
            entry[1].cancel()
