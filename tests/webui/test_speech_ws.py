"""Tests for WebUI spoken-reply envelopes carried over the gateway socket."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from nanobot.audio.speech import EffectiveSpeechConfig, SpeechIngressError
from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.providers.doubao_tts import AudioChunkHandler
from nanobot.webui.speech_ws import SpeechSynthesisSessions


def _config_path(tmp_path: Path, *, api_key: str = "doubao-key") -> Path:
    config = Config()
    config.tts.enabled = True
    config.providers.doubao_asr.api_key = api_key
    path = tmp_path / "config.json"
    save_config(config, path)
    return path


class _Events:
    def __init__(self) -> None:
        self.items: list[tuple[str, dict[str, Any]]] = []
        self.finished = asyncio.Event()

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        self.items.append((event, payload))
        if event in {"tts_end", "tts_error"}:
            self.finished.set()


@pytest.mark.asyncio
async def test_speech_session_streams_excerpt_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spoken: list[str] = []

    async def fake_synthesize(text: str, _config: EffectiveSpeechConfig, on_audio: AudioChunkHandler) -> None:
        spoken.append(text)
        await on_audio(b"\x01\x00\x02\x00")

    monkeypatch.setattr("nanobot.webui.speech_ws.synthesize_speech", fake_synthesize)
    events = _Events()
    sessions = SpeechSynthesisSessions()

    await sessions.start(
        "conn",
        {"request_id": "tts-1", "text": "可以出门。\n\n- 带伞\n- 穿外套"},
        events.send,
        config_path=_config_path(tmp_path),
    )
    await asyncio.wait_for(events.finished.wait(), 1)

    assert spoken == ["可以出门。"]
    assert events.items == [
        ("tts_started", {
            "request_id": "tts-1",
            "text": "可以出门。",
            "truncated": True,
            "sample_rate": 24000,
        }),
        ("tts_audio", {"request_id": "tts-1", "audio": base64.b64encode(b"\x01\x00\x02\x00").decode()}),
        ("tts_end", {"request_id": "tts-1"}),
    ]


@pytest.mark.asyncio
async def test_speech_session_reports_missing_credentials(tmp_path: Path) -> None:
    events = _Events()

    await SpeechSynthesisSessions().start(
        "conn",
        {"request_id": "tts-1", "text": "你好。"},
        events.send,
        config_path=_config_path(tmp_path, api_key=""),
    )
    await asyncio.wait_for(events.finished.wait(), 1)

    assert events.items == [("tts_error", {"request_id": "tts-1", "detail": "not_configured"})]


@pytest.mark.asyncio
async def test_speech_session_reports_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_synthesize(_text: str, _config: EffectiveSpeechConfig, _on_audio: AudioChunkHandler) -> None:
        raise SpeechIngressError("provider_error")

    monkeypatch.setattr("nanobot.webui.speech_ws.synthesize_speech", failing_synthesize)
    events = _Events()

    await SpeechSynthesisSessions().start(
        "conn",
        {"request_id": "tts-1", "text": "你好。"},
        events.send,
        config_path=_config_path(tmp_path),
    )
    await asyncio.wait_for(events.finished.wait(), 1)

    assert events.items[-1] == ("tts_error", {"request_id": "tts-1", "detail": "provider_error"})


@pytest.mark.asyncio
async def test_speech_stop_cancels_running_synthesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def endless_synthesize(_text: str, _config: EffectiveSpeechConfig, _on_audio: AudioChunkHandler) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr("nanobot.webui.speech_ws.synthesize_speech", endless_synthesize)
    events = _Events()
    sessions = SpeechSynthesisSessions()
    await sessions.start(
        "conn",
        {"request_id": "tts-1", "text": "你好。"},
        events.send,
        config_path=_config_path(tmp_path),
    )
    await asyncio.wait_for(started.wait(), 1)

    sessions.stop("conn", {"request_id": "other"})
    assert not cancelled.is_set()
    sessions.stop("conn", {"request_id": "tts-1"})
    await asyncio.wait_for(cancelled.wait(), 1)

    assert [event for event, _ in events.items] == ["tts_started"]
