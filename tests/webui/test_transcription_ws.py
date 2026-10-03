"""Tests for WebUI transcription envelopes carried over the gateway socket."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterable
from pathlib import Path
from typing import Any

import pytest

from nanobot.audio.transcription_registry import TranscriptionProviderError
from nanobot.config.loader import save_config
from nanobot.config.schema import Config
from nanobot.webui.transcription_ws import RealtimeTranscriptionSessions, webui_transcription_event


def _audio_data_url(payload: bytes = b"voice", mime: str = "audio/webm") -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


@pytest.mark.asyncio
async def test_webui_transcribe_audio_rejects_unconfigured_provider(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.transcription.provider = "groq"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)

    event, payload = await webui_transcription_event({
        "request_id": "voice-1",
        "data_url": _audio_data_url(),
    })

    assert event == "transcription_error"
    assert payload == {
        "request_id": "voice-1",
        "detail": "not_configured",
        "provider": "groq",
    }


@pytest.mark.asyncio
async def test_webui_transcription_uses_explicit_gateway_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_path = tmp_path / "default.json"
    gateway_path = tmp_path / "gateway.json"
    default = Config()
    default.transcription.provider = "groq"
    default.providers.groq.api_key = "gsk-global"
    gateway = Config()
    gateway.transcription.provider = "groq"
    save_config(default, default_path)
    save_config(gateway, gateway_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", default_path)

    event, payload = await webui_transcription_event(
        {
            "request_id": "voice-explicit",
            "data_url": _audio_data_url(),
        },
        config_path=gateway_path,
    )

    assert event == "transcription_error"
    assert payload["detail"] == "not_configured"


@pytest.mark.asyncio
async def test_webui_transcribe_audio_rejects_unsupported_mime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.transcription.provider = "groq"
    config.providers.groq.api_key = "gsk-test"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)

    event, payload = await webui_transcription_event({
        "request_id": "voice-1",
        "data_url": _audio_data_url(mime="text/plain"),
    })

    assert event == "transcription_error"
    assert payload["request_id"] == "voice-1"
    assert payload["detail"] == "mime"


@pytest.mark.asyncio
async def test_webui_transcribe_audio_rejects_oversized_audio(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.transcription.provider = "groq"
    config.transcription.max_upload_mb = 1
    config.providers.groq.api_key = "gsk-test"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)
    monkeypatch.setattr("nanobot.audio.transcription.get_media_dir", lambda _channel=None: tmp_path)

    event, payload = await webui_transcription_event({
        "request_id": "voice-1",
        "data_url": _audio_data_url(payload=b"x" * (1024 * 1024 + 1)),
    })

    assert event == "transcription_error"
    assert payload["request_id"] == "voice-1"
    assert payload["detail"] == "size"


@pytest.mark.asyncio
async def test_webui_transcribe_audio_returns_text_and_removes_temp_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    config = Config()
    config.transcription.provider = "groq"
    config.providers.groq.api_key = "gsk-test"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)
    monkeypatch.setattr(
        "nanobot.audio.transcription.get_media_dir",
        lambda _channel=None: media_dir,
    )
    captured_paths: list[Path] = []

    async def fake_transcribe_audio_file(path: str | Path, _resolved: Any, **_kwargs: Any) -> str:
        p = Path(path)
        assert p.exists()
        captured_paths.append(p)
        return "hello voice"

    monkeypatch.setattr(
        "nanobot.audio.transcription.transcribe_audio_file",
        fake_transcribe_audio_file,
    )

    event, payload = await webui_transcription_event({
        "request_id": "voice-1",
        "data_url": _audio_data_url(payload=b"webm voice", mime="audio/webm;codecs=opus"),
        "duration_ms": 1200,
    })

    assert event == "transcription_result"
    assert payload == {"request_id": "voice-1", "text": "hello voice"}
    assert captured_paths
    assert not captured_paths[0].exists()


@pytest.mark.asyncio
async def test_webui_transcription_sends_deltas_before_final_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    config = Config()
    config.transcription.provider = "xiaomi_mimo"
    config.providers.xiaomi_mimo.api_key = "mimo-test"
    save_config(config, config_path)
    monkeypatch.setattr("nanobot.config.loader._current_config_path", config_path)
    monkeypatch.setattr("nanobot.audio.transcription.get_media_dir", lambda _channel=None: tmp_path)

    async def fake_transcribe_audio_file(_path: str | Path, _resolved: Any, *, on_delta) -> str:
        await on_delta("hello ")
        await on_delta("voice")
        return "hello voice"

    monkeypatch.setattr(
        "nanobot.audio.transcription.transcribe_audio_file",
        fake_transcribe_audio_file,
    )
    sent: list[tuple[str, dict[str, Any]]] = []

    async def send_event(event: str, payload: dict[str, Any]) -> None:
        sent.append((event, payload))

    event, payload = await webui_transcription_event(
        {"request_id": "voice-2", "data_url": _audio_data_url(mime="audio/wav")},
        send_event=send_event,
    )

    assert sent == [
        ("transcription_delta", {"request_id": "voice-2", "delta": "hello "}),
        ("transcription_delta", {"request_id": "voice-2", "delta": "voice"}),
    ]
    assert (event, payload) == (
        "transcription_result",
        {"request_id": "voice-2", "text": "hello voice"},
    )


def _doubao_config_path(tmp_path: Path, *, max_duration_sec: int = 120) -> Path:
    config_path = tmp_path / "config.json"
    config = Config()
    config.transcription.provider = "doubao_asr"
    config.transcription.max_duration_sec = max_duration_sec
    config.providers.doubao_asr.api_key = "doubao-key"
    save_config(config, config_path)
    return config_path


class _EventLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.finished = asyncio.Event()

    async def send(self, event: str, payload: dict[str, Any]) -> None:
        self.events.append((event, payload))
        if event in {"transcription_result", "transcription_error"}:
            self.finished.set()


def _pcm_frame(request_id: str, pcm: bytes) -> dict[str, Any]:
    return {"request_id": request_id, "audio": base64.b64encode(pcm).decode("ascii")}


@pytest.mark.asyncio
async def test_realtime_session_streams_partials_and_caps_audio_at_max_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[bytes] = []

    async def fake_transcribe_realtime(_self: Any, pcm_chunks: AsyncIterable[bytes], on_partial) -> str:
        async for chunk in pcm_chunks:
            received.append(chunk)
            await on_partial("你好")
        return "你好。"

    monkeypatch.setattr(
        "nanobot.providers.doubao_transcription.DoubaoTranscriptionProvider.transcribe_realtime",
        fake_transcribe_realtime,
    )
    sessions = RealtimeTranscriptionSessions()
    log = _EventLog()
    owner = object()

    await sessions.start(
        owner,
        {"request_id": "rt-1"},
        log.send,
        config_path=_doubao_config_path(tmp_path, max_duration_sec=1),
    )
    sessions.push_audio(owner, _pcm_frame("rt-1", b"a" * 20_000))
    sessions.push_audio(owner, _pcm_frame("rt-1", b"b" * 20_000))
    sessions.push_audio(owner, _pcm_frame("rt-1", b"c" * 100))
    await asyncio.wait_for(log.finished.wait(), timeout=2)

    assert [len(chunk) for chunk in received] == [20_000, 12_000]
    assert log.events[-1] == ("transcription_result", {"request_id": "rt-1", "text": "你好。"})
    assert ("transcription_partial", {"request_id": "rt-1", "text": "你好"}) in log.events


@pytest.mark.asyncio
async def test_realtime_session_reports_provider_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_transcribe_realtime(_self: Any, _chunks: AsyncIterable[bytes], _on_partial) -> str:
        raise TranscriptionProviderError("resource not granted")

    monkeypatch.setattr(
        "nanobot.providers.doubao_transcription.DoubaoTranscriptionProvider.transcribe_realtime",
        failing_transcribe_realtime,
    )
    sessions = RealtimeTranscriptionSessions()
    log = _EventLog()
    owner = object()

    await sessions.start(owner, {"request_id": "rt-2"}, log.send, config_path=_doubao_config_path(tmp_path))
    await asyncio.wait_for(log.finished.wait(), timeout=2)
    sessions.push_audio(owner, _pcm_frame("rt-2", b"late"))

    assert log.events == [
        (
            "transcription_error",
            {"detail": "provider_error", "provider": "doubao_asr", "request_id": "rt-2"},
        )
    ]


@pytest.mark.asyncio
async def test_realtime_session_is_cancelled_when_connection_is_discarded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def waiting_transcribe_realtime(_self: Any, pcm_chunks: AsyncIterable[bytes], _on_partial) -> str:
        started.set()
        try:
            async for _chunk in pcm_chunks:
                pass
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ""

    monkeypatch.setattr(
        "nanobot.providers.doubao_transcription.DoubaoTranscriptionProvider.transcribe_realtime",
        waiting_transcribe_realtime,
    )
    sessions = RealtimeTranscriptionSessions()
    log = _EventLog()
    owner = object()

    await sessions.start(owner, {"request_id": "rt-3"}, log.send, config_path=_doubao_config_path(tmp_path))
    await asyncio.wait_for(started.wait(), timeout=2)
    sessions.discard(owner)

    await asyncio.wait_for(cancelled.wait(), timeout=2)
    assert log.events == []
