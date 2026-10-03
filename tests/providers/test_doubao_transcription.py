"""Doubao streaming ASR adapter against a local fake SAUC WebSocket server."""

from __future__ import annotations

import gzip
import json
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve
from websockets.http11 import Request, Response

from nanobot.audio.transcription_registry import TranscriptionProviderError
from nanobot.providers.doubao_transcription import DoubaoTranscriptionProvider


def _server_frame(payload: dict[str, Any], *, sequence: int, last: bool = False) -> bytes:
    body = gzip.compress(json.dumps(payload).encode())
    flags = 0b0011 if last else 0b0001
    return bytes([0x11, (0b1001 << 4) | flags, 0x11, 0x00]) + struct.pack(">iI", sequence, len(body)) + body


def _error_frame(code: int, message: str) -> bytes:
    body = gzip.compress(message.encode())
    return bytes([0x11, 0b1111 << 4, 0x11, 0x00]) + struct.pack(">iI", code, len(body)) + body


def _client_frame(raw: bytes) -> tuple[int, int, bytes]:
    message_type = raw[1] >> 4
    sequence, size = struct.unpack(">iI", raw[4:12])
    return message_type, sequence, gzip.decompress(raw[12:12 + size])


@asynccontextmanager
async def _fake_doubao(
    handler: Callable[[ServerConnection], Awaitable[None]],
    process_request: Callable[[ServerConnection, Request], Response | None] | None = None,
) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0, process_request=process_request) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


async def _chunks(*parts: bytes) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_realtime_reports_partials_then_returns_final_text() -> None:
    seen: dict[str, Any] = {"audio": b""}

    async def handler(ws: ServerConnection) -> None:
        assert ws.request is not None
        seen["headers"] = ws.request.headers
        message_type, sequence, payload = _client_frame(await ws.recv())
        seen["config"] = (message_type, sequence, json.loads(payload))
        await ws.send(_server_frame({}, sequence=1))
        sequence = 0
        while sequence >= 0:
            _, sequence, audio = _client_frame(await ws.recv())
            seen["audio"] += audio
            seen["last_sequence"] = sequence
            if audio:
                text = "你好" if len(seen["audio"]) < 8 else "你好，世界"
                await ws.send(_server_frame({"result": {"text": text}}, sequence=sequence))
        await ws.send(_server_frame({"result": {"text": "你好，世界。"}}, sequence=-sequence, last=True))

    partials: list[str] = []

    async def on_partial(text: str) -> None:
        partials.append(text)

    async with _fake_doubao(handler) as url:
        provider = DoubaoTranscriptionProvider(api_key="doubao-key", api_base=url, model="volc.test.resource")
        text = await provider.transcribe_realtime(_chunks(b"pcm1", b"pcm2"), on_partial)

    assert text == "你好，世界。"
    assert partials == ["你好", "你好，世界"]
    assert seen["headers"]["X-Api-Key"] == "doubao-key"
    assert seen["headers"]["X-Api-Resource-Id"] == "volc.test.resource"
    message_type, sequence, config = seen["config"]
    assert (message_type, sequence) == (0b0001, 1)
    assert config["audio"] == {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1}
    assert config["request"]["end_window_size"] == 800
    assert config["request"]["force_to_speech_time"] == 1000
    assert seen["audio"] == b"pcm1pcm2"
    assert seen["last_sequence"] == -4


@pytest.mark.asyncio
async def test_realtime_surfaces_rejected_resource() -> None:
    def reject(connection: ServerConnection, _request: Request) -> Response:
        return connection.respond(HTTPStatus.FORBIDDEN, '{"error":"requested resource not granted"}')

    async def handler(_ws: ServerConnection) -> None:
        raise AssertionError("handshake should be rejected")

    async def on_partial(_text: str) -> None:
        raise AssertionError("no partials expected")

    async with _fake_doubao(handler, reject) as url:
        provider = DoubaoTranscriptionProvider(api_key="doubao-key", api_base=url)
        with pytest.raises(TranscriptionProviderError, match=r"403.*requested resource not granted"):
            await provider.transcribe_realtime(_chunks(b"pcm"), on_partial)


@pytest.mark.asyncio
async def test_file_transcription_sends_container_format_and_returns_empty_on_server_error(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"OggS-voice")
    seen: dict[str, Any] = {}

    async def handler(ws: ServerConnection) -> None:
        _, _, payload = _client_frame(await ws.recv())
        seen["audio_format"] = json.loads(payload)["audio"]
        await ws.send(_server_frame({}, sequence=1))
        await ws.recv()
        await ws.send(_error_frame(45000151, "invalid audio format"))

    async with _fake_doubao(handler) as url:
        provider = DoubaoTranscriptionProvider(api_key="doubao-key", api_base=url)
        assert await provider.transcribe(audio_path) == ""

    assert seen["audio_format"] == {"format": "ogg", "codec": "opus"}


@pytest.mark.asyncio
async def test_realtime_returns_when_doubao_marks_the_utterance_definite() -> None:
    async def handler(ws: ServerConnection) -> None:
        await ws.recv()
        await ws.send(_server_frame({}, sequence=1))
        await ws.recv()
        await ws.send(_server_frame(
            {"result": {"text": "你好", "utterances": [{"text": "你好", "definite": False}]}},
            sequence=2,
        ))
        await ws.recv()
        await ws.send(_server_frame(
            {"result": {"text": "你好。", "utterances": [{"text": "你好。", "definite": True}]}},
            sequence=3,
        ))

    partials: list[str] = []

    async def on_partial(text: str) -> None:
        partials.append(text)

    async with _fake_doubao(handler) as url:
        provider = DoubaoTranscriptionProvider(api_key="doubao-key", api_base=url)
        text = await provider.transcribe_realtime(_chunks(b"pcm1", b"pcm2"), on_partial)

    assert text == "你好。"
    assert partials == ["你好"]
