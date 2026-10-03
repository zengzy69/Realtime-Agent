"""Doubao bidirectional TTS adapter against a local fake WebSocket server."""

from __future__ import annotations

import json
import struct
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from nanobot.providers.doubao_tts import DoubaoTTSProvider, SpeechSynthesisError

_CONNECTION_EVENTS = (1, 2, 50, 51, 52)


def _server_frame(event: int, payload: bytes = b"{}", *, message_type: int = 0b1001) -> bytes:
    frame = bytes([0x11, (message_type << 4) | 0b0100, 0x10, 0x00]) + struct.pack(">i", event)
    identifier = b"connect-1" if event in _CONNECTION_EVENTS else b"session-1"
    frame += struct.pack(">I", len(identifier)) + identifier
    return frame + struct.pack(">I", len(payload)) + payload


def _client_frame(raw: bytes) -> tuple[int, dict[str, Any]]:
    (event,) = struct.unpack_from(">i", raw, 4)
    offset = 8
    if event not in _CONNECTION_EVENTS:
        (size,) = struct.unpack_from(">I", raw, offset)
        offset += 4 + size
    (size,) = struct.unpack_from(">I", raw, offset)
    return event, json.loads(raw[offset + 4:offset + 4 + size])


@asynccontextmanager
async def _fake_doubao(handler: Callable[[ServerConnection], Awaitable[None]]) -> AsyncIterator[str]:
    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        yield f"ws://127.0.0.1:{port}"


async def _open_session(ws: ServerConnection, seen: dict[str, Any]) -> None:
    assert _client_frame(await ws.recv())[0] == 1
    await ws.send(_server_frame(50))
    event, start = _client_frame(await ws.recv())
    seen["start"] = (event, start)
    await ws.send(_server_frame(150))
    seen["task"] = _client_frame(await ws.recv())
    seen["finish"] = _client_frame(await ws.recv())[0]


@pytest.mark.asyncio
async def test_synthesize_streams_audio_in_whole_samples() -> None:
    seen: dict[str, Any] = {}

    async def handler(ws: ServerConnection) -> None:
        assert ws.request is not None
        seen["headers"] = ws.request.headers
        await _open_session(ws, seen)
        await ws.send(_server_frame(350))
        await ws.send(_server_frame(352, b"\x01\x02\x03", message_type=0b1011))
        await ws.send(_server_frame(352, b"\x04\x05\x06", message_type=0b1011))
        await ws.send(_server_frame(152))

    chunks: list[bytes] = []

    async def on_audio(chunk: bytes) -> None:
        chunks.append(chunk)

    async with _fake_doubao(handler) as url:
        provider = DoubaoTTSProvider("doubao-key", voice="zh_female_test", api_base=url)
        await provider.synthesize("你好。", on_audio)

    assert chunks == [b"\x01\x02", b"\x03\x04\x05\x06"]
    assert seen["headers"]["X-Api-Key"] == "doubao-key"
    assert seen["headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"
    event, start = seen["start"]
    assert event == 100
    assert start["req_params"]["speaker"] == "zh_female_test"
    assert start["req_params"]["audio_params"] == {"format": "pcm", "sample_rate": 24000}
    task_event, task = seen["task"]
    assert (task_event, task["req_params"]["text"]) == (200, "你好。")
    assert seen["finish"] == 102


@pytest.mark.asyncio
async def test_synthesize_raises_on_session_failure() -> None:
    async def handler(ws: ServerConnection) -> None:
        await _open_session(ws, {})
        await ws.send(_server_frame(153, b'{"error":"speaker not granted"}'))

    async def on_audio(_chunk: bytes) -> None:
        raise AssertionError("no audio expected")

    async with _fake_doubao(handler) as url:
        provider = DoubaoTTSProvider("doubao-key", voice="zh_female_test", api_base=url)
        with pytest.raises(SpeechSynthesisError, match="speaker not granted"):
            await provider.synthesize("你好。", on_audio)
