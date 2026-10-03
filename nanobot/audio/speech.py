"""Application-level spoken replies.

The gateway decides which part of an assistant reply is read aloud: the
leading paragraph, stripped of markdown. It prefers complete sentences that
fit in ``tts.max_chars``; a first sentence longer than the cap is still
spoken in full. Everything else stays text-only. Provider protocol details live in
``nanobot.providers.doubao_tts``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from loguru import logger

from nanobot.config.loader import resolve_env_refs
from nanobot.config.schema import Config
from nanobot.providers.doubao_tts import AudioChunkHandler, DoubaoTTSProvider, SpeechSynthesisError

# Volcengine issues one speech API key for both recognition and synthesis.
_CREDENTIAL_PROVIDER = "doubao_asr"
_CREDENTIAL_ENV_KEY = "DOUBAO_ASR_API_KEY"

_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW = re.compile(r"^\s*\|")
_BLOCK_PREFIX = re.compile(r"^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
_EMPHASIS = re.compile(r"(\*\*|__|~~|`)")
_SINGLE_STAR = re.compile(r"(?<!\w)\*(\S[^*]*?)\*(?!\w)")
_SENTENCE_END = re.compile(r"[。！？；!?;]|\.(?=\s|$)")
_CLAUSE_END = re.compile(r"[，、,：:]")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass(frozen=True)
class EffectiveSpeechConfig:
    enabled: bool
    voice: str
    max_chars: int
    api_key: str = field(repr=False)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class SpokenExcerpt:
    text: str
    truncated: bool


class SpeechIngressError(Exception):
    """Stable speech synthesis error surfaced to WebUI clients."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def resolve_speech_config(config: Config) -> EffectiveSpeechConfig:
    credentials = getattr(config.providers, _CREDENTIAL_PROVIDER)
    api_key = resolve_env_refs(credentials.api_key or "") or os.environ.get(_CREDENTIAL_ENV_KEY, "")
    return EffectiveSpeechConfig(
        enabled=config.tts.enabled,
        voice=config.tts.voice,
        max_chars=config.tts.max_chars,
        api_key=api_key,
    )


def _plain_line(line: str) -> str:
    line = _BLOCK_PREFIX.sub("", line)
    line = _IMAGE.sub("", line)
    line = _LINK.sub(r"\1", line)
    line = _URL.sub("", line)
    line = _EMPHASIS.sub("", line)
    return _SINGLE_STAR.sub(r"\1", line).strip()


def _cut_at_clause(text: str, max_chars: int) -> str:
    window = text[:max_chars]
    ends = [match.end() for match in _CLAUSE_END.finditer(window)]
    if ends and ends[-1] >= max_chars // 3:
        return window[:ends[-1]].strip()
    return window.strip()


def _complete_sentences(text: str, max_chars: int) -> str:
    """Keep whole sentences, targeting *max_chars* without splitting one."""
    if len(text) <= max_chars:
        return text
    sentences: list[str] = []
    last = 0
    for match in _SENTENCE_END.finditer(text):
        sentences.append(text[last:match.end()])
        last = match.end()
    tail = text[last:]
    if tail.strip():
        sentences.append(tail)
    if not sentences:
        return _cut_at_clause(text, max_chars)
    chosen = sentences[0]
    if _SENTENCE_END.search(chosen) is None:
        return _cut_at_clause(chosen, max_chars) if len(chosen) > max_chars else chosen
    for sentence in sentences[1:]:
        candidate = chosen + sentence
        if len(candidate) <= max_chars:
            chosen = candidate
        else:
            break
    return chosen.strip()


def spoken_excerpt(reply: str, max_chars: int) -> SpokenExcerpt:
    """Pick the leading paragraph of *reply* as plain speech, targeting *max_chars*."""
    lines = reply.replace("\r\n", "\n").split("\n")
    collected: list[str] = []
    rest_index = len(lines)
    for index, line in enumerate(lines):
        if _FENCE.match(line) or _TABLE_ROW.match(line):
            rest_index = index
            break
        if not line.strip():
            if collected:
                rest_index = index
                break
            continue
        if plain := _plain_line(line):
            collected.append(plain)
    paragraph = re.sub(r"\s+", " ", " ".join(collected)).strip()
    has_rest = any(line.strip() for line in lines[rest_index:])
    if not paragraph:
        return SpokenExcerpt(text="", truncated=has_rest)
    text = _complete_sentences(paragraph, max_chars)
    truncated = has_rest or len(text) < len(paragraph)
    if truncated:
        text = text.rstrip("，、,：: ")
        if not _SENTENCE_END.search(text[-1:]):
            text += "。" if _CJK.search(text) else "."
    return SpokenExcerpt(text=text, truncated=truncated)


def speech_excerpt_for(reply: str, config: EffectiveSpeechConfig) -> SpokenExcerpt:
    if not config.enabled:
        raise SpeechIngressError("disabled")
    if not config.configured:
        raise SpeechIngressError("not_configured")
    excerpt = spoken_excerpt(reply, config.max_chars)
    if not excerpt.text:
        raise SpeechIngressError("empty")
    return excerpt


async def synthesize_speech(text: str, config: EffectiveSpeechConfig, on_audio: AudioChunkHandler) -> None:
    """Stream mono PCM16 audio for *text* to *on_audio*."""
    provider = DoubaoTTSProvider(config.api_key, voice=config.voice)
    try:
        await provider.synthesize(text, on_audio)
    except SpeechSynthesisError as exc:
        logger.warning("Speech synthesis failed: {}", exc)
        raise SpeechIngressError("provider_error") from exc
