"""Spoken-reply excerpt selection."""

from __future__ import annotations

import pytest

from nanobot.audio.speech import SpokenExcerpt, spoken_excerpt


@pytest.mark.parametrize(
    ("reply", "max_chars", "expected"),
    [
        (
            "明天北京晴，最高 25 度，适合出门。",
            60,
            SpokenExcerpt("明天北京晴，最高 25 度，适合出门。", truncated=False),
        ),
        (
            "**结论**：用 [uv](https://docs.astral.sh/uv/) 安装即可\n\n## 步骤\n\n```bash\nuv sync\n```",
            60,
            SpokenExcerpt("结论：用 uv 安装即可。", truncated=True),
        ),
        (
            "第一句说明结论。第二句补充一个很长很长很长很长很长很长的背景。第三句继续展开。",
            20,
            SpokenExcerpt("第一句说明结论。", truncated=True),
        ),
        (
            "这是一句超过六十个字的说明" + "啊" * 40 + "。后面还有一句。",
            60,
            SpokenExcerpt("这是一句超过六十个字的说明" + "啊" * 40 + "。", truncated=True),
        ),
        (
            "Use uv to install it. Version 3.5 works, and the rest is optional.",
            40,
            SpokenExcerpt("Use uv to install it.", truncated=True),
        ),
        (
            "没有句号的很长一段说明需要在逗号处收住，后面还要继续说很多很多很多很多很多内容",
            20,
            SpokenExcerpt("没有句号的很长一段说明需要在逗号处收住。", truncated=True),
        ),
        ("```python\nprint('hi')\n```", 60, SpokenExcerpt("", truncated=True)),
    ],
)
def test_spoken_excerpt_reads_only_the_leading_plain_paragraph(
    reply: str, max_chars: int, expected: SpokenExcerpt,
) -> None:
    assert spoken_excerpt(reply, max_chars) == expected
