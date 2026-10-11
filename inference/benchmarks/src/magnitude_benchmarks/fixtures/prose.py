"""One pinned prose source; downloaded content is never repository data."""

import hashlib
import json
import re
from pathlib import Path

import httpx

from .storage import cache_root, fetch


def normalize(content: bytes) -> str:
    text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.M)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.M)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("missing or unordered Gutenberg body markers")
    return text[start.end() : end.start()].strip()


async def prepare(*, cache: Path | None = None) -> tuple[str, dict[str, str]]:
    lock = json.loads((Path(__file__).parent / "data/moby-dick.lock.json").read_text())
    root = cache or cache_root()
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        content = await fetch(
            client, lock["url"], lock["sha256"], root / "sources" / lock["sha256"] / "moby-dick.txt"
        )
    text = normalize(content)
    return text, {**lock, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
