"""Whether two builds render a template's sessions to the same tokens.

Rendering belongs to the engine's serving path (chat template and tokenizer), so a difference
between builds is a finding in itself: template or tokenizer behaviour changed. Quality
comparisons between the builds, and of the candidate against the base's reference, need
identical rendered input.
"""

import json
from pathlib import Path

RENDERED = "rendered.jsonl"


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _first_difference(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right))


def differences(base: list[dict], candidate: list[dict]) -> list[dict]:
    """Per-session differences between two rendered-session files (empty when identical)."""
    base_ids = [[line["template"], line["lane"]] for line in base]
    candidate_ids = [[line["template"], line["lane"]] for line in candidate]
    if base_ids != candidate_ids:
        return [{"sessions": {"base": base_ids, "candidate": candidate_ids}}]
    found = []
    for left, right in zip(base, candidate, strict=True):
        if left == right:
            continue
        fields = sorted(
            name for name in left.keys() | right.keys() if left.get(name) != right.get(name)
        )
        difference = {"template": left["template"], "lane": left["lane"], "fields": fields}
        if "chat_template_sha256" in fields:
            difference["chat_template_sha256"] = {
                "base": left["chat_template_sha256"],
                "candidate": right["chat_template_sha256"],
            }
        if "tokens" in fields:
            difference["tokens"] = {
                "base": len(left["tokens"]),
                "candidate": len(right["tokens"]),
                "first_difference": _first_difference(left["tokens"], right["tokens"]),
            }
        found.append(difference)
    return found


def compare_files(base: Path, candidate: Path) -> list[dict]:
    if base.read_bytes() == candidate.read_bytes():
        return []
    found = differences(read_lines(base), read_lines(candidate))
    # Identical content in different bytes still changes the rendered digest.
    return found or [{"encoding": "the rendered files hold the same sessions in different bytes"}]
