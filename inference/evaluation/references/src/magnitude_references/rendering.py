"""Whether two builds render a fixture set to the same tokens.

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
    """Per-fixture differences between two rendered-fixture files (empty when identical)."""
    base_ids = [line["fixture_id"] for line in base]
    candidate_ids = [line["fixture_id"] for line in candidate]
    if base_ids != candidate_ids:
        return [{"fixtures": {"base": base_ids, "candidate": candidate_ids}}]
    found = []
    for left, right in zip(base, candidate, strict=True):
        if left == right:
            continue
        fields = sorted(
            name for name in left.keys() | right.keys() if left.get(name) != right.get(name)
        )
        difference = {"fixture_id": left["fixture_id"], "fields": fields}
        if "template_sha256" in fields:
            difference["template_sha256"] = {
                "base": left["template_sha256"],
                "candidate": right["template_sha256"],
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
    return found or [{"encoding": "the rendered files hold the same fixtures in different bytes"}]
