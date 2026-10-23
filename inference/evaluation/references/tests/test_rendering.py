"""Detecting that the builds render a fixture set to different tokens."""

import json

from magnitude_references import rendering


def line(fixture_id: str, tokens: list[int], template: str = "t1") -> dict:
    return {
        "fixture_id": fixture_id,
        "template_sha256": template,
        "tokens": tokens,
        "scored_spans": [[2, 4]],
    }


def write(path, lines: list[dict]):
    path.write_text("".join(json.dumps(item) + "\n" for item in lines))
    return path


def test_identical_files(tmp_path):
    lines = [line("a", [1, 2, 3, 4]), line("b", [5, 6, 7, 8])]
    assert rendering.compare_files(write(tmp_path / "x", lines), write(tmp_path / "y", lines)) == []


def test_token_and_template_differences(tmp_path):
    base = [line("a", [1, 2, 3, 4]), line("b", [5, 6, 7, 8])]
    candidate = [line("a", [1, 2, 3, 4]), line("b", [5, 6, 9, 8, 10], template="t2")]
    [difference] = rendering.compare_files(
        write(tmp_path / "x", base), write(tmp_path / "y", candidate)
    )
    assert difference == {
        "fixture_id": "b",
        "fields": ["template_sha256", "tokens"],
        "template_sha256": {"base": "t1", "candidate": "t2"},
        "tokens": {"base": 4, "candidate": 5, "first_difference": 2},
    }


def test_prefix_difference_is_at_the_shorter_end():
    [difference] = rendering.differences([line("a", [1, 2])], [line("a", [1, 2, 3])])
    assert difference["tokens"]["first_difference"] == 2


def test_different_fixtures():
    [difference] = rendering.differences([line("a", [1])], [line("b", [1])])
    assert difference == {"fixtures": {"base": ["a"], "candidate": ["b"]}}


def test_same_content_in_different_bytes(tmp_path):
    lines = [line("a", [1, 2, 3, 4])]
    (tmp_path / "y").write_text(json.dumps(lines[0], indent=None, separators=(",", ":")) + "\n")
    [difference] = rendering.compare_files(write(tmp_path / "x", lines), tmp_path / "y")
    assert "encoding" in difference
