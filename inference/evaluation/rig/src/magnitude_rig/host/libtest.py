"""Reading libtest's output, the one parser of it (standard library, Python 3.9+, as every host
module): each test's outcome line, the `test result:` totals, the `running N tests` line that
marks an executable with the harness, a `--list` listing, and the `Doc-tests CRATE` heading
cargo prints before a crate's documentation tests."""

from __future__ import annotations

import re

RUNNING = re.compile(r"^running \d+ tests?$")
# A documentation test's name has spaces (`src/lib.rs - f (line 3)`): the name ends at ` ... `.
OUTCOME = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)\b")
RESULT = re.compile(r"^test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")
LISTED = re.compile(r"^(.+): test$")
DOC_TESTS = re.compile(r"^Doc-tests (\S+)$")
KINDS = ("ok", "FAILED", "ignored")
STDERR = "[stderr]"  # how a job's log marks the lines of its standard error


def _text(line: str) -> str:
    return (line[len(STDERR):] if line.startswith(STDERR) else line).strip()


def outcomes(lines: list) -> dict:
    """Each test's outcome (`ok`, `FAILED`, `ignored`) by name."""
    found = {}
    for line in lines:
        match = OUTCOME.match(_text(line))
        if match:
            found[match[1]] = match[2]
    return found


def totals(lines: list):
    """libtest's counts summed over every `test result:` line, as [passed, failed, ignored],
    or None when no line has them (an executable without the harness)."""
    counted = None
    for line in lines:
        match = RESULT.match(_text(line))
        if match:
            counted = [a + int(b) for a, b in zip(counted or [0, 0, 0], match.groups())]
    return counted


def tallied(found: dict) -> list:
    """The counts libtest would total for outcomes `found`, as `totals` gives them."""
    return [sum(1 for outcome in found.values() if outcome == kind) for kind in KINDS]


def harness(lines: list) -> bool:
    return any(RUNNING.match(_text(line)) for line in lines)


def listed(lines: list) -> list:
    return [match[1] for line in lines if (match := LISTED.match(_text(line)))]


def doc_sections(lines: list) -> dict:
    """The documentation tests' lines of a `cargo test` output, by crate."""
    sections, crate = {}, None
    for line in lines:
        heading = DOC_TESTS.match(_text(line))
        if heading:
            crate = heading[1]
            sections[crate] = []
        elif crate is not None:
            sections[crate].append(line)
    return sections
