"""Splitting cargo arguments, shared by the instruments that run cargo."""


def split(arguments: list[str]) -> tuple[list[str], list[str]]:
    """Cargo's own arguments, and those after its `--` (for the test binaries)."""
    if "--" in arguments:
        at = arguments.index("--")
        return arguments[:at], arguments[at + 1 :]
    return arguments, []


def failures(lines: list[str], limit: int = 20) -> list[str]:
    """The compiler's error lines (short message format), or nothing."""
    return [line for line in lines if ": error" in line or line.startswith("error")][:limit]
