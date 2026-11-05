"""Splitting cargo arguments, shared by the instruments that run cargo."""


def split(arguments: list[str]) -> tuple[list[str], list[str]]:
    """Cargo's own arguments, and those after its `--` (for the test binaries)."""
    if "--" in arguments:
        at = arguments.index("--")
        return arguments[:at], arguments[at + 1 :]
    return arguments, []


# Cargo's options that take a value, given as the next argument unless joined to it (`--opt=v`,
# `-pv`): what a test name filter is not.
VALUED = {"-p", "--package", "--exclude", "--bin", "--example", "--test", "--bench", "-F", "--features", "-j",
          "--jobs", "--profile", "--target", "--target-dir", "--manifest-path", "--message-format", "--color",
          "--config", "-Z", "--lockfile-path", "--artifact-dir"}


# Cargo's target selections but `--doc`: a test run that names none of them runs the
# documentation tests too.
TARGETS = {"--lib", "--bin", "--bins", "--test", "--tests", "--example", "--examples", "--bench", "--benches",
           "--all-targets"}


def runs_doc_tests(arguments: list[str]) -> bool:
    """Whether cargo test's own `arguments` run the documentation tests: they name `--doc`, or
    no target selection at all."""
    selected = {argument.split("=", 1)[0] for argument in arguments}
    return "--doc" in selected or not selected & TARGETS


def test_name(arguments: list[str]) -> list[str]:
    """The test name filter among cargo test's own `arguments` (its positional TESTNAME, which
    cargo passes each test executable ahead of the arguments after `--`), or nothing. A valued
    option takes the next argument, unless that is itself an option (`--test` alone, which lists
    the targets)."""
    names, taking = [], False
    for argument in arguments:
        if taking and not argument.startswith("-"):
            taking = False
        elif argument.startswith("-"):
            taking = argument in VALUED
        else:
            names.append(argument)
    return names


def failures(lines: list[str], limit: int = 20) -> list[str]:
    """The compiler's error lines (short message format), or nothing."""
    return [line for line in lines if ": error" in line or line.startswith("error")][:limit]
