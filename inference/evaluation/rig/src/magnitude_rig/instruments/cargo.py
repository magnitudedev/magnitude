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
    cargo passes each test executable ahead of the arguments after `--`), or nothing."""
    return [value for option, value in options(arguments) if option == ""]


def options(arguments: list[str]) -> list[tuple[str, str | None]]:
    """Cargo test's own `arguments` as (option, value or None), positional names as ("", name):
    a valued option takes the next argument, unless joined to it (`--opt=v`) or that is itself
    an option (`--test` alone, which lists the targets)."""
    found, taking = [], None
    for argument in arguments:
        if taking is not None and not argument.startswith("-"):
            found.append((taking, argument))
            taking = None
            continue
        if taking is not None:
            found.append((taking, None))
            taking = None
        if argument.startswith("-") and "=" in argument:
            found.append(tuple(argument.split("=", 1)))
        elif argument in VALUED:
            taking = argument
        elif argument.startswith("-"):
            found.append((argument, None))
        else:
            found.append(("", argument))
    if taking is not None:
        found.append((taking, None))
    return found


PACKAGES = {"-p", "--package"}
FEATURES = {"-F", "--features", "--all-features", "--no-default-features"}
PROFILES = {"--release", "-r", "--profile"}


def packages(arguments: list[str]) -> set[str]:
    """The packages cargo test's own `arguments` select."""
    return {value for option, value in options(arguments) if option in PACKAGES and value}


def configuration(arguments: list[str]) -> str:
    """What cargo test's own `arguments` build the selected tests as, beyond the tree: the
    profile, the feature options and the target, in one canonical text (`release`,
    `test +a,b --target x`), which equal trees build equally under."""
    found = options(arguments)
    profile = next((value if option == "--profile" else "release" for option, value in found
                    if option in PROFILES), "test")
    features = sorted({feature for option, value in found if option in {"-F", "--features"} and value
                       for feature in value.replace(",", " ").split()})
    flags = sorted({option for option, _ in found if option in {"--all-features", "--no-default-features"}})
    target = [f"--target {value}" for option, value in found if option == "--target"]
    return " ".join([profile, *([f"+{','.join(features)}"] if features else []), *flags, *target])


def without_targets(arguments: list[str]) -> list[str]:
    """Cargo test's own `arguments` without their target selections and test name filter:
    what builds the selection, and selects its packages."""
    kept = []
    for option, value in options(arguments):
        if option == "" or option.split("=", 1)[0] in TARGETS | {"--doc"}:
            continue
        kept += [option] if value is None else [option, value]
    return kept


def failures(lines: list[str], limit: int = 20) -> list[str]:
    """The compiler's error lines (short message format), or nothing."""
    return [line for line in lines if ": error" in line or line.startswith("error")][:limit]
