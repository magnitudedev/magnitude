"""The selectors every instrument shares, parsed and checked once, before anything queues.

An instrument declares which selectors it takes; each is added to its parser from here and
checked locally (a model against the catalog, a ref against git, a duration's syntax), so a
wrong argument fails in a second instead of after a queue and a build.
"""

import argparse
import re
from dataclasses import dataclass

from magnitude_bench.models import catalog_reference

from . import templates
from .sources import WORKTREE, SourceUnavailable, checkout, commit

BACKENDS = ("metal", "cuda", "vulkan", "cpu")
_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s?)?$")


def seconds(text: str) -> float:
    """`90`, `90s`, `2m`, `1h30m` or `2m30s` as seconds."""
    match = _DURATION.match(text.strip())
    if not text.strip() or match is None or not any(match.groups()):
        raise argparse.ArgumentTypeError(f"{text!r} is not a duration such as 45s, 2m or 1h30m")
    hours, minutes, rest = match.groups()
    return int(hours or 0) * 3600 + int(minutes or 0) * 60 + float(rest or 0)


def model(text: str) -> str:
    """A catalog model `<id>:<variant>`, as the serving benchmark (the catalog's reader here)
    checks it."""
    try:
        return catalog_reference(text, checkout())
    except (ValueError, SourceUnavailable) as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def build(text: str) -> str:
    if text == WORKTREE:
        return text
    try:
        commit(text)
    except SourceUnavailable as error:
        raise argparse.ArgumentTypeError(str(error)) from None
    return text


@dataclass(frozen=True)
class Selector:
    flags: tuple[str, ...]
    options: dict


SELECTORS = {
    "backend": Selector(("--backend",), {"choices": BACKENDS, "required": True,
                                         "help": "what the run needs; places it on a host that has it"}),
    "host": Selector(("--host",), {"help": "a host or pool, only when the question is about it"}),
    "build": Selector(("--build",), {"type": build, "default": WORKTREE,
                                     "help": "`worktree` (default: the git worktree containing the current "
                                             "directory, never mrig's own) or a git ref"}),
    "model": Selector(("--model",), {"type": model, "required": True,
                                     "help": "catalog model <id>:<variant>"}),
    "model-or-checkpoint": Selector(("--model",), {"type": model,
                                                   "help": "catalog model <id>:<variant>, or --checkpoint DIR"}),
    "model-or-catalog": Selector(("--model",), {"type": model,
                                                "help": "catalog model <id>:<variant>, or --catalog for all"}),
    "template": Selector(("--template",), {"type": templates.name, "required": True,
                                           "help": "the request measured, <key>:<context>, "
                                                   "e.g. repeat:8k (1k = 1024 tokens)"}),
    "templates": Selector(("--template",), {"type": templates.name, "action": "append", "required": True,
                                            "help": "a request measured (repeatable), "
                                                    "<key>:<context>, e.g. repeat:8k"}),
    "entry": Selector(("--entry",), {"required": True, "help": "kernel entry name, e.g. attention_decode_k8v4"}),
    "purpose": Selector(("--purpose",), {"required": True,
                                         "help": "what this run settles, in a sentence"}),
    "estimate": Selector(("--estimate",), {"type": seconds, "required": True,
                                           "help": "how long it should take, e.g. 2m"}),
}


def add(parser: argparse.ArgumentParser, names: tuple[str, ...]) -> None:
    for name in names:
        selector = SELECTORS[name]
        parser.add_argument(*selector.flags, **selector.options)
