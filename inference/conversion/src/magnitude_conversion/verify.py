"""verify-publish: whether a converted GGUF, alone, does what a user of it needs. It runs before any upload.

It reads only the file: no checkpoint and no template from outside it. Each check passes or refuses by name:

- **shards:** every shard reads on its own (`check.shard_facts`: alignment, split keys, aligned tensor data);
- **metadata:** upstream llama.cpp's loader, at the release the converter is pinned to, reads every key it reads
  for the file's `general.architecture`, with its own required/optional rules and types (the steps of
  `llama_model_load`, `src/llama.cpp`, short of the tensors). The key set is llama.cpp's code, not a list here;
- **tokenizer:** llama.cpp's tokenizer, from the file's vocabulary and merges, encodes the fixture's texts to the
  checkpoint's `tokenizer.json` ids;
- **chat:** llama.cpp's chat path (`common_chat_templates_apply`, as llama-server calls it) renders the fixture's
  conversations from the file's own `tokenizer.chat_template` byte-equal to the reference renderer.

The fixture is the engine's chat fixture for the architecture, written by `magnitude_references` from the checkpoint's
tokenizer and encoding (`engine/chat/tests/real_tokenizer.rs` checks Magnitude's chat path against the same one).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from conversion.base import gguf

from . import llama
from .check import shard_facts

_PROJECT = Path(__file__).resolve().parents[2]  # inference/conversion
# The renders and token ids each architecture's text file must reproduce
FIXTURES = {
    "deepseek41": _PROJECT.parent / "engine/chat/tests/assets/deepseek41.json",
    "glm5-next": _PROJECT.parent / "engine/chat/tests/assets/glm53.json",
}


class Refused(Exception):
    """The file fails a check; the message names each refused check and what it found."""


# The checks llama.cpp's side reports, each on exactly one line
LLAMA_CHECKS = ("metadata", "tokenizer", "chat")


def _report(run: subprocess.CompletedProcess) -> tuple[list[str], list[str]]:
    """The passed and refused lines of the verifier's report: exactly one, pass or refused, per check, and exit 1
    exactly when one is refused. Any other report is the verifier's fault, not the file's."""
    passed, refused, reported = [], [], []
    for line in run.stdout.splitlines():
        check, _, outcome = line.partition(": ")
        if outcome.startswith("refused: "):
            refused.append(line)
        elif outcome == "pass" or outcome.startswith("pass, "):
            passed.append(line)
        else:
            raise RuntimeError(f"verify-publish reported a line that neither passes nor refuses: {line!r}")
        reported.append(check)
    if sorted(reported) != sorted(LLAMA_CHECKS) or run.returncode != (1 if refused else 0):
        raise RuntimeError(f"verify-publish exited {run.returncode} reporting {reported}, not one line each for "
                           f"{', '.join(LLAMA_CHECKS)}: {run.stdout}{run.stderr}")
    return passed, refused


def verify(first_shard: Path) -> list[str]:
    """Each check's line. Raises `Refused` with every refused check's line."""
    lines, refused = [], []
    try:
        readers = shard_facts(first_shard)
        lines.append(f"shards: pass, {len(readers)} read on their own")
    except ValueError as e:
        refused.append(f"shards: refused: {e}")
        readers = [gguf.GGUFReader(first_shard)]
    arch_field = readers[0].get_field(gguf.Keys.General.ARCHITECTURE)
    arch = arch_field.contents() if arch_field is not None else None
    if (fixture := FIXTURES.get(arch)) is None:
        refused.append(f"fixture: refused: no fixture for general.architecture {arch!r}")
    else:
        run = subprocess.run([str(llama.tool("verify-publish")), str(first_shard), str(fixture)], capture_output=True,
                             text=True)
        passed, llama_refused = _report(run)
        lines += passed
        refused += llama_refused
    if refused:
        raise Refused("\n".join(lines + refused))
    return lines
