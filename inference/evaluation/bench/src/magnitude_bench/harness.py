"""Host-only rendering of the catalog GGUF, through ``magnitude-harness render``.

The harness renders template files exactly as our engine serves them (the GGUF's chat template
and tokenizer, through the serving chat adapter) without loading the model, sizing every lane
to its template's context, and writes the contract's rendered lines. It is the one sizer of a
run that measures any GGUF engine (ours, llama.cpp, Ollama's GGUF runner), and counts their
prompts; each engine's terminal counts stay authoritative for what it actually evaluated.

Each lane's sized conversation is its session's ``conversation_with(history)``, ``history``
being the units the harness kept; its prompt count is the rendered line's
``generation_prefix_tokens``.
"""

import hashlib
import json
from pathlib import Path

from magnitude_fixtures import TemplateFile

from .adapters.base import command
from .plan import Sized

RENDERED = "rendered"


class HarnessRenderer:
    def __init__(self, binary: Path, artifacts: Path, build: Path, logs: Path):
        self.binary, self.artifacts, self.build, self.logs = binary, artifacts, build, logs

    async def render(self, files: list[tuple[TemplateFile, Path]], run: Path) -> dict[tuple[str, int], dict]:
        """Every lane of ``files`` (each read from its path) rendered into the run directory
        (``rendered/rendered.jsonl``, the run's rendered evidence), by (template, lane)."""
        output = run / RENDERED
        output.mkdir(parents=True, exist_ok=True)
        await command(
            [
                str(self.binary), "render",
                "--artifacts", str(self.artifacts),
                *(argument for _, path in files for argument in ("--template", str(path))),
                "--output", str(output),
                "--build-identity", str(self.build),
            ],
            output,
            self.logs / "harness-render.log",
        )
        lines = [json.loads(line) for line in (output / "rendered.jsonl").read_text().splitlines()]
        expected = [(file.template, lane) for file, _ in files for lane in range(len(file.sessions))]
        if [(line["template"], line["lane"]) for line in lines] != expected:
            raise ValueError("magnitude-harness rendered other lanes than the templates hold")
        return {(line["template"], line["lane"]): line for line in lines}

    @staticmethod
    def sized(file: TemplateFile, lane: int, line: dict) -> Sized:
        """Lane ``lane`` of ``file`` as the harness sized it. A lane whose whole history fits
        is refused rather than served short."""
        session = file.sessions[lane]
        if line["history"] == len(session.units()) - 1:
            raise ValueError(f"{file.template} lane {lane} holds too little history to fill "
                             f"{file.context} tokens")
        return Sized(session.conversation_with(line["history"]), line["prompt_sha256"])


def rendered_digest(run: Path) -> str | None:
    path = run / RENDERED / "rendered.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
