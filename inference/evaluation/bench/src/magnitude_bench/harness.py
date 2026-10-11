"""Host-only rendering of the catalog GGUF, through ``magnitude-harness render``.

The harness renders fixtures exactly as our engine serves them (the GGUF's chat template and
tokenizer, through the serving chat adapter) without loading the model, and writes the
contract's rendered-fixture lines. It sizes the run and counts the prompts of every GGUF
engine (ours, llama.cpp, Ollama's GGUF runner); each engine's terminal counts stay
authoritative for what it actually evaluated.

Each chat body is rendered as a one-conversation ``fixed`` fixture, so its token count is
the served length; a body that ends at a user turn is rendered with the generation prompt.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from magnitude_fixtures import Fixture

from .adapters.base import command
from .requests import Plan

RENDERED = "rendered"


class HarnessRenderer:
    def __init__(
        self, binary: Path, artifacts: Path, build: Path, fixture: Fixture, logs: Path
    ):
        #: Family and source every rendered body is attributed to (the run's fixture set).
        self.binary, self.artifacts, self.build = binary, artifacts, build
        self.family, self.source = fixture.family, fixture.source.model_dump()
        self.logs = logs
        self.lock = asyncio.Lock()

    def fixtures(self, bodies: list[dict]) -> str:
        lines = []
        for index, body in enumerate(bodies):
            fixture = {
                "id": f"body-{index}",
                "family": self.family,
                "source": self.source,
                **({"tools": body["tools"]} if body.get("tools") else {}),
                "messages": body["messages"],
                "sizing": {"unit": "fixed"},
            }
            lines.append(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines) + "\n"

    async def render(self, bodies: list[dict], output: Path) -> list[dict]:
        """Rendered-fixture lines of ``bodies``, in order, written under ``output``."""
        output.mkdir(parents=True, exist_ok=True)
        fixtures = output / "fixtures.jsonl"
        fixtures.write_text(self.fixtures(bodies))
        async with self.lock:
            await command(
                [
                    str(self.binary), "render",
                    "--artifacts", str(self.artifacts),
                    "--fixtures", str(fixtures),
                    "--output", str(output),
                    "--build-identity", str(self.build),
                ],
                output,
                self.logs / "harness-render.log",
            )
        lines = [json.loads(line) for line in (output / "rendered.jsonl").read_text().splitlines()]
        if [line["fixture_id"] for line in lines] != [f"body-{i}" for i in range(len(bodies))]:
            raise ValueError("magnitude-harness rendered a different set of conversations")
        return lines

    async def lengths(self, bodies: list[dict]) -> list[int]:
        with tempfile.TemporaryDirectory(prefix="render-", dir=self.logs) as work:
            return [len(line["tokens"]) for line in await self.render(bodies, Path(work))]

    async def prompt_counts(self, plan: Plan, run: Path) -> dict[str, int]:
        """Every served prompt rendered into the run directory: their tokens and template
        identity are the run's rendered evidence (``rendered/rendered.jsonl``)."""
        requests = plan.prepared_requests
        lines = await self.render(
            [request.body("magnitude-bench") for request in requests], run / RENDERED
        )
        return {
            request.id: len(line["tokens"])
            for request, line in zip(requests, lines, strict=True)
        }


def rendered_digest(run: Path) -> str | None:
    path = run / RENDERED / "rendered.jsonl"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
