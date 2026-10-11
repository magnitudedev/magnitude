"""A fake engine for the comparison's tests: a `magnitude-harness` stand-in that writes contract
outputs (result records, rendered fixtures, logits, tuning traces) from synthetic numbers, and
the true logits a reference would produce for the same rendered input.

Behaviour per build is read from `behaviour.json` next to the fake harness:
`{"speed": tokens/s scale, "noise": logits noise scale, "shift_tokens": bool, "drafting": x}`.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

VOCABULARY = 64
MODEL = "fake:gguf:q4"
TARGET_SHA = "0" * 64
TEMPLATE_SHA = "1" * 64
SCORED = 12

HARNESS = """#!{python}
import sys
sys.path.insert(0, {tests!r})
import compare_fake
sys.exit(compare_fake.harness(sys.argv[0], sys.argv[1:]))
"""


def install(target: Path, behaviour: dict) -> None:
    """A fake harness at `<target>/release/magnitude-harness` behaving as `behaviour`."""
    release = target / "release"
    release.mkdir(parents=True)
    script = release / "magnitude-harness"
    script.write_text(HARNESS.format(python=sys.executable, tests=str(Path(__file__).parent)))
    script.chmod(0o755)
    (release / "behaviour.json").write_text(json.dumps(behaviour))


def _options(argv: list[str]) -> dict[str, str]:
    options, index = {}, 0
    while index < len(argv):
        name = argv[index][2:]
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            options[name] = argv[index + 1]
            index += 2
        else:
            options[name] = "true"
            index += 1
    return options


def rendered_lines(fixtures: Path, context: int, shift: bool) -> list[dict]:
    lines = []
    for number, line in enumerate(fixtures.read_text().splitlines()):
        fixture_id = json.loads(line)["id"]
        length = 40 + number
        tokens = [
            (index * 7 + number + (1 if shift and index == 5 else 0)) % VOCABULARY
            for index in range(length)
        ]
        lines.append(
            {
                "fixture_id": fixture_id,
                "fixture_set_digest": hashlib.sha256(fixtures.read_bytes()).hexdigest(),
                "model": MODEL,
                "target_sha256": TARGET_SHA,
                "template_sha256": TEMPLATE_SHA,
                "context_target": context,
                "tokens": tokens,
                "scored_spans": [[length - SCORED, length]],
                "generation_prefix_tokens": None,
            }
        )
    return lines


def rows_of(lines: list[dict]) -> list[dict]:
    return [
        {"fixture_id": line["fixture_id"], "position": position}
        for line in lines
        for start, end in line["scored_spans"]
        for position in range(start - 1, end - 1)
    ]


def truth(rows: list[dict]) -> np.ndarray:
    """The true logits of rows, deterministic per (fixture, position)."""
    values = np.empty((len(rows), VOCABULARY), dtype="<f4")
    for index, row in enumerate(rows):
        seed = int(
            hashlib.sha256(f"{row['fixture_id']}/{row['position']}".encode()).hexdigest()[:8], 16
        )
        values[index] = np.random.default_rng(seed).normal(0, 3, VOCABULARY)
    return values


def write_logits(
    directory: Path, producer: str, rendered_digest: str, rows: list[dict], values: np.ndarray
) -> None:
    np.ascontiguousarray(values, dtype="<f4").tofile(directory / "logits.f32")
    header = {
        "producer": producer,
        "model": MODEL,
        "rendered_digest": rendered_digest,
        "vocabulary": VOCABULARY,
        "rows": rows,
    }
    (directory / "logits.json").write_text(json.dumps(header))


def _tuning(directory: Path, options: dict) -> dict:
    fresh = "fresh-tuning" in options
    budget = float(options["tuning-budget"])
    events = [
        {"event": "began", "schema": "tuning-trace/3", "budget_seconds": budget, "fresh": fresh},
        # Only a load that searched reports what became of its budget.
        *(
            [{"event": "budget", "used_seconds": 1.8, "overrun_seconds": 0.0, "refused": []}]
            if fresh
            else []
        ),
        {
            "event": "concluded",
            "units": 3,
            "searched": 3 if fresh else 0,
            "stored": 3,
            "tuning_seconds": 1.5,
            "seconds": 2.0,
        },
    ]
    (directory / "tuning.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    load = {
        "budget_seconds": budget,
        "fresh": fresh,
        **({"concluded_at_seconds": 1.8, "overrun_seconds": 0.0, "refused": 0} if fresh else {}),
        "units": 3,
        "searched": 3 if fresh else 0,
        "stored": 3,
        "tuning_seconds": 1.5,
        "seconds": 2.0,
    }
    return {"budget_seconds": budget, "fresh": fresh, "outcome": {"loads": [load]}}


def _speed_cell(phase: str, context: int, concurrency: int, scale: float, call: int) -> dict:
    rate = (2000.0 if phase == "prefill" else 50.0) * concurrency * scale * (1 + 0.002 * (call % 3))
    samples = [rate * (1 + 0.001 * offset) for offset in (-2, -1, 0, 1, 2)]
    attention = 10.0 / scale
    return {
        "phase": phase,
        "context": context,
        "concurrency": concurrency,
        "tokens_per_second": rate,
        "samples": samples,
        "detail": {
            "attribution": {
                "steps": 4,
                "entries": [
                    {"entry": "attention", "device_ms_per_step": attention},
                    {"entry": "matmul#1", "device_ms_per_step": 5.0},
                    {"entry": "matmul#2", "device_ms_per_step": 5.0},
                ],
            }
        },
    }


def harness(program: str, argv: list[str]) -> int:
    behaviour = json.loads((Path(program).parent / "behaviour.json").read_text())
    command, options = argv[0], _options(argv[1:])
    directory = Path(options["output"])
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = Path(options["fixtures"])
    build = json.loads(Path(options["build-identity"]).read_text())
    calls = directory.parent / f".calls-{Path(program).parent.parent.name}"
    call = int(calls.read_text()) if calls.exists() else 0
    calls.write_text(str(call + 1))
    measurements = {} if command == "render" else {"tuning": _tuning(directory, options)}
    if command == "time":
        contexts = [int(value) for value in options["contexts"].split(",")]
        lines = [
            line
            for context in contexts
            for line in rendered_lines(fixtures, context, behaviour["shift_tokens"])
        ]
        measurements["speed"] = [
            _speed_cell(phase, context, int(concurrency), behaviour["speed"], call)
            for context in contexts
            for concurrency in options["concurrency"].split(",")
            for phase in ("prefill", "decode")
        ]
        measurements["drafting"] = {
            "proposed": 100,
            "accepted": 60,
            "tokens_per_verify": behaviour["drafting"],
        }
    elif command == "logits" and "rendered" in options:  # another build's exact tokens
        data = Path(options["rendered"]).read_bytes()
        lines = [json.loads(line) for line in data.decode().splitlines()]
    elif command in ("logits", "render"):
        lines = rendered_lines(fixtures, int(options["context"]), behaviour["shift_tokens"])
    else:
        raise SystemExit(f"fake harness: unsupported {command}")
    data = "".join(json.dumps(line) + "\n" for line in lines).encode()
    (directory / "rendered.jsonl").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if command == "logits":
        rows = rows_of(lines)
        noise = np.random.default_rng(7).normal(0, behaviour["noise"], (len(rows), VOCABULARY))
        write_logits(directory, "harness", digest, rows, truth(rows) + noise)
    record = {
        "schema": "evaluation-result/1",
        "tool": f"harness.{command}",
        "started_at": "",
        "finished_at": "",
        "build": build,
        "model": MODEL,
        "artifacts": {},
        "fixture_set": {"path": str(fixtures), "digest": ""},
        "rendered_digest": digest,
        "host": {},
        "command": [program, *argv],
        "measurements": measurements,
        "traces": {} if command == "render" else {"tuning": "tuning.jsonl"},
    }
    (directory / "result.json").write_text(json.dumps(record))
    return 0
