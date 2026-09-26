"""Native llama.cpp reference with the same fixed token workload, independent timers."""

import asyncio
import hashlib
import platform
import shlex
import subprocess
from pathlib import Path

from pydantic import JsonValue

from roofline.contracts import Measurement, digest, identity, now
from roofline.store import atomic_write

from .magnitude import verify_artifact


class Llama:
    def __init__(self, request, attempt, store):
        if "/" in request.experiment.scope:
            raise NotImplementedError(
                "llama.cpp exposes enclosing prefill/decode, not Magnitude formula scopes"
            )
        from benchmark_fixtures.preparation import Fixture, Tokenization, prepare

        self.request, self.attempt, self.store = request, attempt, store
        self.experiment = request.experiment
        assert request.model is not None
        path = verify_artifact(request.model, attempt.target_name, request.experiment.model)
        self.tokens = asyncio.run(
            prepare(
                Fixture(
                    identity="prose.moby-dick",
                    context_tokens=self.experiment.context,
                    continuation_tokens=self.experiment.steps,
                ),
                Tokenization(path),
            )
        )
        flags = subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "llama", "ggml"], text=True
        )
        source = Path(__file__).with_name("llama_driver.cpp")
        version = subprocess.check_output(
            ["pkg-config", "--modversion", "llama"], text=True
        ).strip()
        key = digest(
            {
                "driver": hashlib.sha256(source.read_bytes()).hexdigest(),
                "flags": flags,
                "version": version,
            }
        )
        directory = store.root / "reference" / key
        directory.mkdir(parents=True, exist_ok=True)
        binary = directory / "driver"
        if not binary.exists():
            subprocess.run(
                ["c++", "-std=c++17", "-O2", str(source), "-o", str(binary), *shlex.split(flags)],
                check=True,
                timeout=120,
            )
        fixture = directory / "tokens.txt"
        atomic_write(
            fixture,
            (
                f"{len(self.tokens.prompt)} {len(self.tokens.continuation)}\n"
                + " ".join(map(str, (*self.tokens.prompt, *self.tokens.continuation)))
            ).encode(),
        )
        self.process = subprocess.Popen(
            [
                str(binary),
                str(path),
                str(self.experiment.context + self.experiment.steps),
                attempt.target.device.backend,
                str(attempt.target.device.index),
                str(fixture),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.process.stdout is not None and self.process.stdin is not None
        ready = self.process.stdout.readline().strip()
        if not ready.startswith("READY "):
            self.close()
            raise RuntimeError("llama.cpp did not initialize; see executor log")
        self.hardware = {
            "host": platform.node(),
            "system": platform.platform(),
            "device": ready[6:],
        }
        self.implementation: dict[str, JsonValue] = {"library_version": version, "driver": key}

    def refresh(self):
        # Magnitude authored-kernel changes do not alter this external implementation.
        if self.process.poll() is not None:
            raise RuntimeError("reference executor is no longer alive")

    def sample(self):
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(self.experiment.scope + "\n")
        self.process.stdin.flush()
        value = self.process.stdout.readline().strip()
        if not value:
            raise RuntimeError("llama.cpp reference execution failed")
        return float(value)

    def measure(self):
        for _ in range(self.experiment.protocol.warmups):
            self.sample()
        samples = tuple(self.sample() for _ in range(self.experiment.protocol.samples))
        return [
            Measurement(
                measurement_id=identity(),
                request_id=self.request.request_id,
                attempt_id=self.attempt.attempt_id,
                created=now(),
                model=self.request.experiment.model,
                artifact=self.request.model.sha256,
                source_id=self.experiment.source,
                target=self.attempt.target_name,
                engine="llama.cpp",
                scope=self.experiment.scope,
                workload={
                    "workload": "prose",
                    "context": self.experiment.context,
                    "steps": self.experiment.steps,
                    "tokens": digest(
                        {"prompt": self.tokens.prompt, "continuation": self.tokens.continuation}
                    ),
                    "provenance": self.tokens.provenance,
                },
                hardware=self.hardware,
                protocol={
                    **self.experiment.protocol.model_dump(exclude={"deadline_seconds"}),
                    "boundary": "llama_decode-through-synchronize",
                },
                status="complete",
                correctness="unchecked",
                samples_seconds=samples,
                details={
                    "implementation": self.implementation,
                    "sampling": "no token selection in timer",
                },
                unavailable=(
                    "Independent numerical checking unavailable for this reference integration",
                    "Opaque reference execution has no verified Magnitude formula correspondence",
                    "Native decode timing excludes token selection; Magnitude boundaries differ",
                ),
            )
        ]

    def close(self):
        if getattr(self, "process", None):
            assert self.process.stdin is not None
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
