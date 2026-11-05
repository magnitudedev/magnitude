"""The host side of `mrig served`: drive the real service as a client does, on an installation
of the checkout's binary, from an empty kernel cache. Standard library only."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

RESULTS = "served.json"
SCENARIOS = ("lifecycle", "stop", "failure", "repeat")
# Characters of a fixture's opening a prompt keeps: within any context an evaluated load serves.
PROMPT_CHARACTERS = 9000
ANSWER_TOKENS = 64
# Answers a `repeat` run asks past the first of each prompt.
REPEATS = 16
LOAD_SECONDS = 1500.0
JOB_SECONDS = 180.0
START_SECONDS = 120.0
# The read a `failure` run fails: this much of the model file, from its middle, with EIO once the
# trigger file exists. Small enough that a request can route around it and large enough that one
# of a pair of answers names an expert in it.
FAILED_BYTES = 8 << 20
SHIM = r"""
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
ssize_t pread64(int fd, void *buffer, size_t count, off64_t offset) {
  static ssize_t (*real)(int, void *, size_t, off64_t);
  static char trigger[4096];
  static long long first, last;
  if (!real) {
    real = dlsym(RTLD_NEXT, "pread64");
    sscanf(getenv("MAGNITUDE_FAILED_READS"), "%lld,%lld,%4095s", &first, &last, trigger);
  }
  if (offset < last && offset + (off64_t)count > first && access(trigger, F_OK) == 0) {
    errno = EIO;
    return -1;
  }
  return real(fd, buffer, count, offset);
}
"""


class Failed(Exception):
    """A stage did not do what it must."""


class Log:
    """The service's log as it grows, read from where the last wait left off."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.mark()

    def mark(self) -> None:
        self.offset = self.path.stat().st_size if self.path.exists() else 0

    def text(self) -> str:
        with self.path.open(errors="replace") as file:
            file.seek(self.offset)
            return file.read()

    def wait(self, pattern: str, seconds: float, process: subprocess.Popen) -> str:
        """The first line matching `pattern` past the mark; the mark moves past it."""
        end = time.monotonic() + seconds
        expression = re.compile(pattern)
        while time.monotonic() < end:
            with self.path.open(errors="replace") as file:
                file.seek(self.offset)
                while line := file.readline():
                    self.offset = file.tell()
                    if expression.search(line):
                        return line.rstrip()
            if process.poll() is not None:
                raise Failed(f"the service exited {process.returncode} waiting for {pattern!r}")
            time.sleep(0.05)
        raise Failed(f"no {pattern!r} in {seconds:.0f} s")


class Service:
    """One `serve` process, ended by closing its owner's stdin."""

    def __init__(self, run: "Run", name: str, bounds: tuple[str, ...], environment: dict[str, str] | None = None) -> None:
        self.log_path = run.directory / f"service-{name}.log"
        self.output = self.log_path.open("w")
        self.process = subprocess.Popen(
            [str(run.installed), "serve", "--bind", "127.0.0.1:0", "--exit-on-stdin-eof", "--model-store", str(run.models),
             "--cache-root", str(run.cache), "--installation", str(run.installation), *bounds],
            stdin=subprocess.PIPE, stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, "RUST_LOG": "info", **(environment or {})},
        )
        self.log = Log(self.log_path)
        ready = self.log.wait("MAGNITUDE_ICN_READY", START_SECONDS, self.process)
        self.origin = json.loads(ready.split("MAGNITUDE_ICN_READY ", 1)[1])["origin"]
        self.log.offset = 0

    def call(self, method: str, path: str, body: object = None, seconds: float = 60.0) -> object:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.origin + path, data=data, method=method, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=seconds) as reply:
                return json.loads(reply.read() or b"null")
        except urllib.error.HTTPError as error:
            raise Failed(f"{method} {path}: {error.code} {error.read()[:600]!r}") from None

    def model(self, wanted: str) -> str:
        """`wanted` installed through the catalog, as the app installs it: what the host holds
        of it is found, and what it lacks fetched."""
        path = f"/api/v1/catalog/models/{wanted}"
        end = time.monotonic() + LOAD_SECONDS
        installing = False
        while '"_tag": "Installed"' not in (state := json.dumps(self.call("GET", path))):
            if not installing:
                self.call("POST", f"{path}/install")
                installing = True
            if time.monotonic() > end:
                raise Failed(f"{wanted} never installed: {state[:400]}")
            time.sleep(1)
        return wanted

    def load(self, model: str) -> dict:
        return self.call("POST", "/api/v1/instances", {"modelId": model}, LOAD_SECONDS)

    def instances(self) -> dict:
        return self.call("GET", "/api/v1/instances")

    def stop(self, instance: dict) -> object:
        return self.call("POST", f"/api/v1/instances/{instance['id']}/stop", {})

    def chat(self, model: str, messages: list[dict]) -> dict:
        """One answer: its text, how it ended, and its first token's time and each later one's."""
        started = time.monotonic()
        request = urllib.request.Request(self.origin + "/v1/chat/completions", method="POST", headers={"content-type": "application/json"}, data=json.dumps({
            "model": model, "messages": messages, "max_tokens": ANSWER_TOKENS, "temperature": 0, "seed": 1,
            "stream": True, "stream_options": {"include_usage": True},
        }).encode())
        pieces, times, finish, usage = [], [], None, None
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                for line in reply:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    chunk = json.loads(line[6:])
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", ()):
                        finish = choice.get("finish_reason") or finish
                        delta = choice.get("delta", {})
                        if content := delta.get("content") or delta.get("reasoning_content"):
                            pieces.append(content)
                            times.append(time.monotonic())
        except urllib.error.HTTPError as error:
            raise Failed(f"chat: {error.code} {error.read()[:600]!r}") from None
        tokens = (usage or {}).get("completion_tokens") or len(pieces)
        return {"seconds": round(time.monotonic() - started, 3), "first_seconds": round(times[0] - started, 3) if times else None,
                "decode_ms": round((times[-1] - times[0]) * 1000 / max(tokens - 1, 1), 2) if len(times) > 1 else None,
                "text": "".join(pieces), "pieces": pieces, "finish": finish, "usage": usage}

    def stream(self, model: str, messages: list[dict], started: threading.Event) -> dict:
        """An answer token by token, `started` set at its first; how far it got and how it ended."""
        request = urllib.request.Request(self.origin + "/v1/chat/completions", method="POST", headers={"content-type": "application/json"}, data=json.dumps({
            "model": model, "messages": messages, "max_tokens": ANSWER_TOKENS * 4, "temperature": 0, "seed": 1, "stream": True,
        }).encode())
        pieces, ended = [], None
        try:
            with urllib.request.urlopen(request, timeout=300) as reply:
                for line in reply:
                    if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                        continue
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        ended = chunk["error"]
                        break
                    for choice in chunk.get("choices", ()):
                        delta = choice.get("delta", {})
                        pieces.append(delta.get("content") or delta.get("reasoning_content") or "")
                        ended = choice.get("finish_reason") or ended
                    started.set()
        except urllib.error.HTTPError as error:
            ended = f"{error.code} {error.read()[:300]!r}"
        except OSError as error:
            ended = f"connection: {error}"
        started.set()
        return {"pieces": len(pieces), "ended": ended, "text": "".join(pieces)}

    def end(self) -> int | None:
        self.process.stdin.close()
        try:
            return self.process.wait(30)
        except subprocess.TimeoutExpired:
            self.process.send_signal(signal.SIGKILL)
            return self.process.wait(30)
        finally:
            self.output.close()


def streaming(value: object) -> object:
    """The streaming status an instance states, wherever its lifecycle holds it."""
    if isinstance(value, dict):
        if "streaming" in value:
            return value["streaming"]
        value = list(value.values())
    if isinstance(value, list):
        return next((found for item in value if (found := streaming(item)) is not None), None)
    return None


class Run:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        self.directory = Path(arguments.run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.models = Path(arguments.models).expanduser()
        self.scratch = Path(arguments.scratch) / self.directory.parent.name
        shutil.rmtree(self.scratch, ignore_errors=True)
        self.cache = self.scratch / "cache"
        self.cache.mkdir(parents=True)
        self.report: list[str] = []
        self.record: dict = {"stages": []}
        self.started = time.monotonic()
        self.capped = (*(("--context-tokens", str(arguments.context)) if arguments.context else ()),
                       *(("--weight-cap", str(arguments.weight_cap)) if arguments.weight_cap else ()))
        plain = ("--plain",) if arguments.plain else ()
        self.capped += plain
        self.resident = (*(("--context-tokens", str(arguments.context)) if arguments.context else ()), *plain)
        self.install()
        self.messages = opening(Path(arguments.fixtures), 0)
        self.other = opening(Path(arguments.fixtures), 1, 40)

    def install(self) -> None:
        """An installation as `build-local` prepares one (`installation.rs`): the binary, the
        catalog, and an empty runtime directory, the host's NVRTC found as every run finds it."""
        root = self.scratch / "installation"
        for directory in ("bin", "catalog", "runtime"):
            (root / directory).mkdir(parents=True)
        self.installed = root / "bin" / Path(self.arguments.service).name
        os.link(self.arguments.service, self.installed)
        shutil.copyfile(self.arguments.bundle, root / "catalog" / "model-planner-inputs.bundle")
        identity = json.loads(subprocess.check_output([str(self.installed), "version", "--json"]))
        self.installation = root / "installation.json"
        self.installation.write_text(json.dumps({"schemaVersion": 1, "nativeBuild": identity["native_build"]}))

    def note(self, stage: str, **facts: object) -> None:
        at = round(time.monotonic() - self.started, 1)
        self.record["stages"].append({"stage": stage, "at": at, **facts})
        shown = "  ".join(f"{name}={value}" for name, value in facts.items())
        line = f"{at:7.1f}s {stage}: {shown}"
        print(line, flush=True)
        self.report.append(line[:400])

    def finish(self) -> None:
        self.record["report"] = self.report
        (self.directory / RESULTS).write_text(json.dumps(self.record, indent=1))
        shutil.rmtree(self.scratch, ignore_errors=True)


def opening(path: Path, index: int, skip: int = 0) -> list[dict]:
    """Fixture `index`'s messages from `skip` on, its system message first, within a prompt's
    characters and ending on a user message."""
    fixture = json.loads(path.read_text().splitlines()[index])
    kept, characters = [], 0
    for message in fixture["messages"][:1] + fixture["messages"][1 + skip:]:
        characters += len(str(message["content"]))
        if characters > PROMPT_CHARACTERS:
            break
        kept.append({"role": message["role"], "content": message["content"]})
    while kept[-1]["role"] != "user":
        kept.pop()
    return kept


def priced(line: str) -> dict[str, str]:
    """What a pricing line states of each input, shortened to its state."""
    fields = dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
    return {name: fields.get(name, "?").strip('"')[:40] for name in ("lines", "resident", "pass", "moves", "tier_copy", "convert")}


def in_thread(work) -> tuple[Callable[[], None], dict]:
    """`work` on its own thread, and what joins it: its result, or the service's refusal or a
    lost connection, in the returned dict; anything else it raises is raised again by the join."""
    out: dict = {}
    defect: list[BaseException] = []

    def body() -> None:
        started = time.monotonic()
        try:
            out["result"] = work()
        except (Failed, OSError) as error:
            out["error"] = str(error)
        except BaseException as error:
            defect.append(error)
        out["seconds"] = round(time.monotonic() - started, 2)

    thread = threading.Thread(target=body)
    thread.start()

    def join() -> None:
        thread.join()
        if defect:
            raise defect[0]

    return join, out


def settled(service: Service, run: Run) -> dict[str, str]:
    """Wait for `Reprice` to bring the live load's lines."""
    while True:
        line = service.log.wait("the experts are priced", JOB_SECONDS * 3, service.process)
        state = priced(line)
        run.note("priced", **state)
        if state["lines"] == "measured":
            return state


def lifecycle(run: Run) -> None:
    wanted = run.arguments.model
    capped = Service(run, "capped", run.capped)
    model = capped.model(wanted)
    run.note("service up", origin=capped.origin)
    # A load arriving during the job.
    capped.log.wait("measurement attempting", START_SECONDS, capped.process)
    run.note("job running, load requested")
    join, loaded = in_thread(lambda: capped.load(model))
    join()
    if "error" in loaded:
        raise Failed(f"cold load: {loaded['error']}")
    run.note("load against the job", preempted="load preempts the job" in capped.log_path.read_text(errors="replace"))
    instance = loaded["result"]
    run.note("cold load ready", seconds=loaded["seconds"], streaming=json.dumps(streaming(instance))[:300])
    first = capped.chat(model, run.messages)
    run.note("no-cost answer", first=first["first_seconds"], decode_ms=first["decode_ms"], usage=first["usage"], finish=first["finish"], text=first["text"][:60])
    log = capped.log_path.read_text(errors="replace")
    (run.directory / "pricing-at-load.txt").write_text("\n".join(l for l in log.splitlines() if "experts are priced" in l or "CPU sites" in l))
    # The job beside the idle load, and a request preempting it.
    capped.log.mark()
    capped.log.wait("measurement attempting", JOB_SECONDS, capped.process)
    run.note("job beside the idle load")
    during = capped.chat(model, run.messages)
    run.note("answer during the job", first=during["first_seconds"], decode_ms=during["decode_ms"], same=during["text"] == first["text"])
    # The job resumes when idle again; `Reprice` reaches the load.
    capped.log.mark()
    settled(capped, run)
    time.sleep(1)
    third = capped.chat(model, run.messages)
    status = streaming(capped.instances())
    run.note("priced answer", first=third["first_seconds"], decode_ms=third["decode_ms"], same=third["text"] == first["text"], streaming=json.dumps(status)[:300])
    other = capped.chat(model, run.other)
    run.note("priced answer to another prompt", first=other["first_seconds"], decode_ms=other["decode_ms"])
    fourth = capped.chat(model, run.messages)
    run.note("priced answer repeated", first=fourth["first_seconds"], decode_ms=fourth["decode_ms"], same=fourth["text"] == first["text"])
    run.record["answers"] = {"no_cost": first, "repeated": fourth}
    # Stopped first: its worker reports its streaming terms as it ends.
    capped.log.mark()
    capped.stop(instance)
    capped.log.wait("model residency terminalized", 60, capped.process)
    run.note("capped service ended", code=capped.end())
    # A new service reads what the job stored.
    second = Service(run, "second", run.capped)
    started = time.monotonic()
    second.model(wanted)
    instance = second.load(model)
    loaded = time.monotonic() - started
    pricing = priced(second.log.wait("the experts are priced", 5, second.process))
    run.note("second load ready", seconds=round(loaded, 2), **pricing)
    answer = second.chat(model, run.messages)
    run.note("second load's answer", first=answer["first_seconds"], decode_ms=answer["decode_ms"], same=answer["text"] == first["text"])
    second_other = second.chat(model, run.other)
    run.note("second load's other answer", first=second_other["first_seconds"], decode_ms=second_other["decode_ms"], same=second_other["text"] == other["text"])
    run.note("second service ended", code=second.end())
    # The reference.
    resident = Service(run, "resident", run.resident)
    resident.model(wanted)
    resident.load(model)
    reference = resident.chat(model, run.messages)
    resident_other = resident.chat(model, run.other)
    run.note("resident answer", first=reference["first_seconds"], decode_ms=reference["decode_ms"], same=reference["text"] == first["text"], text=reference["text"][:60])
    run.note("resident service ended", code=resident.end())
    reference_other = resident_other
    run.record["answers"] = {"no_cost": first, "during": during, "priced": third, "second": answer, "resident": reference,
                             "repeated": fourth, "other": other, "second_other": second_other, "resident_other": reference_other}
    if not all(item["text"] == reference["text"] for item in (first, during, third, fourth, answer)):
        raise Failed("an answer differs from the resident load's")
    if not other["text"] == second_other["text"] == reference_other["text"]:
        raise Failed("an answer to another prompt differs from the resident load's")


def stop(run: Run) -> None:
    wanted = run.arguments.model
    service = Service(run, "stop", run.capped)
    model = service.model(wanted)
    # Past the job's first run, so a job starts beside the load, and stop the load under it.
    instance = service.load(model)
    run.note("load ready", streaming=json.dumps(streaming(instance))[:200])
    service.log.mark()
    service.log.wait("measurement attempting", JOB_SECONDS, service.process)
    run.note("job beside the load, stopping it")
    service.stop(instance)
    ended = service.log.wait("measurement job ended", 30, service.process)
    run.note("job ended", line=ended[-60:])
    instance = service.load(model)
    run.note("next load ready", instance=instance["id"])
    answer = service.chat(model, run.messages)
    run.note("its answer", seconds=answer["seconds"], text=answer["text"][:60])
    run.note("service ended", code=service.end())


def failure(run: Run) -> None:
    """Fail a stretch of the model file under two answers: only those that named it fail, and
    the load serves on."""
    size = Path(json.loads(Path(run.arguments.artifacts).read_text())["target"]).stat().st_size
    shim = run.scratch / "failed_reads.so"
    source = run.scratch / "failed_reads.c"
    source.write_text(SHIM)
    subprocess.check_call(["cc", "-shared", "-fPIC", "-o", str(shim), str(source), "-ldl"])
    trigger = run.scratch / "fail"
    environment = {"LD_PRELOAD": str(shim), "MAGNITUDE_FAILED_READS": f"{size // 2},{size // 2 + FAILED_BYTES},{trigger}"}
    service = Service(run, "failure", run.capped, environment)
    model = service.model(run.arguments.model)
    instance = service.load(model)
    run.note("load ready", streaming=json.dumps(streaming(instance))[:200])
    before = service.chat(model, run.messages)
    run.note("answer before", seconds=before["seconds"], text=before["text"][:60])
    gates = threading.Event(), threading.Event()
    first, one = in_thread(lambda: service.stream(model, run.messages, gates[0]))
    second, two = in_thread(lambda: service.stream(model, run.other, gates[1]))
    gates[0].wait(120)
    gates[1].wait(120)
    trigger.touch()
    run.note("reads failing", first_bytes=size // 2, last_bytes=size // 2 + FAILED_BYTES)
    first()
    second()
    trigger.unlink()
    for name, ended in (("first", one), ("second", two)):
        result = ended.get("result", ended)
        run.note(f"{name} answer ended", pieces=result.get("pieces"), ended=result.get("ended") or ended.get("error"))
    run.note("instances", state=json.dumps(service.instances())[:300])
    after = service.chat(model, run.messages)
    run.note("answer after", seconds=after["seconds"], same=after["text"] == before["text"])
    run.note("service ended", code=service.end())
    if after["text"] != before["text"]:
        raise Failed("the load's answer changed after the failed reads")


def repeat(run: Run) -> None:
    """One load answering two prompts in turn: every answer to a prompt held to its first, and
    where the first that differs parts from it."""
    service = Service(run, "repeat", run.capped)
    model = service.model(run.arguments.model)
    service.load(model)
    first = [service.chat(model, run.messages), service.chat(model, run.other)]
    run.note("first answers", decode_ms=[answer["decode_ms"] for answer in first], prefill=[answer["first_seconds"] for answer in first])
    parted = []
    for turn in range(REPEATS):
        which = turn % 2
        answer = service.chat(model, (run.messages, run.other)[which])
        same = answer["pieces"] == first[which]["pieces"]
        at = next((index for index, (a, b) in enumerate(zip(answer["pieces"], first[which]["pieces"])) if a != b), None)
        run.note("answer", prompt=which, decode_ms=answer["decode_ms"], prefill=answer["first_seconds"], same=same, parted_at=at)
        if not same:
            parted.append({"turn": turn, "at": at, "answer": answer["pieces"], "first": first[which]["pieces"]})
    run.record["parted"] = parted
    run.note("service ended", code=service.end())
    if parted:
        raise Failed(f"{len(parted)} of {REPEATS} repeats parted from their first answer")


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("--service", "--bundle", "--models", "--artifacts", "--fixtures", "--scratch", "--run-dir", "--model"):
        parser.add_argument(name, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--context", type=int)
    parser.add_argument("--weight-cap", type=int)
    parser.add_argument("--plain", action="store_true")
    run = Run(parser.parse_args())
    code = 0
    try:
        {"lifecycle": lifecycle, "stop": stop, "failure": failure, "repeat": repeat}[run.arguments.scenario](run)
    except Failed as why:
        run.note("FAILED", why=str(why))
        code = 1
    finally:
        run.finish()
    return code


if __name__ == "__main__":
    sys.exit(main())
