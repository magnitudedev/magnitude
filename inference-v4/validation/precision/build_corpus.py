#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build the fixed D4 precision corpus: prose, code and tool JSON text files.

uv run inference-v4/validation/precision/build_corpus.py

Writes validation/results/precision/corpus/{prose,code,tool_json}.txt and
manifest.json (sha256 of every output and input). Everything is deterministic:
prose is the pinned Session Bench Moby Dick text read from the local cache,
code is a fixed file list read from a pinned git commit (read-only `git show`),
and tool JSON comes from a seeded generator. No network access.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess

ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[2]
OUTPUT = ROOT.parent / "results" / "precision" / "corpus"

# Session Bench pin: inference-v3/src/benchmark_fixtures/data/moby-dick.lock.json.
MOBY_LOCK = REPOSITORY / "inference-v3/src/benchmark_fixtures/data/moby-dick.lock.json"
PROSE_ANCHOR = "Call me Ishmael."
PROSE_CHARACTERS = 240_000

CODE_COMMIT = "1fb31d00c548b2da9b5c496ffc7f7df6155c7dda"
CODE_FILES = (
    "inference-v4/engine/artifacts/src/gguf.rs",
    "inference-v3/src/engine/weights/formats/gguf.py",
    "inference-v4/engine/chat/src/wire.rs",
    "inference-v3/src/engine/generation/constraints.py",
    "inference-v4/engine/generation/src/round.rs",
    "inference-v3/src/session_bench/runner.py",
    "inference-v4/engine/model-batching/src/rows.rs",
    "inference-v3/src/ops/formula.py",
    "inference-v4/engine/model-contracts/src/lib.rs",
)

TOOL_JSON_SEED = 20260923
TOOL_JSON_CHARACTERS = 240_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def gutenberg_body(content: bytes) -> str:
    """Same normalization as benchmark_fixtures.prose.normalize (gutenberg-body-lf-v1)."""
    text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"^\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.M)
    end = re.search(r"^\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*?\*\*\*\s*$", text, re.M)
    if start is None or end is None or start.end() >= end.start():
        raise ValueError("missing or unordered Gutenberg body markers")
    return text[start.end() : end.start()].strip()


def prose() -> tuple[str, dict]:
    lock = json.loads(MOBY_LOCK.read_text())
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "magnitude/benchmarks"
    path = cache / "sources" / lock["sha256"] / "moby-dick.txt"
    content = path.read_bytes()
    if sha256(content) != lock["sha256"]:
        raise ValueError(f"{path} does not match the Session Bench pin {lock['sha256']}")
    body = gutenberg_body(content)
    start = body.index(PROSE_ANCHOR)
    text = body[start : start + PROSE_CHARACTERS]
    return text, {"source": lock, "anchor": PROSE_ANCHOR, "characters": PROSE_CHARACTERS}


def code() -> tuple[str, dict]:
    parts = []
    files = []
    for path in CODE_FILES:
        blob = subprocess.run(
            ["git", "-C", str(REPOSITORY), "show", f"{CODE_COMMIT}:{path}"],
            check=True, capture_output=True,
        ).stdout
        files.append({"path": path, "sha256": sha256(blob)})
        comment = "//" if path.endswith(".rs") else "#"
        parts.append(f"{comment} file: {path}\n{blob.decode()}")
    return "\n".join(parts), {"commit": CODE_COMMIT, "files": files}


# ---------------------------------------------------------------------------
# Tool JSON: OpenAI-style chat completion requests with tool calls and results.

WORDS = (
    "cache", "buffer", "request", "session", "token", "model", "kernel", "batch", "layer", "queue",
    "worker", "config", "schema", "stream", "index", "record", "branch", "commit", "result", "report",
    "metric", "latency", "error", "handler", "client", "server", "route", "tensor", "weight", "shard",
)
DIRECTORIES = ("src", "src/engine", "src/api", "tests", "crates/core/src", "packages/client/src", "scripts", "docs")
EXTENSIONS = (".rs", ".py", ".ts", ".json", ".toml", ".md")
CITIES = ("Oslo", "Lisbon", "Denver", "Osaka", "Nairobi", "Toronto", "Santiago", "Hanoi", "Zurich", "Perth")
LANGUAGES = ("rust", "python", "typescript", "go")
USER_TASKS = (
    "Find where {name} is defined and explain what it does.",
    "The {name} tests are failing on CI. Can you figure out why?",
    "Rename {name} to {other} across the repository.",
    "What's the weather in {city} tomorrow? I need to decide whether to fly.",
    "How many {table} rows were created last week, grouped by day?",
    "Open an issue about the {name} regression in the {other} path.",
    "Search the web for the latest release notes of {project}.",
    "Add a unit test for {name} covering the empty input case.",
)
PROJECTS = ("llama.cpp", "tokio", "numpy", "react", "postgres", "kubernetes", "pytorch", "serde")
TABLES = ("orders", "sessions", "events", "invoices", "users", "deployments")

TOOLS = (
    {"name": "read_file", "description": "Read a file from the workspace.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "Path relative to the workspace root."},
         "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}},
    {"name": "grep", "description": "Search file contents with a regular expression.",
     "parameters": {"type": "object", "properties": {
         "pattern": {"type": "string"}, "path": {"type": "string"},
         "case_sensitive": {"type": "boolean"}}, "required": ["pattern"]}},
    {"name": "run_shell", "description": "Run a shell command and return stdout, stderr and the exit code.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string"}, "timeout_ms": {"type": "integer"}}, "required": ["command"]}},
    {"name": "edit_file", "description": "Replace an exact string in a file.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string"}, "old_string": {"type": "string"},
         "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}},
    {"name": "get_weather", "description": "Get the forecast for a city.",
     "parameters": {"type": "object", "properties": {
         "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
         "days": {"type": "integer"}}, "required": ["city"]}},
    {"name": "query_database", "description": "Run a read-only SQL query.",
     "parameters": {"type": "object", "properties": {
         "sql": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["sql"]}},
    {"name": "create_issue", "description": "Create an issue in the tracker.",
     "parameters": {"type": "object", "properties": {
         "title": {"type": "string"}, "body": {"type": "string"},
         "labels": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "body"]}},
    {"name": "web_search", "description": "Search the web.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string"}, "max_results": {"type": "integer"}}, "required": ["query"]}},
)


def identifier(rng: random.Random) -> str:
    return "_".join(rng.sample(WORDS, rng.randint(2, 3)))


def path(rng: random.Random) -> str:
    return f"{rng.choice(DIRECTORIES)}/{identifier(rng)}{rng.choice(EXTENSIONS)}"


def call_id(rng: random.Random) -> str:
    return "call_" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789") for _ in range(24))


def source_line(rng: random.Random) -> str:
    name, other = identifier(rng), identifier(rng)
    return rng.choice((
        f"fn {name}(&self, {other}: usize) -> Result<(), Error> {{",
        f"    let {name} = self.{other}.get(index).ok_or(Error::Missing)?;",
        f"def {name}(self, {other}: int) -> None:",
        f"        return self.{other}[{rng.randint(0, 64)}]",
        f"export function {name}({other}: string): number {{",
        f"    assert_eq!({name}.len(), {rng.randint(1, 4096)});",
        "}",
    ))


def arguments(rng: random.Random, tool: str) -> dict:
    if tool == "read_file":
        start = rng.randint(1, 400)
        return {"path": path(rng), "start_line": start, "end_line": start + rng.randint(10, 80)}
    if tool == "grep":
        return {"pattern": rf"fn {identifier(rng)}\(", "path": rng.choice(DIRECTORIES), "case_sensitive": rng.random() < 0.5}
    if tool == "run_shell":
        return {"command": rng.choice((
            f"cargo test -p {rng.choice(WORDS)} -- {identifier(rng)}",
            f"python -m pytest tests/test_{identifier(rng)}.py -x -q",
            f"git log --oneline -n {rng.randint(3, 20)} -- {path(rng)}",
            f"ls -la {rng.choice(DIRECTORIES)}",
        )), "timeout_ms": rng.choice((30000, 60000, 120000))}
    if tool == "edit_file":
        return {"path": path(rng), "old_string": source_line(rng), "new_string": source_line(rng)}
    if tool == "get_weather":
        return {"city": rng.choice(CITIES), "unit": rng.choice(("celsius", "fahrenheit")), "days": rng.randint(1, 7)}
    if tool == "query_database":
        table = rng.choice(TABLES)
        return {"sql": f"SELECT date_trunc('day', created_at) AS day, count(*) FROM {table} "
                       f"WHERE created_at >= now() - interval '{rng.randint(2, 30)} days' GROUP BY 1 ORDER BY 1",
                "limit": rng.choice((50, 100, 500))}
    if tool == "create_issue":
        return {"title": f"{identifier(rng).replace('_', ' ').capitalize()} regression after {rng.randint(1, 9)}.{rng.randint(0, 30)}",
                "body": " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 40))) + ".",
                "labels": rng.sample(("bug", "performance", "p1", "p2", "regression", "needs-triage"), 2)}
    return {"query": f"{rng.choice(PROJECTS)} {rng.choice(WORDS)} {rng.choice(WORDS)} release notes", "max_results": rng.randint(3, 10)}


def result(rng: random.Random, tool: str, args: dict) -> dict:
    if tool == "read_file":
        lines = [source_line(rng) for _ in range(rng.randint(6, 18))]
        return {"path": args["path"], "start_line": args["start_line"], "content": "\n".join(lines), "truncated": False}
    if tool == "grep":
        return {"matches": [{"path": path(rng), "line": rng.randint(1, 900), "text": source_line(rng)}
                            for _ in range(rng.randint(0, 6))]}
    if tool == "run_shell":
        code = rng.choice((0, 0, 0, 1, 101))
        stdout = "\n".join(f"test {identifier(rng)} ... {'ok' if code == 0 or rng.random() < 0.7 else 'FAILED'}"
                           for _ in range(rng.randint(2, 10)))
        stderr = "" if code == 0 else f"error[E0{rng.randint(100, 799)}]: mismatched types in `{identifier(rng)}`"
        return {"exit_code": code, "stdout": stdout, "stderr": stderr, "duration_ms": rng.randint(40, 90000)}
    if tool == "edit_file":
        return {"ok": True, "path": args["path"], "replacements": 1}
    if tool == "get_weather":
        return {"city": args["city"], "forecast": [
            {"day": day, "high": round(rng.uniform(-5, 35), 1), "low": round(rng.uniform(-15, 20), 1),
             "precipitation_probability": round(rng.random(), 2),
             "conditions": rng.choice(("clear", "cloudy", "rain", "snow", "thunderstorms", "fog"))}
            for day in range(args["days"])]}
    if tool == "query_database":
        return {"columns": ["day", "count"], "rows": [[f"2026-09-{day:02d}", rng.randint(0, 50000)]
                                                     for day in range(1, rng.randint(3, 12))]}
    if tool == "create_issue":
        number = rng.randint(100, 9999)
        return {"number": number, "url": f"https://tracker.example.com/issues/{number}", "state": "open"}
    return {"results": [{"title": f"{rng.choice(PROJECTS)}: {identifier(rng).replace('_', ' ')}",
                         "url": f"https://example.org/{identifier(rng)}", "snippet": " ".join(rng.choice(WORDS) for _ in range(14))}
                        for _ in range(args["max_results"])]}


def conversation(rng: random.Random) -> dict:
    tools = rng.sample(TOOLS, rng.randint(2, 5))
    task = rng.choice(USER_TASKS).format(name=identifier(rng), other=identifier(rng), city=rng.choice(CITIES),
                                         table=rng.choice(TABLES), project=rng.choice(PROJECTS))
    messages = [{"role": "system", "content": "You are a helpful assistant with access to tools."},
                {"role": "user", "content": task}]
    for _ in range(rng.randint(1, 3)):
        calls = []
        for tool in rng.sample(tools, rng.randint(1, min(2, len(tools)))):
            calls.append((call_id(rng), tool["name"], arguments(rng, tool["name"])))
        messages.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": ident, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for ident, name, args in calls]})
        for ident, name, args in calls:
            messages.append({"role": "tool", "tool_call_id": ident, "content": json.dumps(result(rng, name, args))})
    messages.append({"role": "assistant", "content": " ".join(rng.choice(WORDS) for _ in range(rng.randint(10, 40))) + "."})
    return {"model": "qwen3.5-4b", "temperature": round(rng.choice((0.0, 0.2, 0.7, 1.0)), 1),
            "tools": [{"type": "function", "function": tool} for tool in tools], "messages": messages}


def tool_json() -> tuple[str, dict]:
    rng = random.Random(TOOL_JSON_SEED)
    documents = []
    size = 0
    while size < TOOL_JSON_CHARACTERS:
        document = conversation(rng)
        text = json.dumps(document, indent=2) if rng.random() < 0.5 else json.dumps(document)
        documents.append(text)
        size += len(text) + 1
    return "\n".join(documents), {"seed": TOOL_JSON_SEED, "characters": TOOL_JSON_CHARACTERS, "documents": len(documents)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    options = parser.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    manifest = {"generator_sha256": sha256(Path(__file__).read_bytes()), "categories": {}}
    for name, build in (("prose", prose), ("code", code), ("tool_json", tool_json)):
        text, provenance = build()
        data = text.encode()
        (options.output / f"{name}.txt").write_bytes(data)
        manifest["categories"][name] = {"file": f"{name}.txt", "bytes": len(data), "sha256": sha256(data), **provenance}
        print(f"{name}: {len(data)} bytes sha256={sha256(data)}")
    (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
