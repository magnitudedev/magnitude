from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import AdapterConfiguration, detected_backend, validate_configuration


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _verify_locked_snapshot(path: Path, manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text())
        files = manifest["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid MLX artifact manifest: {manifest_path}") from exc
    if not isinstance(files, list) or not files:
        raise ValueError(f"MLX artifact manifest has no files: {manifest_path}")
    root = path.resolve()
    locked_paths: set[str] = set()
    for entry in files:
        try:
            relative = entry["path"]
            expected_size = entry["sizeBytes"]
            expected_digest = entry["sha256"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"invalid file entry in {manifest_path}") from exc
        if (
            not isinstance(relative, str)
            or not relative
            or relative in locked_paths
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise ValueError(f"invalid file entry in {manifest_path}")
        locked_paths.add(relative)
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError(f"locked snapshot file is missing or unsafe: {relative}")
        if candidate.stat().st_size != expected_size:
            raise ValueError(f"locked snapshot file has the wrong size: {relative}")
        digest = hashlib.sha256()
        with candidate.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_digest:
            raise ValueError(f"locked snapshot file has the wrong digest: {relative}")
    if "config.json" not in locked_paths or not any(
        relative.endswith(".safetensors") for relative in locked_paths
    ):
        raise ValueError("locked snapshot does not contain configuration and weights")


def _wait(url: str, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + 900.0
    last_error = "server did not start"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"qualification server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5.0) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeError(last_error)


def main() -> None:
    parser = argparse.ArgumentParser(description="Qualify a locked model with the owned oMLX adapter")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--context-capacity", type=int, required=True)
    parser.add_argument("--max-concurrent-requests", type=int, required=True)
    parser.add_argument("--speculative-method", choices=["none", "mtp", "dflash", "dspark"], required=True)
    parser.add_argument("--max-draft-tokens", type=int)
    parser.add_argument("--dflash-draft", type=Path)
    parser.add_argument("--dflash-draft-manifest", type=Path)
    parser.add_argument("--dflash-block-size", type=int)
    args = parser.parse_args()
    _verify_locked_snapshot(args.model, args.artifact_manifest)
    if args.dflash_draft is not None:
        if args.dflash_draft_manifest is None:
            raise SystemExit("DFlash draft manifest is required")
        _verify_locked_snapshot(args.dflash_draft, args.dflash_draft_manifest)
    elif args.dflash_draft_manifest is not None:
        raise SystemExit("DFlash draft manifest requires a draft snapshot")
    with tempfile.TemporaryDirectory(prefix="magnitude-omlx-qualify-") as temporary:
        base = Path(temporary)
        config = AdapterConfiguration(
            model=args.model.resolve(), served_model=args.served_model, base_path=base,
            context_capacity=args.context_capacity,
            max_concurrent_requests=args.max_concurrent_requests,
            speculative_method=args.speculative_method,
            max_draft_tokens=args.max_draft_tokens,
            dflash_draft=args.dflash_draft.resolve() if args.dflash_draft else None,
            dflash_block_size=args.dflash_block_size,
        )
        validate_configuration(config)
        expected = detected_backend(config)
        port = _port()
        command = [
            sys.executable, "-m", "magnitude_omlx_benchmark.server",
            "--model", str(config.model), "--served-model", config.served_model,
            "--host", "127.0.0.1", "--port", str(port), "--base-path", str(base),
            "--max-concurrent-requests", str(config.max_concurrent_requests),
            "--context-capacity", str(config.context_capacity),
            "--cache-policy", "disabled", "--memory-guard", "off",
            "--speculative-method", config.speculative_method,
        ]
        if config.max_draft_tokens is not None:
            command += ["--max-draft-tokens", str(config.max_draft_tokens)]
        if config.dflash_draft is not None:
            command += ["--dflash-draft", str(config.dflash_draft)]
        if config.dflash_block_size is not None:
            command += ["--dflash-block-size", str(config.dflash_block_size)]
        log = (base / "qualification-server.log").open("wb")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            status = _wait(f"http://127.0.0.1:{port}/magnitude/benchmark/readiness", process)
            qualification = json.loads((base / "qualification.json").read_text())
            terminal = qualification["terminal"]
            timings = terminal["timings"]
            payload: dict[str, Any] = {
                "kind": "omlx",
                "modelSnapshotComplete": True,
                "tokenizerLoaded": True,
                "chatTemplateLoaded": True,
                "toolsRendered": True,
                "toolCallParsed": True,
                "contextCapacity": status["context_capacity"],
                "parallelSequences": status["max_concurrent_requests"],
                "expectedBackend": expected,
                "actualBackend": status["speculative_backend"],
                "terminalTimingEvidence": True,
            }
            if expected != "none":
                payload["draftTokens"] = timings["draft_n"]
                payload["acceptedDraftTokens"] = timings["draft_n_accepted"]
            print(json.dumps(payload, sort_keys=True))
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.close()


if __name__ == "__main__":
    main()
