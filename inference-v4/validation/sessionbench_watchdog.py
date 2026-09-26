#!/usr/bin/env python3
"""Bound Session Bench stalls and individual requests using its durable progress evidence."""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True)
    parser.add_argument("--seconds", required=True, type=float,
                        help="maximum seconds without meaningful benchmark progress")
    parser.add_argument("--request-seconds", type=float,
                        help="hard elapsed bound for each identifiable request")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.seconds <= 0 or (args.request_seconds is not None and args.request_seconds <= 0):
        parser.error("timeouts must be positive")
    if not args.command:
        parser.error("missing command")
    return args


def terminate_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def read_appended(path, offsets):
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return []
    offset = offsets.get(path, 0)
    if size < offset:
        offset = 0
    if size == offset:
        return []
    with path.open("r", errors="replace") as stream:
        stream.seek(offset)
        lines = stream.readlines()
        offsets[path] = stream.tell()
    return lines


def main():
    args = parse_args()
    log_path = Path(os.environ["SESSIONBENCH_WATCHDOG_LOG"]).resolve()
    process = subprocess.Popen(args.command, start_new_session=True)
    offsets = {}
    run_path = None
    armed = False
    last_progress_at = None
    last_progress = "waiting for marker"
    active = {}
    stream_counts = {}
    pending_warmup = None
    pass_pattern = re.compile(r"^Pass ([0-9]+)/([0-9]+): (.+)$")

    def progress(description, now):
        nonlocal last_progress_at, last_progress
        last_progress_at = now
        last_progress = description

    try:
        while process.poll() is None:
            now = time.monotonic()
            for raw in read_appended(log_path, offsets):
                line = raw.rstrip("\n")
                if line.startswith("Run: "):
                    candidate = Path(line[5:].strip())
                    run_path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
                    run_path = run_path.resolve()
                match = pass_pattern.match(line)
                if match or (not armed and args.marker in line):
                    armed = True
                    progress(f"pass marker: {line}", now)
                    if match:
                        active = {key: value for key, value in active.items()
                                  if not key.startswith("warmup-trace:")}
                        pending_warmup = f"{line} warmup"
                    print(
                        f"watchdog progress={last_progress!r} inactivity_seconds={args.seconds:g} "
                        f"request_seconds={args.request_seconds if args.request_seconds else 'disabled'}",
                        flush=True,
                    )

            if armed and run_path is not None:
                events_path = run_path / "events.jsonl"
                for raw in read_appended(events_path, offsets):
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("event")
                    if kind == "request_started":
                        active = {key: value for key, value in active.items()
                                  if not key.startswith("warmup-trace:")}
                        pending_warmup = None
                        key = "request:" + ":".join(str(event.get(name, "?"))
                                                     for name in ("target", "block", "request"))
                        active[key] = (now, key)
                        progress(f"request started: {key}", now)
                    elif kind == "request_finished":
                        request = str(event.get("request", "?"))
                        if request == "warmup":
                            active = {key: value for key, value in active.items()
                                      if not key.startswith("warmup-trace:")}
                            pending_warmup = None
                        key = "request:" + ":".join(str(event.get(name, "?"))
                                                     for name in ("target", "block", "request"))
                        active.pop(key, None)
                        progress(f"request finished: {key} outcome={event.get('outcome')}", now)

                logs = run_path / "logs"
                if logs.is_dir():
                    for engine_log in logs.glob("*-block-*.log"):
                        new_lines = read_appended(engine_log, offsets)
                        traced = [line for line in new_lines if line.startswith("target block ")]
                        if traced:
                            if pending_warmup is not None:
                                key = f"warmup-trace:{engine_log.name}"
                                active[key] = (now, pending_warmup)
                                pending_warmup = None
                            count = stream_counts.get(engine_log, 0) + len(traced)
                            stream_counts[engine_log] = count
                            progress(
                                f"engine block trace: {engine_log.relative_to(run_path)} line={count}",
                                now,
                            )
                    candidates = [logs / "warmup-streams.jsonl", *logs.glob("*-b*-*.jsonl")]
                    for stream_path in candidates:
                        new_lines = read_appended(stream_path, offsets)
                        if new_lines:
                            count = stream_counts.get(stream_path, 0) + len(new_lines)
                            stream_counts[stream_path] = count
                            progress(
                                f"stream chunk: {stream_path.relative_to(run_path)} line={count}", now
                            )

            if armed:
                if args.request_seconds is not None and active:
                    key, (request_at, description) = min(active.items(), key=lambda item: item[1][0])
                    elapsed = now - request_at
                    if elapsed >= args.request_seconds:
                        print(
                            f"watchdog timeout reason=hard-request request={description!r} "
                            f"elapsed_seconds={elapsed:.3f} limit_seconds={args.request_seconds:g} "
                            f"last_progress={last_progress!r} "
                            f"last_progress_ago_seconds={now - last_progress_at:.3f}",
                            flush=True,
                        )
                        terminate_group(process)
                        return 124
                stalled = now - last_progress_at
                if stalled >= args.seconds:
                    active_text = ",".join(sorted(active)) if active else "none"
                    print(
                        f"watchdog timeout reason=no-progress elapsed_seconds={stalled:.3f} "
                        f"limit_seconds={args.seconds:g} last_progress={last_progress!r} "
                        f"active_requests={active_text}",
                        flush=True,
                    )
                    terminate_group(process)
                    return 124
            time.sleep(0.1)
    except BaseException:
        terminate_group(process)
        raise
    return process.returncode


if __name__ == "__main__":
    sys.exit(main())
