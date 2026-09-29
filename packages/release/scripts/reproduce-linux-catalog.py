"""Run a final Linux ICN archive in a disposable local container and retain its assessment evidence."""

import argparse
import hashlib
import json
import pathlib
import platform as host_platform
import subprocess
import sys
import tarfile
import time
import uuid


def docker(*args, check=True):
    return subprocess.run(["docker", *args], text=True, capture_output=True, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", required=True, type=pathlib.Path)
    parser.add_argument("--host", required=True, choices=("linux-x64-gnu", "linux-arm64-gnu"))
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)
    parser.add_argument("--timeout-seconds", type=int, default=17 * 60)
    parser.add_argument("--cpuset-cpus", help="Docker CPU affinity, for example 0-3")
    args = parser.parse_args()

    descriptor = json.loads((args.artifact_directory / f"icn-base-{args.host}.artifact.json").read_text())
    archive = args.artifact_directory / descriptor["filename"]
    if descriptor["host"] != args.host or descriptor["kind"] != "icn-base":
        raise ValueError("artifact descriptor does not match the requested host")
    with archive.open("rb") as content:
        digest = hashlib.file_digest(content, "sha256").hexdigest()
    if archive.stat().st_size != descriptor["bytes"] or digest != descriptor["sha256"]:
        raise ValueError("artifact archive failed size or SHA-256 verification")

    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=False)
    installation = root / "installation"
    installation.mkdir()
    with tarfile.open(archive) as contents:
        contents.extractall(installation, filter="data")
    (installation / "installation.json").write_text(json.dumps({
        "schemaVersion": 1, "nativeBuild": descriptor["nativeBuild"],
    }))
    (root / "models").mkdir()
    (root / "cache").mkdir()

    platform = "linux/amd64" if args.host == "linux-x64-gnu" else "linux/arm64"
    native_architecture = "x86_64" if args.host == "linux-x64-gnu" else "arm64"
    machine = host_platform.machine().lower().replace("aarch64", "arm64")
    if machine != native_architecture:
        print(f"Running {native_architecture} under emulation; elapsed time is not native-host evidence", flush=True)
    name = "magnitude-catalog-repro-" + uuid.uuid4().hex[:12]
    command = (
        "apt-get update -qq && apt-get install -y -qq ca-certificates curl >/dev/null && "
        "/work/installation/bin/magnitude-inference serve "
        "--installation /work/installation/installation.json "
        "--model-store /work/models --cache-root /work/cache --bind 127.0.0.1:8080"
    )
    # Keep a shell as PID 1: native workers require the service owner to have PID > 1.
    run_args = ["run", "-d", "--name", name, "--platform", platform]
    if args.cpuset_cpus:
        run_args += ["--cpuset-cpus", args.cpuset_cpus]
    docker(*run_args,
           "-e", "MAGNITUDE_MEASUREMENT_PROFILE=1",
           "-e", "RUST_LOG=magnitude_service_server=info",
           "-v", f"{root}:/work", "ubuntu:22.04", "sh", "-c", command)
    last = None
    outcome = "timeout"
    deadline = time.monotonic() + args.timeout_seconds
    try:
        with (root / "assessment-timeline.jsonl").open("w") as timeline:
            while time.monotonic() < deadline:
                response = docker("exec", name, "curl", "-fsS", "http://127.0.0.1:8080/api/v1/model-assessments", check=False)
                if response.returncode == 0:
                    snapshot = json.loads(response.stdout)
                    state = snapshot["state"]
                    summary = {"revision": snapshot["revision"], "phase": state["_tag"]}
                    if state["_tag"] == "Failed":
                        summary["failure"] = state["failure"]
                        outcome = "failed"
                    elif state["_tag"] == "Ready":
                        for domain in ("catalog", "discovered"):
                            value = state[domain]
                            summary[domain] = {"phase": value["_tag"], "count": len(value.get("entries", []))}
                            if value["_tag"] == "Failed":
                                summary[domain]["failure"] = value["failure"]
                                outcome = "failed"
                        if all(state[domain]["_tag"] == "Available" and all(
                            entry["state"]["_tag"] != "Assessing" for entry in state[domain]["entries"]
                        ) for domain in ("catalog", "discovered")) and state["catalog"]["entries"]:
                            outcome = "complete"
                    if summary != last:
                        timeline.write(json.dumps({"observedAt": time.time(), **summary}) + "\n")
                        timeline.flush()
                        print(json.dumps(summary), flush=True)
                        last = summary
                    if outcome != "timeout":
                        break
                elif docker("inspect", "--format", "{{.State.Running}}", name, check=False).stdout.strip() != "true":
                    outcome = "service-exited"
                    break
                time.sleep(2)
    finally:
        with (root / "service.log").open("w") as log:
            subprocess.run(["docker", "logs", "--timestamps", name], stdout=log, stderr=subprocess.STDOUT)
        docker("stop", name, check=False)
        docker("rm", name, check=False)
    print(f"Outcome: {outcome}; evidence: {root}", flush=True)
    return 0 if outcome == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
