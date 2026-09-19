#!/usr/bin/env python3
"""Bounded source-family scaling probe. Raw output belongs in ignored results/.

This measures real source construction/export, not optimization quality. A process
boundary keeps a slow constructor from preventing the remaining cases from running.
"""
import argparse
import hashlib
import json
import pathlib
import selectors
import subprocess
import time


def run(binary, case, timeout):
    started = time.monotonic()
    process = subprocess.Popen([str(binary), *map(str, case)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, bufsize=1)
    events = []
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    # macOS loader startup is not compiler time; separately cap it at 60 seconds.
    deadline = started + 60
    status = "completed"
    while True:
        if time.monotonic() >= deadline:
            status = "timeout" if events else "startup_timeout"
            process.kill()
            break
        available = selector.select(min(0.1, max(0, deadline-time.monotonic())))
        if available:
            line = process.stdout.readline()
            if not line:
                break
            event = json.loads(line)
            events.append(event)
            if event.get("stage") == "ready":
                deadline = time.monotonic() + timeout
        elif process.poll() is not None:
            # Drain buffered final events on the next readable iteration.
            continue
    process.wait()
    stderr = process.stderr.read()
    selector.close()
    if status == "completed" and process.returncode:
        status = "error"
    return dict(case=case, status=status, returncode=process.returncode, events=events,
                wall_seconds=time.monotonic()-started, stderr=stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--config-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.seconds <= 0:
        parser.error("repeats and seconds must be positive")
    config_bytes = (args.config_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)
    expected = dict(hidden_size=2560, intermediate_size=9216, num_attention_heads=16,
                    num_key_value_heads=4, head_dim=256, linear_num_key_heads=16,
                    linear_num_value_heads=32, linear_key_head_dim=128,
                    linear_value_head_dim=128, linear_conv_kernel_dim=4)
    for key, value in expected.items():
        if config["text_config"][key] != value:
            parser.error(f"configuration mismatch for {key}")
    if config["quantization"]["bits"] != 4 or config["quantization"]["group_size"] != 64:
        parser.error("expected four-bit group-64 weights")
    index_bytes = (args.config_dir / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_bytes)["weight_map"]
    for projection in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z"):
        for suffix in ("weight", "scales", "biases"):
            if f"language_model.model.layers.0.linear_attn.{projection}.{suffix}" not in index:
                parser.error("expected quantized recurrent projection")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [("dense", m, s, "q4") for m in (1, 32, 128, 512) for s in range(7)]
    cases += [("decode", 256, s, "q4") for s in range(11)]
    cases += [("decode", t, 5, "q4") for t in (32, 2048, 32768)]
    cases += [("recurrent", 1, s, "q4") for s in range(12)]
    cases += [("prefill", t, 0, "q4") for t in (32, 128, 512)]
    cases += [("dense-entry", 1, 0, "q4"), ("dense-entry", 128, 0, "q4"),
              ("decode-entry", 256, 0, "q4"), ("recurrent-entry", 1, 0, "q4")]
    # Dense control distinguishes quantization support from dimensional scaling.
    cases += [("dense", m, 1, "bf16") for m in (1, 128, 512)]
    metadata = dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                    seconds=args.seconds, repeats=args.repeats, cases=cases,
                    config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    index_sha256=hashlib.sha256(index_bytes).hexdigest(),
                    purpose="source construction/coverage only; no complete target cost model")
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    for repeat in range(args.repeats):
        for case in cases:
            result = run(args.binary.resolve(), case, args.seconds)
            name = "-".join(map(str,case))+f"-{repeat}"
            (args.out / (name+".json")).write_text(json.dumps(result,indent=2)+"\n")
            last = result["events"][-1] if result["events"] else {}
            print(name, result["status"], last.get("stage"),
                  round(last.get("end_to_end_ms", result["wall_seconds"]*1000),2), flush=True)


if __name__ == "__main__":
    main()
