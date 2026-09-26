#!/usr/bin/env python3
"""Join independent GPU observations with native archive inspection.

An external, explicitly selected decoder remains evidence with limitations.
Successful decoding is not permission to construct a production timing profile.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser(__doc__)
p.add_argument("measurements", type=Path)
p.add_argument("archives", type=Path)
p.add_argument("extractor", type=Path)
p.add_argument("decoder", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()

def sha(data):
    return hashlib.sha256(data).hexdigest()

raw_bytes = a.measurements.read_bytes()
raw = json.loads(raw_bytes)
captures = raw["nativeArchives"]
if not captures:
    raise SystemExit("measurement report has no captured native archives")
ids = [row["pipelineID"] for row in captures]
if len(set(ids)) != len(ids) or any(Path(name).name != name for name in ids):
    raise SystemExit("invalid or repeated pipeline identity")
if any(row["pipelineID"] not in ids for row in raw["observations"]):
    raise SystemExit("an observation lacks its native archive")

rows = []
for capture in captures:
    name = capture["pipelineID"]
    archive = a.archives / (name + ".metalar")
    source = a.archives / (name + ".metal")
    if sha(archive.read_bytes()) != capture["archiveSHA256"] or sha(source.read_bytes()) != capture["sourceSHA256"]:
        raise SystemExit(f"capture identity mismatch: {name}")
    compute = a.archives / (name + ".compute.bin")
    code = a.archives / (name + ".code.bin")
    for args in [("--extract-compute", compute, archive), ("--extract-main", code, compute)]:
        subprocess.run([str(a.extractor.resolve()), *map(str, args)], check=True, timeout=30)
    result = subprocess.run([sys.executable, str(a.decoder.resolve()), str(code)], capture_output=True, text=True, timeout=30)
    (a.archives / (name + ".disassembly.txt")).write_text(result.stdout)
    (a.archives / (name + ".disassembly.stderr")).write_text(result.stderr)
    failures = {marker: result.stdout.count(marker) for marker in
                ["<disassembly failed>", "Unrecognized opcode", "Length mismatch"]}
    decoded = result.returncode == 0 and not any(failures.values()) and not result.stderr
    instructions = re.findall(r"^\s*[0-9a-f]+:\s+(?:[0-9A-F]{2}\s+)+\s*(\S+)", result.stdout, re.MULTILINE)
    decoded = decoded and bool(instructions)
    rows.append({**capture, "main_shader_bytes": code.stat().st_size,
                 "main_shader_sha256": sha(code.read_bytes()), "decoder_exit_code": result.returncode,
                 "decode_failure_markers": failures, "decoded_without_reported_errors": decoded,
                 "static_mnemonic_counts": dict(sorted(Counter(instructions).items())) if decoded else None,
                 "partially_named_mnemonics": sorted({name for name in instructions if "todo" in name}) if decoded else None,
                 "observations": [row for row in raw["observations"] if row["pipelineID"] == name]})

report = {
    "status": "independent native inspection; timing profile remains unqualified",
    "device": raw["device"], "registry_id": raw["registryID"], "operating_system": raw["operatingSystem"],
    "measurements_sha256": sha(raw_bytes), "extractor_sha256": sha(a.extractor.read_bytes()),
    "decoder_sources_sha256": {path.name: sha(path.read_bytes()) for path in sorted(a.decoder.parent.glob("*.py"))},
    "pipelines": rows,
    "limitations": ["Decoder output is third-party interpretation, not a validated hardware contract.",
                    "Static mnemonic counts cover the extracted main shader only, excluding the constant prologue; they are not dynamic instruction counts.",
                    "Operand/control interpretation, native register allocation, spills, occupancy, and held-out timing predictions still require qualification.",
                    "These observations do not score or select Seismic candidates."],
}
a.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"pipelines": len(rows), "decoded_without_reported_errors": sum(row["decoded_without_reported_errors"] for row in rows),
                  "failed_pipelines": [row["pipelineID"] for row in rows if not row["decoded_without_reported_errors"]]}))
