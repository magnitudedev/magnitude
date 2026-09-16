"""Verify vendored hashes and apply the extraction patch in the build tree."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "manifest.json").read_text())
    with tempfile.TemporaryDirectory(prefix="templates-extraction-") as temporary:
        staging = Path(temporary)
        for entry in manifest["files"]:
            source = root / "upstream" / entry["path"]
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Vendored source hash mismatch: {entry['path']}")
            destination = staging / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        for patch in sorted((root / "patches").glob("*.patch")):
            subprocess.run(["patch", "--batch", "-p1", "-i", str(patch)], cwd=staging, check=True)
        for entry in manifest["files"]:
            source = staging / entry["path"]
            destination = args.destination / entry["path"]
            if not destination.exists() or destination.read_bytes() != source.read_bytes():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)


if __name__ == "__main__":
    main()
