"""Import the explicit model-independent source corpus from the pinned checkout.

Run with a pristine extracted llama.cpp source directory. Local extraction changes
belong in patches/ and are applied to a build-directory copy, never upstream/.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REVISION = "930e2fa5995789efbf249a8bf61325bb626e417b"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_GROUPS = (
    "LICENSE",
    "common/chat*.cpp",
    "common/chat*.h",
    "common/peg-parser.*",
    "common/json*.cpp",
    "common/json*.h",
    "common/trie.*",
    "common/unicode.*",
    "common/jinja/*",
    "common/parsers/*.cpp",
    "common/parsers/*.h",
    "vendor/nlohmann/*",
    "vendor/sheredom/*",
    "tests/test-jinja.cpp",
    "tests/test-chat*.cpp",
    "tests/test-peg-parser.cpp",
    "tests/test-json-schema.cpp",
    "tests/test-json-schema-to-grammar.cpp",
    "tests/peg-parser/*",
    "tests/testing.h",
    "common/subproc.h",
    "common/subproc.cpp",
    "common/common.cpp",
    "tests/snapshots/*",
    "models/templates/*",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    sources = sorted({p for glob in SOURCE_GROUPS for p in args.checkout.glob(glob) if p.is_file()})
    manifest = []
    for source in sources:
        relative = source.relative_to(args.checkout)
        destination = ROOT / "upstream" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest.append(
            {"path": relative.as_posix(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    (ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/ggml-org/llama.cpp",
                "revision": REVISION,
                "files": manifest,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
