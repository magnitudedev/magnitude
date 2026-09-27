# Native templates

Model-independent extraction of the pinned llama.cpp Jinja, template analysis,
PEG parsing, chat-event mapping, grammar generation, and specialized handlers.
The Python consumer lives in `src/templates` in the engine distribution.

## Build

Requires CMake 3.24+, a C++17 compiler, Python 3, and `patch`. No model download,
llama library, GGML, Torch, or GPU SDK is used by this build.

```sh
cmake -S inference-v3/native/templates -B inference-v3/build/templates -DCMAKE_BUILD_TYPE=Release
cmake --build inference-v3/build/templates --parallel 4
ctest --test-dir inference-v3/build/templates --output-on-failure
```

The Rust engine's `inference-v4/templates` crate uses the same owned sources and
C ABI with `-DTEMPLATES_STATIC=ON`, embedding the library in the executable. Its
safe owners release native handles and copy borrowed stream events before the
next mutation. Build tools are development dependencies, not runtime dependencies.
The default shared-library build remains available to the Python consumer.

To bundle the shared library with the development Python module:

```sh
python3 inference-v3/native/templates/tools/build.py
```

The C ABI and Python wrapper currently expose compilation, capability inspection,
raw rendering, immutable request preparation, and semantic streaming events with
explicit terminal causes. The engine wheel build bundles the native library,
build identity, licenses, provenance manifest, and patches. Engine integration
and cross-platform release qualification remain in progress; this is not yet a
production replacement.

## Provenance and refresh

`manifest.json` records the exact upstream revision and original file hashes.
`upstream/` is unmodified upstream source, tests, fixtures, and licenses. Its
`common/common.cpp` is provenance for the few extracted string helpers, not a
build input. The build lists its translation units explicitly. Subprocess support
is used only by the Jinja reference tests.

`patches/0001-standalone-boundary.patch` replaces broad common/logging/assertion
dependencies, removes token delimiter operations and the legacy inference-linked
renderer, and replaces model discovery with explicit source/BOS/EOS construction.
Artifact selection belongs to the host. Authored source is not silently rewritten
or replaced with a fallback template.

`patches/0002-standalone-tests.patch` adapts upstream tests to this source-only
boundary. The test CLI's GGUF inspection utility is unavailable; artifact loading
remains outside the native library. The template suite's legacy renderer tests are
excluded; its Jinja cases remain. Core test cases remain upstream fixtures.

`patches/0003-argument-presence-and-time.patch` preserves omitted reasoning
controls, freezes native capability/differential probes, and propagates explicit
request time. `patches/0004-utc-and-json-invariants.patch` renders time in UTC and
keeps third-party noexcept assertions separate from recoverable boundary errors.

Template calls are serialized. Raw render context preserves missing special-token
variables; request preparation currently uses upstream's BOS/EOS string contract.
The native build exports only the versioned C entrypoints in `include/templates.h`.
Returned buffers stay valid until released; template and request handles are
checked IDs. The Python wrapper releases the GIL during calls and copies output
before releasing its native owner.

`tools/prepare.py` verifies original hashes and applies patches to a build-tree
copy. To refresh, obtain a pristine checkout of the intended revision, update the
pin in `tools/vendor.py`, import it, review the manifest/source delta, revise the
extraction patches, and rerun the complete qualification corpus. A successful
compile alone is not a compatibility result.

## Grammar qualification

The extraction generates whole-completion GBNF while the native schema tree is
still alive. Serialized PEG data intentionally omits schema nodes and cannot be
used to regenerate enforcing grammars. The matcher must consume only the declared
`grammar_initial_prefix` before generated output; it must not consume the full
conversation prompt. This strategy is undergoing tokenizer/platform qualification.

Patch 0005 fixes strict completion of optional calls and incomplete marker prefixes;
0006 fixes automatic call-ID delimiter ownership. Patch 0007 selects whole-completion
grammars and names nested repetition rules so the pinned converter retains grouping.
The Python converter adapter alpha-renames parsed rule identifiers before upstream
resolution to prevent reserved-name and case-normalization collisions. It checks
llguidance 1.8.0 and the converter source SHA-256 before use.

Schema admission currently accepts an explicit subset: primitive types, nested
objects/arrays, declared required properties, additional properties, enum/const,
`anyOf`, type unions without structural siblings, and local definitions references
without escaped path segments. Intersections that upstream discards are rejected.
Numeric/length bounds, formats, `oneOf`, `allOf`, external references, and unknown
constraint keywords are rejected. Pattern support is conditional on successful
native conversion; patch 0008 turns incomplete-conversion warnings into errors.
Patch 0009 preserves JSON Schema's default of allowing additional properties.

Handler support is narrower than JSON support: tagged raw strings accept plain
string schemas but reject constraints their grammar would discard (patch 0010).
The upstream Gemma handler accepts arbitrary dictionaries, so only unrestricted
object argument schemas are admitted there. A successfully parsed output alone
is not evidence that its argument schema was enforced.

Append-only streaming opts into bounded scanner-prefix caches (patch 0011).
Only fully examined UTF-8 characters and escape sequences are skipped on later
calls; provisional delimiters remain unconsumed. Ordinary upstream parse contexts
keep the original behavior. Patch 0012 additionally retains repetition prefixes
only when they produce no AST nodes and observe no input boundary. Production
streaming retains named-rule nodes for every format. The bounded cache preserves earlier enclosing prefixes when saturated. Differential native tests
cover every prefix, ordered choices, lookahead, Unicode, natural finalization, and
cache saturation.

Patch 0013 retains stable rule results and repetition AST nodes across appended
input. Boundary-dependent decisions remain provisional, including failed choices
and lookahead. Retained AST nodes and cache entries are bounded; exhaustion falls
back to parsing without new retention. Input relocation and context copies rebase
source views before reuse.

Patch 0014 provides shared immutable output fragments, subtree mapping caches,
and stable child-prefix folds. Existing recursive conversion rules use this shared
representation; content, reasoning, raw JSON, and tagged arguments use the same
suffix validation and emission mechanism. Source fragments hold offsets and escape
only newly published bytes. Tagged traversal skips subtrees with no tags without
suppressing named AST nodes. The former Gemma-specific output-sink patch is removed.

The shared implementation passes the native and Python correctness suites. Initial
structured scaling measurements show a substantial improvement, but broader scaling,
cache-exhaustion performance, and final cross-platform qualification remain open.
