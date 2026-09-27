# Roofline

Roofline is the engine's performance development tool: make measurements on local
or remote machines, examine component costs and correctness, compare changes, and
browse stored evidence in a Textual tree. Measurements are stored automatically.

## Install and get started

From `inference-v3/`, install the CLI into its own tool environment:

```sh
uv tool install --editable ./roofline --with-editable ./formula-performance --python 3.12
roofline --help
```

Keep running commands from `inference-v3/` or a directory below it. The CLI reads
`roofline/models.json` and `roofline/targets.json` there. If those files do not exist,
create them as described in [Configuration](#configuration).

With a model configured on `local`, start with a small decode workload:

```sh
roofline models
roofline workers list
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m --context 128 --steps 4
roofline query --model qwen3.5-35b-a3b:gguf:q4_k_m
roofline
```

The first measurement installs the local worker, prepares dependencies and compiler
builds, verifies the model, and loads weights. Later measurements can reuse valid
preparation while collecting fresh samples. Cold preparation can take minutes.
The CLI itself requires no engine or GPU packages for browsing existing results.

## Configuration

Both files are local and **gitignored**. Models define artifact identity and file
locations; targets define machines and devices. Replace the example checksum and
paths with your own values before running.

**`roofline/models.json`**:

```json
{
  "models": {
    "qwen3.5-35b-a3b:gguf:q4_k_m": {
      "sha256": "<actual SHA-256 of the GGUF file>",
      "locations": {
        "local": "/absolute/local/path/Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "m4-pro-01": "/absolute/remote/path/Qwen3.5-35B-A3B-Q4_K_M.gguf"
      }
    }
  }
}
```

The ID has the form `model:format:quantization`. The checksum identifies the exact
artifact bytes. Compute it with `shasum -a 256 /absolute/path/model.gguf` (or
`sha256sum` on Linux). Current integrations consume single-file GGUF models.

Locations are **absolute file paths on each named machine**. There is no separate
model root or catalog lookup. Different filenames are fine, but every copy of one
model must match its declared checksum. Architecture and shapes come from the
artifact; format and quantization are not repeated as configuration fields.

**`roofline/targets.json`**:

```json
{
  "targets": {
    "local": {
      "connection": {"kind": "local"},
      "device": {"backend": "metal", "index": 0, "maximum_bytes": 34359738368}
    },
    "m4-pro-01": {
      "connection": {"kind": "ssh", "host": "m4-pro-01"},
      "device": {"backend": "metal", "index": 0, "maximum_bytes": 34359738368}
    }
  }
}
```

`maximum_bytes` is the engine memory budget; this example allows 32 GiB. Choose it
for the model and available device memory. CUDA targets use `"backend": "cuda"`.
SSH hosts can be aliases from your SSH configuration or addresses such as
`anders@sparky`. A model's location keys must name entries in `targets.json`.

There are no checkout or repository-interpreter settings. An optional `worker_root`
field on a target overrides the default worker directory with an absolute path or
a path starting with `~/`, interpreted on that target.

## Set up remote workers

```sh
roofline workers setup m4-pro-01
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m \
  --context 128 --steps 4 --targets local,m4-pro-01
```

The remote host needs Python 3, uv and working SSH access. Execution also needs the
host compiler and device tooling required by TileLang. It does not need an engine
checkout. Model files must already exist at the configured locations. Linux also
requires hwloc; on Debian/Ubuntu setup can unpack the distribution library inside
the worker directory. Other distributions need it installed on the host.

Setup copies the Roofline package and dependency lock into
`~/.local/share/roofline/` **on the target**, creates managed Python and an isolated
control environment, and starts the worker. Source snapshots and dependency locks
are transferred when measurements or scope discovery need them. The worker builds
inside its own directory, reusing its own matching dependency builds when possible.
Submissions include verified pinned fixture bytes so workers use the same corpus,
including when its upstream download has changed. The execution environment includes
the locked `performance` dependency group for independent artifact decoding.

Repeat setup to update an idle worker. Updates are rejected while work is accepted
or active. Workers survive SSH disconnection and start on demand when contacted;
they are not registered as boot services. The local worker uses the same directory
layout and installs on demand.

```sh
roofline workers list
roofline workers pause m4-pro-01
roofline workers resume m4-pro-01
```

`workers list` shows configured targets, worker directories and model availability;
it is not a live health probe. Pausing prevents new measurement acceptance.

## Measure a workload or component

The defaults are `local`, engine `magnitude`, workload `prose`, 2,048 context tokens,
128 decode steps, three samples and one warmup. `--context 16k` means 16,384 tokens.
The prose workload uses fixture tokens, so runs can use repeatable inputs.

```sh
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m --context 2k --steps 32
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m --scope prefill --context 2k
```

To measure a component, first discover its actual production selector:

```sh
roofline scopes --model qwen3.5-35b-a3b:gguf:q4_k_m --context 128 --steps 4
```

Selectors come from the engine's traced formula hierarchy. Use a returned selector;
paths differ between architectures. For example, this model exposes its first
block's routed feedforward component:

```sh
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m \
  --context 128 --steps 4 \
  --scope 'decode/qwen35.model[0]/qwen35.decoder[0]/qwen35.block[0]/qwen35.routed_feedforward[0]' \
  --step 0
```

`--step 0` selects the first decode occurrence. Quote selectors because they contain
shell-significant brackets. Scope discovery reads metadata; measurement verifies
the full artifact checksum. Each requested target gets its own outcome. Discovery
reuses an already built matching environment and transfers only engine Python
sources, not compiler source or model weights. On a cold target it reports that
preparation is required; a measurement prepares that environment automatically.

`measure --dry-run` checks the selection and reports required preparation without
building an environment or timing anything. Unresolved checks remain explicit.
Use `--samples`, `--warmups` and `--deadline` to change the
measurement protocol; the deadline is in seconds. `--engine llama.cpp` selects the
optional reference integration, which requires its installed library/toolchain and
currently exposes enclosing prefill/decode scopes.

## Query results and compare changes

Reports are JSON on stdout; progress goes to stderr. To save a report:

```sh
roofline measure --model qwen3.5-35b-a3b:gguf:q4_k_m --context 128 --steps 4 > report.json
```

Use IDs returned in reports in place of the uppercase placeholders below:

| What you need | Command |
| --- | --- |
| Model performance, latest results and best-correct history | `roofline query --model MODEL_ID` |
| One measurement and its recorded detail | `roofline query --measurement MEASUREMENT_ID` |
| Generated code, observations or other returned evidence | `roofline query --artifact ARTIFACT_ID` |
| A recorded source snapshot's manifest | `roofline query --source SOURCE_ID` |
| Request progress and per-target outcomes | `roofline query --request REQUEST_ID` |
| Compare two existing measurements | `roofline compare --baseline BASELINE_ID --candidate CANDIDATE_ID` |

A request can run on several targets and produce multiple measurements. A source ID
identifies captured code; an artifact ID identifies a stored evidence payload.
Queries returning a `cursor` have another page: repeat the same query with
`--cursor CURSOR`.

Filter model history to the condition you care about:

```sh
roofline query --model qwen3.5-35b-a3b:gguf:q4_k_m \
  --targets m4-pro-01 --context 128 --scope decode
```

Times are in seconds. Check `status`, `correctness` and `unavailable` alongside the
samples. A timing can survive failed checking or unavailable diagnostics; that does
not make it a qualified result. Comparisons report compatibility and differences,
and distinguish historical comparisons from paired execution. A negative latency
change means the candidate was faster when the comparison is valid.

To rerun captured code, add `--source SOURCE_ID` to `measure`. For fresh paired
execution against a recorded baseline, use `--against-source BASELINE_SOURCE_ID`.
Pairing currently supports safely replaceable Magnitude kernel edits; runtime,
formula or compiler changes can make it unavailable. Keep workload, scope and
sampling conditions consistent when evaluating a change.

To test a decode component on identical inputs across machines, name the source
and target that should produce those inputs:

```sh
roofline measure --model MODEL_ID --scope RETURNED_COMPONENT --step 0 \
  --source SOURCE_ID --input-source PRODUCER_SOURCE_ID --input-target m4-pro-02 \
  --targets m4-pro-02,sparky --context 128 --steps 4
```

Roofline prepares the producer boundary, transfers its values and verifies the
consumer's ports, precision and immutable weight bindings. Each consumer then runs
its independent check and fresh samples. Reports mark these inputs as frozen and
record their producer; they do not claim native current-production inputs. This
currently requires the same production graph and physical representations, one
decode position, and an integration snapshot supporting shared inputs. Unsupported
conversion fails explicitly. Omit both input options for normal production inputs.

## Stop waiting or cancel work

After submission, Ctrl-C stops the CLI waiting and prints the request ID. The
request continues in the background. Inspect or cancel it explicitly:

```sh
roofline query --request REQUEST_ID
roofline cancel --request REQUEST_ID
```

## Browse the Textual tree

Run `roofline` without a subcommand and choose a model. Its formula tree consolidates
all published workloads, hardware bindings and implementation revisions. There is
no workload selector and hardware is not a tree level or a set of columns.

Each component declares its primary quantity: processed token rows, floating-point
work, output elements, or explicit traffic. The row shows that unit and the range
of observed attainment after each point is normalized against its own hardware
binding. For example, 1,000 token/s against a 2,000 token/s ceiling and 4,000 token/s
against an 8,000 token/s ceiling both contribute 50% points to the same relation.
The range describes recorded conditions; it is not an average of raw machine times.

Select a component to see measured rates, hardware-relative ceilings, applicable
conditions, actual in-parent native contributions, and conditional parent
predictions. A point above 100% challenges the bound's applicability. Failed or
unchecked results cannot establish attainment. Achieved calibration rates remain
empirical references and never become theoretical hardware peaks.

Native attribution preserves fused regions at their smallest containing component.
A parent total constrains unresolved children jointly. Serial transfer predictions
require an explicit relation or matching production capture and isolated baseline;
they retain their assumptions and can be checked by subsequent measurements.

Historical records without analytical contracts remain available as raw evidence;
they are not silently assigned units or requalified. Discovering scopes publishes
the numerical graph and its formula declarations for offline browsing.

- `m`: choose a model.
- Arrow keys: navigate and expand/collapse the formula tree.
- Enter: focus the selected subtree; Escape: go back.
- `e`: open measurement history and named evidence for the selected component.
- Tab: move between the model selector, tree and details.
- `r`: refresh stored results; `q`: quit.

The CLI model query and this tree use the same analytical report. Query filters
restrict raw evidence history; they do not discard other points from the model.
Derived snapshots record the analytical rule revision and evidence watermark.

Browsing reads the local evidence store; workers may be offline. It never launches
measurements, traces models or opens devices. Scope discovery records model structure
for subsequent offline browsing, even before a measurement succeeds.

## Other operations

```sh
roofline workloads
roofline characterize m4-pro-01
roofline export --measurement MEASUREMENT_ID --output evidence.zip
roofline import evidence.zip
roofline import-session /absolute/path/to/completed/session
```

Characterization explicitly measures device resources used by performance analysis.
Export bundles contain selected measurements, source and evidence; model weights
and external toolchains are not bundled. Repeat `--measurement` to export several.
Session import matches recorded GGUF checksums to configured models and adds measured
HTTP and native timers to model history, with their original boundaries and dates.
Warmups are excluded. Response validation remains separate from numerical checking;
missing source and rendered-input identity remain unknown. Unmatched records are
retained for `query --artifact` using the returned artifact ID.

## Where files live

The coordinator collects measurements into `inference-v3/runs/performance/`.
Each worker owns `~/.local/share/roofline/` on its machine:

| Worker path | Contents |
| --- | --- |
| `installations/`, `current` | Control releases, their environments and the active-release link |
| `python/`, `bin/` | Managed Python and the worker's uv executable |
| `executors/` | Received source snapshots and execution environments |
| `discovery/` | Metadata-only Python source snapshots using existing dependencies |
| `cache/` | Dependency, compilation and verified fixture caches |
| `native/` | Worker-owned host libraries when setup supplies them |
| `roofline.sqlite`, `blobs/` | Worker journal and received/published evidence |
| `worker.log`, `environment.log`, `executor.log` | Supervisor, build and execution diagnostics |

Existing remote repositories and their virtual environments are not used. Weights
remain at their configured paths and are read directly. Failed cold builds can be
investigated in the target's `environment.log` without touching a checkout.

To run the CLI from elsewhere, set `ROOFLINE_ROOT` to the absolute `inference-v3/`
path. `ROOFLINE_WORKSPACE` can select another coordinator evidence directory; use an
absolute path to keep its meaning independent of the shell's working directory.

## Hardware capacity bindings

A target may declare `capacities`: named capacities with a resource pool, unit,
value, provenance, conditions, and kind (`upper-bound`, `conditional-upper-bound`,
or `achieved`). The formula model binds its resource parameters to these records.
Unbound parameters remain explicit; known necessary terms still establish a
possibly weaker bound. Measurements cannot manufacture missing hardware peaks.

The configured M4 Pro and GB10 machines use advertised 273 GB/s unified-memory
bandwidth; the local 40-core M4 Max uses 546 GB/s. These limits describe external
memory traffic, so their application to a formula is conditional on traffic crossing
that path. Cache-resident execution has different conditions. Sources:
[Apple M4 Pro](https://support.apple.com/en-ie/121555),
[Apple M4 Max](https://support.apple.com/en-us/121553), and
[NVIDIA GB10](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).

The device-free analytical package is `formula-performance`. Ops publishes its
numerical graph, explicit metric declarations, obligations, observations and native
mappings. Roofline transports this closure, assimilates it atomically, and exposes
it to queries and Textual. Reading reports never imports the numerical runtime.

**Qualification status:** implementation and connected physical validation are in
progress. See the [acceptance spec](../../specs/26-09-16/formula-performance-model.md).
