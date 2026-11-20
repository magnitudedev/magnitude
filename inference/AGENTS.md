# Working on inference

## Workflow

**Fix the system, not the symptom.** Make engineering decisions: change what in the design causes
the problem, and redesign if needed. Patching a symptom leaves the cause in place, so the next
symptom appears and gets patched too, and the work gets stuck iterating in a loop.

How to run checks and measurements is in the root `AGENTS.md` ("Running checks and
measurements"); runs on hardware go through `mrig`.

**Clippy blocks.** `mrig check` runs `cargo clippy`, which fails on the lints the workspace denies
(`[workspace.lints]` in `inference/Cargo.toml`). The release runs the same clippy before
publishing, so a change lands only once `mrig check` passes.

**Check Windows when a change touches the OS.** The rig has no Windows host, so code that calls the
OS (memory mapping, symbols, threads, files, processes) is checked for Windows by cross-compiling
the shipped binaries' whole dependency tree: `bun run check:windows` (needs
`rustup target add x86_64-pc-windows-gnu` and a MinGW-w64 C compiler for the native build scripts,
`brew install mingw-w64` on macOS). Each OS call has one owner with an implementation per platform
(host memory and program code pages: `seismic-memory`; the C library functions programs call:
`seismic-cpu`'s `library`), never a call guarded at its use.

## Autonomy

**Act with full autonomy.** Make the decisions the work requires and keep going. Do what has been
authorised without asking again, and bring something back only when it is not yours to decide.
Stopping to ask halts the work until someone answers.

**Clear space with `cargo clean` automatically.** When a local disk is full or filling, run
`cargo clean` on stale target directories at once, without hesitation and without asking. Build
output is regenerable. On the hosts, `mrig` evicts the least recently used checkouts itself.

## Tuning parameters and defaults

Defaults and tuning have distinct jobs:

- **Predictable choices belong in defaults, not tuning.** When the right configuration follows
  from genuine device capabilities or problem geometry (subgroup width, reduction length,
  shared-memory/register pressure, hardware tensor ops), encode it as a default or `where`
  constraint so it holds generically for *any* device with those properties — never overfit to a
  single GPU we happened to measure, and never gate defaults or fixes on a specific PCI ID or
  hardware model.
- **Leave microarchitecture-specific trade-offs to the tuner.** Keep declared search spaces tight
  (the tuning budget never grows) to configurations that can actually win across devices.
- **Pruning requires both reason and evidence:** remove a configuration (via a `where` constraint
  or shorter value list, with the reason and numbers in a comment) only when (1) reading the kernel
  explains why it cannot win in terms of shape/parameters/capabilities, and (2) tuning history on
  that backend's devices confirms it never won.

## Measurement

- Model quality and speed are measured only through `mrig`. Its timed instruments hold the
  device alone and pass the thermal gate first.
- A difference between two builds is evidence only from `mrig ab`: both sides on one host, in
  alternating rounds, judged against what each side resolved.
- A run records the source it measured (`rig.json` in its results); check it is the tree you
  intend.
