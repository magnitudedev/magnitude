---
applies_to:
  - inference/seismic/runtime/src/native/coverage.rs
  - inference/seismic/runtime/src/native/coverage/**
  - inference/service/kernel-formation/**
  - inference/service/server/src/assessment/catalog_kernels.rs
  - inference/scripts/kernels-form.ts
---

# Kernel formation

Kernel formation proves, before release, that every native kernel the engine ships forms with
each GPU backend's real toolchain, and that every kernel every supported catalog model requests
forms. It is an offline check: no shipped binary contains it.

## What a run establishes

For one backend, a run:

- forms every native implementation under configurations chosen so that every preprocessor group
  of its sources (the asset, its includes and Seismic's native library) is compiled at least once,
  and every value of every `form` tuning parameter is formed under each device configuration;
- checks that every weight slot takes every resident representation: planning gives any weight any
  representation of the backend's resident row layout, and each weight its own, so an element
  parameter the entry reads (other than a conversion target, which its source chooses) that is
  implemented at one of them must be implemented at each, with only that parameter changed from
  the entry's base configuration. Preprocessing alone decides, so the check also runs as a test on
  every host;
- forms every distinct kernel request of every supported catalog model at its default
  specialization on the backend's base configuration.

A run succeeds only if every formation succeeded, every group was reached, every form value was
formed, no entry lacks an implemented configuration, no weight slot leaves a resident
representation untaken, this host could form every configuration the search kept, and every
catalog request formed. A backend whose toolchain the host lacks is an
error when named, never a skip.

Formation produces the real artifact preparation produces (a CUBIN through ptxas, sealed SPIR-V,
Metal functions), short of loading it on a device. No lighter proxy substitutes for it.

## Ownership

- **The coverage search** (Seismic) decides what to form. It renders programs and preprocesses
  their skeletons, and hands each formation to the caller's `Forming`, which returns a verdict.
  Toolchain work runs at most one job per core.
- **The formation tool** owns the ledger and progress reporting; it forms in its own process. It
  is a workspace member outside the default members, so the coverage feature never joins default
  builds.

## Verdicts

| Verdict | Meaning |
|---|---|
| Formed | the toolchain produced the artifact |
| Rejected | a `static_assert` fired: the configuration is not implemented, and the search seeks its groups elsewhere |
| Failed | a compile error: a defect |

A row's programs form in order; the first that does not form decides the row.

## Formation identity

A formation is determined by its program (text and entries) and its toolchain identity: the
toolchain namespace and the identity material of the configuration. The material is derived from
the same compiler-identity functions the runtime's program cache uses (CUDA: NVRTC release,
architecture and compile options; Vulkan: glslang release, SPIR-V target, seal version, and the
formation facts; Metal: the OS build, which ships the compiler, and the language version). The
rendered text is the entire compiler input; no headers or include paths are passed separately.
The ledger key hashes exactly these inputs, so an unchanged key is an unchanged formation.

## Ledger

The ledger keeps, per backend, what runs established:

- a **verdict** per formation key: Formed or Rejected. Failed is never recorded;
- a **completion** per coverage key: a coverage run that established everything, with its
  summary.

Records are appended as they are established, so a run that fails or dies keeps everything it
finished. Unparseable lines are ignored and a missing ledger is an empty one. The ledger only
skips work whose outcome is already known; correctness never depends on it.

The coverage key hashes the toolchain identity (every configuration it forms and the preprocessor
that chooses them) and a digest, taken when the tool is built, of every source coverage's result
depends on: the kernel crate and the standard library its bundle is built from, Seismic, and the
locked dependency versions. Engine and service sources do not enter it. Coverage whose key has a
completion reports it and searches nothing; any change to those inputs runs the search again, and
every formation whose key is recorded is recalled. Catalog requests depend on the engine's
planning as well, so they are always enumerated; each request's formation is recalled when its key
is recorded.

## Search memory

The search holds no table of admissible configurations. Where an entry's condition can fail to
evaluate at its statics, the nearest-configuration questions of every frontier row sharing those
statics are answered by one walk over the admissible configurations that keeps only the current
answers, so the search's memory is independent of how many configurations a declaration admits.

## Progress

While a phase runs, the tool reports at a fixed interval the elapsed time, the phase's units done
of its total (entries searched during coverage, requests formed during catalog formation), and the
formations formed and recalled from the ledger. Formations per entry are discovered by the search,
so entry progress is not proportional to time.

## Acceptance

- Identical inputs give identical formation keys; changing the namespace, material, text, or any
  entry's symbol, group size or constants changes the key.
- A run repeated with no coverage input changed reports the coverage completion, searches nothing,
  and recalls every catalog formation.
- An engine or service change alone does not invalidate the coverage completion.
- A run after a change to one kernel family forms only that family's changed programs.
- A run that is killed keeps every verdict it recorded.
- A weight slot that rejects one resident representation of its backend's layout while it takes
  another fails the run, so no model whose weights need it can pass assessment and then fail to
  form.
