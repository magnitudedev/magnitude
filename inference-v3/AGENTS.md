# Agents: Inference Engine Development

## TileLang

TileLang is the git submodule `tilelang`, on the `magnitude` branch of
`magnitudedev/tilelang`, installed as an editable path dependency. `uv sync`
builds it for the backends enabled through the `USE_*` build variables (on a
Mac: `USE_METAL=ON USE_CUDA=OFF USE_ROCM=OFF`); Python edits are live; after
C++ edits run `uv sync --reinstall-package tilelang` with the same variables.

Target-specific compiler behavior (code generation, pipeline defaults, execution
adapters) goes in the fork, never in Magnitude's driver. Magnitude's driver is
one implementation for every backend and passes no per-target configuration.

The fork has two kinds of branches: `magnitude`, which Magnitude uses and which
evolves freely, and `pr/<name>` branches based on upstream `main`, one per
change we want upstreamed. The fork's `3rdparty/tvm` points at
`magnitudedev/tvm`, our fork of `TileLang/tvm`, run the same way (its
upstream branch is `tilelang_main`). The TileLang suites relevant to a change are the
ones for the backends it touches (`testing/python/<backend>`, plus the shared
transform and language tests for cross-backend changes). The workflow:

1. **Change TileLang.** Commit on `magnitude` inside the submodule and push to
   the fork. When Magnitude should use the new commit, bump the submodule
   pointer in its own Magnitude commit, after the relevant TileLang suites and
   `tests/numerics` pass against it.
2. **Prepare a PR branch.** Cut `pr/<name>` from `upstream/main` and apply just
   that change (`git diff upstream/main...magnitude -- <paths>` is the source
   of truth for what we carry). Rebuild the C++ library on that checkout before
   testing (a library built from another branch lowers wrongly), run the
   relevant suites on it alone, run `pre-commit run --files <changed>` and
   `git diff --check`, push it to the fork. Upstream conventions:
   - One commit per logical change, titled with upstream's bracketed tags:
     `[BugFix][Metal] ...`, `[Metal][Cache] ...`.
   - Tests for a target live in `testing/python/<target>`. Assertions about
     generated source lower through `tilelang.lower` inside a `PassContext` so
     they run without a device; only execution is gated with
     `tilelang.testing.requires_<target>`. Never skip a whole module on device
     availability.
   - Describe the PR as Problem, Root cause (bug fixes), Change (by file),
     Tests, Validation (hardware, exact commands, pass/skip counts), Related
     (overlapping upstream PRs and how they reconcile). Tell the user what the
     description will say; it is not a file anywhere.
3. **Open the PR.** A separate step, and only after discussing the branch and
   the description with the user and getting explicit approval. Never open an
   upstream PR on your own. CI for first-time contributors waits for maintainer
   approval, so local validation on the branch is the only pre-review signal.
   Bot feedback (CodeRabbit, pre-commit.ci) is triaged, not obeyed: verify each
   item against the code, reproduce it before adopting a suggested fix (its
   proposed test cases may not exercise what it claims), and ignore generic
   warnings such as docstring coverage on tests. Review fixes are amended into
   the existing commit and force-pushed with `--force-with-lease`; the user
   answers reviewers. Fixes that land on the `pr/` branch are merged back into
   `magnitude`.
4. **Pull upstream in.** When a PR of ours merges, or upstream has something we
   want: in the submodule, `git fetch upstream && git merge upstream/main`
   (merge, never rebase, so older Magnitude commits keep buildable pointers),
   resolve conflicts, rebuild, run the suites, push `magnitude`, bump the
   pointer. Merged PRs drop out of `git diff upstream/main...magnitude`.

## Kernels and backends

Every kernel is portable TileLang written against `tilelang.language`. Backend
behavior enters the engine only through the fork.

- Do not call a backend directly from a kernel: no `T.call_extern`,
  `T.call_pure_extern`, or `T.call_intrin` naming a target function, no
  `tilelang.metal`/`tilelang.cuda` dialect imports, no `tilelang.tvm` access, no
  compile-flag annotations.
- Do not add a compilation path that bypasses tiling: no target-specific pass
  configuration, transforms, compiler patches, shader launchers, or kernel caches in
  Magnitude. The driver calls `tilelang.compile(program, target=...)` and nothing
  else, and is one implementation for every backend.
- When a backend needs an optimization the language cannot express, first look for
  the TileLang abstraction that expresses it. If none exists, propose a fork change
  (a language operation, a lowering, or a target pipeline default) with tests under
  `testing/python/<backend>`, land it on `magnitude`, then use it from the kernel
  portably. The streaming attention rewrite to `T.alloc_fragment`, `T.gemm`, and
  `T.reduce_*` is the reference case.
- A schedule may target one backend's strengths, but it is named by its strategy,
  gated by a capability the driver reports from TileLang, and never by a backend
  name. Backend-specific optimization is a TileLang concern, not a kernel concern.
