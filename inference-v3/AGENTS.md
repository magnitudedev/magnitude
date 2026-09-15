# Agents: Inference Engine Development

## TileLang

TileLang is the git submodule `tilelang`, on the `magnitude` branch of
`magnitudedev/tilelang`, installed as an editable path dependency. `uv sync`
builds it for the backends enabled through the `USE_*` build variables (on a
Mac: `USE_METAL=ON USE_CUDA=OFF USE_ROCM=OFF`); Python edits are live; after
C++ edits run `uv sync --reinstall-package tilelang` with the same variables.

Magnitude owns its optimization schedules, expressed through TileLang's existing
public language and IR. The compiler owns implementing those contracts on each
backend. The fact that a scheduling choice can be implemented in the compiler
does not justify putting it there. Magnitude's driver remains one implementation
for every backend and passes no per-target configuration.

TileLang is the mandatory path for all kernel computation and execution. Never
bypass TileLang for performance reasons, to reach a backend-specific feature,
or because a direct framework, runtime, library, intrinsic, shader, or launch
path is faster or easier to use. A performance gap is a reason to investigate the
authored schedule and its lowering, not permission to modify or bypass TileLang.

### Fork change policy

Optimize Magnitude through TileLang's existing public APIs and IR. Change either
fork only to fix a demonstrated correctness defect or when the required behavior
cannot be expressed without an unreasonable workaround.
Ordinary scheduling work, minor inconvenience, or a potential speedup is not enough.

Before editing a fork, investigate and test existing paths, including alternative
representations and scoped IR controls. Explain the precise limitation and why the
alternatives are inadequate. Keep the fix minimal and general; Magnitude retains
its scheduling policy. Verify performance separately from correctness, including
when removing optimizations, using comparable settings and identified builds.

The fork has two kinds of branches: `magnitude`, which Magnitude uses, and
`pr/<name>` branches based on upstream `main`, one per independently reviewable
change we want upstreamed. The fork's `3rdparty/tvm` points at
`magnitudedev/tvm`, our fork of `TileLang/tvm`, run the same way (its
upstream branch is `tilelang_main`). The TileLang suites relevant to a change are the
ones for the backends it touches (`testing/python/<backend>`, plus the shared
transform and language tests for cross-backend changes). The workflow below does
not itself authorize Git or GitHub writes; those require the user's explicit
request under the repository-wide Git policy.

1. **Prove the gap.** Apply the fork-change policy above. Inspect the relevant
   language, lowering, runtime, tests, and upstream implementation; reproduce the
   defect or explain the unreasonable workaround before proposing a fork change.
2. **Change TileLang locally.** Implement the smallest general fix with relevant
   tests. Keep it uncommitted and unpushed during development and validation.
   Magnitude may use the editable checkout, but must not gain a backend bypass.
3. **Disclose and request approval.** If the agent has an active goal, continue
   the goal using the local fork changes without interrupting it to request
   approval. Keep those changes uncommitted and unpushed throughout the goal.
   After the goal's implementation and validation work is completely finished,
   tell the user exactly what changed in TileLang, why each change belongs there,
   which defect or missing contract it addresses, and what tests were run. Do not
   commit or push the fork changes until the user explicitly approves them.
4. **Reconcile upstream before pushing.** Fetch `upstream/main` and check every
   local fork change against upstream before committing or pushing `magnitude`.
   If upstream already resolves any carried change, merge `upstream/main` into
   `magnitude` and resolve the resulting conflicts instead of pushing a duplicate
   implementation. Rebuild and revalidate the remaining fork delta after the
   merge.
5. **Land on the fork's `magnitude` branch.** After approval and upstream
   reconciliation, split the work into coherent commits on `magnitude` and push
   them to `magnitudedev/tilelang`. Bump Magnitude's submodule pointer only after
   the relevant TileLang suites and `tests/numerics` pass against those exact
   commits.
6. **Prepare contained PR branches.** For every distinct upstreamable change,
   cut a separate `pr/<name>` from `upstream/main` and apply only that change
   (`git diff upstream/main...magnitude -- <paths>` is the source of truth for
   what we carry). Rebuild the C++ library on each checkout before testing (a
   library built from another branch lowers wrongly), run the relevant suites
   on that branch alone, run `pre-commit run --files <changed>` and
   `git diff --check`, then push the branch to the fork. Upstream conventions:
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
7. **Open upstream PRs only after final approval.** Present each tested PR branch
   and its proposed description to the user. Creating a PR against
   `tile-ai/tilelang` is a separate step requiring the user's explicit final
   approval; never open an upstream PR on your own. CI for first-time
   contributors waits for maintainer approval, so local validation on the branch
   is the only pre-review signal.
   Bot feedback (CodeRabbit, pre-commit.ci) is triaged, not obeyed: verify each
   item against the code, reproduce it before adopting a suggested fix (its
   proposed test cases may not exercise what it claims), and ignore generic
   warnings such as docstring coverage on tests. Review fixes are amended into
   the existing commit and force-pushed with `--force-with-lease`; the user
   answers reviewers. Fixes that land on the `pr/` branch are merged back into
   `magnitude`.
8. **Pull upstream in.** When a PR of ours merges, or upstream has something we
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
  backend-specific compile-flag annotations. Existing scoped, backend-neutral
  TileLang IR controls (including unrolling) are allowed in authored schedules;
  they are not a reason to change compiler-wide defaults.
- Do not add a compilation path that bypasses tiling: no target-specific pass
  configuration, transforms, compiler patches, shader launchers, or kernel caches in
  Magnitude. The driver calls `tilelang.compile(program, target=...)` and nothing
  else, and is one implementation for every backend.
- Do not replace a TileLang operation with PyTorch, MLX, MPS, CUDA, Metal, a
  vendor library, or another runtime operation. This prohibition applies equally
  to kernels, driver helpers, fallback paths, and performance experiments intended
  to become production code.
- When the fork-change policy above is satisfied, follow its investigation,
  local-change, disclosure, and approval workflow. The
  resulting Magnitude kernel must use a portable TileLang contract; any
  backend-specific realization belongs in TileLang's lowering or runtime. The
  streaming attention rewrite to `T.alloc_fragment`, `T.gemm`, and `T.reduce_*`
  is the reference case.
- A schedule may target one backend's strengths, but it is named by its strategy,
  gated by a capability the driver reports from TileLang, and never by a backend
  name. Keep scheduling choices in Magnitude unless the fork-change policy above
  justifies a general compiler change.
