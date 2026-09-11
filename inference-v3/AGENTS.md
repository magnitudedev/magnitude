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
change we want upstreamed. The TileLang suites relevant to a change are the
ones for the backends it touches (`testing/python/<backend>`, plus the shared
transform and language tests for cross-backend changes). The workflow:

1. **Change TileLang.** Commit on `magnitude` inside the submodule and push to
   the fork. When Magnitude should use the new commit, bump the submodule
   pointer in its own Magnitude commit, after the relevant TileLang suites and
   `tests/numerics` pass against it.
2. **Prepare a PR branch.** Cut `pr/<name>` from `upstream/main` and apply just
   that change (`git diff upstream/main...magnitude -- <paths>` is the source
   of truth for what we carry). Build it, run the relevant suites on it alone,
   push it to the fork. The branch can sit there as long as needed; this step
   is done when the branch is green and a draft description exists.
3. **Open the PR.** A separate step, and only after discussing the branch and
   the description with the user and getting explicit approval. Never open an
   upstream PR on your own. Review changes land on the `pr/` branch and are
   merged back into `magnitude`.
4. **Pull upstream in.** When a PR of ours merges, or upstream has something we
   want: in the submodule, `git fetch upstream && git merge upstream/main`
   (merge, never rebase, so older Magnitude commits keep buildable pointers),
   resolve conflicts, rebuild, run the suites, push `magnitude`, bump the
   pointer. Merged PRs drop out of `git diff upstream/main...magnitude`.
