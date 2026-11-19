# Magnitude

Magnitude runs AI models locally. Most work is the inference engine in `inference/`: the Rust engine,
the Seismic kernel language and compiler, native kernels for Metal, CUDA, Vulkan and CPU, and `mrig`
for checks and measurements on hardware. See `inference/AGENTS.md` for engine work.

## Engineering principles

1. **Widest first.** Solve the root cause, never the symptom; the class of issues, not the one
   instance; structure before detail.
2. **Minimal entropy.** No more parts, and no more ways of doing the same thing, than needed. Each
   responsibility has a single owner. Logic and data must earn their existence.
3. **Correct by construction.** Invalid states have no representation. Paths believed impossible are
   eliminated by construction, or treated as invariant violations (panic), never hidden.
4. **Strict upgrade.** Every change is equal or better on every axis it touches: correctness,
   performance, cleanliness. Trade-offs are presumed avoidable until proven otherwise.

## Running checks and measurements

Hardware checks and measurements run only through `mrig` (`uv run --project inference/evaluation mrig …`;
`mrig --help` lists the instruments). Run each in the background with `--purpose` and `--estimate`,
and keep working while it runs.

- **Only run to decide.** Run a check only when you need information to decide the next step (or to
  verify a completed change) and reading, arithmetic, or existing results (`mrig runs`) cannot answer
  it. Make sure the run actually exercises the changed path and can resolve the effect above noise.
- **Use the cheapest check that answers the question.** Move up the cost ladder only as far as the
  question requires:
  `read/reason → compile → unit test → kernel bench → model step → whole model`.
  Scope to what changed — one representative case per affected path, model, and backend. Never run
  broad suites or sweeps.
- **Validate in batches, correctness before timing.** Don't re-run after every small edit, and don't
  measure performance until correctness is established. Check `mrig runs` first to avoid duplicate
  runs or busy hosts.
- **React to failures and overruns.** Read a failure fully before running anything again. If a run
  exceeds its estimate, re-evaluate whether it is the right check rather than blindly waiting it out.
