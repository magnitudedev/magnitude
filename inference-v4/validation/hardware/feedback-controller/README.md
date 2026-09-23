# Production controller effectiveness fixtures

Run from `inference-v4`, serially with other Cargo/device work:

```sh
cargo test -p seismic-compiler --lib coupled_fixture_requires_compound_edits_and_can_combine_donors
mkdir -p validation/results/feedback-search
cargo test -p seismic-compiler --lib coupled_cold_cache_controller_qualification -- --ignored --nocapture > validation/results/feedback-search/controller-coupled.log 2>&1
```

The experiment uses the real controller, candidate domain, native realization
cache, confirmation, continuation and published selector with a fake compiler and
controlled observer. Six binary axes select six independent kernel branches;
every candidate also executes one shared kernel. There are 64 optimized
coordinates and one general candidate. This is synthetic algorithm qualification,
not a benchmark of emitted application code or hardware.

The known latency surface has three invocation points and a different target
six-bit assignment at each point. Each matching two-bit group gives a benefit;
matching all three gives an additional benefit. At the complement of a target,
no single-bit mutation improves latency. Separate tests demonstrate compound
mutations improve such a parent and crossover can combine partial donors into
the optimum. The general candidate costs 1100 synthetic nanoseconds; the optimum
at each point is 500. Reported latency vectors keep all three points separate;
there is no assumed workload distribution.

Three seeds (1, 7, 29) compare evolution, fresh proposal search, finite
enumeration, and evolution with cost ordering disabled. Each run has actual
controller wall-time checkpoints at 120 and 240 milliseconds. The second is an
explicit continuation of the first; only its already-published selector is
measured at that checkpoint. Finalization time is charged. Reports include
actual elapsed time, overruns, confirmed coverage, native artifact formation and
reuse, formed native code/metadata sizes, and retained variant metadata. Allocation sizes and native bytes are
those of the synthetic compiler, not estimates of GPU code.

Native formation sleeps for either 300 or 2500 microseconds per new artifact;
complete observation setup and trial work have separate heterogeneous delays.
Actual elapsed wall time, including operating-system sleep overshoot, is charged.
The campaign starts with no formed artifacts. An independent probe session labels
assignment identities for ground truth, but shares no compiler or realization
cache with the measured session. Structural fixture construction and ground-truth
labelling are outside these controller allowances; this is not a measurement of
all source-to-prepared compilation time. Production native formation, executable
creation, observation, confirmation and publication are inside the allowance.

The 240ms checkpoint naturally reuses artifacts formed earlier in its cold-start
campaign. This is distinguished from the existing smaller warm-cache test, which
pre-realizes all its candidates before controller timing.

These experiments do not establish superiority of one search strategy. The
remaining independent Stage 4 ablations are fixed versus adaptive invocation
sampling, artifact reuse disabled versus enabled, and uniform high-precision
measurement versus screening. Hardware qualification with unseen invocation
points remains a separate gate.
