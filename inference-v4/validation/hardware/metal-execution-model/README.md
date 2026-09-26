# Metal execution-model laboratory

A working, standalone prototype of pre-native analytical GPU estimation on
an Apple M4 Pro. It is an experiment, **not a certified production profile**.

The completed three-revision experiment and measured results are in
[the result report](../../../../specs/26-09-21/metal-execution-model-results.md).
Final held-out error: median 16.7%, p90 45.1%, worst factor 5.07×. This prototype
does not establish a universal bounded-error estimator.

`program.py` lowers a closed laboratory grammar into both Metal source and an
instruction/dependency program. `scheduler.cpp` executes SIMD waves against
issue ports, dependencies, register reservations, spills, memory service,
barriers and matrix resources. It also executes compare/exchange retries,
including inactive successful lanes waiting for reconvergence.

`helpers.py` instruments **fixed helper-library AIR**, executes it on the CPU to
select paths for declared input patterns, merges paths across SIMD lanes, and
allocates their live values. CPU execution time is never a predictor input.
Candidate AIR/native binaries and candidate timings are not predictor inputs.
`machine.py` can evaluate from the frozen helper descriptors without LLVM,
Metal, the compiler, or device access.

There are two arithmetic realizations: ordinary generated source, and fixed
compiled blocks of 8/16/32 values with runtime dimensions. The latter tests
whether stabilizing the compiler's register allocation and instruction
structure improves prediction. It has no fitted block-specific correction.

## Protocol

The final corpus contains 252 fixed calibration configurations and 398 held-out
configurations in two partitions. The historical `interval` label names the
first held-out partition; its observations do **not** calibrate uncertainty,
points, parameters or rankings. All predictions are frozen before either
partition is timed. No case is dropped from assessment.
The entire qualification batch is measured again in a second session. Earlier
revisions and their frozen predictions are retained under `results/v1/` and
`results/v2/`; the final cases reuse none of their timed runtime configurations.
Operation families and some compiled kernels are shared across revisions, so
this is validation on new configurations, not unseen operation families.

The endpoint is command-buffer GPU start/end, divided by repeated observations
of the declared dispatch sequence. Buffers already exist. Each observation
follows 32 fixed warm-up dispatches on separate buffers. Two pilot dispatches
precede seven randomized measurement blocks. Atomic observations use one
repetition; others use up to 64. CPU output references check selected outputs;
the experiment does not check every output byte or claim end-to-end application
latency. Host submission and compilation timing are retained separately.

Shared resource parameters are initialized from fixed isolated mechanisms and
jointly reconciled against fixed calibration observations. Calibration reports
local Jacobian rank and unidentified parameters. Core count is observed;
partition count, maximum resident waves, register capacity and cache tiers are
explicit hypotheses. A fitted effective parameter is not automatically an
identified intrinsic hardware fact. The profile records that distinction.

Qualification reports absolute relative time error `abs(predicted/actual - 1)`,
percentiles, worst factor, threshold coverage, selection regret and repeatability.
It also reports a sensitivity envelope over three explicitly listed
residency/register hypotheses. That envelope is **not a confidence interval or
a proven error bound**. Expanding it cannot certify a missing mechanism.

## Reproduce

All generated sources, binaries, profiles and measurements belong in ignored
`results/`. No production crate is modified. Run from this directory.

```sh
python3 -m venv --system-site-packages results/venv
results/venv/bin/python -m pip install llvmlite==0.49.0 numpy scipy matplotlib
export PYTHONDONTWRITEBYTECODE=1
python3 corpus.py
python3 tooling.py prepare
clang++ -O3 -std=c++17 -dynamiclib scheduler.cpp -o results/scheduler.dylib
```

Create an isolated directory on the measuring Mac with `mktemp -d`. Copy
`runner.swift`, `archive.swift`, `results/helpers` and
`results/calibration-sources` there. Compile with `swiftc -O`, then run:

```sh
./archive helpers
./runner calibration-sources cal sealed-calibration --warm > sealed-calibration.jsonl
```

Copy `helpers/archives` to local `results/archives` and the JSONL to
`results/cal.jsonl`. Then:

```sh
results/venv/bin/python tooling.py extract
results/venv/bin/python helpers.py
results/venv/bin/python test_model.py
results/venv/bin/python calibrate.py > results/calibration-fit.log
results/venv/bin/python qualify.py freeze
```

Freeze refuses to overwrite an existing prediction artifact. Preserve the
entire run before starting a new model revision. Only after freezing, upload
`results/qualification-sources` and run on the measuring Mac:

```sh
./runner qualification-sources all qualification --warm > qualification.jsonl
./runner qualification-sources all qualification-repeat --warm > qualification-repeat.jsonl
```

Copy those JSONL files into local `results/`, then:

```sh
results/venv/bin/python qualify.py assess qualification.jsonl
```

`test_model.py` has nine behavioral tests. `diagnostics.py` checks repeatability,
frozen source hashes, category errors and selected output checks, and generates
the comparison plot. Its counter preparation path emits an instrumented CAS
runner for diagnostic retry counts; these counts never calibrate the profile.

Earlier development campaigns of the recorded experiment are not held-out
evidence. See the dated result
document for the final frozen artifact identity and quantitative findings.

## Boundaries and limits

This laboratory grammar covers arithmetic chains and mixtures, lifetime
blocking, streaming/striding, pointer chasing, shared-memory barriers, SIMD
exchange/reduction, staged/resident matrix multiplication, native atomics,
software-float CAS, seven strict helper operations, lane input patterns, and
repeated dispatches. Every declared fixture is assessed. It is not an exhaustive
enumeration of the production `CandidateDomain` or a proof over arbitrary
inputs, contention schedules, OS states, or cache states.

AIR still precedes native register allocation and scheduling. Liveness-based
allocation, coarse cache tiers and uniform core placement remain model
assumptions. The unroll discrimination probes test that correspondence instead
of treating a source-level liveness count as measured native allocation.
Fixed helper probes use a controlled `noinline` policy and representative lane
cohorts, not exhaustive traces of arbitrary inputs. Returning CAS/load recurrence
measurements include their dependency scaffolding. Atomic queues and sharing
between returning and nonreturning variants remain hypotheses. The final model
distinguishes constant arithmetic, loop control, helper branches/calls and
actually used shared storage, but those distinctions do not close native
realization or memory/atomic behavior.
CPU allocation/materialization, transfer workflows, cold start and whole-model
inference are outside the measured GPU endpoint.

An accepted production design would retain the dependency boundary:
fixed characterization → immutable device profile → analytical evaluator.
The generic candidate domain, planner and feedback evaluator gain no dependency
on device characterization. This prototype does not install an evaluator,
narrow a production candidate domain, or use qualification feedback to correct
candidate scores.
