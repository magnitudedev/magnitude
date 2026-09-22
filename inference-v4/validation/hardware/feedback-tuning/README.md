# Standalone Metal feedback-search prototype

**For the actual Qwen3.5-4B experiment, see [QWEN.md](QWEN.md) and the
[five-minute convergence results](../../../../handoffs/26-09-21/qwen-feedback-convergence.md).**
The synthetic fixtures described below are the earlier harness experiment.

This is an experimental observer and generic optimizer for the
[feedback-tuning proposal](../../../../specs/26-09-21/seismic-feedback-tuning.md).
It does not change Seismic's production estimator, candidate domain, numerical
policy, or preparation interfaces. The fixtures generate Metal directly; they
are not evidence that Seismic can construct or qualify the same candidate space.

See the [first M4 Pro results](../../../../handoffs/26-09-21/metal-feedback-prototype.md)
for timings, independent quality comparisons, and limitations.

## What is implemented

- A Python controller that sees only finite opaque coordinate axes and observations.
- A small evolving population, legal mutation, donor crossover, and complementary
  difference experiments. Diagnosed interactions supply crossover groups.
- Exact source/pipeline reuse inside one Metal observer process.
- Complete command-buffer execution for every score; no partial-region proxy.
- A 60-second campaign allowance including observer startup, CPU references,
  buffers, compilation, warm-up, observations, and full-output checks.
- Fresh randomized finalist observations, plus random-search and exhaustive-screening
  comparison modes. The final output is a selected coordinate and observations,
  not a production Seismic prepared policy.

There are no application names, operation preferences, or source features in
the optimizer's proposal/ranking logic. Its sampling parameters are explicit
generic algorithm parameters. The fixture generator owns the restricted legal
space. All coordinates in these initial spaces are legal; constrained or
incompatible topology recombination remains unimplemented.

## Fixtures

Each fixture processes 1,048,576 Float values and has 2,048 variants.

| Fixture | Computation | Choices |
|---|---|---|
| `chain` | Eight elementwise stages | Seven fusion boundaries, four lane counts, four launch widths |
| `reduce` | Six elementwise stages, then row RMS normalization | Five fusion boundaries, four lane counts, four launch widths, four reduction widths |
| `stencil` | Eight stages, alternating elementwise work and a periodic three-point stencil | Seven fusion boundaries, four lane counts, four launch widths |

Every stage applies `v = fma(v, a_stage, b_stage)` followed by
`v / (1 + abs(v) / 32)`. The stencil first combines neighbors with weights
1/4, 1/2, 1/4; fusing it can trade materialization for repeated computation.
The reduction uses rows of width 1024 and epsilon 1e-5. Lane counts are
1/2/4/8 and launch/reduction widths are 64/128/256/512. Zero in a fusion axis
retains materialization. A coordinate of all zeros is the declared baseline;
it is deliberately simple and is not a tuned performance reference.

An independent host loop computes expected values. Every observed candidate
checks all output elements against an absolute/relative tolerance of
`2e-4 * max(1, abs(expected))`, including finiteness. Reduction associations
can differ. This is a fixture-level numerical criterion, not a universal
floating-point equivalence proof or Seismic numerical qualification.

## Timing and interpretation

The objective is the mean GPU duration per complete graph replay inside a
command buffer, with several complete replays batched to improve resolution.
Original inputs are immutable; all scratch is overwritten on each replay.
This measures a warmed repeated-invocation workload, not cold-cache latency or
host-to-completion latency. Host preparation and checking count toward the
tuning budget even though they are outside the GPU objective.

Each search process starts with an empty application pipeline cache. Metal's
system compiler cache is not cleared, so these runs are not guaranteed cold
native compilation. Raw observations retain compilation time, artifact misses,
GPU samples, repetitions, numerical error, and total observer time.

Interaction estimates use previously observed complete scores and a diagnostic
noise threshold. They guide proposals only: they are not confidence-sequence
results or certificates of independence. Finalists are remeasured in fresh
random order. Three seeds are a small prototype comparison, not broad statistical
qualification.

The controller stops search early and reserves time for final observations.
Its admission allowance is empirical. Synchronous native compilation or a GPU
hang can exceed it; this prototype does not implement bounded cancellation.
Report actual deadline misses. The selected coordinate has already executed
and passed checks, but creating a production deployable policy is not measured.

## Run

On a Metal-capable Mac with Swift and Python 3:

```sh
swiftc -O metal_worker.swift -o metal-worker
python3 tune.py --fixture stencil --algorithm evolution --budget 60 --seed 0 \
  --output ../../results/feedback-tuning/stencil-evolution-0.jsonl
```

`run_suite.py` runs sequential campaigns in its own `results/` directory. Copy
the files into a dedicated remote scratch directory before using it, or route
individual commands to `validation/results/feedback-tuning/` locally. It runs
two simple evolutionary campaigns, three evolutionary and three random-search
stencil campaigns, then exhaustively screens each domain with a longer budget
and remeasures up to 32 finalists. A screening pass is exhaustive only if its
reported unique-point count equals 2,048. No finite noisy screen proves the
exact expected-latency optimum.

`confirm.py` subsequently compares all selected coordinates and the screened
reference in the same observer process, in randomized blocks. This independent
comparison is outside each tuning budget and is used to assess selected quality.

The first runs use `m4-pro-01`, under
`/tmp/seismic-feedback-prototype-20260921`. Generated results belong under
`inference-v4/validation/results/feedback-tuning/`; do not stage them.

## Remaining scope

No Qwen run, Seismic integration, regional measurement, context-transfer audit,
dynamic-state workload, concurrent graph, tensor-core kernel, CUDA observer,
or general numerical-contract proof is implemented here. These are subsequent
steps, not capabilities inferred from success on the fixtures.
