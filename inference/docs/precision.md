# Precision policy

Magnitude allows small, explicitly bounded floating-point differences so the native kernel tuner
can choose faster implementations. A candidate must meet the precision policy before it can
continue competing on speed. Speed never compensates for failing a numerical check.

The engine supplies the limits. Seismic supplies the shared policy representation, comparison,
and enforcement in native tuning. This document explains the current behavior; the governing
contracts are [numerical precision](../../design/inference/seismic-numerical-precision.md) and
[execution planning](../../design/inference/engine/execution-plan.md).

## The production limits

For every finite floating-point result or writable-state element, compare the candidate value
with the reference value:

```text
error = abs(candidate - reference)
allowed_error = absolute + relative * abs(reference)
pass when error <= allowed_error
```

**Absolute** is a fixed allowance in the value's units. It provides a meaningful allowance near
zero, where a percentage alone would approach zero. **Relative** is an allowance proportional to
the reference's magnitude. The two allowances add together; an element does not have to pass two
separate tests.

| Result or writable-state type | Absolute | Relative |
| --- | ---: | ---: |
| F32 | `0.00001` | `0.0001` (0.01%) |
| F16 | `0.001` | `0.002` (0.2%) |
| BF16 | `0.01` | `0.01` (1%) |
| Integer, Boolean, index, or packed-code state | Exact | Exact |

These types describe the observed result or state, not the model's weight quantization. A Q4
model can produce BF16 activations, F32 results, and F16 state; each receives its corresponding
policy. Every floating result/state subject is assigned an explicit tolerance, with exact
comparison as the default for other subjects.

One case takes the type of what was computed instead of the type it is stored as: a result that
is a residual plus a value the kernel rounds to the activation type first (the MLP down
projection and the attention output projection write the F32 residual stream this way). Summing
in another order can round that value one step differently, which no F32 tolerance admits, so
such a result takes the activation type's row: BF16 for a BF16 model.

For example, BF16 allows **0.01 plus 1% of the reference's magnitude**:

| Reference | Maximum allowed error | Accepted interval |
| ---: | ---: | ---: |
| 0 | 0.01 | −0.01 to 0.01 |
| 1 | 0.02 | 0.98 to 1.02 |
| 10 | 0.11 | 9.89 to 10.11 |
| 100 | 1.01 | 98.99 to 101.01 |

Every checked element must pass. A bad element cannot be hidden by averaging its error with a
large number of correct elements. Shapes and representations must also match. NaNs, infinities,
signed zeros, and subnormals follow the shared `SpecialPolicy::PRESERVE` rules rather than using
the finite-value formula to excuse changes.

The shared `Tolerance` type also supports a relative floor and a limit on representable
floating-point steps (ULPs). Production uses a zero relative floor and no additional ULP cap.
The constants are defined in the engine's [precision policy](../engine/executor/src/native/tuning/precision.rs).

## Error classes

A kernel form whose error against its entry's default exceeds these limits by design declares an
**error class** (`error_class NAME when ...` in its native declaration). The tuner forms a
configuration of a class only when the load admits the class by name, none by default
(`--admit-error-class` on the engine CLI), and then validates it against the default under the
class's envelope instead of the per-element limits: a bound on the relative RMS difference of
each result and on its largest element difference in units of the reference RMS. Whether a model
tolerates a class is decided by the model's qualification, not at load. The envelopes are in the
engine's [precision policy](../engine/executor/src/native/tuning/precision.rs):

| Class | Forms | Relative RMS | Largest element |
| --- | --- | ---: | ---: |
| `int8_activations` | INT8 of `dense_expand` and `dense_output` on Metal tensor operations: activations as int8 per (row, 32 columns) against exact weight codes | `0.02` | `0.25` |
| `int8_token_packing` | PACK of `dense_expand`, `dense_output`, `project_rows`, `gated_delta_project`, `attention_project` and `attention_output` on Metal without tensor operations, over Q4_K, Q5_K, Q6_K or q4g32s (GGUF Q4_0) weights: activations as integer codes per (row, 32 columns), two rows packed into one F32 matrix operand against exact weight codes | `0.02` | `0.5` |

`int8_token_packing` is row-dependent: the top row's sums of a pair are exact, and the low row's
carry the rounding of the accumulator the two share, so a row's result depends on the row it is
packed with (and through it on how a request is split into forwards).

## What the reference is

Production executes the native kernel with its **declared default tuning parameters** to produce
the reference for each test case. Candidates run with the same inputs and initial writable state.
Reference observations are retained independently so later candidate executions cannot overwrite
them.

This establishes agreement with the native default. It does not establish independent agreement
with the mathematical operation: a defect shared by the default and a candidate can pass.
Independent kernel tests and model regressions therefore remain necessary.

The runtime also supports an explicitly selected portable reference for development tests.
Full-size portable reference execution proved too slow for production startup. Reference selection
is explicit; failure to execute a reference does not silently switch to another one.

## How the tuner enforces it

A candidate is a complete kernel entry with concrete tuning parameters. It can contain several
GPU launches. Validation checks every element of the entry's returned outputs and of the rows
its writable state declares written. Each case declares, per writable parameter, the rows the
entry writes (for attention, the appended rows of the history); the rest of that state is input
the entry only reads, so comparing it would re-read bytes that were just restored. Long
histories would make that the dominant startup cost. One case per tuned unit, the one with the
least state, instead restores and compares its writable state whole, so a candidate that writes
outside its declared rows is still rejected.

For a candidate without matching prior evidence, the tuner performs this sequence:

1. Restore the case's written rows (the whole state at the unit's one whole case) to their
   pristine contents and clear reused output storage. This prevents a missing write from
   inheriting a previous candidate's correct output.
2. Execute and time the candidate's first invocation for the case.
3. Read its outputs and written state, then compare them with the retained reference. Readback,
   comparison, and state initialization are outside the GPU timing interval.
4. On failure, reject the candidate immediately and stop its remaining performance samples.
   On success, check the remaining required cases and input rotations.
5. Once qualified, continue performance sampling, ranking, and confirmation. Reuse the first
   duration as a speed sample when it meets the existing warmup/calibration rules; otherwise it
   serves as calibration. There is no extra kernel invocation solely to obtain validation output.

Later timing passes are batched and do not restore mutable state: every configuration is timed
the same way, so the state they leave behind affects all of them alike, and only the validated
first invocation needs pristine state. Serving state is never used as tuning scratch. Comparison
still has a host-side cost even though the validation execution is shared with timing.

Ordinary search, searches over launch-local choices, startup census seeds, and final assembled
configurations obey the same numerical gate. Timing equivalence between two choices does not
authorize numerical reuse. The default configuration is also a candidate that must qualify;
it is not an unchecked fallback after a failure. If no candidate qualifies, preparation fails.

Test coverage is part of enforcement. Cases include served row classes, real model-weight
rotations where appropriate, and the empty as well as long attention histories. Attention history
lengths are `0, 256, 4096, 16384, 65536`, subject to the model's supported limits. The original
Gemma attention defect reproduces on the fresh-only path of an empty history and passed at long
histories, so checking only long histories would miss it even with a tighter tolerance.

## Cache reuse

Only a completed search whose choice passed every case is stored. Evidence binds the case
structure, policy, source, reference kind and artifact, candidate artifact, and device; the
cache key covers the implementation, the precision policy and the search objective. Tuning
inputs are generated test data and resident weights, so evidence is not bound to their bytes.

A hit uses the stored choice without constructing inputs, validating or timing. A changed
implementation, policy or objective is a different key and requires fresh qualification.
Persisted policy values and search weights retain their exact floating-point values through
JSON round trips.

## Relationship to compiler precision

Both the native tuner and compiler use `PrecisionPolicy` and `Tolerance`. Their enforcement
has different scopes:

| Policy | Meaning |
| --- | --- |
| `Exact` | Preserve the source's declared numerical behavior. |
| `Bounded` | Allow only differences within the declared numerical limits. |
| `Unconstrained` | Relax floating-point error bounds while retaining discrete-value and other correctness requirements. |

Production native tuning uses `Bounded` and rejects `Unconstrained`. It establishes empirical
agreement on its tested cases. Compiler selection requires an established numerical relation
to the portable source; passing tuner samples cannot make an otherwise unproven compiler
transformation eligible. The shared representation keeps the limits and comparison semantics
consistent without treating local test results as a proof for every input.

## Why these bounds, and what they guarantee

The current constants are the selected **1× policy** from local 0.25×, 1×, and 4× experiments on
Gemma 26B A4B and Gemma E2B using Metal on an M4 Max. Tightening to 0.25× slowed E2B by about 17%
without improving the answer checks. Loosening to 4× gained about 1.9% on Gemma 26B but lost
about 4.6% on E2B. The original Gemma/Pi regression passed at 1×, and the numerical gate rejected
the historical bad attention configuration at all three tested scales.

The [experiment report](../../bugs/26-09-29/bounded-precision-status.md) contains the chart,
raw measurements, unchanged answer mistakes, startup costs, and coverage limits. These results
support an engineering default for the tested workloads, not a universal optimal threshold.

Ordinary production builds use fixed constants. The development-only
`tuning-precision-experiment` feature enables `MAGNITUDE_TUNING_PRECISION_SCALE`, which scales
both absolute and relative allowances. It is not a production tuning knob, and the tuner does
not widen limits automatically to admit a faster candidate.

Local bounds do not prove that errors cannot compound across a model or change a near-tied token
choice. Whole-model numerical and output/parser regressions are development and release checks.
Startup does not run a full-model test for each candidate or search combinations of kernel
winners. The guarantee is that every selected native candidate has qualifying evidence for
its required local cases under the declared policy.
