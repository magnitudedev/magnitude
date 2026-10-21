# Precision system

The precision system lets native tuning choose faster implementations while controlling their
added numerical error. One caller policy, α, applies across operations, models, representations
and devices. The engine's standard is α = 8, set to admit every technique with a measured effect
on speed (the packed weight forms measure 3.1 to 6.7 from their defaults). Per-operation
allowances follow from the resolution of the
stored inputs; they are not handwritten tolerances or a table of model-specific permissions.

The system combines three kinds of knowledge. Checked portable bodies and representation
definitions describe the computation and its stored values. Executing native kernels establishes
their actual error and speed. Independent model evaluation measures what the admitted error costs
at model level over the intended workloads. None substitutes for the others.

This document defines the intended error model and admission policy. [Tuning](tuning.md) owns
search, deadlines and retained evidence; [compiler numerical applicability](seismic/compiler/numerics.md)
remains a separate source-preservation contract.

## What admission guarantees

Admission makes one exact statement: at every checked output and written state, the candidate
moved the default's result by no more than α times the movement that storage-resolution noise
causes there. Both quantities are measured at the point. Nothing is fitted, declared or inferred
from the rest of the model, so every candidate that fits α is admitted and no other is.

Admission does not state a ceiling on model-level quality change. The downstream cost of an error
depends on how the rest of the network responds to its direction and covariance, and that response
cannot be observed from an operation's own inputs and outputs. Tuning runs no model-level passes
and ships no sensitivity data, so a local correction for it would be a guess about the network.
There is none. Model-level impact is a measured property of the system, established by evaluation.

Four relations determine how a local ratio becomes a model-level effect. Evaluation measures each
over the intended operating regime; none is a universal fact about floating-point programs, and
none is enforced by admission.

1. **Small-error cost follows a square law.** At a fixed location, scaling an error of fixed
   structure by a factor c approximately scales its model-level distributional effect by c².
   Sensitivity depends on the location and the surrounding computation. Large perturbations,
   changed control decisions and different error directions need not follow the same curve.
2. **Independent error effects add.** Contributions from different operations or techniques
   approximately add when their propagated cross terms are negligible. Dependencies across
   layers, shared operands and time can break this; different kernel names do not establish
   independence.
3. **Existing error does not mask new error.** The same injected error has approximately the
   same incremental reference loss on matched higher- and lower-precision versions of a model.
   A coarse file is not free capacity for further degradation. This transfer is measured across
   file types, not inferred from bit width.
4. **Equal-sized errors cost different amounts.** Candidate error and storage-resolution noise of
   the same size at the same location differ in downstream effect according to their direction
   and covariance. The spread of that cost around the noise's cost is the system's model-level
   approximation. It is measured and reported, never corrected for.

If measurement shows a value of α costs more quality than is acceptable, the remedy is a different
α or a different generic yardstick, not per-model allowances or a correction inside admission.

## Why magnitude, structure and dependence matter

For one fixed rendered input, let δᵢ be the error introduced at operation i, including any state
it writes. Let Jᵢ map a small error there to its first-order change in the final logits. For
baseline probabilities p, the local KL metric on logits is F = diag(p) − ppᵀ. Then:

```text
change in logits ≈ Σᵢ Jᵢ δᵢ
Hᵢⱼ = Jᵢᵀ F Jⱼ

KL(baseline || perturbed)
    ≈ ½ Σᵢ δᵢᵀ Hᵢᵢ δᵢ + Σᵢ<ⱼ δᵢᵀ Hᵢⱼ δⱼ
```

The diagonal terms explain the square law. They also explain why equal-sized errors can have
different costs: the network amplifies some directions and suppresses others. A scalar
sensitivity times squared RMS is exact only for a fixed error structure. The tuner does not
compute or ship these Jacobians, which is why its local rule claims nothing about them; this
model explains what evaluation measures instead.

The cross terms explain compounding. At a fixed input, conditionally zero-mean errors with zero
cross-covariance make the expected cross terms vanish in this local model. Extending that result
across inputs must preserve the condition: H changes with the input. This is an expectation over
perturbations, not a guarantee for each token. Bias and correlated directions can
reinforce each other; favorable cancellation cannot be counted as spare error budget. Pairwise
raw correlation alone does not establish the absence of sensitivity-weighted interactions.
For n equal contributions, independent effects grow like n; perfectly aligned propagated
errors can instead grow like n².

Even independently perturbed stored weights produce correlated output errors when the same
weights are reused across tokens. A cache error is reused by later queries, and recurrent state
feeds subsequent state. Noise must follow that same storage lifetime, not be redrawn on each
read to make the experiment appear independent. Errors in discarded prompt intermediates and
errors published into persistent state have different future effects. A one-step comparison of
written state therefore requires validation over subsequent generation before it can stand in
for its lifetime cost.

There are also two different KL questions:

```text
distribution change = KL(default || candidate)
added reference loss = KL(reference || candidate) − KL(reference || default)
```

They are not interchangeable. In a common quadratic approximation around the independent
reference, let e be the default's existing logit error, d the added logit error, and G the
reference's local KL metric:

```text
added reference loss ≈ ½ dᵀ G d + eᵀ G d
```

The second term can reinforce or cancel the new error; it is first-order in d and can make
added reference loss negative. No masking requires this interaction, and changes in sensitivity
between the two baselines, not to invalidate the incremental-cost model. Small default-relative
KL alone does not establish that condition. Where the errors leave the local quadratic regime,
the expansion itself no longer supplies the argument.

## Storage resolution supplies the yardstick

A stored value can only be one of the values its format represents. The gap between adjacent
representable values is its step. The yardstick is how far the canonical default's output moves
when its stored inputs are perturbed by their own steps. It is a defined comparison scale read
from the stored data and declared types, not an estimate of the error the file already contains:
no original weights or knowledge of how the file was produced is needed.

| Stored input | Step |
| --- | --- |
| Quantized weights | Adjacent codes of each stored plane, including the format's own stored scales |
| Compressed state | Adjacent codes of the codec and adjacent values of its stored coefficients |
| Floating activations and dense state | Adjacent representable values of the declared type |

Two consequences follow. Finer storage gives a smaller allowance, all else equal. Files in one
format receive the same allowance however they were produced: importance-aware quantization and
quantization-aware training leave less error than their step implies, so a given α is a larger
share of such a file's own error than of a plainly rounded file's. α therefore has no fixed
equivalent in bits of precision across kinds of file.

**The reference perturbation.** Each stored value independently moves to the adjacent
representable value above or below, with probability 1/24 each, and otherwise stays. Storage
cannot hold part of a step; this is the realizable perturbation with the variance of rounding
error over one step, s²/12. A move with no adjacent value in range is no move. Indices, counts and
masks are not stored values and are never perturbed. One draw is made per stored value and reused
wherever that value is read.

**Obtaining the yardstick.** Execute the default once more on the perturbed inputs and measure
the movement of every returned output and written-state subject. One method serves every
operation: the real kernel carries the perturbation through its own normalization, rounding and
nonlinear stages, so nothing is derived about its internals. The yardstick does not depend on
the candidate. For an output linear in a perturbed input the expected movement is exact:

```text
y = Wx, perturbing W:    E[‖δy‖²] = Σᵣ Σₖ (sᵣₖ² / 12) xₖ²
```

Elsewhere the measured movement is one draw of the defined perturbation, and a few whole steps
need not move a nonlinear output as many part-steps of equal variance would. Evaluation
establishes its stability across draws.

## From local measurements to one admission rule

For each floating output or writable-state subject at a point, let ε be the candidate's relative
RMS difference from the default, and b the yardstick expressed with the same normalization.
Then r = ε / b is dimensionless. Equivalently, it compares candidate error power with
resolution-noise output power before taking the square root; the shared output normalization
cancels. A zero yardstick permits no nonzero error, and an unavailable yardstick grants no
admission to an alternative. Geometry, discrete values, memory behavior and exceptional-value
rules remain separate hard constraints.

The decision is:

```text
r ≤ α
```

There is no further term. In particular there is no correction for the structure of the error:
any such factor would have to estimate the network's downstream sensitivity from local data,
which local data does not contain.

The model-level consequence follows from the four relations. Let Cᵢ be the expected
default-relative KL from resolution noise at location i over the assessed inputs, and kᵢ the cost
of the candidate's error there relative to resolution noise of the same size:

```text
candidate contribution at i ≈ kᵢ × Cᵢ × rᵢ²,     rᵢ ≤ α

with additive effects:
total candidate contribution ≈ Σᵢ kᵢ × Cᵢ × rᵢ²
```

Admission fixes rᵢ ≤ α. Each location carries its own sensitivity in Cᵢ, so no budget is
assigned or redistributed between layers. The tuner knows neither kᵢ nor Cᵢ. Were every kᵢ equal
to one, the total would be at most α² × Σᵢ Cᵢ, the effect of one resolution-noise budget; how far
kᵢ departs from one is what evaluation measures. Reading the total as added reference loss needs
the no-masking relation as well. Correlated contributions do not simply add, and a candidate that
writes several subjects has joint structure that per-subject ratios do not describe.

α = 1 is not a promise to double the model's existing error, lose a fixed number of effective
bits, or admit every low-precision optimization. α is a ceiling on local movement, not a target to
spend. The squared ratios are contributions relative to resolution noise; neither they nor their
unweighted sum are KL predictions in nats.

## How this governs tuning

The [native tuner](tuning.md) selects the fastest measured admissible complete configuration.
The canonical default defines the local comparison and is always numerically admissible; it
need not be compared with itself. It must still form and execute legally. Its default status is
not independent evidence of correctness.

Inputs, default outputs and yardsticks are shared across candidate comparisons at a point.
A candidate's initialized execution supplies both timing and numerical observations; device-side
reductions compute the error before outputs are reused. An over-budget candidate is
rejected immediately. Performance sampling does not trigger a separate reference execution for
every candidate. The selected artifact must additionally cover required shapes and witnesses
outside the timed workload before it replaces the default.

All of this shares the single first-load allowance, including canonical compilation. There is
no slow CPU reference, offline calibration, shipped sensitivity map, or whole-model startup pass.
The language declares computation and representation semantics, not precision permissions or
claims about how a native kernel accumulates. Reordering, packing and narrower arithmetic face
the same measured rule. There are no `numerics`, `error_class` or `deviates` declarations.

## NVFP4 operands

NVFP4 weights come in two recipes, as their producer calibrated them: W4A16 multiplies the
rounded activation, and W4A4 takes the activation rows as NVFP4 too, so the same function runs
on FP4 tensor operations where a device has them and on ordinary arithmetic elsewhere. NVIDIA
names them `W4A16_NVFP4` and `NVFP4`; a GGUF of the second also stores each weight's static
activation scale `<stem>.input_scale`, which the family binder reads as the weight's recipe
(`ScaledActivations`). Running a W4A16 checkpoint as W4A4 costs it: Lightning NVFP4 (W4A16)
lost about six times its Q4 variant's KL on prose.

This is computation semantics, not a tuning choice. One NVFP4 storage has an element of each
recipe, `nvfp4g16` and `nvfp4g16a4` (likewise `nvfp4_e2m1_block16`), with the same bytes,
decode and conversions; the registry gives each element its operand recipe (`operands(W)`:
NVFP4 rows for the W4A4 elements, the rows as they are for every other), and every portable
product reads its rows through `operand_rows` (`seismic-std`'s `operand_rows.seismic`).
Natives implement the same recipe, so a W4A4 weight is W4A4 on every device, and the faster
forms tuning compares against it start from the same operand rows.

Each row x, already rounded to the activation element A, becomes q · g · s:

- g, the row's global scale, is the least power of two with 6 · 448 · g ≥ amax|x| (6 is E2M1's
  largest value, 448 UE4M3's), within F32's normal binades;
- s, each 16-value block's scale, is the UE4M3 value nearest amax|block| / (6g), ties to even;
- q, each value's code, is the E2M1 value nearest x / (g · s), ties to even, saturating at 6,
  with x's sign, and zero where s is.

Because g is a power of two, g · s and q · g · s are exact in F32 and q · g · s has at most six
significant bits, so the operand row is exact in BF16 and needs no per-row factor in a
product's epilogue. Every rounding takes the exact quotient to nearest. The portable body
writes them as F32 divisions, which round onto both grids exactly as the exact quotient does:
for a grid midpoint t and a divisor d with 2^k ≤ d < 2^(k+1), any dividend other than t · d
differs from it by at least ulp(t) · 2^k, so its quotient lies more than half an ulp of t
away from t. Devices whose division is not correctly rounded (Metal, Vulkan) decide each
rounding without one: every grid point and midpoint times its divisor is exact in F32, so
the dividend compared against it decides the exact quotient.

Rows are formed where a product's operands are staged: a GEMV threadgroup forms its rows while
staging them (one more pass for each row's largest magnitude), a staging or normalizing launch
writes formed rows into the scratch a GEMM reads, and a product row an expert's down projection
reads is formed in place. Weights that share one staged operand share a recipe; a native refuses
a binding that would mix them, and the router, stream-mix and drafter-combine projections, whose
checkpoints keep those weights in BF16, refuse W4A4 weights rather than form rows for them.
The stored `input_scale` marks the recipe and is not applied: each row's global scale is the
row's own power of two.

## Evidence and boundaries

Development evaluation measures the four relations together: scaling at fixed structure;
independent versus combined effects; transfer across file types without masking; and
candidate-versus-noise cost across structures. It must also establish that tuning inputs
represent real activations, that one draw of the yardstick is stable, and that errors
written into state remain within the model over long generation. Testing only one site, only
random noise, or only prompt logits cannot discharge these obligations.

Whole-model checks use rendered chat fixtures and sane independent references, retain the full
quality distribution, and distinguish default-relative movement from added reference loss.
Changing a KV codec has its own model-quality obligation: matching a compressed canonical kernel
does not establish that the compression itself is suitable. `inference/evaluation`
owns these measurements; empirical results and implementation status are recorded separately
from this intended design.

Native agreement cannot detect a defect shared with the default. Kernel tests and independent
semantic checks remain necessary. A rounding-scale estimate, such as √K times accumulator unit
roundoff for a reduction, is an inspection signal under its rounding and conditioning assumptions,
not a second admission threshold or a general correctness proof. Large departures can indicate
an intentional approximation or a defect; small departures do not prove correctness.

Compiler-generated alternatives require an established whole-entry numerical relation to the
portable source. Passing native tuning samples cannot grant that applicability. Compiler proof, native empirical admission
and model-quality evidence stay distinct.
