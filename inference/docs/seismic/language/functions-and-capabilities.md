# Functions and capabilities

A function declares a logical interface: types, shapes, ownership, effects and results. Applicable authored bodies provide implementations of that interface. Function boundaries allow composition without concealing work from the compiler.

## Reference and alternatives

The first applicable portable body defines reference behavior, including order, conversions and rounding. Additional bodies supply alternative structure. Their declaration makes them available for consideration; it does not assert numerical equivalence.

Selecting a child body is a conditional choice inside the containing entry's candidate construction. The compiler binds the actual operands, composes effects and numerical meaning, and returns the complete result product. A helper call does not receive its own independent share of the entry's error tolerance.

The initial [candidate families](../compiler/candidate-domain.md) correspond to applicable authored root bodies. Different tilings of the same body do not require different semantic families. Child alternatives and physical choices can be generated as construction reaches them.

## Typed target capabilities

A backend implementation can use declared capability families such as subgroup or matrix operations. Each operation has a typed signature defining operand interpretation, result ownership, participation, layout constraints, numerical behavior and synchronization.

Capability availability is the intersection of hardware, API/driver, toolchain and implemented backend support. A hardware name alone cannot establish it. Unknown support is distinct from known absence.

Backend-specific helpers expose their required capabilities, and callers declare the necessary set. Undeclared, wrong-backend and unused declarations are errors. Portable calls do not acquire every capability of every possible child implementation: construction filters the particular child choices against the target.

A new capability namespace is justified by a coherent semantic facility authors need to express. A new instruction variant within an existing facility generally belongs in that facility's typed signatures. Physical tiling and placement remain compiler choices where they do not alter the authored semantic structure.

Programmable backend bodies must expose their actual computation when participating in compiler selection. An opaque callback or source string cannot claim to implement an unrelated checked operation merely by attaching its identity.

## Explicit direct native execution

An authored top-level native implementation is a separate, explicitly selected API route. Its asset and launch declaration attach to the portable entry's interface; the author is responsible for implementation conformance. The generated `native_for_device` route does not acquire a compiler-derived numerical guarantee or participate in candidate evaluation.

```text
portable entry contract + authored native asset/launch
                         |
                         v
                  direct native formation
                         |
                         v
                  native binding/execution
```

This route must remain distinguishable from a [prepared kernel](../compiler/prepared-kernel.md). It is not a silent fallback for failed compiler construction. Both routes must still bind actual arguments and retain native resources through completion.

The engine's [native tuner](../../tuning.md) can search the declaration's legal configurations
under the [precision policy](../../precision.md), comparing actual execution with the canonical
native default. Portable bodies and representations supply the facts for that comparison;
declarations carry no precision permissions or trusted descriptions of native rounding behavior.
Empirical admission never grants compiler numerical applicability.


A native declaration may take its consumer forms from a projection rather than state them. A
`projection NAME for TARGET` declaration writes a backend's forms of projecting activation rows
by weights once: each form's parameter, its admission over the weights (`each`, `any`,
`agree`), its shape guards over the formals `M`, `K` and `N`, its scratch (per weight where it
needs one) and its launches, with per-variant items for the `paired` and `single` variants. An
entry's declaration takes it with one line, `project NAME VARIANT (WEIGHT, ..)`, naming its weight
ports; the formals come from the entry's signature. The line expands into the ordinary
declaration it stands for before checking, so checking, the tuner's parameter addresses and
formation see exactly what the entry would have written by hand.

### Batch invariance of row-class launches

A native may serve one row computation through several launches, one per row class (for example
a GEMV launch for one row, one for two and one for three up to a batched form). A launch that
computes each row as another launch of the native computes that row alone declares it with
`alike (LAUNCH)`: the launch is claimed to compute the row as that launch does, and the checker holds
the two to one set of reduction-order settings. Each parameter that fixes that reduction's order (an `arithmetic` parameter: a
lane-group width, a split of the reduction, a summation form) is therefore one setting of the
declaration, declared at the top and read by the `alike` launch and the launch it names; a launch
keeps as its own only what does not change a row's bits, its threadgroup mapping:

```text
params (arithmetic LANES in [16, 32, 8] default 16)
launch project_gemv when M < 2:
    params (SIMDGROUPS in [16, 8] default 16)
launch project_gemv_pair when M == 2:
    params (SIMDGROUPS in [8, 16] default 8)
    alike (project_gemv)
launch project_gemv_rows when M >= 3 and M <= VERIFY_ROWS:
    params (SIMDGROUPS in [8, 16] default 8)
    alike (project_gemv)
launch project_batch when M > VERIFY_ROWS and M <= 16:
```

The checker refuses an `alike` launch, or the launch it names, that declares an `arithmetic`
parameter of its own, or that reads a different set of the native's `arithmetic` parameters than
the other (by its declaration or, in a templated source, its code parameters). That proves the two
share their reduction orders; that the bodies compute a row to the same bits is held by the
row-equality tests, not by the declaration. A launch that only forms operands for another launch
and computes no row's result is declared `staging`, written after any `alike`, and is no row class
(a launch is not both). The tuner chooses a
shared setting together with the launches that read it, by the unit's objective over every point
they serve, so its value is the best for those row counts together rather than for one row alone.
A batched form computes rows by another reduction, so the GEMV classes reach from one row to
`VERIFY_ROWS`, the rows of the widest draft verification (owned by `verify_rows.seismic`;
every decode form, the router and the executor's decode classes read it): a verification row computes
the bits plain decode computes for it. A test holds every launch the row count selects, up to that
width, to being a row class or a declared `staging` pass; no native is exempt. (The example's `project_gemv` serves one row, `project_gemv_pair`
two and `project_gemv_rows` three up to `VERIFY_ROWS`; the launch beyond is a batched form, shown
without its body.)
