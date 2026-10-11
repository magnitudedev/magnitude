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

