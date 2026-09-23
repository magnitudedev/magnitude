# Checked semantics

Checking gives every admitted source operation one coherent meaning and produces the entry contract consumed by construction and invocation binding.

| Input | Output |
| --- | --- |
| Source or reconstructed bundle, semantic registry, selected entry and concrete element bindings | `CheckedModule`, then a `LogicalEntry` with checked bodies, typed values/effects, invocation schema/domain and expression arena |

## One operation vocabulary

The registry defines scalar operations, representations, conversions and typed intrinsics. An operation constructor derives result types, exact symbolic meaning, shapes, bounds and effects from the same operands. Callers cannot separately assert a value expression or mark unrelated instructions as an implementation of the operation.

The interpreter and physical construction consume these checked meanings. Reference transcendental operations use the same versioned primitive recipe, including every conversion and rounding boundary. Exactness means agreement with that language recipe, not a separate claim about a correctly rounded real function.

The checked vocabulary covers primitives, elementwise maps, reductions, intrinsics, calls, allocation/fill/copy, representation conversion, views, reads/writes, atomics, conditionals, loops, checks, tuples and extents. Each admitted case has defined effects and composition. Adding a case requires extending every semantic consumer; wildcard fallbacks cannot silently erase a case.

## Entry ownership

`CheckedModule` can contain many function families. Instantiation selects the entry and element/representation bindings while retaining invocation-dependent dimensions and scalars symbolically. `LogicalEntry` owns the reachable checked structure, reference body selection, parameter/result paths and one expression arena.

The invocation domain is the source contract intersected with external target representability. Internal storage limits, optional implementation guards, tuning ranges and incomplete reasoning do not redefine it. A source content check is an operation with failure behavior; it is not automatically a promise that every caller input passes.

The entry's invocation contract derives parameter binding and dimension inference from its actual schema. Shape inference solves observed integer equations and then checks the original equations on their defined domains. It rejects inconsistent or inexact observations rather than guessing a dimension.

## Values, bounds and availability

An index parameter introduces an actual natural value. A range parameter introduces actual endpoints. `RangeStart` and `RangeEnd` project those bindings; `0 <= start <= end <= N` constrains them. It never substitutes `0` and `N` for supplied endpoints.

Expressions share one mathematical Nat/Int DAG. Each constructor computes its required bindings from its operands. Simplification preserves both value and definedness: lazy selection stays lazy, and multiplication by zero does not automatically erase an undefined operand.

Mathematical quantities and fixed-width I32/U32 scalars have different checked meanings. Dimensions,
range endpoints and iteration coordinates use exact mathematical arithmetic; Index refines that
domain. An explicitly typed word operation converts its operands before applying its wrapping,
division, comparison or shift semantics. A literal adopts its resolved operand context before the
I32 default applies. A saved I32 local remains a word even when later consumed as a quantity.
Symbolic projections describe the actual typed result, including wrapping and conversion, or are
absent; they never substitute unbounded operand algebra for a word result.

Exact host evaluation preserves arbitrary-size intermediates. Physical quantities use bound-sized
signed magnitude limbs in ordinary complete values and storage; native address/publication words
are checked projections at their consumer boundary. Finite loop capacity follows a simultaneous
bounded recurrence over actual incoming bounds and the structured body. Its zero-trip result is
the initial product without body evaluation. Capacity is separate from the actual runtime value,
and exhausted analysis resources cannot become a new source-domain restriction.

The physical [construction environment](construction.md#expression-environments) adds where these bindings are available. Semantic origin remains stable when a value is transported through a slot or ABI word.

## Boundary checking and failure

Malformed syntax, types, borrows, initialization, capability declarations and source domains are source errors. Serialized bytes are an external boundary: decoding must reconstruct through checking. A bundle containing source text is not a serialized trusted checked arena, and its identity is not permission to skip semantic construction.

Once an entry is admitted for a supported target contract, missing lowering for one of its legal source compositions is a compiler gap. It cannot be converted into a smaller invocation domain or silently routed through an interpreter.

See [language contracts](../language/overview.md) for author-visible semantics and [construction](construction.md) for the complete physical traversal.
