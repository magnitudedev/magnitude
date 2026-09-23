# Values and types

Types describe logical values and the permissions needed to use them. Physical layout and placement belong to the compiler, subject to the declared external representation.

## Value forms

| Form | Contract |
| --- | --- |
| Scalar | A typed value with the registry's arithmetic, conversion and exceptional-value semantics. |
| Tensor | Logical shape, element representation and a value for each initialized element. |
| `index[N]` | An actual natural index less than `N`. The bound is not the index's value. |
| `range[N]` | Actual endpoints satisfying `0 <= start <= end <= N`. It does not mean `0..N`. |
| Tuple | A complete product of typed values, preserving each component's ownership. |

Generic element bindings are fixed for a prepared entry. Invocation dimensions, range endpoints and ordinary scalar arguments can remain symbolic until binding. Shape arithmetic uses its mathematical integer or natural meaning; overflow in a native word cannot silently change a logical extent.

An exact value and a bound on that value are separate facts. For `r: range[N]`, a loop over `r` runs over the supplied endpoints. A reservation may use `N` as a capacity bound, but execution still uses `r.start` and `r.end`.

## Ownership and initialization

An owned tensor can be moved and mutated where permitted. `&tensor[...] T` grants shared reading; `&mut tensor[...] T` grants exclusive mutable access over the borrowed region. `let mut` permits mutation of a binding; it does not create ownership or write rights.

Allocation produces storage that can be written. It does not produce readable initialized elements. Fill, complete maps and writes establish initialization over their actual regions. Reads require initialization; joins and carries preserve that requirement through control flow.

A view preserves the original allocation identity, version, access rights and logical geometry. Slicing, transposition and reshaping do not independently acquire storage. An owned copy has independent storage containing the viewed value, rather than unrelated elements of its backing allocation. `to_owned` copies a borrowed value and preserves an already-owned value.

Writes change the contents and version of an existing place. They do not manufacture a new allocation identity. Aliases therefore remain connected for access checking and workflow hazards.

## Values are independent of storage decisions

The compiler represents a produced tensor as either:

- **Computed:** the actual operation, captured operand versions, logical indexing and result rounding.
- **Stored:** the logical allocation/view, initialized region, version and availability.

Both represent the same language-level value. Reading an element evaluates that realization. A consumer requiring storage materializes the same realization. It cannot reconstruct the source using whichever operand contents happen to exist later.

For example, if `y` is computed from a mutable `x`, a later write to `x` must not change `y`. The compiler orders all required old-version uses before the write or preserves the old version through materialization. This applies through calls, views and loop carries.

## Representation boundaries

Representations define decoded values and conversion/publication behavior, including packed planes and rounding. Source arithmetic retains its declared width; a wide internal address does not authorize widening an authored I32 computation. Integer, Boolean, index and range values must not be transported through lossy floating conversions.

Private storage may be direct or segmented without changing the logical tensor type. Public inputs and results obey the checked external ABI. The [physical IR](../compiler/physical-ir.md) owns storage realization; [binding](../execution/binding-and-workflows.md) connects external values to it.

