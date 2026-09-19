# Seismic language

Seismic is a language for **authored logical execution structure**. A kernel author fixes the algorithm, the independent work domains, producer and state scopes, stage order and combination order. The compiler selects among the authored implementations at each static call, selects contiguous fusion groups, and selects numerical dimensions (slice widths, window widths, partition counts). It never invents structure. The governing specifications are `specs/26-09-18/seismic-structured-authoring-spec.md` and `specs/26-09-19/seismic-language-simplification-spec.md`; this document is the concrete surface and its rules.

Every source file is named `<name>.seismic`. Filenames organize source but grant no target capabilities. Target restrictions follow from declarations and transitive dependencies (section 9). Indentation is significant (spaces only), `#` starts a comment, brackets suppress line breaks.

Structural words are reserved everywhere and cannot be shadowed: `fn lower admit where for in if else and or not true false inf tile void let mut parallel ordered pipeline stage yield return merge publish`. The words `tensor view index out inout alias to identity` are contextual: ordinary names recognized only by position.

## 1. Declarations

```text
[admit] fn name[Shape, ...](params) [-> result] [for target] [where predicate]:
    body
lower name[Shape, ...](params) [-> result] for target [where predicate]:
    body
```

Every `fn` has a body. A `fn` without `for target` is portable. A `fn ... for target` is a target-specific helper and may use that target's facilities. A `lower` is an additional implementation of an existing portable function family for exactly one target; it always has a body and a complete signature.

`where` and `for target` may start on an indented continuation line; the body then continues at that indentation:

```text
fn row_dot[N](x: tensor[N] f32, w: tensor[N] f32) -> f32
    where N >= 1:
    let mut result = f32(0.0)
    for i in 0..N:
        result = fma(x[i], w[i], result)
    return result
```

- **Implementation families.** Same-name portable definitions with compatible parameter structure are implementation alternatives wherever more than one applies. Disjoint definitions are legal. Overlapping definitions and lowerings must agree on result, modes, effects, ownership and numerical contract.
- **Contract family.** A connected component of same-name overloads with overlapping applicability.
- **Candidate selection.** For target `T`, a static call's candidates are all applicable portable bodies together with all applicable lowerings declared for `T`. Target lowerings do not hide or outrank portable bodies. Declaration order and predicate specificity do not choose a winner; the solver chooses among every complete candidate.
- **Target coverage.** A lowering adds candidates only to its named target and creates no obligation for other targets. A portable body remains usable on every target that supports its complete dependency closure. Compilation fails only when no complete applicable implementation exists.
- **Entry points.** Any linked function may be requested as a compilation root by the embedding API, manifest, CLI or test harness. Entry-point status is not source syntax. A name-only request must identify exactly one family; disjoint same-name families are an ambiguity error rather than a declaration-order choice.
- **Target-specific functions.** A `fn ... for target` is available only to functions and lowerings for that target. It is not a lowering and cannot itself have lowerings.
- **`admit`** marks a library-admitted numerical contract (section 8).
- **Shape parameters** in `[...]` are semantic extents (`i32`, positive unless a `where` says otherwise). **Element parameters** are implicit: a capitalized name in element position that is not a dtype or representation (`T`, `A`, `GW`).
- **`where` predicates**: conjunctions (`and`) of comparisons, `%` divisibility and equalities over shape parameters and integer literals, plus `full(X)` (R1 below). Nothing else.

### Parameters and modes

```text
x: tensor[M, K] bf16          # read-only (default)
out y: tensor[M, N] bf16      # written; every element must be published
inout acc: tile[M, N] f32     # read and written
eps: f32                      # scalar
pos: index[T]                 # i32 with 0 <= pos < T
```

`out` and `inout` are legal on tensor, view and tile parameters. Distinct parameters do not alias unless the declaration says `alias(a, b)` after the parameter list. Entry bindings are checked against this.

## 2. Types

| Kind | Spelling | Notes |
| --- | --- | --- |
| Scalar | `f32 bf16 f16 i32 u32 bool` | Distinct float dtypes widen to `f32`; int/float or signed/unsigned mixing needs an explicit cast `f32(x)`. |
| Bounded index | `index[N]` | An `i32` with a bound; arithmetic drops the refinement. |
| Tensor reference | `tensor[shape] elem` | External storage. `elem` is a dtype, a representation (`q4g64 q4g32 q4k q5k q6k q8g32s iq4g32 q8g32`) or an element parameter. |
| View | `view[shape] elem` | Borrowed selection of a tensor or tile. Indexing with at least one non-point axis yields a view. |
| Tile | `tile[shape] elem` | Owned logical block. `let` creates an immutable binding; `let mut` permits mutation. |
| Domain | `lo..hi` | Half-open semantic range. Not data. |
| Slice | (introduced by a region binder) | Opaque contiguous subset of a domain. |
| Region result | (result of a region expression) | One yielded value per slice visit, keeping the slice correspondence. |
| Tuple | `(a, b)` | Fixed product; destructure explicitly. |
| `void` | | No value. |
| Native | target-defined (`metal.simdgroup_matrix(f32)`) | Target code only. |

An axis extent is **semantic** (a shape expression, numerically usable) or **structural** (the width of a slice; never a number in portable code). `x[:, cols]` with `x: tensor[M, N]` has axes `(M, cols)`. `extent(v, axis)` is legal only on semantic axes. Two slices are interchangeable only if they are the same binder or alias; equal tuned widths mean nothing. Two axes are identical when they are the same slice or provably equal semantic extents.

## 3. Regions

```text
parallel [cols] in 0..N:                      # independent owners, one slice each
ordered [k] in 0..K:                          # visits in order; each completes before the next
pipeline [k] in 0..K:                         # ordered visits through a fixed stage chain
parallel [rows, cols] in (0..M, 0..N):        # rectangular product; last axis fastest for ordered
ordered [inner] in cols:                      # explicit refinement of an enclosing slice
parallel [p] in partials:                     # revisit an earlier region result with its own slices
```

- The author never writes a width. Each static binder is one numerical site; all its dynamic instances share it. Tails keep their logical bounds.
- A binder used on several tensors (`x[:, k]`, `gate[cols, k]`) is one joint traversal.
- Lexical position fixes reuse: a `let` before an inner region is produced once per enclosing visit; inside, once per inner visit. The compiler never moves, duplicates or merges producers.
- `parallel` bodies cannot mutate enclosing state; they may `publish` to provably disjoint views. `ordered`/`pipeline` bodies may update enclosing `let mut` state.
- Slices are opaque: no width/start/ordinal queries, arithmetic, comparison, indexing (`s[0]`), construction, or escape beyond the binder's scope. They may be aliased with `let`, passed to helpers, and used as indices.

### Element and coordinate loops

```text
for i, j in owned(t):         # every valid coordinate of tile t (all axes)
for k in axis(t, 1):          # valid coordinates of one axis of a tile or view
for h in heads:               # semantic coordinates of a slice, ascending
for i in 0..C - 1:            # ordinary semantic range
```

`owned`/`axis` binders are *tile coordinates*: they index any tile or view sharing that axis identity. `coord(i)` gives the semantic `i32` coordinate. `for h in slice` binds the semantic coordinate directly (an `index` of the parent domain), so `h % NK`, `qkv[NK + h]` are ordinary arithmetic on the problem's coordinates; nothing about the tuned partition is observable. These loops are sequential scalar control inside the current owner.

### Region results

```text
let partials = parallel [p] in 0..N:
    let v = f32(x[p])
    yield reduce(v * v, 0, sum)

let mut total = f32(0.0)
ordered [p] in partials:
    total = total + partials[p]
```

A `parallel` or `ordered` region used as an expression returns one yielded value (or fixed tuple) per visit. Consumers iterate `in results` with any binder name and select `results[p]`; they reuse the producer's partition (no new site, no rerun). Every path through the body yields exactly once with one schema. Results are immutable, may nest, may cross stage ports and helper calls, and cannot be flattened, counted, indexed by number, appended to, or escape the compiled composition. `pipeline` is never a result expression.

### Merge

```text
let total = parallel [part] in 0..K:
    yield reduce(f32(x[part]), 0, sum)
merge (left, right) identity f32(0.0):
    yield left + right
```

Canonical near-equal contiguous partition, adjacent pairs combined level by level, odd value forwarded. Empty domain gives `identity`; one part gives its partial. The partition count is the numerical site. Requires an admitted contract for the combination (section 8).

### Stages

```text
stage prepare:
    let a = f32(x[:, k])
    yield a
stage accumulate(a):
    matmul(a, w, into=acc)
```

Consecutive `stage` statements form one linear chain. A stage's parameter list binds the previous stage's `yield` positionally. Other stages' locals are not visible; enclosing immutable bindings and state are. Outside a pipeline, a stage and all its descendant work complete before the next starts, at the scope of the enclosing owner (invocation, one parallel visit, one ordered visit). Inside a `pipeline`, the body consists only of stages; each state object has exactly one updating stage; preparation may run ahead only over stable reads. The terminal stage of a chain that is the body of a result-producing visit supplies that visit's result.

## 4. Statements

```text
let a = expr                  let a, b = f(x)          let (m, l, acc) = partials[p]
let mut s = f32(0.0)          let mut (m, l) = (f32(-inf), f32(0.0))
s = s + v                     t[i, j] = v              (m, l) = (m2, l * a + r)     # tuple assign reads old values first
publish bf16(y) to out[:, cols]
yield a, b                    return a, b
if cond: … else: …            f(a, b, into=acc)
```

`let` bindings cannot be reassigned or mutated through. `let mut` explicitly permits local mutation or binds a mutable view capability, subject to ownership, aliasing, ordering and initialization rules. A mutable view cannot manufacture write access: its backing root must already be `out`/`inout` or mutable local state. Carried state is not a separate declaration category: a `let mut` declared outside an `ordered` or `pipeline` region and updated across visits is carried state by lexical scope and use.

Assigning a tile element (`t[i, j] = v`) or a whole tile rounds the value to the tile's element type.

`publish` is the only write to tensor storage. It evaluates the value, rounds it to the destination's element type (that rounding is part of the publication and survives every fusion), and writes the destination view, whose domain must match. Publishing into a packed representation needs an explicit encode operation. Writing requires an `out`/`inout` parameter.

## 5. Expressions

- Literals, names, `-inf`/`inf`, arithmetic `+ - * / %`, comparisons, `and or not`, bit ops `& | ^ ~ << >>`. A shift amount may be `i32` or `u32`.
- Casts `f32(e) bf16(e) f16(e) i32(e) u32(e)`: on scalars, tiles and views (a view cast reads and converts, yielding a tile). `f32(v)` of a packed view or tile decodes it. Reading one element of a packed value yields its decoded `f32`.
- **Tile arithmetic is elementwise** over identical axes, with scalar broadcast. Comparisons yield a `bool` tile (a mask). `select(mask, a, b)`. No implicit reduction, no general broadcasting.
- `load(view)`: value snapshot in the view's representation. `decode(view)`: dense `f32` tile of a packed view. `zeros_like(v, dtype=f32)`, `ones_like`. `tile[shape] elem`: uninitialized local tile; every element must be assigned before it is read, yielded or published.
- Indexing `t[a, b]`: point (scalar or index expression), slice binder, `:` (whole axis), `lo:hi` semantic range, tile coordinate. All points gives an element; otherwise a view. `t.T` transposes a rank-2 tile or view. `reshape(v, shape)` where element correspondence is provable.
- A range `lo:hi` whose bounds are data-dependent is clamped to the axis at run time. Its extent is a runtime value: semantic, numerically usable, never a site. A helper may receive it as a runtime-valued shape parameter; an implementation whose `where` depends on such a parameter is inapplicable there. A data-dependent point index is bounds-checked at run time. Two ranges are the same runtime window when their bounds are the same values: bind a data-dependent bound once (`let lo = visible[row, 0]`) and use the binding in every range over it, so tiles over those windows share one extent and coordinates of one are provably coordinates of the other.
- `reduce(tile, axis, sum|max|min|argmax)`: reduction along one axis of a dense tile. Over a semantic axis it accumulates in `f32` in ascending index order and is a complete value. Over a structural axis the result carries a partial obligation (section 8). `reduce(tile, axis, sum|max|min, unordered=true)` permits reassociation (a target may combine lane partials); it is legal only inside an `admit fn`, never for `argmax`, and the reference interpreter still accumulates in ascending order. A library offers such a permission as a separately named admitted contract with an ordered body and an unordered body (for example `sum_any_order`, `matmul_any_order`); callers opt in by calling it, and an authored alternative body of the caller keeps the exact contract selectable.
- Math: `fma exp exp_fast rsqrt sqrt log sin cos abs max min`. On tiles they apply elementwise over identical axes, with scalar broadcast.
- Calls `f(args, name=value)`; `f[R = 64](args)` binds a shape parameter the arguments do not determine. Named arguments bind declared parameters (`into=acc`).
- `coord(i)`, `extent(v, axis)`.

## 6. Target-dependent code

A portable body may use only portable forms, types and dependencies. A target-specific `fn` or `lower` body may name that target's primitives: `metal.`-qualified intrinsics and native types (`metal.simdgroup_matrix(f32)`, `metal.simdgroup_load`, `metal.simdgroup_multiply_accumulate`, …), packed accessors (`t.words`, `t.scale`, `t.bias`), participant indices and `atomic(op, place, value)`. Target code additionally has **geometry authority**: it may use `capacity(t, axis)` and `valid(t, axis)` of structural axes, tile-coordinate arithmetic, and fixed atom sizes for loop bounds, addresses and masks. Such values never flow into portable semantic parameters, public shapes or RNG identities.

A target-specific function may call portable functions and functions for the same target. A lowering may call portable functions and same-target functions. Portable functions may call other portable families whose implementations are selected recursively, but cannot directly call a target-specific function. Cross-target calls are rejected. These capabilities come from declaration context and the complete dependency closure, never from a filename.

Not yet supported in structured target code: `lanes` loops (participant distribution belongs to the selected mapping; target code reads its participant through the target's index intrinsic) and `atomic(max|min, …)`. `atomic(add, …)` is checked but has no instantiation yet.

**Tails (R1).** A `where` predicate on a parameter axis bound to a structural extent constrains the slice *capacity*. A selected body must be correct for every visit whose valid extent is at most the capacity; target lowerings author their own tail path. `where full(M)` additionally requires that the capacity divides the semantic extent.

## 7. Execution units and fusion

Units are formed per authored block; a statement calling a helper is one unit and the helper's body has its own units. Within a block, every single-consumer pure tile-valued `let` adjacent to its consumer is folded into it. The remaining statements of a block are its execution units: tile-valued bindings, state updates, calls, `publish`, and nested regions. The solver may group *contiguous* units for which the target supplies a legal fused realization. It never reorders, skips, duplicates or splits. Therefore: statement order of independent work is author-significant; naming a single-use expression is not; a multi-consumer `let` is one producer in every grouping.

## 8. Numerical contracts

Operation meaning (casts, FMA, accumulation dtype, publication rounding, reduction order) is exactly what the body says. Tuning may not change it. A value yielded from a region, or reduced over a structural axis, is **partial**: outside an `admit fn` it may only be forwarded, stored in results, combined by a `merge` clause, accumulated into `let mut` state by `+`, `max`, `min` inside a traversal of the same result, or passed to an `admit fn`. Inside an `admit fn` it is an ordinary value; the library author vouches that the combination is partition-independent under the intended numerical permission. Admission is a trust boundary, not a proof; inspection lists every admitted function on a selected path.

Ordered accumulation into carried state across `ordered`/`pipeline` windows (for example `matmul(a, b, into=acc)`) preserves the element order of the whole traversal and needs no admission.

## 9. Implementation and source rules

For a static call compiled for target `T`, family construction unions applicable portable `fn` bodies with applicable `lower` bodies declared for `T`, then recursively checks each candidate's dependencies. With one complete candidate the compiler selects it directly; with several it exports a categorical choice to the joint solver; with none it reports the missing specialization and dependency reason. A helper call is never a launch, materialization or completion boundary by itself.

All sources end in exactly `.seismic`. A file may freely contain portable functions, target-specific functions and lowerings for several targets. Paths provide source identity and diagnostics only; they never establish portability or target authority.
