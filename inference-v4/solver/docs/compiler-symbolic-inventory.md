# Compiler-derived symbolic and boundary inventory

This is read-only source tracing for the solver's mathematical vocabulary. It does not migrate selection or claim that synthetic objective units model a device. The solver has no dependency on the source crates referenced here.

## Initial unresolved family: work mapping and dispatch geometry

The actual Metal retained choice in `seismic-metal/src/tuning.rs::decomposition` offers every axis step in `1..=extent.max(1)` through `IntegerRange<MappingDecision>`. No step is preselected or omitted. `seismic-realization/src/dispatch.rs::WorkMapping::new` determines `axis.extent = ceil(logical_extent / step)`, row-major stride products and total work-item product. Its `extents` method computes each tail as `min(step, logical_extent - base)`. `GroupDispatch::new` then sets `threads_per_group = lanes_per_item * items_per_group` and `groups = ceil(work_items / items_per_group)`; dispatched lanes are their product. These are shared emission/accounting declarations, not a companion performance model.

The solver directly supports the unresolved integer step variables and bounded nonnegative `Product`, `CeilDiv`, `Minimum`, `Maximum`, and `DivRem` relations needed to preserve this algebra. Affine equalities are pairs of `LinearLe` relations. `DivRem` exposes both full-piece count and remainder without enumerating widths. Domains stay compact while logical extents grow. Derived variables are finite model variables, so a fixed assignment independently reconstructs the same integer formulas. This preserves symbolic family geometry; it does not eliminate the need for complete compiler-owned legality and objective translation.

`TileDeclaration::layout` adds `max(1, ceil(capacity / lanes_per_item))` for distributed private arrays and `max(1, capacity) * bytes_per_element * items_per_group` for shared storage. These require the same typed relations. Shared storage limits couple grouping and placement and must remain in one constraint relation; local isolated costs cannot decide them.

The required boundary is the full selected geometry: logical extents, steps, work counts, participant widths, group widths, representation/placement and storage resource interactions. Fixing only work count is insufficient because equal work counts may have different tails and declared storage. The initial solver leaves unresolved boundary variables in joint search. It does not identify all widths with the same quotient as equivalent.

## Partition and preparation families

`seismic-lang/src/partition.rs::pointwise` computes `ceil(width/piece)` iterations, `width % piece` tail size, and a distinct guarded tail body. `seismic-metal/src/tuning.rs::PartitionChoices` also retains the unpartitioned alternative, so a correct future export must keep that separate topology choice. A partition model consisting only of the resulting loop count would omit the tail's semantic effects.

Constructive physical alternatives may represent small admissible widths plus an aligned progression. Packet forms require alignment and tail conditions derived from packet groups. This justifies union-of-progression domains and exact remainder relations, rather than a dense table over every width. Physical elaboration remains the authority for constructing the exact legal domain: the solver must not replace packet-specific legality with a guessed divisibility rule.

`seismic-realization/src/execution.rs::Multiplicity` and `seismic-accounting/src/multiplicity.rs::evaluate` retain products, plus-one, iteration counts and activity predicates. Constant-product/reified arithmetic therefore has a direct symbolic source. This alone does not justify collapsing coupled temporal repetitions to scalar multiplication; those interfaces are covered by the scheduling/repetition evidence separately.

## Bounds and validation obligations

`seismic-metal/src/tuning/bounds.rs::Dispatch::derive/demand` already derives ceil-division and product consequences for unresolved mapping intervals. The new mathematical relations allow analogous consequences to remain available before selecting a complete kernel. No timing coefficient has been copied or invented here. Mandatory-publication demand alone is not a full operation-count model.

Each symbolic relation has an exact integer meaning and interval propagation checked against independent complete assignment loops on tiny domains. Products use i128 intermediates. Ceiling division requires positive denominators; zero numerators give zero quotients. Remainders are nonnegative and strictly less than the divisor. Model domains remain i64, so an arithmetically larger result has no feasible result assignment; it is never truncated into the domain. Objective arithmetic still reports overflow as a typed error.

Source-derived geometry can be validated without a compiler dependency by recording the source formula and checking every small admitted choice against independently evaluated formulas. Full compiler-to-solver family correspondence still requires an integration test consuming actual retained IR. The current library and abstract lab do not establish that obligation, native reconstruction, controllability of hardware scheduling, or practical compile latency.
