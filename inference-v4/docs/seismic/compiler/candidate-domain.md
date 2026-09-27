# Candidate domain

`CandidateDomain` owns the complete definition of permitted implementations of one `LogicalEntry` for a target and precision policy. It owns structured candidate coordinates, dependent physical and emission choices, and cached construction results. The evaluator owns exploration strategy and performance decisions.

| Input | Output |
| --- | --- |
| Entry, immutable target/deployment, precision policy | Family definitions, candidate coordinates and transformations, general reference candidate and resumable construction |

`RefinementSession`, where retained, is private construction state owned by this domain. It does not become another mutable domain or public planning service.

## Families and coordinates

Applicable authored root bodies determine the initial family cases. Construction reaches child calls and physical choices conditionally. The number of families is therefore distinct from the number of physical implementations; a new tiling does not need a new root family.

A `CandidateFamily` is an immutable construction definition referencing the existing checked body and implemented realization rules. It is neither a partially built candidate nor a closed executable. The domain separately owns continuations and cached physical results. A parameterized IR may cache work shared by several coordinates without changing the meaning of family or copying the source program into another rule language.

```text
one LogicalEntry
    |
    +-- family: reference root body
    |      +-- child body choices
    |      +-- work mapping / storage / communication / control choices
    |      +-- emission alternatives preserving those commitments
    |             |
    |             +-- complete coordinate -> candidate physical execution
    |             +-- another coordinate -> another execution
    |
    +-- family: another applicable authored root body
           +-- its conditional construction choices
```

A full coordinate fixes all active body, physical and emission choices for a whole-entry implementation. Invocation dimensions, ranges and content-dependent control can remain symbolic. Inactive and derived decisions are absent. Stable coordinate identity uses the structural choice path and normalized value, not allocation order, a factory name or an incidental arena ID. Its immutable selected content survives eviction of physical IR and contains no native handles or fitness data.
Completed construction paths retain their own checked family data even if a structural digest matches another path. Native preparation may share an artifact only through an exact resident request or a separately proved equivalence, never by treating the digest as collision-free equality.

The general reference implementation is a designated fully resolved candidate from reference construction. A family label or `universal` marker does not establish its coverage.

## Transformations and navigation

The domain constructs new coordinates by changing a choice, regenerating a region, or combining compatible selections from existing candidates. Changes can introduce or remove decision sites. The same constructors rebuild affected dependencies, including consumers whose layouts, participation or protocols change. Existing candidates remain immutable; there is no independently editable genome beside the physical IR.

`CandidateNavigator` supplies shared traversal mechanics: local alternatives, coordinated region changes, compatible recombination and fresh systematic exploration. It uses domain-owned continuations and returns constructed alternatives or pending work. Evaluators choose parents, operations and effort; performance ranking and retention remain evaluator responsibilities. Target legality stays in the domain's constructors.

## Complete, minimal choice definition

Domains represent integer intervals, finite typed choices, maps/partitions and recursively constructed execution graphs. They do not define all possibilities by a few workgroup sizes or a fixed list of existing factories.

Coverage requires construction rules for every permitted source-preserving physical form in the declared target scope: producer sharing/recomputation, work partition and participant assignment, storage/representation, communication, synchronization, control and resource lifecycle. A reverse argument must show how any permitted native physical contract is represented. An exhaustive match over the rules alone does not establish that no rule is missing.

Each choice must affect a distinct physical or permitted emission freedom. Layouts, resources and bindings determined by the chosen structure are derived, not independently selected. Constructors normalize established equal contracts; exact content equality resolves hash collisions. Equal computed results do not establish equivalent emission requests or native performance. Physical-contract and emission equivalence require their own arguments; unknown aliases remain represented.

## Effort does not define membership

The domain definition is immutable. Requests may return an established applicability region, an established exclusion, or pending construction/analysis. A candidate applicable on one resolved region may be exposed there while analysis of the remainder continues. The owning construction creates any later expanded view; callers cannot attach raw guards.

Shared navigation maintains resumable traversal under evaluator-supplied effort. A fair exhaustive traversal enumerates finite construction descriptions by increasing length and stable tie order, interleaved with heuristic proposals. Pending analyses are dovetailed so one hard member does not starve every later member. Effort expiration preserves continuations and the general incumbent; it is not cached as exclusion. Resident caches may evict reconstructible candidates without removing them from the domain or ending exploration.

Where actual deployment bounds make the entire state/controller space finite, complete traversal may terminate. Otherwise the design promises fair coverage of finite constructible descriptions, not a finite candidate count or a terminating solution to arbitrary controller equivalence. Arbitrary graph-size caps cannot masquerade as target limits.

## General preparation and failures

The deterministic [general construction](construction.md#reference-completeness) does not wait for exhaustive optimization or alternative-body numerical reasoning. Search ranges direct effort and do not narrow its invocation coverage.

An unsupported optional capability excludes that choice. A demonstrated policy violation excludes its affected applicability region. An unfinished analysis remains pending. Missing reference construction, invalid scoped values or contradictory target behavior are compiler defects. Actual service/resource failures are reported separately.

Reports distinguish represented choices, constructed members, applicable regions, observed candidates and retained candidates. None of those counts can substitute for the others.
