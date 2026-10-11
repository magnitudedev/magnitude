# Native kernel tuning

The native tuner chooses the fastest evidenced legal configuration for each served kernel unit
within one first-load allowance. It optimizes the real serving workload while preserving the
unit's complete numerical scope. It seeks useful improvements before the deadline, not proof
that the entire search space has been exhausted.

The [precision system](precision.md) defines numerical admission. This document defines how
search earns, compares, retains and reuses its evidence. It concerns explicitly authored native
implementations; compiler candidate evaluation and numerical applicability remain separate.

## Responsibilities and units

| Owner | Responsibility |
| --- | --- |
| Checked kernel contract | Computation, element constraints, static dimensions, legal configurations and launch structure |
| Engine | Served units, workload demand, numerical witnesses, input construction, the load's allowance and persistent results |
| Seismic runtime | Native formation, candidate generation, execution, numerical comparison, measurements and pending-work forecasts |
| Evaluation | Independent correctness, whole-model quality, tuning effectiveness and performance against controls |

A unit identifies an entry with its bound representations, static geometry and served workload
class. The same entry can have distinct units for disjoint graph classes. Repeated logical calls
contribute their actual invocation demand; sharing weights does not remove calls or manufacture
independent memory streams. A selected configuration is a whole artifact, including all launches
and coupled parameter values. The serving graph binds that exact choice to its class.

Two descriptions of a unit must remain distinct:

- **Numerical scope:** every required shape and witness for its served contract, including edge
  cases and shapes with no estimated performance demand. This scope does not shrink during search.
- **Performance workload:** representative invocations and their demand, including real weight
  rotations where reuse would otherwise distort timing. Adversarial numerical cases do not
  automatically count as additional serving traffic.

For a configuration c, the objective is:

```text
service_cost(c) = Σₚ invocation_demand(p) × median_device_time(c, p)
```

Candidates are compared on the same workload, normalized by the canonical default's service
cost. A fast result at one shape cannot absorb the demand of an unmeasured shape. Unknown cost
remains unknown; it is neither zero nor an invitation to invent a forecast. Numerical eligibility
is a hard constraint, never a term that faster execution can compensate for.

## One first-load clock

The serving allowance is 60 seconds for the load's tuning work. Evaluation may select an explicit
budget. It covers required canonical formation, input construction, default observations,
yardsticks, search planning, candidate formation, execution, comparisons, confirmation, final
coverage and result publication. No work moves before the clock or after finalization to make
the total appear smaller. Loading and readiness are reported separately from this tuning total.

Counting units and recognizing stored choices is metadata-only. Required canonical artifacts
are established inside the clock before optional work can consume their capacity. A unit with
no performance demand still obtains its canonical artifact, but does not build numerical inputs
or open a search merely to manufacture evidence for that default.

Time allocation follows measured serving importance and observed work costs. Runtime forecasts
include the next operation and any extra obligation it creates for concluding a valid choice.
The allocator admits optional work only when that work and all outstanding obligations fit the
remaining deadline. Already completed formation or coverage is not charged again.

Unknown work receives bounded discovery time, not a guessed candidate price. Confirmation,
winner coverage and publication have protected reserves. A reserve is part of the original
allowance, not a grace period; the same coverage obligation is not reserved once per rival.
Deadline checks also apply to domain traversal, input building and the intervals between device
submissions. Noninterruptible work has forecast uncertainty: an actual overrun is a failed
budget result, not permission to extend the cap or conceal time.

Memory is admitted as time is. A search runs under the room the platform let the engine claim when
its device was opened, its observed availability above the planning reserve, so a search explores
less when the host holds more. Work starts only when its forecast bytes fit what is free. An input
build the device's enforced limit refuses anyway did not fit its room; it is neither the unit's
failure nor the preparation's. Its point is priced at what the build demanded and is built again
only once that much is free; otherwise its unit stops for want of room with its terminal choice
recorded, and the other units continue.

Independent compilation can overlap device measurement, with resources and pending work charged
to the same load. Elapsed wall time is the budget authority; overlapping phase times do not sum
to a second budget. A slower device explores less, not longer. No later request, changed traffic
mix or cached incomplete search acquires another tuning allowance.

## Search and finalization

1. **Establish the default.** Form the canonical configuration, discover an affordable workload
   cohort, and measure its service. Retain the inputs and default outputs needed for comparisons.
   The default supplies the numerical recipe without self-qualification. Missing discovery can
   remain incomplete, but candidates cannot be ranked on different workload cohorts.
2. **Cover different forms.** Explore representatives of structurally distinct implementations
   before spending the allowance refining one form. High-share units receive priority. Domain
   traversal remains lazy and bounded; a large Cartesian product is not materialized just to
   identify the useful structural choices.
3. **Measure and admit together.** The candidate's first initialized execution produces its
   numerical observations and timing. Compare on device with the retained default and yardstick;
   reject a failing candidate before further performance sampling. Reuse inputs and exact
   observations instead of repeating a separate qualification pass per candidate.
4. **Refine admissible contenders.** Explore the useful neighborhood while preserving access to
   distinct legal forms. Factor launch choices only where their dependencies allow it. An
   assembled combination is a new whole contender that must be formed, measured and checked
   during search; passing its pieces independently does not admit the combination.
5. **Confirm and cover the leader.** Rank retained whole artifacts using a common measured
   objective and confirmation policy. Before selection, check the leader on every remaining
   required witness outside its timed coverage. If it fails, a retained alternative may be
   checked only while the remaining allowance permits. Expired confirmation may leave earlier
   timing usable, but never supplies missing numerical evidence.
6. **Publish a terminal choice.** Serve the fastest retained admissible artifact with complete
   required coverage, or the actual canonical default. Finalization creates no unseen
   specialization. Preserve the chosen artifact and its evidence through serving preparation,
   so selection does not trigger duplicate compilation or bind another representation's kernel.

If a composed alternative depends on other tuned choices, those prerequisites must conclude
first. Its comparison and retained identity use the finalized prerequisites, not provisional
census defaults that may subsequently change. Dependency ordering shares the same allowance.

The canonical default is a distinguished legal outcome when optional search or coverage cannot
finish. It is recorded as a default, with any unmeasured scope still explicit. A nondefault
artifact never becomes admissible merely because time expired. Actual failure to form or execute
the canonical implementation is a preparation failure, not an implicit alternative reference.

## Measurements that mean what they claim

Performance samples measure completed device work. Input initialization, numerical reductions,
readback and reference execution stay outside the candidate's device-time sample while remaining
inside the load's wall-clock allowance. Cold formation, first-use effects and warmup do not become
steady-state speed evidence. Confirmation compares contenders under matched conditions and must
establish a gain beyond measurement noise before replacing the default.

Numerical comparison observes every returned subject and affected writable-state region from
identical initialized inputs. Reused outputs start cleared so a missing write cannot inherit a
previous candidate's correct value. Private tuning state never aliases live serving state.
Witnesses also cover writes outside declared regions. Repeated performance batches follow the
same mutable-state rules for every contender; they do not replace the initialized numerical
observation.

Evidence belongs to an exact artifact and witness. Equal timing keys, matching display names,
related parameter values or passing individual launches do not establish numerical equivalence.
Serving legality also includes dimensions and resource limits that a timing representative may
not exercise. Zero performance demand cannot waive those requirements.

## Retention and inspection

Keep measurements for every measured candidate, not only the winner: configurations, point
identity, timing, error ratio, numerical verdict and completed coverage.
Changing α can reselect from retained measurements without execution, but only a candidate with
complete matching evidence for the new selection may serve. Recording a fast partly checked
candidate does not authorize a future coverage pass outside the original allowance.

Measurement identity covers the model data and input-generation contract, checked computation,
bound representations and statics, served scope, native source and artifact, device/toolchain,
and comparison definition. Selection also records its policy and tuning budget. Resident artifact
handoff preserves those same distinctions. A digest or abbreviated label cannot substitute for
the exact identity it names. Compilation caches and tuning-result caches have different purposes:
reusable compiled code grants neither numerical evidence nor a new search budget.

Every normally concluded unit retains a terminal choice, including a default selected because
time ran out. A later load consumes that result without new input construction, timing or
qualification. A same-identity result that violates its served contract is an error, not an
excuse to reopen search. Explicit fresh-tuning evaluation can ignore selections while reusing
compiled artifacts; it must record that condition when making comparisons.

The tuning trace explains each decision in one self-contained unit record: its scope and demand,
measured candidates, numerical yardsticks and their derivation, chosen configuration, reason,
coverage and stop condition. It also accounts for the whole load's elapsed time and phases.
Unmeasured quantities remain unavailable, never zero or fabricated evidence. Precision surveys
and pinned-selection probes consume this same evidence through the evaluation system.

A successful tuning run demonstrates legal, evidenced selection within its allowance. An upgrade
also requires matched tuned-versus-default speed and independent model-quality checks, plus
comparison with the baseline build at equal budgets and serving conditions. A faster kernel
alone does not establish faster serving; a locally admitted kernel alone does not establish
unchanged generation quality or drafting acceptance.
