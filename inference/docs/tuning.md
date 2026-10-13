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
  cases and shapes with no performance demand. This scope does not shrink during search.
- **Performance workload:** the measurements the engine is held to, run as the load runs them:
  prefill of a 4k and of a 64k prompt (the load's context limit where it is shorter) and one
  decode step at 4k and at 64k of history, with drafting as the load drafts (the drafter's
  launches and the target's verification at each width the load prepares). The load's graph
  supplies each unit's launches for every such term, with their rows, sequences and histories;
  real weight rotations stand in where reuse would otherwise distort timing. Adversarial
  numerical cases do not count as serving traffic. A launch is demand of the kernel that serves
  it alone: where two kernels are alternatives for the same layers (the attention prefill that
  lists the history tiles its rows see, and the one that lists none), each takes only the
  launches the graph gives it. A history-dependent unit is timed at three histories: empty (all
  fixed costs), 4k, and the longest its layers see (the context limit or their window). A
  launch's intermediate history is represented by demand split between its neighboring
  points. This representative-workload approximation is explicit; it does not establish
  linear latency at unmeasured histories. Every candidate is timed at every selected point,
  including the longest history. No candidate timing is inferred from a shorter point.
  Intermediate numerical history witnesses remain required even when they carry no timing demand.

Every term counts by its relative change: tuning minimizes the weighted mean over terms of
`time / time at the defaults`. A unit's baseline contribution is the mean over
terms of the unit's fraction of the term's time at the defaults, and a point's weight within a
unit is the same for the point. For a configuration c of a unit, the objective is:

```text
service_cost(c) = Σₚ weight(p) × median_device_time(c, p)
weight(p)       = Σₜ term_weight(t) × launches(t, p) / time_at_defaults(t)
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

Baseline discovery measures every demanded unit before candidate search begins. Unknown
units share discovery by elapsed effort, with declaration order breaking ties. Incomplete
discovery is reported explicitly; it never contributes invented latency or zero service to a
candidate objective. Once discovery completes, the measured baseline term totals normalize
point weights, which remain fixed throughout the search.

Search effort follows measured remaining service. Each available piece is ranked by its
virtual finish time `(effort + price) / (share × opportunity)`, smallest first. `effort` is the
unit's completed search work, including compilation even when it overlaps device work;
`share` is its measured fraction of the baseline objective. For exploration, `opportunity`
is the incumbent's remaining cost relative to its default; for a pending conversion it is
the observed lead. This is an explicit allocation policy, not a probability of discovering a
faster configuration. Neither model identities, reference-operation counts, nor the spread of
previous winners predicts a candidate's value. A unit explores distinct forms before refining
one. Resolved promotions take priority over further exploration so established gains
reach serving before the allowance ends. Other stages impose no global barrier after discovery.

Prices use this load's observed operation costs. Unobserved operations use the conservative
admission forecast until observations exist; coverage costs are shared only within the same
entry. Every admission still protects the original deadline, numerical coverage and publication
obligations. A chosen kernel must establish a lead through matched measurements of the full
performance cohort and satisfy its complete numerical scope.

Unknown work is admitted at a bound, never a guessed candidate price. Confirmation,
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
to a second budget. A compile that outruns its forecast does not consume the conclusion reserve:
when the reserve is reached its piece is abandoned unmeasured, choices publish while it finishes,
and the abandoned work is reported. A promoted choice's serving artifact is formed as the choice
is promoted, inside that work's own admission, so the conclusion holds no artifact compiles,
whose cost a system shader cache makes unpredictable. A compile cannot be interrupted, so it
starts only if it would end within the deadline at the longest compile the load has observed,
including one deliberately compiled with nothing to serve it. A slower device explores less, not longer. No later request, changed traffic
mix or cached incomplete search acquires another tuning allowance.

## Search and finalization

1. **Establish the default.** Form the canonical configuration, discover an affordable workload
   cohort, and measure its service. Retain the inputs and default outputs needed for comparisons.
   The default supplies the numerical recipe without self-qualification. Missing discovery can
   remain incomplete, but candidates cannot be ranked on different workload cohorts.
2. **Cover different forms.** Within a unit, explore representatives of structurally distinct
   implementations before refining one form. Across units, measured service and allocated effort decide: a high-share unit's
   refinement can precede a low-share unit's forms. Domain
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
