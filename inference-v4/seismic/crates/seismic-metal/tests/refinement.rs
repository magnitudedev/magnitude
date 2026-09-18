mod support;

use seismic_accounting::{schedule, selection, workload};
use seismic_compiler::tuner::{self, Backend as _, Preparation};
use seismic_lang::{
    Scope,
    ir::LoadMode,
    lowered_ir::LoweredIr,
    program::{SourceFile, compile},
};
use seismic_metal::{
    execution::{self, Config, Execution},
    family::{GroupFamily, GroupingChoices},
    model, msl, tuning,
};
use seismic_realization::{LoadStrategy, dispatch::TilePlacement};
use std::{cell::Cell, sync::Arc};

fn oracle_model(model: schedule::evaluation::Model) -> schedule::Model {
    match model {
        schedule::evaluation::Model::Flat(model) => model,
        schedule::evaluation::Model::Structured { model, expansion_limit } => model.expand(expansion_limit).unwrap(),
    }
}

const TWO: &str = "fn evaluate(x: tensor[5,64] f32, middle: tensor[5,64] f32, out: tensor[5] f32):\n  for row in parallel:\n    a = load(x[row])\n    y = tile[64] f32\n    for i in owned(y): y[i] = a[(i+1)%64]\n    store(y,middle[row])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row,0]\n    store(y,out[row:row+1])\n";
const SPLIT: &str = "fn evaluate(x: tensor[2,65] f32, middle: tensor[2] f32, out: tensor[2] f32):\n  for row in parallel:\n    acc = tile[1] f32\n    for i in owned(acc): acc[i] = 0.0\n    chunk = load(x[row,0:65])\n    acc[0] += reduce(chunk,0,sum)\n    store(acc,middle[row:row+1])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row] * 2.0 + 1.0\n    store(y,out[row:row+1])\n";

fn lowered(text: &str) -> LoweredIr {
    let program = compile(
        &[SourceFile {
            path: "refinement.seismic.portable".into(),
            scope: Scope::Portable,
            text: text.into(),
        }],
        &[],
    )
    .unwrap();
    seismic_lang::lower::lower(&program, "evaluate", "metal", &Default::default()).unwrap()
}
fn config() -> Config {
    Config {
        loads: LoadStrategy::Materialize,
        sg_per_tg: 1,
        max_threads_per_threadgroup: 96,
        max_threadgroup_bytes: 1024,
        ..Default::default()
    }
}
fn family(function: &LoweredIr, config: Config) -> Arc<GroupFamily> {
    Arc::new(
        GroupFamily::derive(
            execution::prepare_with_choices(
                function,
                config,
                &mut |_, _| Ok(LoadMode::Materialize),
                &mut |_| Ok(TilePlacement::GroupShared),
                &mut |decision| Ok(decision.diagnostic()),
            )
            .unwrap(),
        )
        .unwrap(),
    )
}
fn choice(prepared: Preparation<Execution>) -> selection::Domain {
    let Preparation::Choice { alternatives, .. } = prepared else {
        panic!("expected launch choice")
    };
    alternatives
}
fn refine(family: &Arc<GroupFamily>, values: &[u64]) -> Execution {
    let mut prepared = family.next(Vec::new()).unwrap();
    for value in values {
        let domain = choice(prepared);
        let owner = domain.owner::<GroupingChoices>().unwrap();
        assert!(std::ptr::eq(family.execution(), owner.family().execution()));
        prepared = owner.refine(owner.index(*value).unwrap()).unwrap();
    }
    let Preparation::Execution(execution) = prepared else {
        panic!("unresolved launch")
    };
    execution
}

#[test]
fn independent_launch_refinement_retains_the_prepared_family_and_exact_domains() {
    let family = family(&lowered(TWO), config());
    let initial = choice(family.next(Vec::new()).unwrap());
    let first = initial.owner::<GroupingChoices>().unwrap();
    assert_eq!(first.values(), &[1, 2]);
    let second_domain = choice(first.refine(first.index(2).unwrap()).unwrap());
    let second = second_domain.owner::<GroupingChoices>().unwrap();
    assert_eq!(second.values(), &[1, 2, 3]);
    assert_eq!(second.selected(), &[2]);
    assert!(std::ptr::eq(
        first.family().execution(),
        second.family().execution()
    ));
    for a in [1, 2] {
        for b in [1, 2, 3] {
            let selected = refine(&family, &[a, b]);
            let expected = family.select_launches(&[a, b]).unwrap();
            assert_eq!(
                msl::emit_execution(&selected).unwrap(),
                msl::emit_execution(&expected).unwrap()
            );
        }
    }
    assert!(first.refine(first.values().len()).is_err());
    let mut wrong = first.clone();
    wrong.launch = 1;
    assert!(wrong.refine(0).is_err());
    assert!(family.next(vec![3]).is_err());

    let large = lowered(
        "fn evaluate(x: tensor[65535,65537,1] f32, out: tensor[65535,65537,1] f32):\n  for row,col in parallel:\n    a = load(x[row,col])\n    store(a,out[row,col])\n",
    );
    let large = self::family(
        &large,
        Config {
            max_threads_per_threadgroup: 256,
            ..config()
        },
    );
    let domain = choice(large.next(Vec::new()).unwrap());
    let owner = domain.owner::<GroupingChoices>().unwrap();
    assert_eq!(owner.values(), &[1, 3, 5]);
    assert!(owner.index(2).is_none());
    assert_eq!(refine(&large, &[3]).phases()[0].dispatch.items_per_group, 3);
}

#[test]
fn split_merge_refinement_preserves_allocation_identity_and_launch_handoffs() {
    let function = support::streamed(SPLIT, 17, &["chunk"]);
    let family = family(
        &function,
        Config {
            split: 3,
            ..config()
        },
    );
    assert_eq!(family.execution().memory().launches().len(), 3);
    for groups in [[1, 3, 2], [3, 1, 2], [2, 2, 3]] {
        let selected = refine(&family, &groups);
        let expected = family.select_launches(&groups).unwrap();
        assert_eq!(selected.memory(), expected.memory());
        assert_eq!(selected.storage(), family.execution().storage());
        assert_eq!(selected.reductions(), family.execution().reductions());
        assert_eq!(
            selected.memory().scratch(),
            family.execution().memory().scratch()
        );
        for (actual, baseline) in selected
            .memory()
            .launches()
            .iter()
            .zip(family.execution().memory().launches())
        {
            assert_eq!(
                actual
                    .arrays
                    .iter()
                    .map(|a| (a.id, a.slot))
                    .collect::<Vec<_>>(),
                baseline
                    .arrays
                    .iter()
                    .map(|a| (a.id, a.slot))
                    .collect::<Vec<_>>()
            );
            assert_eq!(actual.barriers, baseline.barriers);
            assert_eq!(actual.predecessor, baseline.predecessor);
        }
        let emitted = msl::emit_execution(&selected).unwrap();
        assert_eq!(emitted, msl::emit_execution(&expected).unwrap());
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.kernel.as_str())
                .collect::<Vec<_>>(),
            ["evaluate_0", "evaluate_0_merge", "evaluate_1"]
        );
        assert_eq!(
            emitted
                .launches
                .iter()
                .map(|l| l.dispatch.as_ref().unwrap().items_per_group)
                .collect::<Vec<_>>(),
            groups
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal hardware"]
fn native_refined_split_launches_keep_independent_groupings() {
    let device = seismic_metal::runtime::Device::open().unwrap();
    let function = support::streamed(SPLIT, 17, &["chunk"]);
    let family = family(
        &function,
        Config {
            split: 3,
            ..config()
        },
    );
    let input = (0..130).map(|i| i as f32 * 0.25 - 7.0).collect::<Vec<_>>();
    let bytes = input
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    let expected = input
        .chunks(65)
        .map(|row| row.iter().sum::<f32>() * 2.0 + 1.0)
        .collect::<Vec<_>>();
    for groups in [[1, 3, 2], [3, 1, 2], [2, 2, 3]] {
        let selected = refine(&family, &groups);
        let kernel = device
            .compile(msl::emit_execution(&selected).unwrap())
            .unwrap();
        let x = device.buffer_from(&bytes).unwrap();
        let middle = device.buffer(8).unwrap();
        let out = device.buffer(8).unwrap();
        device.run(&kernel, &[&x, &middle, &out], &[], 1).unwrap();
        let actual = out
            .read(8)
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "groupings={groups:?}");
    }
}

#[test]
fn retained_owner_identity_excludes_cache_state_and_includes_source_conditions() {
    let function = lowered(TWO);
    let a = family(&function, config());
    let b = family(&function, config());
    let a_domain = choice(a.next(vec![1]).unwrap());
    let b_domain = choice(b.next(vec![1]).unwrap());
    assert_eq!(a_domain, b_domain);
    msl::prepare_execution(a.execution()).unwrap();
    assert_eq!(a_domain, b_domain, "lazy emission must not change identity");
    assert_ne!(
        a_domain,
        choice(a.next(vec![2]).unwrap()),
        "prior launch selections matter"
    );
    let mut changed = function.clone();
    changed
        .alias_requirements
        .push(seismic_lang::lowered_ir::AliasRequirement {
            left: 0,
            right: 1,
            exact_allowed: false,
        });
    assert_ne!(
        a_domain,
        choice(family(&changed, config()).next(vec![1]).unwrap())
    );
    let mut changed = function;
    changed.ownership.intermediates.insert("middle".into());
    assert_ne!(
        a_domain,
        choice(family(&changed, config()).next(vec![1]).unwrap())
    );
}

/// Fix the earlier implementation choices, then forbid full preparation after
/// the first launch family has been captured by search.
struct RetainedBackend {
    inner: tuning::Backend,
    prefix: Vec<usize>,
    preparations: Cell<usize>,
}
impl tuner::Backend for RetainedBackend {
    type Execution = Execution;
    type Conditions = (tuning::Conditions, Vec<usize>);
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn conditions(&self) -> Self::Conditions {
        (self.inner.conditions(), self.prefix.clone())
    }
    fn description(&self) -> tuner::Description {
        self.inner.description()
    }
    fn prepare(
        &self,
        function: &LoweredIr,
        path: &[usize],
    ) -> Result<Preparation<Execution>, String> {
        assert!(
            path.is_empty(),
            "repeated preparation after family capture: {path:?}"
        );
        self.preparations.set(self.preparations.get() + 1);
        self.inner.prepare(function, &self.prefix)
    }
    fn refine(
        &self,
        alternatives: &selection::Domain,
        index: usize,
    ) -> Result<Option<Preparation<Execution>>, String> {
        self.inner.refine(alternatives, index)
    }
    fn relax(
        &self,
        alternatives: &selection::Domain,
        indices: std::ops::Range<usize>,
        invocation: &workload::ScalarWorkload,
        limits: workload::DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        self.inner.relax(alternatives, indices, invocation, limits)
    }
    fn analyze(
        &self,
        execution: &Execution,
        invocation: &workload::ScalarWorkload,
        limits: workload::DerivationLimits,
    ) -> Result<schedule::evaluation::Model, workload::DerivationError> {
        let mut model = oracle_model(self.inner.analyze(execution, invocation, limits)?);
        for operation in &mut model.operations {
            for &predecessor in &operation.start_predecessors {
                if !operation.predecessors.contains(&predecessor) {
                    operation.predecessors.push(predecessor);
                }
            }
        }
        model.identity.push_str(":blocking-test");
        Ok(model.into())
    }
    fn relax_execution(
        &self,
        execution: &Execution,
        invocation: &workload::ScalarWorkload,
        limits: workload::DerivationLimits,
    ) -> Result<Option<schedule::Demand>, String> {
        self.inner.relax_execution(execution, invocation, limits)
    }
    fn materialize(
        &self,
        execution: &Execution,
        objective: &selection::Objective,
    ) -> Result<Execution, String> {
        self.inner.materialize(execution, objective)
    }
    fn check_materialization(
        &self,
        source: &Execution,
        selected: &Execution,
        objective: &selection::Objective,
    ) -> Result<(), String> {
        self.inner
            .check_materialization(source, selected, objective)
    }
}

#[test]
fn two_launch_optimum_and_resume_reuse_the_same_preparation() {
    let function = lowered(
        "fn evaluate(middle: tensor[1] f32, out: tensor[1] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 3.0\n    store(y,middle[row:row+1])\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = middle[row] * 2.0\n    store(y,out[row:row+1])\n",
    );
    let capacities = tuning::Capacities {
        max_threads_per_threadgroup: 64,
        max_threadgroup_bytes: 1024,
    };
    let form = tuning::Form::Fixed(Default::default());
    let mut prefix = Vec::new();
    let initial = loop {
        let domain = choice(tuning::expand(&function, &form, &capacities, &prefix).unwrap());
        if domain.owner::<GroupingChoices>().is_some() {
            break domain;
        }
        let owner = domain
            .owner::<seismic_metal::choices::ExecutionChoice>()
            .unwrap();
        let selected = match owner.decision() {
            seismic_metal::choices::Decision::Storage(_) => {
                seismic_metal::choices::Alternative::Storage(TilePlacement::GroupShared)
            }
            _ => owner.diagnostic(LoadStrategy::Materialize),
        };
        prefix.push(owner.index(&selected).unwrap());
    };
    let owner = initial.owner::<GroupingChoices>().unwrap();
    assert_eq!(owner.values(), &[1, 2]);
    let mut executions = Vec::new();
    let mut primitives = Vec::new();
    for first in 0..2 {
        let next = choice(owner.refine(first).unwrap());
        let second = next.owner::<GroupingChoices>().unwrap();
        for last in 0..2 {
            let Preparation::Execution(execution) = second.refine(last).unwrap() else {
                panic!()
            };
            for primitive in model::requirements(&execution).unwrap().primitives {
                if !primitives.contains(&primitive) {
                    primitives.push(primitive);
                }
            }
            let mut path = prefix.clone();
            path.extend([first, last]);
            let Preparation::Execution(full) =
                tuning::expand(&function, &form, &capacities, &path).unwrap()
            else {
                panic!()
            };
            assert_eq!(
                msl::emit_execution(&execution).unwrap(),
                msl::emit_execution(&full).unwrap()
            );
            executions.push((vec![first, last], execution));
        }
    }
    let hardware = model::Hardware {
        identity: "synthetic blocking Metal test".into(),
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![schedule::Resource {
            name: "issue".into(),
            capacity: 1024,
            unit: schedule::CapacityUnit::Slots,
        }],
        resident_groups: 1,
        resident_shared_bytes: 1024,
        timings: primitives
            .into_iter()
            .map(|primitive| model::Timing {
                primitive,
                latency: 1,
                services: vec![model::Service {
                    resource: 0,
                    offset: 0,
                    duration: 1,
                    units: model::Units::PerLane(1),
                }],
            })
            .collect(),
    };
    let backend = RetainedBackend {
        inner: tuning::Backend::with_conditions(tuning::Conditions {
            target: "fixture".into(),
            capacities,
            form,
            hardware,
        })
        .unwrap(),
        prefix,
        preparations: Cell::new(0),
    };
    let invocation = workload::ScalarWorkload { integer_domains: Vec::new(),
        identity: "disjoint two-phase buffers".into(),
        allocations: (0..2)
            .map(|id| workload::Allocation {
                id,
                bytes: 4,
                alignment: 4,
                known_bytes: Default::default(),
            })
            .collect(),
        buffers: (0..2)
            .map(|allocation| workload::BufferBinding {
                allocation,
                offset: 0,
                bytes: 4,
            })
            .collect(),
        scalars: vec![],
    };
    let limits = workload::DerivationLimits {
        instructions: 100_000,
        operations: 100_000,
    };
    let expected = executions
        .iter()
        .map(|(_, execution)| {
            let solution = oracle_model(backend
                .analyze(execution, &invocation, limits)
                .unwrap())
                .solve(100_000)
                .unwrap();
            assert!(solution.is_optimal());
            solution.schedule().completion
        })
        .min()
        .unwrap();
    let request = tuner::Request {
        input: tuner::Input::Lowered(&function),
        backend: &backend,
        workload: &invocation,
        derivation_limits: limits,
    };
    let tuner::Outcome::Incomplete(progress) = tuner::tune(
        &request,
        selection::Budget {
            nodes: 1,
            schedule_assignments: 0,
        },
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(backend.preparations.get(), 1);
    let tuner::Outcome::Optimal(selected) = tuner::resume(
        &request,
        progress,
        selection::Budget {
            nodes: 100,
            schedule_assignments: 100_000,
        },
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(backend.preparations.get(), 1);
    assert_eq!(selected.modeled_cost().upper(), expected);
    let (_, expected) = executions
        .iter()
        .find(|(path, _)| path == selected.selected_path())
        .unwrap();
    assert_eq!(
        msl::emit_execution(selected.execution()).unwrap(),
        msl::emit_execution(expected).unwrap()
    );
    for ownership in [false, true] {
        let tuner::Outcome::Incomplete(progress) = tuner::tune(
            &request,
            selection::Budget {
                nodes: 1,
                schedule_assignments: 0,
            },
        )
        .unwrap() else {
            panic!()
        };
        let mut changed = function.clone();
        if ownership {
            changed.ownership.intermediates.insert("middle".into());
        } else {
            changed
                .alias_requirements
                .push(seismic_lang::lowered_ir::AliasRequirement {
                    left: 0,
                    right: 1,
                    exact_allowed: false,
                });
        }
        let changed = tuner::Request {
            input: tuner::Input::Lowered(&changed),
            ..request
        };
        assert!(
            matches!(tuner::resume(&changed, progress, selection::Budget { nodes: 100, schedule_assignments: 100_000 }), Err(message) if message == "selection inputs changed")
        );
    }
}

#[test]
fn execution_refinement_reuses_completed_stages_and_matches_full_path() {
    use seismic_metal::choices::{self, Decision, Expansion};
    let function = lowered(&TWO.replace("    store(y,middle[row])", "    store(y,middle[row])\n    z = tile[64] f32\n    for i in owned(z): z[i] = middle[row,i] * 2.0\n    store(z,middle[row])"));
    let config = config();
    let mut prefix = Vec::new();
    let mut next = choices::expand(&function, config.clone(), &[]).unwrap();
    let mut reused = 0;
    let mut saw_load = false;
    let mut saw_storage = false;
    let mut saw_allocation = false;
    loop {
        match next {
            Expansion::Choice(owner) => {
                let Expansion::Choice(rebuilt) =
                    choices::expand(&function, config.clone(), &prefix).unwrap()
                else {
                    panic!("full path must reach the same choice");
                };
                assert_eq!(owner, rebuilt);
                saw_load |= matches!(owner.decision(), Decision::Load(_));
                saw_storage |= matches!(owner.decision(), Decision::Storage(_));
                saw_allocation |= matches!(owner.decision(), Decision::Allocation(_));
                let selected = match owner.decision() {
                    Decision::Storage(_) => {
                        choices::Alternative::Storage(TilePlacement::GroupShared)
                    }
                    _ => owner.diagnostic(LoadStrategy::Materialize),
                };
                let index = owner.index(&selected).unwrap();
                next = owner.refine(index).unwrap();
                if let Expansion::Choice(ref next) = next {
                    if matches!(
                        (owner.decision(), next.decision()),
                        (Decision::Storage(_), Decision::Storage(_))
                            | (Decision::Allocation(_), Decision::Allocation(_))
                    ) {
                        assert!(std::ptr::eq(owner.prepared(), next.prepared()));
                        reused += 1;
                    }
                }
                prefix.push(index);
            }
            Expansion::Execution {
                execution,
                consumed,
            } => {
                assert_eq!(consumed, prefix.len());
                let Expansion::Execution {
                    execution: rebuilt,
                    consumed: full,
                } = choices::expand(&function, config.clone(), &prefix).unwrap()
                else {
                    panic!("full path must complete");
                };
                assert_eq!(consumed, full);
                assert_eq!(execution.function(), rebuilt.function());
                assert_eq!(execution.phases(), rebuilt.phases());
                assert_eq!(execution.memory(), rebuilt.memory());
                assert_eq!(
                    msl::emit_execution(&execution).unwrap().source,
                    msl::emit_execution(&rebuilt).unwrap().source
                );
                break;
            }
            Expansion::Infeasible { .. } => panic!("small prepared example must fit"),
        }
    }
    assert!(
        saw_load && saw_storage && saw_allocation && reused > 0,
        "load={saw_load} storage={saw_storage} allocation={saw_allocation} reused={reused}"
    );
}

#[test]
fn grouping_region_demand_bounds_every_remaining_launch_assignment() {
    use seismic_accounting::selection::Choices;
    use seismic_metal::terminal::Primitive;
    let function = lowered(
        "fn evaluate(middle:tensor[5] f32,out:tensor[5] f32):\n  for row in parallel:\n    value = tile[1] f32\n    for i in owned(value): value[i] = 3.0\n    store(value,middle[row:row+1])\n  for row in parallel:\n    value = tile[1] f32\n    for i in owned(value): value[i] = middle[row] * 2.0\n    store(value,out[row:row+1])\n",
    );
    let family = family(&function, config());
    let domain = choice(family.next(vec![]).unwrap());
    let owner = domain.owner::<GroupingChoices>().unwrap();
    let mut primitives = Vec::new();
    for grouping in family.groupings() {
        let execution = family.select(grouping.items_per_group).unwrap();
        for primitive in model::requirements(&execution).unwrap().primitives {
            if !primitives.contains(&primitive) { primitives.push(primitive); }
        }
    }
    let hardware = model::Hardware {
        identity: "synthetic dispatch-service fixture".into(),
        timebase: schedule::Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1,
        },
        resources: vec![schedule::Resource {
            name: "submission".into(),
            capacity: 1,
            unit: schedule::CapacityUnit::Slots,
        }],
        resident_groups: 1,
        resident_shared_bytes: 1024,
        timings: primitives.into_iter()
            .map(|primitive| {
                let latency = match primitive {
                    Primitive::Launch => 3,
                    Primitive::Group => 10,
                    Primitive::Write { space: seismic_metal::terminal::Space::Device, .. } => 7,
                    _ => 0,
                };
                model::Timing {
                    primitive,
                    latency,
                    services: if latency == 0 {
                        vec![]
                    } else {
                        vec![model::Service {
                            resource: 0,
                            offset: 0,
                            duration: latency,
                            units: model::Units::PerSubgroup(1),
                        }]
                    },
                }
            })
            .collect(),
    };
    let backend = tuning::Backend::with_conditions(tuning::Conditions {
        target: "fixture".into(),
        capacities: tuning::Capacities {
            max_threads_per_threadgroup: 96,
            max_threadgroup_bytes: 1024,
        },
        form: tuning::Form::Fixed(Default::default()),
        hardware,
    })
    .unwrap();
    let invocation = workload::ScalarWorkload { integer_domains: Vec::new(),
        identity: "two outputs".into(),
        allocations: (0..2)
            .map(|id| workload::Allocation {
                id,
                bytes: 20,
                alignment: 4,
                known_bytes: Default::default(),
            })
            .collect(),
        buffers: (0..2)
            .map(|allocation| workload::BufferBinding {
                allocation,
                offset: 0,
                bytes: 20,
            })
            .collect(),
        scalars: vec![],
    };
    let bound = |domain: &selection::Domain, range| {
        backend
            .relax(domain, range, &invocation, workload::DerivationLimits { instructions: 100_000, operations: 100_000 })
            .unwrap()
            .unwrap()
            .lower_bound()
            .unwrap()
    };
    let broad = bound(&domain, 0..owner.len());
    let narrow = bound(&domain, 0..1);
    assert!(broad > 46 && narrow > broad, "the bound must include mandatory publications beyond the two launches and minimum groups");
    let cached = backend.relax(&domain, 0..owner.len(), &invocation, workload::DerivationLimits { instructions: 0, operations: 0 }).unwrap().unwrap().lower_bound().unwrap();
    assert_eq!(cached, broad, "completed body counts survive a smaller budget");
    for first in 0..owner.len() {
        let next = choice(owner.refine(first).unwrap());
        let second = next.owner::<GroupingChoices>().unwrap();
        let region = bound(&next, 0..second.len());
        assert!(region >= broad);
        for last in 0..second.len() {
            let Preparation::Execution(execution) = second.refine(last).unwrap() else {
                panic!()
            };
            let model = oracle_model(backend
                .analyze(
                    &execution,
                    &invocation,
                    workload::DerivationLimits {
                        instructions: 100_000,
                        operations: 100_000,
                    },
                )
                .unwrap());
            assert!(model.unmapped.is_empty(), "{:?}", model.unmapped);
            let floor = model.lower_bound().unwrap();
            assert!(broad <= floor && region <= floor);
            if first == 0 {
                assert!(narrow <= floor);
            }
        }
    }
}
