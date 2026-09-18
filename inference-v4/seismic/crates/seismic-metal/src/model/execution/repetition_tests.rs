use super::*;
use crate::terminal::{Expression as E, Site, Statement as S, Type as T};

fn site(statement: S) -> Site { Site { operation: None, statement } }
fn repeated(count: i64, body: Vec<S>) -> Vec<Site> {
    std::iter::once(site(S::For { name: "k".into(), start: E::integer(0), end: E::integer(count), step: 1 }))
        .chain(body.into_iter().map(site)).chain(std::iter::once(site(S::End))).collect()
}
fn derive(body: &[Site], symbolic: bool, instructions: u64) -> (InvocationAccount, Result<Result<(), String>, DerivationError>) {
    let mut state = Derivation::new(Sink::Count {
        account: InvocationAccount { operations: vec![], unmapped: vec![], exhausted: None, visits: 0 },
        indices: Default::default(),
    }, DerivationLimits { instructions, operations: 100 });
    state.symbolic = symbolic;
    for name in ["acc", "x", "y"] { state.env.insert(name.into(), [None; 32]); }
    let result = state.block(body, 0, body.len());
    let Sink::Count { mut account, .. } = state.sink else { unreachable!() };
    account.visits = state.visits;
    (account, result)
}
fn fma() -> S {
    S::Assign { name: "acc".into(), value: E::Builtin("fma".into(), vec![E::variable("acc", T::F32), E::variable("x", T::F32), E::variable("y", T::F32)], T::F32) }
}
fn assert_counts_equal(a: &InvocationAccount, b: &InvocationAccount) {
    assert_eq!(a.operations.len(), b.operations.len());
    for left in &a.operations {
        let right = b.operations.iter().find(|right| right.primitive == left.primitive && right.lanes == left.lanes && right.access == left.access).expect("matching operation");
        assert_eq!(left.instances, right.instances, "{:?}", left.primitive);
    }
}

#[test]
fn repeated_work_matches_concrete_counts_and_scales_with_structure() {
    for count in [2, 7, 103] {
        let body = repeated(count, vec![fma()]);
        let (concrete, result) = derive(&body, false, 10000);
        result.unwrap().unwrap();
        let (symbolic, result) = derive(&body, true, 10);
        result.unwrap().unwrap();
        assert_counts_equal(&concrete, &symbolic);
        assert_eq!(symbolic.visits, 4);
    }
    let body = repeated(1_000_000_000, vec![fma()]);
    let (symbolic, result) = derive(&body, true, 10);
    result.unwrap().unwrap();
    assert_eq!(symbolic.visits, 4);
    assert!(symbolic.operations.iter().any(|o| matches!(o.primitive, Primitive::Builtin { .. }) && o.instances == 1_000_000_000));
    assert!(matches!(derive(&body, false, 10).1, Err(DerivationError::Exhausted(_))));
}

#[test]
fn data_dependent_iteration_structure_stays_concrete() {
    let body = repeated(7, vec![
        S::If(E::binary(BinaryOp::Lt, E::variable("k", T::I32), E::integer(3), T::Bool)),
        fma(), S::End,
    ]);
    let (concrete, result) = derive(&body, false, 1000);
    result.unwrap().unwrap();
    let (symbolic, result) = derive(&body, true, 1000);
    result.unwrap().unwrap();
    assert_counts_equal(&concrete, &symbolic);
    assert_eq!(concrete.visits, symbolic.visits);
}

#[test]
fn loop_carried_values_do_not_become_first_iteration_specializations() {
    let mut body = vec![site(S::Let { name: "acc".into(), ty: T::I32, value: E::integer(0) })];
    body.extend(repeated(7, vec![S::Assign { name: "acc".into(), value: E::binary(BinaryOp::Add, E::variable("acc", T::I32), E::integer(1), T::I32) }]));
    body.extend([
        site(S::If(E::binary(BinaryOp::Eq, E::variable("acc", T::I32), E::integer(7), T::Bool))),
        site(fma()), site(S::End),
    ]);
    derive(&body, false, 1000).1.unwrap().unwrap();
    let (account, result) = derive(&body, true, 1000);
    assert!(result.unwrap().unwrap_err().contains("predicate"));
    assert!(!account.operations.iter().any(|o| matches!(o.primitive, Primitive::Builtin { .. })));
}

#[test]
fn loop_counter_remains_known_after_symbolic_repetition() {
    let mut body = repeated(7, vec![fma()]);
    body.extend([
        site(S::If(E::binary(BinaryOp::Eq, E::variable("k", T::I32), E::integer(7), T::Bool))),
        site(fma()), site(S::End),
    ]);
    let (concrete, result) = derive(&body, false, 1000);
    result.unwrap().unwrap();
    let (symbolic, result) = derive(&body, true, 1000);
    result.unwrap().unwrap();
    assert_counts_equal(&concrete, &symbolic);
}

#[test]
fn nested_repetition_retains_product_without_expanding_iterations() {
    fn nested(outer: i64, inner: i64) -> Vec<Site> {
        let mut body = repeated(inner, vec![fma()]);
        body.insert(0, site(S::For { name: "j".into(), start: E::integer(0), end: E::integer(outer), step: 1 }));
        body.push(site(S::End));
        body
    }
    let body = nested(7, 13);
    let (concrete, result) = derive(&body, false, 10000);
    result.unwrap().unwrap();
    let (symbolic, result) = derive(&body, true, 20);
    result.unwrap().unwrap();
    assert_counts_equal(&concrete, &symbolic);
    let (large, result) = derive(&nested(1_000_000, 1_000_000), true, 20);
    result.unwrap().unwrap();
    assert_eq!(large.visits, symbolic.visits);
    assert!(large.operations.iter().any(|o| matches!(o.primitive, Primitive::Builtin { .. }) && o.instances == 1_000_000_000_000));
}

#[test]
fn unknown_addresses_supply_only_a_broadcast_safe_transaction_floor() {
    use crate::terminal::Space;
    let primitive = Primitive::VectorRead { space: Space::Device, ty: T::F32, components: 4 };
    let hardware = Hardware {
        identity: "transaction lower-bound fixture".into(),
        timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "requests".into(), capacity: 32, unit: schedule::CapacityUnit::Slots }],
        resident_groups: 1, resident_shared_bytes: 0,
        timings: vec![Timing { primitive: primitive.clone(), latency: 1, services: vec![Service {
            resource: 0, offset: 0, duration: 1, units: Units::PerTransaction { bytes: 8, units: 1 },
        }] }],
    };
    hardware.validate().unwrap();
    assert!(hardware.operation(&primitive, 32, None).unwrap().is_none());
    let lower = hardware.expand_operation(&primitive, 32, None, true).unwrap().unwrap();
    assert_eq!(lower.reservations[0].units, 2);
    for residue in 0..8 {
        for stride in [0, 4, 16, 128] {
            let access = AccessPattern::new(128, (0..32).map(|lane| {
                let start = residue + lane * stride;
                (start, start + 16)
            }).collect()).unwrap();
            let exact = hardware.operation(&primitive, 32, Some(&access)).unwrap().unwrap();
            assert!(lower.reservations[0].units <= exact.reservations[0].units);
        }
    }
    let weak = AccessPattern::new(4, vec![(0, 16)]).unwrap();
    assert!(hardware.operation(&primitive, 32, Some(&weak)).unwrap().is_none());
    assert_eq!(hardware.expand_operation(&primitive, 32, Some(&weak), true).unwrap().unwrap().reservations[0].units, 2);
}

#[test]
fn structured_terminal_loops_keep_exact_order_and_multiplicity() {
    for count in [7, 1_000_000_000] {
        let body = repeated(count, vec![fma()]);
        let (account, result) = derive(&body, true, 20);
        result.unwrap().unwrap();
        let hardware = Hardware {
            identity: "structured terminal fixture".into(),
            timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
            resources: vec![Resource { name: "service".into(), capacity: 1, unit: schedule::CapacityUnit::Slots }],
            resident_groups: 1, resident_shared_bytes: 0,
            timings: account.operations.iter().map(|o| Timing { primitive: o.primitive.clone(), latency: 1,
                services: vec![Service { resource: 0, offset: 0, duration: 1, units: Units::PerSubgroup(1) }] }).collect(),
        };
        let model = Structured {
            relationship: ModelRelationship::hypothetical_execution(), identity: "terminal repeated body".into(),
            timebase: hardware.timebase.clone(), resources: hardware.resources.clone(), unmapped: vec![],
            root: Arc::new(StructuredNode::Compose { order: StructuredOrder::Serial, children: vec![] }),
        };
        let mut state = Derivation::new(Sink::Structured { hardware: &hardware, model, groups_resource: 0,
            shared_resource: None, current: vec![], frames: vec![], operations: 0, invariant_mapping_gap: false }, DerivationLimits { instructions: 20, operations: 20 });
        state.structured_loops = true;
        for name in ["acc", "x", "y"] { state.env.insert(name.into(), [None; 32]); }
        state.block(&body, 0, body.len()).unwrap().unwrap();
        assert_eq!(state.visits, 6);
        let Sink::Structured { mut model, current, frames, .. } = state.sink else { unreachable!() };
        assert!(frames.is_empty());
        assert!(matches!(current[0].as_ref(), StructuredNode::Repeat { count: n, .. } if *n == count as u64));
        model.root = Arc::new(StructuredNode::Compose { order: StructuredOrder::Serial, children: current });
        let witness = model.compact_witness().unwrap().unwrap();
        assert!(witness.is_optimal());
        assert_eq!(witness.completion(), 4 * count as u64 + 2);
        if count == 7 {
            let expanded = model.expand(100).unwrap();
            let names = expanded.operations.iter().map(|o| o.name.as_str()).collect::<Vec<_>>();
            assert!(names[0].contains("Lt"));
            assert!(names[1].contains("Branch"));
            assert!(names[2].contains("fma"));
            assert!(names[3].contains("Add"));
        }
    }
}

#[test]
fn nested_symbolic_loops_keep_checked_helper_indices() {
    let inner = E::binary(BinaryOp::Add,
        E::binary(BinaryOp::Mul, E::variable("i", T::I32), E::integer(1024), T::I32), E::variable("j", T::I32), T::I32);
    let body = vec![
        site(S::For { name: "i".into(), start: E::integer(0), end: E::integer(1_000_000), step: 1 }),
        site(S::For { name: "j".into(), start: E::integer(0), end: E::integer(1024), step: 1 }),
        site(S::Let { name: "offset".into(), ty: T::I32, value: inner }),
        site(S::If(E::binary(BinaryOp::Lt, E::variable("offset", T::I32), E::integer(1_024_000_000), T::Bool))),
        site(fma()), site(S::End), site(S::End), site(S::End),
    ];
    let (account, result) = derive(&body, true, 100);
    result.unwrap().unwrap();
    assert!(account.operations.iter().any(|operation| matches!(operation.primitive, Primitive::Builtin { .. }) && operation.instances == 1_024_000_000));
    assert!(account.visits < 100);
}
