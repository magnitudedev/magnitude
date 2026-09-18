//! Adversarial symbolic interpretation tests. Unknown inputs stay unknown when
//! lexical state, target integer semantics, or loop termination require it.
use super::*;
use crate::terminal::{Expression as E, Site, Statement as S, Type as T};

fn site(statement: S) -> Site { Site { operation: None, statement } }
fn state(instructions: u64) -> Derivation<'static> {
    Derivation::new(Sink::Count {
        account: InvocationAccount { operations: vec![], unmapped: vec![], exhausted: None, visits: 0 },
        indices: Default::default(),
    }, DerivationLimits { instructions, operations: 100 })
}

#[test]
fn helper_scope_restoration_discards_facts_from_a_shadowed_binding() {
    let mut state = state(100);
    state.env.insert("input".into(), [None; 32]);
    state.ranges.insert("input".into(), [Some((0, 10)); 32]);
    state.affine.insert("input".into(), std::array::from_fn(|_| Some(affine::Value::coordinate(0, 0, 10))));
    // An identity helper with a nested local shadow. Supplying its retained
    // typed body exercises the ordinary helper frame and return-fact path.
    state.helpers.insert((crate::support::Helper::Index, T::I64), Arc::new(HelperBody {
        parameters: vec!["value", "extent", "status"],
        statements: vec![
            site(S::Scope),
            site(S::Let { name: "value".into(), ty: T::I64, value: E::Integer(7, T::I64) }),
            site(S::Evaluate(E::variable("value", T::I64))),
            site(S::End),
            site(S::Return(Some(E::variable("value", T::I64)))),
        ],
    }));
    let call = E::Helper(crate::support::Helper::Index, vec![E::variable("input", T::I64), E::Integer(11, T::I64), E::Integer(0, T::U64)], T::I64);
    assert_eq!(state.expr(&call).unwrap().unwrap(), [None; 32]);
    assert_eq!(state.expression_ranges(&call), [Some((0, 10)); 32]);
    assert_eq!(state.expression_affine(&call), state.affine["input"]);
    assert_eq!(state.env["input"], [None; 32]);
}

#[test]
fn symbolic_shifts_obey_the_target_operand_width() {
    for ty in [T::I32, T::U32, T::I64, T::U64] {
        let width = if matches!(ty, T::I64 | T::U64) { 64 } else { 32 };
        for range in [(0, 0), (0, 7)] {
            for shift in [width, width + 8] {
                for op in [BinaryOp::Shl, BinaryOp::Shr] {
                    let mut state = state(100);
                    state.env.insert("x".into(), [None; 32]);
                    state.ranges.insert("x".into(), [Some(range); 32]);
                    let expression = E::binary(op, E::variable("x", ty), E::Integer(shift, ty), ty);
                    assert_eq!(state.expression_ranges(&expression), [None; 32], "{expression:?}");
                    assert_eq!(state.expression_affine(&expression), std::array::from_fn(|_| None), "{expression:?}");
                    assert_eq!(state.expr(&expression).unwrap().unwrap(), [None; 32], "{expression:?}");
                }
            }
        }
    }
    // The largest legal 64-bit shift is admitted when its result fits.
    let mut state = state(100);
    state.env.insert("x".into(), [None; 32]);
    state.ranges.insert("x".into(), [Some((0, 1)); 32]);
    let shift = E::binary(BinaryOp::Shl, E::variable("x", T::U64), E::Integer(63, T::U64), T::U64);
    assert_eq!(state.expression_ranges(&shift), [Some((0, 1i128 << 63)); 32]);
}

#[test]
fn a_loop_bound_that_reads_its_induction_variable_is_not_frozen() {
    let body = vec![
        site(S::For { name: "i".into(), start: E::integer(0),
            end: E::binary(BinaryOp::Add, E::variable("i", T::I32), E::integer(3), T::I32), step: 1 }),
        site(S::Evaluate(E::integer(1))),
        site(S::End),
    ];
    for symbolic in [false, true] {
        let mut state = state(64);
        state.symbolic = symbolic;
        // i < i + 3 keeps extending the endpoint. It cannot become a completed
        // three-iteration trace by capturing its first endpoint value.
        assert!(matches!(state.block(&body, 0, body.len()),
            Err(DerivationError::Exhausted(DerivationLimit::Instructions(64)))));
    }
}

fn dispatch_fixture(count: i64) -> (crate::execution::Execution, Hardware, ScalarWorkload) {
    use seismic_lang::{Scope, program::{SourceFile, compile}};
    use seismic_accounting::workload::{Allocation, BufferBinding};
    let program = compile(&[SourceFile { path: "dispatch-refinement.seismic.portable".into(), scope: Scope::Portable,
        text: "fn write[N](out: tensor[N] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 3.0\n    store(y, out[row:row+1])\n".into() }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "write", "metal", &std::collections::HashMap::from([("N".into(), count)])).unwrap();
    let execution = crate::execution::prepare(&lowered, crate::execution::Config::default()).unwrap();
    let hardware = Hardware { identity: "dispatch gap classification fixture".into(),
        timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "service".into(), capacity: 1024, unit: schedule::CapacityUnit::Slots }],
        resident_groups: 2, resident_shared_bytes: 65536,
        timings: requirements(&execution).unwrap().primitives.into_iter().map(|primitive| Timing {
            primitive, latency: 1, services: vec![Service { resource: 0, offset: 0, duration: 1, units: Units::PerLane(1) }],
        }).collect(),
    };
    let bytes = count as u64 * 4;
    let workload = ScalarWorkload { identity: "dispatch gap classification fixture".into(),
        allocations: vec![Allocation { id: 0, bytes, alignment: 16, known_bytes: Default::default() }],
        buffers: vec![BufferBinding { allocation: 0, offset: 0, bytes }], scalars: vec![], integer_domains: vec![],
    };
    (execution, hardware, workload)
}

#[test]
fn missing_service_stays_unmapped_without_expanding_a_billion_groups() {
    let (execution, hardware, workload) = dispatch_fixture(1_000_000_003);
    for missing in [Primitive::Launch, Primitive::Write { space: crate::terminal::Space::Device, ty: T::F32 }] {
        let mut hardware = hardware.clone();
        let before = hardware.timings.len();
        hardware.timings.retain(|timing| timing.primitive != missing);
        assert!(hardware.timings.len() < before);
        let model = structured_execution(&execution, &hardware, &workload,
            DerivationLimits { instructions: 1000, operations: 1000 }).expect("an invariant missing service is a gap, not an exhausted dispatch walk");
        assert!(!model.unmapped.is_empty());
        assert!(model.compact_witness().is_err(), "invariant gaps still prevent executable witnesses");
    }
}

#[test]
fn unresolved_transaction_geometry_still_refines_dispatch_coordinates() {
    let (execution, mut hardware, workload) = dispatch_fixture(11);
    for timing in &mut hardware.timings {
        if matches!(timing.primitive, Primitive::Write { space: crate::terminal::Space::Device, .. }) {
            for service in &mut timing.services { service.units = Units::PerTransaction { bytes: 16, units: 1 }; }
        }
    }
    let limits = DerivationLimits { instructions: 1000, operations: 1000 };
    let structured = structured_execution(&execution, &hardware, &workload, limits).unwrap();
    let flat = super::execution(&execution, &hardware, &workload, limits).unwrap();
    assert!(structured.unmapped.is_empty(), "{:?}", structured.unmapped);
    let counts = |model: &Model| {
        let mut counts = BTreeMap::new();
        for operation in model.operations.iter().filter(|operation| operation.latency > 0) {
            *counts.entry(format!("{}:{:?}", operation.latency, operation.reservations)).or_insert(0u64) += 1;
        }
        counts
    };
    assert_eq!(counts(&structured.expand(1000).unwrap()), counts(&flat));
}

#[test]
fn subgroup_domains_preserve_exact_counts_and_partial_final_groups() {
    fn counts(account: &InvocationAccount) -> BTreeMap<String, u64> {
        assert!(account.is_complete(), "{:?}; {:?}", account.unmapped, account.exhausted);
        let mut result = BTreeMap::new();
        for operation in &account.operations {
            *result.entry(format!("{:?}:{}", operation.primitive, operation.lanes)).or_default() += operation.instances;
        }
        result
    }
    for count in [31, 32, 33, 63, 64, 65, 1024] {
        let (execution, _, workload) = dispatch_fixture(count);
        let execution = crate::family::GroupFamily::derive(execution).unwrap().select(32).unwrap();
        let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
        let compact = invocation_relaxation(&execution, &workload, limits).unwrap();
        let concrete = invocation_account(&execution, &workload, limits).unwrap();
        assert_eq!(counts(&compact), counts(&concrete), "count={count}");
        if count == 1024 { assert!(compact.visits * 8 < concrete.visits, "{} vs {}", compact.visits, concrete.visits); }
    }
}

#[test]
fn dispatch_refinement_restores_count_multiplicity_before_subdivision() {
    use seismic_lang::{Scope, program::{SourceFile, compile}};
    let program = compile(&[SourceFile { path: "dispatch-branches.seismic.portable".into(), scope: Scope::Portable,
        text: "fn write[N](out: tensor[N] f32):\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 3.0\n    if row < 37: y[0] += 2.0\n    store(y, out[row:row+1])\n".into() }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "write", "metal", &std::collections::HashMap::from([("N".into(), 67)])).unwrap();
    let execution = crate::execution::prepare(&lowered, crate::execution::Config { sg_per_tg: 8, ..Default::default() }).unwrap();
    let (_, _, workload) = dispatch_fixture(67);
    let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
    let compact = invocation_relaxation(&execution, &workload, limits).unwrap();
    let concrete = invocation_account(&execution, &workload, limits).unwrap();
    assert!(compact.is_complete(), "{:?}", compact.unmapped);
    let counts = |account: &InvocationAccount| {
        let mut counts = BTreeMap::new();
        for term in &account.operations { *counts.entry(format!("{:?}:{}", term.primitive, term.lanes)).or_insert(0u64) += term.instances; }
        counts
    };
    assert_eq!(counts(&compact), counts(&concrete));
}

#[test]
fn retained_integer_publications_supply_later_launch_control() {
    use seismic_lang::{Scope, program::{SourceFile, compile}};
    use seismic_accounting::workload::{Allocation, BufferBinding};
    let source = "fn write(out: tensor[67] f32):\n  limit = 3\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 0.0\n    if row < limit:\n      for i in owned(y): y[i] += 1.0\n    store(y,out[row:row+1])\n  limit += 1\n  for row in parallel:\n    y = tile[1] f32\n    for i in owned(y): y[i] = 0.0\n    if row < limit:\n      for i in owned(y): y[i] += 2.0\n    store(y,out[row:row+1])\n";
    let program = compile(&[SourceFile { path: "retained-control.seismic.portable".into(), scope: Scope::Portable, text: source.into() }], &[]).unwrap();
    let lowered = seismic_lang::lower::lower(&program, "write", "metal", &Default::default()).unwrap();
    let execution = crate::execution::prepare(&lowered, crate::execution::Config::default()).unwrap();
    assert!(execution.phases().len() >= 4);
    let workload = ScalarWorkload { identity: "serial integer publication".into(),
        allocations: vec![Allocation { id: 0, bytes: 268, alignment: 16, known_bytes: Default::default() }],
        buffers: vec![BufferBinding { allocation: 0, offset: 0, bytes: 268 }], scalars: vec![], integer_domains: vec![],
    };
    let limits = DerivationLimits { instructions: 100_000, operations: 100_000 };
    let compact = invocation_relaxation(&execution, &workload, limits).unwrap();
    let concrete = invocation_account(&execution, &workload, limits).unwrap();
    assert!(compact.is_complete(), "{:?}", compact.unmapped);
    assert!(concrete.is_complete(), "{:?}", concrete.unmapped);
    let counts = |account: &InvocationAccount| {
        let mut counts = BTreeMap::new();
        for term in &account.operations { *counts.entry(format!("{:?}:{}", term.primitive, term.lanes)).or_insert(0u64) += term.instances; }
        counts
    };
    assert_eq!(counts(&compact), counts(&concrete));
}
