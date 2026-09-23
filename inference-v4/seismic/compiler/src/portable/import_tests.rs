use super::*;

#[test]
fn reached_quantity_arithmetic_and_word_cast_share_the_source_schedule() {
    use crate::candidate_domain::construct_candidate_domain;
    use crate::evaluation_session::boundary_tests::registry;
    use seismic_ir::schedule::{HostValueDestination, HostValueExpr, ScheduleStep};
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::entry::ParameterKind;
    use seismic_lang::expr::{Assignment, SymbolValue};

    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "exact-quantity.seismic".into(),
        text: "fn probe(i: index[1000000000]) -> i32:\n    let q = i * i * i\n    return i32(q)\n".into(),
    }])).unwrap();
    let entry = module.entry(module.entry_named("probe").unwrap(), &Default::default()).unwrap();
    let ParameterKind::Index { symbol: input, .. } = entry.schema().parameters()[0].kind else {
        panic!("checked input is not an Index")
    };
    let device = device();
    let registry = registry();
    let domain = construct_candidate_domain(
        entry, &device, &registry,
        &seismic_lang::precision::PrecisionPolicy::Exact,
        &Default::default(),
    ).unwrap();
    let candidate = domain.read_materialized(&domain.general_construction()).unwrap();
    let steps = candidate.candidate().schedule().steps();
    assert!(steps.iter().any(|step| matches!(step, ScheduleStep::EvaluateHost(value)
        if matches!(value.value, HostValueExpr::Integer(_)))));
    assert!(steps.iter().any(|step| matches!(step, ScheduleStep::EvaluateHost(value)
        if matches!(value.value, HostValueExpr::Word { dtype: DType::I32, .. }))));
    let mut values = Assignment::new();
    values.bind(input, SymbolValue::Nat(999_999_999u64.into()));
    let mut exact = None;
    let mut word = None;
    for step in steps {
        let ScheduleStep::EvaluateHost(evaluation) = step else { continue };
        let (value, destination) = (evaluation.value, evaluation.to);
        match (value, destination) {
            (HostValueExpr::Integer(expression), HostValueDestination::Quantity(slot)) => {
                let value = domain.arena().eval_int(expression, &values).unwrap();
                exact = Some(value.clone());
                values.bind(slot.symbol(), SymbolValue::Int(value));
            }
            (HostValueExpr::Word { dtype: DType::I32, value }, HostValueDestination::Native(_)) => {
                word = Some(domain.arena().eval_int(value, &values).unwrap());
            }
            _ => {}
        }
    }
    let exact = exact.expect("source arithmetic published its reached exact value");
    assert!(exact > u64::MAX.into());
    assert_eq!(word, Some(((999_999_999u128).pow(3) as u32 as i32).into()));
}

pub(super) fn device(
) -> seismic_target::DeviceDescription<crate::realization::demand_driven_tests::FakeTarget> {
    let mut description = crate::realization::demand_driven_tests::device_parts();
    description
        .dtypes
        .representations
        .insert(seismic_lang::registry::dense(DType::F32));
    description
        .dtypes
        .representations
        .insert(seismic_lang::registry::dense(DType::BF16));
    description
        .dtypes
        .representations
        .insert(seismic_lang::registry::representation("q4g64").unwrap());
    description.limits.max_allocation_bytes = 4096;
    description.limits.max_allocation_alignment = 16;
    description.limits.max_index_bits = 64;
    description.limits.max_bindings = 64;
    description.limits.max_argument_bytes = 4096;
    description.limits.max_workgroup_bytes = 4096;
    description.limits.max_grid = [1024; 3];
    seismic_target::DeviceDescription::new(description).unwrap()
}

#[test]
fn authored_root_binds_entry_coordinates_and_nested_result_paths() {
    use crate::candidate_domain::{
        construct_candidate_domain, BodyMapping, ConstructionAllowance, ConstructionCoordinate,
        Materialization,
    };
    use crate::evaluation_session::boundary_tests::registry;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "authored-entry-abi.seismic".into(),
        text: r#"fn probe(x: &tensor[1] f32, pair: (f32, range[4])) -> (tensor[1] f32, (f32, range[4])):
    return (x + x, pair)

lower probe(x: &tensor[1] f32, pair: (f32, range[4])) -> (tensor[1] f32, (f32, range[4])) for cpu:
    return (x + x, pair)
"#.into(),
    }])).unwrap();
    let device = device();
    let registry = registry();
    let entry = module
        .entry(module.entry_named("probe").unwrap(), &Default::default())
        .unwrap();
    let parameters = entry.schema().parameters().to_vec();
    let results = entry.schema().results().to_vec();
    assert_eq!(results.iter().map(|result| result.path.as_slice()).collect::<Vec<_>>(),
        vec![&[0][..], &[1, 0][..], &[1, 1][..]]);
    let mut domain = construct_candidate_domain(
        entry,
        &device,
        &registry,
        &seismic_lang::precision::PrecisionPolicy::Exact,
        &Default::default(),
    )
    .unwrap();
    let body = domain
        .root_selections()
        .into_iter()
        .find(|body| body.mapping == BodyMapping::Authored)
        .unwrap();
    let coordinate = ConstructionCoordinate::root(body);
    let Materialization::Ready(coordinate) = domain
        .advance(
            &coordinate,
            ConstructionAllowance {
                work_units: 100_000,
                wall_time: std::time::Duration::from_secs(30),
            },
        )
        .state
    else {
        panic!("authored root construction completes");
    };
    let read = domain.read_materialized(&coordinate).unwrap();
    let candidate = read.candidate();
    assert_eq!(
        candidate
            .result_publications()
            .iter()
            .map(|result| &result.path)
            .collect::<Vec<_>>(),
        results
            .iter()
            .map(|result| &result.path)
            .collect::<Vec<_>>()
    );
    let imported_parameter = candidate
        .global_allocations()
        .allocations()
        .iter()
        .find_map(|allocation| match allocation.kind {
            seismic_ir::storage::GlobalBufferKind::Argument { value, abi } => Some((value, abi)),
            _ => None,
        })
        .unwrap();
    assert_ne!(
        imported_parameter.0, parameters[0].value,
        "authored formal keeps its own semantic identity"
    );
    assert_eq!(
        imported_parameter.1,
        Some(parameters[0].id),
        "the common entry ABI binds the checked source coordinate"
    );
    assert_eq!(candidate.bindings().arena.parameters.len(), parameters.len());
}

#[test]
fn loop_private_storage_does_not_escape_scalar_result() {
    use crate::candidate_domain::construct_candidate_domain;
    use crate::evaluation_session::boundary_tests::registry;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "historical-import.seismic".into(),
        text: r#"fn scratch() -> f32:
    let mut total = 0.0
    for i in 0..2:
        let mut local = tensor[1] f32
        local[0] = 3.0
        total = total + local[0]
    return total

fn probe() -> f32:
    return scratch()
"#
        .into(),
    }]))
    .unwrap();
    let device = device();
    let registry = registry();
    let entry = module
        .entry(module.entry_named("probe").unwrap(), &Default::default())
        .unwrap();
    let domain = construct_candidate_domain(
        entry,
        &device,
        &registry,
        &seismic_lang::precision::PrecisionPolicy::Exact,
        &Default::default(),
    )
    .unwrap();
    let read = domain
        .read_materialized(&domain.general_construction())
        .unwrap();
    let bindings = &read.candidate().bindings().arena;
    let mut live_storage = 0;
    bindings.visit_stored(&bindings.results, &mut |_| live_storage += 1);
    assert_eq!(
        live_storage, 0,
        "the scalar result does not export loop-private initialized storage"
    );
}

#[test]
fn nested_effect_only_calls_keep_distinct_executable_imports() {
    use crate::candidate_domain::construct_candidate_domain;
    use crate::evaluation_session::boundary_tests::registry;
    use seismic_ir::schedule::ScheduleStep;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "imported-state.seismic".into(),
        text: r#"fn write(dst: &mut tensor[1] f32, value: f32):
    dst[0] = value

fn forward(dst: &mut tensor[1] f32, value: f32):
    write(dst, value)

fn probe(left: &mut tensor[1] f32, right: &mut tensor[1] f32):
    forward(left, 3.0)
    forward(right, 7.0)
"#
        .into(),
    }]))
    .unwrap();
    let device = device();
    let registry = registry();
    let entry = module
        .entry(module.entry_named("probe").unwrap(), &Default::default())
        .unwrap();
    let domain = construct_candidate_domain(
        entry,
        &device,
        &registry,
        &seismic_lang::precision::PrecisionPolicy::Exact,
        &Default::default(),
    )
    .unwrap();
    let read = domain
        .read_materialized(&domain.general_construction())
        .unwrap();
    let candidate = read.candidate();
    fn scopes<'a>(
        steps: &'a [ScheduleStep],
        result: &mut Vec<&'a seismic_ir::schedule::ImportScope>,
    ) {
        for step in steps {
            match step {
                ScheduleStep::Imported { scope, body } => {
                    result.push(scope);
                    scopes(body, result);
                }
                ScheduleStep::If {
                    then_steps,
                    else_steps,
                    ..
                } => {
                    scopes(then_steps, result);
                    scopes(else_steps, result);
                }
                ScheduleStep::Repeat { body, .. } => scopes(body, result),
                ScheduleStep::Choose { options, .. } => {
                    for (_, body) in options {
                        scopes(body, result);
                    }
                }
                _ => {}
            }
        }
    }
    let mut schedule_scopes = Vec::new();
    scopes(candidate.schedule().steps(), &mut schedule_scopes);
    assert_eq!(
        schedule_scopes.len(),
        4,
        "two actual calls, each containing its actual nested call"
    );
    assert_eq!(
        schedule_scopes
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        4
    );
    let mut write_roots = candidate
        .schedule()
        .launches()
        .iter()
        .flat_map(|launch| {
            candidate
                .executable()
                .kernels()
                .kernel(launch.kernel)
                .interface()
                .bindings
                .iter()
                .filter(|binding| binding.access == seismic_ir::kernel::ops::BindingAccess::Write)
                .map(|binding| candidate.executable().storage().view(binding.view).base)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    write_roots.sort();
    write_roots.dedup();
    assert_eq!(write_roots.len(), 2, "the two Unit calls write different actual arguments");
    // Native specialization retains the complete imported execution.
    let assignment = seismic_lang::expr::PartialAssignment::new();
    let specialization = candidate
        .executable()
        .specialize_for_native(read.arena(), &assignment)
        .unwrap();
    assert_eq!(specialization.launches().count(), candidate.schedule().launches().len());
}

#[test]
fn stored_unaligned_packed_slice_is_unresolved_but_logical_reads_construct() {
    use crate::candidate_domain::{construct_candidate_domain, BodyMapping, ConstructionAllowance, ConstructionCoordinate, ConstructionPending, Materialization};
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "packed-view-map.seismic".into(),
        text: "fn probe(w: &tensor[128] q4g64) -> f32:\n    return w[1]\n\nlower probe(w: &tensor[128] q4g64) -> f32 for cpu:\n    let view = w[1:65]\n    return view[0]\n".into(),
    }])).unwrap();
    let entry = module.entry(module.entry_named("probe").unwrap(), &Default::default()).unwrap();
    let device = device();
    let registry = crate::evaluation_session::boundary_tests::registry();
    let mut domain = construct_candidate_domain(entry, &device, &registry, &seismic_lang::precision::PrecisionPolicy::Exact, &Default::default()).unwrap();
    let authored = domain.root_selections().into_iter().find(|body| body.mapping == BodyMapping::Authored).unwrap();
    let coordinate = ConstructionCoordinate::root(authored);
    let mut previous = None;
    for _ in 0..2 {
        let result = domain.advance(&coordinate, ConstructionAllowance { work_units: 100_000, wall_time: std::time::Duration::from_secs(30) });
        let Materialization::Pending(reason @ ConstructionPending::RepresentationView { .. }) = result.state else {
            panic!("stored packed map must remain unresolved: {:?}", result.state);
        };
        if let Some(previous) = &previous {
            assert_eq!(&reason, previous, "retry must retain the same source value/cursor");
            assert_eq!(result.work_units, 1, "retry must suspend at that node without lowering another step");
        }
        previous = Some(reason);
        assert!(domain.read_materialized(&coordinate).is_err(), "an unresolved view must not publish physical IR");
    }
}

#[test]
fn partially_initialized_nonaffine_carry_constructs_raw_storage_relocation() {
    use crate::candidate_domain::{construct_candidate_domain, ConstructionAllowance, Materialization};
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "partial-map-carry.seismic".into(),
        text: "fn probe() -> f32:\n    let mut y = tensor[2,3] f32\n    y[0,0] = 7.0\n    for i in 0..1:\n        y = reshape(y.T, (2,3))\n    return y[0,0]\n".into(),
    }])).unwrap();
    let entry = module.entry(module.entry_named("probe").unwrap(), &Default::default()).unwrap();
    let device = device();
    let registry = crate::evaluation_session::boundary_tests::registry();
    let mut domain = construct_candidate_domain(entry, &device, &registry, &seismic_lang::precision::PrecisionPolicy::Exact, &Default::default()).unwrap();
    let coordinate = domain.general_construction();
    let result = domain.advance(&coordinate, ConstructionAllowance { work_units: 100_000, wall_time: std::time::Duration::from_secs(30) });
    assert!(matches!(result.state, Materialization::Ready(_)), "partial mapped storage must construct without reading unspecified elements: {:?}", result.state);
}

#[test]
fn mixed_quantity_comparison_uses_the_checked_explicit_cast() {
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};
    use seismic_lang::interp::{Arg, Interpreter, OutcomeValue};
    use seismic_lang::reference_math::ReferenceScalar;
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "mixed-quantity-comparison.seismic".into(),
        text: "fn word(i: index[4294967296], x: i32) -> bool:\n    return i < x\n\nfn quantity(i: index[4294967296], x: i32) -> bool:\n    return i < 0\n".into(),
    }])).unwrap();
    for (name, expected, casts) in [("word", true, 1), ("quantity", false, 0)] {
        let entry = module.entry(module.entry_named(name).unwrap(), &Default::default()).unwrap();
        let device = device();
        let registry = crate::evaluation_session::boundary_tests::registry();
        let domain = crate::candidate_domain::construct_candidate_domain(
            entry, &device, &registry, &seismic_lang::precision::PrecisionPolicy::Exact, &Default::default(),
        ).unwrap();
        let candidate = domain.read_materialized(&domain.general_construction()).unwrap();
        let emitted_casts = candidate.candidate().schedule().steps().iter().filter(|step| matches!(step,
            seismic_ir::schedule::ScheduleStep::EvaluateHost(value)
            if matches!(value.value, seismic_ir::schedule::HostValueExpr::Word { dtype: DType::I32, .. })
        )).count();
        assert_eq!(emitted_casts, casts, "{name}: only the checked word boundary emits a cast");
        let entry = module.entry(module.entry_named(name).unwrap(), &Default::default()).unwrap();
        let mut oracle = Interpreter::new(&entry);
        let outcome = oracle.run(&[Arg::Index(2147483648_u64.into()), Arg::Scalar(ReferenceScalar::I32(0))]).unwrap();
        assert!(matches!(outcome.results().next().unwrap().value(), OutcomeValue::Scalar(ReferenceScalar::Bool(value)) if value == expected), "{name}");
    }
}
