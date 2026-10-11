//! Private tensors of the universal member are acquired where the schedule
//! reaches them. Their backing and geometry are the actual per-visit values,
//! never a launch-local envelope over a declared or I32 domain.
use super::*;
use crate::candidate_domain::{
    construct_candidate_domain, BodyMapping, CandidateDomain, CapacityPending, CapacityReason,
    ConstructionAllowance, ConstructionCoordinate, ConstructionPending, Materialization,
};
use crate::realization::demand_driven_tests::FakeTarget;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::expr::Assignment;
use seismic_lang::precision::PrecisionPolicy;

/// One axis of a reached allocation's geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Axis {
    /// Fixed for the whole invocation.
    Fixed(u64),
    /// An execution value of the visit that acquires the tensor.
    Actual(seismic_lang::expr::NatExpr),
}

fn with_source_domain<R>(
    source: &str,
    inspect: impl FnOnce(CandidateDomain<'_, FakeTarget>) -> R,
) -> Result<R, crate::errors::PreparationError> {
    let module = check_source(SourceSet::new(vec![SourceFile {
        path: "source-capacity.seismic".into(),
        text: source.into(),
    }]))
    .unwrap();
    let entry = module
        .entry(
            module.entry_named("probe").unwrap(),
            &ElementBindings::default(),
        )
        .unwrap();
    let mut description = crate::realization::demand_driven_tests::device_parts();
    for dtype in [DType::F32, DType::I32, DType::U32, DType::Bool] {
        description
            .dtypes
            .representations
            .insert(registry::dense(dtype));
        description.dtypes.scalars.insert(dtype);
    }
    description.limits.max_allocation_bytes = 1 << 40;
    description.limits.max_allocation_alignment = 16;
    description.limits.max_index_bits = 64;
    description.limits.max_bindings = 64;
    description.limits.max_argument_bytes = 4096;
    description.limits.max_workgroup_bytes = 65536;
    description.limits.max_grid = [65536; 3];
    let device = seismic_native_target::DeviceDescription::new(description).unwrap();
    let registry = crate::realization::demand_driven_tests::registry();
    let domain = construct_candidate_domain(entry, &device, &registry, &PrecisionPolicy::Exact)?;
    Ok(inspect(domain))
}

fn reached_geometry(source: &str) -> Result<Vec<Vec<Axis>>, crate::errors::PreparationError> {
    with_source_domain(source, |domain| {
        let parts = domain.into_parts();
        let family = &parts.materialized.first().family;
        assert!(
            family
                .local_allocations()
                .into_locals()
                .iter()
                .all(Vec::is_empty),
            "private tensors are not launch-local envelopes"
        );
        family
            .global_allocations()
            .allocations()
            .iter()
            .filter(|allocation| {
                allocation.acquisition == seismic_ir::storage::AllocationAcquisition::Reached
            })
            .map(|allocation| {
                allocation
                    .geometry
                    .as_ref()
                    .expect("a reached tensor owns its geometry")
                    .extents
                    .iter()
                    .map(
                        |axis| match parts.arena.eval_nat_u64(*axis, &Assignment::new()) {
                            Ok(value) => Axis::Fixed(value),
                            Err(_) => Axis::Actual(*axis),
                        },
                    )
                    .collect()
            })
            .collect()
    })
}

#[test]
fn source_capacity_pairs_helper_dimensions_with_actual_checked_slice() {
    let geometry = reached_geometry(
        r#"fn seed[M,K](x: &tensor[M,K] i32) -> i32:
    let mut scratch = tensor[M,K] i32
    scratch[:] = ones_like(scratch)
    return reduce(reduce(scratch, 1, sum), 0, sum)

fn probe(input: &tensor[8,2] i32, visible: &tensor[4,2] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let lo = visible[i,0]
        let hi = visible[i,1]
        if hi > lo:
            out[i] = seed(input[lo:hi,:])
"#,
    )
    .unwrap();
    // The helper's `M` is the actual checked slice extent of each visit and
    // its `K` the caller's fixed axis.
    assert!(
        geometry
            .iter()
            .any(|axes| matches!(axes.as_slice(), [Axis::Actual(_), Axis::Fixed(2)])),
        "{geometry:?}"
    );
}

#[test]
fn source_capacity_preserves_varying_two_dimensional_snapshot_geometry() {
    let geometry = reached_geometry(
        r#"fn probe(out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let mut local = tensor[i+1,i+2] i32
        local[:] = ones_like(local)
        let captured = local + local
        local[0,0] = 7
        out[i] = captured[0,0] + captured[i,i+1]
"#,
    )
    .unwrap();
    // The captured snapshot is stored with exactly the original's actual
    // per-visit axes.
    let varying = geometry
        .iter()
        .filter(|axes| matches!(axes.as_slice(), [Axis::Actual(_), Axis::Actual(_)]))
        .collect::<Vec<_>>();
    assert!(varying.len() >= 2, "{geometry:?}");
    assert!(
        varying.iter().all(|axes| *axes == varying[0]),
        "{geometry:?}"
    );
}

#[test]
fn source_capacity_composes_transpose_point_zero_and_full_width_calls() {
    let geometry = reached_geometry(
        r#"fn seed[N](x: &tensor[N] i32) -> i32 where N >= 0:
    let scratch = ones_like(x)
    return reduce(scratch, 0, sum)

fn probe(input: &tensor[8,2] i32, visible: &tensor[4,2] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let lo = visible[i,0]
        let hi = visible[i,1]
        let moved = input.T
        let dynamic = seed(moved[1,lo:hi])
        let empty = seed(input[0:0,1])
        let full = seed(input[:,1])
        out[i] = dynamic + empty + full
"#,
    )
    .unwrap();
    for expected in [Axis::Fixed(8), Axis::Fixed(0)] {
        assert!(
            geometry
                .iter()
                .any(|axes| axes.as_slice() == [expected.clone()]),
            "{geometry:?}"
        );
    }
    assert!(
        geometry
            .iter()
            .any(|axes| matches!(axes.as_slice(), [Axis::Actual(_)])),
        "{geometry:?}"
    );
}

/// These checks select the participant mapping explicitly. The universal
/// Portable member executes mathematical quantities with its exact host cursor
/// and must remain valid even when an Independent native member cannot form.
fn independent_materialization(source: &str) -> Materialization {
    with_source_domain(source, |mut domain| {
        assert_eq!(
            domain.constructed().count(),
            1,
            "universal host member formed"
        );
        let selection = domain
            .root_selections()
            .into_iter()
            .find(|selection| selection.mapping == BodyMapping::Independent)
            .expect("portable body exposes the independent mapping");
        let state = domain
            .advance(
                &ConstructionCoordinate::root(selection),
                ConstructionAllowance {
                    work_units: 100_000,
                    wall_time: std::time::Duration::from_secs(30),
                },
            )
            .state;
        if let Materialization::Pending(ConstructionPending::Capacity(pending)) = &state {
            let reference = domain.reference();
            let view = reference.view();
            let function = view.program().function(pending.value.function());
            let value = function.value(pending.value);
            eprintln!("pending quantity {:?}: {value:#?}", pending.value);
            if let ValueOrigin::Node(node) = value.origin {
                let node = function.node(node);
                eprintln!("quantity producer: {node:#?}");
                if let SemanticNodeView::Primitive {
                    primitive: PrimitiveId::Symbolic(expression),
                    ..
                } = node.view()
                {
                    eprintln!(
                        "quantity expression: {:#?}",
                        view.arena().view((*expression).into())
                    );
                }
            }
        }
        state
    })
    .expect("exact universal host construction remains lawful")
}

#[test]
fn declared_slice_width_materializes_supported_independent_quantities() {
    let state = independent_materialization(
        r#"fn probe(input: &tensor[8] i32, starts: &tensor[4] i32, out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let start = starts[i]
        let view = input[start:start+2]
        out[i] = view[0]
"#,
    );
    assert!(
        matches!(&state, Materialization::Ready(_)),
        "actual independent state: {state:?}"
    );
}

#[test]
fn declared_slice_width_keeps_host_quantity_and_declines_independent_narrowing() {
    let state = independent_materialization(
        r#"fn probe(input: &tensor[8] i32, starts: &tensor[4] i32, width: index[4294967298], out: &mut tensor[4] i32):
    parallel for i in 0..4:
        let start = starts[i]
        let view = input[start:start+i32(width)]
        out[i] = view[0]
"#,
    );
    assert!(
        matches!(
            &state,
            Materialization::Pending(ConstructionPending::Capacity(CapacityPending {
                reason: CapacityReason::NativeQuantityUnrepresentable,
                ..
            }))
        ),
        "actual independent state: {state:?}"
    );
}
