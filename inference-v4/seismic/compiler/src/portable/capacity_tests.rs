use super::*;
use crate::candidate_domain::construct_candidate_domain;
use seismic_lang::checked::{check_source, SourceFile, SourceSet};
use seismic_lang::entry::ElementBindings;
use seismic_lang::expr::Assignment;
use seismic_lang::precision::PrecisionPolicy;

fn reservations(source: &str) -> Result<Vec<Vec<u64>>, crate::errors::PreparationError> {
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
    let device = seismic_target::DeviceDescription::new(description).unwrap();
    let registry = crate::realization::demand_driven_tests::registry();
    let domain = construct_candidate_domain(
        entry,
        &device,
        &registry,
        &PrecisionPolicy::Exact,
    )?;
    let parts = domain.into_parts();
    Ok(parts
        .materialized
        .first()
        .family
        .local_allocations()
        .into_locals()
        .iter()
        .flatten()
        .map(|local| {
            local
                .extents
                .iter()
                .map(|axis| parts.arena.eval_nat_u64(*axis, &Assignment::new()).unwrap())
                .collect()
        })
        .collect())
}

#[test]
fn source_capacity_pairs_helper_dimensions_with_actual_checked_slice() {
    let capacities = reservations(
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
    assert!(
        capacities.iter().any(|axes| axes == &[8, 2]),
        "{capacities:?}"
    );
    assert!(
        capacities
            .iter()
            .all(|axes| axes.iter().all(|axis| *axis <= 8)),
        "slice capacities must use the original axis, not the full I32 domain: {capacities:?}"
    );
}

#[test]
fn source_capacity_preserves_varying_two_dimensional_snapshot_geometry() {
    let capacities = reservations(
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
    assert!(
        capacities
            .iter()
            .filter(|axes| axes.as_slice() == [4, 5])
            .count()
            >= 2,
        "original and captured storage share the source envelope: {capacities:?}"
    );
    assert!(
        capacities
            .iter()
            .all(|axes| axes.iter().all(|axis| *axis <= 5)),
        "{capacities:?}"
    );
}

#[test]
fn source_capacity_composes_transpose_point_zero_and_full_width_calls() {
    let capacities = reservations(
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
    assert!(
        capacities.iter().any(|axes| axes.as_slice() == [8]),
        "{capacities:?}"
    );
    assert!(
        capacities.iter().any(|axes| axes.as_slice() == [0]),
        "{capacities:?}"
    );
    assert!(
        capacities
            .iter()
            .all(|axes| axes.iter().all(|axis| *axis <= 8)),
        "{capacities:?}"
    );
}
