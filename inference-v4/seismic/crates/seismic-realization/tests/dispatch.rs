//! The one common linear/runtime iteration geometry: the hosted
//! `LinearIterationMap` is the single definition.

use seismic_lang::{
    logical::{RuntimeExtent, RuntimeScalarExpr},
    sym::Sym,
    types::{ExtentExpr, RuntimeExtentId},
};
use seismic_realization::dispatch::{
    LaunchCondition, LinearIterationMap, LinearMapError, LinearTotal, Traversal,
};
use std::collections::BTreeMap;

#[test]
fn one_pass_and_grid_stride_cover_the_same_domain() {
    let extents = vec![ExtentExpr::Static(7), ExtentExpr::Static(3)];
    let grid_stride = LinearIterationMap::linear(&extents, &BTreeMap::new())
        .expect("the map builds")
        .with_participants(Sym::constant(8));
    assert_eq!(grid_stride.traversal, Traversal::GridStride);
    // Grid-stride: participant `i` visits `i, i + p, i + 2p, …`, tail masked.
    let mut strided = Vec::new();
    for participant in 0..8u64 {
        let mut linear = participant;
        while linear < 21 {
            strided.push(grid_stride.delinearize(linear).unwrap());
            linear += 8;
        }
    }
    // One pass: each participant owns exactly one coordinate, no tail.
    let one_pass = LinearIterationMap::serialized(&grid_stride);
    assert_eq!(one_pass.traversal, Traversal::OnePass);
    assert!(!one_pass.tail_mask);
    let mut ascending = (0..21)
        .map(|linear| one_pass.delinearize(linear).unwrap())
        .collect::<Vec<_>>();
    ascending.sort();
    let mut sorted = strided;
    sorted.sort();
    assert_eq!(sorted, ascending);
}

#[test]
fn zero_work_is_a_retained_launch_condition() {
    let empty = LinearIterationMap::linear(&[ExtentExpr::Static(0)], &BTreeMap::new())
        .expect("the map builds");
    assert_eq!(empty.launch_condition(), LaunchCondition::AlwaysSkip);
    assert_eq!(empty.delinearize(0), None);
    let nonempty = LinearIterationMap::linear(&[ExtentExpr::Static(4)], &BTreeMap::new())
        .expect("the map builds");
    assert_eq!(nonempty.launch_condition(), LaunchCondition::Execute);
}

#[test]
fn runtime_extents_are_retained_not_replaced_by_capacities() {
    let id = RuntimeExtentId(0);
    let runtime = RuntimeExtent {
        id,
        value: RuntimeScalarExpr::Extent(id),
        capacity: 4096,
        expected: None,
    };
    let map =
        LinearIterationMap::linear(&[ExtentExpr::Runtime(id)], &BTreeMap::from([(id, runtime)]))
            .expect("the map builds");
    // Planning/resource accounting uses the checked capacity bound…
    assert_eq!(map.total_symbol().as_constant(), Some(4096));
    // …while the retained total is the exact runtime product.
    match &map.total {
        LinearTotal::Runtime {
            product, capacity, ..
        } => {
            assert_eq!(*capacity, 4096);
            assert_eq!(*product, RuntimeScalarExpr::Extent(id));
        }
        other => panic!("the total must be runtime, got {other:?}"),
    }
}

#[test]
fn expected_extents_price_cost_but_never_geometry() {
    let id = RuntimeExtentId(0);
    let runtime = RuntimeExtent {
        id,
        value: RuntimeScalarExpr::Extent(id),
        capacity: 4096,
        expected: Some(128),
    };
    let map = LinearIterationMap::linear(
        &[ExtentExpr::Runtime(id), ExtentExpr::Static(3)],
        &BTreeMap::from([(id, runtime)]),
    )
    .expect("the map builds");
    // Geometry and resources keep the capacity bound…
    assert_eq!(map.total_symbol().as_constant(), Some(4096 * 3));
    assert_eq!(map.total.bound(), 4096 * 3);
    // …while cost prices the workload's expectation.
    assert_eq!(map.total.expected(), 128 * 3);
    assert_eq!(map.cost_symbol().as_constant(), Some(128 * 3));
    // An unstated axis leaves the domain priced at capacity.
    let unstated = RuntimeExtent {
        id,
        value: RuntimeScalarExpr::Extent(id),
        capacity: 4096,
        expected: None,
    };
    let map = LinearIterationMap::linear(
        &[ExtentExpr::Runtime(id)],
        &BTreeMap::from([(id, unstated)]),
    )
    .expect("the map builds");
    assert_eq!(map.cost_symbol(), map.total_symbol());
}

#[test]
fn checked_totals_never_wrap() {
    let huge = LinearIterationMap::linear(
        &[ExtentExpr::Static(u64::MAX), ExtentExpr::Static(2)],
        &BTreeMap::new(),
    );
    assert_eq!(huge.unwrap_err(), LinearMapError::StaticOverflow);
    let id = RuntimeExtentId(0);
    let runtime = RuntimeExtent {
        id,
        value: RuntimeScalarExpr::Extent(id),
        capacity: u64::MAX,
        expected: None,
    };
    let runtime_overflow = LinearIterationMap::linear(
        &[ExtentExpr::Runtime(id), ExtentExpr::Runtime(id)],
        &BTreeMap::from([(id, runtime)]),
    );
    assert_eq!(
        runtime_overflow.unwrap_err(),
        LinearMapError::CapacityOverflow
    );
    let unresolved = LinearIterationMap::linear(
        &[ExtentExpr::Sym(Sym::param("unresolved"))],
        &BTreeMap::new(),
    );
    assert_eq!(unresolved.unwrap_err(), LinearMapError::UnresolvedSymbol);
}
