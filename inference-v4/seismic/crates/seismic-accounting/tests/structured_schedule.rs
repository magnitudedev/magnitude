use seismic_accounting::{authority::ModelRelationship, schedule::{*, structured::*}, workload::DerivationError};
use std::sync::Arc;

fn leaf(latency: u64, duration: u64) -> Arc<Node> {
    Arc::new(Node::Operation(Operation { name: "service".into(), predecessors: vec![], start_predecessors: vec![], latency,
        reservations: vec![Reservation { resource: 0, offset: 0, duration, units: 1 }] }))
}
fn model(root: Arc<Node>, capacity: u64) -> Structured {
    Structured {
        relationship: ModelRelationship::hypothetical_execution(), identity: "structured scheduling fixture".into(),
        timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
        resources: vec![Resource { name: "issue".into(), capacity, unit: CapacityUnit::Slots },
            Resource { name: "resident".into(), capacity: 2, unit: CapacityUnit::Slots }], root, unmapped: vec![],
    }
}
fn check_against_oracle(model: &Structured) {
    let flat = model.expand(100).unwrap();
    let solution = flat.solve(100_000).unwrap();
    assert!(solution.is_optimal());
    assert!(model.lower_bound().unwrap() <= solution.schedule().completion);
    if let Some(witness) = model.compact_witness().unwrap() {
        let (expanded, schedule) = witness.expand(100).unwrap();
        assert_eq!(expanded, flat);
        assert_eq!(witness.completion(), schedule.completion);
        if witness.is_optimal() { assert_eq!(witness.completion(), solution.schedule().completion); }
    }
}

#[test]
fn serial_parallel_and_resident_scopes_match_flat_scheduling_semantics() {
    for order in [Order::Serial, Order::Parallel] {
        for capacity in [1, 2] {
            let body = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: Arc::new(Node::Compose {
                order: Order::Serial, children: vec![leaf(2, 1), leaf(1, 1)],
            }) });
            let root = Arc::new(Node::Repeat { order, count: 2, body });
            check_against_oracle(&model(root, capacity));
        }
    }
}

#[test]
fn trillion_serial_operations_have_a_compact_exact_model_witness() {
    let million = Arc::new(Node::Repeat { order: Order::Serial, count: 1_000_000, body: leaf(3, 1) });
    let model = model(Arc::new(Node::Repeat { order: Order::Serial, count: 1_000_000, body: million }), 1);
    let witness = model.compact_witness().unwrap().unwrap();
    assert_eq!(witness.completion(), 3_000_000_000_000);
    assert!(witness.is_optimal());
    assert!(matches!(model.expand(100), Err(DerivationError::Exhausted(_))));
}

#[test]
fn feasible_serialization_does_not_claim_parallel_optimality() {
    let model = model(Arc::new(Node::Repeat { order: Order::Parallel, count: 2, body: leaf(3, 1) }), 1);
    let witness = model.compact_witness().unwrap().unwrap();
    assert_eq!(witness.lower_bound(), 4);
    assert_eq!(witness.completion(), 6);
    assert!(!witness.is_optimal());
    let solution = model.expand(10).unwrap().solve(10_000).unwrap();
    assert!(solution.is_optimal());
    assert_eq!(solution.schedule().completion, 4);
}

#[test]
fn nested_residency_and_instruction_reservations_share_capacity() {
    let root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: leaf(3, 3) });
    let constrained = model(root.clone(), 1);
    assert!(constrained.compact_witness().unwrap().is_none());
    assert!(matches!(constrained.expand(10).unwrap().search(1000).unwrap(), SearchOutcome::Infeasible));
    check_against_oracle(&model(root, 2));
}

#[test]
fn missing_mappings_and_optimistic_models_cannot_supply_execution_witnesses() {
    let mut model = model(leaf(3, 1), 1);
    model.unmapped.push("missing service".into());
    assert!(model.compact_witness().is_err());
    assert_eq!(model.lower_bound().unwrap(), 3);
    model.unmapped.clear();
    model.relationship = ModelRelationship::OptimisticRelaxation;
    assert!(model.compact_witness().is_err());
}

#[test]
fn empty_parallel_repetition_preserves_the_surrounding_dependency() {
    let empty = Arc::new(Node::Repeat { order: Order::Parallel, count: u64::MAX,
        body: Arc::new(Node::Compose { order: Order::Serial, children: vec![] }) });
    let model = model(Arc::new(Node::Compose { order: Order::Serial, children: vec![leaf(2, 1), empty, leaf(3, 1)] }), 1);
    check_against_oracle(&model);
    assert_eq!(model.expand(2).unwrap().operations[1].predecessors, vec![0]);
}

#[test]
fn resident_parallel_repetition_has_an_exact_compact_wave_schedule() {
    let group = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: leaf(3, 1) });
    for count in 1..=4 {
        let model = model(Arc::new(Node::Repeat { order: Order::Parallel, count, body: group.clone() }), 100);
        let witness = model.compact_witness().unwrap().unwrap();
        assert!(witness.is_optimal());
        assert_eq!(witness.completion(), count.div_ceil(2) * 3);
        check_against_oracle(&model);
    }
    let count = 1_000_000_000_001u64;
    let model = model(Arc::new(Node::Repeat { order: Order::Parallel, count, body: group }), 100);
    let witness = model.compact_witness().unwrap().unwrap();
    assert!(witness.is_optimal());
    assert_eq!(witness.completion(), count.div_ceil(2) * 3);
    assert!(matches!(witness.expand(100), Err(DerivationError::Exhausted(_))));
}

#[test]
fn repeated_work_shares_ancestor_residency_once() {
    let group = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: leaf(3, 1) });
    let root = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: Arc::new(Node::Repeat {
        order: Order::Parallel, count: 3, body: group,
    }) });
    let mut model = model(root, 100);
    model.resources[1].capacity = 3;
    let witness = model.compact_witness().unwrap().unwrap();
    assert_eq!(witness.completion(), 6);
    assert!(witness.is_optimal());
    check_against_oracle(&model);
}

#[test]
fn factoring_parallel_members_preserves_flat_constraints_and_checked_starts() {
    let body = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: leaf(3, 1) });
    let different = Arc::new(Node::Scope { reservations: vec![(1, 2)], body: leaf(3, 1) });
    let original = vec![body.clone(), body.clone(), body.clone(), different, body];
    let mut factored = Vec::new();
    for node in &original { Node::append_parallel(&mut factored, node.clone()).unwrap(); }
    assert_eq!(factored.len(), 3);
    assert!(matches!(factored[0].as_ref(), Node::Repeat { count: 3, order: Order::Parallel, .. }));
    let original = model(Arc::new(Node::Compose { order: Order::Parallel, children: original }), 100);
    let factored = model(Arc::new(Node::Compose { order: Order::Parallel, children: factored }), 100);
    assert_eq!(original.expand(100).unwrap(), factored.expand(100).unwrap());
    factored.compact_witness().unwrap().unwrap().expand(100).unwrap();
}

#[test]
fn parallel_factoring_overflow_leaves_members_unchanged() {
    let body = leaf(1, 1);
    let mut members = vec![Arc::new(Node::Repeat { order: Order::Parallel, count: u64::MAX, body: body.clone() })];
    let before = members.clone();
    assert!(Node::append_parallel(&mut members, body).is_err());
    assert_eq!(members, before);
}

#[test]
fn zero_duration_resident_scopes_have_empty_capacity_intervals() {
    let body = Arc::new(Node::Scope { reservations: vec![(1, 3)], body: Arc::new(Node::Compose {
        order: Order::Serial, children: vec![],
    }) });
    let model = model(Arc::new(Node::Repeat { order: Order::Parallel, count: 4, body }), 1);
    let witness = model.compact_witness().unwrap().unwrap();
    assert!(witness.is_optimal());
    assert_eq!(witness.completion(), 0);
    witness.expand(8).unwrap();
}
