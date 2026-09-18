use seismic_accounting::schedule::*;
fn op(
    name: &str,
    latency: u64,
    resource: usize,
    duration: u64,
    predecessors: Vec<usize>,
) -> Operation {
    Operation {
        name: name.into(),
        latency,
        predecessors,
        reservations: vec![Reservation {
            resource,
            offset: 0,
            duration,
            units: 1,
        }],
    }
}
fn model(operations: Vec<Operation>) -> Model {
    Model {
        identity: "test discrete hardware model, tick=1, isolated workload".into(),
        timebase: Timebase {
            seconds_numerator: 1,
            seconds_denominator: 1_000_000_000,
        },
        resources: vec![
            Resource {
                name: "issue-a".into(),
                capacity: 1,
                unit: CapacityUnit::Slots,
            },
            Resource {
                name: "issue-b".into(),
                capacity: 1,
                unit: CapacityUnit::Slots,
            },
        ],
        operations,
        unmapped: Vec::new(),
    }
}

#[test]
fn latency_is_distinct_from_issue_service_and_proof_is_checked() {
    let m = model(vec![op("a", 5, 0, 1, vec![]), op("b", 5, 0, 1, vec![])]);
    let result = m.solve(1000).unwrap();
    assert_eq!(result.schedule.completion, 6);
    assert_eq!(result.lower_bound, 5);
    assert_eq!(result.proof, Some(Proof::Exhaustive));
    assert!(matches!(
        m.verify_solution(&result, 1000).unwrap(),
        Verification::Verified { .. }
    ));
    assert!(matches!(
        m.verify_solution(&result, 0).unwrap(),
        Verification::BudgetExhausted { .. }
    ));
    let mut false_proof = result.clone();
    false_proof.proof = Some(Proof::LowerBound);
    assert!(m.verify_solution(&false_proof, 1000).is_err());
    false_proof.proof = Some(Proof::Exhaustive);
    false_proof.schedule = Schedule {
        starts: vec![0, 5],
        completion: 10,
    };
    assert!(m
        .verify_solution(&false_proof, 1000)
        .unwrap_err()
        .contains("better feasible"));
}

#[test]
fn explicit_budget_does_not_turn_a_feasible_schedule_into_an_optimum() {
    let m = model(vec![op("a", 3, 0, 1, vec![]), op("b", 3, 1, 1, vec![])]);
    let unfinished = m.solve(0).unwrap();
    assert_eq!(unfinished.proof, None);
    assert_eq!(unfinished.schedule.completion, 6);
    assert_eq!(unfinished.lower_bound, 3);
    assert!(m.verify_solution(&unfinished, 100).is_err());
    let solved = m.solve(100).unwrap();
    assert_eq!(solved.schedule.completion, 3);
    assert_eq!(solved.proof, Some(Proof::LowerBound));
    assert!(matches!(
        m.verify_solution(&solved, 0).unwrap(),
        Verification::Verified { .. }
    ));
}

// Independent small-domain oracle: checks capacity tick by tick rather than with
// event sweeps, and enumerates all start vectors rather than dependency order.
fn oracle(m: &Model) -> u64 {
    fn valid(m: &Model, starts: &[u64], completion: u64) -> bool {
        for (i, op) in m.operations.iter().enumerate() {
            if op
                .predecessors
                .iter()
                .any(|&p| starts[i] < starts[p] + m.operations[p].latency)
            {
                return false;
            }
        }
        for tick in 0..completion {
            for (resource, capacity) in m.resources.iter().enumerate() {
                let mut used = 0;
                for (op, start) in m.operations.iter().zip(starts) {
                    for r in &op.reservations {
                        if r.resource == resource
                            && tick >= start + r.offset
                            && tick < start + r.offset + r.duration
                        {
                            used += r.units;
                        }
                    }
                }
                if used > capacity.capacity {
                    return false;
                }
            }
        }
        true
    }
    fn enumerate(m: &Model, starts: &mut [u64], index: usize, horizon: u64, best: &mut u64) {
        if index == starts.len() {
            let end = starts
                .iter()
                .zip(&m.operations)
                .map(|(s, o)| s + o.latency)
                .max()
                .unwrap_or(0);
            if end < *best && valid(m, starts, end) {
                *best = end;
            }
            return;
        }
        for start in 0..=horizon {
            starts[index] = start;
            enumerate(m, starts, index + 1, horizon, best);
        }
    }
    let horizon = m.operations.iter().map(|o| o.latency).sum();
    let mut best = u64::MAX;
    enumerate(m, &mut vec![0; m.operations.len()], 0, horizon, &mut best);
    best
}

#[test]
fn exact_solver_agrees_with_independent_oracle_across_resource_and_dependency_patterns() {
    for mask in 0..64 {
        let mut m = model(vec![
            op("a", 2, 0, 1, vec![]),
            op(
                "b",
                3,
                usize::from(mask & 1 != 0),
                2,
                if mask & 2 != 0 { vec![0] } else { vec![] },
            ),
            op(
                "c",
                2,
                usize::from(mask & 4 != 0),
                1,
                if mask & 8 != 0 { vec![1] } else { vec![] },
            ),
        ]);
        m.resources[0].capacity = if mask & 16 != 0 { 2 } else { 1 };
        m.operations[2].reservations[0].offset = u64::from(mask & 32 != 0);
        let solved = m.solve(100_000).unwrap();
        assert!(solved.proof.is_some());
        assert_eq!(solved.schedule.completion, oracle(&m), "mask={mask}");
        assert!(matches!(
            m.verify_solution(&solved, 100_000).unwrap(),
            Verification::Verified { .. }
        ));
    }
}

#[test]
fn invalid_or_incomplete_models_and_corrupted_witnesses_reject() {
    let mut m = model(vec![op("a", 2, 0, 1, vec![])]);
    let solution = m.solve(0).unwrap();
    let mut bad = solution.clone();
    bad.schedule.completion += 1;
    assert!(m.verify_solution(&bad, 100).is_err());
    bad = solution.clone();
    bad.lower_bound = 0;
    assert!(m.verify_solution(&bad, 100).is_err());
    m.unmapped
        .push("native instruction mapping unavailable".into());
    assert!(m.solve(100).is_err());
    m.unmapped.clear();
    let duplicate = m.operations[0].reservations[0].clone();
    m.operations[0].reservations.push(duplicate);
    assert!(m.solve(100).is_err()); // Two simultaneous claims cannot fit a capacity of one.
    m.operations[0].reservations.pop();
    m.operations[0].predecessors.push(0);
    assert!(m.solve(100).is_err());
}

#[test]
fn optimum_can_require_delaying_an_earlier_independent_operation() {
    let m = model(vec![
        op("a", 2, 0, 2, vec![]),
        op("b", 1, 0, 1, vec![]),
        op("c", 10, 1, 10, vec![1]),
    ]);
    let result = m.solve(100_000).unwrap();
    assert_eq!(result.schedule.completion, 11);
    assert!(result.schedule.starts[0] > result.schedule.starts[1]);
    assert!(matches!(
        m.verify_solution(&result, 0).unwrap(),
        Verification::Verified { .. }
    ));
    let mut reordered = m.clone();
    reordered.operations.swap(0, 2);
    // The dependency remains on operation b, now between c and a in source order.
    let other = reordered.solve(100_000).unwrap();
    assert_eq!(other.schedule.completion, result.schedule.completion);
}

#[test]
fn empty_and_zero_latency_models_and_simultaneous_resource_claims_are_explicit() {
    let mut m = model(vec![]);
    assert_eq!(m.solve(0).unwrap().schedule.completion, 0);
    m.operations.push(Operation {
        name: "publication".into(),
        predecessors: vec![],
        latency: 0,
        reservations: vec![],
    });
    let zero = m.solve(0).unwrap();
    assert_eq!(zero.schedule.completion, 0);
    assert!(matches!(
        m.verify_solution(&zero, 0).unwrap(),
        Verification::Verified { .. }
    ));
    m = model(vec![op("a", 3, 0, 2, vec![]), op("b", 3, 1, 2, vec![])]);
    m.resources[0].capacity = 2;
    m.operations[0].reservations[0].units = 2;
    m.operations[1].reservations.push(Reservation {
        resource: 0,
        offset: 1,
        duration: 1,
        units: 1,
    });
    let result = m.solve(100_000).unwrap();
    assert_eq!(result.schedule.completion, oracle(&m));
    assert!(matches!(
        m.verify_solution(&result, 100_000).unwrap(),
        Verification::Verified { .. }
    ));
}
