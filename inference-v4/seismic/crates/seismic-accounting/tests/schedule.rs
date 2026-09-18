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
        start_predecessors: vec![],
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
        relationship: seismic_accounting::authority::ModelRelationship::hypothetical_execution(),
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
        lifetimes: vec![],
        static_orders: vec![],
        unmapped: Vec::new(),
    }
}

#[test]
fn selected_issue_order_preserves_overlap_and_is_checked_independently() {
    let mut second = op("second", 4, 1, 1, vec![]);
    second.start_predecessors = vec![0];
    let m = model(vec![op("first", 4, 0, 1, vec![]), second]);
    let solved = m.solve(1000).unwrap();
    assert_eq!(solved.schedule().completion, 4);

    let invalid = Schedule {
        starts: vec![1, 0],
        completion: 5,
    };
    assert!(
        m.check_schedule(&invalid)
            .unwrap_err()
            .contains("issue order")
    );

    let mut chain = model(vec![
        op("producer", 3, 0, 1, vec![]),
        op("consumer", 2, 0, 1, vec![0]),
        op("ordered", 4, 1, 1, vec![]),
    ]);
    chain.operations[2].start_predecessors = vec![1];
    assert_eq!(chain.lower_bound().unwrap(), 7);
    let solution = chain.solve(10000).unwrap();
    assert_eq!(solution.schedule().completion, 7);

    chain.operations[0].start_predecessors = vec![2];
    assert!(chain.solve(10000).unwrap_err().contains("cyclic"));
}

#[test]
fn resident_capacity_is_held_until_the_actual_completion_event() {
    let mut m = model(vec![
        op("block-a", 4, 0, 1, vec![]),
        op("block-b", 4, 0, 1, vec![]),
    ]);
    m.resources[1] = Resource {
        name: "resident bytes".into(),
        capacity: 8,
        unit: CapacityUnit::Bytes,
    };
    m.lifetimes = (0..2)
        .map(|operation| Lifetime {
            resource: 1,
            units: 8,
            begin: Event {
                operation,
                point: Point::Start,
            },
            end: Event {
                operation,
                point: Point::Completion,
            },
        })
        .collect();
    assert_eq!(m.lower_bound().unwrap(), 8);
    let solution = m.solve(10000).unwrap();
    assert_eq!(solution.schedule().completion, 8);

    assert!(
        m.check_schedule(&Schedule {
            starts: vec![0, 1],
            completion: 5
        })
        .is_err()
    );

    m.resources[1].capacity = 16;
    let solution = m.solve(10000).unwrap();
    assert_eq!(solution.schedule().completion, 5);

    m.lifetimes[0].begin.point = Point::Completion;
    m.lifetimes[0].end.point = Point::Start;
    assert!(m.check_schedule(&solution.schedule()).is_err());
}

#[test]
fn latency_is_distinct_from_issue_service() {
    let m = model(vec![op("a", 5, 0, 1, vec![]), op("b", 5, 0, 1, vec![])]);
    assert_eq!(m.lower_bound().unwrap(), 5);
    let result = m.solve(1000).unwrap();
    assert_eq!(result.schedule().completion, 6);
    assert_eq!(result.lower_bound(), 6);
    assert!(result.is_optimal());
    m.check_schedule(result.schedule()).unwrap();
}

#[test]
fn explicit_budget_does_not_turn_a_feasible_schedule_into_an_optimum() {
    let m = model(vec![op("a", 3, 0, 1, vec![]), op("b", 3, 1, 1, vec![])]);
    let unfinished = m.solve(0).unwrap();
    assert!(!unfinished.is_optimal());
    assert_eq!(unfinished.schedule().completion, 6);
    assert_eq!(unfinished.lower_bound(), 3);
    let solved = m.solve(100).unwrap();
    assert_eq!(solved.schedule().completion, 3);
    assert!(solved.is_optimal());
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
            if op.start_predecessors.iter().any(|&p| starts[i] < starts[p]) {
                return false;
            }
        }
        let event = |e: Event| {
            starts[e.operation]
                + if e.point == Point::Completion {
                    m.operations[e.operation].latency
                } else {
                    0
                }
        };
        if m.lifetimes.iter().any(|l| event(l.end) < event(l.begin)) {
            return false;
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
                for l in &m.lifetimes {
                    if l.resource == resource && event(l.begin) <= tick && tick < event(l.end) {
                        used += l.units;
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
fn large_tick_domains_are_solved_by_constraints_without_tick_enumeration() {
    let unit = 1_000_000_000_000;
    let m = model(vec![
        op("independent", 2 * unit, 0, 2 * unit, vec![]),
        op("producer", 1, 0, 1, vec![]),
        op("dependent", 10 * unit, 1, 10 * unit, vec![1]),
    ]);
    let result = m.solve(1_000).unwrap();
    assert!(result.is_optimal(), "{:?}", result);
    assert_eq!(result.schedule().completion, 10 * unit + 1);
    assert!(result.assignments_examined() < 1_000);
    assert!(result.schedule().starts[0] > result.schedule().starts[1]);

    let mut search = m.start_search().unwrap();
    for step in 1..=1_000 {
        if let SearchOutcome::Feasible(resumed) = search.advance(1).unwrap() {
            if resumed.is_optimal() {
                assert_eq!(resumed.schedule(), result.schedule());
                assert_eq!(
                    resumed.assignments_examined(),
                    result.assignments_examined()
                );
                assert_eq!(resumed.assignments_examined(), step);
                return;
            }
        }
    }
    panic!("resumption must continue the retained frontier");
}

#[test]
fn interval_propagation_matches_oracle_for_lifetimes_offsets_and_issue_orders() {
    for mask in 0..128 {
        let mut m = model(vec![
            op("a", 2, 0, 1, vec![]),
            op("b", 1, 0, 1, vec![]),
            op("c", 2, 0, 1, if mask & 1 != 0 { vec![0] } else { vec![] }),
        ]);
        m.resources[1].unit = CapacityUnit::Bytes;
        m.resources[1].capacity = if mask & 2 != 0 { 2 } else { 1 };
        m.operations[2].reservations[0].offset = u64::from(mask & 4 != 0);
        if mask & 8 != 0 {
            m.operations[1].start_predecessors.push(0);
        }
        m.lifetimes = vec![
            Lifetime {
                resource: 1,
                units: 1,
                begin: Event {
                    operation: 0,
                    point: if mask & 16 != 0 {
                        Point::Completion
                    } else {
                        Point::Start
                    },
                },
                end: Event {
                    operation: 2,
                    point: Point::Completion,
                },
            },
            Lifetime {
                resource: 1,
                units: if mask & 32 != 0 { 2 } else { 1 },
                begin: Event {
                    operation: 1,
                    point: Point::Start,
                },
                end: Event {
                    operation: 1,
                    point: if mask & 64 != 0 {
                        Point::Start
                    } else {
                        Point::Completion
                    },
                },
            },
        ];
        let expected = oracle(&m);
        match m.search(100_000).unwrap() {
            SearchOutcome::Feasible(solution) => {
                assert!(solution.is_optimal(), "mask={mask}");
                assert_eq!(solution.schedule().completion, expected, "mask={mask}");
            }
            SearchOutcome::Infeasible => assert_eq!(expected, u64::MAX, "mask={mask}"),
            SearchOutcome::Incomplete { .. } => panic!("small region incomplete: mask={mask}"),
        }
    }
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
        assert!(solved.is_optimal());
        assert_eq!(solved.schedule().completion, oracle(&m), "mask={mask}");
        assert!(solved.is_optimal());
    }
}

#[test]
fn invalid_or_incomplete_models_and_corrupted_witnesses_reject() {
    let mut m = model(vec![op("a", 2, 0, 1, vec![])]);
    let solution = m.solve(0).unwrap();
    let mut bad = solution.schedule().clone();
    bad.completion += 1;
    assert!(m.check_schedule(&bad).is_err());
    let original_model = solution.model().clone();
    m.operations[0].latency = 3;
    assert_eq!(solution.model(), &original_model);
    assert_ne!(solution.model(), &m);
    m.operations[0].latency = 2;
    m.unmapped
        .push("native instruction mapping unavailable".into());
    // Known constraints still imply a weaker floor, but no execution upper.
    assert_eq!(m.lower_bound().unwrap(), 2);
    assert!(m.check_execution_upper(solution.schedule()).is_err());
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
    assert_eq!(result.schedule().completion, 11);
    assert!(result.schedule().starts[0] > result.schedule().starts[1]);
    assert!(result.is_optimal());
    let mut reordered = m.clone();
    reordered.operations.swap(0, 2);
    // The dependency remains on operation b, now between c and a in source order.
    let other = reordered.solve(100_000).unwrap();
    assert_eq!(other.schedule().completion, result.schedule().completion);
}

#[test]
fn empty_and_zero_latency_models_and_simultaneous_resource_claims_are_explicit() {
    let mut m = model(vec![]);
    assert_eq!(m.solve(0).unwrap().schedule().completion, 0);
    m.operations.push(Operation {
        name: "publication".into(),
        predecessors: vec![],
        start_predecessors: vec![],
        latency: 0,
        reservations: vec![],
    });
    let zero = m.solve(0).unwrap();
    assert_eq!(zero.schedule().completion, 0);
    assert!(zero.is_optimal());
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
    assert_eq!(result.schedule().completion, oracle(&m));
    assert!(result.is_optimal());
}

#[test]
fn infeasible_serial_seed_does_not_hide_a_legal_interleaving() {
    let mut m = model(vec![
        op("allocate-a", 1, 0, 1, vec![]),
        op("allocate-b", 1, 0, 1, vec![]),
        op("release-a", 1, 0, 1, vec![0]),
        op("release-b", 1, 0, 1, vec![1]),
    ]);
    m.resources[1].unit = CapacityUnit::Bytes;
    m.lifetimes = (0..2)
        .map(|operation| Lifetime {
            resource: 1,
            units: 1,
            begin: Event {
                operation,
                point: Point::Start,
            },
            end: Event {
                operation: operation + 2,
                point: Point::Completion,
            },
        })
        .collect();
    assert!(
        m.check_schedule(&Schedule {
            starts: vec![0, 1, 2, 3],
            completion: 4
        })
        .is_err()
    );
    assert!(matches!(
        m.search(0).unwrap(),
        SearchOutcome::Incomplete { lower_bound: 4 }
    ));
    let solution = m.solve(100_000).unwrap();
    assert_eq!(solution.schedule().completion, 4);
    assert!(solution.is_optimal());
    m.check_schedule(solution.schedule()).unwrap();
    assert!(
        solution.schedule().starts[2] < solution.schedule().starts[1]
            || solution.schedule().starts[3] < solution.schedule().starts[0]
    );

    // A zero-latency join at the inclusive horizon must remain legal even when
    // no serial witness exists to seed the search.
    m.operations.push(Operation {
        name: "completion".into(),
        predecessors: vec![2, 3],
        start_predecessors: vec![],
        latency: 0,
        reservations: vec![],
    });
    let solution = m.solve(100_000).unwrap();
    assert_eq!(solution.schedule().starts[4], 4);
    assert_eq!(solution.schedule().completion, 4);
}

#[test]
fn schedule_infeasibility_requires_completed_search() {
    let mut m = model(vec![op("oversubscribed", 1, 0, 1, vec![])]);
    m.operations[0].reservations[0].units = 2;
    assert!(matches!(
        m.search(0).unwrap(),
        SearchOutcome::Incomplete { .. }
    ));
    assert_eq!(m.search(100).unwrap(), SearchOutcome::Infeasible);
}
