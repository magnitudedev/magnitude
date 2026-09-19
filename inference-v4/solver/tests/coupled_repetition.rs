use magnitude_solver::{
    result::StopReason,
    scheduling::repetition::{
        CoupledActivity, CoupledOutcome, CoupledRepetition, CoupledTemplate, RepeatedPrecedence,
        RepetitionMethod, RepetitionStrategy,
    },
    Error, Limits, Options,
};
use std::{sync::Arc, time::Duration};

fn activity(duration: u64, demand: &[u64]) -> CoupledActivity {
    CoupledActivity {
        duration,
        demand: demand.to_vec(),
    }
}

fn edge(before: usize, after: usize, distance: u64, lag: u64) -> RepeatedPrecedence {
    RepeatedPrecedence {
        before,
        after,
        distance,
        lag,
    }
}

// Independent integer-time oracle: enumerate starts and check every unit-time
// resource occupancy directly. Do not reuse propagation, template validation,
// energy bounds, or either compact scheduling formula.
fn direct_cost(
    template: &CoupledTemplate,
    count: usize,
    horizon: u64,
    starts: &[u64],
) -> Option<u64> {
    let width = template.activities.len();
    assert_eq!(starts.len(), width * count);
    let mut makespan = 0;
    for copy in 0..count {
        for (stage, activity) in template.activities.iter().enumerate() {
            let end = starts[copy * width + stage] as u128 + activity.duration as u128;
            if end > horizon as u128 {
                return None;
            }
            makespan = makespan.max(end as u64);
        }
        for edge in &template.precedence {
            let next = copy as u128 + edge.distance as u128;
            if next < count as u128
                && starts[copy * width + edge.before] as u128
                    + template.activities[edge.before].duration as u128
                    + edge.lag as u128
                    > starts[next as usize * width + edge.after] as u128
            {
                return None;
            }
        }
    }
    for time in 0..makespan {
        for (resource, &capacity) in template.capacities.iter().enumerate() {
            let mut demand = 0u128;
            for copy in 0..count {
                for (stage, activity) in template.activities.iter().enumerate() {
                    let start = starts[copy * width + stage];
                    if start <= time && time < start + activity.duration {
                        demand += activity.demand[resource] as u128;
                    }
                }
            }
            if demand > capacity as u128 {
                return None;
            }
        }
    }
    Some(makespan)
}

fn exhaustive_optimum(template: &CoupledTemplate, count: usize, horizon: u64) -> Option<u64> {
    fn visit(
        template: &CoupledTemplate,
        count: usize,
        horizon: u64,
        starts: &mut Vec<u64>,
        optimum: &mut Option<u64>,
    ) {
        if starts.len() == count * template.activities.len() {
            if let Some(cost) = direct_cost(template, count, horizon, starts) {
                *optimum = Some(optimum.map_or(cost, |old| old.min(cost)));
            }
            return;
        }
        let duration = template.activities[starts.len() % template.activities.len()].duration;
        if duration > horizon {
            return;
        }
        for start in 0..=horizon - duration {
            starts.push(start);
            visit(template, count, horizon, starts, optimum);
            starts.pop();
        }
    }
    let mut optimum = None;
    visit(template, count, horizon, &mut Vec::new(), &mut optimum);
    optimum
}

fn compare(
    template: CoupledTemplate,
    count: usize,
    horizon: u64,
    strategy: RepetitionStrategy,
) -> Option<RepetitionMethod> {
    let expected = exhaustive_optimum(&template, count, horizon);
    let mut solver = CoupledRepetition::new(
        Arc::new(template.clone()),
        count as u64,
        horizon,
        strategy,
        Options::default(),
    )
    .unwrap();
    match solver
        .advance(Limits {
            work: 1_000_000,
            ..Limits::default()
        })
        .unwrap()
    {
        CoupledOutcome::Optimal(solution) => {
            assert!(
                solver.stats().peak_retained_bytes >= solver.stats().search.peak_retained_bytes,
                "wrapper peak must include search storage"
            );
            assert_eq!(
                Some(solution.cost()),
                expected,
                "{template:?}, count={count}, strategy={strategy:?}"
            );
            solution.validate().unwrap();
            let starts: Vec<_> = (0..count)
                .flat_map(|copy| (0..template.activities.len()).map(move |stage| (copy, stage)))
                .map(|(copy, stage)| solution.start(copy as u64, stage).unwrap())
                .collect();
            assert_eq!(direct_cost(&template, count, horizon, &starts), expected);
            assert_eq!(solution.start(count as u64, 0), None);
            assert_eq!(solution.start(0, template.activities.len()), None);
            Some(solution.method())
        }
        CoupledOutcome::Infeasible => {
            assert_eq!(expected, None, "{template:?}");
            None
        }
        CoupledOutcome::Incomplete(progress) => {
            panic!("tiny schedule did not finish: {template:?}; {progress:?}")
        }
    }
}

#[test]
fn generated_small_joint_models_match_the_independent_oracle() {
    let mut seed = 0xabcde123456u64;
    let mut next = |limit: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % limit
    };
    for _ in 0..64 {
        let capacities = vec![next(3), next(3)];
        let activities = (0..2)
            .map(|_| activity(next(3), &[next(3), next(3)]))
            .collect();
        let mut precedence = Vec::new();
        for _ in 0..next(4) {
            precedence.push(edge(next(2) as usize, next(2) as usize, next(3), next(2)));
        }
        let template = CoupledTemplate {
            capacities,
            activities,
            precedence,
        };
        for strategy in [RepetitionStrategy::Automatic, RepetitionStrategy::Expanded] {
            compare(template.clone(), 2, 3, strategy);
        }
    }
}

#[test]
fn identical_activity_reduction_matches_exhaustive_resource_schedules() {
    for duration in 0..=2 {
        for demand in 0..=3 {
            for capacity in 0..=3 {
                for count in 0..=3 {
                    let template = CoupledTemplate {
                        capacities: vec![capacity],
                        activities: vec![activity(duration, &[demand])],
                        precedence: vec![],
                    };
                    compare(template.clone(), count, 5, RepetitionStrategy::Automatic);
                    if count <= 2 {
                        compare(template, count, 5, RepetitionStrategy::Expanded);
                    }
                }
            }
        }
    }
}

#[test]
fn pipeline_reduction_matches_exhaustive_schedules_and_finite_expansion() {
    for first in 0..=2 {
        for second in 0..=2 {
            for count in 1..=2 {
                let template = CoupledTemplate {
                    capacities: vec![3, 5],
                    // Exclusivity means more than half capacity; demand need
                    // not equal capacity for the proof to apply.
                    activities: vec![activity(first, &[2, 0]), activity(second, &[0, 3])],
                    precedence: vec![edge(0, 1, 0, 0)],
                };
                assert_eq!(
                    compare(template.clone(), count, 6, RepetitionStrategy::Automatic),
                    Some(RepetitionMethod::UnaryPipeline)
                );
                assert_eq!(
                    compare(template, count, 6, RepetitionStrategy::Expanded),
                    Some(RepetitionMethod::FiniteJoint)
                );
            }
        }
    }
}

#[test]
fn three_stage_pipeline_accounts_for_startup_drain_and_a_middle_bottleneck() {
    let template = CoupledTemplate {
        capacities: vec![1, 1, 1],
        activities: vec![
            activity(1, &[1, 0, 0]),
            activity(2, &[0, 1, 0]),
            activity(1, &[0, 0, 1]),
        ],
        precedence: vec![edge(0, 1, 0, 0), edge(1, 2, 0, 0)],
    };
    for strategy in [RepetitionStrategy::Automatic, RepetitionStrategy::Expanded] {
        assert!(compare(template.clone(), 2, 6, strategy).is_some());
        assert_eq!(compare(template.clone(), 2, 5, strategy), None);
    }
}

#[test]
fn shared_resources_nonexclusive_stages_and_cross_copy_edges_require_joint_search() {
    let cases = [
        CoupledTemplate {
            capacities: vec![1],
            activities: vec![activity(1, &[1]), activity(2, &[1])],
            precedence: vec![edge(0, 1, 0, 0)],
        },
        CoupledTemplate {
            capacities: vec![2, 1],
            activities: vec![activity(2, &[1, 0]), activity(1, &[0, 1])],
            precedence: vec![edge(0, 1, 0, 0)],
        },
        CoupledTemplate {
            capacities: vec![1, 1],
            activities: vec![activity(1, &[1, 0]), activity(1, &[0, 1])],
            precedence: vec![edge(0, 1, 0, 1)],
        },
        CoupledTemplate {
            capacities: vec![1, 1],
            activities: vec![activity(1, &[1, 0]), activity(1, &[0, 1])],
            precedence: vec![edge(0, 1, 0, 0), edge(1, 0, 1, 0)],
        },
        CoupledTemplate {
            capacities: vec![1],
            activities: vec![activity(1, &[1]), activity(1, &[1])],
            precedence: vec![],
        },
        CoupledTemplate {
            capacities: vec![],
            activities: vec![activity(1, &[])],
            precedence: vec![edge(0, 0, 1, 2)],
        },
    ];
    for template in cases {
        assert_eq!(
            compare(template.clone(), 2, 6, RepetitionStrategy::Automatic),
            Some(RepetitionMethod::FiniteJoint)
        );
        assert_eq!(
            compare(template, 2, 6, RepetitionStrategy::Expanded),
            Some(RepetitionMethod::FiniteJoint)
        );
    }
}

#[test]
fn cycles_horizon_limits_and_capacity_shortfalls_are_genuine_infeasibility() {
    let cases = [
        CoupledTemplate {
            capacities: vec![],
            activities: vec![activity(1, &[])],
            precedence: vec![edge(0, 0, 0, 0)],
        },
        CoupledTemplate {
            capacities: vec![1, 1],
            activities: vec![activity(2, &[1, 0]), activity(2, &[0, 1])],
            precedence: vec![edge(0, 1, 0, 0)],
        },
        CoupledTemplate {
            capacities: vec![1],
            activities: vec![activity(1, &[2])],
            precedence: vec![],
        },
    ];
    for template in cases {
        for strategy in [RepetitionStrategy::Automatic, RepetitionStrategy::Expanded] {
            assert_eq!(compare(template.clone(), 2, 3, strategy), None);
        }
    }
}

#[test]
fn precedence_beyond_finite_tail_has_no_schedule_obligation() {
    let template = CoupledTemplate {
        capacities: vec![1],
        activities: vec![activity(1, &[1])],
        precedence: vec![edge(0, 0, u64::MAX, u64::MAX)],
    };
    assert_eq!(
        compare(template.clone(), 2, 2, RepetitionStrategy::Automatic),
        Some(RepetitionMethod::IdenticalActivities)
    );
    compare(template, 2, 2, RepetitionStrategy::Expanded);
    let zero = CoupledTemplate {
        capacities: vec![0],
        activities: vec![activity(0, &[u64::MAX])],
        precedence: vec![edge(0, 0, 0, 0)],
    };
    compare(zero, 2, 0, RepetitionStrategy::Automatic);
}

#[test]
fn million_copy_compact_witnesses_do_not_expand_occurrences() {
    let templates = [
        (
            CoupledTemplate {
                capacities: vec![3],
                activities: vec![activity(2, &[1])],
                precedence: vec![],
            },
            666_668,
            RepetitionMethod::IdenticalActivities,
        ),
        (
            CoupledTemplate {
                capacities: vec![1, 1],
                activities: vec![activity(2, &[1, 0]), activity(3, &[0, 1])],
                precedence: vec![edge(0, 1, 0, 0)],
            },
            3_000_002,
            RepetitionMethod::UnaryPipeline,
        ),
    ];
    for (template, expected, method) in templates {
        let mut solver = CoupledRepetition::new(
            Arc::new(template),
            1_000_000,
            expected,
            RepetitionStrategy::Automatic,
            Options::default(),
        )
        .unwrap();
        let CoupledOutcome::Optimal(solution) = solver
            .advance(Limits {
                work: 1,
                ..Limits::default()
            })
            .unwrap()
        else {
            panic!()
        };
        assert_eq!(solution.cost(), expected);
        assert_eq!(solution.method(), method);
        assert_eq!(solver.stats().construction_work, 1);
        assert_eq!(solver.stats().constructed_copies, 0);
        assert_eq!(solver.stats().expanded_variables, 0);
        assert!(solution.feasible().witness_bytes() < 1024);
        assert!(solution.start(999_999, 0).unwrap() < expected);
        solution.validate().unwrap();
    }
}

#[test]
fn budgets_resume_expansion_and_joint_search_without_losing_coverage() {
    let template = Arc::new(CoupledTemplate {
        capacities: vec![1],
        activities: vec![activity(1, &[1]), activity(1, &[1])],
        precedence: vec![edge(0, 1, 0, 0)],
    });
    for (limits, reason) in [
        (
            Limits {
                work: 0,
                ..Limits::default()
            },
            StopReason::Work,
        ),
        (
            Limits {
                time: Some(Duration::ZERO),
                ..Limits::default()
            },
            StopReason::Time,
        ),
        (
            Limits {
                memory_bytes: Some(1),
                ..Limits::default()
            },
            StopReason::Memory,
        ),
    ] {
        let mut solver = CoupledRepetition::new(
            template.clone(),
            2,
            4,
            RepetitionStrategy::Automatic,
            Options::default(),
        )
        .unwrap();
        let CoupledOutcome::Incomplete(progress) = solver.advance(limits).unwrap() else {
            panic!()
        };
        assert_eq!(progress.reason, reason);
        let mut complete = false;
        for _ in 0..10_000 {
            let before = solver.stats().construction_work + solver.stats().search.work;
            match solver
                .advance(Limits {
                    work: 1,
                    ..Limits::default()
                })
                .unwrap()
            {
                CoupledOutcome::Optimal(solution) => {
                    assert_eq!(solution.cost(), 4);
                    complete = true;
                }
                CoupledOutcome::Incomplete(progress) => {
                    assert!(progress.lower_bound <= 4);
                    if let Some(witness) = progress.incumbent {
                        witness.validate().unwrap();
                        assert!(witness.cost() >= 4);
                    }
                }
                CoupledOutcome::Infeasible => panic!("known feasible"),
            }
            let after = solver.stats().construction_work + solver.stats().search.work;
            assert!(after - before <= 1);
            if complete {
                break;
            }
        }
        assert!(complete);
        assert_eq!(solver.stats().constructed_copies, 2);
    }
}

#[test]
fn memory_stop_during_expansion_preserves_the_exact_partial_construction() {
    let template = Arc::new(CoupledTemplate {
        capacities: vec![1],
        activities: vec![activity(1, &[1])],
        precedence: vec![edge(0, 0, 1, 0)],
    });
    let mut solver = CoupledRepetition::new(
        template,
        3,
        3,
        RepetitionStrategy::Expanded,
        Options::default(),
    )
    .unwrap();
    assert!(matches!(
        solver
            .advance(Limits {
                work: 2,
                ..Limits::default()
            })
            .unwrap(),
        CoupledOutcome::Incomplete(_)
    ));
    assert_eq!(solver.stats().constructed_copies, 1);
    let before = solver.stats().construction_work;
    let CoupledOutcome::Incomplete(progress) = solver
        .advance(Limits {
            memory_bytes: Some(solver.stats().retained_bytes),
            ..Limits::default()
        })
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(progress.reason, StopReason::Memory);
    assert_eq!(solver.stats().construction_work, before);
    let CoupledOutcome::Optimal(solution) = solver.advance(Limits::default()).unwrap() else {
        panic!()
    };
    assert_eq!(solution.cost(), 3);
}

#[test]
fn external_start_validation_rejects_each_kind_of_invalid_witness() {
    let template = CoupledTemplate {
        capacities: vec![1],
        activities: vec![activity(1, &[1]), activity(1, &[1])],
        precedence: vec![edge(0, 1, 0, 0), edge(1, 0, 1, 0)],
    };
    assert_eq!(template.validate_starts(2, 4, &[0, 1, 2, 3]).unwrap(), 4);
    for invalid in [
        vec![0, 1, 2],    // Missing occurrence.
        vec![0, 1, 2, 4], // Past the finite horizon.
        vec![1, 0, 2, 3], // Reversed local precedence.
        vec![0, 1, 0, 1], // Resource overlap and cross-copy precedence.
    ] {
        assert!(matches!(
            template.validate_starts(2, 4, &invalid),
            Err(Error::InvalidAssignment(_))
        ));
    }
    let no_edges = CoupledTemplate {
        precedence: vec![],
        ..template
    };
    assert!(matches!(
        no_edges.validate_starts(1, 4, &[0, 0]),
        Err(Error::InvalidAssignment(_))
    ));
    assert!(matches!(
        no_edges.validate_starts(1, u64::MAX, &[u64::MAX, 0]),
        Err(Error::Overflow(_))
    ));
}

#[test]
fn malformed_templates_are_errors_and_arithmetic_failure_stays_an_error() {
    for template in [
        CoupledTemplate {
            capacities: vec![1],
            activities: vec![activity(1, &[])],
            precedence: vec![],
        },
        CoupledTemplate {
            capacities: vec![],
            activities: vec![activity(1, &[])],
            precedence: vec![edge(0, 1, 0, 0)],
        },
    ] {
        assert!(matches!(
            CoupledRepetition::new(
                Arc::new(template),
                1,
                5,
                RepetitionStrategy::Automatic,
                Options::default()
            ),
            Err(Error::InvalidModel(_))
        ));
    }
    let template = Arc::new(CoupledTemplate {
        capacities: vec![1],
        activities: vec![activity(2, &[1])],
        precedence: vec![],
    });
    let mut solver = CoupledRepetition::new(
        template,
        u64::MAX,
        i64::MAX as u64,
        RepetitionStrategy::Automatic,
        Options::default(),
    )
    .unwrap();
    let error = solver.advance(Limits::default()).unwrap_err();
    assert!(matches!(error, Error::Overflow(_)));
    assert_eq!(solver.advance(Limits::default()).unwrap_err(), error);
}
