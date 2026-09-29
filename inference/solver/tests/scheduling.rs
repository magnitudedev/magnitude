use magnitude_solver::model::{Constraint, Cost, Domain, LinearTerm, ModelBuilder, VarId};
use magnitude_solver::scheduling::{
    Activity, ActivityReservation, Demand, Event, Interval, Lifetime, Reservation,
    SchedulingConstraint,
};

fn time(builder: &mut ModelBuilder, name: &str, min: i64, max: i64) -> VarId {
    builder.variable(name, Domain::interval(min, max).unwrap())
}

fn fixed(values: &[i64]) -> Vec<Domain> {
    values.iter().copied().map(Domain::singleton).collect()
}

#[test]
fn capacity_releases_before_acquiring_and_empty_intervals_consume_nothing() {
    let constraint = SchedulingConstraint::Cumulative {
        capacity: 3,
        reservations: vec![
            Reservation {
                interval: Interval::mandatory(VarId(0), VarId(1)),
                demand: Demand::Constant(3),
            },
            Reservation {
                interval: Interval::mandatory(VarId(2), VarId(3)),
                demand: Demand::Constant(3),
            },
            Reservation {
                interval: Interval::mandatory(VarId(4), VarId(5)),
                demand: Demand::Constant(u64::MAX),
            },
        ],
    };
    assert_eq!(
        constraint
            .assess(&fixed(&[0, 2, 2, 4, 1, 1]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
    assert!(
        constraint
            .assess(&fixed(&[0, 3, 2, 4, 1, 1]))
            .unwrap()
            .infeasible
    );
}

#[test]
fn optional_activities_do_not_validate_absent_private_times() {
    let constraint = SchedulingConstraint::Activity(Activity {
        start: VarId(0),
        duration: VarId(1),
        end: VarId(2),
        presence: Some(VarId(3)),
    });
    assert_eq!(
        constraint
            .assess(&fixed(&[-10, -5, 999, 0]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
    assert!(
        constraint
            .assess(&fixed(&[-10, -5, 999, 1]))
            .unwrap()
            .infeasible
    );
    assert_eq!(
        constraint.assess(&fixed(&[2, 3, 5, 1])).unwrap().exact_cost,
        Some(0)
    );
    assert!(constraint.assess(&fixed(&[2, 3, 6, 1])).unwrap().infeasible);
}

#[test]
fn optional_precedence_requires_both_events() {
    let constraint = SchedulingConstraint::Precedence {
        before: Event {
            time: VarId(0),
            presence: Some(VarId(2)),
        },
        after: Event {
            time: VarId(1),
            presence: Some(VarId(3)),
        },
        lag: 2,
    };
    assert_eq!(
        constraint
            .assess(&fixed(&[10, 0, 0, 1]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
    assert_eq!(
        constraint
            .assess(&fixed(&[10, 0, 1, 0]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
    assert!(
        constraint
            .assess(&fixed(&[10, 0, 1, 1]))
            .unwrap()
            .infeasible
    );
    assert_eq!(
        constraint
            .assess(&fixed(&[10, 12, 1, 1]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
}

#[test]
fn event_lifetimes_reserve_until_the_actual_release_event() {
    let lifetime = Lifetime {
        begin: Event::mandatory(VarId(0)),
        end: Event::mandatory(VarId(1)),
        demand: Demand::Constant(4),
    };
    let constraint = SchedulingConstraint::Cumulative {
        capacity: 4,
        reservations: vec![
            lifetime.reservation(),
            Reservation {
                interval: Interval::mandatory(VarId(2), VarId(3)),
                demand: Demand::Constant(1),
            },
        ],
    };
    assert!(constraint.assess(&fixed(&[1, 5, 4, 6])).unwrap().infeasible);
    assert_eq!(
        constraint.assess(&fixed(&[1, 5, 5, 6])).unwrap().exact_cost,
        Some(0)
    );
}

#[test]
fn propagated_activity_ranges_preserve_sparse_domains() {
    let activity = SchedulingConstraint::Activity(Activity {
        start: VarId(0),
        duration: VarId(1),
        end: VarId(2),
        presence: None,
    });
    let mut domains = vec![
        Domain::progression(0, 20, 3).unwrap(),
        Domain::singleton(2),
        Domain::interval(8, 12).unwrap(),
    ];
    let propagation = activity.propagate(&mut domains).unwrap();
    assert!(!propagation.infeasible);
    assert_eq!(domains[0].values().collect::<Vec<_>>(), vec![6, 9]);
    assert_eq!(domains[2].min(), Some(8));
}

#[test]
fn finite_horizon_is_a_model_constraint_and_never_extended() {
    let mut builder = ModelBuilder::new();
    let start = time(&mut builder, "start", 0, 2);
    let duration = time(&mut builder, "duration", 3, 3);
    let end = time(&mut builder, "end", 0, 2);
    builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
        Activity {
            start,
            duration,
            end,
            presence: None,
        },
    )));
    let model = builder.build().unwrap();
    assert!(model.validate_assignment(&[0, 3, 2]).unwrap().infeasible);
}

#[test]
fn cumulative_matches_direct_tick_oracle_for_small_intervals() {
    let constraint = SchedulingConstraint::Cumulative {
        capacity: 3,
        reservations: vec![
            Reservation {
                interval: Interval {
                    start: VarId(0),
                    end: VarId(1),
                    presence: vec![VarId(4)],
                },
                demand: Demand::Constant(2),
            },
            Reservation {
                interval: Interval::mandatory(VarId(2), VarId(3)),
                demand: Demand::Constant(2),
            },
        ],
    };
    for a in 0..=3 {
        for b in a..=3 {
            for c in 0..=3 {
                for d in c..=3 {
                    for presence in 0..=1 {
                        let legal = (0..3).all(|tick| {
                            let first = if presence == 1 && a <= tick && tick < b {
                                2
                            } else {
                                0
                            };
                            let second = if c <= tick && tick < d { 2 } else { 0 };
                            first + second <= 3
                        });
                        let assessment =
                            constraint.assess(&fixed(&[a, b, c, d, presence])).unwrap();
                        assert_eq!(
                            !assessment.infeasible, legal,
                            "[{a},{b}), [{c},{d}), p={presence}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn variable_resource_demands_and_zero_capacity_are_exact() {
    let constraint = SchedulingConstraint::Cumulative {
        capacity: 0,
        reservations: vec![Reservation {
            interval: Interval::mandatory(VarId(0), VarId(1)),
            demand: Demand::Variable(VarId(2)),
        }],
    };
    assert_eq!(
        constraint.assess(&fixed(&[0, 2, 0])).unwrap().exact_cost,
        Some(0)
    );
    assert!(constraint.assess(&fixed(&[0, 2, 1])).unwrap().infeasible);
    assert!(constraint.assess(&fixed(&[0, 2, -1])).unwrap().infeasible);
}

#[test]
fn makespan_is_one_objective_term_not_sum_of_durations() {
    let mut builder = ModelBuilder::new();
    let a = time(&mut builder, "a_end", 3, 3);
    let b = time(&mut builder, "b_end", 5, 5);
    let completion = time(&mut builder, "completion", 0, 10);
    for end in [a, b] {
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
            before: Event::mandatory(end),
            after: Event::mandatory(completion),
            lag: 0,
        }));
    }
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let model = builder.build().unwrap();
    assert!(model.validate_assignment(&[3, 5, 4]).unwrap().infeasible);
    assert_eq!(
        model.validate_assignment(&[3, 5, 5]).unwrap().exact_cost,
        Some(5)
    );
}

#[test]
fn capacity_incompatibility_derives_nine_tick_bound_without_enumerating_schedules() {
    let mut domains = vec![Domain::interval(0, 20).unwrap()];
    let activities: Vec<_> = [2, 3, 4]
        .into_iter()
        .map(|duration| {
            let start = VarId(domains.len());
            domains.push(Domain::interval(0, 20).unwrap());
            let duration_var = VarId(domains.len());
            domains.push(Domain::singleton(duration));
            let end = VarId(domains.len());
            domains.push(Domain::interval(0, 20).unwrap());
            ActivityReservation {
                activity: Activity {
                    start,
                    duration: duration_var,
                    end,
                    presence: None,
                },
                demand: Demand::Constant(2),
            }
        })
        .collect();
    let constraint = SchedulingConstraint::ActivityCumulative {
        completion: VarId(0),
        capacity: 3,
        activities,
    };
    let result = constraint.propagate(&mut domains).unwrap();
    assert!(!result.infeasible);
    assert_eq!(domains[0].min(), Some(9));
    // No start variable has been chosen: the stronger floor follows from the
    // original resource constraints, not fixed child profiles or enumeration.
    for index in [1, 4, 7] {
        assert!(!domains[index].is_singleton());
    }
    assert_eq!(
        constraint
            .assess(&fixed(&[9, 0, 2, 2, 2, 3, 5, 5, 4, 9]))
            .unwrap()
            .exact_cost,
        Some(0)
    );
}

#[test]
fn capacity_clique_bounds_match_all_tiny_duration_and_demand_contexts() {
    for first_duration in 1..=2 {
        for second_duration in 1..=2 {
            for first_demand in 1..=2 {
                for second_demand in 1..=2 {
                    for capacity in 1..=3 {
                        let mut domains = vec![
                            Domain::interval(0, 4).unwrap(),
                            Domain::interval(0, 4).unwrap(),
                            Domain::singleton(first_duration),
                            Domain::interval(0, 4).unwrap(),
                            Domain::interval(0, 4).unwrap(),
                            Domain::singleton(second_duration),
                            Domain::interval(0, 4).unwrap(),
                        ];
                        let constraint = SchedulingConstraint::ActivityCumulative {
                            completion: VarId(0),
                            capacity,
                            activities: vec![
                                ActivityReservation {
                                    activity: Activity {
                                        start: VarId(1),
                                        duration: VarId(2),
                                        end: VarId(3),
                                        presence: None,
                                    },
                                    demand: Demand::Constant(first_demand),
                                },
                                ActivityReservation {
                                    activity: Activity {
                                        start: VarId(4),
                                        duration: VarId(5),
                                        end: VarId(6),
                                        presence: None,
                                    },
                                    demand: Demand::Constant(second_demand),
                                },
                            ],
                        };
                        let propagated = constraint.propagate(&mut domains).unwrap();
                        let mut optimum = None;
                        for first in 0..=4 {
                            for second in 0..=4 {
                                let end_first = first + first_duration;
                                let end_second = second + second_duration;
                                let completion = end_first.max(end_second);
                                if completion > 4 {
                                    continue;
                                }
                                let fits = (0..completion).all(|tick| {
                                    let a = if first <= tick && tick < end_first {
                                        first_demand
                                    } else {
                                        0
                                    };
                                    let b = if second <= tick && tick < end_second {
                                        second_demand
                                    } else {
                                        0
                                    };
                                    a + b <= capacity
                                });
                                if fits {
                                    optimum = Some(
                                        optimum.map_or(completion, |old: i64| old.min(completion)),
                                    );
                                }
                            }
                        }
                        match optimum {
                            Some(optimum) => {
                                assert!(!propagated.infeasible);
                                assert!(domains[0].min().unwrap() <= optimum);
                            }
                            None => assert!(propagated.infeasible),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn optional_activity_does_not_leak_into_a_mandatory_clique_bound() {
    let constraint = SchedulingConstraint::ActivityCumulative {
        completion: VarId(0),
        capacity: 1,
        activities: vec![ActivityReservation {
            activity: Activity {
                start: VarId(1),
                duration: VarId(2),
                end: VarId(3),
                presence: Some(VarId(4)),
            },
            demand: Demand::Constant(1),
        }],
    };
    let mut domains = vec![
        Domain::interval(0, 10).unwrap(),
        Domain::interval(0, 10).unwrap(),
        Domain::singleton(10),
        Domain::interval(0, 10).unwrap(),
        Domain::boolean(),
    ];
    constraint.propagate(&mut domains).unwrap();
    assert_eq!(domains[0].min(), Some(0));
    domains[4] = Domain::singleton(1);
    constraint.propagate(&mut domains).unwrap();
    assert_eq!(domains[0].min(), Some(10));
}

#[test]
fn absent_activity_is_exact_without_branching_over_private_time_domains() {
    let constraint = SchedulingConstraint::Activity(Activity {
        start: VarId(0),
        duration: VarId(1),
        end: VarId(2),
        presence: Some(VarId(3)),
    });
    let domains = vec![
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::singleton(0),
    ];
    let assessment = constraint.assess(&domains).unwrap();
    assert_eq!(assessment.exact_cost, Some(0));
    assert!(constraint.effective_scope(&domains).is_empty());
}

#[test]
fn absent_reservations_are_removed_from_dynamic_scope_but_not_static_scope() {
    let constraint = SchedulingConstraint::Cumulative {
        capacity: 1,
        reservations: vec![Reservation {
            interval: Interval {
                start: VarId(0),
                end: VarId(1),
                presence: vec![VarId(2)],
            },
            demand: Demand::Constant(1),
        }],
    };
    let domains = vec![
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::singleton(0),
    ];
    assert!(!constraint.scope().is_empty());
    assert!(constraint.effective_scope(&domains).is_empty());
    assert_eq!(constraint.assess(&domains).unwrap().exact_cost, Some(0));
}

#[test]
fn absent_precedence_drops_both_event_times_from_dynamic_scope() {
    let constraint = SchedulingConstraint::Precedence {
        before: Event {
            time: VarId(0),
            presence: Some(VarId(2)),
        },
        after: Event {
            time: VarId(1),
            presence: Some(VarId(3)),
        },
        lag: 7,
    };
    let domains = vec![
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::interval(0, 1_000_000).unwrap(),
        Domain::singleton(0),
        Domain::boolean(),
    ];
    assert!(constraint.effective_scope(&domains).is_empty());
    assert_eq!(constraint.assess(&domains).unwrap().exact_cost, Some(0));
}

#[test]
fn absent_composite_still_rejects_negative_completion_domain() {
    let constraint = SchedulingConstraint::ActivityCumulative {
        completion: VarId(0),
        capacity: 1,
        activities: vec![ActivityReservation {
            activity: Activity {
                start: VarId(1),
                duration: VarId(2),
                end: VarId(3),
                presence: Some(VarId(4)),
            },
            demand: Demand::Constant(1),
        }],
    };
    let domains = vec![
        Domain::interval(-1, 1).unwrap(),
        Domain::interval(0, 1_000).unwrap(),
        Domain::singleton(2),
        Domain::interval(0, 1_000).unwrap(),
        Domain::singleton(0),
    ];
    let assessment = constraint.assess(&domains).unwrap();
    assert!(!assessment.infeasible);
    assert_eq!(assessment.exact_cost, None);
    assert_eq!(constraint.effective_scope(&domains), vec![VarId(0)]);
}

#[test]
fn globally_better_modes_can_use_a_locally_slower_resource_profile() {
    // Mode A is locally faster (three ticks) but monopolizes R. Mode B takes
    // four ticks, occupying R only for its first tick and then a private
    // resource. Two copies therefore choose one A and one B and complete in 4;
    // choosing A independently for both copies would take 6.
    let mut builder = ModelBuilder::new();
    let completion = builder.variable("completion", Domain::interval(0, 8).unwrap());
    let mut resource_r = Vec::new();
    let mut private = Vec::new();
    for copy in 0..2 {
        let mode_a = builder.variable(format!("copy_{copy}_a"), Domain::boolean());
        let mode_b = builder.variable(format!("copy_{copy}_b"), Domain::boolean());
        builder.constraint(Constraint::ExactlyOne {
            variables: vec![mode_a, mode_b],
        });
        let a_start = builder.variable(
            format!("copy_{copy}_a_start"),
            Domain::interval(0, 8).unwrap(),
        );
        let a_duration = builder.variable(format!("copy_{copy}_a_duration"), Domain::singleton(3));
        let a_end = builder.variable(
            format!("copy_{copy}_a_end"),
            Domain::interval(0, 8).unwrap(),
        );
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start: a_start,
                duration: a_duration,
                end: a_end,
                presence: Some(mode_a),
            },
        )));
        builder.guarded_constraint(
            vec![magnitude_solver::model::Literal::new(mode_a, 1)],
            Constraint::LinearLe {
                terms: vec![LinearTerm::new(a_end, 1), LinearTerm::new(completion, -1)],
                rhs: 0,
            },
        );
        resource_r.push(Reservation {
            interval: Interval {
                start: a_start,
                end: a_end,
                presence: vec![mode_a],
            },
            demand: Demand::Constant(1),
        });
        let b_start = builder.variable(
            format!("copy_{copy}_b_start"),
            Domain::interval(0, 8).unwrap(),
        );
        let b_duration = builder.variable(format!("copy_{copy}_b_duration"), Domain::singleton(1));
        let b_end = builder.variable(
            format!("copy_{copy}_b_end"),
            Domain::interval(0, 8).unwrap(),
        );
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start: b_start,
                duration: b_duration,
                end: b_end,
                presence: Some(mode_b),
            },
        )));
        resource_r.push(Reservation {
            interval: Interval {
                start: b_start,
                end: b_end,
                presence: vec![mode_b],
            },
            demand: Demand::Constant(1),
        });
        let tail_start = builder.variable(
            format!("copy_{copy}_tail_start"),
            Domain::interval(0, 8).unwrap(),
        );
        let tail_duration =
            builder.variable(format!("copy_{copy}_tail_duration"), Domain::singleton(3));
        let tail_end = builder.variable(
            format!("copy_{copy}_tail_end"),
            Domain::interval(0, 8).unwrap(),
        );
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start: tail_start,
                duration: tail_duration,
                end: tail_end,
                presence: Some(mode_b),
            },
        )));
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
            before: Event {
                time: b_end,
                presence: Some(mode_b),
            },
            after: Event {
                time: tail_start,
                presence: Some(mode_b),
            },
            lag: 0,
        }));
        builder.guarded_constraint(
            vec![magnitude_solver::model::Literal::new(mode_b, 1)],
            Constraint::LinearLe {
                terms: vec![
                    LinearTerm::new(tail_end, 1),
                    LinearTerm::new(completion, -1),
                ],
                rhs: 0,
            },
        );
        private.push(Reservation {
            interval: Interval {
                start: tail_start,
                end: tail_end,
                presence: vec![mode_b],
            },
            demand: Demand::Constant(1),
        });
    }
    builder.constraint(Constraint::Schedule(SchedulingConstraint::Cumulative {
        capacity: 1,
        reservations: resource_r,
    }));
    builder.constraint(Constraint::Schedule(SchedulingConstraint::Cumulative {
        capacity: 2,
        reservations: private,
    }));
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    let model = builder.build().unwrap();
    let mut search = magnitude_solver::Search::new(
        std::sync::Arc::new(model),
        magnitude_solver::Options::default(),
    )
    .unwrap();
    let magnitude_solver::Outcome::Optimal(solution) = search
        .advance(magnitude_solver::Limits {
            work: 2_000_000,
            ..Default::default()
        })
        .unwrap()
    else {
        panic!("tiny coupled modes should complete");
    };
    assert_eq!(solution.cost(), 4);
    let values = solution.values();
    assert_eq!(values[1] + values[12], 1); // exactly one copy uses A
    assert_eq!(values[2] + values[13], 1); // exactly one copy uses B
}
