use magnitude_solver::model::{Domain, VarId};
use magnitude_solver::scheduling::{Activity, ActivityReservation, Demand, SchedulingConstraint};

fn interval(lo: i64, hi: i64) -> Domain {
    Domain::interval(lo, hi).unwrap()
}
fn reservation(i: usize, demand: u64) -> ActivityReservation {
    ActivityReservation {
        activity: Activity {
            start: VarId(3 * i),
            duration: VarId(3 * i + 1),
            end: VarId(3 * i + 2),
            presence: None,
        },
        demand: Demand::Constant(demand),
    }
}

#[test]
fn positive_duration_forces_order_without_compulsory_parts() {
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(6),
        capacity: 1,
        activities: vec![reservation(0, 1), reservation(1, 1)],
    };
    let mut d = vec![
        interval(0, 4),
        Domain::singleton(2),
        interval(2, 6),
        interval(1, 8),
        Domain::singleton(5),
        interval(6, 13),
        interval(0, 20),
    ];
    let p = c.propagate(&mut d).unwrap();
    assert!(!p.infeasible);
    assert_eq!(d[3].min(), Some(2));
}

#[test]
fn late_release_energy_is_not_diluted_by_early_unrelated_work() {
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(9),
        capacity: 3,
        activities: vec![reservation(0, 0), reservation(1, 1), reservation(2, 2)],
    };
    let mut d = vec![
        interval(0, 100),
        Domain::singleton(1),
        interval(1, 101),
        interval(50, 100),
        Domain::singleton(8),
        interval(58, 108),
        interval(50, 100),
        Domain::singleton(8),
        interval(58, 108),
        interval(0, 200),
    ];
    assert!(!c.propagate(&mut d).unwrap().infeasible);
    assert_eq!(d[9].min(), Some(58));
    // Increase the second demand so the late pair must serialize: energy alone
    // only proves ceil(32/3)=11, while the clique proves all sixteen ticks.
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(9),
        capacity: 3,
        activities: vec![reservation(0, 0), reservation(1, 2), reservation(2, 2)],
    };
    assert!(!c.propagate(&mut d).unwrap().infeasible);
    assert_eq!(d[9].min(), Some(66));
}

#[test]
fn release_suffix_energy_strictly_strengthens_individual_end_bounds() {
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(12),
        capacity: 3,
        activities: vec![
            reservation(0, 0),
            reservation(1, 1),
            reservation(2, 1),
            reservation(3, 1),
        ],
    };
    let mut d = vec![interval(0, 100), Domain::singleton(1), interval(1, 101)];
    for _ in 0..3 {
        d.extend([interval(50, 100), Domain::singleton(8), interval(58, 108)]);
    }
    d.push(interval(0, 200));
    assert!(!c.propagate(&mut d).unwrap().infeasible);
    assert_eq!(d[12].min(), Some(58));
    // With capacity two all three can pairwise overlap, so no nontrivial clique
    // exists; energy proves 50+ceil(24/2)=62.
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(12),
        capacity: 2,
        activities: vec![
            reservation(0, 0),
            reservation(1, 1),
            reservation(2, 1),
            reservation(3, 1),
        ],
    };
    assert!(!c.propagate(&mut d).unwrap().infeasible);
    assert_eq!(d[12].min(), Some(62));
}

#[test]
fn exhaustive_small_schedules_preserve_every_feasible_assignment() {
    // Independent tick-by-tick capacity oracle: no solver assessor is used to
    // decide which assignments propagation must preserve. Include optional,
    // zero-duration, variable-demand, and nonconvex start domains.
    for case in 0..384usize {
        let capacity = 1 + case % 3;
        let mut activities = Vec::new();
        let mut domains = Vec::new();
        let mut lengths = [0; 3];
        let mut demands = [0; 3];
        for i in 0..3 {
            lengths[i] = ((case / (i + 1) + i) % 3) as i64;
            demands[i] = (case / (i + 2) + i) % 4;
            let lo = ((case >> i) % 3) as i64;
            let starts = if case % 2 == 0 {
                Domain::set((lo..=4).filter(|v| v % 2 == 0))
            } else {
                interval(lo, 4)
            };
            domains.extend([
                starts,
                interval(lengths[i], 2),
                interval(lo + lengths[i], 6),
            ]);
            activities.push(reservation(i, demands[i] as u64));
        }
        domains.push(interval(0, 6)); // completion
        domains.push(if case % 3 == 0 {
            Domain::singleton(1)
        } else {
            Domain::boolean()
        }); // presence of final activity
        domains.push(interval(demands[0] as i64, 3)); // variable first demand
        activities[0].demand = Demand::Variable(VarId(11));
        activities[2].activity.presence = Some(VarId(10));
        let c = SchedulingConstraint::ActivityCumulative {
            completion: VarId(9),
            capacity: capacity as u64,
            activities,
        };
        let original = domains.clone();
        let status = c.propagate(&mut domains).unwrap();
        for d0 in lengths[0]..=2 {
            for d1 in lengths[1]..=2 {
                for d2 in lengths[2]..=2 {
                    let lengths = [d0, d1, d2];
                    for s0 in 0..=4 {
                        for s1 in 0..=4 {
                            for s2 in 0..=4 {
                                for present in 0..=1 {
                                    let starts = [s0, s1, s2];
                                    let ends = [s0 + lengths[0], s1 + lengths[1], s2 + lengths[2]];
                                    let completion = ends[0].max(ends[1]).max(if present == 1 {
                                        ends[2]
                                    } else {
                                        0
                                    });
                                    let values = [
                                        s0,
                                        lengths[0],
                                        ends[0],
                                        s1,
                                        lengths[1],
                                        ends[1],
                                        s2,
                                        lengths[2],
                                        ends[2],
                                        completion,
                                        present,
                                        demands[0] as i64,
                                    ];
                                    if !values.iter().zip(&original).all(|(&v, d)| d.contains(v)) {
                                        continue;
                                    }
                                    let feasible = (0..6).all(|t| {
                                        (0..3)
                                            .filter(|&i| i != 2 || present == 1)
                                            .filter(|&i| starts[i] <= t && t < ends[i])
                                            .map(|i| demands[i])
                                            .sum::<usize>()
                                            <= capacity
                                    });
                                    if feasible {
                                        assert!(!status.infeasible, "case {case}: {values:?}");
                                        assert!(
                                            values
                                                .iter()
                                                .zip(&domains)
                                                .all(|(&v, d)| d.contains(v)),
                                            "case {case}: {values:?}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn wide_serial_resource_uses_late_release_clique_bound() {
    let count = 512;
    let mut domains = Vec::new();
    for i in 0..count {
        let release = if i < count / 2 { 0 } else { 512 };
        domains.extend([
            interval(release, 1024),
            Domain::singleton(1),
            interval(release + 1, 1025),
        ]);
    }
    domains.push(interval(0, 2048));
    let c = SchedulingConstraint::ActivityCumulative {
        completion: VarId(count * 3),
        capacity: 3,
        activities: (0..count).map(|i| reservation(i, 2)).collect(),
    };
    assert!(!c.propagate(&mut domains).unwrap().infeasible);
    assert_eq!(domains[count * 3].min(), Some(768));
}
