use super::{Parameters, Random, Reference};
use crate::LabResult;
use magnitude_solver::{
    model::{Constraint, Cost, Domain, LinearTerm, Literal, ModelBuilder, VarId},
    scheduling::{
        Activity, ActivityReservation, Demand, Event, Lifetime, Reservation, SchedulingConstraint,
    },
};

fn schedule_factor(builder: &mut ModelBuilder, constraint: SchedulingConstraint) {
    builder.constraint(Constraint::Schedule(constraint));
}
fn activity(
    builder: &mut ModelBuilder,
    name: &str,
    duration: u64,
    horizon: u64,
    presence: Option<VarId>,
) -> LabResult<Activity> {
    let start = builder.variable(
        format!("{name}-start"),
        Domain::interval(0, i64::try_from(horizon)?)?,
    );
    let end = builder.variable(
        format!("{name}-end"),
        Domain::interval(0, i64::try_from(horizon)?)?,
    );
    let duration = builder.variable(
        format!("{name}-duration"),
        Domain::singleton(i64::try_from(duration)?),
    );
    let a = Activity {
        start,
        duration,
        end,
        presence,
    };
    schedule_factor(builder, SchedulingConstraint::Activity(a.clone()));
    if let Some(present) = presence {
        // Canonical inactive private state reduces useless choices without removing executions.
        for var in [start, end] {
            builder.guarded_constraint(
                vec![Literal::new(present, 0)],
                Constraint::LinearLe {
                    terms: vec![LinearTerm::new(var, 1)],
                    rhs: 0,
                },
            );
        }
    }
    Ok(a)
}
fn precedence(builder: &mut ModelBuilder, before: Event, after: Event) {
    schedule_factor(
        builder,
        SchedulingConstraint::Precedence {
            before,
            after,
            lag: 0,
        },
    );
}
fn finish(builder: &mut ModelBuilder, a: &Activity, makespan: VarId) {
    precedence(builder, a.end_event(), Event::mandatory(makespan));
}
fn makespan(builder: &mut ModelBuilder, horizon: u64) -> LabResult<VarId> {
    let t = builder.variable("makespan", Domain::interval(0, i64::try_from(horizon)?)?);
    builder.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(t, 1)],
    });
    Ok(t)
}

pub(super) fn schedule(
    builder: &mut ModelBuilder,
    p: &Parameters,
    rng: &mut Random,
) -> LabResult<()> {
    if p.n.checked_mul(p.d).ok_or("activity count overflow")? > 512 {
        return Err("schedule generator supports at most 512 explicit mode activities".into());
    }
    let t = makespan(builder, p.horizon)?;
    let mut reservations = Vec::new();
    let mut previous: Vec<Activity> = Vec::new();
    for task in 0..p.n {
        let mut modes = Vec::new();
        let mut presences = Vec::new();
        for mode in 0..p.d {
            let present = builder.variable(format!("task-{task}-mode-{mode}"), Domain::boolean());
            presences.push(present);
            let duration = 1 + rng.next_u64() % 3;
            let a = activity(
                builder,
                &format!("task-{task}-mode-{mode}"),
                duration,
                p.horizon,
                Some(present),
            )?;
            let demand = 1 + mode as u64 % 2;
            reservations.push(Reservation {
                interval: a.interval(),
                demand: Demand::Constant(demand),
            });
            finish(builder, &a, t);
            // Every third task begins a new independent chain. The resource still couples chains.
            if task % 3 != 0 {
                for before in &previous {
                    precedence(builder, before.end_event(), a.start_event());
                }
            }
            modes.push(a);
        }
        builder.constraint(Constraint::ExactlyOne {
            variables: presences,
        });
        previous = modes;
    }
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: p.capacity,
            reservations,
        },
    );
    Ok(())
}

/// Synthetic task durations are fully represented as integer activity lengths.
/// Choices activate different complete task graphs. All active alternatives use
/// the same global transfer, compute and retained-storage resource factors.
pub(super) fn pipeline(builder: &mut ModelBuilder, p: &Parameters) -> LabResult<()> {
    if p.outputs == 0 || p.input_length == 0 || p.input_length > 16_384 || p.outputs > 4096 {
        return Err("pipeline requires 1..4096 outputs and 1..16384 input length".into());
    }
    let widths = choices(p.outputs, p.d);
    let windows = divisors(p.input_length, p.d);
    let mut plans = Vec::new();
    for g in widths {
        for &q in &windows {
            let count = p.input_length / q;
            for u in if count > 1 && p.d > 1 {
                vec![1, count.min(2)]
            } else {
                vec![1]
            } {
                for retained in if p.d > 1 {
                    vec![false, true]
                } else {
                    vec![true]
                } {
                    plans.push((g, q, u, retained));
                }
            }
        }
    }
    let mut count = 0_u64;
    for &(g, q, u, retained) in &plans {
        let groups = p.outputs.div_ceil(g);
        let windows = p.input_length / q;
        count = count
            .checked_add(
                windows
                    .checked_mul(groups + if retained { 1 } else { groups })
                    .ok_or("pipeline count overflow")?
                    + windows.div_ceil(u),
            )
            .ok_or("pipeline count overflow")?;
    }
    if count > 4096 {
        return Err(format!("pipeline eagerly expands {count} activities; construction limit is 4096. Reduce sizes or choice count; large compact schedules require a separate representation.").into());
    }
    let t = makespan(builder, p.horizon)?;
    let mut all_presences = Vec::new();
    let mut transfer = Vec::new();
    let mut compute = Vec::new();
    let mut storage = Vec::new();
    for (plan, (g, q, u, retained)) in plans.into_iter().enumerate() {
        let name = format!(
            "plan-{plan}-g{g}-q{q}-u{u}-{}",
            if retained { "retain" } else { "recompute" }
        );
        let present = builder.variable(format!("{name}-present"), Domain::boolean());
        all_presences.push(present);
        let windows = p.input_length / q;
        let groups = p.outputs.div_ceil(g);
        let preparation = p.overhead.checked_add(q).ok_or("duration overflow")?;
        let reserved = q
            .checked_mul(p.element_bytes)
            .ok_or("storage demand overflow")?;
        let mut previous_batch: Option<Activity> = None;
        for batch in 0..windows.div_ceil(u) {
            let overhead = activity(
                builder,
                &format!("{name}-batch-{batch}"),
                p.overhead,
                p.horizon,
                Some(present),
            )?;
            if let Some(previous) = &previous_batch {
                precedence(builder, previous.end_event(), overhead.start_event());
            }
            compute.push(Reservation {
                interval: overhead.interval(),
                demand: Demand::Constant(1),
            });
            finish(builder, &overhead, t);
            for window in batch * u..((batch + 1) * u).min(windows) {
                let shared = if retained {
                    let prep = activity(
                        builder,
                        &format!("{name}-window-{window}-shared-prep"),
                        preparation,
                        p.horizon,
                        Some(present),
                    )?;
                    precedence(builder, overhead.end_event(), prep.start_event());
                    transfer.push(Reservation {
                        interval: prep.interval(),
                        demand: Demand::Constant(1),
                    });
                    finish(builder, &prep, t);
                    Some(prep)
                } else {
                    None
                };
                // Last-use event is constrained after every consumer; storage pressure forces it as early as needed.
                let last_use = if retained {
                    Some(builder.variable(
                        format!("{name}-window-{window}-last-use"),
                        Domain::interval(0, i64::try_from(p.horizon)?)?,
                    ))
                } else {
                    None
                };
                for group in 0..groups {
                    let tail = g.min(p.outputs - group * g);
                    let duration = 1_u64
                        .checked_add(tail.checked_mul(q).ok_or("consumption duration overflow")?)
                        .ok_or("consumption duration overflow")?;
                    let consumer = activity(
                        builder,
                        &format!("{name}-w{window}-consumer-{group}"),
                        duration,
                        p.horizon,
                        Some(present),
                    )?;
                    compute.push(Reservation {
                        interval: consumer.interval(),
                        demand: Demand::Constant(1),
                    });
                    finish(builder, &consumer, t);
                    let prep = match &shared {
                        Some(a) => a.clone(),
                        None => {
                            let a = activity(
                                builder,
                                &format!("{name}-w{window}-prep-{group}"),
                                preparation,
                                p.horizon,
                                Some(present),
                            )?;
                            precedence(builder, overhead.end_event(), a.start_event());
                            transfer.push(Reservation {
                                interval: a.interval(),
                                demand: Demand::Constant(1),
                            });
                            finish(builder, &a, t);
                            storage.push(
                                Lifetime {
                                    begin: a.end_event(),
                                    end: consumer.end_event(),
                                    demand: Demand::Constant(reserved),
                                }
                                .reservation(),
                            );
                            a
                        }
                    };
                    precedence(builder, prep.end_event(), consumer.start_event());
                    if let Some(last) = last_use {
                        precedence(
                            builder,
                            consumer.end_event(),
                            Event {
                                time: last,
                                presence: Some(present),
                            },
                        );
                    }
                }
                if let (Some(prep), Some(last)) = (shared, last_use) {
                    precedence(
                        builder,
                        Event {
                            time: last,
                            presence: Some(present),
                        },
                        Event::mandatory(t),
                    );
                    storage.push(
                        Lifetime {
                            begin: prep.end_event(),
                            end: Event {
                                time: last,
                                presence: Some(present),
                            },
                            demand: Demand::Constant(reserved),
                        }
                        .reservation(),
                    );
                    builder.guarded_constraint(
                        vec![Literal::new(present, 0)],
                        Constraint::LinearLe {
                            terms: vec![LinearTerm::new(last, 1)],
                            rhs: 0,
                        },
                    );
                }
            }
            previous_batch = Some(overhead);
        }
    }
    builder.constraint(Constraint::ExactlyOne {
        variables: all_presences,
    });
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: 1,
            reservations: transfer,
        },
    );
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: p.n as u64,
            reservations: compute,
        },
    );
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: p.capacity,
            reservations: storage,
        },
    );
    Ok(())
}

fn choices(extent: u64, count: usize) -> Vec<u64> {
    let mut result = vec![extent];
    if count > 1 {
        result.push(1);
    }
    let mut width = 2;
    while result.len() < count && width < extent {
        result.push(width);
        width = width.saturating_mul(2);
    }
    result.sort_unstable();
    result.dedup();
    result
}
fn divisors(extent: u64, count: usize) -> Vec<u64> {
    let mut result = vec![extent];
    for q in 1..extent {
        if result.len() >= count {
            break;
        }
        if extent.is_multiple_of(q) {
            result.push(q);
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

pub(super) fn packing(builder: &mut ModelBuilder) -> LabResult<Reference> {
    let horizon = 9;
    let completion = makespan(builder, horizon)?;
    let durations = vec![2, 3, 4];
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut activities = Vec::new();
    for (i, &duration) in durations.iter().enumerate() {
        let a = activity(builder, &format!("packed-{i}"), duration, horizon, None)?;
        starts.push(a.start);
        ends.push(a.end);
        activities.push(ActivityReservation {
            activity: a,
            demand: Demand::Constant(2),
        });
    }
    schedule_factor(
        builder,
        SchedulingConstraint::ActivityCumulative {
            completion,
            capacity: 3,
            activities,
        },
    );
    Ok(Reference::FixedSchedule {
        starts,
        ends,
        durations,
        completion,
        horizon,
    })
}

pub(super) fn repeated_coupled(builder: &mut ModelBuilder) -> LabResult<Reference> {
    let horizon = 10;
    let completion = makespan(builder, horizon)?;
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut durations = Vec::new();
    let mut transfers = Vec::new();
    let mut computes = Vec::new();
    for copy in 0..2 {
        let prep = activity(builder, &format!("copy-{copy}-prepare"), 2, horizon, None)?;
        let consume = activity(builder, &format!("copy-{copy}-consume"), 3, horizon, None)?;
        precedence(builder, prep.end_event(), consume.start_event());
        finish(builder, &consume, completion);
        for (a, duration) in [(&prep, 2), (&consume, 3)] {
            starts.push(a.start);
            ends.push(a.end);
            durations.push(duration);
        }
        transfers.push(Reservation {
            interval: prep.interval(),
            demand: Demand::Constant(1),
        });
        computes.push(Reservation {
            interval: consume.interval(),
            demand: Demand::Constant(1),
        });
    }
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: 1,
            reservations: transfers,
        },
    );
    schedule_factor(
        builder,
        SchedulingConstraint::Cumulative {
            capacity: 1,
            reservations: computes,
        },
    );
    Ok(Reference::FixedSchedule {
        starts,
        ends,
        durations,
        completion,
        horizon,
    })
}
