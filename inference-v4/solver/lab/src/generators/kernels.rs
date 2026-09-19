//! Kernel-shaped finite choices. There is no variable indexing complete plans:
//! geometry, staging/sharing, layout and live storage remain distinct factors.
use super::{Parameters, Random, Reference};
use crate::LabResult;
use magnitude_solver::{
    model::{Arithmetic, Constraint, Cost, Domain, LinearTerm, Literal, ModelBuilder, VarId},
    scheduling::{Activity, ActivityReservation, Demand, SchedulingConstraint},
};

fn choice(b: &mut ModelBuilder, name: String, max: usize) -> LabResult<VarId> {
    Ok(b.variable(name, Domain::interval(1, i64::try_from(max)?)?))
}
fn table(b: &mut ModelBuilder, vars: &[VarId], rows: Vec<(Vec<i64>, u64)>) {
    b.cost(Cost::Table {
        variables: vars.to_vec(),
        entries: rows,
    });
}
fn product(b: &mut ModelBuilder, name: String, a: VarId, c: VarId, max: usize) -> LabResult<VarId> {
    let out = b.variable(name, Domain::interval(0, i64::try_from(max)?)?);
    b.constraint(Constraint::Arithmetic(Arithmetic::Product {
        left: a,
        right: c,
        product: out,
    }));
    Ok(out)
}

pub(super) fn contraction(
    b: &mut ModelBuilder,
    p: &Parameters,
    rng: &mut Random,
) -> LabResult<Reference> {
    let mut resource = Vec::new();
    let mut groups = Vec::new();
    for i in 0..p.n {
        let tile = choice(b, format!("region-{i}-tile"), p.d)?;
        let staged = b.variable(format!("region-{i}-staged"), Domain::boolean());
        let layout = b.variable(format!("region-{i}-layout"), Domain::boolean());
        let storage = product(b, format!("region-{i}-live-storage"), tile, staged, p.d)?;
        resource.push(LinearTerm::new(storage, 1));
        b.guarded_constraint(
            vec![Literal::new(staged, 1)],
            Constraint::InDomain {
                variable: layout,
                domain: Domain::singleton(1),
            },
        );
        let extent = p.input_length + rng.next_u64() % 4;
        let transfer = 1 + rng.next_u64() % 5;
        let compute = 1 + rng.next_u64() % 3;
        let launch = p.overhead + rng.next_u64() % 4;
        let mut costs = Vec::new();
        let mut alternatives = Vec::new();
        for g in 1..=p.d as u64 {
            for s in 0..=1_u64 {
                for l in 0..=1_u64 {
                    let groups_count = extent.div_ceil(g);
                    // Direct addressing coalesces without a shared layout but
                    // repeats traffic across tiles. Staging pays setup, pads
                    // tail lanes and lowers repeated transfer service.
                    let traffic = if s == 0 {
                        extent * transfer * g
                    } else {
                        extent * transfer + groups_count * launch
                    };
                    let arithmetic = groups_count * (compute + if s == 1 { 0 } else { g - 1 });
                    let tail = (groups_count * g - extent) * (1 + s);
                    let conversion = if s == l { 0 } else { extent + launch };
                    let cost = traffic
                        .checked_add(arithmetic)
                        .and_then(|c| c.checked_add(tail + conversion))
                        .ok_or("kernel cost overflow")?;
                    costs.push((vec![g as i64, s as i64, l as i64], cost));
                    if s == 0 || l == 1 {
                        alternatives.push((g * s, cost));
                    }
                }
            }
        }
        table(b, &[tile, staged, layout], costs);
        groups.push(alternatives);
    }
    b.constraint(Constraint::LinearLe {
        terms: resource,
        rhs: p.capacity as i128,
    });
    Ok(Reference::ResourceChoices {
        capacity: p.capacity,
        modes: vec![(0, 0, groups)],
    })
}

pub(super) fn shared(
    b: &mut ModelBuilder,
    p: &Parameters,
    rng: &mut Random,
) -> LabResult<Reference> {
    let sharing = b.variable("share-preparation", Domain::boolean());
    let retaining = b.variable("retain-preparation", Domain::boolean());
    b.constraint(Constraint::Implies {
        premise: Literal::new(retaining, 1),
        consequence: Literal::new(sharing, 1),
    });
    let prep = p.overhead + p.input_length;
    let retain_storage = p.input_length.div_ceil(2).max(1);
    let mut resource = vec![LinearTerm::new(retaining, i64::try_from(retain_storage)?)];
    table(
        b,
        &[sharing, retaining],
        vec![
            (vec![0, 0], prep * p.n as u64),
            (vec![0, 1], prep * p.n as u64),
            (vec![1, 0], prep),
            (vec![1, 1], prep + p.overhead),
        ],
    );
    let mut modes = vec![
        (0, prep * p.n as u64, Vec::new()),
        (0, prep, Vec::new()),
        (retain_storage, prep + p.overhead, Vec::new()),
    ];
    for i in 0..p.n {
        let tile = choice(b, format!("consumer-{i}-tile"), p.d)?;
        let fused = b.variable(format!("consumer-{i}-fused"), Domain::boolean());
        let layout = b.variable(format!("consumer-{i}-layout"), Domain::boolean());
        let storage = product(b, format!("consumer-{i}-live-storage"), tile, fused, p.d)?;
        resource.push(LinearTerm::new(storage, 1));
        b.constraint(Constraint::Implies {
            premise: Literal::new(fused, 1),
            consequence: Literal::new(sharing, 1),
        });
        b.guarded_constraint(
            vec![Literal::new(fused, 1)],
            Constraint::InDomain {
                variable: layout,
                domain: Domain::singleton(1),
            },
        );
        let extent = p.outputs + rng.next_u64() % 4;
        let compute = 1 + rng.next_u64() % 5;
        let transfer = 1 + rng.next_u64() % 4;
        let mut local = Vec::new();
        for g in 1..=p.d as u64 {
            for f in 0..=1_u64 {
                for l in 0..=1_u64 {
                    let arithmetic =
                        extent.div_ceil(g) * compute + (extent.div_ceil(g) * g - extent);
                    let materialize = if f == 1 {
                        p.overhead + g
                    } else {
                        extent * transfer
                    };
                    let conversion = if f == l { 0 } else { extent + compute };
                    let cost = arithmetic + materialize + conversion;
                    local.push((vec![g as i64, f as i64, l as i64], cost));
                    if f == 0 || l == 1 {
                        for (m, (_, _, groups)) in modes.iter_mut().enumerate() {
                            if groups.len() <= i {
                                groups.push(Vec::new());
                            }
                            if m != 0 || f == 0 {
                                let reread = if m == 1 { extent * transfer / 2 } else { 0 };
                                groups[i].push((g * f, cost + reread));
                            }
                        }
                    }
                }
            }
        }
        table(b, &[tile, fused, layout], local);
        // Shared-but-unretained preparation must be reloaded by each consumer.
        b.guarded_cost(
            vec![Literal::new(sharing, 1), Literal::new(retaining, 0)],
            Cost::Constant(extent * transfer / 2),
        );
    }
    b.constraint(Constraint::LinearLe {
        terms: resource,
        rhs: p.capacity as i128,
    });
    Ok(Reference::ResourceChoices {
        capacity: p.capacity,
        modes,
    })
}

pub(super) fn scheduled(
    b: &mut ModelBuilder,
    p: &Parameters,
    rng: &mut Random,
) -> LabResult<Reference> {
    if p.n > 8 {
        return Err("kernel-schedule supports at most eight explicit tasks; scale repetitions through compact classes separately".into());
    }
    let mut modes = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut durations = Vec::new();
    let mut demands = Vec::new();
    let mut reservations = Vec::new();
    let completion = b.variable(
        "completion",
        Domain::interval(0, i64::try_from(p.horizon)?)?,
    );
    for i in 0..p.n {
        let mode = b.variable(format!("task-{i}-mode"), Domain::boolean());
        let slow = 3 + rng.next_u64() % 3;
        let fast = 1 + rng.next_u64() % 2;
        let ds = [slow, fast];
        let rs = [1, 2];
        let duration = b.variable(
            format!("task-{i}-duration"),
            Domain::set(ds.map(|d| d as i64)),
        );
        let demand = b.variable(format!("task-{i}-demand"), Domain::set([1, 2]));
        b.constraint(Constraint::Table {
            variables: vec![mode, duration, demand],
            tuples: (0..2)
                .map(|m| vec![m as i64, ds[m] as i64, rs[m] as i64])
                .collect(),
        });
        let start = b.variable(
            format!("task-{i}-start"),
            Domain::interval(0, i64::try_from(p.horizon)?)?,
        );
        let end = b.variable(
            format!("task-{i}-end"),
            Domain::interval(0, i64::try_from(p.horizon)?)?,
        );
        reservations.push(ActivityReservation {
            activity: Activity {
                start,
                duration,
                end,
                presence: None,
            },
            demand: Demand::Variable(demand),
        });
        modes.push(mode);
        starts.push(start);
        ends.push(end);
        durations.push((duration, ds));
        demands.push((demand, rs));
    }
    b.constraint(Constraint::Schedule(
        SchedulingConstraint::ActivityCumulative {
            completion,
            capacity: p.capacity,
            activities: reservations,
        },
    ));
    b.cost(Cost::Linear {
        constant: 0,
        terms: vec![LinearTerm::new(completion, 1)],
    });
    Ok(Reference::KernelSchedule {
        modes,
        starts,
        ends,
        durations,
        demands,
        completion,
        horizon: p.horizon,
        capacity: p.capacity,
    })
}
