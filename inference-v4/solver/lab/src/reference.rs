//! Independent reference procedures. Enumeration uses a separate direct
//! interpretation, never the production evaluator, propagator or bound rules.
use crate::{
    generators::{Instance, Reference},
    LabResult,
};
use magnitude_solver::model::{
    Arithmetic, Constraint, Cost, Factor, FactorKind, LinearTerm, Model, VarId,
};
use magnitude_solver::scheduling::{Activity, Demand, Interval, Reservation, SchedulingConstraint};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Debug, PartialEq, Eq)]
pub struct Evaluation {
    pub feasible: bool,
    pub cost: u64,
    pub unresolved: bool,
}

pub fn evaluate(model: &Model, values: &[i64]) -> LabResult<Evaluation> {
    if values.len() != model.variables().len() {
        return Err("reference assignment arity".into());
    }
    if model
        .variables()
        .iter()
        .zip(values)
        .any(|(var, &value)| !var.domain.contains(value))
    {
        return Ok(Evaluation {
            feasible: false,
            cost: 0,
            unresolved: false,
        });
    }
    let mut result = Evaluation {
        feasible: true,
        cost: 0,
        unresolved: false,
    };
    for factor in model.factors() {
        evaluate_factor(factor, values, &mut result)?;
        if !result.feasible {
            break;
        }
    }
    Ok(result)
}
fn evaluate_factor(factor: &Factor, values: &[i64], out: &mut Evaluation) -> LabResult<()> {
    if factor
        .guards
        .iter()
        .any(|guard| values[guard.variable.0] != guard.value)
    {
        return Ok(());
    }
    match &factor.kind {
        FactorKind::Constraint(constraint) => out.feasible &= constraint_value(constraint, values)?,
        FactorKind::Cost(cost) => {
            let value = match cost {
                Cost::Constant(value) => *value,
                Cost::Linear { constant, terms } => u64::try_from(
                    affine(terms, values)?
                        .checked_add(*constant)
                        .ok_or("reference objective overflow")?,
                )?,
                Cost::Table { variables, entries } => {
                    entries
                        .iter()
                        .find(|(tuple, _)| {
                            variables
                                .iter()
                                .zip(tuple)
                                .all(|(var, &expected)| values[var.0] == expected)
                        })
                        .ok_or("reference incomplete cost table")?
                        .1
                }
            };
            out.cost = out
                .cost
                .checked_add(value)
                .ok_or("reference sum overflow")?;
        }
        FactorKind::Unresolved { .. } => out.unresolved = true,
        FactorKind::Fragment { factors } => {
            for factor in factors {
                evaluate_factor(factor, values, out)?;
            }
        }
    }
    Ok(())
}
fn affine(terms: &[LinearTerm], values: &[i64]) -> LabResult<i128> {
    terms.iter().try_fold(0_i128, |sum, term| {
        sum.checked_add(
            (term.coefficient as i128)
                .checked_mul(values[term.variable.0] as i128)
                .ok_or("reference product overflow")?,
        )
        .ok_or("reference affine overflow")
        .map_err(Into::into)
    })
}
fn constraint_value(constraint: &Constraint, v: &[i64]) -> LabResult<bool> {
    Ok(match constraint {
        Constraint::Equal { left, right } => v[left.0] == v[right.0],
        Constraint::NotEqual { left, right } => v[left.0] != v[right.0],
        Constraint::InDomain { variable, domain } => domain.contains(v[variable.0]),
        Constraint::LinearLe { terms, rhs } => affine(terms, v)? <= *rhs,
        Constraint::Table { variables, tuples } => tuples.iter().any(|tuple| {
            variables
                .iter()
                .zip(tuple)
                .all(|(var, &value)| v[var.0] == value)
        }),
        Constraint::ExactlyOne { variables } => {
            variables.iter().filter(|var| v[var.0] == 1).count() == 1
        }
        Constraint::BoolAnd { output, inputs } => {
            v[output.0] == i64::from(inputs.iter().all(|var| v[var.0] == 1))
        }
        Constraint::Implies {
            premise,
            consequence,
        } => {
            v[premise.variable.0] != premise.value || v[consequence.variable.0] == consequence.value
        }
        Constraint::InactiveValue {
            active,
            variable,
            inactive,
        } => v[active.variable.0] == active.value || v[variable.0] == *inactive,
        Constraint::Arithmetic(a) => match a {
            Arithmetic::Product {
                left,
                right,
                product,
            } => v[left.0] as i128 * v[right.0] as i128 == v[product.0] as i128,
            Arithmetic::CeilDiv {
                numerator,
                denominator,
                quotient,
            } => {
                v[denominator.0] > 0
                    && (v[numerator.0] as i128 + v[denominator.0] as i128 - 1)
                        / (v[denominator.0] as i128)
                        == v[quotient.0] as i128
            }
            Arithmetic::DivRem {
                numerator,
                denominator,
                quotient,
                remainder,
            } => {
                v[denominator.0] > 0
                    && v[numerator.0] / v[denominator.0] == v[quotient.0]
                    && v[numerator.0] % v[denominator.0] == v[remainder.0]
            }
            Arithmetic::Minimum {
                left,
                right,
                result,
            } => v[left.0].min(v[right.0]) == v[result.0],
            Arithmetic::Maximum {
                left,
                right,
                result,
            } => v[left.0].max(v[right.0]) == v[result.0],
        },
        Constraint::Schedule(schedule) => schedule_value(schedule, v)?,
    })
}
fn present(p: Option<VarId>, v: &[i64]) -> bool {
    p.is_none_or(|p| v[p.0] == 1)
}
fn activity_value(a: &Activity, v: &[i64]) -> bool {
    !present(a.presence, v)
        || (v[a.start.0] >= 0
            && v[a.duration.0] >= 0
            && v[a.end.0] >= 0
            && (v[a.start.0] as i128 + v[a.duration.0] as i128 == v[a.end.0] as i128))
}
fn demand(d: &Demand, v: &[i64]) -> LabResult<Option<u64>> {
    match d {
        Demand::Constant(d) => Ok(Some(*d)),
        Demand::Variable(var) => Ok(u64::try_from(v[var.0]).ok()),
    }
}
fn reserve(interval: &Interval, amount: u64, v: &[i64], events: &mut Vec<(i64, i128)>) -> bool {
    if interval.presence.iter().any(|p| v[p.0] != 1) {
        return true;
    }
    let start = v[interval.start.0];
    let end = v[interval.end.0];
    if start < 0 || end < 0 || end < start {
        return false;
    }
    if start != end {
        events.push((start, amount as i128));
        events.push((end, -(amount as i128)));
    }
    true
}
fn cumulative(capacity: u64, reservations: &[Reservation], v: &[i64]) -> LabResult<bool> {
    let mut events = Vec::new();
    for reservation in reservations {
        if reservation
            .interval
            .presence
            .iter()
            .any(|presence| v[presence.0] != 1)
        {
            continue;
        }
        let Some(amount) = demand(&reservation.demand, v)? else {
            return Ok(false);
        };
        if !reserve(&reservation.interval, amount, v, &mut events) {
            return Ok(false);
        }
    }
    // Sort release before acquire at equal time; exact half-open semantics.
    events.sort_unstable();
    let mut usage = 0_i128;
    for (_, change) in events {
        usage = usage
            .checked_add(change)
            .ok_or("reference cumulative overflow")?;
        if usage > capacity as i128 {
            return Ok(false);
        }
    }
    Ok(true)
}
fn schedule_value(schedule: &SchedulingConstraint, v: &[i64]) -> LabResult<bool> {
    Ok(match schedule {
        SchedulingConstraint::Activity(activity) => activity_value(activity, v),
        SchedulingConstraint::Precedence { before, after, lag } => {
            !present(before.presence, v)
                || !present(after.presence, v)
                || (v[before.time.0] >= 0
                    && v[after.time.0] >= 0
                    && v[before.time.0] as i128 + *lag as i128 <= v[after.time.0] as i128)
        }
        SchedulingConstraint::NoOverlap { intervals } => cumulative(
            1,
            &intervals
                .iter()
                .cloned()
                .map(|interval| Reservation {
                    interval,
                    demand: Demand::Constant(1),
                })
                .collect::<Vec<_>>(),
            v,
        )?,
        SchedulingConstraint::Cumulative {
            capacity,
            reservations,
        } => cumulative(*capacity, reservations, v)?,
        SchedulingConstraint::ActivityCumulative {
            completion,
            capacity,
            activities,
        } => {
            if v[completion.0] < 0
                || activities.iter().any(|a| {
                    !activity_value(&a.activity, v)
                        || (present(a.activity.presence, v)
                            && v[a.activity.end.0] > v[completion.0])
                })
            {
                return Ok(false);
            }
            cumulative(
                *capacity,
                &activities
                    .iter()
                    .map(|a| Reservation {
                        interval: a.activity.interval(),
                        demand: a.demand,
                    })
                    .collect::<Vec<_>>(),
                v,
            )?
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferenceOutcome {
    Optimal(u64),
    Infeasible,
    Incomplete,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReferenceResult {
    pub outcome: ReferenceOutcome,
    pub assignments: u64,
    pub elapsed_ms: f64,
    pub method: String,
    pub reason: Option<String>,
}

pub fn analytic(instance: &Instance) -> LabResult<Option<ReferenceResult>> {
    let start = Instant::now();
    let outcome = match &instance.reference {
        None => return Ok(None),
        Some(Reference::Exact(value)) => ReferenceOutcome::Optimal(*value),
        Some(Reference::Incomplete) => ReferenceOutcome::Incomplete,
        Some(Reference::FixedSchedule { .. } | Reference::KernelSchedule { .. }) => {
            return Ok(None)
        }
        Some(Reference::ResourceChoices { capacity, modes }) => {
            // A sparse multiple-choice knapsack DP. Alternatives are local
            // resource/cost pairs, with global sharing/retention conditioned.
            // Tiny whole-model enumeration audits this oracle separately.
            let mut best = None::<u64>;
            for (initial_use, initial_cost, groups) in modes {
                if initial_use > capacity {
                    continue;
                }
                let mut frontier =
                    std::collections::BTreeMap::from([(*initial_use, *initial_cost)]);
                for group in groups {
                    let mut next = std::collections::BTreeMap::<u64, u64>::new();
                    for (&used, &cost) in &frontier {
                        for &(usage, local_cost) in group {
                            let usage = used
                                .checked_add(usage)
                                .ok_or("reference capacity overflow")?;
                            if usage <= *capacity {
                                let candidate = cost
                                    .checked_add(local_cost)
                                    .ok_or("reference objective overflow")?;
                                next.entry(usage)
                                    .and_modify(|old| *old = (*old).min(candidate))
                                    .or_insert(candidate);
                            }
                        }
                    }
                    // Dominated larger-resource labels can never improve an
                    // additive continuation with one upper capacity constraint.
                    let mut previous = u64::MAX;
                    next.retain(|_, cost| {
                        if *cost < previous {
                            previous = *cost;
                            true
                        } else {
                            false
                        }
                    });
                    frontier = next;
                }
                if let Some(value) = frontier.values().min() {
                    best = Some(best.map_or(*value, |old| old.min(*value)));
                }
            }
            best.map_or(ReferenceOutcome::Infeasible, ReferenceOutcome::Optimal)
        }
        Some(Reference::Chain { local, transitions }) => {
            let mut row = local.first().ok_or("empty chain reference")?.clone();
            if transitions.len() + 1 != local.len() {
                return Err("malformed chain reference".into());
            }
            for (i, local_row) in local.iter().enumerate().skip(1) {
                let mut next = Vec::with_capacity(local_row.len());
                for (r, &cost) in local_row.iter().enumerate() {
                    let minimum = row
                        .iter()
                        .enumerate()
                        .map(|(p, &previous)| {
                            previous
                                .checked_add(transitions[i - 1][p][r])
                                .ok_or("reference objective overflow")
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .into_iter()
                        .min()
                        .ok_or("empty chain row")?;
                    next.push(
                        minimum
                            .checked_add(cost)
                            .ok_or("reference objective overflow")?,
                    );
                }
                row = next;
            }
            ReferenceOutcome::Optimal(*row.iter().min().ok_or("empty chain terminal")?)
        }
    };
    Ok(Some(ReferenceResult {
        outcome,
        assignments: 0,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        method: "independent closed form / dynamic program".into(),
        reason: None,
    }))
}

pub fn exhaustive(instance: &Instance, limit: u64, time: Duration) -> LabResult<ReferenceResult> {
    if matches!(instance.reference, Some(Reference::KernelSchedule { .. })) {
        return implementation_schedule(instance, None, limit, time);
    }
    if let Some(Reference::FixedSchedule {
        starts,
        ends,
        durations,
        completion,
        horizon,
    }) = &instance.reference
    {
        return fixed_schedule(
            instance,
            starts,
            ends,
            durations,
            *completion,
            *horizon,
            limit,
            time,
        );
    }
    let start = Instant::now();
    let domains = instance.model.domains();
    if domains.iter().any(|domain| domain.is_empty()) {
        return Ok(ReferenceResult {
            outcome: ReferenceOutcome::Infeasible,
            assignments: 0,
            elapsed_ms: 0.0,
            method: "exhaustive".into(),
            reason: None,
        });
    }
    let mut values: Vec<_> = domains.iter().map(|domain| domain.min().unwrap()).collect();
    // Per-variable iterators keep million-wide finite domains compact.
    let mut iterators: Vec<_> = domains.iter().map(|domain| domain.values()).collect();
    for it in &mut iterators {
        it.next();
    }
    let mut evaluated = 0_u64;
    let mut best: Option<u64> = None;
    let mut unresolved_bound: Option<u64> = None;
    loop {
        if evaluated >= limit || start.elapsed() >= time {
            return Ok(ReferenceResult {
                outcome: ReferenceOutcome::Incomplete,
                assignments: evaluated,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                method: "exhaustive".into(),
                reason: Some("enumeration budget exhausted".into()),
            });
        }
        let assessment = evaluate(&instance.model, &values)?;
        evaluated += 1;
        if assessment.feasible {
            if assessment.unresolved {
                unresolved_bound =
                    Some(unresolved_bound.map_or(assessment.cost, |old| old.min(assessment.cost)));
            } else {
                best = Some(best.map_or(assessment.cost, |old| old.min(assessment.cost)));
            }
        }
        let mut advance = false;
        for i in (0..values.len()).rev() {
            if let Some(value) = iterators[i].next() {
                values[i] = value;
                advance = true;
                break;
            }
            iterators[i] = domains[i].values();
            values[i] = iterators[i].next().unwrap();
        }
        if !advance {
            break;
        }
    }
    let outcome = match (best, unresolved_bound) {
        (Some(best), Some(lower)) if lower < best => ReferenceOutcome::Incomplete,
        (Some(best), _) => ReferenceOutcome::Optimal(best),
        (None, Some(_)) => ReferenceOutcome::Incomplete,
        (None, None) => ReferenceOutcome::Infeasible,
    };
    Ok(ReferenceResult {
        outcome,
        assignments: evaluated,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        method: "exhaustive".into(),
        reason: unresolved_bound.map(|_| "unresolved domain coverage".into()),
    })
}

#[allow(clippy::too_many_arguments)]
fn fixed_schedule(
    instance: &Instance,
    starts: &[VarId],
    ends: &[VarId],
    durations: &[u64],
    completion: VarId,
    horizon: u64,
    limit: u64,
    time: Duration,
) -> LabResult<ReferenceResult> {
    // This reference class has fixed mandatory durations and objective T=max(end).
    // Enumerating every finite start assignment therefore covers all schedules.
    let begin = Instant::now();
    let mut values: Vec<_> = instance
        .model
        .variables()
        .iter()
        .map(|v| v.domain.min().unwrap_or(0))
        .collect();
    let mut offsets = vec![0_u64; starts.len()];
    let mut count = 0;
    let mut best = None::<u64>;
    loop {
        if count >= limit || begin.elapsed() >= time {
            return Ok(ReferenceResult {
                outcome: ReferenceOutcome::Incomplete,
                assignments: count,
                elapsed_ms: begin.elapsed().as_secs_f64() * 1000.0,
                method: "independent exhaustive fixed-duration schedules".into(),
                reason: Some("enumeration budget exhausted".into()),
            });
        }
        let mut finish = 0;
        for (((&start, &end), &duration), &offset) in
            starts.iter().zip(ends).zip(durations).zip(&offsets)
        {
            values[start.0] = i64::try_from(offset)?;
            let endpoint = offset
                .checked_add(duration)
                .ok_or("reference end overflow")?;
            values[end.0] = i64::try_from(endpoint)?;
            finish = finish.max(endpoint);
        }
        values[completion.0] = i64::try_from(finish)?;
        let result = evaluate(&instance.model, &values)?;
        if result.feasible && !result.unresolved {
            best = Some(best.map_or(result.cost, |old| old.min(result.cost)));
        }
        count += 1;
        let mut changed = false;
        for offset in offsets.iter_mut().rev() {
            if *offset < horizon {
                *offset += 1;
                changed = true;
                break;
            }
            *offset = 0;
        }
        if !changed {
            break;
        }
    }
    Ok(ReferenceResult {
        outcome: best.map_or(ReferenceOutcome::Infeasible, ReferenceOutcome::Optimal),
        assignments: count,
        elapsed_ms: begin.elapsed().as_secs_f64() * 1000.0,
        method: "independent exhaustive fixed-duration schedules".into(),
        reason: None,
    })
}

/// Enumerate complete instruction modes and all bounded integer starts. For a
/// fixed mode vector this returns f*(x), independent of a searcher's timestamps.
/// Completion=max(end) is valid here because these generated models have only
/// an upper-horizon completion domain and a monotone completion objective.
pub fn implementation_schedule(
    instance: &Instance,
    selected: Option<&[i64]>,
    limit: u64,
    time: Duration,
) -> LabResult<ReferenceResult> {
    let Some(Reference::KernelSchedule {
        modes,
        starts,
        ends,
        durations,
        demands,
        completion,
        horizon,
        ..
    }) = &instance.reference
    else {
        return Err("per-implementation schedule reference requires kernel-schedule".into());
    };
    if let Some(values) = selected {
        if values.len() != instance.model.variables().len() {
            return Err("selected assignment arity".into());
        }
    }
    let begin = Instant::now();
    let mut values: Vec<_> = instance
        .model
        .variables()
        .iter()
        .map(|v| v.domain.min().unwrap_or(0))
        .collect();
    let mut count = 0;
    let mut best = None::<u64>;
    let mut winner = None;
    let mode_count = if selected.is_some() {
        1
    } else {
        1_u64
            .checked_shl(modes.len() as u32)
            .ok_or("mode count overflow")?
    };
    for mask in 0..mode_count {
        for i in 0..modes.len() {
            let mode = selected.map_or(((mask >> i) & 1) as i64, |v| v[modes[i].0]);
            if !(0..=1).contains(&mode) {
                return Err("invalid selected instruction mode".into());
            }
            values[modes[i].0] = mode;
            values[durations[i].0 .0] = i64::try_from(durations[i].1[mode as usize])?;
            values[demands[i].0 .0] = i64::try_from(demands[i].1[mode as usize])?;
        }
        let mut offsets = vec![0_u64; starts.len()];
        loop {
            if count >= limit || begin.elapsed() >= time {
                return Ok(ReferenceResult {
                    outcome: ReferenceOutcome::Incomplete,
                    assignments: count,
                    elapsed_ms: begin.elapsed().as_secs_f64() * 1000.0,
                    method: "independent mode/start enumeration".into(),
                    reason: Some("enumeration budget exhausted".into()),
                });
            }
            let mut finish = 0;
            for i in 0..starts.len() {
                values[starts[i].0] = i64::try_from(offsets[i])?;
                let end = offsets[i]
                    .checked_add(values[durations[i].0 .0] as u64)
                    .ok_or("schedule end overflow")?;
                values[ends[i].0] = i64::try_from(end)?;
                finish = finish.max(end);
            }
            values[completion.0] = i64::try_from(finish)?;
            let result = evaluate(&instance.model, &values)?;
            if result.feasible && !result.unresolved && best.is_none_or(|old| result.cost < old) {
                best = Some(result.cost);
                winner = Some(values.clone());
            }
            count += 1;
            let mut changed = false;
            for offset in offsets.iter_mut().rev() {
                if *offset < *horizon {
                    *offset += 1;
                    changed = true;
                    break;
                }
                *offset = 0;
            }
            if !changed {
                break;
            }
        }
    }
    if let Some(witness) = winner {
        let result = instance.model.validate_assignment(&witness)?;
        if result.infeasible || result.exact_cost != best {
            return Err("independent schedule optimum witness failed original model".into());
        }
    }
    Ok(ReferenceResult {
        outcome: best.map_or(ReferenceOutcome::Infeasible, ReferenceOutcome::Optimal),
        assignments: count,
        elapsed_ms: begin.elapsed().as_secs_f64() * 1000.0,
        method: "independent mode/start enumeration".into(),
        reason: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::{conditional_fixture, generate, Parameters};
    use magnitude_solver::model::{Constraint, Domain, ModelBuilder};
    use magnitude_solver::scheduling::{
        Activity, Demand, Interval, Reservation, SchedulingConstraint,
    };
    #[test]
    fn analytic_oracles_match_enumeration() {
        for seed in 0..5 {
            for family in [
                "independent",
                "chain",
                "shared-producer",
                "process-plan",
                "repeated",
                "coverage-gap",
            ] {
                let instance = generate(
                    family,
                    Parameters {
                        n: 3,
                        depth: 1,
                        ..Parameters::default()
                    },
                    seed,
                )
                .unwrap();
                assert_eq!(
                    analytic(&instance).unwrap().unwrap().outcome,
                    exhaustive(&instance, 1_000_000, Duration::from_secs(5))
                        .unwrap()
                        .outcome,
                    "{family} seed {seed}"
                );
            }
        }
        assert_eq!(
            exhaustive(&conditional_fixture().unwrap(), 10, Duration::from_secs(1))
                .unwrap()
                .outcome,
            ReferenceOutcome::Optimal(15)
        );
    }
    #[test]
    fn enumeration_budget_cannot_report_an_optimum() {
        let instance = generate("independent", Parameters::default(), 0).unwrap();
        assert_eq!(
            exhaustive(&instance, 1, Duration::from_secs(1))
                .unwrap()
                .outcome,
            ReferenceOutcome::Incomplete
        );
    }

    #[test]
    fn inactive_negative_resource_fields_are_irrelevant() {
        let mut builder = ModelBuilder::new();
        let active = builder.variable("active", Domain::boolean());
        let start = builder.variable("start", Domain::interval(-1, 1).unwrap());
        let end = builder.variable("end", Domain::interval(-1, 1).unwrap());
        let demand = builder.variable("demand", Domain::interval(-1, 0).unwrap());
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Cumulative {
            capacity: 1,
            reservations: vec![Reservation {
                interval: Interval {
                    start,
                    end,
                    presence: vec![active],
                },
                demand: Demand::Variable(demand),
            }],
        }));
        let model = builder.build().unwrap();
        // Presence=0 means all private interval and demand fields are inactive.
        assert!(evaluate(&model, &[0, -1, -1, -1]).unwrap().feasible);
        // The same negative demand is invalid when the reservation is active.
        assert!(!evaluate(&model, &[1, 0, 0, -1]).unwrap().feasible);
    }

    #[test]
    fn active_negative_timing_is_infeasible() {
        let mut builder = ModelBuilder::new();
        let start = builder.variable("start", Domain::interval(-1, 0).unwrap());
        let duration = builder.variable("duration", Domain::singleton(1));
        let end = builder.variable("end", Domain::interval(0, 1).unwrap());
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            Activity {
                start,
                duration,
                end,
                presence: None,
            },
        )));
        let model = builder.build().unwrap();
        assert!(!evaluate(&model, &[-1, 1, 0]).unwrap().feasible);
    }
}
