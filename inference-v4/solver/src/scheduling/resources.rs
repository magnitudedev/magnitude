use super::ActivityReservation;
use super::{present, propagate_precedence, Narrowing};
use crate::model::{Error, Result, VarId};
use std::collections::BTreeMap;

/// An event may be absent along with its enclosing activity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Event {
    pub time: VarId,
    pub presence: Option<VarId>,
}

impl Event {
    pub fn mandatory(time: VarId) -> Self {
        Self {
            time,
            presence: None,
        }
    }
}

/// A half-open interval. It exists only when every listed presence is true.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interval {
    pub start: VarId,
    pub end: VarId,
    pub presence: Vec<VarId>,
}

impl Interval {
    pub fn mandatory(start: VarId, end: VarId) -> Self {
        Self {
            start,
            end,
            presence: Vec::new(),
        }
    }

    pub(crate) fn add_scope(&self, scope: &mut Vec<VarId>) {
        scope.extend([self.start, self.end]);
        scope.extend(self.presence.iter().copied());
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Demand {
    Constant(u64),
    Variable(VarId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Reservation {
    pub interval: Interval,
    pub demand: Demand,
}

/// Storage held between actual events; no guessed duration is substituted.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lifetime {
    pub begin: Event,
    pub end: Event,
    pub demand: Demand,
}

impl Lifetime {
    pub fn reservation(&self) -> Reservation {
        let mut presence: Vec<_> = self
            .begin
            .presence
            .into_iter()
            .chain(self.end.presence)
            .collect();
        presence.sort_unstable();
        presence.dedup();
        Reservation {
            interval: Interval {
                start: self.begin.time,
                end: self.end.time,
                presence,
            },
            demand: self.demand,
        }
    }
}

#[derive(Clone, Copy)]
struct ActiveReservation {
    start: VarId,
    end: VarId,
    demand: u64,
}

/// Necessary compulsory-part and energy checks. At a complete assignment these
/// become exact capacity validation. All intermediate sums are checked.
pub(crate) fn propagate_capacity(
    capacity: u64,
    reservations: &[Reservation],
    state: &mut Narrowing<'_>,
) -> Result<()> {
    let mut active = Vec::new();
    for reservation in reservations {
        if !present(state.domains, reservation.interval.presence.iter().copied()) {
            continue;
        }
        let interval = &reservation.interval;
        state.restrict(interval.start, 0, i64::MAX as i128)?;
        state.restrict(interval.end, 0, i64::MAX as i128)?;
        propagate_precedence(interval.start, interval.end, 0, state)?;
        if state.infeasible {
            return Ok(());
        }
        let demand = match reservation.demand {
            Demand::Constant(value) => value,
            Demand::Variable(variable) => {
                state.restrict(variable, 0, i64::MAX as i128)?;
                if state.infeasible {
                    return Ok(());
                }
                state.range(variable).0 as u64
            }
        };
        active.push(ActiveReservation {
            start: interval.start,
            end: interval.end,
            demand,
        });
    }

    let mut events: BTreeMap<i128, i128> = BTreeMap::new();
    let mut energy = 0u128;
    let mut earliest = None::<i128>;
    let mut latest = None::<i128>;
    for reservation in &active {
        let (start_min, start_max) = state.range(reservation.start);
        let (end_min, end_max) = state.range(reservation.end);
        let duration_min = (end_min - start_max).max(0) as u128;
        if duration_min > 0 && reservation.demand > capacity {
            state.infeasible = true;
            return Ok(());
        }
        let work = duration_min
            .checked_mul(reservation.demand as u128)
            .ok_or_else(|| Error::Overflow("resource work".into()))?;
        energy = energy
            .checked_add(work)
            .ok_or_else(|| Error::Overflow("resource energy sum".into()))?;
        earliest = Some(earliest.map_or(start_min, |old| old.min(start_min)));
        latest = Some(latest.map_or(end_max, |old| old.max(end_max)));
        if start_max < end_min && reservation.demand > 0 {
            add_event(&mut events, start_max, reservation.demand as i128)?;
            add_event(&mut events, end_min, -(reservation.demand as i128))?;
        }
    }
    // Aggregating all changes at a timestamp implements release-before-acquire.
    let mut usage = 0i128;
    for change in events.values() {
        usage = usage
            .checked_add(*change)
            .ok_or_else(|| Error::Overflow("resource usage".into()))?;
        if usage > capacity as i128 {
            state.infeasible = true;
            return Ok(());
        }
    }
    if let (Some(begin), Some(end)) = (earliest, latest) {
        let available = ((end - begin).max(0) as u128)
            .checked_mul(capacity as u128)
            .ok_or_else(|| Error::Overflow("resource energy envelope".into()))?;
        if energy > available {
            state.infeasible = true;
            return Ok(());
        }
    }

    // Disjunctive ordering is sound only when both intervals must be nonempty.
    // Empty intervals otherwise permit either apparent overlap without capacity.
    for (i, left) in active.iter().enumerate() {
        for right in &active[i + 1..] {
            if left.demand as u128 + right.demand as u128 <= capacity as u128 {
                continue;
            }
            let (_, ls1) = state.range(left.start);
            let (le0, _) = state.range(left.end);
            let (_, rs1) = state.range(right.start);
            let (re0, _) = state.range(right.end);
            if le0 <= ls1 || re0 <= rs1 {
                continue;
            }
            let left_first = le0 <= rs1;
            let right_first = re0 <= ls1;
            match (left_first, right_first) {
                (false, false) => {
                    state.infeasible = true;
                    return Ok(());
                }
                (true, false) => propagate_precedence(left.end, right.start, 0, state)?,
                (false, true) => propagate_precedence(right.end, left.start, 0, state)?,
                (true, true) => {}
            }
            if state.infeasible {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn add_event(events: &mut BTreeMap<i128, i128>, time: i128, delta: i128) -> Result<()> {
    let previous = events.entry(time).or_default();
    *previous = previous
        .checked_add(delta)
        .ok_or_else(|| Error::Overflow("resource event sum".into()))?;
    Ok(())
}

/// Mandatory activities supply energy and disjunctive bounds even when their
/// time windows have no compulsory part. Demand incompatibility is a threshold
/// graph: a clique is determined by its least-demand member. Enumerating those
/// members avoids the cubic greedy-clique scan and covers every maximal clique.
pub(crate) fn propagate_activity_bounds(
    completion: VarId,
    capacity: u64,
    activities: &[ActivityReservation],
    state: &mut Narrowing<'_>,
) -> Result<()> {
    let mut tasks = Vec::new();
    for reservation in activities {
        let activity = &reservation.activity;
        if !present(state.domains, activity.presence) {
            continue;
        }
        let duration = state.range(activity.duration).0 as u128;
        if duration == 0 {
            continue;
        }
        let demand = match reservation.demand {
            Demand::Constant(d) => d,
            Demand::Variable(v) => state.range(v).0 as u64,
        };
        if demand > capacity {
            state.infeasible = true;
            return Ok(());
        }
        tasks.push((activity, duration, demand));
    }
    // Positive duration, rather than a compulsory part, establishes that a pair
    // of incompatible activities must choose one of the two possible orders.
    for (i, &(left, _, ld)) in tasks.iter().enumerate() {
        for &(right, _, rd) in &tasks[i + 1..] {
            if ld as u128 + rd as u128 <= capacity as u128 {
                continue;
            }
            let left_first = state.range(left.end).0 <= state.range(right.start).1;
            let right_first = state.range(right.end).0 <= state.range(left.start).1;
            match (left_first, right_first) {
                (false, false) => state.infeasible = true,
                (true, false) => propagate_precedence(left.end, right.start, 0, state)?,
                (false, true) => propagate_precedence(right.end, left.start, 0, state)?,
                (true, true) => (),
            }
            if state.infeasible {
                return Ok(());
            }
        }
    }
    // Every suffix by earliest start has its own energy envelope. An unrelated
    // early activity must not weaken a bound on work that is all released late.
    tasks.sort_by_key(|&(a, _, _)| std::cmp::Reverse(state.range(a.start).0));
    let mut energy = 0u128;
    let mut lower = state.range(completion).0;
    for &(activity, duration, demand) in &tasks {
        energy = energy
            .checked_add(
                duration
                    .checked_mul(demand as u128)
                    .ok_or_else(|| Error::Overflow("activity energy".into()))?,
            )
            .ok_or_else(|| Error::Overflow("activity energy sum".into()))?;
        if energy > 0 {
            // Positive energy implies positive demand, already checked <= capacity.
            lower = lower.max(completion_bound(
                state.range(activity.start).0,
                energy.div_ceil(capacity as u128),
            )?);
        }
    }
    // For each least-demand seed, include all no-smaller incompatible members.
    // Those members are mutually incompatible too, so every release-time suffix
    // must execute serially. Equal-demand seeds intentionally duplicate a bound;
    // no additional clique membership search or allocation is necessary.
    for (seed, &(_, _, least)) in tasks.iter().enumerate() {
        let mut duration = 0u128;
        for (candidate, &(activity, ticks, demand)) in tasks.iter().enumerate() {
            if seed != candidate
                && !(demand >= least && demand as u128 + least as u128 > capacity as u128)
            {
                continue;
            }
            duration = duration
                .checked_add(ticks)
                .ok_or_else(|| Error::Overflow("incompatibility duration sum".into()))?;
            lower = lower.max(completion_bound(state.range(activity.start).0, duration)?);
        }
    }
    state.restrict(completion, lower, i64::MAX as i128)
}

fn completion_bound(begin: i128, ticks: u128) -> Result<i128> {
    let ticks =
        i128::try_from(ticks).map_err(|_| Error::Overflow("activity completion time".into()))?;
    begin
        .checked_add(ticks)
        .ok_or_else(|| Error::Overflow("activity completion bound".into()))
}
