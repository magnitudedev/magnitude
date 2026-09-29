//! Finite, integer-time scheduling constraints.
//!
//! Times are nonnegative and reservations are half-open: `[start, end)`.
//! Empty intervals consume no capacity, including when their demand exceeds the
//! resource capacity. Every reservation on one resource must occur in the same
//! cumulative factor, so decomposition sees the complete coupling relation.
//! A finite horizon belongs to the supplied model; no guessed horizon is added.

mod arena;
pub mod repetition;
mod resources;

pub use arena::{arena_offsets, ArenaExpression, ArenaItem, ArenaLiteral, ArenaPacking};
pub use resources::{Demand, Event, Interval, Lifetime, Reservation};

use crate::model::{Assessment, Domain, Error, Propagation, Result, VarId};

/// Optional activity with explicit `end = start + duration` semantics.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Activity {
    pub start: VarId,
    pub duration: VarId,
    pub end: VarId,
    /// Absent activities impose no relationship on their private time values.
    pub presence: Option<VarId>,
}

/// An activity holds this demand for its complete modeled duration. Keeping the
/// duration relation explicit enables energy and incompatibility deductions
/// before its start time is selected.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActivityReservation {
    pub activity: Activity,
    pub demand: Demand,
}

impl Activity {
    pub fn interval(&self) -> Interval {
        Interval {
            start: self.start,
            end: self.end,
            presence: self.presence.into_iter().collect(),
        }
    }

    pub fn start_event(&self) -> Event {
        Event {
            time: self.start,
            presence: self.presence,
        }
    }

    pub fn end_event(&self) -> Event {
        Event {
            time: self.end,
            presence: self.presence,
        }
    }
}

/// Closed scheduling vocabulary: evaluation and propagation share these types.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SchedulingConstraint {
    Activity(Activity),
    Precedence {
        before: Event,
        after: Event,
        lag: u64,
    },
    NoOverlap {
        intervals: Vec<Interval>,
    },
    Cumulative {
        capacity: u64,
        reservations: Vec<Reservation>,
    },
    /// Joint activity/capacity/makespan relation. This owns all activity equations
    /// and `end <= completion`, rather than trusting a caller-supplied cost floor.
    ActivityCumulative {
        completion: VarId,
        capacity: u64,
        activities: Vec<ActivityReservation>,
    },
    ArenaPacking(ArenaPacking),
}

impl SchedulingConstraint {
    /// Renames an immutable fragment's local variables into its instance.
    pub fn remap(&self, variables: &[VarId]) -> Self {
        let event = |event: &Event| Event {
            time: variables[event.time.0],
            presence: event.presence.map(|v| variables[v.0]),
        };
        let interval = |interval: &Interval| Interval {
            start: variables[interval.start.0],
            end: variables[interval.end.0],
            presence: interval.presence.iter().map(|v| variables[v.0]).collect(),
        };
        match self {
            Self::Activity(a) => Self::Activity(Activity {
                start: variables[a.start.0],
                duration: variables[a.duration.0],
                end: variables[a.end.0],
                presence: a.presence.map(|v| variables[v.0]),
            }),
            Self::Precedence { before, after, lag } => Self::Precedence {
                before: event(before),
                after: event(after),
                lag: *lag,
            },
            Self::NoOverlap { intervals } => Self::NoOverlap {
                intervals: intervals.iter().map(interval).collect(),
            },
            Self::Cumulative {
                capacity,
                reservations,
            } => Self::Cumulative {
                capacity: *capacity,
                reservations: reservations
                    .iter()
                    .map(|r| Reservation {
                        interval: interval(&r.interval),
                        demand: match r.demand {
                            Demand::Constant(c) => Demand::Constant(c),
                            Demand::Variable(v) => Demand::Variable(variables[v.0]),
                        },
                    })
                    .collect(),
            },
            Self::ActivityCumulative {
                completion,
                capacity,
                activities,
            } => Self::ActivityCumulative {
                completion: variables[completion.0],
                capacity: *capacity,
                activities: activities
                    .iter()
                    .map(|reservation| {
                        let a = &reservation.activity;
                        ActivityReservation {
                            activity: Activity {
                                start: variables[a.start.0],
                                duration: variables[a.duration.0],
                                end: variables[a.end.0],
                                presence: a.presence.map(|v| variables[v.0]),
                            },
                            demand: match reservation.demand {
                                Demand::Constant(c) => Demand::Constant(c),
                                Demand::Variable(v) => Demand::Variable(variables[v.0]),
                            },
                        }
                    })
                    .collect(),
            },
            Self::ArenaPacking(packing) => Self::ArenaPacking(packing.remap(variables)),
        }
    }

    pub fn scope(&self) -> Vec<VarId> {
        let mut scope = Vec::new();
        match self {
            Self::Activity(a) => {
                scope.extend([a.start, a.duration, a.end]);
                scope.extend(a.presence);
            }
            Self::Precedence { before, after, .. } => {
                scope.extend([before.time, after.time]);
                scope.extend(before.presence);
                scope.extend(after.presence);
            }
            Self::NoOverlap { intervals } => {
                for interval in intervals {
                    interval.add_scope(&mut scope);
                }
            }
            Self::Cumulative { reservations, .. } => {
                for reservation in reservations {
                    reservation.interval.add_scope(&mut scope);
                    if let Demand::Variable(v) = reservation.demand {
                        scope.push(v);
                    }
                }
            }
            Self::ActivityCumulative {
                completion,
                activities,
                ..
            } => {
                scope.push(*completion);
                for reservation in activities {
                    scope.extend([
                        reservation.activity.start,
                        reservation.activity.duration,
                        reservation.activity.end,
                    ]);
                    scope.extend(reservation.activity.presence);
                    if let Demand::Variable(v) = reservation.demand {
                        scope.push(v);
                    }
                }
            }
            Self::ArenaPacking(packing) => scope.extend(packing.scope()),
        }
        scope.sort_unstable();
        scope.dedup();
        scope
    }

    pub fn validate(&self, domains: &[Domain]) -> Result<()> {
        for variable in self.scope() {
            domain(domains, variable)?;
        }
        match self {
            Self::Activity(a) => validate_presence(domains, a.presence),
            Self::Precedence { before, after, .. } => {
                validate_presence(domains, before.presence.into_iter().chain(after.presence))
            }
            Self::NoOverlap { intervals } => {
                for interval in intervals {
                    validate_presence(domains, interval.presence.iter().copied())?;
                }
                Ok(())
            }
            Self::Cumulative { reservations, .. } => {
                for reservation in reservations {
                    validate_presence(domains, reservation.interval.presence.iter().copied())?;
                }
                Ok(())
            }
            Self::ActivityCumulative { activities, .. } => {
                for reservation in activities {
                    validate_presence(domains, reservation.activity.presence)?;
                }
                Ok(())
            }
            Self::ArenaPacking(packing) => packing.validate(domains),
        }
    }

    pub fn assess(&self, domains: &[Domain]) -> Result<Assessment> {
        if let Self::ArenaPacking(packing) = self {
            return packing.assess(domains);
        }
        let mut narrowed = domains.to_vec();
        let outcome = self.propagate(&mut narrowed)?;
        // Private values of a definitely absent optional activity are not
        // effective choices. Keeping them in the static scope is necessary for
        // validation/decomposition safety, but would make the solver branch over
        // unconstrained garbage unless exactness uses the active scope.
        let exact = !outcome.infeasible
            && self
                .effective_scope(domains)
                .iter()
                .all(|v| domains[v.0].is_singleton());
        Ok(Assessment {
            infeasible: outcome.infeasible,
            lower_bound: 0,
            exact_cost: exact.then_some(0),
            unresolved: None,
        })
    }

    /// Variables that can still affect this constraint under the current
    /// activation domains. This is deliberately a conservative dynamic scope:
    /// unknown presence retains all private values, while presence=0 removes the
    /// inactive body's timing and demand variables from the residual problem.
    pub fn effective_scope(&self, domains: &[Domain]) -> Vec<VarId> {
        let mut scope = Vec::new();
        let add_event = |event: &Event, scope: &mut Vec<VarId>| {
            if event
                .presence
                .is_some_and(|v| domains[v.0].singleton_value() == Some(0))
            {
                return;
            }
            scope.push(event.time);
            scope.extend(event.presence);
        };
        let add_interval = |interval: &Interval, scope: &mut Vec<VarId>| {
            if interval
                .presence
                .iter()
                .any(|v| domains[v.0].singleton_value() == Some(0))
            {
                return;
            }
            interval.add_scope(scope);
        };
        match self {
            Self::Activity(activity) => {
                if activity
                    .presence
                    .is_none_or(|v| domains[v.0].singleton_value() != Some(0))
                {
                    scope.extend([activity.start, activity.duration, activity.end]);
                    scope.extend(activity.presence);
                }
            }
            Self::Precedence { before, after, .. } => {
                // The relation is inactive when either endpoint is definitely
                // absent; the other endpoint's private time is irrelevant too.
                if before
                    .presence
                    .is_none_or(|v| domains[v.0].singleton_value() != Some(0))
                    && after
                        .presence
                        .is_none_or(|v| domains[v.0].singleton_value() != Some(0))
                {
                    add_event(before, &mut scope);
                    add_event(after, &mut scope);
                }
            }
            Self::NoOverlap { intervals } => {
                for interval in intervals {
                    add_interval(interval, &mut scope);
                }
            }
            Self::Cumulative { reservations, .. } => {
                for reservation in reservations {
                    add_interval(&reservation.interval, &mut scope);
                    if !reservation
                        .interval
                        .presence
                        .iter()
                        .any(|v| domains[v.0].singleton_value() == Some(0))
                    {
                        if let Demand::Variable(v) = reservation.demand {
                            scope.push(v);
                        }
                    }
                }
            }
            Self::ActivityCumulative {
                completion,
                activities,
                ..
            } => {
                let mut active = false;
                for reservation in activities {
                    if reservation
                        .activity
                        .presence
                        .is_some_and(|v| domains[v.0].singleton_value() == Some(0))
                    {
                        continue;
                    }
                    active = true;
                    scope.extend([
                        reservation.activity.start,
                        reservation.activity.duration,
                        reservation.activity.end,
                    ]);
                    scope.extend(reservation.activity.presence);
                    if let Demand::Variable(v) = reservation.demand {
                        scope.push(v);
                    }
                }
                if active {
                    scope.push(*completion);
                } else if domains[completion.0].min().is_some_and(|value| value < 0) {
                    // Propagation enforces the nonnegative event-time domain even
                    // when no activity is present. Retain a scope until that
                    // invalid portion of the caller domain is eliminated.
                    scope.push(*completion);
                }
            }
            Self::ArenaPacking(packing) => scope.extend(packing.scope()),
        }
        scope.sort_unstable();
        scope.dedup();
        scope
    }

    pub fn propagate(&self, domains: &mut [Domain]) -> Result<Propagation> {
        // Validation is also necessary when this entry point is called directly.
        self.validate(domains)?;
        if self.scope().iter().any(|v| domains[v.0].is_empty()) {
            return Ok(Propagation {
                changed: false,
                infeasible: true,
            });
        }
        let mut state = Narrowing {
            domains,
            changed: false,
            infeasible: false,
        };
        match self {
            Self::Activity(activity) => propagate_activity(activity, &mut state)?,
            Self::Precedence { before, after, lag } => {
                if present(
                    state.domains,
                    before.presence.into_iter().chain(after.presence),
                ) {
                    state.restrict(before.time, 0, i64::MAX as i128)?;
                    state.restrict(after.time, 0, i64::MAX as i128)?;
                    propagate_precedence(before.time, after.time, *lag, &mut state)?;
                }
            }
            Self::NoOverlap { intervals } => {
                let reservations: Vec<_> = intervals
                    .iter()
                    .cloned()
                    .map(|interval| Reservation {
                        interval,
                        demand: Demand::Constant(1),
                    })
                    .collect();
                resources::propagate_capacity(1, &reservations, &mut state)?;
            }
            Self::Cumulative {
                capacity,
                reservations,
            } => {
                resources::propagate_capacity(*capacity, reservations, &mut state)?;
            }
            Self::ActivityCumulative {
                completion,
                capacity,
                activities,
            } => {
                state.restrict(*completion, 0, i64::MAX as i128)?;
                for reservation in activities {
                    propagate_activity(&reservation.activity, &mut state)?;
                    if present(state.domains, reservation.activity.presence) {
                        propagate_precedence(reservation.activity.end, *completion, 0, &mut state)?;
                    }
                }
                if !state.infeasible {
                    let reservations: Vec<_> = activities
                        .iter()
                        .map(|r| Reservation {
                            interval: r.activity.interval(),
                            demand: r.demand,
                        })
                        .collect();
                    resources::propagate_capacity(*capacity, &reservations, &mut state)?;
                    if !state.infeasible {
                        resources::propagate_activity_bounds(
                            *completion,
                            *capacity,
                            activities,
                            &mut state,
                        )?;
                    }
                }
            }
            Self::ArenaPacking(packing) => {
                if packing.assess(state.domains)?.infeasible {
                    state.infeasible = true;
                }
            }
        }
        Ok(Propagation {
            changed: state.changed,
            infeasible: state.infeasible,
        })
    }
}

fn propagate_activity(activity: &Activity, state: &mut Narrowing<'_>) -> Result<()> {
    if !present(state.domains, activity.presence) {
        return Ok(());
    }
    for variable in [activity.start, activity.duration, activity.end] {
        state.restrict(variable, 0, i64::MAX as i128)?;
    }
    if state.infeasible {
        return Ok(());
    }
    let (s0, s1) = state.range(activity.start);
    let (d0, d1) = state.range(activity.duration);
    state.restrict(activity.end, s0 + d0, s1 + d1)?;
    if state.infeasible {
        return Ok(());
    }
    let (e0, e1) = state.range(activity.end);
    state.restrict(activity.start, e0 - d1, e1 - d0)?;
    state.restrict(activity.duration, e0 - s1, e1 - s0)?;
    Ok(())
}

pub(crate) fn propagate_precedence(
    before: VarId,
    after: VarId,
    lag: u64,
    state: &mut Narrowing<'_>,
) -> Result<()> {
    if state.infeasible {
        return Ok(());
    }
    let (b0, _) = state.range(before);
    let (_, a1) = state.range(after);
    state.restrict(after, b0 + lag as i128, i64::MAX as i128)?;
    state.restrict(before, 0, a1 - lag as i128)?;
    Ok(())
}

pub(crate) fn domain(domains: &[Domain], variable: VarId) -> Result<&Domain> {
    domains
        .get(variable.0)
        .ok_or_else(|| Error::InvalidModel(format!("unknown scheduling variable {}", variable.0)))
}

fn validate_presence(domains: &[Domain], variables: impl IntoIterator<Item = VarId>) -> Result<()> {
    for variable in variables {
        let d = domain(domains, variable)?;
        if d.min().is_some_and(|x| x < 0) || d.max().is_some_and(|x| x > 1) {
            return Err(Error::InvalidModel(format!(
                "presence variable {} must be Boolean",
                variable.0
            )));
        }
    }
    Ok(())
}

pub(crate) fn present(domains: &[Domain], variables: impl IntoIterator<Item = VarId>) -> bool {
    variables
        .into_iter()
        .all(|v| domains[v.0].singleton_value() == Some(1))
}

/// Range restriction retains the domain's explicit holes and progression step.
/// Wider intermediates represent implied ranges, never saturating objective costs.
pub(crate) struct Narrowing<'a> {
    pub domains: &'a mut [Domain],
    pub changed: bool,
    pub infeasible: bool,
}

impl Narrowing<'_> {
    pub fn range(&self, variable: VarId) -> (i128, i128) {
        let domain = &self.domains[variable.0];
        (
            domain.min().expect("nonempty domain") as i128,
            domain.max().expect("nonempty domain") as i128,
        )
    }

    pub fn restrict(&mut self, variable: VarId, min: i128, max: i128) -> Result<()> {
        if self.infeasible {
            return Ok(());
        }
        let new = if min > max || min > i64::MAX as i128 || max < i64::MIN as i128 {
            Domain::set([])
        } else {
            self.domains[variable.0].restrict(
                min.max(i64::MIN as i128) as i64,
                max.min(i64::MAX as i128) as i64,
            )?
        };
        self.changed |= new != self.domains[variable.0];
        self.infeasible |= new.is_empty();
        self.domains[variable.0] = new;
        Ok(())
    }
}
