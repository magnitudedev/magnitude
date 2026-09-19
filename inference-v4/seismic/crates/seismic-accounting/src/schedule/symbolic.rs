//! Joint scheduling relation for unresolved implementation families. A backend
//! supplies typed, source-derived activities and events; every use of a shared
//! resource is collected here, across conditional fragments and lifetimes.
use super::Resource;
use magnitude_solver::model::{
    Arithmetic, Constraint, Domain, LinearTerm, Literal, ModelBuilder, VarId,
};
use magnitude_solver::scheduling::{
    Activity, ActivityReservation, Demand, Event, Reservation, SchedulingConstraint,
};

/// One objective/resource boundary. Variables and this boundary must belong to
/// the same builder. A horizon is supplied by a finite-domain argument or checked
/// upper witness, never inferred from a search budget.
pub struct Encoding {
    pub completion: VarId,
    horizon: i64,
    resources: Vec<Resource>,
    timebase: Option<super::Timebase>,
    reservations: Vec<Vec<Reservation>>,
    whole: Vec<Vec<ActivityReservation>>,
    ends: Vec<Event>,
}
impl Encoding {
    pub fn horizon(&self) -> i64 {
        self.horizon
    }
    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }
    pub fn new(
        builder: &mut ModelBuilder,
        name: &str,
        resources: &[Resource],
        horizon: i64,
    ) -> magnitude_solver::Result<Self> {
        if horizon < 0 {
            return Err(magnitude_solver::Error::InvalidModel(
                "negative scheduling horizon".into(),
            ));
        }
        Ok(Self {
            completion: builder
                .variable(format!("{name}.completion"), Domain::interval(0, horizon)?),
            horizon,
            resources: resources.to_vec(),
            timebase: None,
            reservations: vec![Vec::new(); resources.len()],
            whole: vec![Vec::new(); resources.len()],
            ends: Vec::new(),
        })
    }
    /// Every contribution to one resource boundary uses the same physical tick.
    /// Equivalent rational time units are accepted without changing their scale.
    pub fn bind_timebase(&mut self, timebase: &super::Timebase) -> magnitude_solver::Result<()> {
        if timebase.seconds_numerator == 0 || timebase.seconds_denominator == 0 {
            return Err(magnitude_solver::Error::InvalidModel(
                "scheduling time unit must be positive".into(),
            ));
        }
        if let Some(previous) = &self.timebase {
            if u128::from(previous.seconds_numerator) * u128::from(timebase.seconds_denominator)
                != u128::from(timebase.seconds_numerator) * u128::from(previous.seconds_denominator)
            {
                return Err(magnitude_solver::Error::InvalidModel(
                    "joint scheduling fragments have different time units".into(),
                ));
            }
        } else {
            self.timebase = Some(timebase.clone());
        }
        Ok(())
    }
    /// Presence belongs to the defining implementation alternative. Event edges
    /// and resource uses retain it; absent alternatives contribute no duration.
    pub fn activity(&mut self, builder: &mut ModelBuilder, activity: Activity) {
        let completion = activity.end_event();
        self.define_activity(builder, activity, completion);
    }
    /// Allocate one operation's private times. Their absent value is zero, so
    /// its end is already the exact contribution to the completion maximum.
    /// A zero-duration operation has one time shared by both of its events.
    pub fn operation(
        &mut self,
        builder: &mut ModelBuilder,
        name: &str,
        duration: crate::algebra::Value,
        presence: Option<VarId>,
    ) -> magnitude_solver::Result<Activity> {
        let start = private_time(builder, presence, format!("{name}.start"),
            Domain::interval(0, self.horizon)?)?;
        let end = if duration.bounds().1 == 0 {
            start
        } else {
            private_time(builder, presence, format!("{name}.end"),
                Domain::interval(0, self.horizon)?)?
        };
        let activity = Activity { start, end, duration: duration.id(), presence };
        self.define_activity(builder, activity.clone(), Event::mandatory(end));
        Ok(activity)
    }
    fn define_activity(&mut self, builder: &mut ModelBuilder, activity: Activity, completion: Event) {
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
            activity.clone(),
        )));
        self.ends.push(completion);
        builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
            before: activity.end_event(),
            after: Event::mandatory(self.completion),
            lag: 0,
        }));
    }
    /// Attach resource service whose duration and demand still depend on family
    /// variables. Operation presence guards the offsets, lifetime and resources.
    /// The service must finish within its defining operation, just as in the
    /// concrete accounting validator.
    pub fn service(
        &mut self,
        builder: &mut ModelBuilder,
        operation: &Activity,
        uses: &[crate::algebra::ResourceUse<crate::algebra::Value>],
    ) -> magnitude_solver::Result<()> {
        let guards: Vec<_> = operation
            .presence
            .into_iter()
            .map(|p| Literal::new(p, 1))
            .collect();
        for (index, service) in uses.iter().enumerate() {
            let start = if service.offset == 0 {
                operation.start
            } else {
                let start = private_time(builder, operation.presence,
                    format!("service_{index}.start"),
                    Domain::interval(0, self.horizon)?,
                )?;
                for sign in [1, -1] {
                    builder.guarded_constraint(
                        guards.clone(),
                        Constraint::LinearLe {
                            terms: vec![
                                LinearTerm::new(start, sign),
                                LinearTerm::new(operation.start, -sign),
                            ],
                            rhs: i128::from(service.offset) * i128::from(sign),
                        },
                    );
                }
                start
            };
            let end = private_time(builder, operation.presence,
                format!("service_{index}.end"),
                Domain::interval(0, self.horizon)?,
            )?;
            let activity = Activity {
                start,
                end,
                duration: service.duration.id(),
                presence: operation.presence,
            };
            builder.constraint(Constraint::Schedule(SchedulingConstraint::Activity(
                activity.clone(),
            )));
            builder.constraint(Constraint::Schedule(SchedulingConstraint::Precedence {
                before: activity.end_event(),
                after: operation.end_event(),
                lag: 0,
            }));
            self.whole_activity(
                service.resource,
                activity,
                Demand::Variable(service.units.id()),
            )?;
        }
        Ok(())
    }
    pub fn reservation(
        &mut self,
        resource: usize,
        reservation: Reservation,
    ) -> magnitude_solver::Result<()> {
        self.reservations
            .get_mut(resource)
            .ok_or_else(|| {
                magnitude_solver::Error::InvalidModel(
                    "reservation references an absent resource".into(),
                )
            })?
            .push(reservation);
        Ok(())
    }
    /// Whole-duration information supplements the complete cumulative relation;
    /// it enables incompatibility deductions before timing variables are fixed.
    pub fn whole_activity(
        &mut self,
        resource: usize,
        activity: Activity,
        demand: Demand,
    ) -> magnitude_solver::Result<()> {
        self.reservation(
            resource,
            Reservation {
                interval: activity.interval(),
                demand: demand.clone(),
            },
        )?;
        self.whole
            .get_mut(resource)
            .ok_or_else(|| {
                magnitude_solver::Error::InvalidModel(
                    "activity references an absent resource".into(),
                )
            })?
            .push(ActivityReservation { activity, demand });
        Ok(())
    }
    /// Closes a complete resource boundary, without supplying an objective cost.
    /// The caller may compose this completion with other launch/phase boundaries.
    pub fn finish(self, builder: &mut ModelBuilder) -> magnitude_solver::Result<VarId> {
        let mut ends = Vec::with_capacity(self.ends.len());
        for (index, event) in self.ends.into_iter().enumerate() {
            let end = if let Some(presence) = event.presence {
                let effective = builder.variable(
                    format!("active_completion_{index}"),
                    Domain::interval(0, self.horizon)?,
                );
                let active = Literal::new(presence, 1);
                builder.guarded_constraint(
                    vec![active],
                    Constraint::Equal {
                        left: effective,
                        right: event.time,
                    },
                );
                builder.constraint(Constraint::InactiveValue {
                    active,
                    variable: effective,
                    inactive: 0,
                });
                effective
            } else {
                event.time
            };
            ends.push(end);
        }
        // The maximum is associative. A balanced tree preserves the exact
        // objective while avoiding a linear chain through every occurrence.
        let mut level = 0;
        while ends.len() > 1 {
            let mut next = Vec::with_capacity(ends.len().div_ceil(2));
            for (index, pair) in ends.chunks(2).enumerate() {
                if pair.len() == 1 { next.push(pair[0]); continue; }
                let result = if ends.len() == 2 { self.completion } else {
                    builder.variable(format!("completion_level_{level}_{index}"),
                        Domain::interval(0, self.horizon)?)
                };
                builder.constraint(Constraint::Arithmetic(Arithmetic::Maximum {
                    left: pair[0],
                    right: pair[1],
                    result,
                }));
                next.push(result);
            }
            ends = next;
            level += 1;
        }
        let maximum = ends.first().copied().unwrap_or_else(||
            builder.variable("empty_completion", Domain::singleton(0)));
        if maximum != self.completion {
            builder.constraint(Constraint::Equal { left: self.completion, right: maximum });
        }
        for ((resource, reservations), activities) in self
            .resources
            .into_iter()
            .zip(self.reservations)
            .zip(self.whole)
        {
            // ActivityCumulative includes the exact cumulative relation over
            // its activities. Avoid emitting that relation twice when it covers
            // every use. Event lifetimes still require the joint relation with
            // ALL reservations, including the activity-backed uses.
            if reservations.len() != activities.len() {
                builder.constraint(Constraint::Schedule(SchedulingConstraint::Cumulative {
                    capacity: resource.capacity,
                    reservations,
                }));
            }
            if !activities.is_empty() {
                builder.constraint(Constraint::Schedule(
                    SchedulingConstraint::ActivityCumulative {
                        completion: self.completion,
                        capacity: resource.capacity,
                        activities,
                    },
                ));
            }
        }
        Ok(self.completion)
    }
}

fn private_time(builder:&mut ModelBuilder,presence:Option<VarId>,name:String,domain:Domain)->magnitude_solver::Result<VarId> {
    let append=|builder:&mut ModelBuilder|builder.local_variable(name,domain);
    match presence {Some(active)=>builder.when(Literal::new(active,1),append),None=>append(builder)}
}
