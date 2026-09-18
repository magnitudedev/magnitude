//! Retained scheduling subproblems and checked composition of their witnesses.
//! Serial optima add; parallel lower bounds take a maximum. Peak envelopes admit
//! safe overlap; retained exact occupancy profiles additionally search staggered
//! starts. Restricted profile optima never exclude unresolved interleavings.
use super::*;

#[derive(Clone)]
pub(crate) enum Outcome { Feasible(Witness), Incomplete { lower_bound: u64 }, Infeasible }
enum State {
    Settled,
    Flat(super::super::Search),
    Compose { order: Order, children: Vec<Refinement>, profile_limit: u64, overlap: Option<super::overlap::Search> },
    Repeat { order: Order, count: u64, body: Box<Refinement>, next_period: u64, expansion_limit: u64, periodic_body: Option<(Witness, Option<super::periodic::Profile>)> },
    Scope(Box<Refinement>),
    Unresolved,
}
pub(crate) struct Refinement {
    model: Arc<Structured>,
    state: State,
    outcome: Outcome,
}
impl Refinement {
    pub(crate) fn new(model: Structured, expansion_limit: u64) -> Result<Self, String> {
        let lower_bound = model.lower_bound()?;
        let witness = if model.unmapped.is_empty() { model.compact_witness()? } else { None };
        let settled = witness.as_ref().is_some_and(Witness::is_optimal);
        let outcome = witness.map_or(Outcome::Incomplete { lower_bound }, Outcome::Feasible);
        let state = if settled { State::Settled }
        else if !model.unmapped.is_empty() { State::Unresolved }
        else {
            match model.root.as_ref() {
                Node::Compose { order, children } if *order == Order::Serial || children.len() <= 1 => State::Compose { order: *order, profile_limit: expansion_limit, overlap: None, children: children.iter()
                    .map(|child| Self::new(model.with_root(child.clone()), expansion_limit)).collect::<Result<_, _>>()? },
                Node::Repeat { order: Order::Serial, count, body } => State::Repeat { order: Order::Serial, count: *count, next_period: 1, expansion_limit, periodic_body: None,
                    body: Box::new(Self::new(model.with_root(body.clone()), expansion_limit)?) },
                Node::Scope { .. } if model.scope_body()?.is_some() => State::Scope(Box::new(Self::new(
                    model.scope_body()?.expect("scope residual model was checked"), expansion_limit)?)),
                _ => match model.expand(expansion_limit) {
                    Ok(flat) => State::Flat(flat.start_search()?),
                    Err(crate::workload::DerivationError::Exhausted(_)) => match model.root.as_ref() {
                        Node::Repeat { order: Order::Parallel, count, body } => State::Repeat { order: Order::Parallel, next_period: 1, expansion_limit, periodic_body: None,
                            count: *count, body: Box::new(Self::new(model.with_root(body.clone()), expansion_limit)?) },
                        Node::Compose { order, children } => State::Compose { order: *order, profile_limit: expansion_limit, overlap: None, children: children.iter()
                            .map(|child| Self::new(model.with_root(child.clone()), expansion_limit)).collect::<Result<_, _>>()? },
                        _ => State::Unresolved,
                    },
                    Err(error) => return Err(error.to_string()),
                },
            }
        };
        Ok(Self { model: Arc::new(model), state, outcome })
    }
    pub(crate) fn model(&self) -> &Structured { &self.model }
    fn can_advance(&self) -> bool {
        match &self.state {
            State::Settled | State::Unresolved => false,
            State::Flat(_) => true,
            State::Compose { order, children, overlap, .. } => children.iter().any(Self::can_advance) ||
                (*order == Order::Parallel && overlap.as_ref().is_none_or(super::overlap::Search::can_advance)),
            State::Repeat { order, body, next_period, .. } => body.can_advance() || (*order == Order::Parallel && matches!(&body.outcome, Outcome::Feasible(w) if *next_period < w.completion)),
            State::Scope(body) => body.can_advance(),
        }
    }
    pub(crate) fn advance(&mut self, budget: u64) -> Result<Outcome, String> {
        if budget == 0 { return Ok(self.outcome.clone()); }
        let plan = match &mut self.state {
            State::Settled | State::Unresolved => return Ok(self.outcome.clone()),
            State::Flat(search) => match search.advance(budget)? {
                super::super::SearchOutcome::Feasible(solution) => Some(Arc::new(Plan::Flat(solution))),
                super::super::SearchOutcome::Incomplete { lower_bound } => {
                    if let Outcome::Incomplete { lower_bound: previous } = &mut self.outcome { *previous = (*previous).max(lower_bound); }
                    None
                }
                super::super::SearchOutcome::Infeasible => { self.outcome = Outcome::Infeasible; self.state = State::Settled; return Ok(self.outcome.clone()); }
            },
            State::Compose { order, children, profile_limit, overlap } => {
                // The budget is shared across unfinished children. Resumption
                // retains each child's interval frontier and incumbent.
                let child_budget = if *order == Order::Parallel { (budget / 2).max(1) } else { budget };
                let mut remaining = child_budget;
                let active = children.iter().filter(|child| child.can_advance()).count() as u64;
                let share = (child_budget / active.max(1)).max(1);
                for child in children.iter_mut() {
                    if remaining == 0 { break; }
                    if child.can_advance() {
                        let assigned = remaining.min(share);
                        child.advance(assigned)?;
                        remaining -= assigned;
                    }
                }
                let mut plans = Vec::new();
                let mut lower = 0u64;
                let mut duration = 0u64;
                for child in children.iter() {
                    match &child.outcome {
                        Outcome::Infeasible => { self.outcome = Outcome::Infeasible; self.state = State::Settled; return Ok(self.outcome.clone()); }
                        Outcome::Incomplete { lower_bound } => { lower = if *order == Order::Serial { lower.checked_add(*lower_bound).ok_or("serial refinement bound overflow")? } else { lower.max(*lower_bound) }; }
                        Outcome::Feasible(witness) => {
                            lower = if *order == Order::Serial { lower.checked_add(witness.lower_bound).ok_or("serial refinement bound overflow")? } else { lower.max(witness.lower_bound) };
                            duration = duration.checked_add(witness.completion).ok_or("serial refinement completion overflow")?;
                            plans.push(Arc::new(Plan::Selected(Arc::new(witness.clone()))));
                        }
                    }
                }
                if plans.len() == children.len() {
                    if *order == Order::Parallel {
                        if self.model.parallel_fits(&plans)? {
                            *overlap = Some(super::overlap::Search::settled(plans.clone(), *profile_limit));
                            let duration = plans.iter().map(|p| p.duration()).max().unwrap_or(0);
                            Some(Arc::new(Plan::Parallel { children: plans, duration }))
                        } else {
                            if overlap.as_ref().is_none_or(|search| !search.matches(&plans)) {
                                *overlap = Some(super::overlap::Search::new(&self.model, plans.clone(), *profile_limit)?);
                            }
                            let budget = budget - (child_budget - remaining);
                            let candidate = overlap.as_mut().expect("retained overlap frontier").advance(budget)?;
                            Some(candidate.unwrap_or_else(|| Arc::new(Plan::Sequence { children: plans, duration })))
                        }
                    } else { Some(Arc::new(Plan::Sequence { children: plans, duration })) }
                }
                else { self.outcome = Outcome::Incomplete { lower_bound: self.model.lower_bound()?.max(lower) }; None }
            }
            State::Repeat { order, count, body, next_period, expansion_limit, periodic_body } => {
                let child_budget = if body.can_advance() { if *order == Order::Parallel { (budget / 2).max(1) } else { budget } } else { 0 };
                match body.advance(child_budget)? {
                    Outcome::Feasible(witness) => {
                        if *order == Order::Parallel {
                            if periodic_body.as_ref().map(|(body, _)| body) != Some(&witness) {
                                let profile = witness.periodic_profile(*expansion_limit)?;
                                *next_period = if profile.is_some() { 1 } else { witness.completion };
                                *periodic_body = Some((witness.clone(), profile));
                            }
                            let mut best = self.model.parallel_plan(witness.clone())?;
                            // A bounded, retained initiation-period frontier.
                            // These are feasible schedules of the same repeated
                            // model, not a restriction on its legal schedules.
                            let remaining = budget - child_budget;
                            let stop = next_period.saturating_add(remaining).min(witness.completion);
                            while *next_period < stop {
                                let period = *next_period;
                                *next_period += 1;
                                if *count == 0 { break; }
                                if let Some(plan) = witness.periodic_plan(*count, period, *expansion_limit, periodic_body.as_ref().and_then(|(_, profile)| profile.as_ref()).expect("admitted periodic profile"))? {
                                    if plan.duration() < best.duration() { best = plan; }
                                    // Subsequent periods strictly increase completion.
                                    *next_period = witness.completion;
                                    break;
                                }
                            }
                            Some(best)
                        }
                        else {
                            let duration = witness.completion.checked_mul(*count).ok_or("repeated refinement completion overflow")?;
                            Some(Arc::new(Plan::Repeat { body: Arc::new(Plan::Selected(Arc::new(witness))), count: *count, concurrent: 1, duration }))
                        }
                    }
                    Outcome::Incomplete { lower_bound } => {
                        let refined = if *order == Order::Parallel { self.model.parallel_lower(&body.model.root, *count, lower_bound)? }
                            else { lower_bound.checked_mul(*count).ok_or("repeated refinement bound overflow")? };
                        self.outcome = Outcome::Incomplete { lower_bound: self.model.lower_bound()?.max(refined) };
                        None
                    }
                    Outcome::Infeasible => { self.outcome = Outcome::Infeasible; self.state = State::Settled; return Ok(self.outcome.clone()); }
                }
            }
            State::Scope(body) => match body.advance(budget)? {
                Outcome::Feasible(witness) => Some(Arc::new(Plan::Scope { duration: witness.completion,
                    body: Arc::new(Plan::Selected(Arc::new(witness))) })),
                Outcome::Incomplete { lower_bound } => {
                    self.outcome = Outcome::Incomplete { lower_bound: self.model.lower_bound()?.max(lower_bound) };
                    None
                }
                Outcome::Infeasible => { self.outcome = Outcome::Infeasible; self.state = State::Settled; return Ok(self.outcome.clone()); }
            },
        };
        if let Some(plan) = plan {
            let witness = self.model.witness(plan)?;
            if !matches!(&self.outcome, Outcome::Feasible(previous) if previous.completion < witness.completion) {
                if witness.is_optimal() { self.state = State::Settled; }
                self.outcome = Outcome::Feasible(witness);
            }
        }
        Ok(self.outcome.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model(count: u64) -> Structured {
        let operation = Arc::new(Node::Operation(Operation { name: "pipelined service".into(),
            predecessors: vec![], start_predecessors: vec![], latency: 3,
            reservations: vec![Reservation { resource: 0, offset: 0, duration: 1, units: 1 }] }));
        Structured { relationship: crate::authority::ModelRelationship::hypothetical_execution(),
            identity: "serial boundaries around overlapping work".into(),
            timebase: Timebase { seconds_numerator: 1, seconds_denominator: 1 },
            resources: vec![Resource { name: "service".into(), capacity: 1, unit: CapacityUnit::Slots }],
            root: Arc::new(Node::Repeat { order: Order::Serial, count,
                body: Arc::new(Node::Repeat { order: Order::Parallel, count: 2, body: operation }) }), unmapped: vec![] }
    }
    fn complete(search: &mut Refinement) -> Witness {
        for _ in 0..1000 {
            if let Outcome::Feasible(witness) = search.advance(1).unwrap() {
                if witness.is_optimal() { witness.check_execution_upper().unwrap(); return witness; }
            }
        }
        panic!("retained subproblem did not complete")
    }
    #[test]
    fn trillion_serial_bodies_refine_one_shared_parallel_frontier() {
        let model = model(1_000_000_000_000);
        let mut search = Refinement::new(model, 2).unwrap();
        let Outcome::Feasible(initial) = search.advance(0).unwrap() else { panic!("initial compact witness") };
        assert_eq!((initial.lower_bound(), initial.completion()), (4_000_000_000_000, 6_000_000_000_000));
        let selected = complete(&mut search);
        assert_eq!(selected.completion(), 4_000_000_000_000);
        assert!(matches!(selected.expand(100), Err(crate::workload::DerivationError::Exhausted(_))));
    }
    #[test]
    fn refined_repeated_and_serial_witnesses_match_flat_oracle() {
        let mut model = model(2);
        model.root = Arc::new(Node::Compose { order: Order::Serial, children: vec![model.root.clone(), model.root.clone()] });
        let reference = model.expand(8).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        let (_, schedule) = selected.expand(8).unwrap();
        assert_eq!(schedule.completion, reference.schedule().completion);
        assert_eq!(schedule.completion, 16);
    }
    #[test]
    fn periodic_refinement_completes_without_expansion_but_missing_mappings_stay_unresolved() {
        let mut model = model(1_000_000);
        let mut search = Refinement::new(model.clone(), 1).unwrap();
        let Outcome::Feasible(witness) = search.advance(100_000).unwrap() else { panic!("compact witness") };
        assert!(witness.is_optimal());
        model.unmapped.push("unqualified service".into());
        let mut search = Refinement::new(model, 2).unwrap();
        assert!(matches!(search.advance(100_000).unwrap(), Outcome::Incomplete { .. }));
    }
    #[test]
    fn impossible_repeated_body_is_infeasible_but_zero_visits_are_empty() {
        let mut model = model(2);
        let operation = Arc::new(Node::Operation(Operation { name: "oversized service".into(),
            predecessors: vec![], start_predecessors: vec![], latency: 1,
            reservations: vec![Reservation { resource: 0, offset: 0, duration: 1, units: 2 }] }));
        model.root = Arc::new(Node::Repeat { order: Order::Serial, count: 2, body: operation.clone() });
        let mut search = Refinement::new(model.clone(), 1).unwrap();
        assert!(matches!(search.advance(100_000).unwrap(), Outcome::Infeasible));
        model.root = Arc::new(Node::Repeat { order: Order::Serial, count: 0, body: operation });
        assert_eq!(complete(&mut Refinement::new(model, 1).unwrap()).completion(), 0);
    }
    #[test]
    fn nested_residency_refines_shared_body_using_residual_capacity() {
        for count in [2, 1_000_000_000_000] {
            let mut model = model(count);
            model.resources[0].capacity = 3;
            // A singleton parallel wrapper has no independent interleavings;
            // it must not conceal the serial repetition from refinement.
            model.root = Arc::new(Node::Compose { order: Order::Parallel, children: vec![model.root] });
            for _ in 0..2 {
                model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
            }
            let selected = complete(&mut Refinement::new(model.clone(), 2).unwrap());
            assert_eq!(selected.completion(), count * 4);
            if count == 2 {
                let reference = model.expand(8).unwrap().solve(100_000).unwrap();
                assert!(reference.is_optimal());
                let (_, schedule) = selected.expand(8).unwrap();
                assert_eq!(schedule.completion, reference.schedule().completion);
            }
        }
    }
    #[test]
    fn fully_held_unused_resource_does_not_block_refinement() {
        let mut model = model(1_000_000_000_000);
        model.resources.push(Resource { name: "resident group".into(), capacity: 1, unit: CapacityUnit::Slots });
        model.root = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: model.root });
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        assert_eq!(selected.completion(), 4_000_000_000_000);
    }
    #[test]
    fn fully_held_used_resource_retains_exact_scope_feasibility() {
        let mut model = model(1);
        model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
        assert!(model.scope_body().unwrap().is_none());
        let mut unresolved = Refinement::new(model.clone(), 2).unwrap();
        assert!(matches!(unresolved.advance(1000).unwrap(), Outcome::Incomplete { .. }));
        let mut exact = Refinement::new(model, 4).unwrap();
        assert!(matches!(exact.advance(100_000).unwrap(), Outcome::Infeasible));
    }
    #[test]
    fn scoped_witness_cannot_reclaim_held_capacity() {
        let mut model = model(2);
        model.resources[0].capacity = 2;
        model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
        let selected = complete(&mut Refinement::new(model.clone(), 2).unwrap());
        let Plan::Scope { body, duration } = selected.plan.as_ref() else { panic!("scope") };
        let Plan::Selected(child) = body.as_ref() else { panic!("selected body") };
        let mut invalid = child.as_ref().clone();
        Arc::make_mut(&mut invalid.model).resources[0].capacity += 1;
        let plan = Arc::new(Plan::Scope { duration: *duration, body: Arc::new(Plan::Selected(Arc::new(invalid))) });
        assert!(model.witness(plan).is_err());
    }
    fn repeated_groups(count: u64, resident: u64) -> Structured {
        let mut model = model(1);
        model.resources.push(Resource { name: "resident groups".into(), capacity: resident, unit: CapacityUnit::Slots });
        let body = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: model.root });
        model.root = Arc::new(Node::Repeat { order: Order::Parallel, count, body });
        model
    }
    #[test]
    fn huge_parallel_groups_reuse_refined_internal_schedules() {
        let model = repeated_groups(1_000_000_000_000, 1);
        let mut search = Refinement::new(model, 2).unwrap();
        let Outcome::Feasible(initial) = search.advance(0).unwrap() else { panic!("initial group plan") };
        assert_eq!(initial.completion(), 6_000_000_000_000);
        let selected = complete(&mut search);
        assert_eq!(selected.completion(), 4_000_000_000_000);
        assert_eq!(selected.peak().unwrap(), vec![1, 1]);
    }
    #[test]
    fn parallel_refinement_preserves_ancestor_and_group_lifetimes() {
        let mut model = repeated_groups(2, 2);
        model.root = Arc::new(Node::Scope { reservations: vec![(1, 1)], body: model.root });
        let reference = model.expand(10).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        assert_eq!(selected.peak().unwrap(), vec![1, 2]);
        assert_eq!(selected.completion(), reference.schedule().completion);
        assert_eq!(selected.expand(10).unwrap().1.completion, 8);
    }
    #[test]
    fn retained_resident_profiles_close_overlapping_group_schedule() {
        let model = repeated_groups(2, 2);
        let reference = model.expand(8).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let mut search = Refinement::new(model, 2).unwrap();
        let Outcome::Feasible(selected) = search.advance(1000).unwrap() else { panic!("refined waves") };
        selected.check_execution_upper().unwrap();
        selected.expand(8).unwrap();
        assert!(selected.is_optimal());
        assert_eq!(selected.completion(), reference.schedule().completion);
        assert_eq!(selected.completion(), 6);
    }
    fn heterogeneous(count: u64, shared: bool) -> Structured {
        let mut model = model(count);
        model.resources.push(Resource { name: "other service".into(), capacity: 1, unit: CapacityUnit::Slots });
        let operation = Arc::new(Node::Operation(Operation { name: "other pipeline".into(),
            predecessors: vec![], start_predecessors: vec![], latency: 3,
            reservations: vec![Reservation { resource: if shared { 0 } else { 1 }, offset: 0, duration: 1, units: 1 }] }));
        let other = Arc::new(Node::Repeat { order: Order::Serial, count: count + 1,
            body: Arc::new(Node::Repeat { order: Order::Parallel, count: 2, body: operation }) });
        model.root = Arc::new(Node::Compose { order: Order::Parallel, children: vec![model.root, other] });
        model
    }
    #[test]
    fn huge_heterogeneous_parallel_children_refine_without_flattening() {
        let model = heterogeneous(1_000_000_000_000, false);
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        assert_eq!(selected.completion(), 4_000_000_000_004);
        assert_eq!(selected.peak().unwrap(), vec![1, 1]);
    }
    #[test]
    fn heterogeneous_parallel_refinement_matches_flat_oracle() {
        let model = heterogeneous(1, false);
        let reference = model.expand(6).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        assert_eq!(selected.completion(), reference.schedule().completion);
        selected.expand(6).unwrap();
    }
    #[test]
    fn heterogeneous_parallel_children_preserve_ancestor_residency() {
        let mut model = heterogeneous(1, false);
        model.resources[0].capacity = 2;
        model.root = Arc::new(Node::Scope { reservations: vec![(0, 1)], body: model.root });
        let reference = model.expand(8).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let selected = complete(&mut Refinement::new(model, 2).unwrap());
        assert_eq!(selected.completion(), reference.schedule().completion);
        assert_eq!(selected.peak().unwrap(), vec![2, 1]);
        selected.expand(8).unwrap();
    }
    #[test]
    fn contended_parallel_children_retain_staggered_overlap() {
        let model = heterogeneous(1, true);
        let reference = model.expand(6).unwrap().solve(100_000).unwrap();
        assert!(reference.is_optimal());
        let mut search = Refinement::new(model, 2).unwrap();
        let Outcome::Feasible(selected) = search.advance(100_000).unwrap() else { panic!("serial upper") };
        selected.check_execution_upper().unwrap();
        selected.expand(6).unwrap();
        assert!(selected.is_optimal());
        assert_eq!(selected.completion(), reference.schedule().completion);
        let Plan::Offset { children, starts, .. } = selected.plan.as_ref() else { panic!("staggered upper") };
        assert_ne!(starts[0], starts[1]);
        let duration = children.iter().map(|p| p.duration()).max().unwrap();
        assert!(selected.model.witness(Arc::new(Plan::Parallel { children: children.clone(), duration })).is_err());
    }
}
