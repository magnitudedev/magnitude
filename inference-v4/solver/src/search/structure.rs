//! Heuristic relationships inferred from the existing typed model. These never
//! remove factors or authorize proofs of independence.
use crate::{
    model::{Arithmetic, Constraint, Cost, Factor, FactorKind},
    scheduling::{Activity, Demand, SchedulingConstraint},
    Model,
};

pub(super) struct Structure {
    pub scopes: Vec<Vec<usize>>,
    pub incident: Vec<Vec<usize>>,
    pub dependents: Vec<Vec<usize>>,
    pub roots: Vec<usize>,
    inputs: Vec<Vec<usize>>,
    pub guards: Vec<usize>,
    factor_paths: Vec<Vec<usize>>,
    retained_bytes: usize,
}
impl Structure {
    pub fn new(model: &Model) -> Self {
        let n = model.variables().len();
        let mut s = Self {
            scopes: vec![],
            incident: vec![vec![]; n],
            dependents: vec![vec![]; n],
            roots: vec![],
            inputs: vec![vec![]; n],
            guards: vec![],
            factor_paths: vec![],
            retained_bytes: 0,
        };
        let mut derived = vec![false; n];
        let mut stack: Vec<(&Factor, Vec<usize>, Vec<usize>)> = model
            .factors()
            .iter()
            .enumerate()
            .map(|(i, f)| (f, vec![], vec![i]))
            .collect();
        while let Some((factor, mut guards, path)) = stack.pop() {
            guards.extend(factor.guards.iter().map(|g| g.variable.0));
            guards.sort_unstable();
            guards.dedup();
            s.guards.extend(&guards);
            if let FactorKind::Fragment { factors } = &factor.kind {
                stack.extend(factors.iter().enumerate().map(|(i, f)| {
                    let mut child = path.clone();
                    child.push(i);
                    (f, guards.clone(), child)
                }));
                continue;
            }
            let mut scope: Vec<_> = factor.scope().into_iter().map(|v| v.0).collect();
            scope.extend(&guards);
            scope.sort_unstable();
            scope.dedup();
            let id = s.scopes.len();
            for &v in &scope {
                s.incident[v].push(id);
            }
            for &g in &guards {
                s.dependents[g].extend(scope.iter().copied().filter(|v| *v != g));
            }
            s.scopes.push(scope);
            s.factor_paths.push(path);
            if let FactorKind::Constraint(c) = &factor.kind {
                match c {
                    // Table columns and equality endpoints can encode mode-to-
                    // resource mappings without an arithmetic output. Connect
                    // their priority graph through a hub, without making them
                    // mandatory release dependencies or constructing a clique.
                    Constraint::Table { variables, .. } => {
                        if let Some(first) = variables.first() {
                            for other in variables.iter().skip(1) {
                                s.inputs[first.0].push(other.0);
                                s.inputs[other.0].push(first.0);
                            }
                        }
                    }
                    Constraint::Equal { left, right } => {
                        s.inputs[left.0].push(right.0);
                        s.inputs[right.0].push(left.0);
                    }
                    Constraint::Arithmetic(a) => match a {
                        Arithmetic::Product {
                            left,
                            right,
                            product,
                        } => s.derive(&[left.0, right.0], &[product.0], &mut derived),
                        Arithmetic::CeilDiv {
                            numerator,
                            denominator,
                            quotient,
                        } => s.derive(&[numerator.0, denominator.0], &[quotient.0], &mut derived),
                        Arithmetic::DivRem {
                            numerator,
                            denominator,
                            quotient,
                            remainder,
                        } => s.derive(
                            &[numerator.0, denominator.0],
                            &[quotient.0, remainder.0],
                            &mut derived,
                        ),
                        Arithmetic::Minimum {
                            left,
                            right,
                            result,
                        }
                        | Arithmetic::Maximum {
                            left,
                            right,
                            result,
                        } => s.derive(&[left.0, right.0], &[result.0], &mut derived),
                    },
                    Constraint::BoolAnd { output, inputs } => s.derive(
                        &inputs.iter().map(|v| v.0).collect::<Vec<_>>(),
                        &[output.0],
                        &mut derived,
                    ),
                    Constraint::InactiveValue {
                        active, variable, ..
                    } => s.dependents[active.variable.0].push(variable.0),
                    Constraint::Schedule(SchedulingConstraint::Activity(a)) => {
                        s.activity(a, &mut derived)
                    }
                    Constraint::Schedule(SchedulingConstraint::Precedence {
                        before,
                        after,
                        ..
                    }) => {
                        s.dependents[before.time.0].push(after.time.0);
                        for p in [before.presence, after.presence].into_iter().flatten() {
                            s.dependents[p.0].extend([before.time.0, after.time.0]);
                        }
                    }
                    Constraint::Schedule(SchedulingConstraint::ActivityCumulative {
                        completion,
                        activities,
                        ..
                    }) => {
                        for a in activities {
                            s.activity(&a.activity, &mut derived);
                            // Completion is a free upper bound rather than an
                            // equality output; releasing it permits improvement.
                            s.dependents[a.activity.end.0].push(completion.0);
                        }
                    }
                    _ => (),
                }
            }
        }
        for d in &mut s.dependents {
            d.sort_unstable();
            d.dedup();
        }
        for (input, outputs) in s.dependents.iter().enumerate() {
            for &output in outputs {
                s.inputs[output].push(input);
            }
        }
        s.guards.sort_unstable();
        s.guards.dedup();
        s.roots = (0..n)
            .filter(|&v| !derived[v] && model.variables()[v].domain.cardinality() > 1)
            .collect();
        // Every free variable must be reachable from at least one selectable
        // root. Cycles and definitions under a fixed inactive guard otherwise
        // disappear even if unrelated roots exist elsewhere in the model.
        let mut reachable = vec![false; n];
        let mut queue = s.roots.clone();
        for &v in &queue {
            reachable[v] = true;
        }
        s.close(&mut reachable, &mut queue);
        s.roots.extend(
            (0..n).filter(|&v| !reachable[v] && model.variables()[v].domain.cardinality() > 1),
        );
        s.roots.sort_unstable();
        s.retained_bytes = s.storage_bytes();
        s
    }
    fn derive(&mut self, inputs: &[usize], outputs: &[usize], derived: &mut [bool]) {
        for &v in outputs {
            derived[v] = true;
        }
        for &v in inputs {
            self.dependents[v].extend(outputs.iter().copied().filter(|o| *o != v));
        }
    }
    fn activity(&mut self, a: &Activity, derived: &mut [bool]) {
        self.derive(&[a.start.0, a.duration.0], &[a.end.0], derived);
        // A duration change can invalidate downstream timing even if no start
        // variable was selected as a primary move.
        self.dependents[a.duration.0].push(a.start.0);
        if let Some(p) = a.presence {
            self.dependents[p.0].extend([a.start.0, a.duration.0, a.end.0]);
        }
    }
    pub fn close(&self, selected: &mut [bool], queue: &mut Vec<usize>) {
        let mut at = 0;
        while at < queue.len() {
            for &v in &self.dependents[queue[at]] {
                if !selected[v] {
                    selected[v] = true;
                    queue.push(v);
                }
            }
            at += 1;
        }
    }
    /// Incumbent-sensitive priorities, never proof information. Tight resource
    /// participants and objective carriers transfer their priority backwards to
    /// the choices that define them. A max-heap handles cyclic definitions once
    /// per priority improvement rather than repeatedly scanning the graph.
    pub fn priorities(&self, model: &Model, values: &[i64]) -> Vec<u64> {
        let mut scores = vec![0; values.len()];
        let mut stack: Vec<_> = model.factors().iter().collect();
        while let Some(factor) = stack.pop() {
            if factor
                .guards
                .iter()
                .any(|g| values[g.variable.0] != g.value)
            {
                continue;
            }
            match &factor.kind {
                FactorKind::Fragment { factors } => stack.extend(factors),
                FactorKind::Cost(cost) => {
                    for v in cost.scope() {
                        scores[v.0] = scores[v.0].max(1);
                    }
                }
                FactorKind::Constraint(Constraint::LinearLe { terms, rhs }) => {
                    let total = terms.iter().try_fold(0i128, |sum, t| {
                        sum.checked_add(
                            i128::from(t.coefficient) * i128::from(values[t.variable.0]),
                        )
                    });
                    if total == Some(*rhs) {
                        for t in terms {
                            scores[t.variable.0] = scores[t.variable.0].max(4);
                        }
                    }
                }
                FactorKind::Constraint(Constraint::Schedule(
                    SchedulingConstraint::ActivityCumulative {
                        capacity,
                        activities,
                        ..
                    },
                )) => {
                    let active: Vec<_> = activities
                        .iter()
                        .filter(|a| a.activity.presence.is_none_or(|p| values[p.0] != 0))
                        .collect();
                    let finish = active.iter().map(|a| values[a.activity.end.0]).max();
                    for a in active {
                        let demand = match a.demand {
                            Demand::Constant(d) => d,
                            Demand::Variable(v) => u64::try_from(values[v.0]).unwrap_or(0),
                        };
                        let pressure = if *capacity == 0 {
                            0
                        } else {
                            (u128::from(demand) * 8 / u128::from(*capacity)).min(8) as u64
                        };
                        let score =
                            1 + pressure + 8 * u64::from(Some(values[a.activity.end.0]) == finish);
                        for v in [a.activity.start, a.activity.duration, a.activity.end] {
                            scores[v.0] = scores[v.0].max(score);
                        }
                        if let Demand::Variable(v) = a.demand {
                            scores[v.0] = scores[v.0].max(score);
                        }
                    }
                }
                FactorKind::Constraint(Constraint::Schedule(
                    SchedulingConstraint::Cumulative {
                        capacity,
                        reservations,
                    },
                )) => {
                    for r in reservations {
                        if r.interval.presence.iter().any(|p| values[p.0] == 0) {
                            continue;
                        }
                        let demand = match r.demand {
                            Demand::Constant(d) => d,
                            Demand::Variable(v) => u64::try_from(values[v.0]).unwrap_or(0),
                        };
                        let score = if *capacity == 0 {
                            1
                        } else {
                            1 + (u128::from(demand) * 8 / u128::from(*capacity)).min(8) as u64
                        };
                        for v in [r.interval.start, r.interval.end] {
                            scores[v.0] = scores[v.0].max(score);
                        }
                        if let Demand::Variable(v) = r.demand {
                            scores[v.0] = scores[v.0].max(score);
                        }
                    }
                }
                _ => (),
            }
        }
        let mut queue: std::collections::BinaryHeap<_> = scores
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, score)| *score > 0)
            .map(|(v, score)| (score, v))
            .collect();
        while let Some((score, v)) = queue.pop() {
            if scores[v] != score {
                continue;
            }
            for &input in &self.inputs[v] {
                if scores[input] < score {
                    scores[input] = score;
                    queue.push((score, input));
                }
            }
        }
        scores
    }

    /// Values at active constraint transitions. These guide branch order only;
    /// every unproposed value remains legal and available to the repair engine.
    pub fn proposals(&self, model: &Model, values: &[i64], variable: usize) -> Vec<i64> {
        let domain = &model.variables()[variable].domain;
        // Binary choices have no interior numeric transition to discover.
        if domain.cardinality() <= 2 {
            return Vec::new();
        }
        let mut proposed = Vec::new();
        let mut add = |value: i128| {
            for delta in [-1i128, 0, 1] {
                if let Some(value) = value.checked_add(delta).and_then(|v| i64::try_from(v).ok()) {
                    if value != values[variable] && domain.contains(value) {
                        proposed.push(value);
                    }
                }
            }
        };
        // Only incident leaves can propose a value for this variable. Follow
        // their paths to preserve all enclosing fragment guards.
        for &id in &self.incident[variable] {
            let mut factors = model.factors();
            let mut leaf = None;
            for &index in &self.factor_paths[id] {
                let factor = &factors[index];
                if factor
                    .guards
                    .iter()
                    .any(|g| values[g.variable.0] != g.value)
                {
                    leaf = None;
                    break;
                }
                leaf = Some(factor);
                if let FactorKind::Fragment { factors: children } = &factor.kind {
                    factors = children;
                }
            }
            let Some(factor) = leaf else { continue };
            match &factor.kind {
                FactorKind::Constraint(Constraint::LinearLe { terms, rhs }) => {
                    let mut coefficient = 0i128;
                    let mut residual = Some(*rhs);
                    for t in terms {
                        if t.variable.0 == variable {
                            coefficient += i128::from(t.coefficient);
                        } else {
                            residual = residual.and_then(|r| {
                                r.checked_sub(
                                    i128::from(t.coefficient) * i128::from(values[t.variable.0]),
                                )
                            });
                        }
                    }
                    if coefficient != 0 {
                        if let Some(boundary) = residual.and_then(|r| r.checked_div(coefficient)) {
                            add(boundary);
                        }
                    }
                }
                FactorKind::Constraint(Constraint::Arithmetic(Arithmetic::CeilDiv {
                    numerator,
                    denominator,
                    quotient,
                })) => {
                    let n = i128::from(values[numerator.0]);
                    let d = i128::from(values[denominator.0]);
                    let q = i128::from(values[quotient.0]);
                    if denominator.0 == variable && q > 0 {
                        add(n / q);
                        if q > 1 {
                            add(n / (q - 1));
                        }
                    }
                    if numerator.0 == variable {
                        add(q * d);
                        add((q - 1) * d);
                    }
                }
                FactorKind::Constraint(Constraint::Schedule(
                    SchedulingConstraint::Precedence { before, after, lag },
                )) => {
                    if before.presence.is_some_and(|p| values[p.0] == 0)
                        || after.presence.is_some_and(|p| values[p.0] == 0)
                    {
                        continue;
                    }
                    if after.time.0 == variable {
                        add(i128::from(values[before.time.0]) + i128::from(*lag));
                    }
                    if before.time.0 == variable {
                        add(i128::from(values[after.time.0]) - i128::from(*lag));
                    }
                }
                FactorKind::Constraint(Constraint::Schedule(
                    SchedulingConstraint::ActivityCumulative { activities, .. },
                )) => {
                    if activities.iter().any(|a| a.activity.start.0 == variable) {
                        for a in activities {
                            if a.activity.presence.is_none_or(|p| values[p.0] != 0) {
                                add(i128::from(values[a.activity.end.0]));
                            }
                        }
                    }
                }
                FactorKind::Cost(Cost::Linear { terms, .. }) => {
                    for t in terms.iter().filter(|t| t.variable.0 == variable) {
                        if let Some(v) = if t.coefficient > 0 {
                            domain.min()
                        } else {
                            domain.max()
                        } {
                            add(i128::from(v));
                        }
                    }
                }
                _ => (),
            }
        }
        proposed.sort_unstable();
        proposed.dedup();
        proposed
    }
    pub fn bytes(&self) -> usize {
        self.retained_bytes
    }
    fn storage_bytes(&self) -> usize {
        [
            &self.scopes,
            &self.incident,
            &self.dependents,
            &self.inputs,
            &self.factor_paths,
        ]
        .iter()
        .map(|x| {
            x.capacity() * std::mem::size_of::<Vec<usize>>()
                + x.iter()
                    .map(|v| v.capacity() * std::mem::size_of::<usize>())
                    .sum::<usize>()
        })
        .sum::<usize>()
            + (self.roots.capacity() + self.guards.capacity()) * std::mem::size_of::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Literal, Domain, ModelBuilder};

    #[test]
    fn active_resource_priorities_reach_mode_choices_but_ignore_absent_activities() {
        use crate::scheduling::ActivityReservation;
        let mut b = ModelBuilder::new();
        let mode = b.variable("mode", Domain::boolean());
        let start = b.variable("start", Domain::interval(0, 20).unwrap());
        let duration = b.variable("duration", Domain::interval(1, 10).unwrap());
        let end = b.variable("end", Domain::interval(1, 30).unwrap());
        let completion = b.variable("completion", Domain::interval(1, 30).unwrap());
        let present = b.variable("present", Domain::boolean());
        let unrelated = b.variable("unrelated", Domain::boolean());
        b.constraint(Constraint::Table {
            variables: vec![mode, duration],
            tuples: vec![vec![0, 10], vec![1, 5]],
        });
        b.constraint(Constraint::Schedule(
            SchedulingConstraint::ActivityCumulative {
                completion,
                capacity: 4,
                activities: vec![ActivityReservation {
                    activity: Activity {
                        start,
                        duration,
                        end,
                        presence: Some(present),
                    },
                    demand: Demand::Constant(4),
                }],
            },
        ));
        let model = b.build().unwrap();
        let structure = Structure::new(&model);
        let active = structure.priorities(&model, &[0, 0, 10, 10, 10, 1, 0]);
        assert!(active[mode.0] >= 8);
        assert_eq!(active[unrelated.0], 0);
        let absent = structure.priorities(&model, &[0, 0, 10, 10, 10, 0, 0]);
        assert_eq!(absent[mode.0], 0);
    }

    #[test]
    fn proposals_follow_capacity_and_division_transitions_without_expanding_domains() {
        use crate::model::LinearTerm;
        let mut b = ModelBuilder::new();
        let tile = b.variable("tile", Domain::interval(1, 100).unwrap());
        let other = b.variable("other", Domain::singleton(20));
        let size = b.variable("size", Domain::singleton(100));
        let count = b.variable("count", Domain::interval(1, 100).unwrap());
        b.constraint(Constraint::LinearLe {
            terms: vec![LinearTerm::new(tile, 2), LinearTerm::new(other, 1)],
            rhs: 100,
        });
        b.constraint(Constraint::Arithmetic(Arithmetic::CeilDiv {
            numerator: size,
            denominator: tile,
            quotient: count,
        }));
        let model = b.build().unwrap();
        let structure = Structure::new(&model);
        let proposals = structure.proposals(&model, &[25, 20, 100, 4], tile.0);
        assert!(proposals.contains(&40)); // remaining capacity / per-unit use
        assert!(proposals.contains(&34)); // ceil(100 / 3), next repetition count
        assert!(proposals.iter().all(|v| (1..=100).contains(v) && *v != 25));
        assert_eq!(model.variables()[tile.0].domain.cardinality(), 100);
    }

    #[test]
    fn inactive_constraints_do_not_contribute_numeric_proposals() {
        use crate::model::LinearTerm;
        let mut b = ModelBuilder::new();
        let x = b.variable("x", Domain::interval(0, 100).unwrap());
        let active = b.variable("active", Domain::boolean());
        b.guarded_constraint(
            vec![Literal::new(active, 1)],
            Constraint::LinearLe {
                terms: vec![LinearTerm::new(x, 1)],
                rhs: 77,
            },
        );
        let model = b.build().unwrap();
        let structure = Structure::new(&model);
        assert!(structure.proposals(&model, &[0, 0], x.0).is_empty());
        assert!(structure.proposals(&model, &[0, 1], x.0).contains(&77));
    }

    #[test]
    fn disconnected_cycles_and_inactive_definitions_remain_selectable() {
        for guarded in [false, true] {
            let mut b = ModelBuilder::new();
            let x = b.variable("x", Domain::boolean());
            let y = b.variable("y", Domain::boolean());
            let one = b.variable("one", Domain::singleton(1));
            let z = b.variable("unrelated", Domain::boolean());
            if guarded {
                let guard = b.variable("inactive", Domain::singleton(0));
                b.guarded_constraint(
                    vec![Literal::new(guard, 1)],
                    Constraint::Arithmetic(Arithmetic::Product {
                        left: one,
                        right: one,
                        product: x,
                    }),
                );
            } else {
                b.constraint(Constraint::Arithmetic(Arithmetic::Product {
                    left: y,
                    right: one,
                    product: x,
                }));
                b.constraint(Constraint::Arithmetic(Arithmetic::Product {
                    left: x,
                    right: one,
                    product: y,
                }));
            }
            let s = Structure::new(&b.build().unwrap());
            assert!(s.roots.contains(&z.0));
            assert!(s.roots.contains(&x.0));
            assert!(s.roots.contains(&y.0));
        }
    }
}
