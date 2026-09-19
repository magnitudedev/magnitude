use super::{
    knowledge::{self, Knowledge, SharedKnowledge},
    Limits, Options, Policy, Stats,
};
use crate::{
    decomposition,
    model::{Factor, FactorKind, Model, Obligation, VarId},
    propagation,
    result::{FeasibleSolution, Progress, StopReason},
    Domain, Error, Result,
};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Instant,
};

/// Terminal claims apply to this engine's initial domain context. Only the
/// public exact dispatcher may turn an unrestricted result into `Solution`.
pub(super) enum Outcome {
    Optimal(FeasibleSolution),
    Infeasible,
    Incomplete(Progress),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    factors: Vec<usize>,
    context: Vec<(VarId, Domain)>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Witness {
    pub cost: u64,
    pub values: Vec<(usize, i64)>,
}
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub(super) struct Summary {
    pub lower: u64,
    pub witness: Option<Witness>,
    pub solved: bool,
    pub infeasible: bool,
}
struct Pending {
    factors: Vec<usize>,
    domains: Vec<(VarId, Domain)>,
    cursor: usize,
    changed: bool,
    analyzing: bool,
    residual: Vec<usize>,
    offset: u64,
    lower: u64,
    term_bounds: HashMap<usize, u64>,
    early_lower: u64,
}
enum State {
    Pending(Pending),
    Or(Vec<usize>),
    And {
        offset: u64,
        values: Vec<(usize, i64)>,
        children: Vec<usize>,
    },
    Done,
    Blocked(Vec<Obligation>),
}
struct Node {
    reuse_checked: bool,
    published: bool,
    key: Key,
    state: State,
    summary: Summary,
    parents: Vec<usize>,
}

/// Search state is bound to one immutable validated model. Its memo contains
/// unconditional residual results; parent pruning never changes a shared child.
pub struct Search {
    model: Arc<Model>,
    options: Options,
    nodes: Vec<Node>,
    memo: HashMap<Key, usize>,
    stats: Stats,
    incumbent: Option<FeasibleSolution>,
    root: usize,
    // One dense evaluation workspace; each pending region retains only its scope.
    domains: Vec<Domain>,
    active: Option<usize>,
    factors: Arc<Vec<Factor>>,
    knowledge: SharedKnowledge,
    expanded: HashMap<usize, Vec<usize>>,
    failure: Option<Error>,
    // Context-local defaults also preserve assumptions on variables with no factors.
    defaults: Vec<i64>,
    preferred: Option<Vec<i64>>,
    reported_cost: Option<u64>,
    advance_peak_retained_bytes: usize,
}
impl Search {
    pub fn new(model: Arc<Model>, options: Options) -> Result<Self> {
        model.validate()?;
        let domains = model.domains();
        Self::with_context(model, options, domains, None, None)
    }
    /// Internal entry point. The caller owns the proof scope and has validated
    /// that these domains restrict this immutable model. Never expose the
    /// resulting terminal status as a proof about an unrestricted model.
    pub(super) fn with_context(
        model: Arc<Model>,
        options: Options,
        domains: Vec<Domain>,
        preferred: Option<Vec<i64>>,
        incumbent: Option<FeasibleSolution>,
    ) -> Result<Self> {
        let knowledge = Knowledge::shared(model.clone());
        Self::with_knowledge(model, options, domains, preferred, incumbent, knowledge)
    }
    pub(super) fn with_knowledge(
        model: Arc<Model>,
        options: Options,
        domains: Vec<Domain>,
        preferred: Option<Vec<i64>>,
        incumbent: Option<FeasibleSolution>,
        knowledge: SharedKnowledge,
    ) -> Result<Self> {
        if !knowledge
            .lock()
            .expect("solver knowledge lock poisoned")
            .belongs_to(&model)
        {
            return Err(Error::Invariant(
                "shared knowledge belongs to another model".into(),
            ));
        }
        let defaults = domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                preferred
                    .as_ref()
                    .and_then(|p| p.get(i))
                    .copied()
                    .filter(|v| d.contains(*v))
                    .or_else(|| d.min())
                    .unwrap_or(0)
            })
            .collect();
        let factors = knowledge
            .lock()
            .expect("solver knowledge lock poisoned")
            .factors
            .clone();
        let reported_cost = incumbent.as_ref().map(FeasibleSolution::cost);
        let mut search = Self {
            factors,
            knowledge,
            expanded: HashMap::new(),
            failure: None,
            domains,
            active: None,
            model,
            options,
            nodes: Vec::new(),
            memo: HashMap::new(),
            stats: Stats::default(),
            incumbent,
            defaults,
            preferred,
            reported_cost,
            advance_peak_retained_bytes: 0,
            root: 0,
        };
        search.root = search.intern((0..search.factors.len()).collect());
        if search.domains.iter().any(Domain::is_empty) {
            search.nodes[search.root].state = State::Done;
            search.nodes[search.root].summary = Summary {
                infeasible: true,
                solved: true,
                ..Default::default()
            };
        }
        search.update_memory();
        Ok(search)
    }
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
    pub(super) fn advance_peak_retained_bytes(&self) -> usize {
        self.advance_peak_retained_bytes
    }
    fn intern(&mut self, mut factors: Vec<usize>) -> usize {
        factors.sort_unstable();
        let scope = decomposition::scope(&self.factors, &factors);
        let key = Key {
            factors: factors.clone(),
            context: scope
                .iter()
                .map(|&v| (v, self.domains[v.0].clone()))
                .collect(),
        };
        self.stats.max_context_variables = self.stats.max_context_variables.max(key.context.len());
        let log_size = key
            .context
            .iter()
            .map(|(_, d)| (d.cardinality().max(1) as f64).log10())
            .sum::<f64>();
        self.stats.max_context_log10 = self.stats.max_context_log10.max(log_size);
        if self.options.memoize {
            if let Some(&id) = self.memo.get(&key) {
                self.stats.memo_hits += 1;
                return id;
            }
        }
        let id = self.nodes.len();
        if self.options.memoize {
            self.memo.insert(key.clone(), id);
        }
        let domains = key.context.clone();
        // Cheap declared costs can exclude a region before constructing its body.
        factors.sort_by_key(|&f| (!matches!(self.factors[f].kind, FactorKind::Cost(_)), f));
        let state = State::Pending(Pending {
            factors,
            domains,
            cursor: 0,
            changed: false,
            analyzing: false,
            residual: Vec::new(),
            offset: 0,
            lower: 0,
            term_bounds: HashMap::new(),
            early_lower: 0,
        });
        self.nodes.push(Node {
            reuse_checked: false,
            published: false,
            key,
            state,
            summary: Summary::default(),
            parents: Vec::new(),
        });
        self.stats.nodes += 1;
        id
    }
    fn activate(&mut self, id: usize) {
        if self.active == Some(id) {
            return;
        }
        if let Some(previous) = self.active {
            if let State::Pending(p) = &mut self.nodes[previous].state {
                for (v, d) in &mut p.domains {
                    d.clone_from(&self.domains[v.0]);
                }
            }
        }
        if let State::Pending(p) = &self.nodes[id].state {
            for (v, d) in &p.domains {
                self.domains[v.0].clone_from(d);
            }
        }
        self.active = Some(id);
    }
    fn link(&mut self, parent: usize, children: &[usize]) {
        for &child in children {
            if !self.nodes[child].parents.contains(&parent) {
                self.nodes[child].parents.push(parent);
            }
        }
    }
    fn add(a: u64, b: u64) -> Result<u64> {
        a.checked_add(b)
            .ok_or_else(|| Error::Overflow("objective sum".into()))
    }
    fn refresh(&mut self, initial: usize) -> Result<()> {
        let mut queue = VecDeque::from([initial]);
        while let Some(id) = queue.pop_front() {
            let mut next = match &self.nodes[id].state {
                State::Or(children) => {
                    let live: Vec<_> = children
                        .iter()
                        .map(|&c| &self.nodes[c].summary)
                        .filter(|s| !s.infeasible)
                        .collect();
                    if live.is_empty() {
                        Summary {
                            infeasible: true,
                            solved: true,
                            ..Default::default()
                        }
                    } else {
                        let witness = live
                            .iter()
                            .filter_map(|s| s.witness.as_ref())
                            .min_by_key(|w| w.cost)
                            .cloned();
                        let lower = live.iter().map(|s| s.lower).min().unwrap();
                        let solved = witness.as_ref().is_some_and(|w| lower >= w.cost);
                        if let Some(w) = &witness {
                            if lower > w.cost {
                                return Err(Error::Invariant("OR bound exceeds witness".into()));
                            }
                        }
                        Summary {
                            lower,
                            witness,
                            solved,
                            infeasible: false,
                        }
                    }
                }
                State::And {
                    offset,
                    values,
                    children,
                } => {
                    if children.iter().any(|&c| self.nodes[c].summary.infeasible) {
                        Summary {
                            infeasible: true,
                            solved: true,
                            ..Default::default()
                        }
                    } else {
                        let mut lower = *offset;
                        let mut cost = *offset;
                        let mut all = true;
                        for &c in children {
                            let s = &self.nodes[c].summary;
                            lower = Self::add(lower, s.lower)?;
                            if let Some(w) = &s.witness {
                                cost = Self::add(cost, w.cost)?;
                            } else {
                                all = false;
                            }
                        }
                        let witness = if all {
                            let mut assignments: BTreeMap<_, _> = values.iter().copied().collect();
                            for &c in children {
                                for &(v, x) in &self.nodes[c]
                                    .summary
                                    .witness
                                    .as_ref()
                                    .expect("all children feasible")
                                    .values
                                {
                                    if assignments.insert(v, x).is_some_and(|old| old != x) {
                                        return Err(Error::Invariant(
                                            "independent contexts disagree on a boundary".into(),
                                        ));
                                    }
                                }
                            }
                            Some(Witness {
                                cost,
                                values: assignments.into_iter().collect(),
                            })
                        } else {
                            None
                        };
                        if all && lower > cost {
                            return Err(Error::Invariant("AND bound exceeds witness".into()));
                        }
                        Summary {
                            lower,
                            witness,
                            solved: all && lower == cost,
                            infeasible: false,
                        }
                    }
                }
                _ => self.nodes[id].summary.clone(),
            };
            // Refinement partitions the same region; new children initially have
            // weaker bounds, but cannot invalidate a bound already proved here.
            if !next.infeasible {
                next.lower = next.lower.max(self.nodes[id].summary.lower);
                if let Some(w) = &next.witness {
                    if next.lower > w.cost {
                        return Err(Error::Invariant("retained bound exceeds witness".into()));
                    }
                    next.solved = next.lower == w.cost;
                }
            }
            let changed = next != self.nodes[id].summary;
            self.nodes[id].summary = next;
            if self.options.memoize && self.nodes[id].summary.solved && !self.nodes[id].published {
                let node = &self.nodes[id];
                let (stored, conflict) = self
                    .knowledge
                    .lock()
                    .expect("solver knowledge lock poisoned")
                    .record(
                        &self.factors,
                        &node.key.factors,
                        &node.key.context,
                        &node.summary,
                    )?;
                self.stats.proof_stores += u64::from(stored);
                self.stats.conflicts_learned += u64::from(conflict);
                self.nodes[id].published = true;
            }
            if changed || id == initial {
                queue.extend(self.nodes[id].parents.iter().copied());
            }
        }
        Ok(())
    }
    fn finish_pending(&mut self, id: usize, p: Pending) -> Result<()> {
        if p.residual.is_empty() {
            let values = self.nodes[id]
                .key
                .context
                .iter()
                .map(|(v, _)| {
                    (
                        v.0,
                        self.domains[v.0].min().expect("nonempty propagated domain"),
                    )
                })
                .collect();
            self.nodes[id].state = State::Done;
            self.nodes[id].summary = Summary {
                lower: p.offset,
                witness: Some(Witness {
                    cost: p.offset,
                    values,
                }),
                solved: true,
                infeasible: false,
            };
            return self.refresh(id);
        }
        let residual_scope = decomposition::scope(&self.factors, &p.residual);
        // Discharged factors may be the only users of a narrowed, non-singleton
        // variable. Retain a valid value before its domain leaves the residual.
        let fixed = self.nodes[id]
            .key
            .context
            .iter()
            .filter_map(|(v, _)| {
                let domain = &self.domains[v.0];
                if domain.is_singleton() || residual_scope.binary_search(v).is_err() {
                    domain.min().map(|value| (v.0, value))
                } else {
                    None
                }
            })
            .collect();
        let groups = if self.options.decompose {
            decomposition::components(&self.factors, &p.residual, &self.domains)
        } else {
            vec![p.residual.clone()]
        };
        // Strip exact factors into a local constant before memoizing the residual.
        // This lets the same remaining component be reused after different prefixes.
        let narrowed = self.nodes[id]
            .key
            .context
            .iter()
            .any(|(v, domain)| *domain != self.domains[v.0]);
        if groups.len() > 1 || p.residual != p.factors || narrowed {
            let children: Vec<_> = groups.into_iter().map(|g| self.intern(g)).collect();
            self.link(id, &children);
            self.stats.decompositions += u64::from(children.len() > 1);
            self.nodes[id].state = State::And {
                offset: p.offset,
                values: fixed,
                children,
            };
            return self.refresh(id);
        }
        if let Some(var) = decomposition::branch_variable(&self.factors, &p.residual, &self.domains)
        {
            let domain = &self.domains[var.0];
            let partition = self
                .preferred
                .as_ref()
                .and_then(|values| values.get(var.0))
                .copied()
                .filter(|v| domain.contains(*v) && !domain.is_singleton())
                .map(|v| (Domain::singleton(v), domain.without(v)))
                .or_else(|| domain.split());
            let (left, right) = partition
                .ok_or_else(|| Error::Invariant("branch variable has no partition".into()))?;
            let mut children = Vec::new();
            for domain in [left, right] {
                self.domains[var.0] = domain;
                children.push(self.intern(p.residual.clone()));
            }
            self.link(id, &children);
            self.nodes[id].state = State::Or(children);
            self.stats.branches += 1;
            self.refresh(id)
        } else {
            let mut reasons = Vec::new();
            for &f in &p.residual {
                if let Some(reason) = self.factors[f].assess_validated(&self.domains)?.unresolved {
                    reasons.push(reason);
                }
            }
            if reasons.is_empty() {
                return Err(Error::Invariant(
                    "fixed residual factors did not resolve".into(),
                ));
            }
            self.nodes[id].state = State::Blocked(reasons);
            self.refresh(id)
        }
    }
    /// Materialize one declared child per work unit. The immutable bundle keeps
    /// every possible dependency visible until its children have all arrived.
    fn expand_fragment(&mut self, id: usize, p: &mut Pending) -> Result<bool> {
        let fid = p.factors[p.cursor];
        let factor = &self.factors[fid];
        let FactorKind::Fragment { factors } = &factor.kind else {
            return Ok(false);
        };
        if factor.guard_state(&self.domains)? != Some(true) {
            return Ok(false);
        }
        let ready = self.expanded.get(&fid).map_or(0, Vec::len);
        if ready < factors.len() {
            let mut child = factors[ready].clone();
            child.guards.extend_from_slice(&factor.guards);
            child.guards.sort_by_key(|g| (g.variable, g.value));
            child.guards.dedup();
            let child_id = self.factors.len();
            Arc::make_mut(&mut self.factors).push(child);
            self.expanded.entry(fid).or_default().push(child_id);
            self.stats.materializations += 1;
            self.early_cost_bound(id, p, child_id)?;
        } else {
            let children = self.expanded.get(&fid).cloned().unwrap_or_default();
            p.factors.splice(p.cursor..=p.cursor, children);
            p.changed = true;
        }
        Ok(true)
    }
    fn early_cost_bound(&mut self, id: usize, p: &mut Pending, fid: usize) -> Result<()> {
        if !matches!(self.factors[fid].kind, FactorKind::Cost(_)) {
            return Ok(());
        }
        let a = self.factors[fid].assess_validated(&self.domains)?;
        self.stats.assessments += 1;
        let lower = if self.options.bounds {
            a.lower_bound
        } else {
            a.exact_cost.unwrap_or(0)
        };
        let old = p.term_bounds.get(&fid).copied().unwrap_or(0);
        if lower > old {
            p.term_bounds.insert(fid, lower);
            p.early_lower = Self::add(p.early_lower, lower - old)?;
            if p.early_lower > self.nodes[id].summary.lower {
                self.nodes[id].summary.lower = p.early_lower;
                self.refresh(id)?;
            }
        }
        Ok(())
    }
    fn step(&mut self, id: usize) -> Result<()> {
        self.activate(id);
        if self.options.memoize && !self.nodes[id].reuse_checked {
            self.nodes[id].reuse_checked = true;
            let node = &self.nodes[id];
            let (proof, conflict) = self
                .knowledge
                .lock()
                .expect("solver knowledge lock poisoned")
                .lookup(&self.factors, &node.key.factors, &node.key.context)?;
            if let Some(proof) = proof {
                self.stats.structural_hits += u64::from(!conflict);
                self.stats.conflict_hits += u64::from(conflict);
                self.nodes[id].summary = proof;
                self.nodes[id].state = State::Done;
                self.nodes[id].published = true;
                return self.refresh(id);
            }
        }
        let state = std::mem::replace(&mut self.nodes[id].state, State::Done);
        let State::Pending(mut p) = state else {
            return Err(Error::Invariant("selected nonpending work".into()));
        };
        if self.nodes[id]
            .key
            .context
            .iter()
            .any(|(v, _)| self.domains[v.0].is_empty())
        {
            self.nodes[id].summary = Summary {
                infeasible: true,
                solved: true,
                ..Default::default()
            };
            return self.refresh(id);
        }
        if !p.analyzing {
            if p.cursor < p.factors.len() {
                if self.expand_fragment(id, &mut p)? {
                    self.nodes[id].state = State::Pending(p);
                    return Ok(());
                }
                let fid = p.factors[p.cursor];
                self.early_cost_bound(id, &mut p, fid)?;
                let result = propagation::propagate_factor_validated(
                    &self.factors[p.factors[p.cursor]],
                    &mut self.domains,
                )?;
                self.stats.propagations += 1;
                if result.infeasible {
                    self.nodes[id].summary = Summary {
                        infeasible: true,
                        solved: true,
                        ..Default::default()
                    };
                    return self.refresh(id);
                }
                p.changed |= result.changed;
                p.cursor += 1;
            } else if p.changed {
                p.cursor = 0;
                p.changed = false;
            } else {
                p.analyzing = true;
                p.cursor = 0;
            }
        } else if p.cursor < p.factors.len() {
            let fid = p.factors[p.cursor];
            let a = self.factors[fid].assess_validated(&self.domains)?;
            self.stats.assessments += 1;
            if a.infeasible {
                self.nodes[id].summary = Summary {
                    infeasible: true,
                    solved: true,
                    ..Default::default()
                };
                return self.refresh(id);
            }
            if let Some(cost) = a.exact_cost {
                p.offset = Self::add(p.offset, cost)?;
            } else {
                p.residual.push(fid);
            }
            p.lower = Self::add(
                p.lower,
                if self.options.bounds {
                    a.lower_bound
                } else {
                    a.exact_cost.unwrap_or(0)
                },
            )?;
            p.cursor += 1;
            if self.nodes[id].summary.lower < p.lower {
                self.nodes[id].summary.lower = p.lower;
                self.refresh(id)?;
            }
        } else {
            return self.finish_pending(id, p);
        }
        self.nodes[id].state = State::Pending(p);
        Ok(())
    }
    fn next(&mut self) -> (Option<usize>, Vec<Obligation>) {
        let mut stack = vec![(self.root, Vec::<bool>::new())];
        let mut candidates = Vec::new();
        let mut blocked = Vec::new();
        let mut visited = HashSet::new();
        while let Some((id, preference_path)) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let node = &self.nodes[id];
            if node.summary.solved || node.summary.infeasible {
                continue;
            }
            match &node.state {
                State::Pending(_) => {
                    if self.options.policy == Policy::DepthFirst {
                        return (Some(id), Vec::new());
                    }
                    candidates.push((preference_path, node.summary.lower, id));
                }
                State::Or(children) => {
                    // Only an actual preferred-value partition introduces a
                    // priority. Once that value is absent, ordinary BestFirst
                    // bound ordering applies within the remaining region.
                    let preferred_partition = self.options.policy == Policy::BestFirst
                        && self.preferred.as_ref().is_some_and(|values| {
                            self.nodes[children[0]]
                                .key
                                .context
                                .iter()
                                .zip(&self.nodes[children[1]].key.context)
                                .any(|((left_var, left), (right_var, right))| {
                                    left_var == right_var
                                        && left != right
                                        && left.contains(values[left_var.0])
                                        && !right.contains(values[left_var.0])
                                })
                        });
                    let upper = node.summary.witness.as_ref().map(|w| w.cost);
                    let mut ordered: Vec<_> = children
                        .iter()
                        .copied()
                        .filter(|&c| {
                            let s = &self.nodes[c].summary;
                            !s.infeasible && !s.solved && upper.is_none_or(|u| s.lower < u)
                        })
                        .collect();
                    // Repair contexts deliberately try preferred assignments
                    // first. Reordering them by cost bounds can explore all
                    // siblings before finishing even the first feasible leaf.
                    if self.preferred.is_none() {
                        ordered.sort_by_key(|&c| (self.nodes[c].summary.lower, c));
                    }
                    stack.extend(ordered.into_iter().rev().map(|c| {
                        let mut path = preference_path.clone();
                        if preferred_partition {
                            path.push(c != children[0]);
                        }
                        (c, path)
                    }));
                }
                State::And { children, .. } => {
                    stack.extend(children.iter().rev().map(|&c| (c, preference_path.clone())))
                }
                State::Blocked(reasons) => blocked.extend_from_slice(reasons),
                _ => {}
            }
        }
        blocked.sort();
        blocked.dedup();
        (candidates.into_iter().min().map(|(_, _, id)| id), blocked)
    }
    fn update_incumbent(&mut self, elapsed: f64) -> Result<()> {
        if let Some(w) = self.nodes[self.root].summary.witness.as_ref() {
            if self
                .incumbent
                .as_ref()
                .is_none_or(|old| w.cost < old.cost())
            {
                let w = w.clone();
                let mut values = self.defaults.clone();
                for (v, x) in w.values {
                    values[v] = x;
                }
                let assessment = self.model.validate_assignment(&values)?;
                if assessment.infeasible
                    || assessment.unresolved.is_some()
                    || assessment.exact_cost != Some(w.cost)
                {
                    return Err(Error::Invariant(
                        "root witness disagrees with original model".into(),
                    ));
                }
                self.incumbent = Some(FeasibleSolution {
                    model: self.model.clone(),
                    values,
                    cost: w.cost,
                });
                let now = self.stats.solve_seconds + elapsed;
                self.stats.first_feasible_seconds.get_or_insert(now);
                self.stats.winner_found_seconds = Some(now);
            }
        }
        Ok(())
    }
    pub(super) fn update_memory(&mut self) {
        let mut bytes = self.nodes.capacity() * std::mem::size_of::<Node>()
            + self.defaults.capacity() * std::mem::size_of::<i64>()
            + self
                .preferred
                .as_ref()
                .map_or(0, |p| p.capacity() * std::mem::size_of::<i64>())
            + self
                .incumbent
                .as_ref()
                .map_or(0, |w| w.values.capacity() * std::mem::size_of::<i64>())
            + self
                .domains
                .iter()
                .map(Domain::retained_bytes)
                .sum::<usize>();
        bytes += knowledge::retained_bytes(&self.knowledge);
        if !Arc::ptr_eq(
            &self.factors,
            &self
                .knowledge
                .lock()
                .expect("solver knowledge lock poisoned")
                .factors,
        ) {
            bytes += self.factors.capacity() * std::mem::size_of::<Factor>()
                + self
                    .factors
                    .iter()
                    .map(|f| f.retained_bytes() - std::mem::size_of::<Factor>())
                    .sum::<usize>();
        }
        bytes += self.expanded.capacity() * std::mem::size_of::<(usize, Vec<usize>)>()
            + self
                .expanded
                .values()
                .map(|v| v.capacity() * std::mem::size_of::<usize>())
                .sum::<usize>();
        let mut pending = 0;
        let mut pruned = 0;
        for node in &self.nodes {
            bytes += node.key.factors.capacity() * std::mem::size_of::<usize>()
                + node.key.context.capacity() * std::mem::size_of::<(VarId, Domain)>()
                + node.parents.capacity() * std::mem::size_of::<usize>();
            bytes += node
                .key
                .context
                .iter()
                .map(|(_, d)| d.retained_bytes() - std::mem::size_of::<Domain>())
                .sum::<usize>();
            if let State::Pending(p) = &node.state {
                pending += 1;
                bytes += p
                    .domains
                    .iter()
                    .map(|(_, d)| std::mem::size_of::<VarId>() + d.retained_bytes())
                    .sum::<usize>()
                    + (p.factors.capacity() + p.residual.capacity()) * std::mem::size_of::<usize>()
                    + p.term_bounds.capacity() * std::mem::size_of::<(usize, u64)>();
            }
            match &node.state {
                State::Or(children) => {
                    bytes += children.capacity() * std::mem::size_of::<usize>();
                    if let Some(w) = &node.summary.witness {
                        pruned += children
                            .iter()
                            .filter(|&&id| {
                                let s = &self.nodes[id].summary;
                                !s.solved && s.lower >= w.cost
                            })
                            .count() as u64;
                    }
                }
                State::And {
                    values, children, ..
                } => {
                    bytes += values.capacity() * std::mem::size_of::<(usize, i64)>()
                        + children.capacity() * std::mem::size_of::<usize>()
                }
                State::Blocked(reasons) => {
                    bytes += reasons
                        .iter()
                        .map(|r| std::mem::size_of::<Obligation>() + r.reason.capacity())
                        .sum::<usize>()
                }
                _ => {}
            }
            if let Some(w) = &node.summary.witness {
                bytes += w.values.capacity() * std::mem::size_of::<(usize, i64)>();
            }
        }
        // Hash table keys duplicate contexts; include their inline payloads.
        bytes += self.memo.capacity() * std::mem::size_of::<(Key, usize)>();
        for k in self.memo.keys() {
            bytes += k.context.capacity() * std::mem::size_of::<(VarId, Domain)>()
                + k.factors.capacity() * std::mem::size_of::<usize>()
                + k.context
                    .iter()
                    .map(|(_, d)| d.retained_bytes() - std::mem::size_of::<Domain>())
                    .sum::<usize>();
        }
        self.stats.pruned = pruned;
        self.stats.retained_bytes = bytes;
        self.stats.peak_retained_bytes = self.stats.peak_retained_bytes.max(bytes);
        self.advance_peak_retained_bytes = self.advance_peak_retained_bytes.max(bytes);
        self.stats.max_frontier = self.stats.max_frontier.max(pending);
    }
    /// Fold completed regions into their exact summaries and evict cache-only
    /// descendants. Live pending regions and every proof summary used by a live
    /// parent remain reachable. A later request for an evicted memo key starts
    /// the exact region again, rather than interpreting absence as completion.
    fn evict_reclaimable(&mut self) -> bool {
        self.update_memory();
        let before = self.stats.retained_bytes;
        self.knowledge
            .lock()
            .expect("solver knowledge lock poisoned")
            .clear();
        if let Some(id) = self.active.take() {
            if let State::Pending(p) = &mut self.nodes[id].state {
                for (v, d) in &mut p.domains {
                    d.clone_from(&self.domains[v.0]);
                }
            }
        }
        for node in &mut self.nodes {
            if node.summary.solved {
                node.state = State::Done;
            }
        }
        let mut live = vec![false; self.nodes.len()];
        let mut stack = vec![self.root];
        while let Some(id) = stack.pop() {
            if std::mem::replace(&mut live[id], true) {
                continue;
            }
            match &self.nodes[id].state {
                State::Or(children) | State::And { children, .. } => {
                    stack.extend(children.iter().copied());
                }
                _ => {}
            }
        }
        let removed = live.iter().filter(|&&live| !live).count();
        self.stats.evictions += removed as u64;
        let mut mapping = vec![usize::MAX; self.nodes.len()];
        let mut nodes = Vec::with_capacity(self.nodes.len() - removed);
        for (old, mut node) in self.nodes.drain(..).enumerate() {
            if live[old] {
                mapping[old] = nodes.len();
                node.parents.clear();
                node.parents.shrink_to_fit();
                nodes.push(node);
            }
        }
        self.root = mapping[self.root];
        for node in &mut nodes {
            match &mut node.state {
                State::Or(children) | State::And { children, .. } => {
                    for child in children {
                        *child = mapping[*child];
                    }
                }
                _ => {}
            }
        }
        self.nodes = nodes;
        self.memo.clear();
        self.memo.shrink_to_fit();
        if self.options.memoize {
            for (id, node) in self.nodes.iter().enumerate() {
                self.memo.insert(node.key.clone(), id);
            }
        }
        for parent in 0..self.nodes.len() {
            let children = match &self.nodes[parent].state {
                State::Or(children) | State::And { children, .. } => children.clone(),
                _ => continue,
            };
            self.link(parent, &children);
        }
        self.update_memory();
        self.stats.retained_bytes < before
    }
    pub fn advance(&mut self, limits: Limits) -> Result<Outcome> {
        self.advance_mode(limits, false)
    }
    pub(super) fn advance_to_witness(&mut self, limits: Limits) -> Result<Outcome> {
        self.advance_mode(limits, true)
    }
    fn advance_mode(&mut self, limits: Limits, yield_witness: bool) -> Result<Outcome> {
        self.advance_peak_retained_bytes = self.stats.retained_bytes;
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        let result = self.advance_inner(limits, yield_witness);
        if let Err(error) = &result {
            self.failure = Some(error.clone());
        }
        result
    }
    fn advance_inner(&mut self, limits: Limits, yield_witness: bool) -> Result<Outcome> {
        let started = Instant::now();
        let work_start = self.stats.work;
        let reason = loop {
            self.update_incumbent(started.elapsed().as_secs_f64())?;
            let root = &self.nodes[self.root].summary;
            if root.infeasible {
                self.stats.solve_seconds += started.elapsed().as_secs_f64();
                return Ok(Outcome::Infeasible);
            }
            if root.solved {
                self.stats.solve_seconds += started.elapsed().as_secs_f64();
                self.update_memory();
                return Ok(Outcome::Optimal(self.incumbent.clone().ok_or_else(
                    || Error::Invariant("solved root has no witness".into()),
                )?));
            }
            // A validated starting witness need not have been rediscovered by
            // traversal. Only the root bound of this same context can certify it.
            if let Some(witness) = &self.incumbent {
                if root.lower >= witness.cost() {
                    self.stats.solve_seconds += started.elapsed().as_secs_f64();
                    return Ok(Outcome::Optimal(witness.clone()));
                }
            }
            if yield_witness {
                if let Some(w) = &self.incumbent {
                    if self.reported_cost.is_none_or(|cost| w.cost() < cost) {
                        self.reported_cost = Some(w.cost());
                        break StopReason::Work;
                    }
                }
            }
            if self.stats.work - work_start >= limits.work {
                break StopReason::Work;
            }
            if limits.time.is_some_and(|d| started.elapsed() >= d) {
                break StopReason::Time;
            }
            if (self.stats.work - work_start).is_multiple_of(64) {
                self.update_memory();
            }
            if let Some(cap) = limits.memory_bytes {
                if self.stats.retained_bytes >= cap {
                    // Completed memo/node payloads are recomputable. Give
                    // eviction a chance before reporting that retained
                    // storage is exhausted; pending regions are never
                    // discarded.
                    if self.evict_reclaimable() {
                        if self.stats.retained_bytes < cap {
                            continue;
                        }
                    }
                    break StopReason::Memory;
                }
            }
            let (pending, reasons) = self.next();
            let Some(id) = pending else {
                if reasons.is_empty() {
                    return Err(Error::Invariant("unfinished root has neither pending work nor a competitive coverage obligation".into()));
                }
                break StopReason::Coverage(reasons);
            };
            self.activate(id);
            if let Some(cap) = limits.memory_bytes {
                if let State::Pending(p) = &self.nodes[id].state {
                    let expansion = if !p.analyzing && p.cursor < p.factors.len() {
                        let fid = p.factors[p.cursor];
                        if let FactorKind::Fragment { factors } = &self.factors[fid].kind {
                            let next = self.expanded.get(&fid).map_or(0, Vec::len);
                            if self.factors[fid].guard_state(&self.domains)? == Some(true) {
                                factors
                                    .get(next)
                                    .map_or(0, |f| f.retained_bytes().saturating_add(1024))
                            } else {
                                0
                            }
                        } else {
                            0
                        }
                    } else if p.analyzing && p.cursor == p.factors.len() && !p.residual.is_empty() {
                        let groups = if self.options.decompose {
                            decomposition::components(&self.factors, &p.residual, &self.domains)
                        } else {
                            vec![p.residual.clone()]
                        };
                        let copies = if groups.len() == 1 { 2 } else { 1 };
                        groups
                            .iter()
                            .map(|g| {
                                let context = decomposition::scope(&self.factors, g)
                                    .iter()
                                    .map(|v| {
                                        self.domains[v.0].retained_bytes()
                                            + std::mem::size_of::<VarId>()
                                    })
                                    .sum::<usize>();
                                context
                                    .saturating_mul(3)
                                    .saturating_add(g.len() * std::mem::size_of::<usize>() * 3)
                                    .saturating_add(1024)
                            })
                            .sum::<usize>()
                            .saturating_mul(copies)
                    } else {
                        0
                    };
                    if expansion > 0 {
                        self.update_memory();
                        if self.stats.retained_bytes.saturating_add(expansion) > cap {
                            if self.evict_reclaimable() {
                                // Compaction renumbers nodes and saves the
                                // active workspace. Select it again before
                                // performing the postponed expansion.
                                continue;
                            }
                        }
                        if self.stats.retained_bytes.saturating_add(expansion) > cap {
                            break StopReason::Memory;
                        }
                    }
                }
            }
            self.step(id)?;
            self.stats.work += 1;
        };
        self.update_memory();
        self.stats.solve_seconds += started.elapsed().as_secs_f64();
        Ok(Outcome::Incomplete(Progress {
            lower_bound: self.nodes[self.root].summary.lower,
            incumbent: self.incumbent.clone(),
            reason,
            stats: self.stats.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasherDefault, Hasher};
    #[derive(Default)]
    struct Collision;
    impl Hasher for Collision {
        fn write(&mut self, _: &[u8]) {}
        fn finish(&self) -> u64 {
            0
        }
    }
    #[test]
    fn structural_context_equality_survives_hash_collisions() {
        let key = Key {
            factors: vec![0],
            context: vec![(VarId(0), Domain::singleton(1))],
        };
        let changed_value = Key {
            factors: vec![0],
            context: vec![(VarId(0), Domain::singleton(2))],
        };
        let changed_factor = Key {
            factors: vec![1],
            context: key.context.clone(),
        };
        let changed_variable = Key {
            factors: vec![0],
            context: vec![(VarId(1), Domain::singleton(1))],
        };
        let mut memo: HashMap<Key, u64, BuildHasherDefault<Collision>> = HashMap::default();
        for (i, k) in [
            key.clone(),
            changed_value.clone(),
            changed_factor.clone(),
            changed_variable.clone(),
        ]
        .into_iter()
        .enumerate()
        {
            memo.insert(k, i as u64);
        }
        assert_eq!(memo.len(), 4);
        assert_eq!(memo[&key], 0);
        assert_eq!(memo[&changed_value], 1);
        assert_eq!(memo[&changed_factor], 2);
        assert_eq!(memo[&changed_variable], 3);
    }
}
