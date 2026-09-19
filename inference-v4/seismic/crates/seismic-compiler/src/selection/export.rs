//! Family + backend contributions -> one immutable solver model, and back.
//!
//! Every factor names exactly the decisions it depends on: its own site variables, the
//! choice variables of the ancestors that make it active, and its interval selections.
//! Activity is a conjunction of choice literals carried as factor guards; inactive
//! variables are pinned to one canonical value so they never multiply the search.
use super::{Constraint, Factor, Interval, SelectionError};
use magnitude_solver::model::{Constraint as Rel, Cost, Literal};
use magnitude_solver::{Domain, Model, ModelBuilder, VarId};
use seismic_lang::family::{CandidateRef, Family, OccurrenceId, Requirement, SequenceId, SiteId, Witness};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Largest tabulated scope product.
const TABLE_LIMIT: usize = 200_000;

#[derive(Clone)]
enum Slot {
    /// Searched site; `values` ascending, `values[0]` is the canonical inactive value.
    Var { var: VarId, values: Vec<i64> },
    /// Singleton domain: no variable, reconstruction identity kept.
    Const(i64),
    /// No admissible value: the owner is unselectable.
    Dead,
}

pub(super) struct Export<'f> {
    pub model: Arc<Model>,
    family: &'f Family,
    /// Choice variable per occurrence; `None` for fixed (single-candidate) occurrences.
    choices: Vec<Option<VarId>>,
    sites: Vec<Slot>,
    /// Selection variable and `(sequence, start, end)` per backend interval.
    intervals: Vec<(VarId, SequenceId, u32, u32)>,
    /// Canonical inactive assignment of every variable.
    inactive: Vec<i64>,
}

struct Rows {
    vars: Vec<VarId>,
    /// `(variable tuple, full closure arguments)` over the scope product.
    rows: Vec<(Vec<i64>, Vec<i64>)>,
}

fn defect(message: impl Into<String>) -> SelectionError {
    SelectionError::Reconstruction(message.into())
}

pub(super) fn solver_error(error: magnitude_solver::Error) -> SelectionError {
    match error {
        magnitude_solver::Error::Overflow(m) => SelectionError::AnalysisUnavailable(format!("estimate overflow: {m}")),
        other => defect(format!("solver: {other}")),
    }
}

fn required_site(r: &Requirement) -> SiteId {
    match r {
        Requirement::Multiple { site, .. }
        | Requirement::AtLeast { site, .. }
        | Requirement::AtMost { site, .. }
        | Requirement::Equal { site, .. }
        | Requirement::Divides { site, .. } => *site,
    }
}

fn admits(r: &Requirement, v: i64) -> bool {
    match r {
        Requirement::Multiple { unit, .. } => *unit != 0 && v % unit == 0,
        Requirement::AtLeast { value, .. } => v >= *value,
        Requirement::AtMost { value, .. } => v <= *value,
        Requirement::Equal { value, .. } => v == *value,
        Requirement::Divides { extent, .. } => v != 0 && extent % v == 0,
    }
}

struct Builder<'f> {
    family: &'f Family,
    b: ModelBuilder,
    choices: Vec<Option<VarId>>,
    sites: Vec<Slot>,
    inactive: Vec<i64>,
}

impl<'f> Builder<'f> {
    fn variable(&mut self, name: String, domain: Domain, inactive: i64, activity: &[Literal]) -> VarId {
        let var = self.b.variable(name, domain);
        self.inactive.push(inactive);
        for active in activity {
            self.b.constraint(Rel::InactiveValue { active: *active, variable: var, inactive });
        }
        var
    }

    fn checked(&self, c: CandidateRef) -> Result<(), SelectionError> {
        let known = self.family.occurrences.get(c.occurrence.0 as usize).is_some_and(|o| (c.candidate as usize) < o.candidates.len());
        if known { Ok(()) } else { Err(defect(format!("unknown candidate {}.{}", c.occurrence.0, c.candidate))) }
    }

    /// Choice literals whose conjunction is "`c` and all its ancestors are selected".
    fn activity(&self, c: CandidateRef) -> Vec<Literal> {
        let mut literals = Vec::new();
        let mut current = Some(c);
        let mut steps = 0;
        while let Some(c) = current {
            if let Some(var) = self.choices[c.occurrence.0 as usize] {
                literals.push(Literal::new(var, c.candidate as i64));
            }
            current = self.family.occurrence(c.occurrence).parent;
            steps += 1;
            if steps > self.family.occurrences.len() {
                break;
            }
        }
        literals
    }

    fn site(&self, id: SiteId) -> Result<Slot, SelectionError> {
        self.sites.get(id.0 as usize).cloned().ok_or_else(|| defect(format!("unknown site {}", id.0)))
    }

    fn site_activity(&self, id: SiteId) -> Result<Vec<Literal>, SelectionError> {
        let site = self.family.sites.get(id.0 as usize).ok_or_else(|| defect(format!("unknown site {}", id.0)))?;
        Ok(self.activity(site.owner))
    }

    /// The conjunction `guards` has no execution.
    fn never(&mut self, guards: Vec<Literal>) {
        let var = self.b.variable("never", Domain::singleton(0));
        self.inactive.push(0);
        self.b.guarded_constraint(guards, Rel::InDomain { variable: var, domain: Domain::singleton(1) });
    }

    /// Product of the scope's domains; `None` when a scope site has no admissible value.
    fn rows(&self, scope: &[SiteId], what: &str) -> Result<Option<Rows>, SelectionError> {
        let mut vars: Vec<VarId> = Vec::new();
        let mut domains: Vec<&[i64]> = Vec::new();
        // Per scope position: constant, or index into `vars`.
        let mut positions = Vec::with_capacity(scope.len());
        for id in scope {
            match &self.sites.get(id.0 as usize).ok_or_else(|| defect(format!("unknown site {}", id.0)))? {
                Slot::Dead => return Ok(None),
                Slot::Const(v) => positions.push(Err(*v)),
                Slot::Var { var, values } => match vars.iter().position(|v| v == var) {
                    Some(i) => positions.push(Ok(i)),
                    None => {
                        positions.push(Ok(vars.len()));
                        vars.push(*var);
                        domains.push(values);
                    }
                },
            }
        }
        let count = domains.iter().try_fold(1usize, |n, d| n.checked_mul(d.len())).filter(|n| *n <= TABLE_LIMIT);
        let Some(count) = count else {
            return Err(SelectionError::AnalysisUnavailable(format!("{what}: scope product exceeds {TABLE_LIMIT} tuples")));
        };
        let mut rows = Vec::with_capacity(count);
        let mut index = vec![0usize; domains.len()];
        loop {
            let tuple: Vec<i64> = index.iter().zip(&domains).map(|(i, d)| d[*i]).collect();
            let args = positions.iter().map(|p| match p { Ok(i) => tuple[*i], Err(v) => *v }).collect();
            rows.push((tuple, args));
            let mut axis = domains.len();
            loop {
                if axis == 0 {
                    return Ok(Some(Rows { vars, rows }));
                }
                axis -= 1;
                index[axis] += 1;
                if index[axis] < domains[axis].len() {
                    break;
                }
                index[axis] = 0;
            }
        }
    }
}

impl<'f> Export<'f> {
    pub fn build(family: &'f Family, domains: &BTreeMap<SiteId, Vec<i64>>, constraints: &[Constraint], intervals: &[Interval], factors: &[Factor]) -> Result<Export<'f>, SelectionError> {
        match family.occurrences.first() {
            None => return Err(SelectionError::InvalidSource(format!("entry `{}` has no occurrence", family.entry))),
            Some(root) if root.candidates.is_empty() => {
                return Err(SelectionError::MissingCoverage(format!("entry `{}` has no implementation adopted for target `{}`", family.entry, family.target)))
            }
            Some(_) => (),
        }
        let mut b = ModelBuilder::new();
        b.units("ns");
        let mut x = Builder { family, b, choices: vec![None; family.occurrences.len()], sites: Vec::new(), inactive: Vec::new() };

        // Implementation choices. Activity needs ancestors' variables, so allocate first.
        for (i, o) in family.occurrences.iter().enumerate() {
            if o.id.0 as usize != i {
                return Err(defect(format!("occurrence {} stored at index {i}", o.id.0)));
            }
            if let Some(p) = o.parent {
                x.checked(p)?;
            }
            if o.candidates.len() > 1 {
                let domain = Domain::interval(0, o.candidates.len() as i64 - 1).map_err(solver_error)?;
                x.choices[i] = Some(x.b.variable(format!("choice{i}"), domain));
                x.inactive.push(0);
            }
        }
        for (i, o) in family.occurrences.iter().enumerate() {
            let Some(parent) = o.parent else { continue };
            let activity = x.activity(parent);
            match x.choices[i] {
                Some(var) => {
                    for active in activity {
                        x.b.constraint(Rel::InactiveValue { active, variable: var, inactive: 0 });
                    }
                }
                None if o.candidates.is_empty() => x.never(activity),
                None => (),
            }
        }

        // Numerical sites over backend domain ∩ owner requirements.
        for (i, site) in family.sites.iter().enumerate() {
            if site.id.0 as usize != i {
                return Err(defect(format!("site {} stored at index {i}", site.id.0)));
            }
            x.checked(site.owner)?;
            let backend = domains.get(&site.id).ok_or_else(|| defect(format!("backend bound no domain for site {i}")))?;
            let own = &family.candidate(site.owner).requirements;
            let mut values: Vec<i64> = backend.iter().copied().filter(|v| own.iter().filter(|r| required_site(r) == site.id).all(|r| admits(r, *v))).collect();
            values.sort_unstable();
            values.dedup();
            let activity = x.activity(site.owner);
            let slot = match values.as_slice() {
                [] => {
                    x.never(activity);
                    Slot::Dead
                }
                [v] => Slot::Const(*v),
                [first, ..] => {
                    let first = *first;
                    let var = x.variable(format!("site{i}"), Domain::set(values.iter().copied()), first, &activity);
                    Slot::Var { var, values }
                }
            };
            x.sites.push(slot);
        }

        // Requirements a candidate places on a site it does not own (structural parameters).
        for o in &family.occurrences {
            for (k, candidate) in o.candidates.iter().enumerate() {
                let me = CandidateRef { occurrence: o.id, candidate: k as u32 };
                for r in &candidate.requirements {
                    let id = required_site(r);
                    if family.sites.get(id.0 as usize).ok_or_else(|| defect(format!("requirement on unknown site {}", id.0)))?.owner == me {
                        continue;
                    }
                    let mut guards = x.activity(me);
                    guards.extend(x.site_activity(id)?);
                    match x.site(id)? {
                        Slot::Dead => (),
                        Slot::Const(v) => {
                            if !admits(r, v) {
                                x.never(guards);
                            }
                        }
                        Slot::Var { var, values } => {
                            let allowed: Vec<i64> = values.iter().copied().filter(|v| admits(r, *v)).collect();
                            if allowed.is_empty() {
                                x.never(guards);
                            } else if allowed.len() < values.len() {
                                x.b.guarded_constraint(guards, Rel::InDomain { variable: var, domain: Domain::set(allowed) });
                            }
                        }
                    }
                }
            }
        }

        // Fusion intervals and the exact cover of every active sequence.
        let mut selections = Vec::with_capacity(intervals.len());
        let mut containing: Vec<Vec<Vec<VarId>>> = family.sequences.iter().map(|s| vec![Vec::new(); s.units.len()]).collect();
        let mut seen = BTreeMap::new();
        for (i, interval) in intervals.iter().enumerate() {
            let sequence = family.sequences.get(interval.sequence.0 as usize).ok_or_else(|| defect(format!("interval {i} names unknown sequence {}", interval.sequence.0)))?;
            if interval.start >= interval.end || interval.end as usize > sequence.units.len() {
                return Err(defect(format!("interval {i} [{}, {}) is outside sequence {}", interval.start, interval.end, sequence.id.0)));
            }
            if seen.insert((interval.sequence, interval.start, interval.end), i).is_some() {
                return Err(defect(format!("sequence {} interval [{}, {}) has two realizations", sequence.id.0, interval.start, interval.end)));
            }
            x.checked(sequence.owner)?;
            let activity = x.activity(sequence.owner);
            let z = x.variable(format!("interval{i}"), Domain::boolean(), 0, &activity);
            let selected = Literal::new(z, 1);
            for unit in interval.start..interval.end {
                containing[interval.sequence.0 as usize][unit as usize].push(z);
            }
            for required in &interval.requires {
                x.checked(*required)?;
                for consequence in x.activity(*required) {
                    x.b.constraint(Rel::Implies { premise: selected, consequence });
                }
            }
            for (left, right) in &interval.equal_sites {
                let mut guards = vec![selected];
                guards.extend(x.site_activity(*left)?);
                guards.extend(x.site_activity(*right)?);
                match (x.site(*left)?, x.site(*right)?) {
                    (Slot::Dead, _) | (_, Slot::Dead) => (),
                    (Slot::Const(a), Slot::Const(b)) => {
                        if a != b {
                            x.never(guards);
                        }
                    }
                    (Slot::Var { var, values }, Slot::Const(v)) | (Slot::Const(v), Slot::Var { var, values }) => {
                        if values.contains(&v) {
                            x.b.guarded_constraint(guards, Rel::InDomain { variable: var, domain: Domain::singleton(v) });
                        } else {
                            x.never(guards);
                        }
                    }
                    (Slot::Var { var: left, .. }, Slot::Var { var: right, .. }) => {
                        if left != right {
                            x.b.guarded_constraint(guards, Rel::Equal { left, right });
                        }
                    }
                }
            }
            selections.push((z, interval.sequence, interval.start, interval.end));
        }
        for ((i, sequence), units) in family.sequences.iter().enumerate().zip(containing) {
            if sequence.id.0 as usize != i {
                return Err(defect(format!("sequence {} is not stored at its index", sequence.id.0)));
            }
            x.checked(sequence.owner)?;
            let activity = x.activity(sequence.owner);
            for variables in units {
                if variables.is_empty() {
                    x.never(activity.clone());
                } else {
                    x.b.guarded_constraint(activity.clone(), Rel::ExactlyOne { variables });
                }
            }
        }

        // Backend legality, tabulated over exactly its scope.
        for constraint in constraints {
            let what = format!("constraint `{}`", constraint.reason);
            let Some(table) = x.rows(&constraint.scope, &what)? else { continue };
            let mut guards = Vec::new();
            for c in &constraint.guard {
                x.checked(*c)?;
                guards.extend(x.activity(*c));
            }
            for id in &constraint.scope {
                guards.extend(x.site_activity(*id)?);
            }
            let total = table.rows.len();
            let tuples: Vec<Vec<i64>> = table.rows.into_iter().filter(|(_, args)| (constraint.holds)(args)).map(|(tuple, _)| tuple).collect();
            if tuples.is_empty() {
                x.never(guards);
            } else if tuples.len() < total {
                x.b.guarded_constraint(guards, Rel::Table { variables: table.vars, tuples });
            }
        }

        // Local cost factors, each over exactly its scope.
        for factor in factors {
            let what = format!("factor `{}`", factor.label);
            let Some(table) = x.rows(&factor.scope, &what)? else { continue };
            let mut guards = Vec::new();
            for c in &factor.guard {
                x.checked(*c)?;
                guards.extend(x.activity(*c));
            }
            for id in &factor.scope {
                guards.extend(x.site_activity(*id)?);
            }
            for r in &factor.intervals {
                let (z, ..) = selections.get(r.0 as usize).ok_or_else(|| defect(format!("{what} names unknown interval {}", r.0)))?;
                guards.push(Literal::new(*z, 1));
            }
            let mut entries = Vec::with_capacity(table.rows.len());
            for (tuple, args) in table.rows {
                let cost = (factor.cost)(&args).map_err(|e| SelectionError::AnalysisUnavailable(format!("{what} at {args:?}: {e}")))?;
                entries.push((tuple, cost));
            }
            let cost = match entries.as_slice() {
                [(tuple, cost)] if tuple.is_empty() => Cost::Constant(*cost),
                _ => Cost::Table { variables: table.vars, entries },
            };
            x.b.guarded_cost(guards, cost);
        }

        let Builder { b, choices, sites, inactive, .. } = x;
        let model = Arc::new(b.build().map_err(solver_error)?);
        Ok(Export { model, family, choices, sites, intervals: selections, inactive })
    }

    pub fn family(&self) -> &'f Family {
        self.family
    }

    /// Ascending admissible values of a site; empty when its owner is unselectable.
    pub fn site_values(&self, site: SiteId) -> &[i64] {
        match &self.sites[site.0 as usize] {
            Slot::Var { values, .. } => values,
            Slot::Const(value) => std::slice::from_ref(value),
            Slot::Dead => &[],
        }
    }

    /// Every interval the backend offers for `sequence`, as `(start, end)`.
    pub fn offered(&self, sequence: SequenceId) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.intervals.iter().filter(move |(_, s, ..)| *s == sequence).map(|(_, _, start, end)| (*start, *end))
    }

    /// The witness a full solver assignment denotes: only active decisions appear.
    pub fn witness(&self, values: &[i64]) -> Witness {
        let mut witness = Witness::default();
        let mut pending = vec![OccurrenceId(0)];
        while let Some(id) = pending.pop() {
            let occurrence = self.family.occurrence(id);
            let choice = match self.choices[id.0 as usize] {
                Some(var) => values[var.0] as u32,
                None => 0,
            };
            let Some(candidate) = occurrence.candidates.get(choice as usize) else { continue };
            witness.choices.insert(id, choice);
            for site in &candidate.sites {
                match &self.sites[site.0 as usize] {
                    Slot::Var { var, .. } => {
                        witness.sites.insert(*site, values[var.0]);
                    }
                    Slot::Const(v) => {
                        witness.sites.insert(*site, *v);
                    }
                    Slot::Dead => (),
                }
            }
            for sequence in &candidate.sequences {
                let mut cover: Vec<(u32, u32)> = self.intervals.iter().filter(|(z, s, ..)| s == sequence && values[z.0] == 1).map(|(_, _, start, end)| (*start, *end)).collect();
                cover.sort_unstable();
                witness.covers.insert(*sequence, cover);
            }
            pending.extend(candidate.children.iter().copied());
        }
        witness
    }

    /// Full solver assignment of a complete witness, inactive variables at their
    /// canonical values. Structural completeness is checked here; legality and cost by
    /// `Model::validate_assignment`.
    pub fn assignment(&self, witness: &Witness) -> Result<Vec<i64>, String> {
        let mut values = self.inactive.clone();
        let (mut choices, mut sites, mut covers) = (0, 0, 0);
        let mut pending = vec![OccurrenceId(0)];
        while let Some(id) = pending.pop() {
            let occurrence = self.family.occurrence(id);
            let choice = *witness.choices.get(&id).ok_or_else(|| format!("active occurrence {} has no choice", id.0))?;
            let candidate = occurrence.candidates.get(choice as usize).ok_or_else(|| format!("occurrence {} has no candidate {choice}", id.0))?;
            choices += 1;
            if let Some(var) = self.choices[id.0 as usize] {
                values[var.0] = choice as i64;
            }
            for site in &candidate.sites {
                let value = *witness.sites.get(site).ok_or_else(|| format!("active site {} has no value", site.0))?;
                sites += 1;
                match &self.sites[site.0 as usize] {
                    Slot::Var { var, values: domain } if domain.contains(&value) => values[var.0] = value,
                    Slot::Const(v) if *v == value => (),
                    _ => return Err(format!("site {} does not admit {value}", site.0)),
                }
            }
            for sequence in &candidate.sequences {
                let cover = witness.covers.get(sequence).ok_or_else(|| format!("active sequence {} has no cover", sequence.0))?;
                covers += 1;
                for (start, end) in cover {
                    let (z, ..) = self
                        .intervals
                        .iter()
                        .find(|(_, s, a, b)| s == sequence && a == start && b == end)
                        .ok_or_else(|| format!("sequence {} has no legal interval [{start}, {end})", sequence.0))?;
                    values[z.0] = 1;
                }
            }
            pending.extend(candidate.children.iter().copied());
        }
        if choices != witness.choices.len() || sites != witness.sites.len() || covers != witness.covers.len() {
            return Err("witness assigns inactive occurrences, sites or sequences".into());
        }
        Ok(values)
    }
}
