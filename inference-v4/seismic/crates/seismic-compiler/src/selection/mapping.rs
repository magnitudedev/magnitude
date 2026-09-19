//! The target-neutral half of every backend's `Backend` hooks: the site domain rule, the
//! legal fusion intervals instantiation can realize, the conversion of hard limits into
//! guarded constraints, the additive factor skeleton and the constructive seed. A backend
//! supplies its `Accounting`, its limits (`Legality`), its cost formula (`Costs`) and its
//! seed piece target; nothing here ranks alternatives by profitability.
use super::quantity::{lookup, scope, Quantity};
use super::structure::{referenced, Account, Accounting, Analysis, Launch};
use super::{Constraint, Factor, Interval, IntervalRef, SelectionError};
use seismic_lang::family::{CandidateRef, Family, OccurrenceId, Requirement, Site, SiteId, SiteKind, UnitKind, Witness};
use seismic_lang::sir::{DefKind, Program, SliceParent};
use seismic_lang::syntax::ast::RegionMode;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A site domain holds at most this many values.
pub const DOMAIN_VALUES: usize = 24;
/// Largest partition count of a `merge` axis.
pub const MAX_PARTS: i64 = 1024;

pub fn requirement_site(requirement: &Requirement) -> SiteId {
    match requirement {
        Requirement::Multiple { site, .. } | Requirement::AtLeast { site, .. } | Requirement::AtMost { site, .. } | Requirement::Equal { site, .. } | Requirement::Divides { site, .. } => *site,
    }
}

pub fn requirement_holds(requirement: &Requirement, value: i64) -> bool {
    match *requirement {
        Requirement::Multiple { unit, .. } => unit > 0 && value % unit == 0,
        Requirement::AtLeast { value: least, .. } => value >= least,
        Requirement::AtMost { value: most, .. } => value <= most,
        Requirement::Equal { value: equal, .. } => value == equal,
        Requirement::Divides { extent, .. } => value > 0 && extent % value == 0,
    }
}

/// Domain definition of one site (not a profitability filter).
///
/// Width: the divisors `d` of `extent` (so no tail pieces exist) with `d == 1`, `d == extent`,
/// `d` pinned by an `Equal` requirement, or `d == unit * 2^k` for `unit` 1 or the unit of any
/// `Multiple` requirement on the site; then intersected with `owner` (the requirements of
/// the site's owning candidate, active whenever the site is). Parts: 1 and the powers of
/// two up to `min(extent, 1024)`, intersected likewise. Above 24 values the ascending list
/// is thinned to 24 evenly spaced ordinals, always keeping both ends. `domains` further
/// restricts a width over a runtime-bounded domain to one (the only width instantiation realizes).
pub fn domain(site: &Site, mentioned: &[Requirement], owner: &[Requirement]) -> Vec<i64> {
    let extent = site.extent;
    if extent < 1 {
        return Vec::new();
    }
    let mut values = BTreeSet::new();
    match site.kind {
        SiteKind::Width { .. } => {
            let units: BTreeSet<i64> = mentioned.iter().filter_map(|r| match r { Requirement::Multiple { unit, .. } if *unit > 0 => Some(*unit), _ => None }).chain([1]).collect();
            let pinned: BTreeSet<i64> = mentioned.iter().filter_map(|r| match r { Requirement::Equal { value, .. } => Some(*value), _ => None }).collect();
            let mut keep = |d: i64| {
                let structured = units.iter().any(|u| d % u == 0 && u64::try_from(d / u).is_ok_and(u64::is_power_of_two));
                if d == 1 || d == extent || pinned.contains(&d) || structured {
                    values.insert(d);
                }
            };
            let mut d = 1;
            while d <= extent / d {
                if extent % d == 0 {
                    keep(d);
                    keep(extent / d);
                }
                d += 1;
            }
        }
        SiteKind::Parts { .. } => {
            let mut parts = 1;
            while parts <= extent.min(MAX_PARTS) {
                values.insert(parts);
                parts *= 2;
            }
        }
    }
    let values: Vec<i64> = values.into_iter().filter(|v| owner.iter().all(|r| requirement_holds(r, *v))).collect();
    if values.len() <= DOMAIN_VALUES {
        return values;
    }
    let last = values.len() - 1;
    let ordinals: BTreeSet<usize> = (0..DOMAIN_VALUES).map(|k| k * last / (DOMAIN_VALUES - 1)).collect();
    ordinals.into_iter().map(|i| values[i]).collect()
}

/// The finite value domain of every site of `family` (`Backend::bind_structure`).
pub fn domains(program: &Program, family: &Family) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError> {
    let all: Vec<&Requirement> = family.occurrences.iter().flat_map(|o| &o.candidates).flat_map(|c| &c.requirements).collect();
    let mut domains = BTreeMap::new();
    for site in &family.sites {
        let mentioned: Vec<Requirement> = all.iter().filter(|r| requirement_site(r) == site.id).map(|r| (*r).clone()).collect();
        let owner: Vec<Requirement> = family.candidate(site.owner).requirements.iter().filter(|r| requirement_site(r) == site.id).cloned().collect();
        let mut values = domain(site, &mentioned, &owner);
        // A width over a domain with runtime bounds (`extent` is only its static upper
        // bound) has no static piece count; instantiation realizes it at width one only.
        if let SiteKind::Width { slice, .. } = site.kind {
            let candidate = family.candidate(site.owner);
            let template = family.template(candidate.template);
            let parent = program.definition(template.definition).body.as_ref().and_then(|body| body.slices.get(slice.0 as usize)).map(|declared| &declared.parent);
            if let Some(SliceParent::Domain { lo, hi }) = parent {
                let fixed = |name: &String| template.shapes.contains_key(name) || candidate.structural.iter().any(|(structural, _)| structural == name);
                if !hi.sub(lo).params().iter().all(fixed) {
                    values.retain(|v| *v == 1);
                }
            }
        }
        if values.is_empty() {
            let definition = &program.definition(family.template(family.candidate(site.owner).template).definition).name;
            return Err(SelectionError::UnsupportedMapping(format!("`{definition}` site {}: no divisor of extent {} satisfies its requirements", site.id.0, site.extent)));
        }
        domains.insert(site.id, values);
    }
    Ok(domains)
}

/// The root launch of `owner` that a region unit denotes, if it is one.
pub fn unit_launch<'b, A: Accounting>(analysis: &'b Analysis<'_, A>, owner: CandidateRef, kind: &UnitKind) -> Option<&'b Launch<A>> {
    let UnitKind::Region(region) = kind else { return None };
    analysis.accounts.get(&owner)?.launches.iter().find(|l| l.region == *region)
}

/// Singletons for every unit, then the two fused realizations instantiation has:
/// (a) contiguous runs (length >= 2) of `Elementwise` units of one block, not crossing a
///     completion; (b) contiguous runs of root-level `parallel` region units with equal
///     binder counts and domain extents, no merge, no completion between them, their
///     corresponding width sites tied by `equal_sites`.
pub fn intervals<A: Accounting>(analysis: &Analysis<'_, A>, family: &Family) -> Result<Vec<Interval>, SelectionError> {
    let mut out = Vec::new();
    for sequence in &family.sequences {
        let interval = |start: usize, end: usize, equal_sites| Interval { sequence: sequence.id, start: start as u32, end: end as u32, requires: Vec::new(), equal_sites };
        out.extend((0..sequence.units.len()).map(|i| interval(i, i + 1, Vec::new())));
        let invocation = analysis.sequence_block(sequence).map_err(SelectionError::Reconstruction)?.1.invocation;
        let fusible = |left: usize| -> bool {
            let (a, b) = (&sequence.units[left], &sequence.units[left + 1]);
            if a.completion_after {
                return false;
            }
            if a.kind == UnitKind::Elementwise && b.kind == UnitKind::Elementwise {
                return true;
            }
            let (Some(x), Some(y)) = (unit_launch(analysis, sequence.owner, &a.kind), unit_launch(analysis, sequence.owner, &b.kind)) else { return false };
            let plain = |l: &Launch<A>| l.mode == RegionMode::Parallel && !l.merge && l.binders.len() == l.binder_count;
            invocation && plain(x) && plain(y) && x.binder_count == y.binder_count && x.binders.iter().zip(&y.binders).all(|(p, q)| p.1 == q.1)
        };
        for start in 0..sequence.units.len() {
            let region_run = matches!(sequence.units[start].kind, UnitKind::Region(_));
            for end in start + 2..=sequence.units.len() {
                if !fusible(end - 2) {
                    break;
                }
                let mut equal_sites = Vec::new();
                if region_run {
                    let first = unit_launch(analysis, sequence.owner, &sequence.units[start].kind);
                    for unit in &sequence.units[start + 1..end] {
                        let pairs = first.zip(unit_launch(analysis, sequence.owner, &unit.kind));
                        equal_sites.extend(pairs.into_iter().flat_map(|(a, b)| a.binders.iter().zip(&b.binders).map(|(p, q)| (p.0, q.0))));
                    }
                }
                out.push(interval(start, end, equal_sites));
            }
        }
    }
    Ok(out)
}

/// A value of site assignments, as the limit and cost closures read it.
pub type Sites<'s> = &'s dyn Fn(SiteId) -> Option<i64>;

/// One hard limit of a backend over derived quantities, active under `guard`.
#[derive(Clone)]
pub struct Legality {
    pub guard: Vec<CandidateRef>,
    /// Every quantity the limit reads; their sites are the constraint's scope.
    pub reads: Vec<Quantity>,
    pub holds: Arc<dyn Fn(Sites<'_>) -> bool + Send + Sync>,
    /// What the limit needs, for the diagnostic of a limit no site can repair.
    pub needed: Arc<dyn Fn(Sites<'_>) -> Result<String, String> + Send + Sync>,
    /// Seed repairs, tried in order: step the highest site of these quantities up (`true`) or
    /// down to its next admissible value.
    pub repairs: Vec<(Vec<Quantity>, bool)>,
    pub reason: String,
}

impl Legality {
    pub fn scope(&self) -> Vec<SiteId> {
        scope(&self.reads)
    }
}

/// A refinement partitions the pieces of the binder it refines (`Family::refinements`), and
/// instantiation realizes dividing widths only: the refinement's width divides the refined
/// width. Shared by every backend; repaired in the seed by lowering the refinement, else by
/// raising the refined width.
pub fn refinements(family: &Family) -> Vec<Legality> {
    family
        .refinements
        .iter()
        .map(|&(inner, outer)| {
            let mut guard = Vec::new();
            let mut current = family.sites.get(inner.0 as usize).map(|site| site.owner);
            while let Some(candidate) = current {
                guard.push(candidate);
                current = family.occurrence(candidate.occurrence).parent;
            }
            guard.reverse();
            Legality {
                guard,
                reads: vec![Quantity::Site(inner), Quantity::Site(outer)],
                holds: Arc::new(move |site| matches!((site(inner), site(outer)), (Some(i), Some(o)) if i > 0 && o % i == 0)),
                needed: Arc::new(move |site| Ok(format!("width {:?} of the refinement dividing width {:?}", site(inner), site(outer)))),
                repairs: vec![(vec![Quantity::Site(inner)], false), (vec![Quantity::Site(outer)], true)],
                reason: format!("site {} refines site {}: its width divides the refined width", inner.0, outer.0),
            }
        })
        .collect()
}

/// Hard limits as guarded solver constraints. A limit no site can repair, on a candidate
/// chain selection cannot avoid, is a diagnosed composition failure rather than an
/// unexplained infeasible family.
pub fn constraints(family: &Family, legalities: Vec<Legality>) -> Result<Vec<Constraint>, SelectionError> {
    let legalities: Vec<Legality> = legalities.into_iter().chain(refinements(family)).collect();
    for legality in legalities.iter().filter(|l| l.scope().is_empty() && !(l.holds)(&|_| None)) {
        if legality.guard.iter().all(|g| family.occurrence(g.occurrence).candidates.len() == 1) {
            return Err(SelectionError::IncompatibleComposition(format!("{} (needs {})", legality.reason, (legality.needed)(&|_| None).unwrap_or_else(|e| e))));
        }
    }
    Ok(legalities
        .into_iter()
        .map(|legality| {
            let sites = legality.scope();
            let (table, holds) = (sites.clone(), legality.holds.clone());
            Constraint { guard: legality.guard, scope: sites, holds: Box::new(move |values| holds(&lookup(&table, values))), reason: legality.reason }
        })
        .collect())
}

/// One execution scope of a candidate, offered to the backend's cost formula.
pub struct CostScope<'s, 'a, A: Accounting> {
    pub analysis: &'s Analysis<'a, A>,
    pub account: &'s Account<A>,
    /// The root launch the scope is, or executes in; `None` for invocation-scope serial work.
    pub launch: Option<(CandidateRef, usize)>,
    /// Launches this factor pays for (zero when a selected interval pays the launch).
    pub launches: u64,
    pub work: &'s A::Work,
    pub pieces: &'s Quantity,
}

/// The cost of one scope: the quantities it reads beyond the scope's work, and the closure.
pub struct ScopeCost {
    pub reads: Vec<Quantity>,
    pub cost: Box<dyn Fn(Sites<'_>) -> Result<u64, String> + Send + Sync>,
}

/// A backend's estimate formula over the shared factor skeleton.
pub trait Costs<A: Accounting> {
    /// Text after "`name` part: " in a scope factor's label.
    fn scope_label(&self) -> String;
    fn scope(&self, scope: CostScope<'_, '_, A>) -> ScopeCost;
    /// One launch with no work: what a selected run of root regions pays once.
    fn launch(&self) -> Box<dyn Fn() -> Result<u64, String> + Send + Sync>;
    /// Write then read of `bits` of tile storage.
    fn materialization(&self) -> Box<dyn Fn(u64) -> Result<u64, String> + Send + Sync>;
}

/// Additive factors. Per candidate: one factor per root launch and one for the remaining
/// body, each the backend's cost of its own work. Child calls carry their own factors,
/// guarded by their own `CandidateRef` chain. Per interval: one launch overhead for root
/// region units (fused regions pay one), and the write + read of every elementwise output
/// that survives the interval (a fused run keeps only its last output and outputs
/// referenced after it).
pub fn factors<A: Accounting>(analysis: &Analysis<'_, A>, family: &Family, intervals: &[Interval], costs: &dyn Costs<A>) -> Result<Vec<Factor>, SelectionError> {
    let mut out = Vec::new();
    // Root regions that are sequence units pay their launch through the selected interval.
    let mut sequenced: BTreeSet<(CandidateRef, u32)> = BTreeSet::new();
    for sequence in &family.sequences {
        sequenced.extend(sequence.units.iter().filter_map(|u| unit_launch(analysis, sequence.owner, &u.kind)).map(|l| (sequence.owner, l.region.0)));
    }
    for (&candidate, account) in &analysis.accounts {
        let name = &analysis.bounds[&candidate].definition.name;
        type Launched = Option<(CandidateRef, usize)>;
        let mut scopes: Vec<(String, u64, &A::Work, &Quantity, Launched)> = account
            .launches
            .iter()
            .enumerate()
            .map(|(ordinal, l)| (format!("region#{}", l.region.0), u64::from(!sequenced.contains(&(candidate, l.region.0))), &l.work, &l.pieces, Some((candidate, ordinal))))
            .collect();
        scopes.push(("body".into(), account.serial_launches, &account.rest, &account.context.scope.pieces, account.context.scope.launch));
        for (part, launches, work, pieces, launch) in scopes {
            if launches == 0 && A::quantities(work).is_empty() {
                continue;
            }
            let ScopeCost { reads, cost } = costs.scope(CostScope { analysis, account, launch, launches, work, pieces });
            let sites = scope(A::quantities(work).into_iter().chain([pieces]).chain(&reads));
            let table = sites.clone();
            out.push(Factor {
                guard: account.context.guards.clone(),
                intervals: Vec::new(),
                scope: sites,
                cost: Box::new(move |values| cost(&lookup(&table, values))),
                label: format!("`{name}` {part}: {}", costs.scope_label()),
            });
        }
    }
    for (ordinal, interval) in intervals.iter().enumerate() {
        let Some(sequence) = family.sequences.get(interval.sequence.0 as usize).filter(|s| s.id == interval.sequence) else {
            return Err(SelectionError::Reconstruction(format!("interval names sequence {} absent from the family", interval.sequence.0)));
        };
        let Some(units) = sequence.units.get(interval.start as usize..interval.end as usize).filter(|u| !u.is_empty()) else {
            return Err(SelectionError::Reconstruction(format!("interval [{}, {}) exceeds sequence {}", interval.start, interval.end, sequence.id.0)));
        };
        let Some(account) = analysis.accounts.get(&sequence.owner) else {
            return Err(SelectionError::Reconstruction(format!("sequence {} has no owner account", sequence.id.0)));
        };
        let name = &analysis.bounds[&sequence.owner].definition.name;
        let guard = account.context.guards.clone();
        let reference = vec![IntervalRef(ordinal as u32)];
        if units.iter().all(|u| unit_launch(analysis, sequence.owner, &u.kind).is_some()) {
            let launch = costs.launch();
            out.push(Factor {
                guard,
                intervals: reference,
                scope: Vec::new(),
                cost: Box::new(move |_| launch()),
                label: format!("`{name}` sequence {} [{}, {}): one launch, unqualified estimate", sequence.id.0, interval.start, interval.end),
            });
            continue;
        }
        if !units.iter().any(|u| u.kind == UnitKind::Elementwise) {
            continue;
        }
        // Surviving elementwise outputs, or the reason they are not derivable.
        let surviving: Result<Quantity, String> = analysis.sequence_block(sequence).and_then(|(block, executed)| {
            let after = block.get(units[units.len() - 1].statements.end..).unwrap_or(&[]);
            let mut terms = Vec::new();
            for (position, unit) in units.iter().enumerate().filter(|(_, u)| u.kind == UnitKind::Elementwise) {
                let (variable, bits) = analysis.unit_output(sequence.owner, block, unit)?;
                if position + 1 == units.len() || variable.is_none_or(|v| referenced(after, v)) {
                    terms.push(Quantity::product(executed.multiplicity.iter().cloned().chain([bits])));
                }
            }
            Ok(Quantity::Sum(terms))
        });
        let sites = surviving.as_ref().map_or_else(|_| Vec::new(), |q| scope([q]));
        let (table, materialization) = (sites.clone(), costs.materialization());
        out.push(Factor {
            guard,
            intervals: reference,
            scope: sites,
            cost: Box::new(move |values| materialization(surviving.as_ref().map_err(Clone::clone)?.eval(&lookup(&table, values))?)),
            label: format!("`{name}` sequence {} [{}, {}): surviving tile write + read, unqualified estimate", sequence.id.0, interval.start, interval.end),
        });
    }
    Ok(out)
}

/// Seed policy: deterministic feasibility construction, never profitability ranking.
/// Choices: occurrences in pre-order; each takes its first candidate (those adopted
/// through a `lower ... for <target>` body first, then the rest, each in ordinal order) under
/// which every site named by the requirements of the candidates chosen so far plus its own
/// keeps a non-empty `domain ∩ requirements`, and whose remaining occurrences can be
/// completed the same way; otherwise the next candidate is tried (backtracking).
/// Sites: root `parallel` binders take the smallest admissible widths whose piece count
/// does not exceed `piece_target`, raising the last binder first; every other width site
/// takes its largest admissible value; parts take 1. A violated limit is then repaired by
/// its own `repairs`. Covers: all singletons.
pub fn seed<A: Accounting>(
    program: &Program,
    family: &Family,
    target: &str,
    analysis: &Analysis<'_, A>,
    domains: &BTreeMap<SiteId, Vec<i64>>,
    intervals: &[Interval],
    piece_target: u64,
    legalities: Vec<Legality>,
) -> Result<Witness, SelectionError> {
    let mut witness = Witness::default();
    struct Construction<'c> {
        program: &'c Program,
        family: &'c Family,
        target: &'c str,
        domains: &'c BTreeMap<SiteId, Vec<i64>>,
        /// Deepest occurrence that had no admissible candidate, for the diagnostic.
        blocked: Option<OccurrenceId>,
    }
    impl Construction<'_> {
        fn admissible(&self, chosen: &[CandidateRef], candidate: CandidateRef) -> bool {
            let requirements: Vec<&Requirement> = chosen.iter().chain([&candidate]).flat_map(|c| &self.family.candidate(*c).requirements).collect();
            self.family.candidate(candidate).requirements.iter().map(requirement_site).all(|site| {
                self.domains.get(&site).is_some_and(|values| values.iter().any(|v| requirements.iter().filter(|r| requirement_site(r) == site).all(|r| requirement_holds(r, *v))))
            })
        }
        /// Complete `chosen` over `pending` (a pre-order stack); restores both on failure.
        fn complete(&mut self, pending: &mut Vec<OccurrenceId>, chosen: &mut Vec<CandidateRef>) -> bool {
            let Some(id) = pending.pop() else { return true };
            let Some(occurrence) = self.family.occurrences.get(id.0 as usize) else {
                pending.push(id);
                return false;
            };
            let lowered = |c: &u32| matches!(&self.program.definition(occurrence.candidates[*c as usize].via).kind, DefKind::Lower { target } if target == self.target);
            let ordinals = 0..occurrence.candidates.len() as u32;
            let preferred: Vec<u32> = ordinals.clone().filter(lowered).chain(ordinals.filter(|c| !lowered(c))).collect();
            for candidate in preferred.into_iter().map(|candidate| CandidateRef { occurrence: id, candidate }) {
                if !self.admissible(chosen, candidate) {
                    continue;
                }
                let depth = pending.len();
                chosen.push(candidate);
                pending.extend(self.family.candidate(candidate).children.iter().rev());
                if self.complete(pending, chosen) {
                    return true;
                }
                pending.truncate(depth);
                chosen.pop();
            }
            self.blocked.get_or_insert(id);
            pending.push(id);
            false
        }
    }
    let mut construction = Construction { program, family, target, domains, blocked: None };
    let mut active = Vec::new();
    if !construction.complete(&mut vec![OccurrenceId(0)], &mut active) {
        let id = construction.blocked.unwrap_or(OccurrenceId(0));
        let name = family.occurrences.get(id.0 as usize).and_then(|o| program.families.get(o.family)).map_or("?", |f| f.name.as_str());
        return Err(SelectionError::MissingCoverage(format!("seed: no `{target}` implementation of `{name}` (occurrence {}) is admissible under the requirements of its callers", id.0)));
    }
    witness.choices.extend(active.iter().map(|c| (c.occurrence, c.candidate)));
    // Admissible values: the domain under every active single-site requirement.
    let requirements: Vec<&Requirement> = active.iter().flat_map(|c| &family.candidate(*c).requirements).collect();
    let mut admissible: BTreeMap<SiteId, Vec<i64>> = BTreeMap::new();
    for site in active.iter().flat_map(|c| &family.candidate(*c).sites) {
        let values: Vec<i64> = domains.get(site).into_iter().flatten().copied().filter(|v| requirements.iter().filter(|r| requirement_site(r) == *site).all(|r| requirement_holds(r, *v))).collect();
        let (Some(least), Some(most)) = (values.first().copied(), values.last().copied()) else {
            return Err(SelectionError::UnsupportedMapping(format!("seed: site {} has no value satisfying the selected requirements", site.0)));
        };
        let parts = matches!(family.sites.get(site.0 as usize).map(|s| &s.kind), Some(SiteKind::Parts { .. }));
        witness.sites.insert(*site, if parts { least } else { most });
        admissible.insert(*site, values);
    }
    let step = |witness: &mut Witness, site: SiteId, up: bool| -> bool {
        let (Some(values), Some(current)) = (admissible.get(&site), witness.sites.get(&site).copied()) else { return false };
        let next = if up { values.iter().copied().find(|v| *v > current) } else { values.iter().rev().copied().find(|v| *v < current) };
        next.map(|v| witness.sites.insert(site, v)).is_some()
    };
    for launch in active.iter().flat_map(|c| &analysis.accounts[c].launches).filter(|l| l.mode == RegionMode::Parallel) {
        for (site, _) in &launch.binders {
            if let Some(least) = admissible.get(site).and_then(|values| values.first()) {
                witness.sites.insert(*site, *least);
            }
        }
        while launch.pieces.eval(&|id| witness.sites.get(&id).copied()).is_ok_and(|p| p > piece_target) {
            if !launch.binders.iter().rev().any(|(site, _)| step(&mut witness, *site, true)) {
                break;
            }
        }
    }
    let legalities: Vec<Legality> = legalities.into_iter().chain(refinements(family)).filter(|l| l.guard.iter().all(|g| active.contains(g))).collect();
    let mut budget = 2 * admissible.values().map(Vec::len).sum::<usize>() + 1;
    while let Some(violated) = legalities.iter().find(|l| !(l.holds)(&|id| witness.sites.get(&id).copied())) {
        budget -= 1;
        let repaired = budget > 0 && violated.repairs.iter().any(|(quantities, up)| scope(quantities).iter().rev().any(|site| step(&mut witness, *site, *up)));
        if !repaired {
            return Err(SelectionError::UnsupportedMapping(format!("seed: no admissible widths satisfy: {}", violated.reason)));
        }
    }
    for sequence in active.iter().flat_map(|c| &family.candidate(*c).sequences) {
        let units = family.sequences.get(sequence.0 as usize).map_or(0, |s| s.units.len()) as u32;
        if let Some(missing) = (0..units).find(|i| !intervals.iter().any(|v| v.sequence == *sequence && v.start == *i && v.end == i + 1)) {
            return Err(SelectionError::UnsupportedMapping(format!("seed: sequence {} unit {missing} has no separate execution", sequence.0)));
        }
        witness.covers.insert(*sequence, (0..units).map(|i| (i, i + 1)).collect());
    }
    Ok(witness)
}
