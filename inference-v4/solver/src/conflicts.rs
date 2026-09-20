//! Reusable contradictions scoped to an immutable model and residual context.
//!
//! A fact means that the conjunction of its retained factors has no feasible
//! assignment inside its premise domains. It remains true after domains shrink
//! or more factors are added. It says nothing about a disjoint, wider or differently
//! guarded region. Facts originate only from a completed infeasibility proof;
//! objective cutoffs and unfinished coverage never establish such a fact.
//!
//! This module is private: the search is the proof-producing authority. There is
//! deliberately no public callback that lets callers assert contradictions.

use crate::model::{Domain, Error, Factor, Model, Result, VarId};
use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    sync::Arc,
};

#[derive(Clone, Debug, Default)]
pub(crate) struct ConflictStats {
    pub learned: u64,
    pub hits: u64,
    pub evictions: u64,
    pub subsumed: u64,
}

#[derive(Clone, Debug)]
struct Conflict {
    // Structural factors include guards and retained fragment definitions.
    factors: Vec<Factor>,
    // Includes every scoped variable, even fixed boundary/guard variables.
    context: Vec<(VarId, Domain)>,
    // Immutable payload accounting is computed once, not on every search step.
    bytes: usize,
}

/// FIFO eviction only forgets redundant deductions; it never owns unresolved
/// search coverage. Zero capacity disables learning without changing answers.
pub(crate) struct ConflictStore {
    model: Arc<Model>,
    capacity: usize,
    facts: VecDeque<Conflict>,
    stats: ConflictStats,
    payload_bytes: usize,
}

impl ConflictStore {
    pub fn new(model: Arc<Model>, capacity: usize) -> Self {
        Self {
            model,
            capacity,
            facts: VecDeque::new(),
            stats: ConflictStats::default(),
            payload_bytes: 0,
        }
    }

    #[cfg(test)]
    pub fn stats(&self) -> &ConflictStats {
        &self.stats
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        self.trim();
        // In particular, releasing all facts must release the arena as well.
        if capacity == 0 {
            self.facts = VecDeque::new();
        }
    }

    fn trim(&mut self) {
        while self.facts.len() > self.capacity {
            self.payload_bytes -= self.facts.pop_front().unwrap().bytes;
            self.stats.evictions += 1;
        }
    }

    /// Learn a narrower, single-factor contradiction only after the typed factor
    /// independently confirms it. Other factors used to narrow the workspace do
    /// not become hidden premises: every resulting scoped domain is retained.
    /// This permits reuse across regions with unrelated extra constraints.
    #[cfg(test)]
    pub fn record_factor_infeasible(
        &mut self,
        factor: &Factor,
        domains: &[Domain],
    ) -> Result<bool> {
        if self.capacity == 0 || !factor.assess(domains)?.infeasible {
            return Ok(false);
        }
        let context = factor
            .scope()
            .into_iter()
            .map(|variable| {
                domains
                    .get(variable.0)
                    .map(|domain| (variable, domain.clone()))
                    .ok_or_else(|| {
                        Error::Invariant("conflict factor references an absent domain".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        self.record_infeasible(std::slice::from_ref(factor), &[0], &context)
    }

    /// Retained owned bytes; the shared immutable model is charged by its owner.
    pub fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.facts.capacity() * std::mem::size_of::<Conflict>()
            + self.payload_bytes
    }

    /// Records an unconditional, established contradiction over this whole
    /// region. The search MUST call this only after propagation proves a
    /// contradiction, exact assessment rejects the region, or all exhaustive OR
    /// children are infeasible. "Cannot improve the incumbent" is insufficient.
    ///
    /// Use the original region context, not narrowed domains that depended on
    /// factors omitted from `region`. Keeping the complete proof premises is
    /// essential. No conflict minimization is attempted here.
    pub fn record_infeasible(
        &mut self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Result<bool> {
        if self.capacity == 0 {
            return Ok(false);
        }
        self.validate_region(factors, region, context)?;
        if self.matches(factors, region, context)? {
            self.stats.subsumed += 1;
            return Ok(false);
        }
        let mut retained = Vec::with_capacity(region.len());
        let mut seen = HashSet::with_capacity(region.len());
        for &index in region {
            let factor = &factors[index];
            if seen.insert(factor) {
                retained.push(factor.clone());
            }
        }
        let mut context = context.to_vec();
        context.sort_by_key(|(variable, _)| *variable);
        let bytes = retained.capacity() * std::mem::size_of::<Factor>()
            + retained
                .iter()
                .map(|factor| factor.retained_bytes() - std::mem::size_of::<Factor>())
                .sum::<usize>()
            + context.capacity() * std::mem::size_of::<(VarId, Domain)>()
            + context
                .iter()
                .map(|(_, domain)| domain.retained_bytes() - std::mem::size_of::<Domain>())
                .sum::<usize>();
        self.payload_bytes += bytes;
        self.facts.push_back(Conflict {
            factors: retained,
            context,
            bytes,
        });
        self.stats.learned += 1;
        self.trim();
        Ok(true)
    }

    /// Returns true only when a retained proof applies to this exact constraint
    /// conjunction or a stronger conjunction within its premise domains.
    pub fn excludes(
        &mut self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Result<bool> {
        if self.facts.is_empty() {
            return Ok(false);
        }
        self.validate_region(factors, region, context)?;
        let found = self.matches(factors, region, context)?;
        if found {
            self.stats.hits += 1;
        }
        Ok(found)
    }

    fn matches(
        &self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Result<bool> {
        // Search contexts are sorted. Preserve the private API's support for
        // unordered validated contexts without allocating a map in the hot path.
        let sorted = context.windows(2).all(|pair| pair[0].0 < pair[1].0);
        let mut conjunction = None;
        for fact in &self.facts {
            let mut applies = true;
            for (variable, premise) in &fact.context {
                let candidate = if sorted {
                    context
                        .binary_search_by_key(variable, |(id, _)| *id)
                        .ok()
                        .map(|index| &context[index].1)
                } else {
                    context
                        .iter()
                        .find(|(id, _)| id == variable)
                        .map(|(_, domain)| domain)
                };
                let Some(candidate) = candidate else {
                    applies = false;
                    break;
                };
                if !subset(candidate, premise)? {
                    applies = false;
                    break;
                }
            }
            if !applies {
                continue;
            }
            // Only construct the structural index after a domain match. Most
            // retained contradictions concern another branch's domains. Full
            // Factor equality resolves hash collisions and includes guards and
            // fragments; numeric factor IDs alone never authorize reuse.
            let conjunction = conjunction.get_or_insert_with(|| {
                region
                    .iter()
                    .map(|&index| &factors[index])
                    .collect::<HashSet<_>>()
            });
            if fact
                .factors
                .iter()
                .all(|premise| conjunction.contains(premise))
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn validate_region(
        &self,
        factors: &[Factor],
        region: &[usize],
        context: &[(VarId, Domain)],
    ) -> Result<()> {
        let sorted = context.windows(2).all(|pair| pair[0].0 < pair[1].0);
        let mut variables = (!sorted).then(BTreeSet::new);
        for (variable, domain) in context {
            let Some(original) = self.model.variables().get(variable.0) else {
                return Err(Error::Invariant(
                    "conflict context references a variable outside its model".into(),
                ));
            };
            if variables
                .as_mut()
                .is_some_and(|variables| !variables.insert(*variable))
            {
                return Err(Error::Invariant(
                    "duplicate variable in conflict context".into(),
                ));
            }
            domain.validate()?;
            if !subset(domain, &original.domain)? {
                return Err(Error::Invariant(
                    "conflict context exceeds its immutable model domain".into(),
                ));
            }
        }
        for &index in region {
            let Some(factor) = factors.get(index) else {
                return Err(Error::Invariant(
                    "conflict region references an absent factor".into(),
                ));
            };
            for variable in factor.scope() {
                let present = match &variables {
                    Some(variables) => variables.contains(&variable),
                    None => context
                        .binary_search_by_key(&variable, |(id, _)| *id)
                        .is_ok(),
                };
                if !present {
                    return Err(Error::Invariant(
                        "conflict context omits a factor or guard dependency".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn subset(candidate: &Domain, premise: &Domain) -> Result<bool> {
    if candidate == premise || candidate.is_empty() {
        return Ok(true);
    }
    if let Some(value) = candidate.singleton_value() {
        return Ok(premise.contains(value));
    }
    if premise.is_empty() || candidate.min() < premise.min() || candidate.max() > premise.max() {
        return Ok(false);
    }
    // A contiguous premise is completely described by its endpoints. The
    // subtraction uses i128 to include the complete i64 interval safely.
    if premise.cardinality()
        == (premise.max().unwrap() as i128 - premise.min().unwrap() as i128 + 1) as u128
    {
        return Ok(true);
    }
    Ok(candidate.intersect(premise)?.cardinality() == candidate.cardinality())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, FactorKind, Literal, ModelBuilder};

    fn disequality() -> (Arc<Model>, VarId, VarId) {
        let mut builder = ModelBuilder::new();
        let x = builder.variable("x", Domain::interval(0, 2).unwrap());
        let y = builder.variable("y", Domain::interval(0, 2).unwrap());
        builder.constraint(Constraint::NotEqual { left: x, right: y });
        (Arc::new(builder.build().unwrap()), x, y)
    }

    #[test]
    fn subset_fast_paths_match_finite_sets_including_holes() {
        for a in 0..128 {
            for b in 0..128 {
                let candidate = Domain::set((-3..4).filter(|v| a & (1 << (v + 3)) != 0));
                let premise = Domain::set((-3..4).filter(|v| b & (1 << (v + 3)) != 0));
                assert_eq!(subset(&candidate, &premise).unwrap(), a & !b == 0);
            }
        }
        let all = Domain::interval(i64::MIN, i64::MAX).unwrap();
        assert!(subset(&Domain::set([i64::MIN, 0, i64::MAX]), &all).unwrap());
    }

    #[test]
    fn unordered_contexts_and_duplicate_factors_preserve_reuse_and_accounting() {
        let (model, x, y) = disequality();
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 1);
        let empty_bytes = store.retained_bytes();
        let context = [(y, Domain::singleton(1)), (x, Domain::singleton(1))];
        store
            .record_infeasible(&factors, &[0, 0], &context)
            .unwrap();
        assert_eq!(store.facts[0].factors.len(), 1);
        assert!(store.excludes(&factors, &[0], &context).unwrap());
        assert!(store.retained_bytes() > empty_bytes);
        store
            .record_infeasible(
                &factors,
                &[0],
                &[(x, Domain::singleton(0)), (y, Domain::singleton(0))],
            )
            .unwrap();
        assert_eq!(
            store.payload_bytes,
            store.facts.iter().map(|fact| fact.bytes).sum::<usize>()
        );
        store.set_capacity(0);
        assert_eq!(store.retained_bytes(), empty_bytes);
        assert_eq!(store.payload_bytes, 0);
    }

    #[test]
    fn every_matched_tiny_context_is_independently_infeasible() {
        let (model, x, y) = disequality();
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 4);
        let premise = vec![(x, Domain::singleton(1)), (y, Domain::singleton(1))];
        store.record_infeasible(&factors, &[0], &premise).unwrap();
        for xmask in 0..8 {
            for ymask in 0..8 {
                let dx = Domain::set((0..3).filter(|v| xmask & (1 << v) != 0));
                let dy = Domain::set((0..3).filter(|v| ymask & (1 << v) != 0));
                let actual_infeasible = dx.values().all(|a| dy.values().all(|b| a == b));
                let hit = store.excludes(&factors, &[0], &[(x, dx), (y, dy)]).unwrap();
                assert!(
                    !hit || actual_infeasible,
                    "unsound cache hit {xmask} {ymask}"
                );
            }
        }
        assert!(store.stats().hits > 0);
    }

    #[test]
    fn guards_and_structural_factors_cannot_be_changed_or_omitted() {
        let mut builder = ModelBuilder::new();
        let guard = builder.variable("guard", Domain::boolean());
        let x = builder.variable("x", Domain::boolean());
        builder.guarded_constraint(
            vec![Literal::new(guard, 1)],
            Constraint::NotEqual { left: x, right: x },
        );
        let model = Arc::new(builder.build().unwrap());
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 2);
        let premise = [(guard, Domain::singleton(1)), (x, Domain::boolean())];
        store.record_infeasible(&factors, &[0], &premise).unwrap();
        assert!(store
            .excludes(
                &factors,
                &[0],
                &[(guard, Domain::singleton(1)), (x, Domain::singleton(0))]
            )
            .unwrap());
        assert!(!store
            .excludes(
                &factors,
                &[0],
                &[(guard, Domain::singleton(0)), (x, Domain::boolean())]
            )
            .unwrap());
        assert!(!store
            .excludes(
                &factors,
                &[0],
                &[(guard, Domain::boolean()), (x, Domain::boolean())]
            )
            .unwrap());
        assert!(store
            .excludes(&factors, &[0], &[(x, Domain::boolean())])
            .is_err());
        let mut changed = factors.clone();
        changed[0].kind = FactorKind::Constraint(Constraint::Equal { left: x, right: x });
        assert!(!store.excludes(&changed, &[0], &premise).unwrap());
        assert!(!store.excludes(&factors, &[], &premise).unwrap());
    }

    #[test]
    fn complete_factor_conjunction_is_required_and_stronger_queries_are_allowed() {
        let mut builder = ModelBuilder::new();
        let x = builder.variable("x", Domain::boolean());
        let y = builder.variable("y", Domain::boolean());
        builder.constraint(Constraint::Equal { left: x, right: y });
        builder.constraint(Constraint::NotEqual { left: x, right: y });
        builder.constraint(Constraint::Equal { left: x, right: x });
        let model = Arc::new(builder.build().unwrap());
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 4);
        let context = [(x, Domain::boolean()), (y, Domain::boolean())];
        store
            .record_infeasible(&factors, &[0, 1], &context)
            .unwrap();
        assert!(!store.excludes(&factors, &[0], &context).unwrap());
        assert!(!store.excludes(&factors, &[1], &context).unwrap());
        assert!(store.excludes(&factors, &[2, 1, 0], &context).unwrap());
        assert!(!store
            .record_infeasible(&factors, &[0, 1, 2], &context)
            .unwrap());
        assert_eq!(store.stats().subsumed, 1);
    }

    #[test]
    fn eviction_and_zero_capacity_forget_only_deductions() {
        let (model, x, y) = disequality();
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 1);
        let a = [(x, Domain::singleton(0)), (y, Domain::singleton(0))];
        let b = [(x, Domain::singleton(1)), (y, Domain::singleton(1))];
        store.record_infeasible(&factors, &[0], &a).unwrap();
        let bytes = store.retained_bytes();
        store.record_infeasible(&factors, &[0], &b).unwrap();
        assert_eq!(store.len(), 1);
        assert_eq!(store.stats().evictions, 1);
        assert!(!store.excludes(&factors, &[0], &a).unwrap());
        assert!(store.excludes(&factors, &[0], &b).unwrap());
        store.set_capacity(0);
        assert!(store.retained_bytes() < bytes);
        assert!(!store.excludes(&factors, &[0], &b).unwrap());
        assert!(!store.record_infeasible(&factors, &[0], &a).unwrap());
    }

    #[test]
    fn context_validation_keeps_model_and_guard_dependencies() {
        let (model, x, y) = disequality();
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 1);
        assert!(store
            .record_infeasible(&factors, &[0], &[(x, Domain::singleton(0))])
            .is_err());
        assert!(store
            .record_infeasible(
                &factors,
                &[0],
                &[(x, Domain::singleton(4)), (y, Domain::singleton(4))]
            )
            .is_err());
        assert!(store
            .record_infeasible(
                &factors,
                &[0],
                &[
                    (x, Domain::singleton(0)),
                    (x, Domain::singleton(0)),
                    (y, Domain::singleton(0))
                ]
            )
            .is_err());
        assert!(store
            .record_infeasible(
                &factors,
                &[1],
                &[(x, Domain::singleton(0)), (y, Domain::singleton(0))]
            )
            .is_err());
    }

    #[test]
    fn subset_checks_preserve_holes_in_large_domains() {
        let mut builder = ModelBuilder::new();
        let x = builder.variable("x", Domain::interval(0, 1_000_000).unwrap());
        builder.constraint(Constraint::InDomain {
            variable: x,
            domain: Domain::singleton(1),
        });
        let model = Arc::new(builder.build().unwrap());
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 1);
        let premise = [(x, Domain::progression(0, 1_000_000, 2).unwrap())];
        store.record_infeasible(&factors, &[0], &premise).unwrap();
        assert!(store
            .excludes(&factors, &[0], &[(x, Domain::set([0, 2, 20]))])
            .unwrap());
        assert!(!store
            .excludes(&factors, &[0], &[(x, Domain::interval(0, 20).unwrap())])
            .unwrap());
    }

    #[test]
    fn factor_learning_requires_a_real_typed_contradiction() {
        let (model, x, y) = disequality();
        let factors = model.factors().to_vec();
        let mut store = ConflictStore::new(model, 2);
        assert!(!store
            .record_factor_infeasible(&factors[0], &[Domain::boolean(), Domain::boolean()])
            .unwrap());
        assert_eq!(store.len(), 0);
        assert!(store
            .record_factor_infeasible(&factors[0], &[Domain::singleton(1), Domain::singleton(1)])
            .unwrap());
        assert!(store
            .excludes(
                &factors,
                &[0],
                &[(x, Domain::singleton(1)), (y, Domain::singleton(1))]
            )
            .unwrap());
        assert!(!store
            .excludes(
                &factors,
                &[0],
                &[(x, Domain::singleton(0)), (y, Domain::singleton(1))]
            )
            .unwrap());
    }
}
