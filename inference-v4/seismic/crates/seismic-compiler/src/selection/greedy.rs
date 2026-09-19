//! `Strategy::Greedy`: a diagnostic alternative to solver search. Deterministic
//! coordinate-wise improvement of the validated seed over the exported model. It ranks by
//! the exported objective only (every trial is a complete witness priced by
//! `Model::validate_assignment`), proves nothing, and never reports more than `Feasible`.
//!
//! One sweep visits, in this fixed order:
//! 1. occurrence choices, in pre-order of the current witness;
//! 2. the cover of every active sequence, in sequence order: all singletons, or one
//!    maximal offered fused interval with singletons around it;
//! 3. every active site, in id order, over its exported domain.
//! Each decision tries every other value with all other decisions fixed and keeps the
//! cheapest feasible complete witness (ties keep the current value). Sweeps repeat until
//! one improves nothing, at most `MAX_SWEEPS` times.
//!
//! Changing an occurrence's choice deactivates the old candidate's subtree and activates
//! the new one. `Backend::seed` cannot be constrained to a partial choice, so newly active
//! decisions take the seed policy's base values instead: the seed's own choice and site
//! value wherever the seed has one, otherwise candidate 0, the largest admissible width,
//! parts 1 (the smallest), and all-singleton covers. When that completion is infeasible
//! the all-smallest-values completion is tried once; if both are, the move is skipped.
//! The seed's piece-target and limit-repair steps are not replayed: later site sweeps
//! tune the new sites under the same objective.
use super::export::Export;
use seismic_lang::family::{CandidateRef, OccurrenceId, SequenceId, SiteKind, Witness};

/// Upper bound on sweeps; every counted sweep strictly lowers the estimate.
pub(super) const MAX_SWEEPS: u32 = 16;

pub(super) struct Improved {
    pub witness: Witness,
    pub estimate: u64,
    pub sweeps: u32,
    pub trials: u64,
}

struct Sweeps<'a> {
    export: &'a Export<'a>,
    /// Exact estimate of a complete witness; `None` when it is not feasible.
    cost: &'a dyn Fn(&Witness) -> Option<u64>,
    seed: &'a Witness,
    best: Witness,
    estimate: u64,
    trials: u64,
}

pub(super) fn improve(export: &Export, cost: &dyn Fn(&Witness) -> Option<u64>, seed: &Witness, seed_estimate: u64) -> Improved {
    let mut sweeps = Sweeps { export, cost, seed, best: seed.clone(), estimate: seed_estimate, trials: 0 };
    let mut completed = 0;
    while completed < MAX_SWEEPS {
        let before = sweeps.estimate;
        sweeps.choices();
        sweeps.covers();
        sweeps.sites();
        completed += 1;
        if sweeps.estimate == before {
            break;
        }
    }
    Improved { witness: sweeps.best, estimate: sweeps.estimate, sweeps: completed, trials: sweeps.trials }
}

impl Sweeps<'_> {
    /// Price `trial`; adopt it when strictly cheaper. `None` when it is infeasible.
    fn offer(&mut self, trial: Witness) -> Option<u64> {
        self.trials += 1;
        let cost = (self.cost)(&trial)?;
        if cost < self.estimate {
            (self.best, self.estimate) = (trial, cost);
        }
        Some(cost)
    }

    fn deactivate(&self, witness: &mut Witness, id: OccurrenceId) {
        let Some(choice) = witness.choices.remove(&id) else { return };
        let candidate = self.export.family().candidate(CandidateRef { occurrence: id, candidate: choice });
        for site in &candidate.sites {
            witness.sites.remove(site);
        }
        for sequence in &candidate.sequences {
            witness.covers.remove(sequence);
        }
        for child in &candidate.children {
            self.deactivate(witness, *child);
        }
    }

    /// Complete `witness` below `id`; false when a site of the subtree admits no value.
    fn activate(&self, witness: &mut Witness, id: OccurrenceId, forced: Option<u32>, smallest: bool) -> bool {
        let family = self.export.family();
        let choice = forced.or_else(|| self.seed.choices.get(&id).copied()).unwrap_or(0);
        let Some(candidate) = family.occurrence(id).candidates.get(choice as usize) else { return false };
        witness.choices.insert(id, choice);
        for site in &candidate.sites {
            let values = self.export.site_values(*site);
            let (Some(least), Some(most)) = (values.first(), values.last()) else { return false };
            let parts = matches!(family.sites[site.0 as usize].kind, SiteKind::Parts { .. });
            let base = if smallest || parts { *least } else { *most };
            witness.sites.insert(*site, if smallest { base } else { self.seed.sites.get(site).copied().unwrap_or(base) });
        }
        for sequence in &candidate.sequences {
            witness.covers.insert(*sequence, self.singletons(*sequence, None));
        }
        candidate.children.iter().all(|child| self.activate(witness, *child, None, smallest))
    }

    fn choices(&mut self) {
        let family = self.export.family();
        let mut pending = vec![OccurrenceId(0)];
        while let Some(id) = pending.pop() {
            for candidate in 0..family.occurrence(id).candidates.len() as u32 {
                if self.best.choices.get(&id) == Some(&candidate) {
                    continue;
                }
                for smallest in [false, true] {
                    let mut trial = self.best.clone();
                    self.deactivate(&mut trial, id);
                    if self.activate(&mut trial, id, Some(candidate), smallest) && self.offer(trial).is_some() {
                        break;
                    }
                }
            }
            let Some(chosen) = self.best.choices.get(&id) else { continue };
            pending.extend(family.candidate(CandidateRef { occurrence: id, candidate: *chosen }).children.iter().rev());
        }
    }

    /// Singletons of every unit of `sequence` outside `fused`, and `fused` itself.
    fn singletons(&self, sequence: SequenceId, fused: Option<(u32, u32)>) -> Vec<(u32, u32)> {
        let units = self.export.family().sequences[sequence.0 as usize].units.len() as u32;
        let mut cover: Vec<(u32, u32)> = (0..units).filter(|u| fused.is_none_or(|(start, end)| *u < start || *u >= end)).map(|u| (u, u + 1)).collect();
        cover.extend(fused);
        cover.sort_unstable();
        cover
    }

    fn covers(&mut self) {
        let sequences: Vec<SequenceId> = self.best.covers.keys().copied().collect();
        for sequence in sequences {
            let offered: Vec<(u32, u32)> = self.export.offered(sequence).collect();
            let maximal = offered.iter().copied().filter(|(start, end)| end - start >= 2 && !offered.iter().any(|(s, e)| (s, e) != (start, end) && s <= start && end <= e));
            let options: Vec<Vec<(u32, u32)>> = std::iter::once(None).chain(maximal.map(Some)).map(|fused| self.singletons(sequence, fused)).collect();
            for cover in options {
                if self.best.covers.get(&sequence) != Some(&cover) {
                    let mut trial = self.best.clone();
                    trial.covers.insert(sequence, cover);
                    self.offer(trial);
                }
            }
        }
    }

    fn sites(&mut self) {
        let sites: Vec<_> = self.best.sites.keys().copied().collect();
        for site in sites {
            for value in self.export.site_values(site).to_vec() {
                if self.best.sites.get(&site) != Some(&value) {
                    let mut trial = self.best.clone();
                    trial.sites.insert(site, value);
                    self.offer(trial);
                }
            }
        }
    }
}
