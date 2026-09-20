//! Execution-unit normalization, witness validation, and the family queries that
//! selection, backends and instantiation share.

use super::{
    CandidateRef, Family, OccurrenceId, Requirement, SequenceId, SiteId, SiteKind, Witness,
};
use crate::sir::{Body, Program, SliceParent, StmtKind};
use crate::sym::Sym;
use crate::sir::RegionMode;
use crate::types::{Elem, Extent, RegionId, SliceId};
use std::collections::{BTreeMap, BTreeSet};

mod units;

pub(super) use units::{sequences, BlockUnits};

/// The binder that owns `slice`'s partition: `Rebind` chains resolved to the `Domain` or
/// `Refine` binder that is a numerical site.
pub fn owning_slice(body: &Body, slice: SliceId) -> Option<SliceId> {
    let mut current = slice;
    for _ in 0..=body.slices.len() {
        match body.slices.get(current.0 as usize)?.parent {
            SliceParent::Rebind(parent) => current = parent,
            SliceParent::Domain { .. } | SliceParent::Refine(_) => return Some(current),
        }
    }
    None
}

impl Requirement {
    pub fn site(&self) -> SiteId {
        match self {
            Requirement::Multiple { site, .. }
            | Requirement::AtLeast { site, .. }
            | Requirement::AtMost { site, .. }
            | Requirement::Equal { site, .. }
            | Requirement::Divides { site, .. } => *site,
        }
    }

    pub fn holds(&self, value: i64) -> bool {
        match self {
            Requirement::Multiple { unit, .. } => *unit != 0 && value % unit == 0,
            Requirement::AtLeast { value: bound, .. } => value >= *bound,
            Requirement::AtMost { value: bound, .. } => value <= *bound,
            Requirement::Equal {
                value: required, ..
            } => value == *required,
            Requirement::Divides { extent, .. } => value != 0 && extent % value == 0,
        }
    }
}

impl Family {
    /// Active candidates under `witness`, parents before children, in occurrence order.
    /// Stops descending at an occurrence without an in-range choice.
    pub fn active_candidates(&self, witness: &Witness) -> Vec<CandidateRef> {
        let mut out = Vec::new();
        let mut pending = vec![OccurrenceId(0)];
        while let Some(id) = pending.pop() {
            let Some(occurrence) = self.occurrences.get(id.0 as usize) else {
                continue;
            };
            let Some(&choice) = witness.choices.get(&id) else {
                continue;
            };
            let Some(candidate) = occurrence.candidates.get(choice as usize) else {
                continue;
            };
            out.push(CandidateRef {
                occurrence: id,
                candidate: choice,
            });
            pending.extend(candidate.children.iter().rev());
        }
        out
    }

    /// Concrete semantic shape environment of a candidate's body. Structurally bound shape
    /// parameters are absent; see `structural_site`.
    pub fn shapes(&self, candidate: CandidateRef) -> &BTreeMap<String, i64> {
        &self.template(self.candidate(candidate).template).shapes
    }

    /// Concrete element environment of a candidate's body.
    pub fn elems(&self, candidate: CandidateRef) -> &BTreeMap<String, Elem> {
        &self.template(self.candidate(candidate).template).elems
    }

    /// The site bound to structural shape parameter `param` of the candidate's template.
    /// It belongs to an ancestor candidate.
    pub fn structural_site(&self, candidate: CandidateRef, param: &str) -> Option<SiteId> {
        self.candidate(candidate)
            .structural
            .iter()
            .find(|(name, _)| name == param)
            .map(|(_, site)| site.0)
    }

    /// For each dynamic shape parameter of the candidate's template, the caller-side extent
    /// expression bound at the call occurrence (`CandidateBinding::shape_args` of the
    /// candidate's definition). It is stated over the parent candidate's body: its shape
    /// parameters (themselves possibly dynamic there) and runtime atoms. Empty at the entry.
    pub fn dynamic_args(&self, program: &Program, candidate: CandidateRef) -> Vec<(String, Sym)> {
        let occurrence = self.occurrence(candidate.occurrence);
        let template = self.template(self.candidate(candidate).template);
        let (Some(parent), Some(call)) = (occurrence.parent, occurrence.call) else {
            return Vec::new();
        };
        let caller = program.definition(self.template(self.candidate(parent).template).definition);
        let binding = caller.body.calls.get(call.0 as usize).and_then(|site| {
            site.bindings
                .iter()
                .find(|b| b.definition == template.definition)
        });
        let Some(binding) = binding else {
            return Vec::new();
        };
        binding
            .shape_args
            .iter()
            .filter(|(name, _)| template.dynamic.contains(name))
            .filter_map(|(name, extent)| extent.semantic().map(|sym| (name.clone(), sym.clone())))
            .collect()
    }

    /// The root regions of a candidate whose body is nothing else: every statement of its
    /// root block is a statement-position `parallel` region without `merge` and without a
    /// result, in authored order (one element for a single-region kernel). `None` for any
    /// other body shape. A backend may treat a `UnitKind::Call` unit like these `Region`
    /// units when every candidate of the occurrence answers `Some`.
    pub fn root_regions(
        &self,
        program: &Program,
        candidate: CandidateRef,
    ) -> Option<Vec<RegionId>> {
        let body = &program
            .definition(self.template(self.candidate(candidate).template).definition)
            .body;
        let regions: Option<Vec<RegionId>> = body
            .block
            .iter()
            .map(|s| match &s.kind {
                StmtKind::Region(r)
                    if r.mode == RegionMode::Parallel
                        && r.merge.is_none()
                        && r.result.is_none() =>
                {
                    Some(r.id)
                }
                _ => None,
            })
            .collect();
        regions.filter(|regions| !regions.is_empty())
    }

    /// The site governing `slice` of the candidate's body (`Rebind` chains resolved).
    pub fn slice_site(
        &self,
        program: &Program,
        candidate: CandidateRef,
        slice: SliceId,
    ) -> Option<SiteId> {
        let c = self.candidate(candidate);
        let body = &program
            .definition(self.template(c.template).definition)
            .body;
        let owner = owning_slice(body, slice)?;
        c.sites
            .iter()
            .copied()
            .find(|id| match &self.sites[id.0 as usize].kind {
                SiteKind::Width { slice, .. } | SiteKind::Parts { slice, .. } => *slice == owner,
            })
    }

    /// The site governing an axis extent seen from the candidate's body: a slice of the
    /// body, or a shape parameter bound structurally by an ancestor. `None` for a
    /// semantic extent.
    pub fn extent_site(
        &self,
        program: &Program,
        candidate: CandidateRef,
        extent: &Extent,
    ) -> Option<SiteId> {
        match extent {
            Extent::Structural(slice) => self.slice_site(program, candidate, *slice),
            Extent::Semantic(sym) => self
                .candidate(candidate)
                .structural
                .iter()
                .find(|(name, _)| *sym == Sym::param(name))
                .map(|(_, site)| site.0),
        }
    }
}

fn describe(family: &Family, id: OccurrenceId) -> String {
    format!(
        "occurrence {} (family #{})",
        id.0,
        family.occurrence(id).family
    )
}

pub fn validate(family: &Family, witness: &Witness) -> Result<(), String> {
    if family.occurrences.is_empty() {
        return Err("family has no entry occurrence".into());
    }
    let mut occurrences = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut pending = vec![OccurrenceId(0)];
    while let Some(id) = pending.pop() {
        let occurrence = family
            .occurrences
            .get(id.0 as usize)
            .ok_or_else(|| format!("family references missing occurrence {}", id.0))?;
        occurrences.insert(id);
        let choice = *witness
            .choices
            .get(&id)
            .ok_or_else(|| format!("{} is active and has no choice", describe(family, id)))?;
        let candidate = occurrence.candidates.get(choice as usize).ok_or_else(|| {
            format!(
                "{}: choice {choice} is out of range of its {} candidates",
                describe(family, id),
                occurrence.candidates.len()
            )
        })?;
        candidates.push(candidate);
        pending.extend(candidate.children.iter().rev());
    }
    if let Some(id) = witness.choices.keys().find(|id| !occurrences.contains(id)) {
        return Err(format!("occurrence {} is inactive and has a choice", id.0));
    }

    let sites: BTreeSet<SiteId> = candidates
        .iter()
        .flat_map(|c| c.sites.iter().copied())
        .collect();
    for id in &sites {
        let site = family
            .sites
            .get(id.0 as usize)
            .ok_or_else(|| format!("family references missing site {}", id.0))?;
        let value = *witness
            .sites
            .get(id)
            .ok_or_else(|| format!("site {} is active and has no value", id.0))?;
        if !(1..=site.extent).contains(&value) {
            return Err(format!(
                "site {}: value {value} is outside 1..={}",
                id.0, site.extent
            ));
        }
    }
    if let Some(id) = witness.sites.keys().find(|id| !sites.contains(id)) {
        return Err(format!("site {} is inactive and has a value", id.0));
    }
    for requirement in candidates.iter().flat_map(|c| &c.requirements) {
        let site = requirement.site();
        let value = *witness.sites.get(&site).ok_or_else(|| {
            format!(
                "{requirement:?} of an active candidate names inactive site {}",
                site.0
            )
        })?;
        if !requirement.holds(value) {
            return Err(format!(
                "site {} = {value} violates {requirement:?}",
                site.0
            ));
        }
    }

    for (refinement, refined) in &family.refinements {
        if let (Some(inner), Some(outer)) =
            (witness.sites.get(refinement), witness.sites.get(refined))
        {
            if *inner < 1 || outer % inner != 0 {
                return Err(format!(
                    "site {} = {inner} refines site {} = {outer} and does not divide it",
                    refinement.0, refined.0
                ));
            }
        }
    }

    let sequences: BTreeSet<SequenceId> = candidates
        .iter()
        .flat_map(|c| c.sequences.iter().copied())
        .collect();
    for id in &sequences {
        let sequence = family
            .sequences
            .get(id.0 as usize)
            .ok_or_else(|| format!("family references missing sequence {}", id.0))?;
        let cover = witness
            .covers
            .get(id)
            .ok_or_else(|| format!("sequence {} is active and has no cover", id.0))?;
        let mut next = 0u32;
        for &(start, end) in cover {
            if start != next || end <= start {
                return Err(format!("sequence {}: interval [{start}, {end}) does not continue the cover at unit {next}", id.0));
            }
            next = end;
        }
        if next as usize != sequence.units.len() {
            return Err(format!(
                "sequence {}: cover ends at unit {next} of {}",
                id.0,
                sequence.units.len()
            ));
        }
    }
    if let Some(id) = witness.covers.keys().find(|id| !sequences.contains(id)) {
        return Err(format!("sequence {} is inactive and has a cover", id.0));
    }
    Ok(())
}
