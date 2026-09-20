//! Authored execution-unit normalization and family queries used by logical specialization.

use super::{CandidateRef, Family, Requirement, SiteId, SiteKind};
use crate::sir::RegionMode;
use crate::sir::{Body, Program, SliceParent, StmtKind};
use crate::sym::Sym;
use crate::types::{Elem, Extent, RegionId, SliceId};
use std::collections::BTreeMap;

mod units;
pub(super) use units::{sequences, BlockUnits};

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
            Self::Multiple { site, .. }
            | Self::AtLeast { site, .. }
            | Self::AtMost { site, .. }
            | Self::Equal { site, .. }
            | Self::Divides { site, .. } => *site,
        }
    }

    pub fn holds(&self, value: i64) -> bool {
        match self {
            Self::Multiple { unit, .. } => *unit != 0 && value % unit == 0,
            Self::AtLeast { value: bound, .. } => value >= *bound,
            Self::AtMost { value: bound, .. } => value <= *bound,
            Self::Equal {
                value: required, ..
            } => value == *required,
            Self::Divides { extent, .. } => value != 0 && extent % value == 0,
        }
    }
}

impl Family {
    pub fn shapes(&self, candidate: CandidateRef) -> &BTreeMap<String, i64> {
        &self.template(self.candidate(candidate).template).shapes
    }

    pub fn elems(&self, candidate: CandidateRef) -> &BTreeMap<String, Elem> {
        &self.template(self.candidate(candidate).template).elems
    }

    pub fn structural_site(&self, candidate: CandidateRef, param: &str) -> Option<SiteId> {
        self.candidate(candidate)
            .structural
            .iter()
            .find(|(name, _)| name == param)
            .map(|(_, site)| site.0)
    }

    pub fn dynamic_args(&self, program: &Program, candidate: CandidateRef) -> Vec<(String, Sym)> {
        let occurrence = self.occurrence(candidate.occurrence);
        let template = self.template(self.candidate(candidate).template);
        let (Some(parent), Some(call)) = (occurrence.parent, occurrence.call) else {
            return Vec::new();
        };
        let caller = program.definition(self.template(self.candidate(parent).template).definition);
        let Some(binding) = caller.body.calls.get(call.0 as usize).and_then(|site| {
            site.bindings
                .iter()
                .find(|binding| binding.definition == template.definition)
        }) else {
            return Vec::new();
        };
        binding
            .shape_args
            .iter()
            .filter(|(name, _)| template.dynamic.contains(name))
            .filter_map(|(name, extent)| {
                extent.semantic().map(|value| (name.clone(), value.clone()))
            })
            .collect()
    }

    pub fn root_regions(
        &self,
        program: &Program,
        candidate: CandidateRef,
    ) -> Option<Vec<RegionId>> {
        let body = &program
            .definition(self.template(self.candidate(candidate).template).definition)
            .body;
        let regions: Option<Vec<_>> = body
            .block
            .iter()
            .map(|statement| match &statement.kind {
                StmtKind::Region(region)
                    if region.mode == RegionMode::Parallel
                        && region.merge.is_none()
                        && region.result.is_none() =>
                {
                    Some(region.id)
                }
                _ => None,
            })
            .collect();
        regions.filter(|regions| !regions.is_empty())
    }

    pub fn slice_site(
        &self,
        program: &Program,
        candidate: CandidateRef,
        slice: SliceId,
    ) -> Option<SiteId> {
        let selected = self.candidate(candidate);
        let body = &program
            .definition(self.template(selected.template).definition)
            .body;
        let owner = owning_slice(body, slice)?;
        selected
            .sites
            .iter()
            .copied()
            .find(|id| match &self.sites[id.0 as usize].kind {
                SiteKind::Width { slice, .. } | SiteKind::Parts { slice, .. } => *slice == owner,
            })
    }

    pub fn extent_site(
        &self,
        program: &Program,
        candidate: CandidateRef,
        extent: &Extent,
    ) -> Option<SiteId> {
        match extent {
            Extent::Structural(slice) => self.slice_site(program, candidate, *slice),
            Extent::Semantic(symbol) => self
                .candidate(candidate)
                .structural
                .iter()
                .find(|(name, _)| *symbol == Sym::param(name))
                .map(|(_, site)| site.0),
        }
    }
}
