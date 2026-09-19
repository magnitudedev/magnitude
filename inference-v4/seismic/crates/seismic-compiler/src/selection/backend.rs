//! What a backend contributes to joint selection. Every hook is a deterministic
//! function of the family (and, for `realize`, the witness): no hook may search,
//! choose among alternatives, or filter by estimated profitability.
use super::SelectionError;
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::family::{CandidateRef, Family, SequenceId, SiteId, Witness};
use seismic_lang::sir::Program;
use std::collections::BTreeMap;

/// A legal contiguous fusion group of `sequence` units `[start, end)` with one prescribed
/// realization. Singletons (separate execution) must be listed when supported.
#[derive(Clone, Debug)]
pub struct Interval {
    pub sequence: SequenceId,
    pub start: u32,
    pub end: u32,
    /// Child implementation selections this realization's interfaces depend on.
    pub requires: Vec<CandidateRef>,
    /// Site pairs that must be equal when this interval is selected (common geometry).
    pub equal_sites: Vec<(SiteId, SiteId)>,
}

/// Index into the backend's interval list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntervalRef(pub u32);

/// A hard legality relation among site values (capacity, alignment, participant limits),
/// active when every `guard` candidate is selected. Tabulated by the exporter over the
/// product of the scope's domains, so scopes must be small and domains finite.
pub struct Constraint {
    pub guard: Vec<CandidateRef>,
    pub scope: Vec<SiteId>,
    pub holds: Box<dyn Fn(&[i64]) -> bool + Send + Sync>,
    pub reason: String,
}

/// One local term of the estimated execution cost. It depends on exactly `scope`, is
/// active when every `guard` candidate and every `intervals` member is selected, and adds
/// to the objective. Units are the backend's estimate units (nanoseconds).
pub struct Factor {
    pub guard: Vec<CandidateRef>,
    pub intervals: Vec<IntervalRef>,
    pub scope: Vec<SiteId>,
    pub cost: Box<dyn Fn(&[i64]) -> Result<u64, String> + Send + Sync>,
    pub label: String,
}

pub trait Backend {
    type Execution;

    fn target(&self) -> &'static str;

    /// Identity of the estimate model behind `factors`.
    fn estimate_model(&self) -> String;

    /// Identity of the native numerical environment for evidence reuse. This includes the
    /// target's relevant device/compiler facts and is deliberately independent of the
    /// performance estimate identity.
    fn numerical_environment(&self) -> String;

    /// Bind prescribed structural mappings for every candidate body. Fails with
    /// `UnsupportedMapping` when a reachable structure has no mapping; returns the finite
    /// value domain of every site.
    fn bind_structure(&self, program: &Program, family: &Family) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError>;

    fn constraints(&self, program: &Program, family: &Family) -> Result<Vec<Constraint>, SelectionError>;

    /// Every legal contiguous interval of every sequence (spec section 11.3), without
    /// profitability filtering.
    fn intervals(&self, program: &Program, family: &Family) -> Result<Vec<Interval>, SelectionError>;

    fn factors(&self, program: &Program, family: &Family, intervals: &[Interval]) -> Result<Vec<Factor>, SelectionError>;

    /// Constructive complete seed (spec section 10.4): search initialization only.
    fn seed(&self, program: &Program, family: &Family, domains: &BTreeMap<SiteId, Vec<i64>>, intervals: &[Interval]) -> Result<Witness, SelectionError>;

    /// Realize the instantiated execution of `witness`. Applies only deterministic mapping
    /// rules (storage, allocation, transfers, unrolling); returns one execution or an error.
    fn realize(&self, lowered: LoweredIr, family: &Family, witness: &Witness) -> Result<Self::Execution, SelectionError>;
}
