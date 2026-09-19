//! CUDA's side of joint selection (spec 8, 10-12; plan 4.2, R7, R8, R12, Wave 6B).
//!
//! The backend contributes site domains, hard legality, legal fusion intervals, local
//! unqualified-estimate factors, a constructive seed and `realize`. Former backend
//! decisions are fixed by one deterministic rule each (see `realize`); none is a solver
//! decision and no hook searches or ranks by profitability.
//!
//! Coverage is scalar. CUDA's intrinsic table exposes lane index, shuffle and sum only;
//! there are no matrix operations and no matrix coverage is claimed. Every portable body
//! runs as the shared scalar realization printed as PTX.
mod accounting;
mod estimate;
mod realize;

pub use estimate::{EstimateModel, Totals, IDENTITY};
pub use realize::BLOCK_THREADS;

use crate::execution::Launches;
use accounting::CudaAccounting;
use seismic_compiler::selection::mapping::{self, CostScope, Costs, Legality, ScopeCost};
use seismic_compiler::selection::quantity::Quantity;
use seismic_compiler::selection::{Backend, Constraint, Factor, Interval, SelectionError};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::family::{CandidateRef, Family, SiteId, Witness};
use seismic_lang::sir::Program;
use seismic_lang::syntax::ast::RegionMode;
use std::collections::BTreeMap;
use std::sync::Arc;

type Analysis<'a> = seismic_compiler::selection::structure::Analysis<'a, CudaAccounting>;
pub use seismic_compiler::selection::mapping::{DOMAIN_VALUES, MAX_PARTS};

pub const TARGET: &str = "cuda";
/// Lanes of one work item when a body names a participant intrinsic.
pub const WARP: u32 = 32;

/// Device limits the mapping turns into solver constraints and `realize` rechecks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Threads of one block (`CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, and the x block
    /// dimension: launches are one-dimensional).
    pub max_threads_per_block: u32,
    /// Blocks of one grid row (`CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`).
    pub max_grid_x: u32,
    /// Lanes of one warp. Participant intrinsics are realized for 32 only.
    pub warp_size: u32,
    /// Invocation-owned local memory of one launch: per-thread scratch (tiles, materialized
    /// snapshots) and status words, allocated in device global memory for every
    /// participating thread at once.
    pub max_scratch_bytes: u64,
}

impl Limits {
    /// NVIDIA GB10 (compute capability 12.1, driver 13.0) as queried on 2026-09-19: 1024
    /// threads per block, 2^31 - 1 blocks along x, 32-lane warps, 130,663,231,488 bytes of
    /// global memory (unified with the host). The scratch limit is this mapping's rule
    /// (`from_device`), one quarter of that memory; it is not a driver fact.
    pub fn gb10() -> Self {
        Limits { max_threads_per_block: 1024, max_grid_x: i32::MAX as u32, warp_size: 32, max_scratch_bytes: 130_663_231_488 / 4 }
    }

    /// Limits of a queried device. Scratch rule: a launch may hold at most one quarter of
    /// the device's global memory as invocation scratch.
    pub fn from_device(device: &crate::DeviceInfo) -> Self {
        Limits { max_threads_per_block: device.max_threads_per_block, max_grid_x: device.max_grid_x, warp_size: device.warp_size, max_scratch_bytes: device.global_memory_bytes / 4 }
    }

    pub(crate) fn launch(&self) -> crate::execution::Limits {
        crate::execution::Limits { max_threads_per_block: self.max_threads_per_block, max_grid_x: self.max_grid_x }
    }

    /// Work items one launch can address: a full grid row of full blocks of whole items.
    fn max_pieces(&self, lanes: u64) -> u64 {
        (u64::from(self.max_threads_per_block) / lanes.max(1)).saturating_mul(u64::from(self.max_grid_x))
    }
}

pub struct Cuda {
    limits: Limits,
    estimate: EstimateModel,
}

impl Cuda {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, SelectionError> {
        if limits.max_threads_per_block == 0 || limits.max_grid_x == 0 || limits.warp_size == 0 || i64::try_from(limits.max_scratch_bytes).is_err() {
            return Err(SelectionError::IncompatibleComposition(format!(
                "CUDA needs positive block, grid and warp capacities; the device offers {} threads per block, {} blocks, {}-lane warps",
                limits.max_threads_per_block, limits.max_grid_x, limits.warp_size
            )));
        }
        estimate.validate().map_err(SelectionError::AnalysisUnavailable)?;
        Ok(Cuda { limits, estimate })
    }

    /// Queried limits; the driver exposes no throughput facts, so the estimate keeps its
    /// unmeasured defaults.
    pub fn from_device(device: &crate::DeviceInfo) -> Result<Self, SelectionError> {
        Cuda::new(Limits::from_device(device), EstimateModel::default())
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }
}

/// A hard device limit over derived quantities.
enum Limit {
    /// Pieces of one root `parallel` launch fit one grid row of full blocks.
    Grid { pieces: Quantity },
    /// Scratch bits of one thread times the participating threads of the launch fit the
    /// invocation scratch limit.
    Scratch { bits: Quantity, pieces: Quantity },
}

impl Limit {
    fn holds(&self, limits: &Limits, lanes: u64, site: &dyn Fn(SiteId) -> Option<i64>) -> bool {
        match self {
            Limit::Grid { pieces } => pieces.eval(site).is_ok_and(|p| p <= limits.max_pieces(lanes)),
            Limit::Scratch { bits, pieces } => bits.eval(site).and_then(|b| Ok((b, pieces.eval(site)?))).is_ok_and(|(bits, pieces)| {
                // Per thread: its scratch and one four-byte status word.
                let threads = pieces.checked_mul(lanes);
                threads.and_then(|t| bits.div_ceil(8).checked_add(4)?.checked_mul(t)).is_some_and(|bytes| bytes <= limits.max_scratch_bytes)
            }),
        }
    }

    /// A violated scratch limit raises the highest site of the launch's pieces (fewer threads
    /// hold scratch at once) and, failing that, lowers the highest site of the offending
    /// tiles; a violated grid limit raises the highest site of the launch.
    fn legality(self, limits: &Limits, lanes: u64, guard: Vec<CandidateRef>, reason: String) -> Legality {
        let limit = Arc::new(self);
        let (limits, held, needed) = (limits.clone(), limit.clone(), limit.clone());
        Legality {
            guard,
            reads: match &*limit {
                Limit::Grid { pieces } => vec![pieces.clone()],
                Limit::Scratch { bits, pieces } => vec![bits.clone(), pieces.clone()],
            },
            holds: Arc::new(move |site| held.holds(&limits, lanes, site)),
            needed: Arc::new(move |site| match &*needed {
                Limit::Grid { pieces } => pieces.eval(site).map(|p| format!("{p} pieces")),
                Limit::Scratch { bits, pieces } => bits.eval(site).and_then(|b| Ok(format!("{} bytes per thread over {} pieces", b.div_ceil(8), pieces.eval(site)?))),
            }),
            repairs: match &*limit {
                Limit::Grid { pieces } => vec![(vec![pieces.clone()], true)],
                Limit::Scratch { bits, pieces } => vec![(vec![pieces.clone()], true), (vec![bits.clone()], false)],
            },
            reason,
        }
    }
}

impl Cuda {
    fn analysis<'a>(&self, program: &'a Program, family: &'a Family) -> Result<Analysis<'a>, SelectionError> {
        if family.target != TARGET {
            return Err(SelectionError::UnsupportedMapping(format!("family of `{}` targets `{}`, not `{TARGET}`", family.entry, family.target)));
        }
        let analysis = Analysis::new(program, family, TARGET)?;
        if analysis.accounts.values().any(|a| a.ledger.participants) && self.limits.warp_size != WARP {
            return Err(SelectionError::UnsupportedMapping(format!("`{}`: participant intrinsics are realized for {WARP}-lane warps; the device has {}", family.entry, self.limits.warp_size)));
        }
        Ok(analysis)
    }

    /// Lanes per piece the limits are stated for. Participation is a property of the whole
    /// instantiated entry (one kernel per launch inlines every selected candidate), which a
    /// guard chain does not determine, so a family in which any candidate names a
    /// participant intrinsic is limited at the warp width throughout. That is the exact
    /// value whenever such a candidate is selected and a conservative one otherwise;
    /// `realize` rechecks the realized launch exactly.
    fn lanes(&self, analysis: &Analysis<'_>) -> u64 {
        if analysis.accounts.values().any(|a| a.ledger.participants) { u64::from(WARP) } else { 1 }
    }

    fn legalities(&self, analysis: &Analysis<'_>) -> Vec<Legality> {
        let lanes = self.lanes(analysis);
        let mut out = Vec::new();
        for (&candidate, account) in &analysis.accounts {
            let name = &analysis.bounds[&candidate].definition.name;
            for launch in account.launches.iter().filter(|l| l.mode == RegionMode::Parallel) {
                let reason = format!("`{name}` region#{}: pieces fit a grid row of {} blocks of {} threads", launch.region.0, self.limits.max_grid_x, self.limits.max_threads_per_block);
                out.push(Limit::Grid { pieces: launch.pieces.clone() }.legality(&self.limits, lanes, account.context.guards.clone(), reason));
            }
            // Scratch of one kernel: this candidate's allocations and its ancestors' in the same
            // launch (their selection is implied by the guard chain; sibling occurrences are
            // separate occurrences and are covered by the exact recheck in `realize`).
            let mut launches: BTreeMap<Option<(CandidateRef, usize)>, Quantity> = BTreeMap::new();
            for tile in &account.ledger.tiles {
                if tile.launch.is_some() || account.context.scope.invocation {
                    launches.entry(tile.launch).or_insert_with(|| tile.pieces.clone());
                }
            }
            for (launch, pieces) in launches {
                let bits: Vec<Quantity> = account.context.guards.iter().filter_map(|g| analysis.accounts.get(g))
                    .flat_map(|a| a.ledger.tiles.iter().filter(|t| t.launch == launch).map(|t| t.scratch_bits.clone())).collect();
                let reason = format!("`{name}`: scratch of every thread of one launch fits {} bytes", self.limits.max_scratch_bytes);
                out.push(Limit::Scratch { bits: Quantity::Sum(bits), pieces }.legality(&self.limits, lanes, account.context.guards.clone(), reason));
            }
        }
        out
    }

}

impl Costs<CudaAccounting> for EstimateModel {
    fn scope_label(&self) -> String {
        "unqualified estimate (launches + serial work per piece over concurrent pieces + memory traffic; runtime extents at their static upper bound)".into()
    }

    fn scope(&self, scope: CostScope<'_, '_, CudaAccounting>) -> ScopeCost {
        let (launches, work, pieces, model) = (scope.launches, scope.work.clone(), scope.pieces.clone(), self.clone());
        ScopeCost { reads: Vec::new(), cost: Box::new(move |site| model.scope_ns(launches, &work.totals(site)?, pieces.eval(site)?)) }
    }

    fn launch(&self) -> Box<dyn Fn() -> Result<u64, String> + Send + Sync> {
        let model = self.clone();
        Box::new(move || model.scope_ns(1, &Totals::default(), 1))
    }

    fn materialization(&self) -> Box<dyn Fn(u64) -> Result<u64, String> + Send + Sync> {
        let model = self.clone();
        Box::new(move |bits| model.materialization_ns(bits))
    }
}

impl Backend for Cuda {
    type Execution = Launches;

    fn target(&self) -> &'static str {
        TARGET
    }

    fn estimate_model(&self) -> String {
        IDENTITY.into()
    }

    fn bind_structure(&self, program: &Program, family: &Family) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError> {
        self.analysis(program, family)?;
        mapping::domains(program, family)
    }

    fn constraints(&self, program: &Program, family: &Family) -> Result<Vec<Constraint>, SelectionError> {
        mapping::constraints(family, self.legalities(&self.analysis(program, family)?))
    }

    fn intervals(&self, program: &Program, family: &Family) -> Result<Vec<Interval>, SelectionError> {
        mapping::intervals(&self.analysis(program, family)?, family)
    }

    /// Additive estimate (`cuda-estimate-unqualified-v0`; every coefficient unmeasured) over
    /// the shared factor skeleton.
    fn factors(&self, program: &Program, family: &Family, intervals: &[Interval]) -> Result<Vec<Factor>, SelectionError> {
        mapping::factors(&self.analysis(program, family)?, family, intervals, &self.estimate)
    }

    /// Seed policy (`mapping::seed`), with `lower ... for cuda` bodies first. Root `parallel`
    /// binders take the smallest admissible widths whose piece count does not exceed the
    /// threads the estimate runs concurrently (within the grid limit).
    fn seed(&self, program: &Program, family: &Family, domains: &BTreeMap<SiteId, Vec<i64>>, intervals: &[Interval]) -> Result<Witness, SelectionError> {
        let analysis = self.analysis(program, family)?;
        let concurrent = self.estimate.concurrent_threads.clamp(1, self.limits.max_pieces(self.lanes(&analysis)).max(1));
        mapping::seed(program, family, TARGET, &analysis, domains, intervals, concurrent, self.legalities(&analysis))
    }

    fn realize(&self, lowered: LoweredIr, _family: &Family, _witness: &Witness) -> Result<Launches, SelectionError> {
        if lowered.backend != TARGET {
            return Err(SelectionError::Reconstruction(format!("`{}` was instantiated for `{}`, not `{TARGET}`", lowered.name, lowered.backend)));
        }
        realize::realize(&self.limits, &lowered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::family::{OccurrenceId, Site, SiteKind};
    use seismic_lang::types::{RegionId, SliceId};

    #[test]
    fn width_domain_is_structured_divisors() {
        let owner = CandidateRef { occurrence: OccurrenceId(0), candidate: 0 };
        let site = Site { id: SiteId(0), owner, kind: SiteKind::Width { region: RegionId(0), slice: SliceId(0) }, extent: 2560 };
        assert_eq!(mapping::domain(&site, &[], &[]), vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2560]);
        let parts = Site { kind: SiteKind::Parts { region: RegionId(0), slice: SliceId(0) }, extent: 5, ..site };
        assert_eq!(mapping::domain(&parts, &[], &[]), vec![1, 2, 4]);
    }

    #[test]
    fn block_rule_covers_the_work_with_whole_items() {
        let limits = Limits::gb10();
        assert_eq!(realize::threads_per_block(&limits, 1, 1), Ok(1));
        assert_eq!(realize::threads_per_block(&limits, 100, 1), Ok(100));
        assert_eq!(realize::threads_per_block(&limits, 100_000, 1), Ok(256));
        assert_eq!(realize::threads_per_block(&limits, 100, 32), Ok(256));
        let narrow = Limits { max_grid_x: 4, ..Limits::gb10() };
        assert_eq!(realize::threads_per_block(&narrow, 4000, 1), Ok(1000));
        assert!(realize::threads_per_block(&narrow, 5000, 1).is_err());
    }
}
