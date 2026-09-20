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

pub use estimate::{EstimateModel, IDENTITY, Totals};
pub use realize::BLOCK_THREADS;

use crate::execution::Launches;
use accounting::CudaAccounting;
use seismic_compiler::selection::mapping::{self, CostScope, Costs, Legality, ScopeCost};
use seismic_compiler::selection::quantity::{self, Quantity};
use seismic_compiler::selection::{
    Backend, Constraint, Factor, Interval, ResourceConstraint, ResourceTerm, SelectionError,
};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::family::{CandidateRef, Family, SiteId, Witness};
use seismic_lang::sir::Program;
use seismic_lang::sir::RegionMode;
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
        Limits {
            max_threads_per_block: 1024,
            max_grid_x: i32::MAX as u32,
            warp_size: 32,
            max_scratch_bytes: 130_663_231_488 / 4,
        }
    }

    /// Limits of a queried device. Scratch rule: a launch may hold at most one quarter of
    /// the device's global memory as invocation scratch.
    pub fn from_device(device: &crate::DeviceInfo) -> Self {
        Limits {
            max_threads_per_block: device.max_threads_per_block,
            max_grid_x: device.max_grid_x,
            warp_size: device.warp_size,
            max_scratch_bytes: device.global_memory_bytes / 4,
        }
    }

    pub(crate) fn launch(&self) -> crate::execution::Limits {
        crate::execution::Limits {
            max_threads_per_block: self.max_threads_per_block,
            max_grid_x: self.max_grid_x,
        }
    }

    fn target(&self) -> crate::target::TargetLimits {
        crate::target::TargetLimits {
            max_threads_per_block: self.max_threads_per_block,
            max_grid_x: self.max_grid_x,
            warp_size: self.warp_size,
            max_scratch_bytes: self.max_scratch_bytes,
        }
    }

    /// Work items one launch can address: a full grid row of full blocks of whole items.
    fn max_pieces(&self, lanes: u64) -> u64 {
        (u64::from(self.max_threads_per_block) / lanes.max(1))
            .saturating_mul(u64::from(self.max_grid_x))
    }
}

pub struct Cuda {
    limits: Limits,
    estimate: EstimateModel,
    target_profile: crate::target::TargetProfile,
    numerical_environment: String,
}

impl Cuda {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, SelectionError> {
        if limits.max_threads_per_block == 0
            || limits.max_grid_x == 0
            || limits.warp_size == 0
            || i64::try_from(limits.max_scratch_bytes).is_err()
        {
            return Err(SelectionError::IncompatibleComposition(format!(
                "CUDA needs positive block, grid and warp capacities; the device offers {} threads per block, {} blocks, {}-lane warps",
                limits.max_threads_per_block, limits.max_grid_x, limits.warp_size
            )));
        }
        estimate
            .validate()
            .map_err(SelectionError::AnalysisUnavailable)?;
        let target_profile = crate::target::TargetProfile::synthetic_baseline(limits.target());
        let numerical_environment = target_profile.fingerprint().to_owned();
        Ok(Cuda {
            limits,
            estimate,
            target_profile,
            numerical_environment,
        })
    }

    /// Queried limits; the driver exposes no throughput facts, so the estimate keeps its
    /// unmeasured defaults.
    pub fn from_device(device: &crate::DeviceInfo) -> Result<Self, SelectionError> {
        let mut backend = Cuda::new(Limits::from_device(device), EstimateModel::default())?;
        backend.target_profile = crate::target::TargetProfile::from_observation(
            crate::target::TargetObservation::driver(
                device.compute_capability,
            device.driver_version,
            )
            .map_err(|error| SelectionError::UnsupportedMapping(error.to_string()))?,
            backend.limits.target(),
        )
        .map_err(|error| SelectionError::UnsupportedMapping(error.to_string()))?;
        backend.numerical_environment = backend.target_profile.fingerprint().to_owned();
        Ok(backend)
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile {
        &self.target_profile
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
            Limit::Grid { pieces } => pieces
                .eval(site)
                .is_ok_and(|p| p <= limits.max_pieces(lanes)),
            Limit::Scratch { bits, pieces } => bits
                .eval(site)
                .and_then(|b| Ok((b, pieces.eval(site)?)))
                .is_ok_and(|(bits, pieces)| {
                // Per thread: its scratch and one four-byte status word.
                let threads = pieces.checked_mul(lanes);
                    threads
                        .and_then(|t| bits.div_ceil(8).checked_add(4)?.checked_mul(t))
                        .is_some_and(|bytes| bytes <= limits.max_scratch_bytes)
            }),
        }
    }

    /// A violated scratch limit raises the highest site of the launch's pieces (fewer threads
    /// hold scratch at once) and, failing that, lowers the highest site of the offending
    /// tiles; a violated grid limit raises the highest site of the launch.
    fn legality(
        self,
        limits: &Limits,
        lanes: u64,
        guard: Vec<CandidateRef>,
        reason: String,
    ) -> Legality {
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
                Limit::Scratch { bits, pieces } => bits.eval(site).and_then(|b| {
                    Ok(format!(
                        "{} bytes per thread over {} pieces",
                        b.div_ceil(8),
                        pieces.eval(site)?
                    ))
                }),
            }),
            repairs: match &*limit {
                Limit::Grid { pieces } => vec![(vec![pieces.clone()], true)],
                Limit::Scratch { bits, pieces } => {
                    vec![(vec![pieces.clone()], true), (vec![bits.clone()], false)]
                }
            },
            reason,
        }
    }
}

impl Cuda {
    fn analysis<'a>(
        &self,
        program: &'a Program,
        family: &'a Family,
    ) -> Result<Analysis<'a>, SelectionError> {
        if family.target != TARGET {
            return Err(SelectionError::UnsupportedMapping(format!(
                "family of `{}` targets `{}`, not `{TARGET}`",
                family.entry, family.target
            )));
        }
        let analysis = Analysis::new(program, family, TARGET)?;
        if analysis.accounts.values().any(|a| a.ledger.participants)
            && self.limits.warp_size != WARP
        {
            return Err(SelectionError::UnsupportedMapping(format!(
                "`{}`: participant intrinsics are realized for {WARP}-lane warps; the device has {}",
                family.entry, self.limits.warp_size
            )));
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
        if analysis.accounts.values().any(|a| a.ledger.participants) {
            u64::from(WARP)
        } else {
            1
        }
    }

    fn legalities(&self, analysis: &Analysis<'_>, seed_guidance: bool) -> Vec<Legality> {
        let participants: Vec<CandidateRef> = analysis
            .accounts
            .iter()
            .filter_map(|(&candidate, account)| account.ledger.participants.then_some(candidate))
            .collect();
        let mut out = Vec::new();
        for (&candidate, account) in &analysis.accounts {
            let name = &analysis.bounds[&candidate].definition.name;
            for launch in account
                .launches
                .iter()
                .filter(|l| l.mode == RegionMode::Parallel)
            {
                let reason = format!(
                    "`{name}` region#{}: pieces fit a grid row of {} blocks of {} threads",
                    launch.region.0, self.limits.max_grid_x, self.limits.max_threads_per_block
                );
                out.push(
                    Limit::Grid {
                        pieces: launch.pieces.clone(),
                    }
                    .legality(
                        &self.limits,
                        1,
                        account.context.guards.clone(),
                        reason.clone(),
                    ),
                );
                for participant in &participants {
                    let mut guard = account.context.guards.clone();
                    guard.extend(
                        analysis.accounts[participant]
                            .context
                            .guards
                            .iter()
                            .copied(),
                    );
                    out.push(
                        Limit::Grid {
                            pieces: launch.pieces.clone(),
                        }
                        .legality(
                            &self.limits,
                            u64::from(WARP),
                            guard,
                            format!("{reason} when CUDA subgroup participation is selected"),
                        ),
                    );
                }
            }
            if !seed_guidance {
                continue;
            }
            // Scratch of one kernel: this candidate's allocations and its ancestors' in the same
            // launch (their selection is implied by the guard chain; sibling occurrences are
            // separate occurrences and are covered by the exact recheck in `realize`).
            let mut launches: BTreeMap<Option<(CandidateRef, usize)>, Quantity> = BTreeMap::new();
            for tile in &account.ledger.tiles {
                if tile.launch.is_some() || account.context.scope.invocation {
                    launches
                        .entry(tile.launch)
                        .or_insert_with(|| tile.pieces.clone());
                }
            }
            for (launch, pieces) in launches {
                let bits: Vec<Quantity> = account
                    .context
                    .guards
                    .iter()
                    .filter_map(|g| analysis.accounts.get(g))
                    .flat_map(|a| {
                        a.ledger
                            .tiles
                            .iter()
                            .filter(|t| t.launch == launch)
                            .map(|t| t.scratch_bits.clone())
                    })
                    .collect();
                let reason = format!(
                    "`{name}`: scratch of every thread of one launch fits {} bytes",
                    self.limits.max_scratch_bytes
                );
                out.push(
                    Limit::Scratch {
                        bits: Quantity::Sum(bits),
                        pieces,
                    }
                    .legality(
                        &self.limits,
                        self.lanes(analysis),
                        account.context.guards.clone(),
                        reason,
                    ),
                );
            }
        }
        out
    }

    /// Exact additive scratch of every co-selected occurrence, grouped by the launch in which
    /// its allocation is live. The older per-candidate scratch legality remains useful repair
    /// guidance for the constructive seed; this is the authoritative pre-selection capacity.
    fn scratch_resources(&self, analysis: &Analysis<'_>) -> Vec<ResourceConstraint> {
        type LaunchKey = (Option<(CandidateRef, usize)>, CandidateRef);
        let mut launches: BTreeMap<LaunchKey, Vec<(Vec<CandidateRef>, Quantity, Quantity)>> =
            BTreeMap::new();
        for (&candidate, account) in &analysis.accounts {
            for tile in &account.ledger.tiles {
                if tile.launch.is_some() || account.context.scope.invocation {
                    let owner = tile.launch.map_or(candidate, |(owner, _)| owner);
                    launches.entry((tile.launch, owner)).or_default().push((
                        account.context.guards.clone(),
                        tile.scratch_bits.clone(),
                        tile.pieces.clone(),
                    ));
                }
            }
        }

        let participants: Vec<CandidateRef> = analysis
            .accounts
            .iter()
            .filter_map(|(&candidate, account)| account.ledger.participants.then_some(candidate))
            .collect();
        let scenarios = std::iter::once((1, None)).chain(
            participants
                .into_iter()
                .map(|candidate| (u64::from(WARP), Some(candidate))),
        );
        let mut resources = Vec::new();
        for (lanes, participant) in scenarios {
            for (&(launch, owner), allocations) in &launches {
                let fallback_pieces = allocations.first().map(|(_, _, pieces)| pieces.clone());
                let mut terms = Vec::with_capacity(allocations.len() + 1);
                for (base_guard, bits, pieces) in allocations {
                    let mut guard = base_guard.clone();
                    if let Some(participant) = participant {
                        guard.push(participant);
                    }
                    let scope = quantity::scope([bits, pieces]);
                    let table = scope.clone();
                    let (bits, pieces) = (bits.clone(), pieces.clone());
                    terms.push(ResourceTerm {
                        guard,
                        scope,
                        amount: Box::new(move |values| {
                            let site = quantity::lookup(&table, values);
                            bits.eval(&site)?
                                .div_ceil(8)
                                .checked_mul(pieces.eval(&site)?)
                                .and_then(|bytes| bytes.checked_mul(lanes))
                                .ok_or_else(|| "CUDA invocation scratch bytes overflow u64".into())
                        }),
                    });
                }

                // One four-byte status word per participating thread, independent of the
                // number of allocations in the launch.
                let pieces = launch
                    .and_then(|(candidate, ordinal)| {
                        analysis
                            .accounts
                            .get(&candidate)
                            .and_then(|account| account.launches.get(ordinal))
                            .map(|launch| launch.pieces.clone())
                    })
                    .or(fallback_pieces)
                    .unwrap_or_else(Quantity::one);
                let scope = quantity::scope([&pieces]);
                let table = scope.clone();
                let mut guard = analysis
                    .accounts
                    .get(&owner)
                    .map(|account| account.context.guards.clone())
                    .unwrap_or_else(|| vec![owner]);
                if let Some(participant) = participant {
                    guard.push(participant);
                }
                terms.push(ResourceTerm {
                    guard,
                    scope,
                    amount: Box::new(move |values| {
                        pieces
                            .eval(&quantity::lookup(&table, values))?
                            .checked_mul(lanes)
                            .and_then(|threads| threads.checked_mul(4))
                            .ok_or_else(|| "CUDA status bytes overflow u64".into())
                    }),
                });
                resources.push(ResourceConstraint { terms, capacity: self.limits.max_scratch_bytes, reason: format!("CUDA launch {:?} co-selected invocation scratch fits {} bytes at {} lane(s)", launch, self.limits.max_scratch_bytes, lanes) });
            }
        }
        resources
    }
}

impl Costs<CudaAccounting> for EstimateModel {
    fn scope_label(&self) -> String {
        "unqualified estimate (launches + serial work per piece over concurrent pieces + memory traffic; runtime extents at their static upper bound)".into()
    }

    fn scope(&self, scope: CostScope<'_, '_, CudaAccounting>) -> ScopeCost {
        let (launches, work, pieces, model) = (
            scope.launches,
            scope.work.clone(),
            scope.pieces.clone(),
            self.clone(),
        );
        ScopeCost {
            reads: Vec::new(),
            cost: Box::new(move |site| {
                model.scope_ns(launches, &work.totals(site)?, pieces.eval(site)?)
            }),
        }
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

    fn capability_fingerprint(&self) -> String {
        self.target_profile.fingerprint().to_owned()
    }

    fn supports_intrinsic(
        &self,
        intrinsic: &seismic_lang::sir::IntrinsicUse,
    ) -> Result<(), String> {
        self.target_profile.supports_intrinsic(intrinsic)
    }

    fn estimate_model(&self) -> String {
        IDENTITY.into()
    }

    fn numerical_environment(&self) -> String {
        self.numerical_environment.clone()
    }

    fn bind_structure(
        &self,
        program: &Program,
        family: &Family,
    ) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError> {
        self.analysis(program, family)?;
        mapping::domains(program, family)
    }

    fn constraints(
        &self,
        program: &Program,
        family: &Family,
    ) -> Result<Vec<Constraint>, SelectionError> {
        mapping::constraints(
            family,
            self.legalities(&self.analysis(program, family)?, false),
        )
    }

    fn resources(
        &self,
        program: &Program,
        family: &Family,
    ) -> Result<Vec<ResourceConstraint>, SelectionError> {
        Ok(self.scratch_resources(&self.analysis(program, family)?))
    }

    fn intervals(
        &self,
        program: &Program,
        family: &Family,
    ) -> Result<Vec<Interval>, SelectionError> {
        mapping::intervals(&self.analysis(program, family)?, family)
    }

    /// Additive estimate (`cuda-estimate-unqualified-v0`; every coefficient unmeasured) over
    /// the shared factor skeleton.
    fn factors(
        &self,
        program: &Program,
        family: &Family,
        intervals: &[Interval],
    ) -> Result<Vec<Factor>, SelectionError> {
        mapping::factors(
            &self.analysis(program, family)?,
            family,
            intervals,
            &self.estimate,
        )
    }

    /// Seed policy (`mapping::seed`), with `lower ... for cuda` bodies first. Root `parallel`
    /// binders take the smallest admissible widths whose piece count does not exceed the
    /// threads the estimate runs concurrently (within the grid limit).
    fn seed(
        &self,
        program: &Program,
        family: &Family,
        domains: &BTreeMap<SiteId, Vec<i64>>,
        intervals: &[Interval],
    ) -> Result<Witness, SelectionError> {
        let analysis = self.analysis(program, family)?;
        let concurrent = self
            .estimate
            .concurrent_threads
            .clamp(1, self.limits.max_pieces(self.lanes(&analysis)).max(1));
        mapping::seed(
            program,
            family,
            TARGET,
            &analysis,
            domains,
            intervals,
            concurrent,
            self.legalities(&analysis, true),
        )
    }

    fn realize(
        &self,
        lowered: LoweredIr,
        _family: &Family,
        _witness: &Witness,
    ) -> Result<Launches, SelectionError> {
        if lowered.backend != TARGET {
            return Err(SelectionError::Reconstruction(format!(
                "`{}` was instantiated for `{}`, not `{TARGET}`",
                lowered.name, lowered.backend
            )));
        }
        realize::realize(&self.limits, &self.target_profile, &lowered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::family::{OccurrenceId, Site, SiteKind};
    use seismic_lang::intrinsics::{CapabilityId, IntrinsicId, Operation};
    use seismic_lang::sir::IntrinsicUse;
    use seismic_lang::types::{DType, Ty};
    use seismic_lang::types::{RegionId, SliceId};

    fn intrinsic(operation: Operation, arguments: Vec<Ty>, result: Ty) -> IntrinsicUse {
        IntrinsicUse {
            id: IntrinsicId {
                capability: CapabilityId::new(
                    "cuda",
                    if matches!(
                        operation,
                        Operation::MatrixMatmul | Operation::MatrixMatmulAdd
                    ) {
                        "matrix"
                    } else {
                        "subgroup"
                    },
                ),
                name: operation.name().into(),
            },
            operation,
            arguments,
            result,
        }
    }

    #[test]
    fn capability_support_matches_realized_cuda_abis() {
        let backend = Cuda::new(Limits::gb10(), EstimateModel::default()).unwrap();
        let f32 = Ty::Scalar(DType::F32);
        assert!(
            backend
                .supports_intrinsic(&intrinsic(
                    Operation::SimdSum,
                    vec![f32.clone()],
                    f32.clone()
                ))
                .is_ok()
        );
        assert!(
            backend
                .supports_intrinsic(&intrinsic(
                    Operation::SimdMax,
                    vec![f32.clone()],
                    f32.clone()
                ))
                .is_err()
        );
        assert!(
            backend
                .supports_intrinsic(&intrinsic(
                    Operation::ShuffleIndex,
                    vec![Ty::Scalar(DType::F16), Ty::Scalar(DType::I32)],
                    Ty::Scalar(DType::F16)
                ))
                .is_err()
        );
        let matrix = intrinsic(Operation::MatrixMatmul, Vec::new(), Ty::Void);
        assert!(
            backend
                .supports_intrinsic(&matrix)
                .unwrap_err()
                .contains("BackendNotImplemented")
        );
    }

    #[test]
    fn width_domain_is_structured_divisors() {
        let owner = CandidateRef {
            occurrence: OccurrenceId(0),
            candidate: 0,
        };
        let site = Site {
            id: SiteId(0),
            owner,
            kind: SiteKind::Width {
                region: RegionId(0),
                slice: SliceId(0),
            },
            extent: 2560,
        };
        assert_eq!(
            mapping::domain(&site, &[], &[]),
            vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2560]
        );
        let parts = Site {
            kind: SiteKind::Parts {
                region: RegionId(0),
                slice: SliceId(0),
            },
            extent: 5,
            ..site
        };
        assert_eq!(mapping::domain(&parts, &[], &[]), vec![1, 2, 4]);
    }

    #[test]
    fn block_rule_covers_the_work_with_whole_items() {
        let limits = Limits::gb10();
        assert_eq!(realize::threads_per_block(&limits, 1, 1), Ok(1));
        assert_eq!(realize::threads_per_block(&limits, 100, 1), Ok(100));
        assert_eq!(realize::threads_per_block(&limits, 100_000, 1), Ok(256));
        assert_eq!(realize::threads_per_block(&limits, 100, 32), Ok(256));
        let narrow = Limits {
            max_grid_x: 4,
            ..Limits::gb10()
        };
        assert_eq!(realize::threads_per_block(&narrow, 4000, 1), Ok(1000));
        assert!(realize::threads_per_block(&narrow, 5000, 1).is_err());
    }
}
