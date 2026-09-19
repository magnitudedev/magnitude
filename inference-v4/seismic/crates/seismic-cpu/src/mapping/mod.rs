//! The CPU's side of joint selection.
//!
//! The backend contributes site domains, hard legality, legal fusion intervals, local
//! unqualified-estimate factors, a constructive seed and `realize`. Every former backend
//! decision is fixed by one deterministic rule; no hook searches or ranks by profitability.
//! Coverage is scalar: every candidate body is realized through the shared scalar
//! realization and compiled by Cranelift. Vector microkernel lowerings do not exist yet.
mod accounting;
mod estimate;
mod realize;

pub use estimate::{EstimateModel, Totals, IDENTITY};
pub use realize::Execution;

use accounting::CpuAccounting;
use seismic_compiler::selection::mapping::{self, CostScope, Costs, Legality, ScopeCost};
use seismic_compiler::selection::quantity::Quantity;
use seismic_compiler::selection::{Backend, Constraint, Factor, Interval, SelectionError};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::family::{CandidateRef, Family, SiteId, Witness};
use seismic_lang::sir::Program;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

type Analysis<'a> = seismic_compiler::selection::structure::Analysis<'a, CpuAccounting>;
pub use seismic_compiler::selection::mapping::{DOMAIN_VALUES, MAX_PARTS};

pub const TARGET: &str = "cpu";
/// The seed gives a root `parallel` phase about this many pieces per worker, so that uneven
/// pieces still balance across the pool.
pub const SEED_PIECES_PER_WORKER: u64 = 4;
/// Scratch bytes one worker offers a phase. Every worker of the device's pool holds one
/// scratch allocation that the phases of all kernels reuse; a phase's tiles, region results
/// and materialized snapshots are bump-allocated in it without reuse, so their sum is a hard
/// capacity of the phase.
pub const SCRATCH_BYTES: u64 = 64 << 20;

/// Host facts the mapping depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Worker threads that execute the pieces of a root `parallel` phase.
    pub workers: u64,
    pub max_scratch_bytes: u64,
}

impl Limits {
    pub fn host(workers: u64) -> Self {
        Limits { workers, max_scratch_bytes: SCRATCH_BYTES }
    }
}

pub struct Cpu {
    limits: Limits,
    estimate: EstimateModel,
}

impl Cpu {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, SelectionError> {
        if limits.workers == 0 || i64::try_from(limits.max_scratch_bytes).is_err() {
            return Err(SelectionError::IncompatibleComposition(format!("the CPU mapping needs at least one worker; the host offers {}", limits.workers)));
        }
        estimate.validate().map_err(SelectionError::AnalysisUnavailable)?;
        Ok(Cpu { limits, estimate })
    }

    /// The mapping for a pool of `workers` threads, with the unqualified estimate.
    pub fn host(workers: u64) -> Result<Self, SelectionError> {
        Cpu::new(Limits::host(workers), EstimateModel::unqualified(workers))
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }
}

impl Cpu {
    fn analysis<'a>(&self, program: &'a Program, family: &'a Family) -> Result<Analysis<'a>, SelectionError> {
        if family.target != TARGET {
            return Err(SelectionError::UnsupportedMapping(format!("family of `{}` targets `{}`, not `{TARGET}`", family.entry, family.target)));
        }
        Analysis::new(program, family, TARGET)
    }

    /// Scratch of one phase, as far as one candidate chain determines it: the tiles of the
    /// candidate and of its ancestors that are declared in the same phase. Sibling calls
    /// inlined into the phase are separate occurrences; `realize` rechecks the whole phase.
    fn legalities(&self, analysis: &Analysis<'_>) -> Vec<Legality> {
        let mut out = Vec::new();
        for (&candidate, account) in &analysis.accounts {
            let name = &analysis.bounds[&candidate].definition.name;
            let mut launches: BTreeSet<Option<(CandidateRef, usize)>> = account.ledger.tiles.iter().map(|t| t.launch).collect();
            launches.retain(|launch| launch.is_some() || account.context.scope.invocation);
            for launch in launches {
                let bits: Vec<Quantity> = account.context.guards.iter().filter_map(|g| analysis.accounts.get(g))
                    .flat_map(|a| a.ledger.tiles.iter().filter(|t| t.launch == launch).map(|t| t.bits.clone())).collect();
                // The one hard limit: scratch bytes one worker holds for one phase. A violation
                // lowers the highest site of the offending tiles.
                let (bits, limit) = (Quantity::Sum(bits), self.limits.max_scratch_bytes);
                let (held, needed) = (bits.clone(), bits.clone());
                out.push(Legality {
                    guard: account.context.guards.clone(),
                    reads: vec![bits.clone()],
                    holds: Arc::new(move |site| held.eval(site).is_ok_and(|bits| bits.div_ceil(8) <= limit)),
                    needed: Arc::new(move |site| needed.eval(site).map(|bits| format!("{} bytes", bits.div_ceil(8)))),
                    repairs: vec![(vec![bits], false)],
                    reason: format!("`{name}`: scratch tiles of one phase fit {} bytes per worker", self.limits.max_scratch_bytes),
                });
            }
        }
        out
    }

}

impl Costs<CpuAccounting> for EstimateModel {
    fn scope_label(&self) -> String {
        "unqualified estimate (phases + pieces / workers + operations / (rate x workers) + bytes / bandwidth; runtime extents at their static upper bound)".into()
    }

    fn scope(&self, scope: CostScope<'_, '_, CpuAccounting>) -> ScopeCost {
        let (phases, work, pieces, model) = (scope.launches, scope.work.clone(), scope.pieces.clone(), self.clone());
        ScopeCost { reads: Vec::new(), cost: Box::new(move |site| model.scope_ns(phases, &work.totals(site)?, pieces.eval(site)?)) }
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

impl Backend for Cpu {
    type Execution = Execution;

    fn target(&self) -> &'static str {
        TARGET
    }

    fn estimate_model(&self) -> String {
        IDENTITY.into()
    }

    fn numerical_environment(&self) -> String {
        format!(
            "seismic-cpu-v1:{}:workers={}:scratch={}",
            std::env::consts::ARCH,
            self.limits.workers,
            self.limits.max_scratch_bytes
        )
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

    fn factors(&self, program: &Program, family: &Family, intervals: &[Interval]) -> Result<Vec<Factor>, SelectionError> {
        mapping::factors(&self.analysis(program, family)?, family, intervals, &self.estimate)
    }

    /// Seed policy (`mapping::seed`), with `lower ... for cpu` bodies first. Root `parallel`
    /// binders take the smallest admissible widths whose piece count does not exceed
    /// `workers x SEED_PIECES_PER_WORKER`; a violated scratch limit lowers the highest site
    /// of the offending tiles.
    fn seed(&self, program: &Program, family: &Family, domains: &BTreeMap<SiteId, Vec<i64>>, intervals: &[Interval]) -> Result<Witness, SelectionError> {
        let analysis = self.analysis(program, family)?;
        let target = self.limits.workers.saturating_mul(SEED_PIECES_PER_WORKER);
        mapping::seed(program, family, TARGET, &analysis, domains, intervals, target, self.legalities(&analysis))
    }

    fn realize(&self, lowered: LoweredIr, _family: &Family, _witness: &Witness) -> Result<Execution, SelectionError> {
        realize::realize(&self.limits, lowered)
    }
}
