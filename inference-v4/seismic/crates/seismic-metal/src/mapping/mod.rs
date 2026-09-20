//! Metal's side of joint selection (spec 8, 10-12; plan 4.2, R7, R8, R12).
//!
//! The backend contributes site domains, hard legality, legal fusion intervals, local
//! unqualified-estimate factors, a constructive seed and `realize`. Former backend
//! decisions are fixed here by one deterministic rule each (spec 8.5, 14.2); none is a
//! solver decision and no hook searches or ranks by profitability.
mod accounting;
mod estimate;
mod realize;

pub use estimate::{EstimateModel, Group, Totals, IDENTITY};

use crate::execution::{Execution, SUBGROUP};
use accounting::MetalAccounting;
use seismic_compiler::selection::mapping::{self, CostScope, Costs, Legality, ScopeCost};
use seismic_compiler::selection::quantity::{self, Quantity};
use seismic_compiler::selection::structure::Account;
use seismic_compiler::selection::{Backend, Constraint, Factor, Interval, ResourceConstraint, ResourceTerm, SelectionError};
use seismic_lang::exec::lowered_ir::LoweredIr;
use seismic_lang::family::{CandidateRef, Family, SiteId, Witness};
use seismic_lang::sir::{IntrinsicUse, Program, VarKind};
use seismic_lang::sir::RegionMode;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

type Analysis<'a> = seismic_compiler::selection::structure::Analysis<'a, MetalAccounting>;

pub const TARGET: &str = "metal";
/// Threadgroups of one launch grid.
pub const MAX_GROUPS: u64 = 65_535;
pub use seismic_compiler::selection::mapping::{DOMAIN_VALUES, MAX_PARTS};

/// Resource bounds used by mapping; each field retains queried or assumed provenance in the
/// originating target profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_threads_per_threadgroup: u64,
    pub max_threadgroup_bytes: u64,
    /// Compiler budget for declared thread-address-space array bytes of one kernel. Metal does
    /// not document or expose the per-thread stack, so this is never a queried device limit.
    pub max_private_bytes: u64,
}

/// Measured on Apple M4 Max (macOS 15): a kernel whose only private storage is one array
/// links up to 258,032 declared bytes and fails above, i.e. a 256 KiB thread stack less
/// the kernel's own frame. Half of it is left to compiler temporaries and register spills. This
/// conservative assumption is attached to the device profile with explicit provenance; it must
/// not be reported as hardware capability.
pub const CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES: u64 = 128 * 1024;

impl Limits {
    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Self {
        Limits { max_threads_per_threadgroup: device.max_threads_per_threadgroup, max_threadgroup_bytes: device.max_threadgroup_bytes, max_private_bytes: device.profile.private_storage_budget_bytes.value }
    }

    /// Pieces share a threadgroup only when one grid cannot hold them all.
    fn items_per_group(&self, pieces: u64) -> u64 {
        pieces.div_ceil(MAX_GROUPS).max(1)
    }

    fn max_items_per_group(&self) -> u64 {
        self.max_threads_per_threadgroup / SUBGROUP as u64
    }

    /// One threadgroup per piece. Realization groups pieces by the largest launch of the
    /// whole execution (`ceil(pieces / 65535)` pieces per threadgroup for every launch), so a
    /// launch beyond one grid would multiply the threadgroup memory and thread count of its
    /// sibling launches, which no per-launch constraint can see. Every width domain holds
    /// wider values, so this bound removes no entry from the family.
    fn max_pieces(&self) -> u64 {
        MAX_GROUPS
    }
}

pub struct Metal {
    limits: Limits,
    estimate: EstimateModel,
    target_profile: crate::target::TargetProfile,
    numerical_environment: String,
}

impl Metal {
    pub fn new(limits: Limits, estimate: EstimateModel) -> Result<Self, SelectionError> {
        if limits.max_items_per_group() == 0 || i64::try_from(limits.max_threads_per_threadgroup).is_err() || i64::try_from(limits.max_threadgroup_bytes).is_err() {
            return Err(SelectionError::IncompatibleComposition(format!(
                "Metal needs {SUBGROUP} threads per threadgroup; the device offers {}",
                limits.max_threads_per_threadgroup
            )));
        }
        estimate.validate().map_err(SelectionError::AnalysisUnavailable)?;
        let numerical_environment = format!(
            "seismic-metal-v1:synthetic:threads={}:threadgroup={}:private={}",
            limits.max_threads_per_threadgroup,
            limits.max_threadgroup_bytes,
            limits.max_private_bytes
        );
        let target_profile = crate::target::TargetProfile::synthetic(
            limits.max_threads_per_threadgroup,
            limits.max_threadgroup_bytes,
            u64::MAX,
            limits.max_private_bytes,
        );
        Ok(Metal { limits, estimate, target_profile, numerical_environment })
    }

    #[cfg(target_os = "macos")]
    pub fn from_device(device: &crate::runtime::DeviceInfo) -> Result<Self, SelectionError> {
        let mut backend = Metal::new(Limits::from_device(device), EstimateModel::from_device(device))?;
        backend.numerical_environment = format!("seismic-metal-v2:{}", device.target_fingerprint());
        backend.target_profile = crate::target::TargetProfile::from_evidence(
            &device.capability_fingerprint(),
            &device.profile.scalar_dtypes.value,
            &device.profile.matrix_dtypes.value,
            &device.profile.matrix_combinations.value,
            device.max_threads_per_threadgroup,
            device.max_threadgroup_bytes,
            device.max_buffer_bytes,
            device.profile.private_storage_budget_bytes.value,
        );
        Ok(backend)
    }

    pub fn target_profile(&self) -> &crate::target::TargetProfile { &self.target_profile }
}

/// A hard device limit over derived quantities.
enum Limit {
    /// Pieces of one root `parallel` launch fit the grid.
    Dispatch { pieces: Quantity },
    /// Threadgroup-placed tile bytes of the pieces sharing one threadgroup fit its memory.
    Threadgroup { bits: Quantity, pieces: Quantity },
    /// Privately placed tile bytes one kernel declares per thread fit the thread stack.
    Private { bits: Quantity },
}

impl Limit {
    /// Quantities the constraint reads.
    fn reads(&self) -> Vec<Quantity> {
        match self {
            Limit::Dispatch { pieces } => vec![pieces.clone()],
            Limit::Threadgroup { bits, pieces } => vec![bits.clone(), pieces.clone()],
            Limit::Private { bits } => vec![bits.clone()],
        }
    }

    fn holds(&self, limits: &Limits, site: &dyn Fn(SiteId) -> Option<i64>) -> bool {
        match self {
            Limit::Dispatch { pieces } => pieces.eval(site).is_ok_and(|p| p <= limits.max_pieces()),
            Limit::Private { bits } => bits.eval(site).is_ok_and(|bits| bits.div_ceil(8) <= limits.max_private_bytes),
            Limit::Threadgroup { bits, pieces } => bits.eval(site).and_then(|b| Ok((b, pieces.eval(site)?))).is_ok_and(|(bits, pieces)| {
                bits.div_ceil(8).checked_mul(limits.items_per_group(pieces)).is_some_and(|bytes| bytes <= limits.max_threadgroup_bytes)
            }),
        }
    }

    /// A violated memory limit lowers the highest site of the offending tiles; a violated
    /// dispatch limit raises the highest site of the launch.
    fn legality(self, limits: &Limits, guard: Vec<CandidateRef>, reason: String) -> Legality {
        let limit = Arc::new(self);
        let (limits, held, needed) = (limits.clone(), limit.clone(), limit.clone());
        Legality {
            guard,
            reads: limit.reads(),
            holds: Arc::new(move |site| held.holds(&limits, site)),
            needed: Arc::new(move |site| match &*needed {
                Limit::Dispatch { pieces } => pieces.eval(site).map(|p| format!("{p} pieces")),
                Limit::Threadgroup { bits, .. } | Limit::Private { bits } => bits.eval(site).map(|b| format!("{} bytes", b.div_ceil(8))),
            }),
            repairs: match &*limit {
                Limit::Dispatch { pieces } => vec![(vec![pieces.clone()], true)],
                Limit::Threadgroup { bits, .. } | Limit::Private { bits } => vec![(vec![bits.clone()], false)],
            },
            reason,
        }
    }
}

impl Metal {
    fn analysis<'a>(&self, program: &'a Program, family: &'a Family) -> Result<Analysis<'a>, SelectionError> {
        if family.target != TARGET {
            return Err(SelectionError::UnsupportedMapping(format!("family of `{}` targets `{}`, not `{TARGET}`", family.entry, family.target)));
        }
        Analysis::new(program, family, TARGET)
    }

    fn legalities(&self, analysis: &Analysis<'_>) -> Result<Vec<Legality>, SelectionError> {
        let family = analysis.family;
        let mut out = Vec::new();
        for (&candidate, account) in &analysis.accounts {
            let bound = &analysis.bounds[&candidate];
            let name = &bound.definition.name;
            for launch in account.launches.iter().filter(|l| l.mode == RegionMode::Parallel) {
                out.push(Limit::Dispatch { pieces: launch.pieces.clone() }.legality(&self.limits, account.context.guards.clone(), format!("`{name}` region#{}: pieces fit {MAX_GROUPS} threadgroups, one piece each", launch.region.0)));
            }
            // Storage rule: a tile is threadgroup-placed when it is a matrix-intrinsic operand, or
            // holds at least one subgroup of elements that an owned loop reads at foreign
            // coordinates (`Ledger::placed`).
            let shared = |account: &Account<MetalAccounting>| -> Vec<Quantity> { account.ledger.placed().into_iter().filter_map(|v| account.ledger.placed_bits(v)).collect() };
            let own: Vec<_> = account.ledger.placed().into_iter().filter_map(|v| account.ledger.tiles.get(&v)).collect();
            if let Some(first) = own.first() {
                out.push(Limit::Threadgroup { bits: Quantity::Sum(shared(account)), pieces: first.pieces.clone() }.legality(&self.limits, account.context.guards.clone(), format!("`{name}`: threadgroup-placed tiles fit {} bytes", self.limits.max_threadgroup_bytes)));
            }
            // Private tiles of one kernel: this candidate's and its ancestors' in the same launch
            // (their selection is implied by the guard chain; sibling calls are separate
            // occurrences and are checked again by `realize`).
            let mut launches: BTreeSet<Option<(CandidateRef, usize)>> = account.ledger.tiles.values().map(|t| t.launch).collect();
            launches.retain(|launch| launch.is_some() || account.context.scope.invocation);
            for launch in launches {
                let bits: Vec<Quantity> = account.context.guards.iter().filter_map(|g| analysis.accounts.get(g))
                    .flat_map(|a| { let placed = a.ledger.placed(); a.ledger.tiles.iter().filter(move |(v, t)| t.launch == launch && !placed.contains(v)).map(|(_, t)| t.private_bits.clone()) }).collect();
                out.push(Limit::Private { bits: Quantity::Sum(bits) }.legality(&self.limits, account.context.guards.clone(), format!("`{name}`: privately placed tiles of one kernel fit {} bytes of thread stack", self.limits.max_private_bytes)));
            }
            // An operand that is a parameter places the caller's tile; trace it to its allocation.
            for operand in account.ledger.placed() {
                let matrix = account.ledger.operands.contains(&operand);
                let Some(VarKind::Param(mut parameter)) = bound.body.vars.get(operand).map(|v| v.kind.clone()) else { continue };
                let mut callee = candidate;
                loop {
                    let occurrence = family.occurrence(callee.occurrence);
                    let (Some(parent), Some(call)) = (occurrence.parent, occurrence.call) else { break };
                    let (caller, caller_account) = (&analysis.bounds[&parent], &analysis.accounts[&parent]);
                    let definition = family.template(family.candidate(callee).template).definition;
                    let argument = caller.body.calls.get(call.0 as usize).and_then(|site| site.bindings.iter().find(|b| b.definition == definition)).and_then(|b| b.arg_order.get(parameter));
                    let Some(&(_, _, root)) = argument.and_then(|ordinal| caller_account.ledger.call_arguments.iter().find(|(c, o, _)| *c == call && o == ordinal)) else { break };
                    if let Some(VarKind::Param(outer)) = caller.body.vars.get(root).map(|v| v.kind.clone()) {
                        (callee, parameter) = (parent, outer);
                        continue;
                    }
                    // A borrowed snapshot read at foreign coordinates declares no array.
                    if let Some(tile) = caller_account.ledger.tiles.get(&root).filter(|tile| matrix || !tile.snapshot) {
                        let mut bits = shared(caller_account);
                        if !caller_account.ledger.placed().contains(&root) {
                            bits.push(if matrix { tile.bits.clone() } else { tile.replicated_bits.clone() });
                        }
                        out.push(Limit::Threadgroup { bits: Quantity::Sum(bits), pieces: tile.pieces.clone() }.legality(&self.limits, account.context.guards.clone(), format!("`{}`: tile placed in threadgroup memory by `{name}` fits {} bytes", caller.definition.name, self.limits.max_threadgroup_bytes)));
                    }
                    break;
                }
            }
        }
        Ok(out)
    }

    /// Polynomial resource model: one guarded additive term per candidate contribution, plus
    /// one upper bound per physical launch. This represents sibling co-selection directly and
    /// never enumerates candidate subsets or Cartesian products.
    fn launch_resources(&self, analysis: &Analysis<'_>) -> Vec<ResourceConstraint> {
        type LaunchKey = (Option<(CandidateRef, usize)>, CandidateRef);
        type SharedContribution = (Vec<CandidateRef>, Quantity, Quantity);
        let mut shared: BTreeMap<LaunchKey, Vec<SharedContribution>> = BTreeMap::new();
        let mut private: BTreeMap<LaunchKey, Vec<(Vec<CandidateRef>, Quantity)>> = BTreeMap::new();
        for (&candidate, account) in &analysis.accounts {
            let mut launches = account.ledger.tiles.values().map(|tile| tile.launch).collect::<Vec<_>>();
            launches.sort_unstable();
            launches.dedup();
            launches.retain(|launch| launch.is_some() || account.context.scope.invocation);
            for launch in launches {
                let owner = launch.map_or(candidate, |(owner, _)| owner);
                let shared_bits = account.ledger.shared_bits_for(launch);
                if !shared_bits.is_empty() {
                    let pieces = account.ledger.tiles.values().find(|tile| tile.launch == launch)
                        .map(|tile| tile.pieces.clone()).unwrap_or_else(Quantity::one);
                    shared.entry((launch, owner)).or_default().push((
                        account.context.guards.clone(), Quantity::Sum(shared_bits), pieces,
                    ));
                }
                let private_bits = account.ledger.private_bits(launch);
                if !private_bits.is_empty() {
                    private.entry((launch, owner)).or_default().push((
                        account.context.guards.clone(), Quantity::Sum(private_bits),
                    ));
                }
            }
        }

        let mut resources = Vec::with_capacity(shared.len() + private.len());
        for ((launch, _), contributions) in shared {
            let terms = contributions.into_iter().map(|(guard, bits, pieces)| {
                let scope = quantity::scope([&bits, &pieces]);
                let table = scope.clone();
                let limits = self.limits.clone();
                ResourceTerm {
                    guard,
                    scope,
                    amount: Box::new(move |values| {
                        let site = quantity::lookup(&table, values);
                        bits.eval(&site)?.div_ceil(8)
                            .checked_mul(limits.items_per_group(pieces.eval(&site)?))
                            .ok_or_else(|| "Metal threadgroup bytes overflow u64".into())
                    }),
                }
            }).collect();
            resources.push(ResourceConstraint {
                terms,
                capacity: self.limits.max_threadgroup_bytes,
                reason: format!("Metal launch {launch:?} co-selected threadgroup arrays fit {} bytes", self.limits.max_threadgroup_bytes),
            });
        }
        for ((launch, _), contributions) in private {
            let terms = contributions.into_iter().map(|(guard, bits)| {
                let scope = quantity::scope([&bits]);
                let table = scope.clone();
                ResourceTerm {
                    guard,
                    scope,
                    amount: Box::new(move |values| bits.eval(&quantity::lookup(&table, values)).map(|bits| bits.div_ceil(8))),
                }
            }).collect();
            resources.push(ResourceConstraint {
                terms,
                capacity: self.limits.max_private_bytes,
                reason: format!("Metal launch {launch:?} co-selected private arrays fit {} bytes", self.limits.max_private_bytes),
            });
        }
        resources
    }

}

impl Costs<MetalAccounting> for EstimateModel {
    fn scope_label(&self) -> String {
        "estimate (launch + norm(max(compute x private-storage pressure, tile traffic), bus traffic); runtime extents at their static upper bound)".into()
    }

    fn scope(&self, scope: CostScope<'_, '_, MetalAccounting>) -> ScopeCost {
        let CostScope { analysis, account, launch, launches, work, pieces } = scope;
        // Thread-private array bits the kernel of this scope declares per thread, as far
        // as this candidate's chain determines them: its own and its ancestors' privately
        // placed tiles of the same launch. A tile handed to a call is held whole by every
        // lane (a callee addresses it by computed coordinates); others follow the
        // distribution rule. A snapshot of external storage is taken as borrowed (no
        // array). Sibling occurrences inlined into the launch are not seen.
        let private: Vec<Quantity> = match launch {
            None => Vec::new(),
            Some(_) => account.context.guards.iter().filter_map(|g| analysis.accounts.get(g))
                .flat_map(|a| {
                    let placed = a.ledger.placed();
                    a.ledger.tiles.iter().filter(move |(v, t)| t.launch == launch && !t.snapshot && !placed.contains(v)).map(|(v, t)| {
                        if a.ledger.call_arguments.iter().any(|(_, _, root)| root == v) { t.replicated_bits.clone() } else { t.private_bits.clone() }
                    })
                })
                .collect(),
        };
        let private = Quantity::Sum(private);
        // Threadgroup geometry of the scope's kernel, as far as the candidate's chain determines
        // it: the threadgroup-placed tiles of the chain in this launch (per threadgroup, an
        // inner owner's tile once per SIMD group), and the SIMD groups of one threadgroup (the
        // inner owners of a launch piece, else one).
        let placed: Vec<Quantity> = match launch {
            None => Vec::new(),
            Some(_) => account.context.guards.iter().filter_map(|g| analysis.accounts.get(g))
                .flat_map(|a| a.ledger.placed().into_iter().filter(|v| a.ledger.tiles.get(v).is_some_and(|t| t.launch == launch)).filter_map(|v| a.ledger.placed_bits(v)).collect::<Vec<_>>())
                .collect(),
        };
        let placed = Quantity::Sum(placed);
        let subgroups = match launch {
            Some((owner, ordinal)) => analysis.accounts.get(&owner).and_then(|a| a.launches.get(ordinal)).and_then(|l| l.owners.first().cloned())
                .or_else(|| account.context.scope.owners.clone()),
            None => None,
        }
        .unwrap_or_else(Quantity::one);
        let (work, pieces, model) = (work.clone(), pieces.clone(), self.clone());
        let reads = vec![private.clone(), placed.clone(), subgroups.clone()];
        ScopeCost {
            reads,
            cost: Box::new(move |site| {
                let group = Group { subgroups: subgroups.eval(site)?, bytes: placed.eval(site)?.div_ceil(8) };
                model.scope_ns(launches, &work.totals(site)?, pieces.eval(site)?, private.eval(site)?.div_ceil(8), group)
            }),
        }
    }

    fn launch(&self) -> Box<dyn Fn() -> Result<u64, String> + Send + Sync> {
        let model = self.clone();
        Box::new(move || model.scope_ns(1, &Totals::default(), 1, 0, Group::default()))
    }

    fn materialization(&self) -> Box<dyn Fn(u64) -> Result<u64, String> + Send + Sync> {
        let model = self.clone();
        Box::new(move |bits| model.materialization_ns(bits))
    }
}

impl Backend for Metal {
    type Execution = Execution;

    fn target(&self) -> &'static str {
        TARGET
    }

    fn capability_fingerprint(&self) -> String {
        self.target_profile.fingerprint().into()
    }

    fn supports_intrinsic(&self, intrinsic: &IntrinsicUse) -> Result<(), String> {
        self.target_profile.supports_intrinsic(intrinsic)
    }

    fn estimate_model(&self) -> String {
        IDENTITY.into()
    }

    fn numerical_environment(&self) -> String {
        self.numerical_environment.clone()
    }

    fn bind_structure(&self, program: &Program, family: &Family) -> Result<BTreeMap<SiteId, Vec<i64>>, SelectionError> {
        self.analysis(program, family)?;
        mapping::domains(program, family)
    }

    fn constraints(&self, program: &Program, family: &Family) -> Result<Vec<Constraint>, SelectionError> {
        let analysis = self.analysis(program, family)?;
        mapping::constraints(family, self.legalities(&analysis)?)
    }

    fn resources(&self, program: &Program, family: &Family) -> Result<Vec<ResourceConstraint>, SelectionError> {
        Ok(self.launch_resources(&self.analysis(program, family)?))
    }

    fn intervals(&self, program: &Program, family: &Family) -> Result<Vec<Interval>, SelectionError> {
        mapping::intervals(&self.analysis(program, family)?, family)
    }

    /// Additive estimate over the shared factor skeleton: each scope is the span of its own
    /// work (`EstimateModel::span`) under the private-storage pressure of its kernel. A
    /// borrowed external snapshot passed to a call is charged in the callee's body factor,
    /// where its bus traffic overlaps the callee's compute.
    fn factors(&self, program: &Program, family: &Family, intervals: &[Interval]) -> Result<Vec<Factor>, SelectionError> {
        mapping::factors(&self.analysis(program, family)?, family, intervals, &self.estimate)
    }

    /// Seed policy (`mapping::seed`), with `lower ... for metal` bodies first.
    /// Sites: root `parallel` binders take the smallest admissible widths whose piece count
    /// does not exceed the subgroups the device runs concurrently
    /// (`ceil(concurrent_lanes / 32)`, within the grid limit), raising the last binder first,
    /// so pieces ~ available subgroups; every other width site takes its largest admissible
    /// value; parts take 1. A violated threadgroup limit then lowers the highest site of the
    /// offending tiles; a violated dispatch limit raises the highest site of the launch.
    /// Covers: all singletons.
    fn seed(&self, program: &Program, family: &Family, domains: &BTreeMap<SiteId, Vec<i64>>, intervals: &[Interval]) -> Result<Witness, SelectionError> {
        let analysis = self.analysis(program, family)?;
        let concurrent = self.estimate.concurrent_lanes.div_ceil(SUBGROUP as u64).clamp(1, self.limits.max_pieces());
        mapping::seed(program, family, TARGET, &analysis, domains, intervals, concurrent, self.legalities(&analysis)?)
    }

    fn realize(&self, lowered: LoweredIr, family: &Family, _witness: &Witness) -> Result<Execution, SelectionError> {
        realize::realize(&self.limits, &lowered, family.allow_numerical_effects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::family::{OccurrenceId, Requirement, Site, SiteKind};
    use seismic_lang::sir::CallId;
    use seismic_lang::types::{RegionId, SliceId};

    #[test]
    fn width_domain_is_structured_divisors() {
        let owner = CandidateRef { occurrence: OccurrenceId(0), candidate: 0 };
        let site = Site { id: SiteId(0), owner, kind: SiteKind::Width { region: RegionId(0), slice: SliceId(0) }, extent: 2560 };
        assert_eq!(mapping::domain(&site, &[], &[]), vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2560]);
        let multiple = Requirement::Multiple { site: SiteId(0), unit: 40 };
        assert_eq!(mapping::domain(&site, &[multiple.clone()], &[multiple]), vec![40, 80, 160, 320, 640, 1280, 2560]);
        let parts = Site { kind: SiteKind::Parts { region: RegionId(0), slice: SliceId(0) }, extent: 5, ..site };
        assert_eq!(mapping::domain(&parts, &[], &[]), vec![1, 2, 4]);
    }

    #[test]
    fn estimate_is_monotone_in_launches() {
        let model = EstimateModel::default();
        let totals = Totals { lane_ops: 1 << 20, visits: 16, device_bits: 1 << 23, ..Totals::default() };
        let costs: Vec<u64> = (0..4).map(|launches| model.scope_ns(launches, &totals, 64, 0, Group::default()).unwrap()).collect();
        assert!(costs.windows(2).all(|w| w[0] < w[1]), "{costs:?}");
    }

    #[test]
    fn complete_call_operand_is_rejected_at_the_1081504_byte_stack_regression() {
        let owner = CandidateRef { occurrence: OccurrenceId(0), candidate: 0 };
        let launch = Some((owner, 0));
        let mut ledger = accounting::Ledger::default();
        ledger.tiles.insert(0, accounting::Tile {
            bits: Quantity::Constant(1_081_504 * 8),
            // Ordinary lane distribution looked legal; a complete call operand instead needs
            // the full replicated declaration in every lane's private address space.
            private_bits: Quantity::Constant(33_800 * 8),
            replicated_bits: Quantity::Constant(1_081_504 * 8),
            snapshot: false,
            launch,
            pieces: Quantity::one(),
            owner: seismic_compiler::selection::structure::TileOwner::Piece,
        });
        ledger.call_arguments.push((CallId(0), 0, 0));
        let requirement = Quantity::Sum(ledger.private_bits(launch));
        assert_eq!(requirement.eval(&|_| None).unwrap().div_ceil(8), 1_081_504);
        assert!(requirement.eval(&|_| None).unwrap().div_ceil(8) > 131_072);
    }

    #[test]
    fn realize_emits_a_kernel() {
        use seismic_lang::exec::ir::{Expr, ExprKind, Index, Stmt, StmtKind, Var, VarKind as IrVarKind};
        use seismic_lang::exec::types::{Shaped, Ty};
        use seismic_lang::sym::{Atom, Sym};
        use seismic_lang::types::{DType, Elem};
        let span = Default::default();
        let tensor = Ty::Tensor(Shaped::new(vec![Sym::constant(4)], Elem::Dtype(DType::F32)));
        let atom = Atom::Param("i#1".into());
        let expr = |kind, ty, sym| Expr { kind, ty, sym, span };
        let target = expr(
            ExprKind::Index {
                base: Box::new(expr(ExprKind::Var(0), tensor.clone(), None)),
                indices: vec![Index::Point(expr(ExprKind::Var(1), Ty::Scalar(DType::I32), Some(Sym::atom(atom.clone()))))],
            },
            Ty::Scalar(DType::F32),
            None,
        );
        let assign = Stmt { id: None, span, kind: StmtKind::Assign { target, op: seismic_lang::syntax::ast::AssignOp::Assign, value: expr(ExprKind::Float(1.0), Ty::Scalar(DType::F32), None) } };
        let lowered = LoweredIr {
            name: "fill".into(),
            backend: TARGET.into(),
            ownership: Default::default(),
            alias_requirements: Vec::new(),
            params: vec![("y".into(), tensor.clone())],
            source_param_count: 1,
            result: Ty::Void,
            result_bindings: Vec::new(),
            index_params: Vec::new(),
            range_params: Vec::new(),
            vars: vec![
                Var { name: "y".into(), ty: tensor, span, kind: IrVarKind::Param(0) },
                Var { name: "i".into(), ty: Ty::Scalar(DType::I32), span, kind: IrVarKind::Index(atom) },
            ],
            body: vec![Stmt { id: None, span, kind: StmtKind::Parallel { vars: vec![1], extents: vec![Sym::constant(4)], body: vec![assign] } }],
            shapes: Default::default(),
        };
        let limits = Limits { max_threads_per_threadgroup: 1024, max_threadgroup_bytes: 32768, max_private_bytes: CONSERVATIVE_PRIVATE_STORAGE_BUDGET_BYTES };
        let execution = realize::realize(&limits, &lowered, false).unwrap();
        assert!(crate::msl::emit_execution(&execution).unwrap().source.contains("kernel void"));
    }
}
