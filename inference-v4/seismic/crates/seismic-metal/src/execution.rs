//! IR-only application of explicit execution choices, before source emission.
//! This is an intermediate compiler boundary, not a complete Tuned IR: storage
//! lifetimes, intrinsic scheduling, and model-driven selection remain incomplete.
use seismic_lang::{
    ir::*,
    lowered_ir::LoweredIr,
    sym::{Atom, Sym},
    types::{DType, Ty},
};
use seismic_realization::dispatch::{GroupDispatch, WorkMapping};

pub const SUBGROUP: i64 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldOwnership {
    Serial,
    Participants,
    ParticipantsInsertSeed,
    ParticipantsWavefront,
    ParticipantsWavefrontInsertSeed,
    ParticipantsRootSeed,
    ParticipantsWavefrontRootSeed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldChoice {
    pub site: usize,
    pub lanes: u32,
    pub wavefront: bool,
    pub root_seed: bool,
}
impl seismic_accounting::selection::Choices for FoldChoice {
    type Alternative = FoldOwnership;
    fn len(&self) -> usize {
        if self.root_seed { if self.wavefront { 3 } else { 2 } } else if self.wavefront { 5 } else { 3 }
    }
    fn get(&self, index: usize) -> Option<FoldOwnership> {
        if self.root_seed {
            return [FoldOwnership::Serial, FoldOwnership::ParticipantsRootSeed, FoldOwnership::ParticipantsWavefrontRootSeed][..self.len()].get(index).copied();
        }
        [
            FoldOwnership::Serial,
            FoldOwnership::Participants,
            FoldOwnership::ParticipantsInsertSeed,
            FoldOwnership::ParticipantsWavefront,
            FoldOwnership::ParticipantsWavefrontInsertSeed,
        ][..self.len()]
        .get(index)
        .copied()
    }
}

/// Realization choices the model will close; defaults for now.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Explicit load realization; borrowing requires the shared lifetime proof.
    pub loads: seismic_realization::LoadStrategy,
    pub sg_per_tg: i64,
    /// Piece capacity for static streaming extents; `None` streams whole axes.
    pub piece: Option<i64>,
    /// How many consecutive values of the innermost `parallel` index one work item covers.
    /// A free integer of the realization: kernels never name it.
    pub per_item: i64,
    /// Work items sharing a streamed range through separate slices and scratch.
    /// A subsequent merge launch completes the phase. 1 leaves the range whole.
    pub split: i64,
    /// Explicit independent pointwise tile extent per work item. Requires a
    /// checked partition proof and runtime overlap validation; no implicit policy.
    pub tile_piece: Option<i64>,
    /// Device capacity facts, queried. The performance model will weigh
    /// parallelism against reuse; it is carried so no part of the compiler invents it.
    pub max_threads_per_threadgroup: i64,
    pub max_threadgroup_bytes: i64,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            loads: seismic_realization::LoadStrategy::BorrowProvenReadOnly,
            sg_per_tg: 4,
            piece: None,
            per_item: 1,
            split: 1,
            tile_piece: None,
            max_threads_per_threadgroup: 1024,
            max_threadgroup_bytes: 32768,
        }
    }
}

/// A selected parallel mapping and the explicit split handoff, if present.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub mapping: WorkMapping,
    pub parts: i64,
    pub dispatch: GroupDispatch,
    pub merge_dispatch: Option<GroupDispatch>,
    pub split: Option<Split>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Split {
    pub loop_at: usize,
    pub carried: Vec<VarId>,
    pub part: VarId,
    /// Derived from the updates and identity, never from state count or names.
    pub merges: Vec<Merge>,
    /// Validate the original runtime domain before narrowing it into slices.
    pub original_views: Vec<Expr>,
    pub validation_bindings: Vec<Stmt>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Merge {
    Sum,
}

fn derive_merges(
    before: &[Stmt],
    body: &[Stmt],
    carried: &[VarId],
    streams: &[VarId],
    vars: &[Var],
) -> Result<Vec<Merge>, String> {
    use seismic_lang::ast::AssignOp;
    let failure = || "split reduction has no proven merge rule for its carried state".to_string();
    // This admitted merge form is an unordered sum of the streamed elements.
    // Independence of carried state alone is insufficient: e.g. summing chunk
    // maxima changes meaning when a split changes chunk boundaries.
    let mut updated = std::collections::HashSet::new();
    for statement in body {
        let StmtKind::Assign {
            target,
            op: AssignOp::Add,
            value,
        } = &statement.kind
        else {
            return Err(failure());
        };
        let ExprKind::Index { base, indices } = &target.kind else {
            return Err(failure());
        };
        let ExprKind::Var(v) = base.kind else {
            return Err(failure());
        };
        if !carried.contains(&v)
            || !updated.insert(v)
            || indices.len() != 1
            || !matches!(&indices[0], Index::Point(p) if p.sym.as_ref().and_then(Sym::as_constant) == Some(0))
        {
            return Err(failure());
        }
        let ExprKind::Builtin {
            name: Builtin::Reduce,
            args,
        } = &value.kind
        else {
            return Err(failure());
        };
        if args.len() < 3
            || !matches!(args[0].kind, ExprKind::Var(v) if streams.contains(&v))
            || !matches!(args[1].kind, ExprKind::Int(0))
            || !matches!(args[2].kind, ExprKind::Int(0))
            || args
                .get(3)
                .is_some_and(|a| !matches!(a.kind, ExprKind::Bool(false)))
        {
            return Err(failure());
        }
        let Ty::Tile(tile) = &args[0].ty else {
            return Err(failure());
        };
        if tile.shape.len() != 1 || tile.elem != seismic_lang::types::Elem::Dtype(DType::F32) {
            return Err(failure());
        }
    }
    if updated.len() != carried.len() {
        return Err(failure());
    }
    for &v in carried {
        let Ty::Tile(tile) = &vars[v].ty else {
            return Err(failure());
        };
        if tile.elem != seismic_lang::types::Elem::Dtype(DType::F32)
            || tile.shape != [Sym::constant(1)]
        {
            return Err(failure());
        }
        let initialized = before.iter().any(|s| {
            let StmtKind::Owned { vars: indices, tile, body } = &s.kind else { return false; };
            if !matches!(tile.kind, ExprKind::Var(id) if id == v) || indices.len() != 1 || body.len() != 1 { return false; }
            let StmtKind::Assign { target, op: AssignOp::Assign, value } = &body[0].kind else { return false; };
            let ExprKind::Index { base, indices: points } = &target.kind else { return false; };
            matches!(base.kind, ExprKind::Var(id) if id == v) && points.len() == 1
                && matches!(&points[0], Index::Point(p) if matches!(p.kind, ExprKind::Var(id) if id == indices[0]))
                && matches!(value.kind, ExprKind::Float(x) if x == 0.0)
        });
        if !initialized {
            return Err(failure());
        }
        // No later mutation between the identity initializer and the stream is
        // admitted; allocations and that initializer are the only state writes.
        let mut identities = 0;
        let mut allocations = 0;
        for s in before {
            let mut writes = std::collections::HashSet::new();
            seismic_lang::rewrite::writes(s, &mut writes);
            if writes.contains(&v) {
                match &s.kind {
                    StmtKind::Assign { value, .. }
                        if matches!(value.kind, ExprKind::TileAlloc { .. }) && identities == 0 =>
                    {
                        allocations += 1;
                    }
                    StmtKind::Owned { .. } => identities += 1,
                    _ => return Err(failure()),
                }
            }
        }
        if identities != 1 || allocations != 1 {
            return Err(failure());
        }
    }
    Ok(carried.iter().map(|_| Merge::Sum).collect())
}

/// The same merge applicability used by preparation bounds its supported split
/// choice. The current admitted form uses one part count across eligible phases.
pub(crate) fn split_domain(function: &LoweredIr) -> Result<u64, String> {
    let candidates = seismic_lang::split::split_candidates(&function.body, &function.vars);
    if candidates.is_empty() {
        return Ok(1);
    }
    let mut phases = std::collections::HashSet::new();
    let mut maximum = u64::MAX;
    for candidate in candidates {
        if !phases.insert(candidate.stmt) {
            return Ok(1);
        }
        let StmtKind::Parallel { body, extents, .. } = &function.body[candidate.stmt].kind else {
            unreachable!()
        };
        let StmtKind::LoadLoop {
            body: stream,
            vars,
            domain,
            ..
        } = &body[candidate.loop_at].kind
        else {
            unreachable!()
        };
        if seismic_lang::rewrite::check_split(
            stream,
            &candidate.carried,
            &seismic_lang::rewrite::namer(&function.vars),
        )
        .is_err()
            || derive_merges(
                &body[..candidate.loop_at],
                stream,
                &candidate.carried,
                vars,
                &function.vars,
            )
            .is_err()
        {
            return Ok(1);
        }
        let extent = domain
            .view
            .ty
            .shaped()
            .and_then(|s| s.shape.get(domain.axis))
            .and_then(Sym::as_constant)
            .and_then(|n| u64::try_from(n).ok());
        // Dynamic ranges remain supported for explicit diagnostic splits. A
        // finite symbolic domain needs a retained bound, not a guessed maximum.
        let Some(extent) = extent else {
            return Ok(1);
        };
        let items = extents
            .iter()
            .try_fold(1u64, |p, e| {
                p.checked_mul(u64::try_from(e.as_constant()?).ok()?)
            })
            .ok_or("invalid split parallel extent")?;
        maximum = maximum
            .min(extent.max(1))
            .min(u64::from(u32::MAX) / items.max(1))
            .min(i64::MAX as u64);
    }
    Ok(maximum.max(1))
}

/// Owns the transformed IR. It cannot be changed independently of its mappings.
#[derive(Clone)]
pub struct Execution {
    pub(crate) terminal: std::sync::Arc<TerminalImplementation>,
    pub(crate) transfers: Vec<crate::terminal::transfer::Selection>,
    pub(crate) traversals: Vec<crate::terminal::traversal::Selection>,
    pub(crate) function: LoweredIr,
    /// Checked source with selected structured contracts before backend
    /// realization. The emitted function below is its selected implementation.
    pub(crate) source: LoweredIr,
    pub(crate) config: Config,
    pub(crate) phases: Vec<Phase>,
    pub(crate) retained: Vec<seismic_realization::phases::RetainedValue>,
    pub(crate) memory: crate::memory::MemoryPlan,
    pub(crate) support: crate::support::Plan,
    pub(crate) reductions: crate::reduction::ReductionPlan,
    pub(crate) storage: crate::storage::StoragePlan,
    pub(crate) partition_parameters: Vec<(usize, DType)>,
}

/// Emission and its workload-bound account belong to the same immutable
/// terminal implementation. Identity refinements retain both; any code or
/// dispatch change replaces this owner before either result can be reused.
#[derive(Default)]
pub(crate) struct TerminalImplementation {
    pub(crate) emission: std::sync::OnceLock<Result<crate::msl::Emitted, String>>,
    account: RefinementAccountCache,
}

impl Execution {
    pub(crate) fn invalidate_terminal(&mut self) {
        self.terminal = Default::default();
    }
    pub(crate) fn relaxation(
        &self,
        workload: &seismic_accounting::workload::ScalarWorkload,
        limits: seismic_accounting::workload::DerivationLimits,
    ) -> Result<std::sync::Arc<crate::model::InvocationAccount>, String> {
        self.terminal.account.derive(self, workload, limits)
    }
    /// Compare the actual prepared implementation independently of whether its
    /// lazy target emission has already been requested.
    pub(crate) fn same_implementation(&self, other: &Self) -> bool {
        self.transfers == other.transfers && self.traversals == other.traversals
            && self.function == other.function
            && self.source == other.source
            && self.config == other.config
            && self.phases == other.phases
            && self.retained == other.retained
            && self.memory == other.memory
            && self.support == other.support
            && self.reductions == other.reductions
            && self.storage == other.storage
            && self.partition_parameters == other.partition_parameters
    }
    pub fn source(&self) -> &LoweredIr {
        &self.source
    }
    pub fn support(&self) -> &crate::support::Plan {
        &self.support
    }
    pub fn memory(&self) -> &crate::memory::MemoryPlan {
        &self.memory
    }
    pub fn reductions(&self) -> &crate::reduction::ReductionPlan {
        &self.reductions
    }
    pub fn storage(&self) -> &crate::storage::StoragePlan {
        &self.storage
    }
    pub fn function(&self) -> &LoweredIr {
        &self.function
    }
    pub fn phases(&self) -> &[Phase] {
        &self.phases
    }
    pub fn retained(&self) -> &[seismic_realization::phases::RetainedValue] {
        &self.retained
    }
}

/// Apply a caller's choices exactly, without native compilation or querying a device.
pub fn prepare(function: &LoweredIr, config: Config) -> Result<Execution, String> {
    prepare_storage_selected(function, config, &mut |decision| Ok(decision.diagnostic()))
}

pub fn prepare_storage_selected(
    function: &LoweredIr,
    config: Config,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
) -> Result<Execution, String> {
    prepare_selected(function, config, select, &mut |decision| {
        Ok(decision.diagnostic())
    })
}

pub fn prepare_selected(
    function: &LoweredIr,
    config: Config,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
) -> Result<Execution, String> {
    let borrow = config.loads == seismic_realization::LoadStrategy::BorrowProvenReadOnly;
    prepare_with_choices(
        function,
        config,
        &mut |_, site| {
            Ok(if borrow && site.can_borrow {
                LoadMode::Borrow
            } else {
                LoadMode::Materialize
            })
        },
        select,
        select_reduction,
    )
}

/// Prepare all site choices after decomposition, so widening and splitting
/// cannot silently create loads outside the selected domain.
pub fn prepare_with_choices(
    function: &LoweredIr,
    config: Config,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::normalize::loads::Site,
    ) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
) -> Result<Execution, String> {
    prepare_with_allocation_choices(
        function,
        config,
        select_load,
        select,
        select_reduction,
        &mut |choice| Ok(choice.new_slot),
    )
}
pub fn prepare_with_allocation_choices(
    function: &LoweredIr,
    config: Config,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::normalize::loads::Site,
    ) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
) -> Result<Execution, String> {
    prepare_with_mappings(
        function,
        config,
        None,
        select_load,
        select,
        select_reduction,
        select_allocation,
    )
}
pub fn prepare_with_mappings(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::normalize::loads::Site,
    ) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
) -> Result<Execution, String> {
    prepare_with_participants(
        function,
        config,
        mappings,
        &mut |_| Ok(FoldOwnership::Serial),
        select_load,
        select,
        select_reduction,
        select_allocation,
    )
}
pub fn prepare_with_participants(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
    select_fold: &mut dyn FnMut(&FoldChoice) -> Result<FoldOwnership, String>,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::normalize::loads::Site,
    ) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
) -> Result<Execution, String> {
    prepare_with_traversals(function, config, mappings, select_fold, select_load, select, select_reduction, select_allocation, &mut |_| Ok(1))
}

pub fn prepare_with_traversals(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
    select_fold: &mut dyn FnMut(&FoldChoice) -> Result<FoldOwnership, String>,
    select_load: &mut dyn FnMut(usize, &seismic_lang::normalize::loads::Site) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(&crate::storage::StorageDecision) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(&crate::reduction::Decision) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
    select_traversal: &mut dyn FnMut(&crate::terminal::traversal::Choice) -> Result<usize, String>,
) -> Result<Execution, String> {
    prepare_with_transfers(function, config, mappings, select_fold, select_load, select, select_reduction, select_allocation, &mut |_| Ok(1), select_traversal)
}

pub fn prepare_with_transfers(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
    select_fold: &mut dyn FnMut(&FoldChoice) -> Result<FoldOwnership, String>,
    select_load: &mut dyn FnMut(usize, &seismic_lang::normalize::loads::Site) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(&crate::storage::StorageDecision) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(&crate::reduction::Decision) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
    select_transfer: &mut dyn FnMut(&crate::terminal::transfer::Choice) -> Result<u8, String>,
    select_traversal: &mut dyn FnMut(&crate::terminal::traversal::Choice) -> Result<usize, String>,
) -> Result<Execution, String> {
    let mut stage = prepare_initial(function, config, mappings)?;
    loop {
        match advance(
            &stage,
            select_fold,
            select_load,
            select,
            select_reduction,
            select_allocation,
            select_transfer,
            select_traversal,
        )? {
            Advance::Stage(next) => stage = next,
            Advance::Execution(execution) => return Ok(execution),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Prepared {
    function: LoweredIr,
    source: LoweredIr,
    config: Config,
    phases: Vec<Phase>,
    retained: Vec<seismic_realization::phases::RetainedValue>,
    partition_parameters: Vec<(usize, DType)>,
    private_values: Vec<usize>,
}

/// Immutable existing execution boundaries. Later decisions retain their owning
/// computation and completed plans rather than reconstructing earlier phases.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Stage {
    Transfers(std::sync::Arc<PreparedTransfers>),
    Traversals(std::sync::Arc<PreparedTraversals>),
    Folds(std::sync::Arc<Prepared>),
    Loads(std::sync::Arc<Prepared>),
    Storage(std::sync::Arc<Prepared>),
    Reductions(
        std::sync::Arc<Prepared>,
        std::sync::Arc<crate::storage::StoragePlan>,
    ),
    Allocations(
        std::sync::Arc<Prepared>,
        std::sync::Arc<crate::storage::StoragePlan>,
        std::sync::Arc<crate::reduction::ReductionPlan>,
    ),
}
impl Stage {
    pub(crate) fn refinement_account(
        &self,
        workload: &seismic_accounting::workload::ScalarWorkload,
        limits: seismic_accounting::workload::DerivationLimits,
    ) -> Result<Option<std::sync::Arc<crate::model::InvocationAccount>>, String> {
        let execution = match self {
            Self::Transfers(prepared) => &prepared.execution,
            Self::Traversals(prepared) => &prepared.execution,
            _ => return Ok(None),
        };
        execution.relaxation(workload, limits).map(Some)
    }

    pub(crate) fn preserves(&self, primitive: &crate::terminal::Primitive) -> bool {
        match self {
            Self::Transfers(_) => crate::terminal::transfer::preserves(primitive),
            Self::Traversals(_) => crate::terminal::traversal::preserves(primitive),
            _ => false,
        }
    }
    pub(crate) fn phases(&self) -> &[Phase] {
        match self {
            Self::Traversals(p) => p.execution.phases(),
            Self::Transfers(p) => p.execution.phases(),
            Self::Folds(p)
            | Self::Loads(p)
            | Self::Storage(p)
            | Self::Reductions(p, _)
            | Self::Allocations(p, _, _) => &p.phases,
        }
    }
    pub(crate) fn function(&self) -> &LoweredIr {
        match self {
            Self::Traversals(p) => p.execution.function(),
            Self::Transfers(p) => p.execution.function(),
            Self::Folds(p)
            | Self::Loads(p)
            | Self::Storage(p)
            | Self::Reductions(p, _)
            | Self::Allocations(p, _, _) => &p.function,
        }
    }
}
#[derive(Clone)]
pub(crate) struct PreparedTransfers {
    execution: Execution,
    choices: Vec<crate::terminal::transfer::Choice>,
}
impl std::fmt::Debug for PreparedTransfers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.debug_struct("PreparedTransfers").field("choices", &self.choices).finish() }
}
impl PartialEq for PreparedTransfers {
    fn eq(&self, other: &Self) -> bool { self.execution.same_implementation(&other.execution) && self.choices == other.choices }
}
#[derive(Clone)]
pub(crate) struct PreparedTraversals {
    execution: Execution,
    choices: Vec<crate::terminal::traversal::Choice>,
}
/// An account belongs to one immutable prepared execution. Complete derivations
/// survive smaller budgets; an exhausted prefix is recomputed for larger ones.
/// This memo is deliberately outside execution identity and selection equality.
#[derive(Default)]
pub(crate) struct RefinementAccountCache(std::sync::Mutex<Option<RefinementAccount>>);
impl RefinementAccountCache {
    pub(crate) fn derive(
        &self,
        execution: &Execution,
        workload: &seismic_accounting::workload::ScalarWorkload,
        limits: seismic_accounting::workload::DerivationLimits,
    ) -> Result<std::sync::Arc<crate::model::InvocationAccount>, String> {
        let mut cache = self.0.lock().map_err(|_| "retained terminal account was poisoned")?;
        if let Some(cached) = cache.as_ref() {
            if &cached.workload == workload && (cached.account.exhausted.is_none()
                || (limits.instructions <= cached.limits.instructions && limits.operations <= cached.limits.operations))
            { return Ok(cached.account.clone()); }
        }
        let account = std::sync::Arc::new(crate::model::invocation_relaxation(execution, workload, limits).map_err(|error| error.to_string())?);
        *cache = Some(RefinementAccount { workload: workload.clone(), limits, account: account.clone() });
        Ok(account)
    }
}
struct RefinementAccount {
    workload: seismic_accounting::workload::ScalarWorkload,
    limits: seismic_accounting::workload::DerivationLimits,
    account: std::sync::Arc<crate::model::InvocationAccount>,
}
impl std::fmt::Debug for PreparedTraversals {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedTraversals").field("function", &self.execution.function.name).field("choices", &self.choices).finish()
    }
}
impl PartialEq for PreparedTraversals {
    fn eq(&self, other: &Self) -> bool {
        self.execution.same_implementation(&other.execution) && self.choices == other.choices
    }
}
pub(crate) enum Advance {
    Stage(Stage),
    Execution(Execution),
}

pub(crate) fn prepare_initial(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
) -> Result<Stage, String> {
    if function.backend != "metal" {
        return Err("Metal execution requires Metal Lowered IR".into());
    }
    if config.sg_per_tg <= 0
        || config.per_item <= 0
        || config.split <= 0
        || config
            .sg_per_tg
            .checked_mul(SUBGROUP)
            .is_none_or(|n| n > config.max_threads_per_threadgroup)
        || config.max_threadgroup_bytes < 0
    {
        return Err("invalid Metal candidate or device resource limits".into());
    }
    if (config.per_item > 1
        || mappings.is_some_and(|m| m.iter().flat_map(WorkMapping::axes).any(|a| a.step > 1)))
        && config.split > 1
    {
        return Err("combined widening and splitting is not realized".into());
    }
    let source = function.clone();
    let mut function = function.clone();
    seismic_lang::normalize::work_domain(&mut function.body);
    let mut partition_parameters = Vec::new();
    if let Some(piece) = config.tile_piece {
        if config.split != 1 {
            return Err("pointwise partition cannot combine with reduction splitting".into());
        }
        let partitioned = seismic_lang::partition::pointwise(&function, piece)?;
        function = partitioned.function;
        partition_parameters = partitioned.parameters;
    }
    if config.split > 1 {
        function = seismic_lang::reduction::structured::materialize(&function)?;
    }
    let phase_plan = seismic_realization::phases::construct(&function)?;
    let retained = phase_plan.retained;
    function = phase_plan.function;
    if mappings.is_some_and(|m| m.len() != function.body.len()) {
        return Err("one work mapping is required per normalized phase".into());
    }
    let splits = if config.split > 1 {
        seismic_lang::split::split_candidates(&function.body, &function.vars)
    } else {
        Vec::new()
    };
    if config.split > 1 && splits.is_empty() {
        return Err("requested split has no legal streamed reduction".into());
    }
    let mut split_by_phase = std::collections::HashMap::new();
    for split in splits {
        if split_by_phase.contains_key(&split.stmt) {
            return Err("multiple split reductions in one phase require separate handoffs".into());
        }
        let StmtKind::Parallel { body, .. } = &function.body[split.stmt].kind else {
            unreachable!()
        };
        let before = &body[..split.loop_at];
        let StmtKind::LoadLoop {
            body,
            vars: streams,
            domain,
            views,
            ..
        } = &body[split.loop_at].kind
        else {
            unreachable!()
        };
        seismic_lang::rewrite::check_split(
            body,
            &split.carried,
            &seismic_lang::rewrite::namer(&function.vars),
        )
        .map_err(|e| e.to_string())?;
        let merges = derive_merges(before, body, &split.carried, streams, &function.vars)?;
        // The iteration view carries guards even when producer projection
        // removes every transfer. Splitting must preserve that original domain.
        let original_views = std::iter::once(domain.view.clone())
            .chain(views.iter().cloned())
            .collect();
        let part = function.vars.len();
        let atom = Atom::Param(format!("part#{part}"));
        function.vars.push(Var {
            name: "part".into(),
            ty: Ty::Scalar(DType::I32),
            span: split.lo.span,
            kind: VarKind::Index(atom.clone()),
        });
        let loop_at = seismic_lang::split::narrow_range(
            &split,
            &mut function.body,
            &mut function.vars,
            part,
            &atom,
            config.split,
        )?;
        split_by_phase.insert(
            split.stmt,
            Split {
                loop_at,
                carried: split.carried,
                part,
                merges,
                original_views,
                validation_bindings: Vec::new(),
            },
        );
    }
    let mut phases = Vec::new();
    for (index, statement) in function.body.iter_mut().enumerate() {
        let StmtKind::Parallel {
            vars,
            extents,
            body,
        } = &mut statement.kind
        else {
            return Err("every top-level statement of a kernel must be a `parallel` block".into());
        };
        let extents = extents
            .iter()
            .map(|e| e.as_constant().ok_or("parallel extent is not concrete"))
            .collect::<Result<Vec<_>, _>>()?;
        if extents.iter().any(|&e| e < 0 || e > i64::from(i32::MAX)) {
            return Err("parallel extent must fit a nonnegative Metal index".into());
        }
        let mut steps = vec![1u64; extents.len()];
        if let Some(step) = steps.last_mut() {
            *step = config.per_item as u64;
        }
        let mapping = if let Some(mappings) = mappings {
            let mapping = mappings[index].clone();
            if mapping
                .axes()
                .iter()
                .map(|a| a.logical_extent)
                .ne(extents.iter().map(|&n| n as u64))
            {
                return Err(
                    "selected work mapping disagrees with the normalized iteration domain".into(),
                );
            }
            mapping
        } else {
            WorkMapping::new(
                &extents.iter().map(|&e| e as u64).collect::<Vec<_>>(),
                &steps,
            )?
        };
        let base_items = i64::try_from(mapping.work_items())
            .map_err(|_| "work item count exceeds signed domain")?;
        let mut split = split_by_phase.remove(&index);
        let parts = if split.is_some() { config.split } else { 1 };
        let items = base_items
            .checked_mul(parts)
            .ok_or("split work item count overflow")?;
        if items > i64::from(u32::MAX) {
            return Err("work item count exceeds Metal slot index width".into());
        }
        let dispatch = GroupDispatch::new(items as u64, SUBGROUP as u64, config.sg_per_tg as u64)?;
        if dispatch.dispatched_lanes() / SUBGROUP as u64 > u64::from(u32::MAX) {
            return Err("padded work item count exceeds Metal slot index width".into());
        }
        for (&inner, axis) in vars.iter().zip(mapping.axes()).rev() {
            if axis.step == 1 {
                continue;
            }
            let VarKind::Index(atom) = &function.vars[inner].kind else {
                return Err("widening requires an index variable".into());
            };
            let atom = atom.clone();
            let base = Expr {
                kind: ExprKind::Var(inner),
                ty: Ty::Scalar(DType::I32),
                sym: Some(Sym::atom(atom.clone())),
                span: statement.span,
            };
            *body = seismic_lang::widen::apply_bounded(
                body,
                inner,
                &atom,
                i64::try_from(axis.step).map_err(|_| "mapping step exceeds Metal extent domain")?,
                i64::try_from(axis.logical_extent)
                    .map_err(|_| "mapping extent exceeds Metal index domain")?,
                &mut function.vars,
                &base,
            );
        }
        let positions = seismic_lang::normalize::bind_values(body, &mut function.vars);
        if let Some(split) = &mut split {
            split.loop_at = positions[split.loop_at];
            for view in &mut split.original_views {
                split
                    .validation_bindings
                    .extend(seismic_lang::normalize::bind_expression_values(
                        view,
                        &mut function.vars,
                    ));
            }
        }
        phases.push(Phase {
            mapping,
            parts,
            dispatch,
            merge_dispatch: split
                .as_ref()
                .map(|_| {
                    GroupDispatch::new(base_items as u64, SUBGROUP as u64, config.sg_per_tg as u64)
                })
                .transpose()?,
            split,
        });
    }
    Ok(Stage::Folds(std::sync::Arc::new(Prepared {
        function,
        source,
        config,
        phases,
        retained,
        partition_parameters,
        private_values: Vec::new(),
    })))
}

pub(crate) fn advance(
    stage: &Stage,
    select_fold: &mut dyn FnMut(&FoldChoice) -> Result<FoldOwnership, String>,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::normalize::loads::Site,
    ) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(
        &crate::storage::StorageDecision,
    ) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(
        &crate::reduction::Decision,
    ) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
    select_transfer: &mut dyn FnMut(&crate::terminal::transfer::Choice) -> Result<u8, String>,
    select_traversal: &mut dyn FnMut(&crate::terminal::traversal::Choice) -> Result<usize, String>,
) -> Result<Advance, String> {
    use std::sync::Arc;
    Ok(match stage {
        Stage::Transfers(prepared) => {
            let mut selections = Vec::new();
            for choice in &prepared.choices {
                let width = select_transfer(choice)?;
                if width == 0 || width > choice.maximum { return Err("invalid terminal vector transfer".into()); }
                selections.push(crate::terminal::transfer::Selection { choice: choice.clone(), width });
            }
            let mut execution = prepared.execution.clone();
            if selections.iter().any(|s| s.width != 1) { execution.invalidate_terminal(); }
            execution.transfers = selections;
            let choices = crate::terminal::traversal::choices(&crate::msl::prepare_execution(&execution)?.terminal)?;
            Advance::Stage(Stage::Traversals(Arc::new(PreparedTraversals { execution, choices })))
        }
        Stage::Traversals(prepared) => {
            let mut selections = Vec::new();
            for choice in &prepared.choices {
                let width = select_traversal(choice)?;
                if width == 0 || width > choice.iterations { return Err("invalid terminal loop traversal".into()); }
                selections.push(crate::terminal::traversal::Selection { choice: choice.clone(), width });
            }
            let mut execution = prepared.execution.clone();
            if selections.iter().any(|s| s.width != 1) { execution.invalidate_terminal(); }
            execution.traversals = selections;
            Advance::Execution(execution)
        }
        Stage::Folds(prepared) => {
            let Prepared {
                mut function,
                source,
                config,
                mut phases,
                retained,
                partition_parameters,
                ..
            } = (**prepared).clone();
            let mut selected_folds = Vec::new();
            let root_seed_sites = seismic_lang::reduction::structured::participants::root_seed_candidates(&function, SUBGROUP as u32);
            let wavefront_sites = seismic_lang::reduction::structured::participants::wavefront_candidates(&function, SUBGROUP as u32);
            for site in seismic_lang::reduction::structured::participants::candidates(
                &function,
                SUBGROUP as u32,
            ) {
                use seismic_lang::reduction::structured::participants::{Completion, SeedPlacement, Selection};
                let wavefront = wavefront_sites.contains(&site);
                let choice = FoldChoice { site, lanes: SUBGROUP as u32, wavefront, root_seed: root_seed_sites.contains(&site) };
                let selected = select_fold(&choice)?;
                use seismic_accounting::selection::Choices;
                if !(0..choice.len()).any(|i| choice.get(i) == Some(selected)) {
                    return Err("participant ownership is outside this fold's selected tree".into());
                }
                let (seed, completion) = match selected {
                    FoldOwnership::Serial => continue,
                    FoldOwnership::ParticipantsRootSeed => (SeedPlacement::AtRoot, Completion::RetainLeaves),
                    FoldOwnership::ParticipantsWavefrontRootSeed if wavefront => (SeedPlacement::AtRoot, Completion::CompleteWaves),
                    FoldOwnership::Participants => (SeedPlacement::LeadingLeaf, Completion::RetainLeaves),
                    FoldOwnership::ParticipantsInsertSeed => (SeedPlacement::InsertAfterSegments, Completion::RetainLeaves),
                    FoldOwnership::ParticipantsWavefront if wavefront => (SeedPlacement::LeadingLeaf, Completion::CompleteWaves),
                    FoldOwnership::ParticipantsWavefrontInsertSeed if wavefront => (SeedPlacement::InsertAfterSegments, Completion::CompleteWaves),
                    _ => return Err("wave completion is outside this fold's selected tree".into()),
                };
                selected_folds.push(Selection { site, seed, completion });
            }
            let refinement = seismic_lang::reduction::structured::participants::apply(
                &function,
                &selected_folds,
                SUBGROUP as u32,
            )?;
            let private_values = refinement.private_values;
            function = seismic_lang::reduction::structured::materialize(&refinement.function)?;
            for (root, phase) in function.body.iter_mut().zip(&mut phases) {
                let StmtKind::Parallel { body, .. } = &mut root.kind else {
                    unreachable!()
                };
                let positions = seismic_lang::normalize::lift_owned_reductions(body);
                if let Some(split) = &mut phase.split {
                    split.loop_at = positions[split.loop_at];
                }
            }
            for (root, phase) in function.body.iter_mut().zip(&mut phases) {
                let StmtKind::Parallel { body, .. } = &mut root.kind else {
                    unreachable!()
                };
                let positions = seismic_lang::normalize::remove_empty_ranges(body);
                if let Some(split) = &mut phase.split {
                    split.loop_at = positions[split.loop_at];
                }
            }
            Advance::Stage(Stage::Loads(Arc::new(Prepared {
                function,
                source,
                config,
                phases,
                retained,
                partition_parameters,
                private_values,
            })))
        }
        Stage::Loads(prepared) => {
            let Prepared {
                mut function,
                source,
                config,
                mut phases,
                retained,
                partition_parameters,
                private_values,
            } = (**prepared).clone();
            let load_modes = seismic_lang::normalize::loads::sites(&function.body)
                .iter()
                .enumerate()
                .map(|(index, site)| select_load(index, site))
                .collect::<Result<Vec<_>, _>>()?;
            seismic_lang::normalize::loads::resolve(&mut function.body, &load_modes)?;
            for phase in &mut phases {
                if let Some(split) = &mut phase.split {
                    // Validation bindings are consumed by separately represented views.
                    // Until that lifetime is unified with the phase, retain value snapshots.
                    seismic_lang::normalize::select_loads(&mut split.validation_bindings, false);
                }
            }
            let mut next_operation = 0;
            seismic_lang::normalize::identify(&mut function.body, &mut next_operation);
            for phase in &mut phases {
                if let Some(split) = &mut phase.split {
                    seismic_lang::normalize::identify(
                        &mut split.validation_bindings,
                        &mut next_operation,
                    );
                }
            }
            Advance::Stage(Stage::Storage(Arc::new(Prepared {
                function,
                source,
                config,
                phases,
                retained,
                partition_parameters,
                private_values,
            })))
        }
        Stage::Storage(prepared) => {
            let Prepared {
                function,
                phases,
                private_values,
                ..
            } = prepared.as_ref();
            let extra = phases
                .iter()
                .filter_map(|p| p.split.as_ref().map(|s| s.validation_bindings.as_slice()))
                .collect::<Vec<_>>();
            let storage =
                crate::storage::plan(&function.vars, &function.body, &extra, &mut |decision| {
                    if private_values.contains(&decision.variable) {
                        let mut decision = decision.clone();
                        decision.alternatives.retain(|p| {
                            *p == seismic_realization::dispatch::TilePlacement::Replicated
                        });
                        let placement = select(&decision)?;
                        decision.select(placement.clone())?;
                        Ok(placement)
                    } else {
                        select(decision)
                    }
                })?;
            Advance::Stage(Stage::Reductions(prepared.clone(), Arc::new(storage)))
        }
        Stage::Reductions(prepared, storage) => {
            let Prepared {
                function, phases, ..
            } = prepared.as_ref();
            let reductions = crate::reduction::plan(
                &function.vars,
                &function.body,
                &phases,
                &storage,
                select_reduction,
            )?;
            Advance::Stage(Stage::Allocations(
                prepared.clone(),
                storage.clone(),
                Arc::new(reductions),
            ))
        }
        Stage::Allocations(prepared, storage, reductions) => {
            let Prepared {
                function,
                source,
                config,
                phases,
                retained,
                partition_parameters,
                ..
            } = prepared.as_ref();
            let memory = crate::memory::plan_selected(
                &function.vars,
                &function.body,
                &phases,
                &storage,
                &reductions,
                config.max_threadgroup_bytes as u64,
                select_allocation,
            )?.with_retained(retained, phases)?;
            let execution = Execution {
                terminal: Default::default(),
                transfers: Vec::new(),
                traversals: Vec::new(),
                support: crate::support::Plan::new(),
                memory,
                reductions: (**reductions).clone(),
                storage: (**storage).clone(),
                function: function.clone(),
                source: source.clone(),
                config: config.clone(),
                phases: phases.clone(),
                retained: retained.clone(),
                partition_parameters: partition_parameters.clone(),
            };
            let choices = crate::terminal::transfer::choices(&crate::msl::prepare_execution(&execution)?.terminal);
            Advance::Stage(Stage::Transfers(Arc::new(PreparedTransfers { execution, choices })))
        }
    })
}
