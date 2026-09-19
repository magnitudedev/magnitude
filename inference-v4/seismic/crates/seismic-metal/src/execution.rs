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
impl seismic_accounting::choices::Choices for FoldChoice {
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
    pub retained: Option<RetainedSplitPhase>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RetainedSplitPhase {
    pub parts_symbol: String,
    pub selector: String,
    /// Scope holding the ordinary widened program. Prefix, stream and merge
    /// tail precede it and retain their original split identities.
    pub ordinary_at: usize,
}
#[derive(Clone)]
pub(crate) struct RetainedSplit {
    pub parts_symbol: String,
    pub selector: String,
    pub maximum: i64,
    pub ordinary: std::collections::BTreeMap<usize, RetainedOrdinary>,
}
#[derive(Clone)]
pub(crate) struct RetainedOrdinary {
    pub body: Vec<Stmt>,
    pub aliases: Vec<(VarId, VarId)>,
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
    pub(crate) implementation: Option<std::sync::Arc<crate::family::layout::Family>>,
    pub(crate) launch_parameters: Option<std::sync::Arc<Vec<crate::msl::parameters::Launch>>>,
    pub(crate) numeric_parameters: std::collections::BTreeMap<String, seismic_accounting::algebra::Value>,
    pub(crate) numeric_definitions: std::collections::BTreeMap<String, Sym>,
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

/// Emission belongs to one immutable terminal implementation. A code or
/// dispatch change replaces this owner before the result can be reused.
#[derive(Default)]
pub(crate) struct TerminalImplementation {
    pub(crate) emission: std::sync::OnceLock<Result<crate::msl::Emitted, String>>,
}

impl Execution {
    /// Install the selected coordinates and work counts without repeating IR
    /// normalization, widening, ownership, reduction or allocation planning.
    pub(crate) fn install_mappings(&mut self, mappings: &[WorkMapping], values: &[i64]) -> Result<(), String> {
        if mappings.len() != self.phases.len() { return Err("selected mapping count differs from retained phases".into()); }
        let read = |value: seismic_accounting::algebra::Value| -> Result<u64, String> {
            values.get(value.id().0).and_then(|&value| u64::try_from(value).ok())
                .filter(|&selected| value.bounds().0 <= selected && selected <= value.bounds().1)
                .ok_or_else(|| "missing or invalid original mapping operand".into())
        };
        let mut launch = 0usize;
        for (phase, mapping) in self.phases.iter_mut().zip(mappings) {
            let mut selected_work_items = None;
            let mut selected_merge_items = None;
            if let Some(parameters) = &self.launch_parameters {
                let original = parameters.get(launch).ok_or("selected mapping has no original launch definition")?;
                if original.axes.len() != mapping.axes().len() { return Err("selected mapping rank differs from its original definition".into()); }
                for (original, selected) in original.axes.iter().zip(mapping.axes()) {
                    if read(original.extent)? != selected.logical_extent || read(original.step)? != selected.step || read(original.count)? != selected.extent || read(original.stride)? != selected.stride {
                        return Err("selected mapping disagrees with original symbolic coordinates".into());
                    }
                }
                phase.parts = i64::try_from(read(original.parts)?).map_err(|_| "selected split exceeds integer range")?;
                if phase.parts <= 0 || (phase.split.is_none() && phase.parts != 1) {
                    return Err("selected split differs from retained phase topology".into());
                }
                selected_work_items = Some(read(original.work_items)?);
                if phase.merge_dispatch.is_some() {
                    let merge = parameters.get(launch + 1).ok_or("selected merge has no original launch definition")?;
                    if read(merge.parts)? != 1 { return Err("selected merge must own one completed item".into()); }
                    selected_merge_items = Some(read(merge.work_items)?);
                }
            } else if phase.mapping.axes().iter().map(|axis| axis.logical_extent)
                .ne(mapping.axes().iter().map(|axis| axis.logical_extent)) {
                return Err("selected mapping changed its original logical iteration domain".into());
            }
            let maximum_work_items = mapping.work_items().checked_mul(phase.parts as u64).ok_or("selected work domain overflow")?;
            let work_items = selected_work_items.unwrap_or(maximum_work_items);
            if work_items != 0 && work_items != maximum_work_items { return Err("selected launch work count differs from its selected coordinates".into()); }
            phase.mapping = mapping.clone();
            phase.dispatch = GroupDispatch::new(work_items, phase.dispatch.lanes_per_item, phase.dispatch.items_per_group)?;
            if let Some(dispatch) = &mut phase.merge_dispatch {
                let expected = if phase.parts > 1 && work_items != 0 { mapping.work_items() } else { 0 };
                let merge_items = selected_merge_items.unwrap_or(expected);
                if merge_items != expected { return Err("selected merge work count differs from its active split domain".into()); }
                *dispatch = GroupDispatch::new(merge_items, dispatch.lanes_per_item, dispatch.items_per_group)?;
            }
            launch += 1 + usize::from(phase.merge_dispatch.is_some());
        }
        if self.launch_parameters.as_ref().is_some_and(|parameters| parameters.len() != launch) {
            return Err("retained launch parameter count differs from selected phases".into());
        }
        self.memory = self.memory.redispatch(&self.phases)?;
        self.invalidate_terminal();
        Ok(())
    }
    /// Install dispatch equations already selected in the shared model. This
    /// consumes the retained witness directly rather than deriving another
    /// grouping family from a construction envelope during reconstruction.
    pub(crate) fn install_dispatches(&mut self, dispatches: &[GroupDispatch]) -> Result<(), String> {
        let mut index = 0usize;
        for phase in &mut self.phases {
            for dispatch in std::iter::once(&mut phase.dispatch).chain(phase.merge_dispatch.iter_mut()) {
                let selected = dispatches.get(index).ok_or("selected dispatch is missing")?;
                if selected.work_items != dispatch.work_items
                    || selected.lanes_per_item != dispatch.lanes_per_item {
                    return Err("selected dispatch changed retained work ownership".into());
                }
                if selected.threads_per_group > self.config.max_threads_per_threadgroup as u64 {
                    return Err("selected dispatch exceeds target thread capacity".into());
                }
                *dispatch = selected.clone();
                index += 1;
            }
        }
        if index != dispatches.len() { return Err("selected dispatch has no retained launch".into()); }
        self.config.sg_per_tg = dispatches.first().map_or(Ok(1), |dispatch|
            i64::try_from(dispatch.items_per_group).map_err(|_| "selected grouping exceeds integer range"))?;
        self.memory = self.memory.redispatch(&self.phases)?;
        self.invalidate_terminal();
        Ok(())
    }
    pub(crate) fn invalidate_terminal(&mut self) {
        self.terminal = Default::default();
    }
    /// Instantiate the retained backend computation from the same assignment
    /// as its terminal program. No lowering, preparation or decision discovery
    /// runs here; source choice, operation and storage identities are retained.
    pub(crate) fn install_source(&mut self, source: &LoweredIr, decomposition: &crate::tuning::Decomposition,
        values: &[i64]) -> Result<(), String> {
        if source.name != self.function.name || source.backend != self.function.backend {
            return Err("selected source differs from its retained Metal function".into());
        }
        let layout = self.implementation.as_ref().ok_or("selected Metal source has no retained implementation")?;
        let selection = crate::family::source::Selection::new(&self.function, &self.numeric_parameters, layout, values)?;
        let mut function = self.function.clone();
        let mut phases = self.phases.clone();
        if function.body.len() != phases.len() { return Err("selected source phase count differs from its retained mapping".into()); }
        for (phase_index, (statement, phase)) in function.body.iter_mut().zip(&mut phases).enumerate() {
            let StmtKind::Parallel { vars, extents, body } = &mut statement.kind else { return Err("selected Metal phase has no original work domain".into()); };
            if vars.len() != phase.mapping.axes().len() { return Err("selected Metal phase rank differs from its mapping".into()); }
            *extents = phase.mapping.axes().iter().map(|axis| i64::try_from(axis.logical_extent).map(Sym::constant)
                .map_err(|_| "selected Metal work extent exceeds its index type".to_string())).collect::<Result<Vec<_>, _>>()?;
            let expected_work = if selection.phase_active(phase_index)? {
                phase.mapping.work_items().checked_mul(u64::try_from(phase.parts).map_err(|_| "selected phase has a negative split count")?)
                    .ok_or("selected source work count overflow")?
            } else { 0 };
            if phase.dispatch.work_items != expected_work { return Err("selected source presence differs from its launch work count".into()); }
            if phase.dispatch.work_items == 0 {
                body.clear();
                phase.split = None;
                continue;
            }
            if let Some(split) = &mut phase.split {
                if phase.parts != decomposition.split { return Err("selected phase split count differs from its original decomposition".into()); }
                if let Some(retained) = &split.retained {
                    if selection.predicate(&retained.selector)? != (phase.parts > 1) {
                        return Err("selected split topology differs from its original compiler predicate".into());
                    }
                }
                let ordinary_at = split.retained.as_ref().map_or(body.len(), |retained| retained.ordinary_at);
                if split.loop_at >= ordinary_at || ordinary_at > body.len() { return Err("selected split lost its retained statement boundaries".into()); }
                if phase.parts == 1 {
                    if split.retained.is_none() { return Err("ordinary execution lacks its retained split alternative".into()); }
                    *body = selection.body(&body[ordinary_at..])?;
                    phase.split = None;
                } else {
                    let mut selected = selection.body(&body[..split.loop_at])?;
                    let loop_at = selected.len();
                    let stream = selection.body(&body[split.loop_at..=split.loop_at])?;
                    if stream.len() != 1 || !matches!(stream[0].kind, StmtKind::LoadLoop { .. }) {
                        return Err("selected split no longer contains its original stream".into());
                    }
                    selected.extend(stream);
                    selected.extend(selection.body(&body[split.loop_at + 1..ordinary_at])?);
                    *body = selected;
                    split.loop_at = loop_at;
                    split.retained = None;
                    let mut inputs = selection.body(&split.validation_bindings)?;
                    let validation_count = inputs.len();
                    inputs.extend(split.original_views.iter().cloned().map(|view| Stmt { id: None, span: view.span, kind: StmtKind::Expr(view) }));
                    seismic_lang::lowered_ir::specialize_statements(&mut inputs, &selection.parameters);
                    split.original_views = inputs.split_off(validation_count).into_iter().map(|statement| {
                        let StmtKind::Expr(view) = statement.kind else { unreachable!() }; view
                    }).collect();
                    split.validation_bindings = inputs;
                }
            } else { *body = selection.body(body)?; }
        }
        function.specialize_parameters(&selection.parameters);
        if function.params.get(..source.params.len()) != Some(source.params.as_slice()) {
            return Err("selected Metal function ABI differs from its original source".into());
        }
        function.selections = source.selections.clone();
        function.decisions = source.decisions.clone();
        function.shapes = source.shapes.clone();
        self.function = function;
        self.source = source.clone();
        self.phases = phases;
        self.config.split = decomposition.split;
        self.config.per_item = decomposition.per_item;
        self.config.tile_piece = decomposition.tile_piece;
        self.invalidate_terminal();
        Ok(())
    }
    /// Install the specialization of an already accounted typed terminal family.
    /// Every geometry and backing quantity must match this selected execution.
    pub(crate) fn install_terminal(&mut self, emitted: crate::msl::Emitted) -> Result<(), String> {
        let indices = self.phases.iter().map(|phase| phase.split.iter().map(|split| split.part).collect()).collect::<Vec<_>>();
        seismic_lang::verify::executable_phases(&self.function, &indices)?;
        let dispatches = self.phases.iter().flat_map(|phase| std::iter::once(&phase.dispatch).chain(phase.merge_dispatch.as_ref())).collect::<Vec<_>>();
        if emitted.launches.len() != dispatches.len() || emitted.launches.len() != self.memory.launches().len() { return Err("terminal family launch count differs from selected execution".into()); }
        for ((launch, dispatch), memory) in emitted.launches.iter().zip(dispatches).zip(self.memory.launches()) {
            if launch.dispatch.as_ref() != Some(dispatch) || launch.threadgroups != dispatch.groups || launch.threads_per_threadgroup != dispatch.threads_per_group
                || launch.declared_threadgroup_bytes != memory.shared_bytes_per_group
                || launch.tiles.iter().ne(memory.arrays.iter().map(|array| &array.declaration)) {
                return Err("terminal family geometry or storage differs from selected execution".into());
            }
        }
        if emitted.scratch.len() != self.memory.scratch().len() || emitted.scratch_bindings.len() != self.memory.scratch().len()
            || self.memory.scratch().iter().enumerate().any(|(index, allocation)| {
                emitted.scratch[index] != allocation.bytes || emitted.scratch_bindings[index].bytes != allocation.bytes
                    || emitted.scratch_bindings[index].alignment != allocation.dtype.bytes() as usize
            }) {
            return Err("terminal family scratch ABI differs from selected execution".into());
        }
        emitted.terminal.validate_typed()?;
        self.terminal = std::sync::Arc::new(TerminalImplementation { emission: std::sync::OnceLock::from(Ok(emitted)) });
        Ok(())
    }
    /// Compare the actual prepared implementation independently of whether its
    /// lazy target emission has already been requested.
    pub(crate) fn same_implementation(&self, other: &Self) -> bool {
        match (&self.implementation, &other.implementation) {
            (Some(left), Some(right)) if !std::sync::Arc::ptr_eq(left, right) => return false,
            (None, Some(_)) | (Some(_), None) => return false,
            _ => {},
        }
        match (&self.launch_parameters, &other.launch_parameters) {
            (Some(left), Some(right)) if !std::sync::Arc::ptr_eq(left, right) => return false,
            (None, Some(_)) | (Some(_), None) => return false,
            _ => {},
        }
        self.numeric_parameters.len() == other.numeric_parameters.len()
            && self.numeric_definitions == other.numeric_definitions
            && self.numeric_parameters.iter().all(|(name, value)| other.numeric_parameters.get(name)
                .is_some_and(|other| value.id() == other.id() && value.bounds() == other.bounds()))
            && self.transfers == other.transfers && self.traversals == other.traversals
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
    pub(crate) function: LoweredIr,
    pub(crate) source: LoweredIr,
    pub(crate) config: Config,
    pub(crate) phases: Vec<Phase>,
    pub(crate) retained: Vec<seismic_realization::phases::RetainedValue>,
    pub(crate) partition_parameters: Vec<(usize, DType)>,
    pub(crate) private_values: Vec<usize>,
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
    /// Hand the fold boundary's immutable prepared computation to the retained
    /// region constructor without repeating initial normalization or work mapping.
    pub(crate) fn prepared_folds(&self) -> Option<&Prepared> {
        match self { Self::Folds(prepared) => Some(prepared), _ => None }
    }
    /// Local definitions visible at this boundary. Independent folds, loads
    /// and materializations do not depend on an earlier selected ordinal.
    pub(crate) fn local_domains(&self) -> Result<Vec<crate::choices::Domain>, String> {
        use crate::choices::{Decision, Domain};
        use seismic_accounting::choices::Choices;
        let definitions = match self {
            Self::Folds(prepared) => {
                let roots = seismic_lang::reduction::structured::participants::root_seed_candidates(&prepared.function, SUBGROUP as u32);
                let waves = seismic_lang::reduction::structured::participants::wavefront_candidates(&prepared.function, SUBGROUP as u32);
                seismic_lang::reduction::structured::participants::candidates(&prepared.function, SUBGROUP as u32)
                    .into_iter().map(|site| Decision::Fold(FoldChoice { site, lanes: SUBGROUP as u32,
                        wavefront: waves.contains(&site), root_seed: roots.contains(&site) })).collect()
            }
            Self::Loads(prepared) => seismic_lang::normalize::loads::sites(&prepared.function.body)
                .into_iter().enumerate().filter(|(_,site)| site.selected.is_none() && site.can_borrow)
                .map(|(site,definition)| Decision::Load(seismic_lang::normalize::loads::Choice {
                    site, variable: definition.variable })).collect(),
            Self::Storage(_) => self.storage_family()?.ok_or("storage stage lost its local family")?
                .decisions().iter().cloned().map(Decision::Storage).collect(),
            Self::Transfers(prepared) => prepared.choices.iter().cloned().map(Decision::Transfer).collect(),
            Self::Traversals(prepared) => prepared.choices.iter().cloned().map(Decision::Traversal).collect(),
            Self::Reductions(..) | Self::Allocations(..) => Vec::new(),
        };
        let definitions = definitions.into_iter().map(|decision| Domain { decision }).collect::<Vec<_>>();
        if definitions.iter().any(|domain| domain.len() == 0) { return Err("empty retained Metal implementation domain".into()); }
        Ok(definitions)
    }
    pub(crate) fn storage_family(&self) -> Result<Option<std::sync::Arc<crate::storage::StorageFamily>>, String> {
        let Stage::Storage(prepared) = self else { return Ok(None) };
        let extra = prepared.phases.iter().filter_map(|phase| phase.split.as_ref().map(|split| split.validation_bindings.as_slice())).collect::<Vec<_>>();
        let mut family = crate::storage::StorageFamily::derive(&prepared.function.vars, &prepared.function.body, &extra)?;
        family.force_replicated(&prepared.private_values)?;
        Ok(Some(std::sync::Arc::new(family)))
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
    prepare_with_parameters(function, config, mappings, &Default::default(), &Default::default(), None)
}
pub(crate) fn prepare_retained(
    function: &LoweredIr,
    config: Config,
    mappings: &[WorkMapping],
    parameters: &std::collections::BTreeMap<String, seismic_accounting::algebra::Value>,
    selectors: &std::collections::BTreeSet<VarId>,
    retained_split: Option<&RetainedSplit>,
) -> Result<Stage, String> {
    let numeric = parameters.iter().map(|(name, value)| {
        let (minimum, maximum) = value.bounds();
        Ok((name.clone(), (i64::try_from(minimum).map_err(|_| "Metal parameter minimum exceeds signed range")?,
            i64::try_from(maximum).map_err(|_| "Metal parameter maximum exceeds signed range")?)))
    }).collect::<Result<std::collections::BTreeMap<_, _>, String>>()?;
    prepare_with_parameters(function, config, Some(mappings), &numeric, selectors, retained_split)
}
fn prepare_with_parameters(
    function: &LoweredIr,
    config: Config,
    mappings: Option<&[WorkMapping]>,
    numeric: &std::collections::BTreeMap<String, (i64, i64)>,
    selectors: &std::collections::BTreeSet<VarId>,
    retained_split: Option<&RetainedSplit>,
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
    let maximum_parts = retained_split.map_or(config.split, |split| split.maximum);
    if maximum_parts > 1 && retained_split.is_none() {
        function = seismic_lang::reduction::structured::materialize(&function)?;
    }
    let phase_plan = seismic_realization::phases::construct_retained(&function, numeric, selectors)?;
    let handoffs = phase_plan.handoffs;
    let retained = phase_plan.retained;
    function = phase_plan.function;
    if mappings.is_some_and(|m| m.len() != function.body.len()) {
        return Err("one work mapping is required per normalized phase".into());
    }
    let splits = if maximum_parts > 1 {
        seismic_lang::split::split_candidates(&function.body, &function.vars)
    } else {
        Vec::new()
    };
    if maximum_parts > 1 && splits.is_empty() {
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
        let loop_at = if let Some(retained) = retained_split {
            seismic_lang::split::parameterized::narrow_range(
                &split,
                &mut function.body,
                &mut function.vars,
                part,
                &atom,
                &seismic_lang::sym::Sym::param(&retained.parts_symbol),
            )?
        } else {
            seismic_lang::split::narrow_range(
                &split,
                &mut function.body,
                &mut function.vars,
                part,
                &atom,
                config.split,
            )?
        };
        let retained = if let Some(retained) = retained_split {
            let ordinary = retained.ordinary.get(&split.stmt).ok_or("split phase has no retained ordinary program")?;
            let handoff = handoffs.get(split.stmt).ok_or("split phase lost its retained value handoff")?;
            let mut ordinary_body = ordinary.body.clone();
            let Some(Stmt { kind: StmtKind::If { then, els, .. }, .. }) = ordinary_body.last_mut() else {
                return Err("ordinary split alternative lost its compiler guard".into());
            };
            if !then.is_empty() { return Err("ordinary split alternative has an unexpected split arm".into()); }
            let mut completed = seismic_lang::widen::parameterized::remap_bindings(&handoff.restores, &ordinary.aliases);
            completed.append(els);
            completed.extend(seismic_lang::widen::parameterized::remap_bindings(&handoff.publications, &ordinary.aliases));
            *els = completed;
            let StmtKind::Parallel { body, .. } = &mut function.body[split.stmt].kind else { unreachable!() };
            let ordinary_at = body.len(); body.extend(ordinary_body);
            Some(RetainedSplitPhase { parts_symbol: retained.parts_symbol.clone(), selector: retained.selector.clone(), ordinary_at })
        } else { None };
        split_by_phase.insert(
            split.stmt,
            Split {
                loop_at,
                carried: split.carried,
                part,
                merges,
                original_views,
                validation_bindings: Vec::new(),
                retained,
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
        let parts = if split.is_some() { maximum_parts } else { 1 };
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
            if let Some(retained) = &mut split.retained { retained.ordinary_at = positions[retained.ordinary_at]; }
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
                use seismic_accounting::choices::Choices;
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
                    if let Some(retained) = &mut split.retained { retained.ordinary_at = positions[retained.ordinary_at]; }
                }
            }
            for (root, phase) in function.body.iter_mut().zip(&mut phases) {
                let StmtKind::Parallel { body, .. } = &mut root.kind else {
                    unreachable!()
                };
                let positions = seismic_lang::normalize::remove_empty_ranges(body);
                if let Some(split) = &mut phase.split {
                    split.loop_at = positions[split.loop_at];
                    if let Some(retained) = &mut split.retained { retained.ordinary_at = positions[retained.ordinary_at]; }
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
                implementation: None,
                launch_parameters: None,
                numeric_parameters: Default::default(),
                numeric_definitions: Default::default(),
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
