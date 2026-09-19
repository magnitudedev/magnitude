//! IR-only application of explicit execution choices, before source emission.
//! This is an intermediate compiler boundary, not a complete Tuned IR: storage
//! lifetimes, intrinsic scheduling, and model-driven selection remain incomplete.
use seismic_lang::exec::{
    ir::*,
    lowered_ir::LoweredIr,
};
use seismic_realization::dispatch::{GroupDispatch, WorkMapping};

pub const SUBGROUP: i64 = 32;

/// Realization choices the model will close; defaults for now.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub sg_per_tg: i64,
    /// Device capacity facts, queried. The performance model will weigh
    /// parallelism against reuse; it is carried so no part of the compiler invents it.
    pub max_threads_per_threadgroup: i64,
    pub max_threadgroup_bytes: i64,
}

/// Piece counts of the inner owner regions of one launch body (`Parallel` statements nested
/// in a root `Parallel`), or `None` when it has none. Structural mapping: the launch piece is
/// one threadgroup and each inner owner one of its SIMD groups, so every inner owner region
/// of a launch has the same static piece counts, sits under uniform control only (the owner
/// body itself or ordered windows with static bounds), and contains no further owner region.
pub(crate) fn inner_owners(body: &[Stmt]) -> Result<Option<Vec<i64>>, String> {
    fn contains(body: &[Stmt]) -> bool {
        body.iter().any(|s| match &s.kind {
            StmtKind::Parallel { .. } => true,
            StmtKind::Owned { body, .. } | StmtKind::Range { body, .. } | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } => contains(body),
            StmtKind::If { then, els, .. } => contains(then) || contains(els),
            StmtKind::Assign { .. } | StmtKind::Expr(_) => false,
        })
    }
    fn visit(body: &[Stmt], found: &mut Option<Vec<i64>>) -> Result<(), String> {
        for statement in body {
            match &statement.kind {
                StmtKind::Parallel { extents, body, .. } => {
                    let counts = extents.iter().map(|e| e.as_constant().filter(|n| *n > 0)).collect::<Option<Vec<i64>>>().ok_or("an inner owner region needs static positive piece counts")?;
                    if contains(body) {
                        return Err("an inner owner region contains another owner region, which has no Metal mapping".into());
                    }
                    match found {
                        Some(existing) if *existing != counts => return Err("the inner owner regions of one launch have different piece counts; one threadgroup geometry cannot serve both".into()),
                        Some(_) => {}
                        None => *found = Some(counts),
                    }
                }
                StmtKind::Range { lo, hi, body, .. } => {
                    if contains(body) && (lo.as_constant().is_none() || hi.as_constant().is_none()) {
                        return Err("an inner owner region inside a loop with runtime bounds has no Metal mapping: every thread of the threadgroup must reach its barriers".into());
                    }
                    visit(body, found)?;
                }
                StmtKind::Owned { body, .. } | StmtKind::Lanes { body, .. } | StmtKind::LoadLoop { body, .. } if contains(body) => {
                    return Err("an inner owner region inside an element loop has no Metal mapping".into());
                }
                StmtKind::If { then, els, .. } if contains(then) || contains(els) => {
                    return Err("an inner owner region inside a branch has no Metal mapping".into());
                }
                _ => {}
            }
        }
        Ok(())
    }
    let mut found = None;
    visit(body, &mut found)?;
    Ok(found)
}

/// A selected parallel mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct Phase {
    pub mapping: WorkMapping,
    pub dispatch: GroupDispatch,
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
}

/// Emission belongs to one immutable terminal implementation. A code or
/// dispatch change replaces this owner before the result can be reused.
#[derive(Default)]
pub(crate) struct TerminalImplementation {
    pub(crate) emission: std::sync::OnceLock<Result<crate::msl::Emitted, String>>,
}

impl Execution {
    pub(crate) fn invalidate_terminal(&mut self) {
        self.terminal = Default::default();
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
    pub fn retained(&self) -> &[seismic_realization::phases::RetainedValue] {
        &self.retained
    }
}

/// Apply a caller's choices exactly, without native compilation or querying a device.
pub fn prepare_with_transfers(
    function: &LoweredIr,
    config: Config,
    select_load: &mut dyn FnMut(usize, &seismic_lang::exec::normalize::loads::Site) -> Result<LoadMode, String>,
    select: &mut dyn FnMut(&crate::storage::StorageDecision) -> Result<seismic_realization::dispatch::TilePlacement, String>,
    select_reduction: &mut dyn FnMut(&crate::reduction::Decision) -> Result<crate::reduction::Algorithm, String>,
    select_allocation: &mut dyn FnMut(&crate::memory::AllocationChoices) -> Result<usize, String>,
    select_transfer: &mut dyn FnMut(&crate::terminal::transfer::Choice) -> Result<u8, String>,
    select_traversal: &mut dyn FnMut(&crate::terminal::traversal::Choice) -> Result<usize, String>,
) -> Result<Execution, String> {
    let mut stage = prepare_initial(function, config)?;
    loop {
        match advance(
            &stage,
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

#[derive(Clone)]
pub(crate) struct Prepared {
    pub(crate) function: LoweredIr,
    pub(crate) source: LoweredIr,
    pub(crate) config: Config,
    pub(crate) phases: Vec<Phase>,
    pub(crate) retained: Vec<seismic_realization::phases::RetainedValue>,
}

/// Immutable existing execution boundaries. Later decisions retain their owning
/// computation and completed plans rather than reconstructing earlier phases.
pub(crate) enum Stage {
    Transfers(std::sync::Arc<PreparedTransfers>),
    Traversals(std::sync::Arc<PreparedTraversals>),
    Normalize(std::sync::Arc<Prepared>),
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
pub(crate) struct PreparedTransfers {
    execution: Execution,
    choices: Vec<crate::terminal::transfer::Choice>,
}
pub(crate) struct PreparedTraversals {
    execution: Execution,
    choices: Vec<crate::terminal::traversal::Choice>,
}
pub(crate) enum Advance {
    Stage(Stage),
    Execution(Execution),
}

pub(crate) fn prepare_initial(
    function: &LoweredIr,
    config: Config,
) -> Result<Stage, String> {
    if function.backend != "metal" {
        return Err("Metal execution requires Metal Lowered IR".into());
    }
    if config.sg_per_tg <= 0
        || config
            .sg_per_tg
            .checked_mul(SUBGROUP)
            .is_none_or(|n| n > config.max_threads_per_threadgroup)
        || config.max_threadgroup_bytes < 0
    {
        return Err("invalid Metal candidate or device resource limits".into());
    }
    let source = function.clone();
    let mut function = function.clone();
    seismic_lang::exec::normalize::work_domain(&mut function.body);
    let phase_plan = seismic_realization::phases::construct(&function)?;
    let retained = phase_plan.retained;
    function = phase_plan.function;
    let mut phases = Vec::new();
    for statement in function.body.iter_mut() {
        let StmtKind::Parallel {
            extents,
            body,
            ..
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
        // Inner owners are the trailing work axes: the work items of one threadgroup are the
        // inner owners of one launch piece, so the threadgroup holds exactly their product.
        let mut extents = extents;
        let owners = inner_owners(body)?;
        let items_per_group = match &owners {
            Some(counts) => {
                let per_group = counts.iter().try_fold(1i64, |n, c| n.checked_mul(*c)).ok_or("inner owner count overflow")?;
                if config.sg_per_tg != 1 {
                    return Err("a launch with inner owner regions needs one launch piece per threadgroup".into());
                }
                if per_group.checked_mul(SUBGROUP).is_none_or(|threads| threads > config.max_threads_per_threadgroup) {
                    return Err(format!("{per_group} inner owners per launch piece exceed the threadgroup capacity of {} threads", config.max_threads_per_threadgroup));
                }
                extents.extend(counts);
                per_group
            }
            None => config.sg_per_tg,
        };
        let steps = vec![1u64; extents.len()];
        let mapping = WorkMapping::new(
            &extents.iter().map(|&e| e as u64).collect::<Vec<_>>(),
            &steps,
        )?;
        let items = i64::try_from(mapping.work_items())
            .map_err(|_| "work item count exceeds signed domain")?;
        if items > i64::from(u32::MAX) {
            return Err("work item count exceeds Metal slot index width".into());
        }
        let dispatch = GroupDispatch::new(items as u64, SUBGROUP as u64, items_per_group as u64)?;
        if dispatch.dispatched_lanes() / SUBGROUP as u64 > u64::from(u32::MAX) {
            return Err("padded work item count exceeds Metal slot index width".into());
        }
        seismic_lang::exec::normalize::bind_values(body, &mut function.vars);
        phases.push(Phase {
            mapping,
            dispatch,
        });
    }
    Ok(Stage::Normalize(std::sync::Arc::new(Prepared {
        function,
        source,
        config,
        phases,
        retained,
    })))
}

pub(crate) fn advance(
    stage: &Stage,
    select_load: &mut dyn FnMut(
        usize,
        &seismic_lang::exec::normalize::loads::Site,
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
        Stage::Normalize(prepared) => {
            let Prepared {
                mut function,
                source,
                config,
                phases,
                retained,
            } = (**prepared).clone();
            for root in function.body.iter_mut() {
                let StmtKind::Parallel { body, .. } = &mut root.kind else {
                    unreachable!()
                };
                seismic_lang::exec::normalize::lift_owned_reductions(body);
            }
            for root in function.body.iter_mut() {
                let StmtKind::Parallel { body, .. } = &mut root.kind else {
                    unreachable!()
                };
                seismic_lang::exec::normalize::remove_empty_ranges(body);
            }
            Advance::Stage(Stage::Loads(Arc::new(Prepared {
                function,
                source,
                config,
                phases,
                retained,
            })))
        }
        Stage::Loads(prepared) => {
            let Prepared {
                mut function,
                source,
                config,
                phases,
                retained,
            } = (**prepared).clone();
            let load_modes = seismic_lang::exec::normalize::loads::sites(&function.body)
                .iter()
                .enumerate()
                .map(|(index, site)| select_load(index, site))
                .collect::<Result<Vec<_>, _>>()?;
            seismic_lang::exec::normalize::loads::resolve(&mut function.body, &load_modes)?;
            let mut next_operation = 0;
            seismic_lang::exec::normalize::identify(&mut function.body, &mut next_operation);
            Advance::Stage(Stage::Storage(Arc::new(Prepared {
                function,
                source,
                config,
                phases,
                retained,
            })))
        }
        Stage::Storage(prepared) => {
            let Prepared {
                function,
                ..
            } = prepared.as_ref();
            let storage = crate::storage::plan(&function.vars, &function.body, select)?;
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
            } = prepared.as_ref();
            let memory = crate::memory::plan_selected(
                &function.vars,
                &function.body,
                &phases,
                &storage,
                &reductions,
                &function.alias_requirements,
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
            };
            let choices = crate::terminal::transfer::choices(&crate::msl::prepare_execution(&execution)?.terminal);
            Advance::Stage(Stage::Transfers(Arc::new(PreparedTransfers { execution, choices })))
        }
    })
}
