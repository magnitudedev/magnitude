//! Allocations and memory barriers planned from selected IR, before emission.
//! Barrier sites specify static program locations, not dynamic execution counts.
//! These publication boundaries are not yet a minimal synchronization schedule.
//! Lexical scopes describe declaration lifetime boundaries, not native register
//! allocation or a proof that all private arrays coexist. Intrinsic fragments
//! remain explicit unmodeled storage, rather than contributing a false zero.
use crate::{
    execution::Phase,
    reduction::{ReductionPlan, Site},
    storage::StoragePlan,
};
use seismic_lang::{ir::*, sym::Sym, types::Ty};
use seismic_realization::dispatch::{GroupDispatch, TileDeclaration, TilePlacement};
use seismic_realization::execution::Multiplicity;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use seismic_realization::memory::{
    AllocationId, ArrayAllocation, Barrier, BarrierPurpose, BarrierSite, ControlValue,
    LaunchMemory, MemoryPlan, MemorySpace, Purpose, Scope, ScratchAllocation,
};

pub fn plan(
    vars: &[Var],
    body: &[Stmt],
    phases: &[Phase],
    storage: &StoragePlan,
    reductions: &ReductionPlan,
    shared_limit: u64,
) -> Result<MemoryPlan, String> {
    struct Planner<'a> {
        storage: &'a StoragePlan,
        reductions: &'a ReductionPlan,
        bound: HashMap<VarId, Option<TilePlacement>>,
        barriers: BTreeMap<BarrierSite, Barrier>,
        ids: HashSet<AllocationId>,
        arrays: Vec<ArrayAllocation>,
        scope: Vec<Scope>,
        opaque: Vec<OperationId>,
        executions: Arc<Multiplicity<ControlValue>>,
        /// Known potentially partial owned-loop participation. False does not
        /// prove convergence of arbitrary branch control flow.
        partial_owned: bool,
    }
    impl Planner<'_> {
        fn barrier(
            &mut self,
            operation: OperationId,
            variable: VarId,
            purpose: BarrierPurpose,
            memory: MemorySpace,
        ) -> Result<(), String> {
            if self.partial_owned {
                return Err(format!("memory barrier {purpose:?} at {operation:?} requires full-lane participation inside an owned domain"));
            }
            let site = BarrierSite {
                operation,
                variable,
                purpose,
            };
            if self
                .barriers
                .insert(
                    site,
                    Barrier {
                        memory,
                        scope: self.scope.clone(),
                        executions: self.executions.clone(),
                    },
                )
                .is_some()
            {
                return Err("duplicate memory barrier identity".into());
            }
            Ok(())
        }
        fn publish(
            &mut self,
            operation: OperationId,
            variable: VarId,
            purpose: BarrierPurpose,
            placement: Option<TilePlacement>,
        ) -> Result<(), String> {
            if placement == Some(TilePlacement::GroupShared) {
                self.barrier(operation, variable, purpose, MemorySpace::Threadgroup)?;
            }
            Ok(())
        }
        fn snapshot(
            &mut self,
            operation: OperationId,
            variable: VarId,
            purpose: Purpose,
        ) -> Result<(), String> {
            self.materialize(operation, variable, purpose)?;
            self.publish(
                operation,
                variable,
                BarrierPurpose::Snapshot(purpose),
                Some(self.storage.declaration(variable)?.placement.clone()),
            )
        }
        fn allocate(
            &mut self,
            operation: OperationId,
            variable: VarId,
            purpose: Purpose,
            declaration: TileDeclaration,
        ) -> Result<(), String> {
            let id = AllocationId {
                operation,
                variable,
                purpose,
            };
            if !self.ids.insert(id) {
                return Err("duplicate allocation identity".into());
            }
            self.arrays.push(ArrayAllocation {
                id,
                declaration: declaration.clone(),
                scope: if declaration.placement
                    == seismic_realization::dispatch::TilePlacement::GroupShared
                {
                    self.scope[..1].to_vec()
                } else {
                    self.scope.clone()
                },
            });
            Ok(())
        }
        fn materialize(
            &mut self,
            operation: OperationId,
            var: VarId,
            purpose: Purpose,
        ) -> Result<(), String> {
            self.allocate(
                operation,
                var,
                purpose,
                self.storage.declaration(var)?.clone(),
            )
        }
        fn nested(
            &mut self,
            body: &[Stmt],
            scope: Scope,
            count: Multiplicity<ControlValue>,
        ) -> Result<(), String> {
            let outer = self.executions.clone();
            self.executions = Multiplicity::product(outer.clone(), Arc::new(count));
            self.scope.push(scope);
            self.body(body)?;
            self.scope.pop();
            self.executions = outer;
            Ok(())
        }
        fn body(&mut self, body: &[Stmt]) -> Result<(), String> {
            for statement in body {
                let operation = statement
                    .id
                    .ok_or("allocation planning requires operation identities")?;
                match &statement.kind {
                    StmtKind::LoadLoop {
                        vars,
                        views,
                        axis,
                        capacity,
                        modes,
                        body,
                        ..
                    } => {
                        // Whole-axis empty streams have no emitted body. Chunked
                        // streams retain a loop and its declarations even when empty.
                        if capacity.is_none()
                            && views
                                .first()
                                .and_then(|v| v.ty.shaped())
                                .and_then(|s| s.shape.get(*axis))
                                .and_then(|s| s.as_constant())
                                == Some(0)
                        {
                            continue;
                        }
                        let modes = modes
                            .as_ref()
                            .filter(|m| m.len() == vars.len())
                            .ok_or("unresolved streamed allocations")?;
                        let outer = self.executions.clone();
                        if let Some(capacity) = capacity {
                            let extent = views
                                .first()
                                .and_then(|v| v.ty.shaped())
                                .and_then(|s| s.shape.get(*axis))
                                .ok_or("stream memory domain has no extent")?;
                            if *capacity <= 0 {
                                return Err("stream memory capacity must be positive".into());
                            }
                            let chunks = extent
                                .add(&Sym::constant(*capacity - 1))
                                .quot(&Sym::constant(*capacity));
                            self.executions = Multiplicity::product(
                                outer.clone(),
                                Arc::new(Multiplicity::Iterations {
                                    lower: ControlValue::Integer(Sym::constant(0)),
                                    upper: ControlValue::Integer(chunks),
                                }),
                            );
                            self.scope.push(Scope::Body(operation));
                        }
                        for (var, mode) in vars.iter().zip(modes) {
                            if *mode == LoadMode::Materialize {
                                self.snapshot(operation, *var, Purpose::Value)?;
                            }
                            self.bound.insert(
                                *var,
                                if *mode == LoadMode::Materialize {
                                    Some(self.storage.declaration(*var)?.placement.clone())
                                } else {
                                    None
                                },
                            );
                        }
                        self.body(body)?;
                        if capacity.is_some() {
                            self.scope.pop();
                        }
                        self.executions = outer;
                    }
                    StmtKind::Assign { target, value, .. } => {
                        let ExprKind::Var(var) = target.kind else {
                            continue;
                        };
                        let previous = self.bound.get(&var).cloned();
                        match &value.kind {
                            ExprKind::TileAlloc { .. } => {
                                self.materialize(operation, var, Purpose::Value)?;
                                self.bound.insert(
                                    var,
                                    Some(self.storage.declaration(var)?.placement.clone()),
                                );
                            }
                            ExprKind::Load {
                                mode: LoadMode::Materialize,
                                ..
                            } => {
                                self.snapshot(operation, var, Purpose::Value)?;
                                if let Some(placement) = previous {
                                    self.publish(operation, var, BarrierPurpose::Copy, placement)?;
                                } else {
                                    self.bound.insert(
                                        var,
                                        Some(self.storage.declaration(var)?.placement.clone()),
                                    );
                                }
                            }
                            ExprKind::Var(_) if matches!(target.ty, Ty::Tile(_)) => {
                                if previous.is_none() {
                                    self.materialize(operation, var, Purpose::Value)?;
                                    self.bound.insert(
                                        var,
                                        Some(self.storage.declaration(var)?.placement.clone()),
                                    );
                                }
                                self.publish(
                                    operation,
                                    var,
                                    BarrierPurpose::Copy,
                                    self.bound[&var].clone(),
                                )?;
                            }
                            ExprKind::Builtin {
                                name: Builtin::Reduce,
                                ..
                            } => {
                                let selected = self.reductions.get(Site {
                                    operation,
                                    output: var,
                                })?;
                                if selected.decision.materialize_input {
                                    self.snapshot(
                                        operation,
                                        selected.decision.input,
                                        Purpose::ReductionInput,
                                    )?;
                                    self.bound.insert(
                                        selected.decision.input,
                                        Some(
                                            self.storage
                                                .declaration(selected.decision.input)?
                                                .placement
                                                .clone(),
                                        ),
                                    );
                                }
                                if let Some(declaration) = &selected.output {
                                    self.allocate(
                                        operation,
                                        var,
                                        Purpose::Value,
                                        declaration.clone(),
                                    )?;
                                    if previous.is_none() {
                                        self.bound.insert(var, Some(declaration.placement.clone()));
                                    }
                                }
                                if let Some(placement) = previous {
                                    self.publish(operation, var, BarrierPurpose::Copy, placement)?;
                                }
                            }
                            ExprKind::Intrinsic { op: name, .. } if *name == seismic_lang::intrinsics::Operation::Matrix => {
                                self.opaque.push(operation)
                            }
                            _ => {}
                        }
                        self.bound.entry(var).or_insert(None);
                    }
                    StmtKind::Owned { tile, body, .. } => {
                        let ExprKind::Var(var) = tile.kind else {
                            return Err("owned memory domain has no binding".into());
                        };
                        let placement = self
                            .bound
                            .get(&var)
                            .cloned()
                            .ok_or("owned memory domain has no placement")?;
                        let outer_partial = self.partial_owned;
                        self.partial_owned |= placement != Some(TilePlacement::Replicated);
                        self.nested(body, Scope::Body(operation), Multiplicity::Unknown { reason: "owned-domain lane participation and runtime tile extents are unresolved".into() })?;
                        self.partial_owned = outer_partial;
                        self.publish(operation, var, BarrierPurpose::Owned, placement)?;
                    }
                    StmtKind::Expr(Expr {
                        kind: ExprKind::Intrinsic { op: name, args },
                        ..
                    }) if *name == seismic_lang::intrinsics::Operation::MatrixStore => {
                        let operand = args.get(1).ok_or("fragment store has no destination")?;
                        let ExprKind::Var(var) = operand.kind else {
                            return Err("fragment store requires a shared tile binding".into());
                        };
                        if self.bound.get(&var) != Some(&Some(TilePlacement::GroupShared)) {
                            return Err("fragment store requires owned shared tile storage".into());
                        }
                        let ExprKind::Var(_) = args[0].kind else {
                            return Err("fragment store has no fragment binding".into());
                        };
                        self.barrier(
                            operation,
                            var,
                            BarrierPurpose::IntrinsicStore,
                            MemorySpace::Threadgroup,
                        )?;
                    }
                    StmtKind::Parallel { body, .. } => {
                        self.nested(body, Scope::Body(operation), Multiplicity::Constant(1))?
                    }
                    StmtKind::Range { lo, hi, body, .. } => self.nested(
                        body,
                        Scope::Body(operation),
                        Multiplicity::Iterations {
                            lower: ControlValue::Integer(lo.clone()),
                            upper: ControlValue::Integer(hi.clone()),
                        },
                    )?,
                    StmtKind::Lanes {
                        extent,
                        width,
                        body,
                        ..
                    } => {
                        let width = u64::try_from(*width)
                            .ok()
                            .filter(|w| *w > 0)
                            .ok_or("invalid lane width")?;
                        let run = (crate::execution::SUBGROUP as u64)
                            .checked_mul(width)
                            .and_then(|n| i64::try_from(n).ok())
                            .ok_or("lane domain overflow")?;
                        self.nested(
                            body,
                            Scope::Body(operation),
                            Multiplicity::Product(
                                Arc::new(Multiplicity::Constant(width)),
                                Arc::new(Multiplicity::Iterations {
                                    lower: ControlValue::Integer(Sym::constant(0)),
                                    upper: ControlValue::Integer(extent.quot(&Sym::constant(run))),
                                }),
                            ),
                        )?;
                    }
                    StmtKind::If { cond, then, els } => {
                        self.nested(
                            then,
                            Scope::Then(operation),
                            Multiplicity::Predicate {
                                value: ControlValue::Predicate(Box::new(cond.clone())),
                                expected: true,
                            },
                        )?;
                        self.nested(
                            els,
                            Scope::Else(operation),
                            Multiplicity::Predicate {
                                value: ControlValue::Predicate(Box::new(cond.clone())),
                                expected: false,
                            },
                        )?;
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        fn finish(
            &mut self,
            dispatch: &GroupDispatch,
            limit: u64,
            predecessor: Option<usize>,
        ) -> Result<LaunchMemory, String> {
            let mut private = 0u64;
            let mut shared = 0u64;
            for array in &self.arrays {
                let layout = array.declaration.layout(dispatch)?;
                private = private
                    .checked_add(layout.private_bytes_per_lane)
                    .ok_or("private declaration sum overflow")?;
                shared = shared
                    .checked_add(layout.shared_bytes_per_group)
                    .ok_or("shared declaration sum overflow")?;
            }
            if shared > limit {
                return Err(format!("planned realization needs {shared} bytes of threadgroup memory, over this device's {limit}"));
            }
            Ok(LaunchMemory {
                predecessor,
                arrays: std::mem::take(&mut self.arrays),
                barriers: std::mem::take(&mut self.barriers),
                declared_private_bytes_per_lane: private,
                shared_bytes_per_group: shared,
                unmodeled_fragments: std::mem::take(&mut self.opaque),
            })
        }
    }
    if body.len() != phases.len() {
        return Err("allocation plan phase/domain mismatch".into());
    }
    let mut planner = Planner {
        storage,
        reductions,
        bound: vars
            .iter()
            .enumerate()
            .filter_map(|(i, v)| matches!(v.kind, VarKind::Param(_)).then_some((i, None)))
            .collect(),
        barriers: BTreeMap::new(),
        ids: HashSet::new(),
        arrays: Vec::new(),
        scope: Vec::new(),
        opaque: Vec::new(),
        executions: Arc::new(Multiplicity::Constant(1)),
        partial_owned: false,
    };
    let mut launches = Vec::new();
    let mut scratch = Vec::new();
    for (phase_index, (root, phase)) in body.iter().zip(phases).enumerate() {
        let operation = root.id.ok_or("phase has no operation identity")?;
        let StmtKind::Parallel { body, .. } = &root.kind else {
            return Err("allocation phase is not a parallel domain".into());
        };
        planner.scope = vec![Scope::Body(operation)];
        if let Some(split) = &phase.split {
            for &variable in &split.carried {
                let Ty::Tile(tile) = &vars[variable].ty else {
                    return Err("split scratch requires a tile value".into());
                };
                let dtype = tile
                    .elem
                    .read_dtype()
                    .ok_or("split scratch dtype is unresolved")?;
                let elements_per_item = tile.shape.iter().try_fold(1u64, |size, extent| {
                    let extent = extent
                        .as_constant()
                        .and_then(|n| u64::try_from(n).ok())
                        .ok_or("split scratch capacity must be nonnegative and static")?;
                    size.checked_mul(extent)
                        .ok_or("split scratch capacity overflow")
                })?;
                let work_items = phase.mapping.work_items();
                let parts =
                    u64::try_from(phase.parts).map_err(|_| "invalid split scratch part count")?;
                let bytes = work_items
                    .checked_mul(parts)
                    .and_then(|n| n.checked_mul(elements_per_item))
                    .and_then(|n| n.checked_mul(u64::from(dtype.bytes())))
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or("split scratch size overflow")?;
                let producer = launches.len();
                scratch.push(ScratchAllocation {
                    index: scratch.len(),
                    phase: phase_index,
                    variable,
                    dtype,
                    elements_per_item,
                    work_items,
                    parts,
                    bytes,
                    producer,
                    consumer: producer + 1,
                });
            }
            let (prefix, suffix) = body
                .split_at_checked(split.loop_at)
                .ok_or("allocation split position is invalid")?;
            let (stream, tail) = suffix
                .split_first()
                .ok_or("allocation split stream is missing")?;
            planner.body(prefix)?;
            planner.body(&split.validation_bindings)?;
            planner.body(std::slice::from_ref(stream))?;
            launches.push(planner.finish(
                &phase.dispatch,
                shared_limit,
                launches.len().checked_sub(1),
            )?);
            for var in &split.carried {
                planner.materialize(operation, *var, Purpose::Merge)?;
                let placement = Some(storage.declaration(*var)?.placement.clone());
                planner.publish(operation, *var, BarrierPurpose::Merge, placement.clone())?;
                planner.bound.insert(*var, placement);
            }
            planner.body(tail)?;
            launches.push(
                planner.finish(
                    phase
                        .merge_dispatch
                        .as_ref()
                        .ok_or("split merge dispatch is missing")?,
                    shared_limit,
                    launches.len().checked_sub(1),
                )?,
            );
        } else {
            planner.body(body)?;
            launches.push(planner.finish(
                &phase.dispatch,
                shared_limit,
                launches.len().checked_sub(1),
            )?);
        }
    }
    MemoryPlan::new(launches, scratch)
}
