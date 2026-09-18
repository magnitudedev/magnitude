//! Allocations and memory barriers planned from selected IR, before emission.
//! Barrier sites specify static program locations, not dynamic execution counts.
//! These publication boundaries are not yet a minimal synchronization schedule.
//! Lexical scopes describe declaration lifetime boundaries, not native register
//! allocation or a proof that all private arrays coexist. Fragment payloads and
//! collective implementation requests are explicit without guessing native registers.
use crate::collective::{Collective, FragmentAllocation, Implementation, Site as CollectiveSite};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Purpose {
    Value,
    ReductionInput,
    Merge,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId {
    pub operation: OperationId,
    pub variable: VarId,
    pub purpose: Purpose,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Body(OperationId),
    Then(OperationId),
    Else(OperationId),
}
#[derive(Clone, Debug)]
pub struct ArrayAllocation {
    pub id: AllocationId,
    pub declaration: TileDeclaration,
    pub scope: Vec<Scope>,
}
/// Memory accesses ordered among lanes of one SIMD group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySpace {
    Threadgroup,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BarrierPurpose {
    Snapshot(Purpose),
    Copy,
    Owned,
    Merge,
    IntrinsicStore,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarrierSite {
    pub operation: OperationId,
    pub variable: VarId,
    pub purpose: BarrierPurpose,
}
/// References to structured IR, before scalar SSA or native emission.
#[derive(Clone, Debug)]
pub enum ControlValue {
    Integer(seismic_lang::sym::Sym),
    Predicate(Box<seismic_lang::ir::Expr>),
}
#[derive(Clone, Debug)]
pub struct Barrier {
    pub memory: MemorySpace,
    pub scope: Vec<Scope>,
    /// Per-work-item executions, conditional on valid collective participation.
    pub executions: Arc<Multiplicity<ControlValue>>,
}
#[derive(Clone, Debug)]
pub struct LaunchMemory {
    pub prologue: crate::support::LaunchRecipe,
    /// This launch must observe completion of its predecessor before it starts.
    pub predecessor: Option<usize>,
    pub arrays: Vec<ArrayAllocation>,
    pub barriers: BTreeMap<BarrierSite, Barrier>,
    /// Sum of declared private arrays per lane, not a simultaneous residency claim.
    pub declared_private_bytes_per_lane: u64,
    /// Shared arrays are hoisted to launch scope by the current emission contract.
    pub shared_bytes_per_group: u64,
    pub fragments: Vec<crate::collective::FragmentAllocation>,
    pub collectives: BTreeMap<crate::collective::Site, crate::collective::Collective>,
}
/// Device storage carrying a split phase's partial values to its merge launch.
/// Buffers currently have separate allocations; producer/consumer identities
/// describe the required lifetime, not an assertion that reuse is implemented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchAllocation {
    pub index: usize,
    pub phase: usize,
    pub variable: VarId,
    pub dtype: seismic_lang::types::DType,
    pub elements_per_item: u64,
    pub work_items: u64,
    pub parts: u64,
    pub bytes: usize,
    pub producer: usize,
    pub consumer: usize,
}
#[derive(Clone, Debug, Default)]
pub struct MemoryPlan {
    launches: Vec<LaunchMemory>,
    scratch: Vec<ScratchAllocation>,
}
impl MemoryPlan {
    pub fn new(
        launches: Vec<LaunchMemory>,
        scratch: Vec<ScratchAllocation>,
    ) -> Result<Self, String> {
        let mut ids = HashSet::new();
        for (index, launch) in launches.iter().enumerate() {
            if launch.predecessor.is_some_and(|p| p >= index) {
                return Err("memory launch predecessor must precede the launch".into());
            }
            let mut fragment_variables = HashSet::new();
            for fragment in &launch.fragments {
                fragment.layout.metal_type()?;
                if !fragment_variables.insert(fragment.variable) {
                    return Err("duplicate launch fragment binding".into());
                }
            }
            for (site, collective) in &launch.collectives {
                if *site != collective.site {
                    return Err("collective identity differs from its lookup key".into());
                }
                collective.implementation.validate()?;
            }
            for allocation in &launch.arrays {
                if !ids.insert(allocation.id) {
                    return Err("duplicate memory allocation identity".into());
                }
            }
        }
        let mut bindings = HashSet::new();
        for (index, allocation) in scratch.iter().enumerate() {
            if allocation.index != index
                || !bindings.insert((allocation.phase, allocation.variable))
            {
                return Err("scratch allocation identity is ambiguous".into());
            }
            if allocation.parts == 0
                || allocation.producer >= allocation.consumer
                || allocation.consumer >= launches.len()
            {
                return Err("invalid scratch producer/consumer domain".into());
            }
            let mut cursor = Some(allocation.consumer);
            while let Some(launch) = cursor {
                if launch == allocation.producer {
                    break;
                }
                cursor = launches[launch].predecessor;
            }
            if cursor != Some(allocation.producer) {
                return Err("scratch consumer is not ordered after its producer".into());
            }
            let bytes = allocation
                .work_items
                .checked_mul(allocation.parts)
                .and_then(|n| n.checked_mul(allocation.elements_per_item))
                .and_then(|n| n.checked_mul(u64::from(allocation.dtype.bytes())))
                .and_then(|n| usize::try_from(n).ok());
            if bytes != Some(allocation.bytes) {
                return Err("scratch byte size disagrees with its layout".into());
            }
        }
        Ok(Self { launches, scratch })
    }
    pub fn scratch(&self) -> &[ScratchAllocation] {
        &self.scratch
    }
    pub fn launches(&self) -> &[LaunchMemory] {
        &self.launches
    }
}

pub fn plan(
    vars: &[Var],
    body: &[Stmt],
    phases: &[Phase],
    storage: &StoragePlan,
    reductions: &ReductionPlan,
    shared_limit: u64,
) -> Result<MemoryPlan, String> {
    struct Planner<'a> {
        vars: &'a [Var],
        uniform: HashSet<VarId>,
        uniform_views: HashSet<VarId>,
        uniform_atoms: HashSet<seismic_lang::sym::Atom>,
        /// Tensor reads at a common address agree until this launch can write
        /// tensor memory. A containing loop's effects apply before its body, so
        /// the fact cannot accidentally describe only the first iteration.
        uniform_tensor_reads: bool,
        storage: &'a StoragePlan,
        reductions: &'a ReductionPlan,
        bound: HashMap<VarId, Option<TilePlacement>>,
        barriers: BTreeMap<BarrierSite, Barrier>,
        ids: HashSet<AllocationId>,
        arrays: Vec<ArrayAllocation>,
        scope: Vec<Scope>,
        fragments: Vec<FragmentAllocation>,
        collectives: BTreeMap<CollectiveSite, Collective>,
        executions: Arc<Multiplicity<ControlValue>>,
        /// True when some lanes may skip this lexical scope. Uniformity is
        /// derived conservatively from parameter/index/scalar dataflow.
        partial_owned: bool,
    }
    impl Planner<'_> {
        fn uniform_sym(&self, value: &Sym) -> bool {
            value.atoms().iter().all(|atom| {
                self.uniform_atoms.contains(atom)
                    || match atom {
                        seismic_lang::sym::Atom::Param(name) => {
                            self.vars.iter().enumerate().any(|(v, var)| {
                                self.uniform.contains(&v)
                                    && (matches!(&var.kind, VarKind::Index(a) if a==atom)
                                        || (matches!(var.kind, VarKind::Param(_))
                                            && var.name == *name))
                            })
                        }
                        seismic_lang::sym::Atom::Quot(a, b)
                        | seismic_lang::sym::Atom::Rem(a, b) => {
                            self.uniform_sym(a) && self.uniform_sym(b)
                        }
                    }
            })
        }
        fn uniform_view(&self, expr: &Expr) -> bool {
            match &expr.kind {
                ExprKind::Var(v) => self.uniform_views.contains(v),
                ExprKind::Load { view, .. }
                | ExprKind::Transpose(view)
                | ExprKind::Accessor { base: view, .. } => self.uniform_view(view),
                ExprKind::Index { base, indices } => {
                    self.uniform_view(base)
                        && indices.iter().all(|i| match i {
                            Index::Point(e) => self.uniform_value(e),
                            Index::Slice { start, end } => {
                                start.iter().chain(end).all(|e| self.uniform_value(e))
                            }
                        })
                }
                ExprKind::TileAlloc { shape, .. } => shape.iter().all(|s| self.uniform_sym(s)),
                ExprKind::Builtin {
                    name: Builtin::Reshape,
                    args,
                } => args.first().is_some_and(|a| self.uniform_view(a)),
                _ => false,
            }
        }
        fn uniform_value(&self, expr: &Expr) -> bool {
            match &expr.kind {
                ExprKind::Int(_)
                | ExprKind::Float(_)
                | ExprKind::Bool(_)
                | ExprKind::ShapeParam(_) => true,
                ExprKind::Var(v) => self.uniform.contains(v),
                ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } => {
                    self.uniform_value(expr)
                }
                ExprKind::Binary { lhs, rhs, .. } => {
                    self.uniform_value(lhs) && self.uniform_value(rhs)
                }
                ExprKind::Index { base, indices } => self.uniform_tensor_reads
                    && matches!(base.ty, Ty::Tensor(_))
                    && self.uniform_view(base)
                    && indices.iter().all(
                        |index| matches!(index, Index::Point(value) if self.uniform_value(value)),
                    ),
                ExprKind::Intrinsic { op, .. } => matches!(
                    op,
                    seismic_lang::intrinsics::Operation::SimdSum
                        | seismic_lang::intrinsics::Operation::SimdMax
                        | seismic_lang::intrinsics::Operation::SimdMin
                ),
                ExprKind::Builtin {
                    name: Builtin::Extent,
                    args,
                } => {
                    args.first()
                        .and_then(|a| a.ty.shaped())
                        .is_some_and(|s| s.shape.iter().all(|e| self.uniform_sym(e)))
                        && args.iter().skip(1).all(|e| self.uniform_value(e))
                }
                ExprKind::Builtin { name, args }
                    if matches!(
                        name,
                        Builtin::Fma
                            | Builtin::Exp
                            | Builtin::ExpFast
                            | Builtin::Rsqrt
                            | Builtin::Sqrt
                            | Builtin::Log
                            | Builtin::Sin
                            | Builtin::Cos
                            | Builtin::Abs
                            | Builtin::Max
                            | Builtin::Min
                    ) =>
                {
                    args.iter().all(|a| self.uniform_value(a))
                }
                _ => false,
            }
        }
        fn barrier(
            &mut self,
            operation: OperationId,
            variable: VarId,
            purpose: BarrierPurpose,
            memory: MemorySpace,
        ) -> Result<(), String> {
            if self.partial_owned {
                return Err(format!("memory barrier {purpose:?} at {operation:?} requires proven full-lane participation inside owned or conditional control"));
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
        fn expression(
            &mut self,
            expr: &Expr,
            operation: OperationId,
            ordinal: &mut usize,
            output: Option<VarId>,
        ) -> Result<(), String> {
            match &expr.kind {
                ExprKind::Intrinsic { op, args } => {
                    for arg in args {
                        self.expression(arg, operation, ordinal, None)?;
                    }
                    if self.partial_owned && op.collective() {
                        return Err("subgroup intrinsic requires full-lane participation".into());
                    }
                    if matches!(
                        op,
                        seismic_lang::intrinsics::Operation::MatrixLoad
                            | seismic_lang::intrinsics::Operation::MatrixLoadTranspose
                            | seismic_lang::intrinsics::Operation::MatrixStore
                    ) && (args.len() != 4
                        || !self.uniform_view(&args[1])
                        || !self.uniform_value(&args[2])
                        || !self.uniform_value(&args[3]))
                    {
                        return Err(
                            "matrix transfer requires subgroup-uniform view and coordinates".into(),
                        );
                    }
                    let implementation =
                        crate::collective::implementation(*op, args, output, &self.bound)?;
                    if let Implementation::Declare { fragment, layout } = &implementation {
                        self.fragments.push(FragmentAllocation {
                            operation,
                            variable: *fragment,
                            layout: *layout,
                            scope: self.scope.clone(),
                        });
                    }
                    let site = CollectiveSite {
                        operation,
                        ordinal: *ordinal,
                    };
                    *ordinal += 1;
                    if self
                        .collectives
                        .insert(
                            site,
                            Collective {
                                site,
                                implementation,
                                scope: self.scope.clone(),
                                executions: self.executions.clone(),
                            },
                        )
                        .is_some()
                    {
                        return Err("duplicate collective implementation site".into());
                    }
                }
                ExprKind::Index { base, indices } => {
                    self.expression(base, operation, ordinal, None)?;
                    for index in indices {
                        match index {
                            Index::Point(e) => self.expression(e, operation, ordinal, None)?,
                            Index::Slice { start, end } => {
                                for e in start.iter().chain(end) {
                                    self.expression(e, operation, ordinal, None)?;
                                }
                            }
                        }
                    }
                }
                ExprKind::Load { view: base, .. }
                | ExprKind::Transpose(base)
                | ExprKind::Accessor { base, .. }
                | ExprKind::Lanes { base, .. }
                | ExprKind::Unary { expr: base, .. }
                | ExprKind::Cast { expr: base, .. } => {
                    self.expression(base, operation, ordinal, None)?
                }
                ExprKind::Binary { lhs, rhs, .. } => {
                    self.expression(lhs, operation, ordinal, None)?;
                    self.expression(rhs, operation, ordinal, None)?;
                }
                ExprKind::Builtin { args, .. }
                | ExprKind::Call { args, .. }
                | ExprKind::Tuple(args) => {
                    for arg in args {
                        self.expression(arg, operation, ordinal, None)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        fn body(&mut self, body: &[Stmt]) -> Result<(), String> {
            for statement in body {
                // Reuse the language's effect classification, including effects
                // anywhere in a repeated body. Private tile storage has separate
                // per-lane values and never acquires this tensor-read fact.
                self.uniform_tensor_reads &= !seismic_lang::effects::tensor_effect(statement);
                let operation = statement
                    .id
                    .ok_or("allocation planning requires operation identities")?;
                let mut ordinal = 0;
                match &statement.kind {
                    StmtKind::Assign { target, value, .. } => {
                        self.expression(target, operation, &mut ordinal, None)?;
                        let output = if let ExprKind::Var(v) = target.kind {
                            Some(v)
                        } else {
                            None
                        };
                        self.expression(value, operation, &mut ordinal, output)?;
                    }
                    StmtKind::Expr(e) => self.expression(e, operation, &mut ordinal, None)?,
                    StmtKind::If { cond, .. } => {
                        self.expression(cond, operation, &mut ordinal, None)?
                    }
                    StmtKind::Owned { tile, .. } => {
                        self.expression(tile, operation, &mut ordinal, None)?
                    }
                    StmtKind::LoadLoop { views, .. } => {
                        for view in views {
                            self.expression(view, operation, &mut ordinal, None)?;
                        }
                    }
                    _ => {}
                }
                match &statement.kind {
                    StmtKind::LoadLoop {
                        vars,
                        views,
                        axis,
                        capacity,
                        piece,
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
                        let outer_partial = self.partial_owned;
                        let uniform_extent = views.iter().all(|view| self.uniform_view(view));
                        if uniform_extent {
                            for view in views {
                                if let Some(s) = view.ty.shaped() {
                                    for extent in &s.shape {
                                        self.uniform_atoms.extend(extent.atoms());
                                    }
                                }
                            }
                        }
                        for (var, view) in vars.iter().zip(views) {
                            if self.uniform_view(view) {
                                self.uniform_views.insert(*var);
                            } else {
                                self.uniform_views.remove(var);
                            }
                        }
                        self.partial_owned |= !uniform_extent;
                        if uniform_extent {
                            self.uniform_atoms.insert(piece.clone());
                        } else {
                            self.uniform_atoms.remove(piece);
                        }
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
                        self.partial_owned = outer_partial;
                    }
                    StmtKind::Assign { target, value, op } => {
                        let ExprKind::Var(var) = target.kind else {
                            continue;
                        };
                        let previous = self.bound.get(&var).cloned();
                        if !self.partial_owned && self.uniform_view(value) {
                            self.uniform_views.insert(var);
                        } else {
                            self.uniform_views.remove(&var);
                        }
                        if !self.partial_owned
                            && self.uniform_value(value)
                            && (*op == seismic_lang::ast::AssignOp::Assign
                                || self.uniform.contains(&var))
                        {
                            self.uniform.insert(var);
                        } else {
                            self.uniform.remove(&var);
                        }
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
                            _ => {}
                        }
                        self.bound.entry(var).or_insert(None);
                    }
                    StmtKind::Owned {
                        vars: indices,
                        tile,
                        body,
                    } => {
                        let ExprKind::Var(var) = tile.kind else {
                            return Err("owned memory domain has no binding".into());
                        };
                        let placement = self
                            .bound
                            .get(&var)
                            .cloned()
                            .ok_or("owned memory domain has no placement")?;
                        let outer_partial = self.partial_owned;
                        self.partial_owned |= placement != Some(TilePlacement::Replicated)
                            || !tile
                                .ty
                                .shaped()
                                .is_some_and(|s| s.shape.iter().all(|e| self.uniform_sym(e)));
                        for index in indices {
                            if self.partial_owned {
                                self.uniform.remove(index);
                            } else {
                                self.uniform.insert(*index);
                            }
                        }
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
                    StmtKind::Parallel { vars, body, .. } => {
                        self.uniform.extend(vars);
                        self.nested(body, Scope::Body(operation), Multiplicity::Constant(1))?
                    }
                    StmtKind::Range { var, lo, hi, body } => {
                        let outer = self.partial_owned;
                        self.partial_owned |= !self.uniform_sym(lo) || !self.uniform_sym(hi);
                        if self.partial_owned {
                            self.uniform.remove(var);
                        } else {
                            self.uniform.insert(*var);
                        }
                        self.nested(
                            body,
                            Scope::Body(operation),
                            Multiplicity::Iterations {
                                lower: ControlValue::Integer(lo.clone()),
                                upper: ControlValue::Integer(hi.clone()),
                            },
                        )?;
                        self.partial_owned = outer;
                    }
                    StmtKind::Lanes {
                        var,
                        extent,
                        width,
                        body,
                        ..
                    } => {
                        self.uniform.remove(var);
                        let outer_partial = self.partial_owned;
                        self.partial_owned |= !self.uniform_sym(extent);
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
                        self.partial_owned = outer_partial;
                    }
                    StmtKind::If { cond, then, els } => {
                        let outer = self.partial_owned;
                        self.partial_owned |= !self.uniform_value(cond);
                        let incoming = self.uniform.clone();
                        let incoming_views = self.uniform_views.clone();
                        let incoming_atoms = self.uniform_atoms.clone();
                        self.nested(
                            then,
                            Scope::Then(operation),
                            Multiplicity::Predicate {
                                value: ControlValue::Predicate(Box::new(cond.clone())),
                                expected: true,
                            },
                        )?;
                        let after_then = self.uniform.clone();
                        let then_views = self.uniform_views.clone();
                        let then_atoms = self.uniform_atoms.clone();
                        self.uniform = incoming;
                        self.uniform_views = incoming_views;
                        self.uniform_atoms = incoming_atoms;
                        self.nested(
                            els,
                            Scope::Else(operation),
                            Multiplicity::Predicate {
                                value: ControlValue::Predicate(Box::new(cond.clone())),
                                expected: false,
                            },
                        )?;
                        self.uniform.retain(|v| after_then.contains(v));
                        self.uniform_views.retain(|v| then_views.contains(v));
                        self.uniform_atoms.retain(|v| then_atoms.contains(v));
                        self.partial_owned = outer;
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
            prologue: crate::support::LaunchRecipe,
        ) -> Result<LaunchMemory, String> {
            prologue.instantiate(dispatch)?;
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
                prologue,
                predecessor,
                arrays: std::mem::take(&mut self.arrays),
                barriers: std::mem::take(&mut self.barriers),
                declared_private_bytes_per_lane: private,
                shared_bytes_per_group: shared,
                fragments: std::mem::take(&mut self.fragments),
                collectives: std::mem::take(&mut self.collectives),
            })
        }
    }
    if body.len() != phases.len() {
        return Err("allocation plan phase/domain mismatch".into());
    }
    let mut planner = Planner {
        vars,
        uniform: vars
            .iter()
            .enumerate()
            .filter_map(|(v, var)| matches!(var.kind, VarKind::Param(_)).then_some(v))
            .collect(),
        uniform_atoms: HashSet::new(),
        uniform_tensor_reads: true,
        uniform_views: vars
            .iter()
            .enumerate()
            .filter_map(|(v, var)| matches!(var.kind, VarKind::Param(_)).then_some(v))
            .collect(),
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
        fragments: Vec::new(),
        collectives: BTreeMap::new(),
        executions: Arc::new(Multiplicity::Constant(1)),
        partial_owned: false,
    };
    let mut launches = Vec::new();
    let mut scratch = Vec::new();
    for (phase_index, (root, phase)) in body.iter().zip(phases).enumerate() {
        // Ordered launches publish predecessor tensor writes before this phase.
        planner.uniform_tensor_reads = true;
        let operation = root.id.ok_or("phase has no operation identity")?;
        let StmtKind::Parallel {
            vars: indices,
            body,
            ..
        } = &root.kind
        else {
            return Err("allocation phase is not a parallel domain".into());
        };
        planner.uniform.extend(indices);
        planner.scope = vec![Scope::Body(operation)];
        if let Some(split) = &phase.split {
            planner.uniform.insert(split.part);
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
                crate::support::LaunchRecipe::new(phase.mapping.clone(), phase.parts as u64)?,
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
                    crate::support::LaunchRecipe::new(phase.mapping.clone(), 1)?,
                )?,
            );
        } else {
            planner.body(body)?;
            launches.push(planner.finish(
                &phase.dispatch,
                shared_limit,
                launches.len().checked_sub(1),
                crate::support::LaunchRecipe::new(phase.mapping.clone(), 1)?,
            )?);
        }
    }
    MemoryPlan::new(launches, scratch)
}
