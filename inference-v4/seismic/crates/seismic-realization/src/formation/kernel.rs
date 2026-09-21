//! Kernel block formation (package K1).
//!
//! `form` lowers every owned node of one routed block, in the block's
//! structured order, against its closed interface into `CoreKernelOp`s or
//! typed backend intrinsics, and seals the result by exact definition/use
//! and publication-set equality.
//!
//! Lowering policy (exact):
//!
//! - Every value is a binding: a kernel scalar, a range (two scalars), a
//!   tuple, or a tensor. A tensor binding is an addressable place of the
//!   interface, an in-block view over a place (lowered to coordinate
//!   arithmetic at every access), a fused element of the open elementwise
//!   iteration, a snapshot alias (`load`), or a representation-plane alias
//!   (`.words`, `.scale`, ...).
//! - Whole-tensor nodes (elementwise, cast, select, fill, copies, decode,
//!   residual reads/writes) are lowered per element inside one *open
//!   iteration* over their elementwise domain: the block's binder-less axes
//!   cover a prefix of that domain, the residual axes become nested
//!   `Repeat`s. Consecutive whole-tensor nodes with the same domain and no
//!   storage hazard fuse into the same iteration; a computed tensor is then a
//!   scalar SSA per element visible to the fused consumers only. A consumer
//!   at another domain, or after the iteration closed, finds no element and
//!   no residence: that is a defect of the proposing rule (the tensor must
//!   have been spilled by D1), reported by rule name.
//! - Scalar nodes are emitted at the enclosing level (before an open
//!   iteration's nest, which is emitted when it closes); a storage hazard
//!   (read/write, write/read, write/write) between a node and the open
//!   iteration closes the iteration first, so the logical state order is
//!   preserved exactly.
//! - Reductions fold a place with `Fold`, or, over a fused operand, become an
//!   explicit `Repeat` with `Carry` accumulators under the registry schema.
//!
//! Forbidden here: any default dtype/axis/shape/representation, any
//! backend-specific semantic reconstruction, any allocation.

use crate::failure::{CompilerDefect, Package};
use crate::ids::{
    BlockId, CanonicalLeafId, CanonicalStorageId, CanonicalValueId, KernelAxisId, KernelInputId,
    KernelLocalId, KernelOutputId, KernelSsaId, ObligationRef, OwnedGraphKey, OwnedNodeRef,
    OwnedRegionRef, OwnedStateRef, OwnedValueRef, OwnedViewRef, StatusFieldTemplateId,
};
use crate::kernel::{
    AtomicMode, CheckPredicate, ClosedKernelBlock, ConstantValue, CoreKernelOp, ExecutableDialect,
    IntrinsicCatalog, IntrinsicOperand, IntrinsicResult, KernelJoin, KernelOp, KernelPlaceRef,
    KernelSliceAxis, KernelSsaDecl, KernelValueRef, KernelValueType, KernelViewChain,
    KernelViewStep, RelOp, TypedConstant,
};
use crate::occurrence::OccurrenceFacts;
use crate::residence::{ClosedKernelInterface, RoutedStrategy};
use crate::routes::ValueRoute;
use crate::strategy::{
    child_regions, loop_value_carries, node_state_inputs, region_nodes, root_region,
    subtree_nodes, AlgorithmChoice, BlockCut, ParticipantPolicy,
};
use seismic_lang::abi::RangeEndpoint;
use seismic_lang::intrinsics::{
    lowering, reduce_schema, CoreLoweringFamily, IntrinsicId, MathOp, PlaneField, PrimitiveId,
    ReduceOp, ReduceSchema,
};
use seismic_lang::logical::value::{GraphValue, GraphValueKind, TensorSource};
use seismic_lang::logical::{
    Access, GraphValueId, IdVec, IfNode, JoinSlot, LogicalNode, LogicalNodeKind, LoopNode,
    PrimitiveOp, ReductionNode, RegionResult, SafetyObligation, SliceAxis, StateTokenId,
    ViewTransform,
};
use seismic_lang::repr::{lookup as lookup_repr, CodeInterpretation, DecodeStep, PlaneEncoding, PlaneSchema};
use seismic_lang::sir::{IntrinsicUse, Literal, LoopKind};
use seismic_lang::syntax::ast::BinaryOp;
use seismic_lang::types::{DType, Elem, ExtentExpr, NonEmpty, TensorType, ValuePath, ValueType};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn form<D: ExecutableDialect>(
    facts: &OccurrenceFacts<'_>,
    strategy: &RoutedStrategy,
    block: BlockId,
    catalog: &dyn IntrinsicCatalog<D>,
) -> Result<ClosedKernelBlock<D::Intrinsic>, CompilerDefect> {
    let Some(cut) = strategy.shape().blocks().get(block) else {
        return Err(CompilerDefect::new(
            Package::K1,
            format!("block {} is not a block cut of the strategy shape", block.0),
        ));
    };
    let Some(routed) = strategy.blocks().get(block) else {
        return Err(CompilerDefect::new(
            Package::D1,
            format!("block {} has no routed interface", block.0),
        ));
    };
    let mut former = Former::<D>::new(facts, strategy, block, cut, &routed.interface, catalog)?;
    let ops = former.lower_block()?;
    former.seal(ops)
}

type Ops<D> = Vec<KernelOp<<D as ExecutableDialect>::Intrinsic>>;

/// One kernel guard attached to a binding: emitted around every access
/// through that binding.
#[derive(Clone, Debug)]
struct AttachedGuard {
    status: StatusFieldTemplateId,
    obligation: ObligationRef,
    predicate: CheckPredicate,
}

#[derive(Clone, Debug)]
enum Binding {
    Scalar(KernelValueRef),
    Range {
        start: KernelValueRef,
        end: KernelValueRef,
        guards: Vec<AttachedGuard>,
    },
    Tuple(Vec<Binding>),
    Tensor(TensorBinding),
}

#[derive(Clone, Debug)]
enum TensorBinding {
    /// An addressable place of the interface in its view coordinates.
    Place {
        place: KernelPlaceRef,
        ty: TensorType,
    },
    /// An in-block view of `source`.
    View {
        source: Box<TensorBinding>,
        step: KernelViewStep,
        ty: TensorType,
        guards: Vec<AttachedGuard>,
    },
    /// A fused element of the open iteration at its coordinates.
    Element {
        value: KernelValueRef,
        ty: TensorType,
    },
    /// A `load` snapshot: reads its source, invalidated by any later write
    /// of a storage it reads (`value` is the snapshot's identity).
    Snapshot {
        source: Box<TensorBinding>,
        ty: TensorType,
        value: CanonicalValueId,
    },
    /// A representation-plane alias of a packed source (`.words`, ...).
    Plane {
        source: Box<TensorBinding>,
        repr: String,
        field: PlaneField,
        ty: TensorType,
        value: CanonicalValueId,
    },
}

impl TensorBinding {
    fn ty(&self) -> &TensorType {
        match self {
            TensorBinding::Place { ty, .. }
            | TensorBinding::View { ty, .. }
            | TensorBinding::Element { ty, .. }
            | TensorBinding::Snapshot { ty, .. }
            | TensorBinding::Plane { ty, .. } => ty,
        }
    }

    /// The snapshot/plane alias identities this binding reads through.
    fn alias_values(&self, out: &mut Vec<CanonicalValueId>) {
        match self {
            TensorBinding::Place { .. } | TensorBinding::Element { .. } => {}
            TensorBinding::View { source, .. } => source.alias_values(out),
            TensorBinding::Snapshot { source, value, .. }
            | TensorBinding::Plane { source, value, .. } => {
                out.push(*value);
                source.alias_values(out);
            }
        }
    }
}

/// The binding scopes of a block: one root scope that is never popped,
/// plus the nested scopes of control bodies. `current` is total — with no
/// nested scope open it is the root — so a binding is always insertable.
struct Scopes {
    root: BTreeMap<CanonicalValueId, Binding>,
    /// Nested control scopes, innermost last.
    nested: Vec<BTreeMap<CanonicalValueId, Binding>>,
}

impl Scopes {
    fn current(&mut self) -> &mut BTreeMap<CanonicalValueId, Binding> {
        match self.nested.last_mut() {
            Some(scope) => scope,
            None => &mut self.root,
        }
    }

    fn get(&self, canonical: CanonicalValueId) -> Option<&Binding> {
        self.nested
            .iter()
            .rev()
            .find_map(|scope| scope.get(&canonical))
            .or_else(|| self.root.get(&canonical))
    }
}

/// The open elementwise iteration of one op sequence.
struct OpenIteration<I> {
    domain: Vec<ExtentExpr>,
    /// Full-domain coordinates: block axes first, residual binders after.
    coords: Vec<KernelValueRef>,
    /// Residual `Repeat` binders with their `[start, end)`, outermost first.
    residual: Vec<(KernelSsaId, KernelValueRef, KernelValueRef)>,
    /// Ops emitted before the nest (constants, extents, carry initials).
    prelude: Vec<KernelOp<I>>,
    body: Vec<KernelOp<I>>,
    elements: BTreeMap<CanonicalValueId, (KernelValueRef, TensorType)>,
    reads: BTreeSet<CanonicalStorageId>,
    writes: BTreeSet<CanonicalStorageId>,
}

/// Where a grid-wide barrier may be emitted for an intra-launch
/// whole-result edge (a `Store` into a kernel-local intermediate followed by
/// a `Load` of it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BarrierPolicy {
    /// Not a grid-cooperative block: intermediates are never shared across
    /// participants (D1 spills them per participant or across launches).
    None,
    /// The top-level sequence of a grid-cooperative block.
    Grid,
    /// A nested body of a grid-cooperative block: a barrier there would be
    /// inside possibly non-uniform control, so an edge needing one is a
    /// defect.
    Forbidden,
}

/// The op sequence under construction at one nesting level.
struct Emitter<I> {
    ops: Vec<KernelOp<I>>,
    open: Option<OpenIteration<I>>,
    barrier: BarrierPolicy,
    /// A kernel-local intermediate was written and not yet synchronized.
    dirty_local: bool,
    /// A barrier was needed under `BarrierPolicy::Forbidden`.
    violation: bool,
}

fn touches_local<I>(ops: &[KernelOp<I>], stores: &mut bool, loads: &mut bool) {
    for op in ops {
        if let KernelOp::Core(core) = op {
            let written = matches!(
                core,
                CoreKernelOp::Store { .. } | CoreKernelOp::PlaneStore { .. } | CoreKernelOp::Atomic { .. }
            );
            let read = matches!(
                core,
                CoreKernelOp::Load { .. }
                    | CoreKernelOp::PlaneLoad { .. }
                    | CoreKernelOp::PackedPlaneRead { .. }
                    | CoreKernelOp::Fold { .. }
                    | CoreKernelOp::Atomic { .. }
            );
            for place in core.places() {
                if let KernelPlaceRef::Local(_) = place {
                    *stores |= written;
                    *loads |= read;
                }
            }
        }
        for body in op.bodies() {
            touches_local(body, stores, loads);
        }
    }
}

impl<I> Emitter<I> {
    fn new(barrier: BarrierPolicy) -> Emitter<I> {
        Emitter {
            ops: Vec::new(),
            open: None,
            barrier,
            dirty_local: false,
            violation: false,
        }
    }

    /// The nested-body policy of this emitter.
    fn nested(&self) -> BarrierPolicy {
        match self.barrier {
            BarrierPolicy::None => BarrierPolicy::None,
            BarrierPolicy::Grid | BarrierPolicy::Forbidden => BarrierPolicy::Forbidden,
        }
    }

    /// Synchronize before `ops` read a dirty kernel-local intermediate, and
    /// mark the intermediate dirty when `ops` write one.
    fn synchronize(&mut self, ops: &[KernelOp<I>]) {
        let (mut stores, mut loads) = (false, false);
        touches_local(ops, &mut stores, &mut loads);
        if loads && self.dirty_local {
            match self.barrier {
                BarrierPolicy::None => {}
                BarrierPolicy::Grid => {
                    self.ops.push(KernelOp::Core(CoreKernelOp::GridBarrier));
                    self.dirty_local = false;
                }
                BarrierPolicy::Forbidden => self.violation = true,
            }
        }
        if stores {
            self.dirty_local = true;
        }
    }

    /// Append ops at this level (before any open iteration's nest).
    fn push_ops(&mut self, ops: Vec<KernelOp<I>>) {
        self.synchronize(&ops);
        self.ops.extend(ops);
    }

    /// Close the open iteration: emit its prelude and its `Repeat` nest.
    fn flush(&mut self) {
        let Some(open) = self.open.take() else {
            return;
        };
        if open.body.is_empty() {
            return;
        }
        self.ops.extend(open.prelude);
        let mut body = open.body;
        for (binder, start, end) in open.residual.into_iter().rev() {
            body = vec![KernelOp::Core(CoreKernelOp::Repeat {
                binder,
                start,
                end,
                body,
            })];
        }
        self.ops.extend(body);
    }

    fn finish(mut self) -> (Vec<KernelOp<I>>, bool, bool) {
        self.flush();
        (self.ops, self.dirty_local, self.violation)
    }
}

struct Former<'f, 'l, D: ExecutableDialect> {
    facts: &'f OccurrenceFacts<'l>,
    strategy: &'f RoutedStrategy,
    block: BlockId,
    cut: &'f BlockCut,
    interface: &'f ClosedKernelInterface,
    catalog: &'f dyn IntrinsicCatalog<D>,
    members: BTreeSet<OwnedNodeRef>,
    visited: BTreeSet<OwnedNodeRef>,
    ssa: Vec<KernelSsaDecl>,
    scopes: Scopes,
    input_leaves: BTreeMap<CanonicalLeafId, KernelInputId>,
    output_leaves: BTreeMap<CanonicalLeafId, KernelOutputId>,
    local_values: BTreeMap<CanonicalValueId, KernelLocalId>,
    axis_binders: BTreeMap<CanonicalValueId, KernelAxisId>,
    /// Binder-less block axes with their extents, in axis order.
    free_axes: Vec<(KernelAxisId, ExtentExpr)>,
    /// `Check` sites per guard of `cut.guards` (same index).
    guard_sites: Vec<u32>,
    /// Snapshot/plane aliases with the storages they read.
    alias_reads: BTreeMap<CanonicalValueId, BTreeSet<CanonicalStorageId>>,
    invalid_aliases: BTreeSet<CanonicalValueId>,
    /// Publication/store sites per output, indexed by the dense interface
    /// output id: `new` rejects a misnumbered or duplicate-leaf output
    /// table, and every output id in circulation comes from that table.
    output_sites: Vec<u32>,
}

impl<'f, 'l, D: ExecutableDialect> Former<'f, 'l, D> {
    fn new(
        facts: &'f OccurrenceFacts<'l>,
        strategy: &'f RoutedStrategy,
        block: BlockId,
        cut: &'f BlockCut,
        interface: &'f ClosedKernelInterface,
        catalog: &'f dyn IntrinsicCatalog<D>,
    ) -> Result<Former<'f, 'l, D>, CompilerDefect> {
        let mut former = Former {
            facts,
            strategy,
            block,
            cut,
            interface,
            catalog,
            members: cut.nodes.iter().cloned().collect(),
            visited: BTreeSet::new(),
            ssa: Vec::new(),
            scopes: Scopes {
                root: BTreeMap::new(),
                nested: Vec::new(),
            },
            input_leaves: BTreeMap::new(),
            output_leaves: BTreeMap::new(),
            local_values: BTreeMap::new(),
            axis_binders: BTreeMap::new(),
            free_axes: Vec::new(),
            guard_sites: vec![0; cut.guards.len()],
            alias_reads: BTreeMap::new(),
            invalid_aliases: BTreeSet::new(),
            output_sites: Vec::new(),
        };
        for (id, decl) in interface.inputs.entries() {
            if decl.id != id || former.input_leaves.insert(decl.leaf, id).is_some() {
                return Err(former.defect(
                    Package::D1,
                    format!("input {} is misnumbered or its leaf is declared twice", id.0),
                ));
            }
        }
        for (id, decl) in interface.outputs.entries() {
            if decl.id != id || former.output_leaves.insert(decl.leaf, id).is_some() {
                return Err(former.defect(
                    Package::D1,
                    format!("output {} is misnumbered or its leaf is declared twice", id.0),
                ));
            }
        }
        former.output_sites = vec![0; interface.outputs.len()];
        for (id, decl) in interface.locals.entries() {
            if decl.id != id || former.local_values.insert(decl.value, id).is_some() {
                return Err(former.defect(
                    Package::D1,
                    format!("local {} is misnumbered or its value is declared twice", id.0),
                ));
            }
        }
        if interface.axes.len() != interface.iteration.extents.len() {
            return Err(former.defect(
                Package::D1,
                format!(
                    "the interface declares {} axes but its iteration map has {} extents",
                    interface.axes.len(),
                    interface.iteration.extents.len()
                ),
            ));
        }
        for ((id, decl), extent) in interface.axes.entries().zip(&interface.iteration.extents) {
            if decl.id != id {
                return Err(former.defect(Package::D1, format!("axis {} is misnumbered", id.0)));
            }
            match decl.binder {
                Some(binder) => {
                    former.axis_binders.insert(binder, id);
                    former.scopes.current().insert(binder, Binding::Scalar(KernelValueRef::Axis(id)));
                }
                None => former.free_axes.push((id, extent.clone())),
            }
        }
        for binding in &cut.participants.independent_axes {
            if !former.axis_binders.contains_key(&binding.binder) {
                return Err(former.defect(
                    Package::D1,
                    format!(
                        "independent axis {} (binder value {}) has no kernel axis with that binder",
                        binding.ordinal, binding.binder.0
                    ),
                ));
            }
        }
        for (obligation, safety) in &cut.guards {
            match safety {
                SafetyObligation::ExtentPositive { .. } | SafetyObligation::ShapeProductFits { .. } => {
                    return Err(former.defect(
                        Package::S1,
                        format!(
                            "obligation {} of {:?} is an extent-only obligation and cannot be a kernel guard; S1 disposes it as an executor guard",
                            obligation.index, obligation.node
                        ),
                    ));
                }
                SafetyObligation::IndexInBounds { .. }
                | SafetyObligation::RangeInBounds { .. }
                | SafetyObligation::DivisorNonZero { .. }
                | SafetyObligation::SignedDivisionNoOverflow { .. }
                | SafetyObligation::ShiftInRange { .. } => {}
            }
            if !former.members.contains(&obligation.node) {
                return Err(former.defect(
                    Package::S1,
                    format!("guard {:?} names a node the block does not own", obligation),
                ));
            }
        }
        Ok(former)
    }

    fn defect(&self, package: Package, invariant: impl std::fmt::Display) -> CompilerDefect {
        CompilerDefect::new(
            package,
            format!(
                "block {} of rule `{}`: {invariant}",
                self.block.0,
                self.strategy.shape().rule()
            ),
        )
    }

    /// A contradiction of the proposing rule's structural promise (S1 seals
    /// the shape, so the rule owner is the one to fix).
    fn rule_defect(&self, invariant: impl std::fmt::Display) -> CompilerDefect {
        self.defect(Package::S1, invariant)
    }

    // -- environment -------------------------------------------------------

    fn canonical(
        &self,
        graph: OwnedGraphKey,
        value: GraphValueId,
    ) -> Result<CanonicalValueId, CompilerDefect> {
        self.facts.canonical_value(OwnedValueRef { graph, value })
    }

    fn fresh(&mut self, ty: KernelValueType) -> KernelSsaId {
        let id = KernelSsaId(self.ssa.len() as u32);
        self.ssa.push(KernelSsaDecl { id, ty });
        id
    }

    fn fresh_scalar(&mut self, dtype: DType) -> KernelSsaId {
        self.fresh(KernelValueType::Scalar(dtype))
    }

    fn push_scope(&mut self) {
        self.scopes.nested.push(BTreeMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.nested.pop();
    }

    fn bind(&mut self, canonical: CanonicalValueId, binding: Binding) {
        self.scopes.current().insert(canonical, binding);
    }

    fn scalar_dtype_of(&self, kind: &GraphValueKind) -> Result<DType, CompilerDefect> {
        match kind {
            GraphValueKind::Scalar(dtype) => Ok(*dtype),
            GraphValueKind::Index { .. } => Ok(DType::I32),
            GraphValueKind::Void
            | GraphValueKind::Range { .. }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Capability(_)
            | GraphValueKind::Tensor { .. } => Err(self.defect(
                Package::L1,
                format!("a kernel scalar was expected but the value kind is {kind:?}"),
            )),
        }
    }

    fn kernel_type_of(&self, canonical: CanonicalValueId) -> Result<KernelValueType, CompilerDefect> {
        match self.facts.value_kind(canonical) {
            GraphValueKind::Scalar(dtype) => Ok(KernelValueType::Scalar(*dtype)),
            GraphValueKind::Index { .. } => Ok(KernelValueType::Scalar(DType::I32)),
            GraphValueKind::Capability(ty) => Ok(KernelValueType::Capability(ty.clone())),
            kind @ (GraphValueKind::Void
            | GraphValueKind::Range { .. }
            | GraphValueKind::Tuple(_)
            | GraphValueKind::Tensor { .. }) => Err(self.defect(
                Package::L1,
                format!("value {} of kind {kind:?} is not a kernel scalar", canonical.0),
            )),
        }
    }

    fn leaf_of(
        &self,
        canonical: CanonicalValueId,
        path: &ValuePath,
        endpoint: Option<RangeEndpoint>,
    ) -> Result<CanonicalLeafId, CompilerDefect> {
        let facts = self.facts;
        facts
            .leaves(canonical)
            .iter()
            .copied()
            .find(|leaf| {
                let record = facts.leaf(*leaf);
                record.path == *path && record.endpoint == endpoint
            })
            .ok_or_else(|| {
                self.defect(
                    Package::O1,
                    format!(
                        "value {} has no leaf at path {path} endpoint {endpoint:?}",
                        canonical.0
                    ),
                )
            })
    }

    /// Build the binding of a block input from the interface, following the
    /// value type structure through the canonical leaf registry.
    fn input_binding(
        &self,
        canonical: CanonicalValueId,
        ty: &ValueType,
        path: &mut Vec<u32>,
    ) -> Result<Binding, CompilerDefect> {
        let input_of = |former: &Self, endpoint: Option<RangeEndpoint>| -> Result<KernelInputId, CompilerDefect> {
            let leaf = former.leaf_of(canonical, &ValuePath(path.clone()), endpoint)?;
            former.input_leaves.get(&leaf).copied().ok_or_else(|| {
                former.defect(
                    Package::D1,
                    format!(
                        "leaf {} of value {} is consumed by the block but is not an interface input",
                        leaf.0, canonical.0
                    ),
                )
            })
        };
        match ty {
            ValueType::Scalar(_) | ValueType::Index { .. } | ValueType::CapabilityValue(_) => {
                let id = input_of(self, None)?;
                self.expect_scalar_route(id)?;
                Ok(Binding::Scalar(KernelValueRef::Input(id)))
            }
            ValueType::Range { .. } => {
                let start = input_of(self, Some(RangeEndpoint::Start))?;
                let end = input_of(self, Some(RangeEndpoint::End))?;
                self.expect_scalar_route(start)?;
                self.expect_scalar_route(end)?;
                Ok(Binding::Range {
                    start: KernelValueRef::Input(start),
                    end: KernelValueRef::Input(end),
                    guards: Vec::new(),
                })
            }
            ValueType::Tensor(tensor) => {
                let id = input_of(self, None)?;
                match &self.interface.inputs[id].route {
                    ValueRoute::Tensor(_) => Ok(Binding::Tensor(TensorBinding::Place {
                        place: KernelPlaceRef::Input(id),
                        ty: tensor.clone(),
                    })),
                    route @ (ValueRoute::Void | ValueRoute::Scalar(_)) => {
                        Err(self.defect(
                            Package::D1,
                            format!("tensor input {} is routed as {route:?}", id.0),
                        ))
                    }
                }
            }
            ValueType::Tuple(items) => {
                let mut bindings = Vec::with_capacity(items.len());
                for (index, item) in items.iter().enumerate() {
                    path.push(index as u32);
                    bindings.push(self.input_binding(canonical, item, path)?);
                    path.pop();
                }
                Ok(Binding::Tuple(bindings))
            }
            ValueType::Void => Err(self.defect(
                Package::L1,
                format!("void value {} is consumed by the block", canonical.0),
            )),
        }
    }

    fn expect_scalar_route(&self, id: KernelInputId) -> Result<(), CompilerDefect> {
        match &self.interface.inputs[id].route {
            ValueRoute::Scalar(_) => Ok(()),
            route @ (ValueRoute::Void | ValueRoute::Tensor(_)) => Err(self.defect(
                Package::D1,
                format!("scalar input {} is routed as {route:?}", id.0),
            )),
        }
    }

    fn lookup(&self, canonical: CanonicalValueId) -> Result<Binding, CompilerDefect> {
        if let Some(binding) = self.scopes.get(canonical) {
            return Ok(binding.clone());
        }
        let leaves = self.facts.leaves(canonical);
        if leaves.iter().all(|leaf| self.input_leaves.contains_key(leaf)) && !leaves.is_empty() {
            let ty = self.facts.value_type(canonical);
            return self.input_binding(canonical, &ty, &mut Vec::new());
        }
        // A value the block itself writes (its leaf is a block output): the
        // destination binding is the output's place — an in-block write
        // through an output destination addresses it directly.
        if let [leaf] = leaves {
            if let Some(output) = self.output_leaves.get(leaf).copied() {
                let place = self.output_place(output)?;
                let ValueType::Tensor(ty) = self.facts.value_type(canonical) else {
                    return Err(self.defect(
                        Package::D1,
                        format!("tensor output value {} is not a tensor", canonical.0),
                    ));
                };
                return Ok(Binding::Tensor(TensorBinding::Place { place, ty }));
            }
        }
        Err(self.rule_defect(format!(
            "value {} is consumed inside the block but is neither defined earlier in the block nor a block input (a computed tensor consumed at another elementwise domain, or after its fused iteration closed, must be spilled by D1) [kind={:?} leaves={:?} block inputs={:?} members={:?}]",
            canonical.0,
            self.facts.value_kind(canonical),
            leaves,
            self.input_leaves.keys().collect::<Vec<_>>(),
            self.facts.members(canonical),
        )))
    }

    fn scalar(&self, canonical: CanonicalValueId) -> Result<KernelValueRef, CompilerDefect> {
        match self.lookup(canonical)? {
            Binding::Scalar(value) => Ok(value),
            Binding::Range { .. } | Binding::Tuple(_) | Binding::Tensor(_) => Err(self.defect(
                Package::L1,
                format!("value {} is used as a kernel scalar but is not one", canonical.0),
            )),
        }
    }

    fn scalar_typed(&self, canonical: CanonicalValueId) -> Result<(KernelValueRef, DType), CompilerDefect> {
        let value = self.scalar(canonical)?;
        let dtype = self.scalar_dtype_of(self.facts.value_kind(canonical))?;
        Ok((value, dtype))
    }

    /// The tensor binding of one value: the open iteration's fused element
    /// first, then the environment, then the interface inputs.
    fn resolve_tensor(
        &self,
        canonical: CanonicalValueId,
        em: &Emitter<D::Intrinsic>,
    ) -> Result<TensorBinding, CompilerDefect> {
        if let Some(open) = &em.open {
            if let Some((value, ty)) = open.elements.get(&canonical) {
                return Ok(TensorBinding::Element {
                    value: *value,
                    ty: ty.clone(),
                });
            }
        }
        let binding = match self.lookup(canonical)? {
            Binding::Tensor(binding) => binding,
            Binding::Scalar(_) | Binding::Range { .. } | Binding::Tuple(_) => {
                return Err(self.defect(
                    Package::L1,
                    format!("value {} is used as a tensor but is not one", canonical.0),
                ))
            }
        };
        let mut aliases = Vec::new();
        binding.alias_values(&mut aliases);
        if let Some(alias) = aliases.iter().find(|alias| self.invalid_aliases.contains(alias)) {
            return Err(self.rule_defect(format!(
                "snapshot value {} is consumed after a storage it reads was written inside the block; the rule must spill the snapshot",
                alias.0
            )));
        }
        Ok(binding)
    }

    // -- guards --------------------------------------------------------------

    fn node_guard_indices(&self, node: &OwnedNodeRef) -> Vec<usize> {
        self.cut
            .guards
            .iter()
            .enumerate()
            .filter(|(_, (obligation, _))| obligation.node == *node)
            .map(|(index, _)| index)
            .collect()
    }

    /// The attached form of every kernel guard of `node`, with the guarded
    /// values resolved through `resolved` (the node's already-resolved
    /// operands) or the environment.
    fn guards_for(
        &self,
        node: &OwnedNodeRef,
        resolved: &BTreeMap<GraphValueId, (KernelValueRef, DType)>,
    ) -> Result<Vec<AttachedGuard>, CompilerDefect> {
        let mut guards = Vec::new();
        for index in self.node_guard_indices(node) {
            let (obligation, safety) = &self.cut.guards[index];
            let value = |id: GraphValueId| -> Result<(KernelValueRef, DType), CompilerDefect> {
                match resolved.get(&id) {
                    Some(entry) => Ok(*entry),
                    None => self.scalar_typed(self.canonical(node.graph, id)?),
                }
            };
            let predicate = match safety {
                SafetyObligation::IndexInBounds { index, extent } => CheckPredicate::IndexInBounds {
                    index: value(*index)?.0,
                    extent: extent.clone(),
                },
                SafetyObligation::RangeInBounds { start, end, extent } => CheckPredicate::RangeInBounds {
                    start: value(*start)?.0,
                    end: value(*end)?.0,
                    extent: extent.clone(),
                },
                SafetyObligation::DivisorNonZero { value: divisor } => {
                    let (divisor, dtype) = value(*divisor)?;
                    CheckPredicate::DivisorNonZero {
                        value: divisor,
                        dtype,
                    }
                }
                SafetyObligation::SignedDivisionNoOverflow { lhs, rhs } => {
                    CheckPredicate::SignedDivisionNoOverflow {
                        lhs: value(*lhs)?.0,
                        rhs: value(*rhs)?.0,
                    }
                }
                SafetyObligation::ShiftInRange { value: count } => CheckPredicate::ShiftInRange {
                    value: value(*count)?.0,
                },
                SafetyObligation::ExtentPositive { .. } | SafetyObligation::ShapeProductFits { .. } => {
                    return Err(self.defect(
                        Package::S1,
                        format!("extent-only obligation {obligation:?} reached kernel guard lowering"),
                    ))
                }
            };
            guards.push(AttachedGuard {
                status: StatusFieldTemplateId(index as u32),
                obligation: obligation.clone(),
                predicate,
            });
        }
        Ok(guards)
    }

    /// Wrap `ops` in one `Check` per guard (the first guard outermost).
    fn wrap(&mut self, mut ops: Ops<D>, guards: &[AttachedGuard]) -> Ops<D> {
        for guard in guards.iter().rev() {
            self.guard_sites[guard.status.0 as usize] += 1;
            ops = vec![KernelOp::Core(CoreKernelOp::Check {
                obligation: guard.obligation.clone(),
                predicate: guard.predicate.clone(),
                status: guard.status,
                guarded: ops,
            })];
        }
        ops
    }

    /// A record-only site: the obligation is checked where the guarded value
    /// (a view or range) is created, even when nothing in the block accesses
    /// it.
    fn record_only(&mut self, guards: &[AttachedGuard], ops: &mut Ops<D>) {
        for guard in guards {
            ops.extend(self.wrap(Vec::new(), std::slice::from_ref(guard)));
        }
    }

    // -- constants and extents ---------------------------------------------

    fn const_op(&mut self, value: ConstantValue, dtype: DType, ops: &mut Ops<D>) -> KernelValueRef {
        let into = self.fresh_scalar(dtype);
        ops.push(KernelOp::Core(CoreKernelOp::Const {
            into,
            value: TypedConstant { value, dtype },
        }));
        KernelValueRef::Ssa(into)
    }

    fn const_i32(&mut self, value: i64, ops: &mut Ops<D>) -> KernelValueRef {
        self.const_op(ConstantValue::Int(value), DType::I32, ops)
    }

    fn zero_of(&mut self, dtype: DType, ops: &mut Ops<D>) -> KernelValueRef {
        let value = match dtype {
            DType::F32 | DType::BF16 | DType::F16 => ConstantValue::Float { bits: 0f64.to_bits() },
            DType::I32 | DType::U32 => ConstantValue::Int(0),
            DType::Bool => ConstantValue::Bool(false),
        };
        self.const_op(value, dtype, ops)
    }

    fn extent_ref(&mut self, extent: &ExtentExpr, ops: &mut Ops<D>) -> Result<KernelValueRef, CompilerDefect> {
        let constant = match extent {
            ExtentExpr::Static(n) => *n,
            ExtentExpr::Runtime(id) => {
                let into = self.fresh_scalar(DType::I32);
                ops.push(KernelOp::Core(CoreKernelOp::RuntimeExtent { into, extent: *id }));
                return Ok(KernelValueRef::Ssa(into));
            }
            ExtentExpr::Sym(sym) => match sym.as_constant() {
                Some(constant) if constant >= 0 => u64::try_from(constant).map_err(|_| {
                    self.defect(
                        Package::W1,
                        format!("symbolic extent `{sym}` resolves to {constant}, above the u64 extent space"),
                    )
                })?,
                Some(constant) => {
                    return Err(self.defect(
                        Package::W1,
                        format!("symbolic extent `{sym}` resolves to the negative constant {constant}"),
                    ))
                }
                None => {
                    return Err(self.defect(
                        Package::W1,
                        format!("symbolic extent `{sym}` survived specialization"),
                    ))
                }
            },
        };
        let value = i64::from(i32::try_from(constant).map_err(|_| {
            self.defect(
                Package::L1,
                format!("static extent {constant} exceeds the i32 index space"),
            )
        })?);
        Ok(self.const_i32(value, ops))
    }

    fn dense_dtype(&self, ty: &TensorType) -> Result<DType, CompilerDefect> {
        match &ty.elem {
            Elem::Dtype(dtype) => Ok(*dtype),
            Elem::Repr(name) => Err(self.defect(
                Package::L1,
                format!("a dense element was required but the tensor is packed as `{name}`"),
            )),
            Elem::Param(name) => Err(self.defect(
                Package::W1,
                format!("element parameter `{name}` survived specialization"),
            )),
        }
    }

    fn cast_if_needed(
        &mut self,
        value: KernelValueRef,
        from: DType,
        to: DType,
        ops: &mut Ops<D>,
    ) -> KernelValueRef {
        if from == to {
            return value;
        }
        let into = self.fresh_scalar(to);
        ops.push(KernelOp::Core(CoreKernelOp::Cast {
            into,
            operand: value,
            from,
            to,
        }));
        KernelValueRef::Ssa(into)
    }

    fn binary(
        &mut self,
        op: BinaryOp,
        left: KernelValueRef,
        right: KernelValueRef,
        dtype: DType,
        ops: &mut Ops<D>,
    ) -> KernelValueRef {
        let into = self.fresh_scalar(dtype);
        ops.push(KernelOp::Core(CoreKernelOp::Binary {
            into,
            op,
            left,
            right,
            dtype,
        }));
        KernelValueRef::Ssa(into)
    }

    // -- structure -----------------------------------------------------------

    fn absorbed_callee(&self, call: &OwnedNodeRef) -> Result<OwnedGraphKey, CompilerDefect> {
        let occurrence = self.facts.call_occurrence(call)?;
        match self.strategy.shape().absorbed().get(&occurrence) {
            Some(alternative) => Ok(OwnedGraphKey {
                occurrence,
                logical_alternative: *alternative,
            }),
            None => Err(self.rule_defect(format!(
                "call {call:?} is inside the block but its occurrence#{} is not absorbed",
                occurrence.0
            ))),
        }
    }

    /// Every node of an absorbed callee graph, nested regions and further
    /// absorbed callees included.
    fn callee_nodes(&self, key: OwnedGraphKey, out: &mut Vec<OwnedNodeRef>) -> Result<(), CompilerDefect> {
        for node in region_nodes(self.facts, &root_region(key)) {
            for inner in subtree_nodes(self.facts, &node) {
                if let LogicalNodeKind::Call(_) = &self.facts.node(&inner)?.kind {
                    if self.members.contains(&inner) {
                        let callee = self.absorbed_callee(&inner)?;
                        self.callee_nodes(callee, out)?;
                    }
                }
                out.push(inner);
            }
        }
        Ok(())
    }

    /// The block's top-level nodes in order: every owned node not nested in
    /// the child regions of another owned control node and not part of an
    /// absorbed callee.
    fn top_level_nodes(&self) -> Result<Vec<OwnedNodeRef>, CompilerDefect> {
        let mut nested: BTreeSet<OwnedNodeRef> = BTreeSet::new();
        for node in self.cut.nodes.iter() {
            match &self.facts.node(node)?.kind {
                LogicalNodeKind::Loop(_) | LogicalNodeKind::If(_) => {
                    nested.extend(subtree_nodes(self.facts, node).into_iter().skip(1));
                }
                LogicalNodeKind::Call(_) => {
                    let callee = self.absorbed_callee(node)?;
                    let mut nodes = Vec::new();
                    self.callee_nodes(callee, &mut nodes)?;
                    nested.extend(nodes);
                }
                LogicalNodeKind::Primitive(_) | LogicalNodeKind::Reduction(_) => {}
            }
        }
        Ok(self
            .cut
            .nodes
            .iter()
            .filter(|node| !nested.contains(node))
            .cloned()
            .collect())
    }

    /// The nodes of one region in order, each absorbed call followed by its
    /// callee's root sequence.
    fn sequence(&self, region: &OwnedRegionRef) -> Result<Vec<OwnedNodeRef>, CompilerDefect> {
        let mut out = Vec::new();
        for node in region_nodes(self.facts, region) {
            let callee = match &self.facts.node(&node)?.kind {
                LogicalNodeKind::Call(_) => Some(self.absorbed_callee(&node)?),
                LogicalNodeKind::Primitive(_)
                | LogicalNodeKind::Reduction(_)
                | LogicalNodeKind::Loop(_)
                | LogicalNodeKind::If(_) => None,
            };
            out.push(node);
            if let Some(callee) = callee {
                out.extend(self.sequence(&root_region(callee))?);
            }
        }
        Ok(out)
    }

    fn barrier_policy(&self) -> BarrierPolicy {
        match &self.cut.participants.policy {
            ParticipantPolicy::GridCooperative { .. } => BarrierPolicy::Grid,
            ParticipantPolicy::Serial
            | ParticipantPolicy::Linear { .. }
            | ParticipantPolicy::Cooperative { .. }
            | ParticipantPolicy::DynamicPull { .. } => BarrierPolicy::None,
        }
    }

    /// Finish a nested emitter, reporting an intra-launch whole-result edge
    /// that would need a grid barrier inside nested control.
    fn finish_nested(&self, em: Emitter<D::Intrinsic>, parent: &mut Emitter<D::Intrinsic>, what: &str) -> Result<Ops<D>, CompilerDefect> {
        let (ops, dirty, violation) = em.finish();
        if violation {
            return Err(self.rule_defect(format!(
                "a kernel-local intermediate is written and read inside {what} of a grid-cooperative block; a grid barrier cannot be placed inside nested control, the edge must be cut or spilled per participant"
            )));
        }
        parent.dirty_local |= dirty;
        Ok(ops)
    }

    fn lower_block(&mut self) -> Result<Ops<D>, CompilerDefect> {
        let top = self.top_level_nodes()?;
        let mut em = Emitter::new(self.barrier_policy());
        self.lower_into(&top, &mut em)?;
        let (ops, _, violation) = em.finish();
        if violation {
            return Err(self.defect(Package::K1, "a barrier violation was recorded at the block level"));
        }
        Ok(ops)
    }

    fn lower_into(&mut self, nodes: &[OwnedNodeRef], em: &mut Emitter<D::Intrinsic>) -> Result<(), CompilerDefect> {
        for node in nodes {
            self.lower_node(node, em)?;
        }
        Ok(())
    }

    fn storage_of(
        &self,
        graph: OwnedGraphKey,
        token: StateTokenId,
    ) -> Result<CanonicalStorageId, CompilerDefect> {
        self.facts
            .canonical_storage(self.facts.state_storage(OwnedStateRef { graph, state: token }))
    }

    /// The storages one node reads (its state inputs) and writes (its state
    /// outputs).
    fn node_storages(
        &self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
    ) -> Result<(BTreeSet<CanonicalStorageId>, BTreeSet<CanonicalStorageId>), CompilerDefect> {
        let mut reads = BTreeSet::new();
        for token in node_state_inputs(logical) {
            reads.insert(self.storage_of(node.graph, *token)?);
        }
        let mut writes = BTreeSet::new();
        for token in &logical.state_outputs {
            writes.insert(self.storage_of(node.graph, token.id)?);
        }
        Ok((reads, writes))
    }

    /// Close the open iteration when `node` conflicts with it through storage
    /// state, and invalidate every snapshot alias the node's writes affect.
    fn hazards(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let (reads, writes) = self.node_storages(node, logical)?;
        if let Some(open) = &em.open {
            let conflict = writes
                .iter()
                .any(|storage| open.reads.contains(storage) || open.writes.contains(storage))
                || reads.iter().any(|storage| open.writes.contains(storage));
            if conflict {
                em.flush();
            }
        }
        if !writes.is_empty() {
            for (alias, sources) in &self.alias_reads {
                if sources.iter().any(|storage| writes.contains(storage)) {
                    self.invalid_aliases.insert(*alias);
                }
            }
        }
        Ok(())
    }

    fn lower_node(&mut self, node: &OwnedNodeRef, em: &mut Emitter<D::Intrinsic>) -> Result<(), CompilerDefect> {
        if !self.members.contains(node) {
            return Err(self.rule_defect(format!(
                "{node:?} is reached by the block's structure but the block does not own it"
            )));
        }
        if !self.visited.insert(node.clone()) {
            return Err(self.defect(Package::K1, format!("{node:?} is lowered twice")));
        }
        let logical = self.facts.node(node)?;
        match &logical.kind {
            LogicalNodeKind::Call(_) => {
                let callee = self.absorbed_callee(node)?;
                let body = self.sequence(&root_region(callee))?;
                self.lower_into(&body, em)
            }
            LogicalNodeKind::Loop(loop_node) => self.lower_loop(node, logical, loop_node, em),
            LogicalNodeKind::If(if_node) => self.lower_if(node, logical, if_node, em),
            LogicalNodeKind::Reduction(reduction) => {
                self.hazards(node, logical, em)?;
                self.lower_reduction(node, logical, reduction, em)
            }
            LogicalNodeKind::Primitive(application) => {
                self.hazards(node, logical, em)?;
                self.lower_primitive(node, logical, &application.op, em)
            }
        }
    }

    // -- control -------------------------------------------------------------

    fn lower_loop(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        loop_node: &LoopNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let facts = self.facts;
        let binder = self.canonical(node.graph, loop_node.binder)?;
        let regions = child_regions(facts, node);
        let body_region = regions
            .first()
            .cloned()
            .ok_or_else(|| self.defect(Package::L1, format!("loop {node:?} has no body region")))?;
        if self.axis_binders.contains_key(&binder) {
            if loop_node.kind != LoopKind::Independent {
                return Err(self.rule_defect(format!(
                    "loop {node:?} is bound to a kernel axis but is not an independent loop"
                )));
            }
            let body = self.sequence(&body_region)?;
            return self.lower_into(&body, em);
        }
        if !self.cut.participants.serial_binders.contains(&binder) {
            return Err(self.rule_defect(format!(
                "absorbed loop {node:?} has binder value {} which is neither a kernel axis nor a serial binder",
                binder.0
            )));
        }
        em.flush();
        let start = self.scalar(self.canonical(node.graph, loop_node.range.start)?)?;
        let end = self.scalar(self.canonical(node.graph, loop_node.range.end)?)?;
        let carries = loop_value_carries(facts, node, loop_node)?;
        let mut pending = Vec::with_capacity(carries.len());
        for carry in &carries {
            let ty = self.kernel_type_of(carry.initial)?;
            let initial = self.scalar(carry.initial)?;
            let current = self.fresh(ty.clone());
            let result = self.fresh(ty.clone());
            pending.push((carry.clone(), initial, current, result, ty));
        }
        let binder_ssa = self.fresh_scalar(DType::I32);
        self.push_scope();
        self.bind(binder, Binding::Scalar(KernelValueRef::Ssa(binder_ssa)));
        for (carry, _, current, _, _) in &pending {
            self.bind(carry.parameter, Binding::Scalar(KernelValueRef::Ssa(*current)));
        }
        let mut body_em = Emitter::new(em.nested());
        let sequence = self.sequence(&body_region)?;
        self.lower_into(&sequence, &mut body_em)?;
        let body_ops = self.finish_nested(body_em, em, "the body of an absorbed loop")?;
        let mut body = Vec::with_capacity(pending.len() + body_ops.len());
        for (carry, initial, current, result, ty) in &pending {
            let update = self.scalar(carry.update)?;
            body.push(KernelOp::Core(CoreKernelOp::Carry {
                initial: *initial,
                current: *current,
                update,
                result: *result,
                ty: ty.clone(),
            }));
        }
        body.extend(body_ops);
        self.pop_scope();
        em.push_ops(vec![KernelOp::Core(CoreKernelOp::Repeat {
            binder: binder_ssa,
            start,
            end,
            body,
        })]);
        for (carry, _, _, result, _) in &pending {
            self.define_scalar(carry.result, KernelValueRef::Ssa(*result), em)?;
        }
        // Every value output of the loop is a carry result (sealed by L1).
        for output in &logical.outputs {
            let canonical = self.canonical(node.graph, output.id())?;
            if !pending.iter().any(|(carry, ..)| carry.result == canonical) {
                return Err(self.defect(
                    Package::L1,
                    format!("loop {node:?} output {} is not a carried value", canonical.0),
                ));
            }
        }
        Ok(())
    }

    fn region_result_value(
        &self,
        region: &OwnedRegionRef,
        ordinal: usize,
    ) -> Result<CanonicalValueId, CompilerDefect> {
        match self.facts.region(region)?.results.get(ordinal) {
            Some(RegionResult::Value { id, .. }) => self.canonical(region.graph, *id),
            Some(RegionResult::State { .. }) | None => Err(self.defect(
                Package::L1,
                format!("result {ordinal} of {region:?} is not a joined value"),
            )),
        }
    }

    fn lower_if(
        &mut self,
        node: &OwnedNodeRef,
        _logical: &LogicalNode,
        if_node: &IfNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        em.flush();
        let condition = self.scalar(self.canonical(node.graph, if_node.condition)?)?;
        let regions = child_regions(self.facts, node);
        let [then_region, else_region] = regions.as_slice() else {
            return Err(self.defect(Package::L1, format!("conditional {node:?} has no two regions")));
        };
        let (then_body, then_results) = self.branch_side(if_node, then_region, then_region, em)?;
        let (else_body, else_results) = self.branch_side(if_node, else_region, then_region, em)?;
        let mut joins = Vec::new();
        let mut joined_values = Vec::new();
        let mut index = 0;
        for join in &if_node.joins {
            match join {
                JoinSlot::Value { joined, .. } => {
                    let canonical = self.canonical(node.graph, *joined)?;
                    let ty = match self.facts.value_kind(canonical) {
                        GraphValueKind::Tensor { .. } => {
                            return Err(self.rule_defect(format!(
                                "conditional {node:?} joins tensor value {} inside a block; absorbed conditionals join kernel scalars only",
                                canonical.0
                            )))
                        }
                        _ => self.kernel_type_of(canonical)?,
                    };
                    let id = self.fresh(ty.clone());
                    joins.push(KernelJoin {
                        then_value: then_results[index],
                        else_value: else_results[index],
                        joined: id,
                        ty,
                    });
                    joined_values.push((canonical, id));
                    index += 1;
                }
                JoinSlot::State { .. } => {}
            }
        }
        em.push_ops(vec![KernelOp::Core(CoreKernelOp::Branch {
            condition,
            then_body,
            else_body,
            joins,
        })]);
        for (canonical, id) in joined_values {
            self.define_scalar(canonical, KernelValueRef::Ssa(id), em)?;
        }
        Ok(())
    }

    /// Lower one branch region of an absorbed conditional: its ops and the
    /// kernel scalars of its joined value results.
    fn branch_side(
        &mut self,
        if_node: &IfNode,
        region: &OwnedRegionRef,
        then_region: &OwnedRegionRef,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(Ops<D>, Vec<KernelValueRef>), CompilerDefect> {
        self.push_scope();
        let mut side_em = Emitter::new(em.nested());
        let sequence = self.sequence(region)?;
        self.lower_into(&sequence, &mut side_em)?;
        let ops = self.finish_nested(side_em, em, "a branch of an absorbed conditional")?;
        let mut results = Vec::new();
        for join in &if_node.joins {
            if let JoinSlot::Value {
                then_result,
                else_result,
                ..
            } = join
            {
                let ordinal = if region == then_region { then_result } else { else_result };
                let value = self.region_result_value(region, ordinal.index())?;
                results.push(self.scalar(value)?);
            }
        }
        self.pop_scope();
        Ok((ops, results))
    }

    // -- publication -----------------------------------------------------------

    /// Bind a scalar result and publish it when it is a block output.
    fn define_scalar(
        &mut self,
        canonical: CanonicalValueId,
        value: KernelValueRef,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        self.bind(canonical, Binding::Scalar(value));
        let leaf = self.leaf_of(canonical, &ValuePath::default(), None)?;
        if let Some(output) = self.output_leaves.get(&leaf).copied() {
            let mut ops = Vec::new();
            self.publish_scalar(output, value, &[], &mut ops)?;
            em.push_ops(ops);
        }
        Ok(())
    }

    fn publish_scalar(
        &mut self,
        output: KernelOutputId,
        value: KernelValueRef,
        guards: &[AttachedGuard],
        ops: &mut Ops<D>,
    ) -> Result<(), CompilerDefect> {
        match &self.interface.outputs[output].route {
            ValueRoute::Scalar(_) => {}
            route @ (ValueRoute::Void | ValueRoute::Tensor(_)) => {
                return Err(self.defect(
                    Package::D1,
                    format!("scalar output {} is routed as {route:?}", output.0),
                ))
            }
        }
        let publish = vec![KernelOp::Core(CoreKernelOp::Publish { value, output })];
        let wrapped = self.wrap(publish, guards);
        ops.extend(wrapped);
        self.output_sites[output.0 as usize] += 1;
        Ok(())
    }

    /// Publish every leaf of a structured value that is a block output.
    fn publish_value(
        &mut self,
        canonical: CanonicalValueId,
        binding: &Binding,
        ty: &ValueType,
        path: &mut Vec<u32>,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output_of = |former: &Self, endpoint: Option<RangeEndpoint>| -> Result<Option<KernelOutputId>, CompilerDefect> {
            let leaf = former.leaf_of(canonical, &ValuePath(path.clone()), endpoint)?;
            Ok(former.output_leaves.get(&leaf).copied())
        };
        match (ty, binding) {
            (ValueType::Scalar(_) | ValueType::Index { .. } | ValueType::CapabilityValue(_), Binding::Scalar(value)) => {
                if let Some(output) = output_of(self, None)? {
                    let mut ops = Vec::new();
                    self.publish_scalar(output, *value, &[], &mut ops)?;
                    em.push_ops(ops);
                }
                Ok(())
            }
            (ValueType::Range { .. }, Binding::Range { start, end, guards }) => {
                let guards = guards.clone();
                let mut ops = Vec::new();
                if let Some(output) = output_of(self, Some(RangeEndpoint::Start))? {
                    self.publish_scalar(output, *start, &guards, &mut ops)?;
                }
                if let Some(output) = output_of(self, Some(RangeEndpoint::End))? {
                    self.publish_scalar(output, *end, &guards, &mut ops)?;
                }
                em.push_ops(ops);
                Ok(())
            }
            (ValueType::Tensor(tensor), Binding::Tensor(source)) => {
                if let Some(output) = output_of(self, None)? {
                    let place = self.output_place(output)?;
                    let source = source.clone();
                    self.copy_standalone(&source, place, tensor, em)?;
                }
                Ok(())
            }
            (ValueType::Tuple(items), Binding::Tuple(bindings)) => {
                if items.len() != bindings.len() {
                    return Err(self.defect(
                        Package::L1,
                        format!("tuple value {} has {} components but {} bindings", canonical.0, items.len(), bindings.len()),
                    ));
                }
                for (index, (item, binding)) in items.iter().zip(bindings).enumerate() {
                    path.push(index as u32);
                    self.publish_value(canonical, binding, item, path, em)?;
                    path.pop();
                }
                Ok(())
            }
            (ValueType::Void, _) => Ok(()),
            (ty, binding) => Err(self.defect(
                Package::K1,
                format!("value {} of type {ty} is bound as {binding:?}", canonical.0),
            )),
        }
    }

    fn output_place(&self, output: KernelOutputId) -> Result<KernelPlaceRef, CompilerDefect> {
        match &self.interface.outputs[output].route {
            ValueRoute::Tensor(_) => Ok(KernelPlaceRef::Output(output)),
            route @ (ValueRoute::Void | ValueRoute::Scalar(_)) => Err(self.defect(
                Package::D1,
                format!("tensor output {} is routed as {route:?}", output.0),
            )),
        }
    }

    /// The places a tensor value produced by this block must be written to:
    /// its output residence (when it crosses a cut) and its kernel-local
    /// residence (when D1 spilled it inside the block).
    fn destinations(&self, canonical: CanonicalValueId) -> Result<Vec<KernelPlaceRef>, CompilerDefect> {
        let mut places = Vec::new();
        let leaf = self.leaf_of(canonical, &ValuePath::default(), None)?;
        if let Some(output) = self.output_leaves.get(&leaf).copied() {
            places.push(self.output_place(output)?);
        }
        if let Some(local) = self.local_values.get(&canonical).copied() {
            places.push(KernelPlaceRef::Local(local));
        }
        Ok(places)
    }

    fn note_store_site(&mut self, place: KernelPlaceRef) {
        if let KernelPlaceRef::Output(output) = place {
            self.output_sites[output.0 as usize] += 1;
        }
    }

    // -- iterations --------------------------------------------------------------

    /// Make `domain` the open iteration of `em` (closing another domain's
    /// iteration first) and record the node's storage effects on it.
    fn ensure_open(
        &mut self,
        em: &mut Emitter<D::Intrinsic>,
        domain: &[ExtentExpr],
        reads: &BTreeSet<CanonicalStorageId>,
        writes: &BTreeSet<CanonicalStorageId>,
    ) -> Result<(), CompilerDefect> {
        if let Some(open) = &em.open {
            if open.domain != domain {
                em.flush();
            }
        }
        if em.open.is_none() {
            let free = self.free_axes.clone();
            let prefix_matches = free.len() <= domain.len()
                && free.iter().zip(domain).all(|((_, axis), extent)| axis == extent);
            if !prefix_matches {
                return Err(self.defect(
                    Package::D1,
                    format!(
                        "the block's binder-less axes {:?} are not a prefix of the elementwise domain {domain:?}",
                        free.iter().map(|(_, extent)| extent.clone()).collect::<Vec<_>>()
                    ),
                ));
            }
            let mut coords: Vec<KernelValueRef> = free.iter().map(|(id, _)| KernelValueRef::Axis(*id)).collect();
            let mut prelude = Vec::new();
            let mut residual = Vec::new();
            if domain.len() > free.len() {
                let zero = self.const_i32(0, &mut prelude);
                for extent in &domain[free.len()..] {
                    let end = self.extent_ref(extent, &mut prelude)?;
                    let binder = self.fresh_scalar(DType::I32);
                    residual.push((binder, zero, end));
                    coords.push(KernelValueRef::Ssa(binder));
                }
            }
            em.open = Some(OpenIteration {
                domain: domain.to_vec(),
                coords,
                residual,
                prelude,
                body: Vec::new(),
                elements: BTreeMap::new(),
                reads: BTreeSet::new(),
                writes: BTreeSet::new(),
            });
        }
        let open = self.open_iter(em)?;
        open.reads.extend(reads.iter().copied());
        open.writes.extend(writes.iter().copied());
        Ok(())
    }

    /// The open iteration of `em`. Every caller has just opened (or kept
    /// open) an iteration through `ensure_open`; a missing one here is a
    /// K1 formation defect.
    fn open_iter<'e>(
        &self,
        em: &'e mut Emitter<D::Intrinsic>,
    ) -> Result<&'e mut OpenIteration<D::Intrinsic>, CompilerDefect> {
        em.open
            .as_mut()
            .ok_or_else(|| self.defect(Package::K1, "an elementwise op is lowered into an open iteration, but none is open"))
    }

    /// Append ops to the open iteration's body (after `ensure_open`).
    fn push_body(
        &mut self,
        em: &mut Emitter<D::Intrinsic>,
        ops: Ops<D>,
    ) -> Result<(), CompilerDefect> {
        em.synchronize(&ops);
        self.open_iter(em)?.body.extend(ops);
        Ok(())
    }

    /// A standalone `Repeat` nest over `domain` (fresh binders, no block
    /// axes), with `body` producing the per-coordinate ops.
    fn nest(
        &mut self,
        domain: &[ExtentExpr],
        ops: &mut Ops<D>,
        body: &mut dyn FnMut(&mut Self, &[KernelValueRef], &mut Ops<D>) -> Result<(), CompilerDefect>,
    ) -> Result<(), CompilerDefect> {
        let mut prelude = Vec::new();
        let mut residual = Vec::new();
        let mut coords = Vec::new();
        if !domain.is_empty() {
            let zero = self.const_i32(0, &mut prelude);
            for extent in domain {
                let end = self.extent_ref(extent, &mut prelude)?;
                let binder = self.fresh_scalar(DType::I32);
                residual.push((binder, zero, end));
                coords.push(KernelValueRef::Ssa(binder));
            }
        }
        let mut inner = Vec::new();
        body(self, &coords, &mut inner)?;
        if inner.is_empty() {
            return Ok(());
        }
        ops.extend(prelude);
        for (binder, start, end) in residual.into_iter().rev() {
            inner = vec![KernelOp::Core(CoreKernelOp::Repeat {
                binder,
                start,
                end,
                body: inner,
            })];
        }
        ops.extend(inner);
        Ok(())
    }

    /// Record a computed dense tensor result at the open iteration's
    /// coordinates: store it into every destination, expose it to fused
    /// consumers, and bind its kernel-local residence for later ones.
    fn computed_result(
        &mut self,
        canonical: CanonicalValueId,
        ty: &TensorType,
        value: KernelValueRef,
        coords: &[KernelValueRef],
        body: &mut Ops<D>,
    ) -> Result<(), CompilerDefect> {
        let dtype = self.dense_dtype(ty)?;
        let mut local = None;
        for place in self.destinations(canonical)? {
            body.push(KernelOp::Core(CoreKernelOp::Store {
                place,
                coords: coords.to_vec(),
                value,
                dtype,
            }));
            self.note_store_site(place);
            if let KernelPlaceRef::Local(_) = place {
                local = Some(place);
            }
        }
        if let Some(place) = local {
            self.bind(
                canonical,
                Binding::Tensor(TensorBinding::Place {
                    place,
                    ty: ty.clone(),
                }),
            );
        }
        Ok(())
    }

    fn expose_element(
        &self,
        em: &mut Emitter<D::Intrinsic>,
        canonical: CanonicalValueId,
        value: KernelValueRef,
        ty: &TensorType,
    ) -> Result<(), CompilerDefect> {
        self.open_iter(em)?
            .elements
            .insert(canonical, (value, ty.clone()));
        Ok(())
    }

    // -- element access --------------------------------------------------------

    /// Resolve an addressable binding to its base place and coordinates,
    /// emitting the coordinate arithmetic of every in-block view and
    /// collecting the guards of every view crossed (outermost first).
    fn resolve_place(
        &mut self,
        binding: &TensorBinding,
        coords: &[KernelValueRef],
        ops: &mut Ops<D>,
        guards: &mut Vec<AttachedGuard>,
    ) -> Result<(KernelPlaceRef, Vec<KernelValueRef>, TensorType), CompilerDefect> {
        if coords.len() != binding.ty().rank() {
            return Err(self.defect(
                Package::K1,
                format!("{} coordinates address a rank-{} tensor", coords.len(), binding.ty().rank()),
            ));
        }
        match binding {
            TensorBinding::Place { place, ty } => Ok((*place, coords.to_vec(), ty.clone())),
            TensorBinding::View {
                source,
                step,
                ty,
                guards: own,
            } => {
                guards.extend(own.iter().cloned());
                let source_coords = self.view_coords(step, ty, source.ty(), coords, ops)?;
                self.resolve_place(source, &source_coords, ops, guards)
            }
            TensorBinding::Snapshot { value, .. } => Err(self.rule_defect(format!(
                "snapshot value {} is not addressable for writing; a snapshot is read-only",
                value.0
            ))),
            TensorBinding::Plane { value, .. } => Err(self.defect(
                Package::L1,
                format!("plane accessor value {} is never writable", value.0),
            )),
            TensorBinding::Element { .. } => Err(self.rule_defect(
                "a fused computed tensor has no residence to address; the rule must spill it",
            )),
        }
    }

    /// Map view coordinates to source coordinates for one in-block step.
    fn view_coords(
        &mut self,
        step: &KernelViewStep,
        view_ty: &TensorType,
        source_ty: &TensorType,
        coords: &[KernelValueRef],
        ops: &mut Ops<D>,
    ) -> Result<Vec<KernelValueRef>, CompilerDefect> {
        match step {
            KernelViewStep::Reshape { source_shape } => {
                if source_ty.axes != *source_shape {
                    return Err(self.defect(
                        Package::L1,
                        format!("reshape source shape {source_shape:?} disagrees with the source tensor {:?}", source_ty.axes),
                    ));
                }
                // Row-major linearization over the view shape, then
                // delinearization over the source shape.
                let mut linear: Option<KernelValueRef> = None;
                for (axis, coord) in coords.iter().enumerate() {
                    let term = match linear {
                        None => *coord,
                        Some(acc) => {
                            let extent = self.extent_ref(&view_ty.axes[axis], ops)?;
                            let scaled = self.binary(BinaryOp::Mul, acc, extent, DType::I32, ops);
                            self.binary(BinaryOp::Add, scaled, *coord, DType::I32, ops)
                        }
                    };
                    linear = Some(term);
                }
                let linear = match linear {
                    Some(linear) => linear,
                    None => self.const_i32(0, ops),
                };
                let rank = source_shape.len();
                let mut out = Vec::with_capacity(rank);
                let mut rest = linear;
                for axis in 0..rank {
                    if axis + 1 == rank {
                        out.push(rest);
                    } else {
                        let stride = self.row_major_stride(&source_shape[axis + 1..], ops)?;
                        out.push(self.binary(BinaryOp::Div, rest, stride, DType::I32, ops));
                        rest = self.binary(BinaryOp::Rem, rest, stride, DType::I32, ops);
                    }
                }
                Ok(out)
            }
            KernelViewStep::Transpose { permutation } => {
                let rank = source_ty.rank();
                let mut out = vec![None; rank];
                for (axis, coord) in coords.iter().enumerate() {
                    let source_axis = permutation.get(axis).map(|p| *p as usize).ok_or_else(|| {
                        self.defect(Package::L1, format!("transpose permutation {permutation:?} is shorter than rank {rank}"))
                    })?;
                    match out.get_mut(source_axis) {
                        Some(slot @ None) => *slot = Some(*coord),
                        _ => {
                            return Err(self.defect(
                                Package::L1,
                                format!("transpose permutation {permutation:?} is not a permutation of rank {rank}"),
                            ))
                        }
                    }
                }
                out.into_iter()
                    .map(|coord| {
                        coord.ok_or_else(|| {
                            self.defect(Package::L1, format!("transpose permutation {permutation:?} leaves a source axis unmapped"))
                        })
                    })
                    .collect()
            }
            KernelViewStep::Slice { axes } => {
                let mut out = Vec::with_capacity(axes.len());
                let mut cursor = 0usize;
                for axis in axes {
                    match axis {
                        KernelSliceAxis::Full => {
                            out.push(*coords.get(cursor).ok_or_else(|| self.slice_rank_defect(axes, coords))?);
                            cursor += 1;
                        }
                        KernelSliceAxis::Point(point) => out.push(*point),
                        KernelSliceAxis::Range { start } => {
                            let coord = *coords.get(cursor).ok_or_else(|| self.slice_rank_defect(axes, coords))?;
                            cursor += 1;
                            out.push(match start {
                                Some(start) => self.binary(BinaryOp::Add, coord, *start, DType::I32, ops),
                                None => coord,
                            });
                        }
                    }
                }
                if cursor != coords.len() {
                    return Err(self.slice_rank_defect(axes, coords));
                }
                Ok(out)
            }
        }
    }

    fn slice_rank_defect(&self, axes: &[KernelSliceAxis], coords: &[KernelValueRef]) -> CompilerDefect {
        self.defect(
            Package::L1,
            format!("slice axes {axes:?} do not match {} view coordinates", coords.len()),
        )
    }

    fn row_major_stride(&mut self, axes: &[ExtentExpr], ops: &mut Ops<D>) -> Result<KernelValueRef, CompilerDefect> {
        let mut stride: Option<KernelValueRef> = None;
        for extent in axes {
            let value = self.extent_ref(extent, ops)?;
            stride = Some(match stride {
                None => value,
                Some(acc) => self.binary(BinaryOp::Mul, acc, value, DType::I32, ops),
            });
        }
        Ok(match stride {
            Some(stride) => stride,
            None => self.const_i32(1, ops),
        })
    }

    /// The dense element of a tensor binding at `coords` (a packed element
    /// decodes to `f32` through the registry recipe).
    fn element_at(
        &mut self,
        binding: &TensorBinding,
        coords: &[KernelValueRef],
        ops: &mut Ops<D>,
    ) -> Result<(KernelValueRef, DType), CompilerDefect> {
        match binding {
            TensorBinding::Element { value, ty } => Ok((*value, self.dense_dtype(ty)?)),
            TensorBinding::Snapshot { source, .. } => self.element_at(source, coords, ops),
            TensorBinding::Plane {
                source,
                repr,
                field,
                ty,
                ..
            } => {
                let dtype = self.dense_dtype(ty)?;
                let mut inner = Vec::new();
                let mut guards = Vec::new();
                let (place, place_coords) = self.plane_place(source, repr, *field, coords, &mut inner, &mut guards)?;
                let into = self.fresh_scalar(dtype);
                inner.push(KernelOp::Core(CoreKernelOp::PlaneLoad {
                    into,
                    place,
                    coords: place_coords,
                    repr: repr.clone(),
                    plane: *field,
                    dtype,
                }));
                let wrapped = self.wrap(inner, &guards);
                ops.extend(wrapped);
                Ok((KernelValueRef::Ssa(into), dtype))
            }
            TensorBinding::Place { .. } | TensorBinding::View { .. } => {
                let mut inner = Vec::new();
                let mut guards = Vec::new();
                let (place, place_coords, ty) = self.resolve_place(binding, coords, &mut inner, &mut guards)?;
                let result = match &ty.elem {
                    Elem::Dtype(dtype) => {
                        let into = self.fresh_scalar(*dtype);
                        inner.push(KernelOp::Core(CoreKernelOp::Load {
                            into,
                            place,
                            coords: place_coords,
                            dtype: *dtype,
                        }));
                        (KernelValueRef::Ssa(into), *dtype)
                    }
                    Elem::Repr(name) => {
                        let value = self.decode_at(place, &place_coords, name, DType::F32, &mut inner)?;
                        (value, DType::F32)
                    }
                    Elem::Param(name) => {
                        return Err(self.defect(
                            Package::W1,
                            format!("element parameter `{name}` survived specialization"),
                        ))
                    }
                };
                let wrapped = self.wrap(inner, &guards);
                ops.extend(wrapped);
                Ok(result)
            }
        }
    }

    /// Store `value` (already at the place's dtype) at `coords` of an
    /// addressable binding.
    fn store_at(
        &mut self,
        binding: &TensorBinding,
        coords: &[KernelValueRef],
        value: KernelValueRef,
        dtype: DType,
        ops: &mut Ops<D>,
    ) -> Result<(), CompilerDefect> {
        let mut inner = Vec::new();
        let mut guards = Vec::new();
        let (place, place_coords, ty) = self.resolve_place(binding, coords, &mut inner, &mut guards)?;
        let place_dtype = self.dense_dtype(&ty)?;
        if place_dtype != dtype {
            return Err(self.defect(
                Package::K1,
                format!("a {} value is stored into a {} place", dtype.name(), place_dtype.name()),
            ));
        }
        inner.push(KernelOp::Core(CoreKernelOp::Store {
            place,
            coords: place_coords,
            value,
            dtype,
        }));
        self.note_store_site(place);
        let wrapped = self.wrap(inner, &guards);
        ops.extend(wrapped);
        Ok(())
    }

    /// Expand the registry decode recipe of one packed element into plane
    /// reads and scalar ops; the result is the recipe's output temporary.
    fn decode_at(
        &mut self,
        place: KernelPlaceRef,
        coords: &[KernelValueRef],
        repr: &str,
        output: DType,
        ops: &mut Ops<D>,
    ) -> Result<KernelValueRef, CompilerDefect> {
        let Some(representation) = lookup_repr(repr) else {
            return Err(self.defect(Package::L1, format!("unknown representation `{repr}`")));
        };
        let recipe = representation.decode_recipe_to(output);
        let mut temps: Vec<Option<KernelValueRef>> = vec![None; recipe.temporaries.len()];
        let temp = |former: &Self, temps: &[Option<KernelValueRef>], id: seismic_lang::repr::DecodeTemp| -> Result<KernelValueRef, CompilerDefect> {
            temps
                .get(id.0 as usize)
                .copied()
                .flatten()
                .ok_or_else(|| former.defect(Package::K1, format!("decode temporary {} of `{repr}` is used before its definition", id.0)))
        };
        for step in &recipe.steps {
            let defined = step.defines();
            let value = match step {
                DecodeStep::ReadPlaneField { plane, field, .. } => {
                    let schema: &PlaneSchema = recipe.planes.get(*plane as usize).ok_or_else(|| {
                        self.defect(Package::K1, format!("decode of `{repr}` reads plane {plane} which it does not declare"))
                    })?;
                    let into = self.fresh_scalar(schema.storage_dtype);
                    ops.push(KernelOp::Core(CoreKernelOp::PackedPlaneRead {
                        into,
                        place,
                        coords: coords.to_vec(),
                        repr: repr.to_string(),
                        plane: schema.field,
                        entry: *field,
                        dtype: schema.storage_dtype,
                    }));
                    KernelValueRef::Ssa(into)
                }
                DecodeStep::InterpretCode {
                    raw,
                    bits,
                    interpretation,
                    ..
                } => {
                    let raw_value = temp(self, &temps, *raw)?;
                    let raw_dtype = recipe.dtype(*raw);
                    match interpretation {
                        CodeInterpretation::Unsigned => {
                            self.cast_if_needed(raw_value, raw_dtype, DType::I32, ops)
                        }
                        CodeInterpretation::TwosComplement => {
                            let shift = self.const_i32(i64::from(32 - *bits), ops);
                            let shifted = self.binary(BinaryOp::Shl, raw_value, shift, raw_dtype, ops);
                            let signed = self.cast_if_needed(shifted, raw_dtype, DType::I32, ops);
                            self.binary(BinaryOp::Shr, signed, shift, DType::I32, ops)
                        }
                        CodeInterpretation::Offset(zero) => {
                            let signed = self.cast_if_needed(raw_value, raw_dtype, DType::I32, ops);
                            let zero = self.const_i32(i64::from(*zero), ops);
                            self.binary(BinaryOp::Sub, signed, zero, DType::I32, ops)
                        }
                        CodeInterpretation::Table(table) => {
                            let index = self.cast_if_needed(raw_value, raw_dtype, DType::U32, ops);
                            let into = self.fresh_scalar(DType::I32);
                            ops.push(KernelOp::Core(CoreKernelOp::TableLookup {
                                into,
                                index,
                                table: table.to_vec(),
                            }));
                            KernelValueRef::Ssa(into)
                        }
                    }
                }
                DecodeStep::ConvertToF32 { from, .. } => {
                    let value = temp(self, &temps, *from)?;
                    self.cast_if_needed(value, recipe.dtype(*from), DType::F32, ops)
                }
                DecodeStep::Multiply { left, right, .. } => {
                    let left = temp(self, &temps, *left)?;
                    let right = temp(self, &temps, *right)?;
                    self.binary(BinaryOp::Mul, left, right, DType::F32, ops)
                }
                DecodeStep::Negate { from, .. } => {
                    let operand = temp(self, &temps, *from)?;
                    let into = self.fresh_scalar(DType::F32);
                    ops.push(KernelOp::Core(CoreKernelOp::Unary {
                        into,
                        op: seismic_lang::syntax::ast::UnaryOp::Neg,
                        operand,
                        dtype: DType::F32,
                    }));
                    KernelValueRef::Ssa(into)
                }
                DecodeStep::MultiplyAdd {
                    factor,
                    multiplicand,
                    addend,
                    ..
                } => {
                    let a = temp(self, &temps, *factor)?;
                    let b = temp(self, &temps, *multiplicand)?;
                    let c = temp(self, &temps, *addend)?;
                    let into = self.fresh_scalar(DType::F32);
                    ops.push(KernelOp::Core(CoreKernelOp::Fma {
                        into,
                        a,
                        b,
                        c,
                        dtype: DType::F32,
                    }));
                    KernelValueRef::Ssa(into)
                }
                DecodeStep::Cast { from, to, .. } => {
                    let value = temp(self, &temps, *from)?;
                    self.cast_if_needed(value, recipe.dtype(*from), *to, ops)
                }
            };
            temps[defined.0 as usize] = Some(value);
        }
        temp(self, &temps, recipe.output)
    }

    // -- representation planes ------------------------------------------------

    fn packed_repr(&self, ty: &TensorType) -> Result<(String, usize), CompilerDefect> {
        let Elem::Repr(name) = &ty.elem else {
            return Err(self.defect(Package::L1, format!("tensor {ty:?} is not packed")));
        };
        let axis = ty.packed_axis.ok_or_else(|| {
            self.defect(Package::L1, format!("packed tensor `{name}` has no packing axis"))
        })?;
        Ok((name.clone(), axis))
    }

    fn plane_schema(&self, repr: &str, field: PlaneField) -> Result<PlaneSchema, CompilerDefect> {
        let Some(representation) = lookup_repr(repr) else {
            return Err(self.defect(Package::L1, format!("unknown representation `{repr}`")));
        };
        representation
            .plane_schemas()
            .into_iter()
            .find(|schema| schema.field == field)
            .ok_or_else(|| self.defect(Package::L1, format!("`{repr}` has no `{}` plane", field.name())))
    }

    /// The number of storage elements of one plane row holding `width`
    /// logical values along the packing axis.
    fn plane_row_elements(
        &mut self,
        schema: &PlaneSchema,
        width: &ExtentExpr,
        ops: &mut Ops<D>,
    ) -> Result<KernelValueRef, CompilerDefect> {
        let group = u64::from(schema.group);
        let fields = u64::from(schema.fields);
        let bits = u64::from(schema.entry_bits);
        if let Some(width) = width.as_static() {
            let entries = width.div_ceil(group) * fields;
            let elements = match schema.encoding {
                PlaneEncoding::Dense(_) => entries,
                PlaneEncoding::Packed { .. } => (entries * bits).div_ceil(32),
            };
            let elements = i64::from(i32::try_from(elements).map_err(|_| {
                self.defect(Package::L1, format!("plane row of {elements} elements exceeds the i32 index space"))
            })?);
            return Ok(self.const_i32(elements, ops));
        }
        let width = self.extent_ref(width, ops)?;
        let group_minus_one = self.const_i32(i64::from(schema.group) - 1, ops);
        let group_ref = self.const_i32(i64::from(schema.group), ops);
        let sum = self.binary(BinaryOp::Add, width, group_minus_one, DType::I32, ops);
        let groups = self.binary(BinaryOp::Div, sum, group_ref, DType::I32, ops);
        let fields_ref = self.const_i32(i64::from(schema.fields), ops);
        let entries = self.binary(BinaryOp::Mul, groups, fields_ref, DType::I32, ops);
        Ok(match schema.encoding {
            PlaneEncoding::Dense(_) => entries,
            PlaneEncoding::Packed { .. } => {
                let bits_ref = self.const_i32(i64::from(schema.entry_bits), ops);
                let total = self.binary(BinaryOp::Mul, entries, bits_ref, DType::I32, ops);
                let thirty_one = self.const_i32(31, ops);
                let rounded = self.binary(BinaryOp::Add, total, thirty_one, DType::I32, ops);
                let word = self.const_i32(32, ops);
                self.binary(BinaryOp::Div, rounded, word, DType::I32, ops)
            }
        })
    }

    /// Resolve plane coordinates (outer view coordinates plus the
    /// storage-element ordinal along the packing axis) through the in-block
    /// slice views of a packed binding to its base place.
    fn plane_place(
        &mut self,
        binding: &TensorBinding,
        repr: &str,
        field: PlaneField,
        coords: &[KernelValueRef],
        ops: &mut Ops<D>,
        guards: &mut Vec<AttachedGuard>,
    ) -> Result<(KernelPlaceRef, Vec<KernelValueRef>), CompilerDefect> {
        match binding {
            TensorBinding::Place { place, .. } => Ok((*place, coords.to_vec())),
            TensorBinding::Snapshot { source, .. } => self.plane_place(source, repr, field, coords, ops, guards),
            TensorBinding::View {
                source,
                step,
                ty,
                guards: own,
            } => {
                guards.extend(own.iter().cloned());
                let (_, packed_axis) = self.packed_repr(ty)?;
                let KernelViewStep::Slice { axes } = step else {
                    return Err(self.defect(
                        Package::L1,
                        format!("a packed tensor cannot be reshaped or transposed (`{repr}`)"),
                    ));
                };
                let (_, source_axis) = self.packed_repr(source.ty())?;
                let schema = self.plane_schema(repr, field)?;
                let mut out = Vec::with_capacity(axes.len());
                let mut cursor = 0usize;
                for (index, axis) in axes.iter().enumerate() {
                    let on_packing = index == source_axis;
                    match axis {
                        KernelSliceAxis::Full => {
                            out.push(*coords.get(cursor).ok_or_else(|| self.slice_rank_defect(axes, coords))?);
                            cursor += 1;
                        }
                        KernelSliceAxis::Point(point) => {
                            if on_packing {
                                return Err(self.defect(
                                    Package::L1,
                                    format!("a point selection on the packing axis of `{repr}` leaves no packed view"),
                                ));
                            }
                            out.push(*point);
                        }
                        KernelSliceAxis::Range { start } => {
                            let coord = *coords.get(cursor).ok_or_else(|| self.slice_rank_defect(axes, coords))?;
                            cursor += 1;
                            let mapped = match start {
                                None => coord,
                                Some(start) if !on_packing => {
                                    self.binary(BinaryOp::Add, coord, *start, DType::I32, ops)
                                }
                                Some(start) => {
                                    // Group-aligned start (checker): the
                                    // storage-element offset of its entries.
                                    let group = self.const_i32(i64::from(schema.group), ops);
                                    let groups = self.binary(BinaryOp::Div, *start, group, DType::I32, ops);
                                    let fields = self.const_i32(i64::from(schema.fields), ops);
                                    let entries = self.binary(BinaryOp::Mul, groups, fields, DType::I32, ops);
                                    let offset = match schema.encoding {
                                        PlaneEncoding::Dense(_) => entries,
                                        PlaneEncoding::Packed { .. } => {
                                            let bits = self.const_i32(i64::from(schema.entry_bits), ops);
                                            let total = self.binary(BinaryOp::Mul, entries, bits, DType::I32, ops);
                                            let word = self.const_i32(32, ops);
                                            self.binary(BinaryOp::Div, total, word, DType::I32, ops)
                                        }
                                    };
                                    self.binary(BinaryOp::Add, coord, offset, DType::I32, ops)
                                }
                            };
                            out.push(mapped);
                        }
                    }
                }
                let _ = packed_axis;
                if cursor != coords.len() {
                    return Err(self.slice_rank_defect(axes, coords));
                }
                self.plane_place(source, repr, field, &out, ops, guards)
            }
            TensorBinding::Plane { value, .. } => Err(self.defect(
                Package::L1,
                format!("plane accessor value {} is not itself packed", value.0),
            )),
            TensorBinding::Element { .. } => Err(self.rule_defect(
                "a fused computed tensor has no representation planes; the rule must spill it",
            )),
        }
    }

    /// Copy every plane of a packed tensor from `source` into `destination`
    /// (both of type `ty`) as standalone loops over storage elements.
    fn copy_planes(
        &mut self,
        source: &TensorBinding,
        destination: &TensorBinding,
        ty: &TensorType,
        ops: &mut Ops<D>,
    ) -> Result<(), CompilerDefect> {
        let (repr, packed_axis) = self.packed_repr(ty)?;
        let Some(representation) = lookup_repr(&repr) else {
            return Err(self.defect(Package::L1, format!("unknown representation `{repr}`")));
        };
        let outer: Vec<ExtentExpr> = ty
            .axes
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != packed_axis)
            .map(|(_, extent)| extent.clone())
            .collect();
        let width = ty.axes[packed_axis].clone();
        for schema in representation.plane_schemas() {
            let source = source.clone();
            let destination = destination.clone();
            let repr = repr.clone();
            let width = width.clone();
            self.nest(&outer, ops, &mut |former, outer_coords, body| {
                let elements = former.plane_row_elements(&schema, &width, body)?;
                let zero = former.const_i32(0, body);
                let binder = former.fresh_scalar(DType::I32);
                let mut coords: Vec<KernelValueRef> = outer_coords.to_vec();
                coords.insert(packed_axis, KernelValueRef::Ssa(binder));
                let mut inner = Vec::new();
                let mut load_guards = Vec::new();
                let (src_place, src_coords) = former.plane_place(&source, &repr, schema.field, &coords, &mut inner, &mut load_guards)?;
                let into = former.fresh_scalar(schema.storage_dtype);
                inner.push(KernelOp::Core(CoreKernelOp::PlaneLoad {
                    into,
                    place: src_place,
                    coords: src_coords,
                    repr: repr.clone(),
                    plane: schema.field,
                    dtype: schema.storage_dtype,
                }));
                let mut inner = former.wrap(inner, &load_guards);
                let mut store = Vec::new();
                let mut store_guards = Vec::new();
                let (dst_place, dst_coords) = former.plane_place(&destination, &repr, schema.field, &coords, &mut store, &mut store_guards)?;
                store.push(KernelOp::Core(CoreKernelOp::PlaneStore {
                    place: dst_place,
                    coords: dst_coords,
                    repr: repr.clone(),
                    plane: schema.field,
                    value: KernelValueRef::Ssa(into),
                    dtype: schema.storage_dtype,
                }));
                former.note_store_site(dst_place);
                inner.extend(former.wrap(store, &store_guards));
                body.push(KernelOp::Core(CoreKernelOp::Repeat {
                    binder,
                    start: zero,
                    end: elements,
                    body: inner,
                }));
                Ok(())
            })?;
        }
        Ok(())
    }

    /// A standalone copy of a tensor binding into `destination` (dense
    /// element loop or packed plane copies).
    fn copy_standalone(
        &mut self,
        source: &TensorBinding,
        destination: KernelPlaceRef,
        ty: &TensorType,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        em.flush();
        let target = TensorBinding::Place {
            place: destination,
            ty: ty.clone(),
        };
        let mut ops = Vec::new();
        match &ty.elem {
            Elem::Repr(_) => self.copy_planes(source, &target, ty, &mut ops)?,
            Elem::Dtype(dtype) => {
                let dtype = *dtype;
                let axes = ty.axes.clone();
                let source = source.clone();
                self.nest(&axes, &mut ops, &mut |former, coords, body| {
                    let (value, from) = former.element_at(&source, coords, body)?;
                    let value = former.cast_if_needed(value, from, dtype, body);
                    former.store_at(&target, coords, value, dtype, body)
                })?;
            }
            Elem::Param(name) => {
                return Err(self.defect(
                    Package::W1,
                    format!("element parameter `{name}` survived specialization"),
                ))
            }
        }
        em.push_ops(ops);
        Ok(())
    }

    // -- primitives ------------------------------------------------------------

    fn single_output<'n>(&self, node: &OwnedNodeRef, logical: &'n LogicalNode) -> Result<&'n GraphValue, CompilerDefect> {
        match logical.outputs.as_slice() {
            [output] => Ok(output),
            outputs => Err(self.defect(
                Package::L1,
                format!("{node:?} has {} outputs where exactly one was expected", outputs.len()),
            )),
        }
    }

    fn input(&self, node: &OwnedNodeRef, logical: &LogicalNode, index: usize) -> Result<CanonicalValueId, CompilerDefect> {
        match logical.inputs.get(index) {
            Some(value) => self.canonical(node.graph, *value),
            None => Err(self.defect(Package::L1, format!("{node:?} has no input {index}"))),
        }
    }

    fn lower_primitive(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        op: &PrimitiveOp,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        match op {
            PrimitiveOp::Constant(literal) => {
                let output = self.single_output(node, logical)?;
                let canonical = self.canonical(node.graph, output.id())?;
                let dtype = self.scalar_dtype_of(output.kind())?;
                let value = match literal {
                    Literal::Int(value) => ConstantValue::Int(*value),
                    Literal::Float(value) => ConstantValue::Float { bits: value.to_bits() },
                    Literal::Bool(value) => ConstantValue::Bool(*value),
                    Literal::ShapeParam(name) => {
                        return Err(self.defect(
                            Package::W1,
                            format!("shape parameter `{name}` survived specialization as a constant"),
                        ))
                    }
                };
                let mut ops = Vec::new();
                let value = self.const_op(value, dtype, &mut ops);
                let guards = self.guards_for(node, &BTreeMap::new())?;
                let ops = self.wrap(ops, &guards);
                em.push_ops(ops);
                self.define_scalar(canonical, value, em)
            }
            PrimitiveOp::RuntimeExtent(id) => {
                let output = self.single_output(node, logical)?;
                let canonical = self.canonical(node.graph, output.id())?;
                let into = self.fresh_scalar(DType::I32);
                em.push_ops(vec![KernelOp::Core(CoreKernelOp::RuntimeExtent { into, extent: *id })]);
                self.define_scalar(canonical, KernelValueRef::Ssa(into), em)
            }
            PrimitiveOp::Capability(id) => {
                let operands: Vec<GraphValueId> = logical.inputs.clone();
                self.lower_intrinsic_use(node, logical, id, &operands, em)
            }
            PrimitiveOp::Primitive(id) => match lowering(id) {
                CoreLoweringFamily::Structural => self.lower_structural(node, logical, id, em),
                CoreLoweringFamily::TensorAlloc => self.lower_alloc(node, logical, em),
                CoreLoweringFamily::ViewTransform => self.lower_view(node, logical, em),
                CoreLoweringFamily::ExtentRead | CoreLoweringFamily::RuntimeExtent => {
                    self.lower_extent(node, logical, id, em)
                }
                CoreLoweringFamily::Constant => Err(self.defect(
                    Package::K1,
                    format!("registry primitive `{id}` lowers as the typed-literal family, which has no primitive id"),
                )),
                CoreLoweringFamily::ScalarArithmetic
                | CoreLoweringFamily::ElementwiseMap
                | CoreLoweringFamily::Cast
                | CoreLoweringFamily::Select => self.lower_elementwise(node, logical, id, em),
                CoreLoweringFamily::ElementRead => self.lower_element_read(node, logical, id, em),
                CoreLoweringFamily::ElementWrite => self.lower_element_write(node, logical, id, em),
                CoreLoweringFamily::CopyInto => self.lower_copy_into(node, logical, em),
                CoreLoweringFamily::Fill => self.lower_fill(node, logical, id, em),
                CoreLoweringFamily::BulkCopy => self.lower_bulk_copy(node, logical, id, em),
                CoreLoweringFamily::PackedDecode => self.lower_decode(node, logical, em),
                CoreLoweringFamily::PackedPlaneRead => self.lower_plane_read(node, logical, id, em),
                CoreLoweringFamily::Atomic => self.lower_atomic(node, logical, id, em),
                CoreLoweringFamily::Reduce => Err(self.defect(
                    Package::L1,
                    format!("reduce primitive `{id}` reached kernel formation; reductions are reduction nodes"),
                )),
            },
        }
    }

    fn lower_structural(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let binding = match id {
            PrimitiveId::TuplePack => {
                let mut items = Vec::with_capacity(logical.inputs.len());
                for index in 0..logical.inputs.len() {
                    items.push(self.lookup(self.input(node, logical, index)?)?);
                }
                Binding::Tuple(items)
            }
            PrimitiveId::TupleGet(index) => match self.lookup(self.input(node, logical, 0)?)? {
                Binding::Tuple(items) => items.get(*index).cloned().ok_or_else(|| {
                    self.defect(Package::L1, format!("{node:?} projects component {index} of a {}-tuple", items.len()))
                })?,
                other => {
                    return Err(self.defect(Package::L1, format!("{node:?} projects a non-tuple binding {other:?}")))
                }
            },
            PrimitiveId::RangeMake => {
                let start = self.scalar(self.input(node, logical, 0)?)?;
                let end = self.scalar(self.input(node, logical, 1)?)?;
                let guards = self.guards_for(node, &BTreeMap::new())?;
                let mut ops = Vec::new();
                self.record_only(&guards, &mut ops);
                em.push_ops(ops);
                Binding::Range { start, end, guards }
            }
            PrimitiveId::RangeStart | PrimitiveId::RangeEnd => {
                let Binding::Range { start, end, guards } = self.lookup(self.input(node, logical, 0)?)? else {
                    return Err(self.defect(Package::L1, format!("{node:?} projects an endpoint of a non-range binding")));
                };
                let endpoint = if matches!(id, PrimitiveId::RangeStart) { start } else { end };
                if guards.is_empty() {
                    Binding::Scalar(endpoint)
                } else {
                    let into = self.fresh_scalar(DType::I32);
                    let copy = vec![KernelOp::Core(CoreKernelOp::Cast {
                        into,
                        operand: endpoint,
                        from: DType::I32,
                        to: DType::I32,
                    })];
                    let wrapped = self.wrap(copy, &guards);
                    em.push_ops(wrapped);
                    Binding::Scalar(KernelValueRef::Ssa(into))
                }
            }
            other => {
                return Err(self.defect(Package::K1, format!("`{other}` is not a structural primitive")))
            }
        };
        self.bind(canonical, binding.clone());
        let ty = output.ty();
        self.publish_value(canonical, &binding, &ty, &mut Vec::new(), em)
    }

    fn lower_alloc(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        _em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let GraphValueKind::Tensor { ty, .. } = output.kind() else {
            return Err(self.defect(Package::L1, format!("{node:?} allocates a non-tensor")));
        };
        // A declaration of storage that no node of this block uses: the
        // storage is written or read by other blocks or the boundary and
        // routes through the plan's residence graph. Nothing is lowered and
        // nothing is bound; the value has no in-block uses.
        let used_in_block = self.members.iter().any(|other| {
            *other != *node
                && self
                    .facts
                    .node(other)
                    .map(|logical| {
                        logical
                            .inputs
                            .iter()
                            .any(|input| {
                                self.canonical(other.graph, *input)
                                    .map(|c| c == canonical)
                                    .unwrap_or(false)
                            })
                    })
                    .unwrap_or(false)
        });
        if !used_in_block {
            return Ok(());
        }
        let place = self.storage_place(canonical, node)?;
        self.bind(
            canonical,
            Binding::Tensor(TensorBinding::Place {
                place,
                ty: ty.clone(),
            }),
        );
        Ok(())
    }

    /// The residence of a value naming fresh storage produced in the block:
    /// its kernel-local residence when D1 declared one, else its output
    /// residence when the value crosses a cut.
    fn storage_place(&self, canonical: CanonicalValueId, node: &OwnedNodeRef) -> Result<KernelPlaceRef, CompilerDefect> {
        let destinations = self.destinations(canonical)?;
        destinations
            .iter()
            .find(|place| matches!(place, KernelPlaceRef::Local(_)))
            .or_else(|| destinations.first())
            .copied()
            .ok_or_else(|| {
                self.defect(
                    Package::D1,
                    format!(
                        "{node:?} produces fresh storage (value {}) but the block interface declares neither a kernel-local nor an output residence for it",
                        canonical.0
                    ),
                )
            })
    }

    fn lower_view(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let GraphValueKind::Tensor {
            ty,
            source: TensorSource::View(view_id),
        } = output.kind()
        else {
            return Err(self.defect(Package::L1, format!("{node:?} produces a non-view tensor")));
        };
        let view = self.facts.view(OwnedViewRef {
            graph: node.graph,
            view: *view_id,
        });
        let source = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let source_ty = source.ty().clone();
        let step = match &view.transform {
            ViewTransform::Identity => {
                self.bind(canonical, Binding::Tensor(source));
                return Ok(());
            }
            ViewTransform::Reshape { source_shape } => {
                if *source_shape != source_ty.axes {
                    return Err(self.defect(
                        Package::L1,
                        format!("{node:?} reshapes from {source_shape:?} but its source has axes {:?}", source_ty.axes),
                    ));
                }
                KernelViewStep::Reshape {
                    source_shape: source_shape.clone(),
                }
            }
            ViewTransform::Transpose { permutation } => {
                if permutation.len() != source_ty.rank() || ty.rank() != source_ty.rank() {
                    return Err(self.defect(Package::L1, format!("{node:?} transposes with a permutation of the wrong rank")));
                }
                for (axis, source_axis) in permutation.iter().enumerate() {
                    if source_ty.axes.get(*source_axis as usize) != ty.axes.get(axis) {
                        return Err(self.defect(
                            Package::L1,
                            format!("{node:?} transpose axis {axis} extent disagrees with source axis {source_axis}"),
                        ));
                    }
                }
                KernelViewStep::Transpose {
                    permutation: permutation.clone(),
                }
            }
            ViewTransform::Slice { axes } => {
                if axes.len() != source_ty.rank() {
                    return Err(self.defect(Package::L1, format!("{node:?} slices {} axes of a rank-{} source", axes.len(), source_ty.rank())));
                }
                let mut lowered = Vec::with_capacity(axes.len());
                for axis in axes {
                    lowered.push(match axis {
                        SliceAxis::Full => KernelSliceAxis::Full,
                        SliceAxis::Point(point) => KernelSliceAxis::Point(self.scalar(self.canonical(node.graph, *point)?)?),
                        SliceAxis::Range { start, .. } => KernelSliceAxis::Range {
                            start: match start {
                                Some(start) => Some(self.scalar(self.canonical(node.graph, *start)?)?),
                                None => None,
                            },
                        },
                    });
                }
                KernelViewStep::Slice { axes: lowered }
            }
        };
        let guards = self.guards_for(node, &BTreeMap::new())?;
        let mut ops = Vec::new();
        self.record_only(&guards, &mut ops);
        em.push_ops(ops);
        self.bind(
            canonical,
            Binding::Tensor(TensorBinding::View {
                source: Box::new(source),
                step,
                ty: ty.clone(),
                guards,
            }),
        );
        Ok(())
    }

    fn lower_extent(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let axis = match id {
            PrimitiveId::Extent { axis } | PrimitiveId::ValidExtent { axis } => *axis,
            other => return Err(self.defect(Package::K1, format!("`{other}` is not an extent primitive"))),
        };
        let operand = self.input(node, logical, 0)?;
        let extent = match self.facts.value_kind(operand) {
            GraphValueKind::Tensor { ty, .. } => ty.axes.get(axis).cloned().ok_or_else(|| {
                self.defect(Package::L1, format!("{node:?} reads axis {axis} of a rank-{} tensor", ty.rank()))
            })?,
            other => return Err(self.defect(Package::L1, format!("{node:?} reads an extent of {other:?}"))),
        };
        let mut ops = Vec::new();
        let value = self.extent_ref(&extent, &mut ops)?;
        em.push_ops(ops);
        self.define_scalar(canonical, value, em)
    }

    /// The core op of one elementwise primitive over resolved scalar operands.
    fn elementwise_op(
        &mut self,
        node: &OwnedNodeRef,
        id: &PrimitiveId,
        operands: &[(KernelValueRef, DType)],
        result: DType,
    ) -> Result<(KernelOp<D::Intrinsic>, KernelSsaId), CompilerDefect> {
        let arity = |former: &Self, n: usize| -> Result<(), CompilerDefect> {
            if operands.len() == n {
                Ok(())
            } else {
                Err(former.defect(Package::L1, format!("`{id}` at {node:?} has {} operands, expected {n}", operands.len())))
            }
        };
        let op = match id {
            PrimitiveId::Unary(op) => {
                arity(self, 1)?;
                let (operand, dtype) = operands[0];
                let into = self.fresh_scalar(dtype);
                (CoreKernelOp::Unary { into, op: *op, operand, dtype }, into)
            }
            PrimitiveId::Binary(op) => {
                arity(self, 2)?;
                let (left, left_dtype) = operands[0];
                let (right, right_dtype) = operands[1];
                match op {
                    BinaryOp::Eq | BinaryOp::Ne | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        let dtype = DType::promote(left_dtype, right_dtype).ok_or_else(|| {
                            self.defect(Package::L1, format!("`{id}` at {node:?} compares {} with {}", left_dtype.name(), right_dtype.name()))
                        })?;
                        let rel = match op {
                            BinaryOp::Eq => RelOp::Eq,
                            BinaryOp::Ne => RelOp::Ne,
                            BinaryOp::Lt => RelOp::Lt,
                            BinaryOp::Le => RelOp::Le,
                            BinaryOp::Gt => RelOp::Gt,
                            _ => RelOp::Ge,
                        };
                        let into = self.fresh_scalar(DType::Bool);
                        (CoreKernelOp::Compare { into, op: rel, left, right, dtype }, into)
                    }
                    BinaryOp::And | BinaryOp::Or => {
                        let into = self.fresh_scalar(DType::Bool);
                        (CoreKernelOp::Binary { into, op: *op, left, right, dtype: DType::Bool }, into)
                    }
                    BinaryOp::Shl | BinaryOp::Shr => {
                        let into = self.fresh_scalar(left_dtype);
                        (CoreKernelOp::Binary { into, op: *op, left, right, dtype: left_dtype }, into)
                    }
                    BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::BitAnd
                    | BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Div
                    | BinaryOp::Rem => {
                        let into = self.fresh_scalar(result);
                        (CoreKernelOp::Binary { into, op: *op, left, right, dtype: result }, into)
                    }
                }
            }
            PrimitiveId::Cast(to) => {
                arity(self, 1)?;
                let (operand, from) = operands[0];
                let into = self.fresh_scalar(*to);
                (CoreKernelOp::Cast { into, operand, from, to: *to }, into)
            }
            PrimitiveId::Math(op) => {
                arity(self, op.arity())?;
                let into = self.fresh_scalar(result);
                (
                    CoreKernelOp::Math {
                        into,
                        op: *op,
                        operands: operands.iter().map(|(value, _)| *value).collect(),
                        dtype: result,
                    },
                    into,
                )
            }
            PrimitiveId::Select => {
                arity(self, 3)?;
                let (condition, condition_dtype) = operands[0];
                if condition_dtype != DType::Bool {
                    return Err(self.defect(Package::L1, format!("select at {node:?} has a {} condition", condition_dtype.name())));
                }
                let into = self.fresh_scalar(result);
                (
                    CoreKernelOp::Select {
                        into,
                        condition,
                        then_value: operands[1].0,
                        else_value: operands[2].0,
                        dtype: result,
                    },
                    into,
                )
            }
            other => return Err(self.defect(Package::K1, format!("`{other}` is not an elementwise primitive"))),
        };
        Ok((KernelOp::Core(op.0), op.1))
    }

    fn lower_elementwise(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        match output.kind() {
            GraphValueKind::Scalar(_) | GraphValueKind::Index { .. } => {
                let result = self.scalar_dtype_of(output.kind())?;
                let mut resolved = BTreeMap::new();
                let mut operands = Vec::with_capacity(logical.inputs.len());
                for value in &logical.inputs {
                    let entry = self.scalar_typed(self.canonical(node.graph, *value)?)?;
                    resolved.insert(*value, entry);
                    operands.push(entry);
                }
                let (op, into) = self.elementwise_op(node, id, &operands, result)?;
                let guards = self.guards_for(node, &resolved)?;
                let ops = self.wrap(vec![op], &guards);
                em.push_ops(ops);
                self.define_scalar(canonical, KernelValueRef::Ssa(into), em)
            }
            GraphValueKind::Tensor {
                ty,
                source: TensorSource::Computed,
            } => {
                let result = self.dense_dtype(ty)?;
                let (reads, writes) = self.node_storages(node, logical)?;
                self.ensure_open(em, &ty.axes, &reads, &writes)?;
                let coords = self.open_iter(em)?.coords.clone();
                let mut body = Vec::new();
                let mut resolved = BTreeMap::new();
                let mut operands = Vec::with_capacity(logical.inputs.len());
                for value in &logical.inputs {
                    let operand = self.canonical(node.graph, *value)?;
                    let entry = match self.lookup(operand)? {
                        Binding::Scalar(_) => self.scalar_typed(operand)?,
                        Binding::Tensor(_) => {
                            let binding = self.resolve_tensor(operand, em)?;
                            self.element_at(&binding, &coords, &mut body)?
                        }
                        Binding::Range { .. } | Binding::Tuple(_) => {
                            return Err(self.defect(Package::L1, format!("{node:?} has a range or tuple operand")))
                        }
                    };
                    resolved.insert(*value, entry);
                    operands.push(entry);
                }
                let (op, into) = self.elementwise_op(node, id, &operands, result)?;
                let guards = self.guards_for(node, &resolved)?;
                body.extend(self.wrap(vec![op], &guards));
                let value = KernelValueRef::Ssa(into);
                self.computed_result(canonical, ty, value, &coords, &mut body)?;
                self.expose_element(em, canonical, value, ty)?;
                self.push_body(em, body)?;
                Ok(())
            }
            other => Err(self.defect(Package::L1, format!("`{id}` at {node:?} produces {other:?}"))),
        }
    }

    fn lower_element_read(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let PrimitiveId::ElementRead { arity } = id else {
            return Err(self.defect(Package::K1, format!("`{id}` is not an element read")));
        };
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let source = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let mut resolved = BTreeMap::new();
        let mut indices = Vec::with_capacity(*arity);
        for index in 1..=*arity {
            let value = logical.inputs.get(index).copied().ok_or_else(|| {
                self.defect(Package::L1, format!("{node:?} has no index operand {index}"))
            })?;
            let entry = self.scalar_typed(self.canonical(node.graph, value)?)?;
            resolved.insert(value, entry);
            indices.push(entry.0);
        }
        match output.kind() {
            GraphValueKind::Scalar(dtype) => {
                if let TensorBinding::Element { .. } = source {
                    return Err(self.rule_defect(format!(
                        "{node:?} reads one element of a fused computed tensor at another domain; the rule must spill the tensor"
                    )));
                }
                let mut ops = Vec::new();
                let (value, read_dtype) = self.element_at(&source, &indices, &mut ops)?;
                if read_dtype != *dtype {
                    return Err(self.defect(Package::L1, format!("{node:?} reads {} but produces {}", read_dtype.name(), dtype.name())));
                }
                let guards = self.guards_for(node, &resolved)?;
                let ops = self.wrap(ops, &guards);
                em.push_ops(ops);
                self.define_scalar(canonical, value, em)
            }
            GraphValueKind::Tensor {
                ty,
                source: TensorSource::Computed,
            } => {
                let result = self.dense_dtype(ty)?;
                let (reads, writes) = self.node_storages(node, logical)?;
                self.ensure_open(em, &ty.axes, &reads, &writes)?;
                let coords = self.open_iter(em)?.coords.clone();
                let mut full = indices;
                full.extend(coords.iter().copied());
                let mut body = Vec::new();
                let (value, read_dtype) = self.element_at(&source, &full, &mut body)?;
                let value = self.cast_if_needed(value, read_dtype, result, &mut body);
                let guards = self.guards_for(node, &resolved)?;
                let mut body = self.wrap(body, &guards);
                self.computed_result(canonical, ty, value, &coords, &mut body)?;
                self.expose_element(em, canonical, value, ty)?;
                self.push_body(em, body)?;
                Ok(())
            }
            other => Err(self.defect(Package::L1, format!("{node:?} produces {other:?}"))),
        }
    }

    fn lower_element_write(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let PrimitiveId::ElementWrite { arity } = id else {
            return Err(self.defect(Package::K1, format!("`{id}` is not an element write")));
        };
        let destination = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let destination_dtype = self.dense_dtype(destination.ty())?;
        let mut resolved = BTreeMap::new();
        let mut indices = Vec::with_capacity(*arity);
        for index in 1..=*arity {
            let value = logical.inputs.get(index).copied().ok_or_else(|| {
                self.defect(Package::L1, format!("{node:?} has no index operand {index}"))
            })?;
            let entry = self.scalar_typed(self.canonical(node.graph, value)?)?;
            resolved.insert(value, entry);
            indices.push(entry.0);
        }
        let value_id = *logical.inputs.last().ok_or_else(|| self.defect(Package::L1, format!("{node:?} has no value operand")))?;
        if logical.inputs.len() != arity + 2 {
            return Err(self.defect(Package::L1, format!("{node:?} has {} operands for arity {arity}", logical.inputs.len())));
        }
        let value_canonical = self.canonical(node.graph, value_id)?;
        match self.facts.value_kind(value_canonical) {
            GraphValueKind::Scalar(_) | GraphValueKind::Index { .. } => {
                let (value, dtype) = self.scalar_typed(value_canonical)?;
                resolved.insert(value_id, (value, dtype));
                let mut ops = Vec::new();
                let value = self.cast_if_needed(value, dtype, destination_dtype, &mut ops);
                self.store_at(&destination, &indices, value, destination_dtype, &mut ops)?;
                let guards = self.guards_for(node, &resolved)?;
                let ops = self.wrap(ops, &guards);
                em.push_ops(ops);
                Ok(())
            }
            GraphValueKind::Tensor { ty, .. } => {
                let domain = ty.axes.clone();
                let (reads, writes) = self.node_storages(node, logical)?;
                self.ensure_open(em, &domain, &reads, &writes)?;
                let coords = self.open_iter(em)?.coords.clone();
                let source = self.resolve_tensor(value_canonical, em)?;
                let mut body = Vec::new();
                let (value, dtype) = self.element_at(&source, &coords, &mut body)?;
                let value = self.cast_if_needed(value, dtype, destination_dtype, &mut body);
                let mut full = indices;
                full.extend(coords.iter().copied());
                self.store_at(&destination, &full, value, destination_dtype, &mut body)?;
                let guards = self.guards_for(node, &resolved)?;
                let body = self.wrap(body, &guards);
                self.push_body(em, body)?;
                Ok(())
            }
            other => Err(self.defect(Package::L1, format!("{node:?} writes a {other:?} value"))),
        }
    }

    fn lower_copy_into(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let destination = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let source_canonical = self.input(node, logical, 1)?;
        let GraphValueKind::Tensor { ty, .. } = self.facts.value_kind(source_canonical).clone() else {
            return Err(self.defect(Package::L1, format!("{node:?} copies a non-tensor")));
        };
        // Shape agreement of a copy is decided upstream: the checker's
        // CopyInto signature and normalize's assignment typing (the slice
        // destination's extent is minted from its bounds). K1 lowers the
        // copy over the source's domain through the destination's view
        // chain; it does not re-check logical typing structurally.
        let guards = self.guards_for(node, &BTreeMap::new())?;
        match &ty.elem {
            Elem::Repr(_) => {
                em.flush();
                let source = self.resolve_tensor(source_canonical, em)?;
                let mut ops = Vec::new();
                self.copy_planes(&source, &destination, &ty, &mut ops)?;
                let ops = self.wrap(ops, &guards);
                em.push_ops(ops);
                Ok(())
            }
            Elem::Dtype(_) | Elem::Param(_) => {
                let destination_dtype = self.dense_dtype(destination.ty())?;
                let (reads, writes) = self.node_storages(node, logical)?;
                self.ensure_open(em, &ty.axes, &reads, &writes)?;
                let coords = self.open_iter(em)?.coords.clone();
                let source = self.resolve_tensor(source_canonical, em)?;
                let mut body = Vec::new();
                let (value, dtype) = self.element_at(&source, &coords, &mut body)?;
                let value = self.cast_if_needed(value, dtype, destination_dtype, &mut body);
                self.store_at(&destination, &coords, value, destination_dtype, &mut body)?;
                let body = self.wrap(body, &guards);
                self.push_body(em, body)?;
                Ok(())
            }
        }
    }

    fn lower_fill(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let PrimitiveId::Fill { value, dtype } = id else {
            return Err(self.defect(Package::K1, format!("`{id}` is not a fill")));
        };
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let GraphValueKind::Tensor { ty, .. } = output.kind() else {
            return Err(self.defect(Package::L1, format!("{node:?} fills a non-tensor")));
        };
        let element = self.dense_dtype(ty)?;
        if element != *dtype {
            return Err(self.defect(Package::L1, format!("{node:?} fills {} storage with {}", element.name(), dtype.name())));
        }
        let constant = match dtype {
            DType::F32 | DType::BF16 | DType::F16 => ConstantValue::Float { bits: value.to_bits() },
            DType::I32 | DType::U32 => {
                if value.fract() != 0.0 {
                    return Err(self.defect(Package::L1, format!("{node:?} fills an integer tensor with {value}")));
                }
                if value.abs() > 9_007_199_254_740_992.0 {
                    return Err(self.defect(
                        Package::L1,
                        format!("{node:?} fills an integer tensor with {value}, outside the exactly representable integer range of f64"),
                    ));
                }
                // Exact by the range proof above.
                ConstantValue::Int(*value as i64)
            }
            DType::Bool => ConstantValue::Bool(*value != 0.0),
        };
        let place = self.storage_place(canonical, node)?;
        let destinations = self.destinations(canonical)?;
        let (reads, writes) = self.node_storages(node, logical)?;
        self.ensure_open(em, &ty.axes, &reads, &writes)?;
        let mut prelude = Vec::new();
        let value = self.const_op(constant, *dtype, &mut prelude);
        let coords = self.open_iter(em)?.coords.clone();
        let mut body = Vec::new();
        for destination in destinations {
            body.push(KernelOp::Core(CoreKernelOp::Store {
                place: destination,
                coords: coords.clone(),
                value,
                dtype: *dtype,
            }));
            self.note_store_site(destination);
        }
        let guards = self.guards_for(node, &BTreeMap::new())?;
        let body = self.wrap(body, &guards);
        self.open_iter(em)?.prelude.extend(prelude);
        self.push_body(em, body)?;
        self.expose_element(em, canonical, value, ty)?;
        self.bind(
            canonical,
            Binding::Tensor(TensorBinding::Place {
                place,
                ty: ty.clone(),
            }),
        );
        Ok(())
    }

    fn lower_bulk_copy(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let source_canonical = self.input(node, logical, 0)?;
        let GraphValueKind::Tensor { ty, source: kind } = output.kind().clone() else {
            return Err(self.defect(Package::L1, format!("{node:?} copies into a non-tensor")));
        };
        let (reads, writes) = self.node_storages(node, logical)?;
        match (id, kind) {
            (PrimitiveId::Materialize | PrimitiveId::Clone, TensorSource::View(_)) => {
                let place = self.storage_place(canonical, node)?;
                let destinations = self.destinations(canonical)?;
                match &ty.elem {
                    Elem::Repr(_) => {
                        em.flush();
                        let source = self.resolve_tensor(source_canonical, em)?;
                        let mut ops = Vec::new();
                        for destination in destinations {
                            let target = TensorBinding::Place {
                                place: destination,
                                ty: ty.clone(),
                            };
                            self.copy_planes(&source, &target, &ty, &mut ops)?;
                        }
                        em.push_ops(ops);
                    }
                    Elem::Dtype(_) | Elem::Param(_) => {
                        let dtype = self.dense_dtype(&ty)?;
                        self.ensure_open(em, &ty.axes, &reads, &writes)?;
                        let coords = self.open_iter(em)?.coords.clone();
                        let source = self.resolve_tensor(source_canonical, em)?;
                        let mut body = Vec::new();
                        let (value, from) = self.element_at(&source, &coords, &mut body)?;
                        let value = self.cast_if_needed(value, from, dtype, &mut body);
                        for destination in destinations {
                            body.push(KernelOp::Core(CoreKernelOp::Store {
                                place: destination,
                                coords: coords.clone(),
                                value,
                                dtype,
                            }));
                            self.note_store_site(destination);
                        }
                        self.push_body(em, body)?;
                        self.expose_element(em, canonical, value, &ty)?;
                    }
                }
                self.bind(canonical, Binding::Tensor(TensorBinding::Place { place, ty }));
                Ok(())
            }
            (PrimitiveId::Load, TensorSource::Computed) => {
                let source = self.resolve_tensor(source_canonical, em)?;
                match &ty.elem {
                    Elem::Repr(_) => {
                        let snapshot = TensorBinding::Snapshot {
                            source: Box::new(source),
                            ty: ty.clone(),
                            value: canonical,
                        };
                        self.alias_reads.insert(canonical, reads.clone());
                        let destinations = self.destinations(canonical)?;
                        if !destinations.is_empty() {
                            em.flush();
                            let mut ops = Vec::new();
                            for destination in &destinations {
                                let target = TensorBinding::Place {
                                    place: *destination,
                                    ty: ty.clone(),
                                };
                                self.copy_planes(&snapshot, &target, &ty, &mut ops)?;
                            }
                            em.push_ops(ops);
                        }
                        let binding = match destinations.iter().find(|place| matches!(place, KernelPlaceRef::Local(_))) {
                            Some(place) => TensorBinding::Place { place: *place, ty },
                            None => snapshot,
                        };
                        self.bind(canonical, Binding::Tensor(binding));
                        Ok(())
                    }
                    Elem::Dtype(_) | Elem::Param(_) => {
                        let dtype = self.dense_dtype(&ty)?;
                        self.ensure_open(em, &ty.axes, &reads, &writes)?;
                        let coords = self.open_iter(em)?.coords.clone();
                        let mut body = Vec::new();
                        let (value, from) = self.element_at(&source, &coords, &mut body)?;
                        let value = self.cast_if_needed(value, from, dtype, &mut body);
                        self.computed_result(canonical, &ty, value, &coords, &mut body)?;
                        self.expose_element(em, canonical, value, &ty)?;
                        self.push_body(em, body)?;
                        Ok(())
                    }
                }
            }
            (PrimitiveId::Materialize | PrimitiveId::Clone | PrimitiveId::Load, kind) => Err(self.defect(
                Package::L1,
                format!("`{id}` at {node:?} produces a tensor with source {kind:?}"),
            )),
            (other, _) => Err(self.defect(Package::K1, format!("`{other}` is not a bulk copy"))),
        }
    }

    fn lower_decode(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let GraphValueKind::Tensor {
            ty,
            source: TensorSource::Computed,
        } = output.kind()
        else {
            return Err(self.defect(Package::L1, format!("{node:?} decodes into a non-computed tensor")));
        };
        let result = self.dense_dtype(ty)?;
        let (reads, writes) = self.node_storages(node, logical)?;
        self.ensure_open(em, &ty.axes, &reads, &writes)?;
        let coords = self.open_iter(em)?.coords.clone();
        let source = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let mut body = Vec::new();
        let (value, from) = self.element_at(&source, &coords, &mut body)?;
        let value = self.cast_if_needed(value, from, result, &mut body);
        self.computed_result(canonical, ty, value, &coords, &mut body)?;
        self.expose_element(em, canonical, value, ty)?;
        self.push_body(em, body)?;
        Ok(())
    }

    fn lower_plane_read(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let PrimitiveId::PackedRead(field) = id else {
            return Err(self.defect(Package::K1, format!("`{id}` is not a plane read")));
        };
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let GraphValueKind::Tensor { ty, .. } = output.kind() else {
            return Err(self.defect(Package::L1, format!("{node:?} reads a plane into a non-tensor")));
        };
        let source_canonical = self.input(node, logical, 0)?;
        let source = self.resolve_tensor(source_canonical, em)?;
        let (repr, _) = self.packed_repr(source.ty())?;
        let schema = self.plane_schema(&repr, *field)?;
        if self.dense_dtype(ty)? != schema.storage_dtype {
            return Err(self.defect(
                Package::L1,
                format!("{node:?} reads `{repr}` plane `{}` as {} but its storage dtype is {}", field.name(), ty.elem, schema.storage_dtype.name()),
            ));
        }
        let (reads, _) = self.node_storages(node, logical)?;
        self.alias_reads.insert(canonical, reads);
        let plane = TensorBinding::Plane {
            source: Box::new(source),
            repr,
            field: *field,
            ty: ty.clone(),
            value: canonical,
        };
        let destinations = self.destinations(canonical)?;
        for destination in destinations {
            self.copy_standalone(&plane, destination, ty, em)?;
        }
        self.bind(canonical, Binding::Tensor(plane));
        Ok(())
    }

    fn lower_atomic(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &PrimitiveId,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let PrimitiveId::Atomic { op, arity } = id else {
            return Err(self.defect(Package::K1, format!("`{id}` is not an atomic update")));
        };
        let place = self.resolve_tensor(self.input(node, logical, 0)?, em)?;
        let place_dtype = self.dense_dtype(place.ty())?;
        let mut resolved = BTreeMap::new();
        let mut indices = Vec::with_capacity(*arity);
        for index in 1..=*arity {
            let value = logical.inputs.get(index).copied().ok_or_else(|| {
                self.defect(Package::L1, format!("{node:?} has no index operand {index}"))
            })?;
            let entry = self.scalar_typed(self.canonical(node.graph, value)?)?;
            resolved.insert(value, entry);
            indices.push(entry.0);
        }
        let value_id = logical.inputs.get(arity + 1).copied().ok_or_else(|| {
            self.defect(Package::L1, format!("{node:?} has no value operand"))
        })?;
        let (value, dtype) = self.scalar_typed(self.canonical(node.graph, value_id)?)?;
        resolved.insert(value_id, (value, dtype));
        let mode = match &self.cut.participants.policy {
            ParticipantPolicy::Serial => AtomicMode::Serialized,
            ParticipantPolicy::Linear { .. }
            | ParticipantPolicy::Cooperative { .. }
            | ParticipantPolicy::DynamicPull { .. }
            | ParticipantPolicy::GridCooperative { .. } => AtomicMode::Device,
        };
        match (mode, place_dtype) {
            (AtomicMode::Serialized, _) | (AtomicMode::Device, DType::F32 | DType::I32 | DType::U32) => {}
            (AtomicMode::Device, DType::BF16 | DType::F16 | DType::Bool) => {
                return Err(self.rule_defect(format!(
                    "{node:?} updates a {} element atomically under a concurrent participant policy; device atomics admit f32/i32/u32 only",
                    place_dtype.name()
                )))
            }
        }
        let mut ops = Vec::new();
        let value = self.cast_if_needed(value, dtype, place_dtype, &mut ops);
        let mut guards = Vec::new();
        let (base, coords, _) = self.resolve_place(&place, &indices, &mut ops, &mut guards)?;
        ops.push(KernelOp::Core(CoreKernelOp::Atomic {
            place: base,
            coords,
            op: *op,
            value,
            dtype: place_dtype,
            mode,
        }));
        let ops = self.wrap(ops, &guards);
        let node_guards = self.guards_for(node, &resolved)?;
        let ops = self.wrap(ops, &node_guards);
        em.push_ops(ops);
        Ok(())
    }

    // -- intrinsics -------------------------------------------------------------

    /// The whole-tensor operand form of a binding: its base place and the
    /// in-block view chain over it.
    fn tensor_operand(&self, binding: &TensorBinding) -> Result<IntrinsicOperand, CompilerDefect> {
        let mut steps = Vec::new();
        let mut current = binding;
        loop {
            match current {
                TensorBinding::Place { place, .. } => {
                    steps.reverse();
                    return Ok(IntrinsicOperand::Tensor {
                        place: *place,
                        ty: binding.ty().clone(),
                        view: KernelViewChain { steps },
                    });
                }
                TensorBinding::View { source, step, .. } => {
                    steps.push(step.clone());
                    current = source;
                }
                TensorBinding::Snapshot { source, .. } => current = source,
                TensorBinding::Plane { value, .. } => {
                    return Err(self.defect(
                        Package::L1,
                        format!("plane accessor value {} cannot be a capability operand", value.0),
                    ))
                }
                TensorBinding::Element { .. } => {
                    return Err(self.rule_defect(
                        "a fused computed tensor is passed to a capability intrinsic; the rule must spill it",
                    ))
                }
            }
        }
    }

    fn lower_intrinsic_use(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        id: &IntrinsicId,
        operand_ids: &[GraphValueId],
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let mut arguments = Vec::with_capacity(operand_ids.len());
        let mut operands = Vec::with_capacity(operand_ids.len());
        for value in operand_ids {
            let canonical = self.canonical(node.graph, *value)?;
            arguments.push(self.facts.value_type(canonical));
            operands.push(match self.lookup(canonical)? {
                Binding::Scalar(value) => IntrinsicOperand::Value(value),
                Binding::Tensor(_) => {
                    let binding = self.resolve_tensor(canonical, em)?;
                    self.tensor_operand(&binding)?
                }
                Binding::Range { .. } | Binding::Tuple(_) => {
                    return Err(self.defect(
                        Package::L1,
                        format!("`{id}` at {node:?} takes a range or tuple operand"),
                    ))
                }
            });
        }
        let output = match logical.outputs.as_slice() {
            [] => None,
            [output] => Some(output),
            outputs => {
                return Err(self.defect(Package::L1, format!("`{id}` at {node:?} has {} outputs", outputs.len())))
            }
        };
        // A use with no result node produces void: the registry result type
        // of a zero-output signature is void, so `None` states that fact.
        let result_ty = match output {
            Some(output) => output.ty(),
            None => ValueType::Void,
        };
        let (result, binding) = match output {
            None => (IntrinsicResult::Void, None),
            Some(output) => {
                let canonical = self.canonical(node.graph, output.id())?;
                match output.kind() {
                    GraphValueKind::Tensor { ty, .. } => {
                        let destinations = self.destinations(canonical)?;
                        let place = destinations
                            .iter()
                            .find(|place| matches!(place, KernelPlaceRef::Local(_)))
                            .copied()
                            .or_else(|| destinations.first().copied())
                            .ok_or_else(|| {
                                self.rule_defect(format!(
                                    "`{id}` at {node:?} produces tensor value {} with no residence; a capability tensor result needs a kernel-local or output residence",
                                    canonical.0
                                ))
                            })?;
                        self.note_store_site(place);
                        (
                            IntrinsicResult::Tensor { place, ty: ty.clone() },
                            Some((canonical, Binding::Tensor(TensorBinding::Place { place, ty: ty.clone() }))),
                        )
                    }
                    GraphValueKind::Scalar(_) | GraphValueKind::Index { .. } | GraphValueKind::Capability(_) => {
                        let ty = self.kernel_type_of(canonical)?;
                        let ssa = self.fresh(ty.clone());
                        (
                            IntrinsicResult::Ssa { id: ssa, ty },
                            Some((canonical, Binding::Scalar(KernelValueRef::Ssa(ssa)))),
                        )
                    }
                    other => {
                        return Err(self.defect(Package::L1, format!("`{id}` at {node:?} produces {other:?}")))
                    }
                }
            }
        };
        let intrinsic = IntrinsicUse {
            id: id.clone(),
            arguments,
            result: result_ty,
        };
        let op = self.catalog.lower(&intrinsic, &operands, result);
        let guards = self.guards_for(node, &BTreeMap::new())?;
        let ops = self.wrap(vec![KernelOp::Intrinsic(op)], &guards);
        em.push_ops(ops);
        match binding {
            Some((canonical, Binding::Scalar(value))) => self.define_scalar(canonical, value, em),
            Some((canonical, binding)) => {
                self.bind(canonical, binding);
                Ok(())
            }
            None => Ok(()),
        }
    }

    // -- reductions ------------------------------------------------------------

    /// Emit the registry accumulation of one element into the carries of the
    /// enclosing reduced-axis `Repeat`; returns the carry ops (to head the
    /// loop body) and the result SSA.
    fn accumulate(
        &mut self,
        op: ReduceOp,
        schema: &ReduceSchema,
        input: DType,
        element: KernelValueRef,
        index: KernelValueRef,
        prelude: &mut Ops<D>,
        body: &mut Ops<D>,
    ) -> Result<(Ops<D>, KernelSsaId), CompilerDefect> {
        let mut carries = Vec::new();
        let scalar = |dtype: DType| KernelValueType::Scalar(dtype);
        match op {
            ReduceOp::Sum => {
                let acc = schema.accumulator;
                let initial = self.zero_of(acc, prelude);
                let current = self.fresh_scalar(acc);
                let result = self.fresh_scalar(acc);
                let element = self.cast_if_needed(element, input, acc, body);
                let update = self.binary(BinaryOp::Add, KernelValueRef::Ssa(current), element, acc, body);
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial,
                    current,
                    update,
                    result,
                    ty: scalar(acc),
                }));
                Ok((carries, result))
            }
            ReduceOp::Max | ReduceOp::Min => {
                let acc = schema.accumulator;
                let initial = self.zero_of(acc, prelude);
                let first_initial = self.const_op(ConstantValue::Bool(true), DType::Bool, prelude);
                let first_update = self.const_op(ConstantValue::Bool(false), DType::Bool, prelude);
                let current = self.fresh_scalar(acc);
                let result = self.fresh_scalar(acc);
                let first = self.fresh_scalar(DType::Bool);
                let first_result = self.fresh_scalar(DType::Bool);
                let element = self.cast_if_needed(element, input, acc, body);
                let combined = self.fresh_scalar(acc);
                body.push(KernelOp::Core(CoreKernelOp::Math {
                    into: combined,
                    op: if op == ReduceOp::Max { MathOp::Max } else { MathOp::Min },
                    operands: vec![KernelValueRef::Ssa(current), element],
                    dtype: acc,
                }));
                let update = self.fresh_scalar(acc);
                body.push(KernelOp::Core(CoreKernelOp::Select {
                    into: update,
                    condition: KernelValueRef::Ssa(first),
                    then_value: element,
                    else_value: KernelValueRef::Ssa(combined),
                    dtype: acc,
                }));
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial,
                    current,
                    update: KernelValueRef::Ssa(update),
                    result,
                    ty: scalar(acc),
                }));
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial: first_initial,
                    current: first,
                    update: first_update,
                    result: first_result,
                    ty: scalar(DType::Bool),
                }));
                Ok((carries, result))
            }
            ReduceOp::Argmax => {
                let best_dtype = schema.accumulator;
                let best_initial = self.zero_of(best_dtype, prelude);
                let index_initial = self.const_i32(0, prelude);
                let first_initial = self.const_op(ConstantValue::Bool(true), DType::Bool, prelude);
                let first_update = self.const_op(ConstantValue::Bool(false), DType::Bool, prelude);
                let best = self.fresh_scalar(best_dtype);
                let best_result = self.fresh_scalar(best_dtype);
                let best_index = self.fresh_scalar(DType::I32);
                let index_result = self.fresh_scalar(DType::I32);
                let first = self.fresh_scalar(DType::Bool);
                let first_result = self.fresh_scalar(DType::Bool);
                let element = self.cast_if_needed(element, input, best_dtype, body);
                let greater = self.fresh_scalar(DType::Bool);
                body.push(KernelOp::Core(CoreKernelOp::Compare {
                    into: greater,
                    op: RelOp::Gt,
                    left: element,
                    right: KernelValueRef::Ssa(best),
                    dtype: best_dtype,
                }));
                let take = self.binary(BinaryOp::Or, KernelValueRef::Ssa(first), KernelValueRef::Ssa(greater), DType::Bool, body);
                let best_update = self.fresh_scalar(best_dtype);
                body.push(KernelOp::Core(CoreKernelOp::Select {
                    into: best_update,
                    condition: take,
                    then_value: element,
                    else_value: KernelValueRef::Ssa(best),
                    dtype: best_dtype,
                }));
                let index_update = self.fresh_scalar(DType::I32);
                body.push(KernelOp::Core(CoreKernelOp::Select {
                    into: index_update,
                    condition: take,
                    then_value: index,
                    else_value: KernelValueRef::Ssa(best_index),
                    dtype: DType::I32,
                }));
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial: best_initial,
                    current: best,
                    update: KernelValueRef::Ssa(best_update),
                    result: best_result,
                    ty: scalar(best_dtype),
                }));
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial: index_initial,
                    current: best_index,
                    update: KernelValueRef::Ssa(index_update),
                    result: index_result,
                    ty: scalar(DType::I32),
                }));
                carries.push(KernelOp::Core(CoreKernelOp::Carry {
                    initial: first_initial,
                    current: first,
                    update: first_update,
                    result: first_result,
                    ty: scalar(DType::Bool),
                }));
                Ok((carries, index_result))
            }
        }
    }

    fn lower_reduction(
        &mut self,
        node: &OwnedNodeRef,
        logical: &LogicalNode,
        reduction: &ReductionNode,
        em: &mut Emitter<D::Intrinsic>,
    ) -> Result<(), CompilerDefect> {
        let output = self.single_output(node, logical)?;
        let canonical = self.canonical(node.graph, output.id())?;
        let operand = self.canonical(node.graph, reduction.operand)?;
        let GraphValueKind::Tensor { ty, .. } = self.facts.value_kind(operand).clone() else {
            return Err(self.defect(Package::L1, format!("{node:?} reduces a non-tensor")));
        };
        let input = self.dense_dtype(&ty)?;
        let schema = reduce_schema(reduction.op, input);
        if reduction.accumulator != schema.accumulator && reduction.op != ReduceOp::Argmax {
            return Err(self.defect(
                Package::L1,
                format!("{node:?} declares accumulator {} but the registry schema is {}", reduction.accumulator.name(), schema.accumulator.name()),
            ));
        }
        if reduction.axis >= ty.rank() {
            return Err(self.defect(Package::L1, format!("{node:?} reduces axis {} of a rank-{} tensor", reduction.axis, ty.rank())));
        }
        let axis = u32::try_from(reduction.axis).map_err(|_| {
            self.defect(Package::L1, format!("{node:?} reduces axis {}, above the u32 axis space", reduction.axis))
        })?;
        let cooperative = match &self.cut.participants.policy {
            ParticipantPolicy::Cooperative { family, .. } => Some(family.clone()),
            ParticipantPolicy::Serial
            | ParticipantPolicy::Linear { .. }
            | ParticipantPolicy::DynamicPull { .. }
            | ParticipantPolicy::GridCooperative { .. } => None,
        };
        match (&self.cut.algorithm, cooperative) {
            (AlgorithmChoice::Intrinsic(id), _) => {
                let id = id.clone();
                return self.lower_intrinsic_use(node, logical, &id, &[reduction.operand], em);
            }
            (AlgorithmChoice::Universal, Some(family)) => {
                return self.lower_intrinsic_use(node, logical, &family, &[reduction.operand], em);
            }
            (AlgorithmChoice::Universal, None) => {}
            (AlgorithmChoice::Reduction(topology), _) => {
                if !topology.is_reference_order() {
                    return Err(self.rule_defect(format!(
                        "reduction {node:?} is proposed with the reassociating topology {topology:?}; the core lowers the reference fold only, a reassociating realization is a typed intrinsic family (`AlgorithmChoice::Intrinsic`) or a cooperative participant policy"
                    )));
                }
            }
            (AlgorithmChoice::Blocked { .. }, _) => {
                return Err(self.rule_defect(format!(
                    "reduction {node:?} is proposed with a blocked algorithm; the core has no blocked reduction form, a blocked realization is a typed intrinsic family"
                )));
            }
        }
        let result_ty = match output.kind() {
            GraphValueKind::Scalar(dtype) => {
                if *dtype != schema.result {
                    return Err(self.defect(Package::L1, format!("{node:?} produces {} but the schema result is {}", dtype.name(), schema.result.name())));
                }
                None
            }
            GraphValueKind::Tensor { ty, source: TensorSource::Computed } => Some(ty.clone()),
            other => return Err(self.defect(Package::L1, format!("{node:?} produces {other:?}"))),
        };
        let outer_axes: Vec<ExtentExpr> = ty
            .axes
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != reduction.axis)
            .map(|(_, extent)| extent.clone())
            .collect();
        if let Some(result) = &result_ty {
            if result.axes != outer_axes {
                return Err(self.defect(Package::L1, format!("{node:?} result axes {:?} are not the outer axes {outer_axes:?}", result.axes)));
            }
        }
        let binding = self.resolve_tensor(operand, em)?;
        match binding {
            TensorBinding::Element { value, .. } => {
                let open = em.open.take().ok_or_else(|| self.defect(Package::K1, "a fused element exists without an open iteration"))?;
                if open.domain != ty.axes {
                    return Err(self.defect(Package::K1, format!("fused reduction operand domain {:?} differs from the open iteration {:?}", ty.axes, open.domain)));
                }
                let free_count = open.coords.len() - open.residual.len();
                if reduction.axis < free_count {
                    return Err(self.defect(
                        Package::D1,
                        format!("{node:?} reduces axis {}, which is a block axis of the interface", reduction.axis),
                    ));
                }
                let k = reduction.axis - free_count;
                let (binder, start, end) = open.residual[k];
                let mut prelude = open.prelude;
                let mut body = open.body;
                let (carries, result) = self.accumulate(reduction.op, &schema, input, value, KernelValueRef::Ssa(binder), &mut prelude, &mut body)?;
                let mut inner = carries;
                inner.extend(body);
                let reduced = KernelOp::Core(CoreKernelOp::Repeat {
                    binder,
                    start,
                    end,
                    body: inner,
                });
                let mut residual = open.residual;
                residual.remove(k);
                let mut coords = open.coords;
                coords.remove(reduction.axis);
                match result_ty {
                    None => {
                        let mut nest = vec![reduced];
                        for (binder, start, end) in residual.into_iter().rev() {
                            nest = vec![KernelOp::Core(CoreKernelOp::Repeat { binder, start, end, body: nest })];
                        }
                        let mut ops = prelude;
                        ops.extend(nest);
                        em.push_ops(ops);
                        self.define_scalar(canonical, KernelValueRef::Ssa(result), em)
                    }
                    Some(result_tensor) => {
                        let mut outer_body = vec![reduced];
                        let value = KernelValueRef::Ssa(result);
                        self.computed_result(canonical, &result_tensor, value, &coords, &mut outer_body)?;
                        let mut elements = BTreeMap::new();
                        elements.insert(canonical, (value, result_tensor));
                        em.synchronize(&outer_body);
                        em.open = Some(OpenIteration {
                            domain: outer_axes,
                            coords,
                            residual,
                            prelude,
                            body: outer_body,
                            elements,
                            reads: open.reads,
                            writes: open.writes,
                        });
                        Ok(())
                    }
                }
            }
            TensorBinding::Place { place, .. } => {
                let (reads, writes) = self.node_storages(node, logical)?;
                match result_ty {
                    None => {
                        em.flush();
                        let into = self.fresh_scalar(schema.result);
                        em.push_ops(vec![KernelOp::Core(CoreKernelOp::Fold {
                            into,
                            op: reduction.op,
                            place,
                            axis,
                            coords: Vec::new(),
                            schema,
                            shape: ty,
                        })]);
                        self.define_scalar(canonical, KernelValueRef::Ssa(into), em)
                    }
                    Some(result_tensor) => {
                        self.ensure_open(em, &outer_axes, &reads, &writes)?;
                        let coords = self.open_iter(em)?.coords.clone();
                        let into = self.fresh_scalar(schema.result);
                        let mut body = vec![KernelOp::Core(CoreKernelOp::Fold {
                            into,
                            op: reduction.op,
                            place,
                            axis,
                            coords: coords.clone(),
                            schema,
                            shape: ty,
                        })];
                        let value = KernelValueRef::Ssa(into);
                        self.computed_result(canonical, &result_tensor, value, &coords, &mut body)?;
                        self.expose_element(em, canonical, value, &result_tensor)?;
                        self.push_body(em, body)?;
                        Ok(())
                    }
                }
            }
            binding @ (TensorBinding::View { .. } | TensorBinding::Snapshot { .. } | TensorBinding::Plane { .. }) => {
                // Explicit fold through the binding's element access.
                let (reads, writes) = self.node_storages(node, logical)?;
                let axis = reduction.axis;
                let length = ty.axes[axis].clone();
                let op = reduction.op;
                let fold = |former: &mut Self, outer: &[KernelValueRef], ops: &mut Ops<D>| -> Result<KernelSsaId, CompilerDefect> {
                    let zero = former.const_i32(0, ops);
                    let end = former.extent_ref(&length, ops)?;
                    let binder = former.fresh_scalar(DType::I32);
                    let mut coords = outer.to_vec();
                    coords.insert(axis, KernelValueRef::Ssa(binder));
                    let mut body = Vec::new();
                    let (element, from) = former.element_at(&binding, &coords, &mut body)?;
                    let mut prelude = Vec::new();
                    let (carries, result) = former.accumulate(op, &schema, from, element, KernelValueRef::Ssa(binder), &mut prelude, &mut body)?;
                    ops.extend(prelude);
                    let mut inner = carries;
                    inner.extend(body);
                    ops.push(KernelOp::Core(CoreKernelOp::Repeat { binder, start: zero, end, body: inner }));
                    Ok(result)
                };
                match result_ty {
                    None => {
                        em.flush();
                        let mut ops = Vec::new();
                        let result = fold(self, &[], &mut ops)?;
                        em.push_ops(ops);
                        self.define_scalar(canonical, KernelValueRef::Ssa(result), em)
                    }
                    Some(result_tensor) => {
                        self.ensure_open(em, &outer_axes, &reads, &writes)?;
                        let coords = self.open_iter(em)?.coords.clone();
                        let mut body = Vec::new();
                        let result = fold(self, &coords, &mut body)?;
                        let value = KernelValueRef::Ssa(result);
                        self.computed_result(canonical, &result_tensor, value, &coords, &mut body)?;
                        self.expose_element(em, canonical, value, &result_tensor)?;
                        self.push_body(em, body)?;
                        Ok(())
                    }
                }
            }
        }
    }

    // -- seal -------------------------------------------------------------------

    fn seal(self, ops: Ops<D>) -> Result<ClosedKernelBlock<D::Intrinsic>, CompilerDefect> {
        if let Some(node) = self.members.difference(&self.visited).next() {
            return Err(self.rule_defect(format!(
                "owned node {node:?} is not reached by the block's structure (top-level, nested region, or absorbed callee)"
            )));
        }
        for (index, sites) in self.guard_sites.iter().enumerate() {
            if *sites == 0 {
                return Err(self.defect(
                    Package::K1,
                    format!("kernel guard {:?} has no `Check` site", self.cut.guards[index].0),
                ));
            }
        }
        for (id, decl) in self.interface.outputs.entries() {
            let sites = self.output_sites[id.0 as usize];
            match &decl.route {
                ValueRoute::Scalar(_) => {
                    if sites != 1 {
                        if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
                            let value = self.facts.leaf(decl.leaf).value;
                            eprintln!(
                                "  FAILING scalar leaf {} value {} kind={:?} members={:?}",
                                decl.leaf.0,
                                value.0,
                                self.facts.value_kind(value),
                                self.facts.members(value),
                            );
                            for node in self.cut.nodes.iter() {
                                if let Ok(logical) = self.facts.node(node) {
                                    for input in &logical.inputs {
                                        if let Ok(c) = self.canonical(node.graph, *input) {
                                            if c == value {
                                                eprintln!("    used by node {:?}", node.node);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        return Err(self.defect(
                            Package::K1,
                            format!("scalar output {} (leaf {}) is published {sites} times", id.0, decl.leaf.0),
                        ));
                    }
                }
                ValueRoute::Tensor(_) => {
                    if sites == 0 {
                        if std::env::var_os("SEISMIC_DEBUG_SEAL").is_some() {
                            eprintln!(
                                "DEBUG seal block {} rule {:?}:\n  nodes={:?}\n  inputs={:?}\n  outputs={:?}\n  sites={:?}\n  leaves={:?}\n  cut inputs={:?}\n  members={:?}",
                                self.block.0,
                                self.strategy.shape().rule(),
                                self.cut.nodes,
                                self.interface.inputs.entries().map(|(i, d)| (i.0, d.leaf.0, format!("{:?}", d.route))).collect::<Vec<_>>(),
                                self.interface.outputs.entries().map(|(i, d)| (i.0, d.leaf.0, format!("{:?}", d.route))).collect::<Vec<_>>(),
                                self.output_sites,
                                self.interface.inputs.iter().map(|d| (d.leaf.0, format!("{:?}", d.route))).chain(self.interface.outputs.iter().map(|d| (d.leaf.0, format!("{:?}", d.route)))).collect::<Vec<_>>(),
                                self.cut.inputs,
                                self.members,
                            );
                            for node in self.cut.nodes.iter() {
                                eprintln!(
                                    "  MEMBER node {:?} kind={:?}",
                                    node,
                                    self.facts.node(node).unwrap().kind,
                                );
                            }
                            for (_, decl) in self.interface.inputs.entries() {
                                let record = self.facts.leaf(decl.leaf);
                                eprintln!(
                                    "  INPUT leaf {} value {} kind={:?}",
                                    decl.leaf.0,
                                    record.value.0,
                                    self.facts.value_kind(record.value),
                                );
                            }
                            for (_, decl) in self.interface.outputs.entries() {
                                let record = self.facts.leaf(decl.leaf);
                                eprintln!(
                                    "  OUTPUT leaf {} value {} members={:?} kind={:?}",
                                    decl.leaf.0,
                                    record.value.0,
                                    self.facts.members(record.value),
                                    self.facts.value_kind(record.value),
                                );
                            }
                        }
                        return Err(self.rule_defect(format!(
                            "tensor output {} (leaf {}) is never written by the block; the producing node is not a block node or produces no residence write",
                            id.0, decl.leaf.0
                        )));
                    }
                }
                route @ ValueRoute::Void => {
                    return Err(self.defect(Package::D1, format!("output {} is routed as {route:?}", id.0)))
                }
            }
        }
        let mut checker = DefUse {
            former: &self,
            defined_once: BTreeSet::new(),
        };
        let mut visible = BTreeSet::new();
        checker.sequence(&ops, &mut visible)?;
        if checker.defined_once.len() != self.ssa.len() {
            let missing: Vec<u32> = self
                .ssa
                .iter()
                .map(|decl| decl.id)
                .filter(|id| !checker.defined_once.contains(id))
                .map(|id| id.0)
                .collect();
            return Err(self.defect(
                Package::K1,
                format!("SSA values {missing:?} are declared but never defined"),
            ));
        }
        let status_fields = self
            .cut
            .guards
            .iter()
            .enumerate()
            .map(|(index, (obligation, _))| (StatusFieldTemplateId(index as u32), obligation.clone()))
            .collect();
        let ops = NonEmpty::new(ops).ok_or_else(|| {
            self.defect(
                Package::K1,
                "the block's structure emits no kernel ops; a sealed block is non-empty",
            )
        })?;
        Ok(ClosedKernelBlock::seal(
            self.interface.clone(),
            IdVec::new(self.ssa),
            ops,
            status_fields,
        ))
    }
}

/// The definition/use walk of the seal: every SSA is defined exactly once
/// before every use within its scope; every input, axis, place, output, and
/// status field names a declaration of the interface or the block.
struct DefUse<'a, 'f, 'l, D: ExecutableDialect> {
    former: &'a Former<'f, 'l, D>,
    defined_once: BTreeSet<KernelSsaId>,
}

impl<'a, 'f, 'l, D: ExecutableDialect> DefUse<'a, 'f, 'l, D> {
    fn defect(&self, invariant: impl std::fmt::Display) -> CompilerDefect {
        self.former.defect(Package::K1, invariant)
    }

    fn define(&mut self, id: KernelSsaId, visible: &mut BTreeSet<KernelSsaId>) -> Result<(), CompilerDefect> {
        if id.0 as usize >= self.former.ssa.len() {
            return Err(self.defect(format!("SSA {} is defined but not declared", id.0)));
        }
        if !self.defined_once.insert(id) {
            return Err(self.defect(format!("SSA {} is defined twice", id.0)));
        }
        visible.insert(id);
        Ok(())
    }

    fn use_value(&self, value: KernelValueRef, visible: &BTreeSet<KernelSsaId>) -> Result<(), CompilerDefect> {
        match value {
            KernelValueRef::Ssa(id) => {
                if visible.contains(&id) {
                    Ok(())
                } else {
                    Err(self.defect(format!("SSA {} is used before its definition or outside its scope", id.0)))
                }
            }
            KernelValueRef::Input(id) => match self.former.interface.inputs.get(id) {
                Some(decl) => match &decl.route {
                    ValueRoute::Scalar(_) => Ok(()),
                    route => Err(self.defect(format!("input {} is used as a scalar but routed as {route:?}", id.0))),
                },
                None => Err(self.defect(format!("input {} does not exist", id.0))),
            },
            KernelValueRef::Axis(id) => {
                if self.former.interface.axes.get(id).is_some() {
                    Ok(())
                } else {
                    Err(self.defect(format!("axis {} does not exist", id.0)))
                }
            }
        }
    }

    fn use_place(&self, place: KernelPlaceRef, written: bool) -> Result<(), CompilerDefect> {
        match place {
            KernelPlaceRef::Input(id) => match self.former.interface.inputs.get(id) {
                Some(decl) => match &decl.route {
                    ValueRoute::Tensor(route) => {
                        if written && route.access == Access::Shared {
                            Err(self.defect(format!("input {} is written through a shared view", id.0)))
                        } else {
                            Ok(())
                        }
                    }
                    route => Err(self.defect(format!("input {} is addressed but routed as {route:?}", id.0))),
                },
                None => Err(self.defect(format!("input place {} does not exist", id.0))),
            },
            KernelPlaceRef::Output(id) => match self.former.interface.outputs.get(id) {
                Some(decl) => match &decl.route {
                    ValueRoute::Tensor(_) => Ok(()),
                    route => Err(self.defect(format!("output {} is addressed but routed as {route:?}", id.0))),
                },
                None => Err(self.defect(format!("output place {} does not exist", id.0))),
            },
            KernelPlaceRef::Local(id) => {
                if self.former.interface.locals.get(id).is_some() {
                    Ok(())
                } else {
                    Err(self.defect(format!("local place {} does not exist", id.0)))
                }
            }
        }
    }

    fn sequence(&mut self, ops: &[KernelOp<D::Intrinsic>], visible: &mut BTreeSet<KernelSsaId>) -> Result<(), CompilerDefect> {
        for op in ops {
            self.op(op, visible)?;
        }
        Ok(())
    }

    fn op(&mut self, op: &KernelOp<D::Intrinsic>, visible: &mut BTreeSet<KernelSsaId>) -> Result<(), CompilerDefect> {
        match op {
            KernelOp::Intrinsic(intrinsic) => {
                let references = D::intrinsic_references(intrinsic);
                for value in references.uses {
                    self.use_value(value, visible)?;
                }
                for place in references.places {
                    self.use_place(place, true)?;
                }
                for id in references.defines {
                    self.define(id, visible)?;
                }
                Ok(())
            }
            KernelOp::Core(core) => match core {
                CoreKernelOp::Repeat { binder, start, end, body } => {
                    self.use_value(*start, visible)?;
                    self.use_value(*end, visible)?;
                    let mut inner = visible.clone();
                    self.define(*binder, &mut inner)?;
                    // Carries head the body: their initial values are outer,
                    // their current values are visible to the body, and their
                    // updates must be defined by the body's end.
                    let mut carries = Vec::new();
                    let mut rest = body.as_slice();
                    while let Some((KernelOp::Core(CoreKernelOp::Carry { initial, current, update, result, .. }), tail)) = rest.split_first() {
                        self.use_value(*initial, visible)?;
                        self.define(*current, &mut inner)?;
                        carries.push((*update, *result));
                        rest = tail;
                    }
                    self.sequence(rest, &mut inner)?;
                    for (update, _) in &carries {
                        self.use_value(*update, &inner)?;
                    }
                    for (_, result) in carries {
                        self.define(result, visible)?;
                    }
                    Ok(())
                }
                CoreKernelOp::Carry { .. } => Err(self.defect("a `Carry` appears outside the head of a `Repeat` body")),
                CoreKernelOp::Branch { condition, then_body, else_body, joins } => {
                    self.use_value(*condition, visible)?;
                    let mut then_visible = visible.clone();
                    self.sequence(then_body, &mut then_visible)?;
                    let mut else_visible = visible.clone();
                    self.sequence(else_body, &mut else_visible)?;
                    for join in joins {
                        self.use_value(join.then_value, &then_visible)?;
                        self.use_value(join.else_value, &else_visible)?;
                        self.define(join.joined, visible)?;
                    }
                    Ok(())
                }
                CoreKernelOp::Check { status, guarded, .. } => {
                    for value in core.uses() {
                        self.use_value(value, visible)?;
                    }
                    if status.0 as usize >= self.former.cut.guards.len() {
                        return Err(self.defect(format!("status field {} is not a guard of the block", status.0)));
                    }
                    self.sequence(guarded, visible)
                }
                CoreKernelOp::Publish { value, output } => {
                    self.use_value(*value, visible)?;
                    match self.former.interface.outputs.get(*output) {
                        Some(decl) => match &decl.route {
                            ValueRoute::Scalar(_) => Ok(()),
                            route => Err(self.defect(format!("output {} is published but routed as {route:?}", output.0))),
                        },
                        None => Err(self.defect(format!("output {} does not exist", output.0))),
                    }
                }
                CoreKernelOp::GridBarrier => match &self.former.cut.participants.policy {
                    ParticipantPolicy::GridCooperative { .. } => Ok(()),
                    policy => Err(self.defect(format!(
                        "a `GridBarrier` appears in a block with participant policy {policy:?}; grid barriers are legal in grid-cooperative blocks only"
                    ))),
                },
                CoreKernelOp::Atomic { mode, dtype, .. } => {
                    match (mode, dtype) {
                        (AtomicMode::Serialized, _) | (AtomicMode::Device, DType::F32 | DType::I32 | DType::U32) => {}
                        (AtomicMode::Device, DType::BF16 | DType::F16 | DType::Bool) => {
                            return Err(self.defect(format!("a device atomic updates a {} element", dtype.name())))
                        }
                    }
                    for value in core.uses() {
                        self.use_value(value, visible)?;
                    }
                    for place in core.places() {
                        self.use_place(place, true)?;
                    }
                    Ok(())
                }
                CoreKernelOp::Const { .. }
                | CoreKernelOp::RuntimeExtent { .. }
                | CoreKernelOp::Unary { .. }
                | CoreKernelOp::Binary { .. }
                | CoreKernelOp::Compare { .. }
                | CoreKernelOp::Math { .. }
                | CoreKernelOp::Fma { .. }
                | CoreKernelOp::Cast { .. }
                | CoreKernelOp::Select { .. }
                | CoreKernelOp::TableLookup { .. }
                | CoreKernelOp::Load { .. }
                | CoreKernelOp::Store { .. }
                | CoreKernelOp::PackedPlaneRead { .. }
                | CoreKernelOp::PlaneLoad { .. }
                | CoreKernelOp::PlaneStore { .. }
                | CoreKernelOp::Fold { .. }
                | CoreKernelOp::Barrier { .. } => {
                    for value in core.uses() {
                        self.use_value(value, visible)?;
                    }
                    let written = matches!(
                        core,
                        CoreKernelOp::Store { .. } | CoreKernelOp::PlaneStore { .. } | CoreKernelOp::Atomic { .. }
                    );
                    for place in core.places() {
                        self.use_place(place, written)?;
                    }
                    for id in core.defines() {
                        self.define(id, visible)?;
                    }
                    Ok(())
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{AxisMapping, LinearIterationMap};
    use crate::ids::{OccurrenceId, PlanParamId, ResidenceId};
    use crate::kernel::{IntrinsicConsequences, IntrinsicReferences, KernelFormer};
    use crate::occurrence::OccurrenceForest;
    use crate::residence::{
        ClosedKernelInterface, KernelAxisDecl, KernelInputDecl, KernelOutputDecl, Lifetime,
        PlanePlan, PullCounter, Replication, Residence, ResidenceChoice, ResidenceGraph,
        ResidenceSource, RoutedBlock, StoragePlane, StorageScope,
    };
    use crate::routes::{RouteTable, ScalarRoute, TensorRoute, ViewTransformTemplate};
    use crate::strategy::{ClosedStrategyShape, MappingProposal, RuleQuery};
    use crate::target::{EffectiveTargetProfile, TargetLimits};
    use seismic_lang::logical::boundary::BoundaryLeaf;
    use seismic_lang::logical::specialization::{ShapeBinding, SpecializationDomain};
    use seismic_lang::logical::{construct, EffectiveTargetIdentity, LogicalProgram};
    use seismic_lang::program::{compile, SourceFile};
    use seismic_lang::sir::Program;
    use seismic_lang::sym::Sym;
    use seismic_lang::types::NonEmpty;

    // -- a tiny dialect with no intrinsics ------------------------------------

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum NoIntrinsic {}

    struct TestDialect;

    impl crate::kernel::sealed::Sealed for TestDialect {}

    impl ExecutableDialect for TestDialect {
        type Intrinsic = NoIntrinsic;
        type LayoutTemplate = ();
        type ResolvedLayout = ();

        fn intrinsic_references(op: &NoIntrinsic) -> IntrinsicReferences {
            match *op {}
        }
        fn intrinsic_consequences(op: &NoIntrinsic) -> IntrinsicConsequences {
            match *op {}
        }
        fn public_layout(_: &TensorType, _: crate::kernel::PlaneRef) {}
        fn internal_layout(_: &TensorType, _: crate::kernel::PlaneRef) {}
        fn resolve_layout(_: &(), _: &crate::plan_space::SolvedValues) {}
    }

    struct NoCatalog;

    impl IntrinsicCatalog<TestDialect> for NoCatalog {
        fn lower(&self, intrinsic: &IntrinsicUse, _: &[IntrinsicOperand], _: IntrinsicResult) -> NoIntrinsic {
            panic!(
                "compiler defect (B1): the test target authorizes no capability, yet `{}` was lowered",
                intrinsic.id
            )
        }
    }

    type Op = KernelOp<NoIntrinsic>;

    // -- program construction (mirrors the S1 tests) ---------------------------

    fn check(source: &str) -> Program {
        let files = vec![SourceFile {
            path: "test.seismic".to_string(),
            text: source.to_string(),
        }];
        compile(&files).unwrap_or_else(|diagnostics| {
            panic!(
                "the source checks: {}",
                diagnostics
                    .iter()
                    .map(|d| d.render())
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
    }

    fn supports_all(_: &IntrinsicUse) -> Result<(), String> {
        Ok(())
    }

    fn logical(source: &str, entry: &str, shapes: &[(&str, u64)]) -> LogicalProgram {
        let program = check(source);
        let shapes = shapes
            .iter()
            .map(|(name, value)| (name.to_string(), ShapeBinding::Exact(*value)))
            .collect();
        let domain = SpecializationDomain::new(&program, entry, shapes, BTreeMap::new())
            .expect("the exact domain binds every entry parameter");
        let target = EffectiveTargetIdentity {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
        };
        construct(&program, &target, &supports_all, &domain).expect("construction succeeds")
    }

    fn profile() -> EffectiveTargetProfile {
        EffectiveTargetProfile {
            backend: "cpu".to_string(),
            capability_fingerprint: "test-fingerprint".to_string(),
            toolchain_fingerprint: "test-toolchain".to_string(),
            effective_signatures: BTreeSet::new(),
            limits: TargetLimits {
                max_participants: 1024,
                max_workgroups_axis: [65535; 3],
                max_workgroup_bytes: 32768,
                max_explicit_private_bytes: 4096,
                max_direct_bindings: 31,
                max_device_bytes: 1 << 30,
                cooperative_grid: None,
            },
        }
    }

    fn proposals(
        facts: &OccurrenceFacts<'_>,
        profile: &EffectiveTargetProfile,
        occurrence: OccurrenceId,
    ) -> Vec<MappingProposal> {
        let query = RuleQuery {
            facts,
            profile,
            occurrence,
            logical_alternative: 0,
        };
        crate::formation::strategy::universal_rules()
            .iter()
            .flat_map(|rule| rule.propose(&query))
            .collect()
    }

    fn shape_of(facts: &OccurrenceFacts<'_>, profile: &EffectiveTargetProfile, proposal: &MappingProposal) -> ClosedStrategyShape {
        crate::formation::strategy::form(facts, profile, proposal.clone()).unwrap_or_else(|defect| panic!("{defect}"))
    }

    // -- a test-only router: the D1 interface contract by hand ------------------

    /// The type of one leaf of a value: the component at the leaf's path.
    fn leaf_type(ty: &ValueType, path: &[u32]) -> ValueType {
        let mut current = ty.clone();
        for index in path {
            let ValueType::Tuple(items) = current else {
                panic!("leaf path {path:?} descends into a non-tuple");
            };
            current = items.as_slice()[*index as usize].clone();
        }
        current
    }

    struct Router<'a, 'l> {
        facts: &'a OccurrenceFacts<'l>,
        residences: Vec<Residence>,
        routes: BTreeMap<CanonicalLeafId, ValueRoute>,
    }

    impl Router<'_, '_> {
        fn route(&mut self, leaf: CanonicalLeafId, produced: bool) -> ValueRoute {
            if let Some(route) = self.routes.get(&leaf) {
                return route.clone();
            }
            let record = self.facts.leaf(leaf);
            let ty = leaf_type(&self.facts.value_type(record.value), &record.path.0);
            let abi_leaf = BoundaryLeaf::Input {
                param: 0,
                leaf: ValuePath::default(),
            };
            let route = match ty {
                ValueType::Tensor(tensor) => {
                    let id = ResidenceId(self.residences.len() as u32);
                    self.residences.push(Residence {
                        source: if produced {
                            ResidenceSource::ComputedSpill(record.value)
                        } else {
                            ResidenceSource::RootInput(abi_leaf)
                        },
                        shape: tensor,
                        planes: NonEmpty::new(vec![PlanePlan {
                            plane: StoragePlane::Dense,
                            bytes: Sym::constant(0),
                            alignment: 4,
                        }])
                        .expect("one plane"),
                        lifetime: Lifetime::Whole,
                        choice: ResidenceChoice::Fixed(StorageScope::DeviceArena),
                        replication: Replication::Once,
                    });
                    ValueRoute::Tensor(TensorRoute {
                        residence: id,
                        access: Access::Exclusive,
                        transform: ViewTransformTemplate::default(),
                    })
                }
                ValueType::Void => ValueRoute::Void,
                _ => ValueRoute::Scalar(if produced {
                    ScalarRoute::ResultField {
                        leaf: abi_leaf,
                        endpoint: record.endpoint,
                    }
                } else {
                    ScalarRoute::RootAbi {
                        leaf: abi_leaf,
                        endpoint: record.endpoint,
                    }
                }),
            };
            self.routes.insert(leaf, route.clone());
            route
        }
    }

    fn routed(facts: &OccurrenceFacts<'_>, shape: ClosedStrategyShape) -> RoutedStrategy {
        let mut router = Router {
            facts,
            residences: Vec::new(),
            routes: BTreeMap::new(),
        };
        let mut blocks = Vec::new();
        for (id, cut) in shape.blocks().entries() {
            let inputs: Vec<KernelInputDecl> = cut
                .inputs
                .iter()
                .enumerate()
                .map(|(index, leaf)| KernelInputDecl {
                    id: KernelInputId(index as u32),
                    leaf: *leaf,
                    route: router.route(*leaf, false),
                })
                .collect();
            let outputs: Vec<KernelOutputDecl> = cut
                .outputs
                .iter()
                .enumerate()
                .map(|(index, leaf)| KernelOutputDecl {
                    id: KernelOutputId(index as u32),
                    leaf: *leaf,
                    route: router.route(*leaf, true),
                })
                .collect();
            let axes: Vec<KernelAxisDecl> = cut
                .participants
                .independent_axes
                .iter()
                .map(|axis| KernelAxisDecl {
                    id: KernelAxisId(axis.ordinal),
                    binder: Some(axis.binder),
                })
                .collect();
            let extents: Vec<ExtentExpr> = cut
                .participants
                .independent_axes
                .iter()
                .map(|axis| axis.extent.clone())
                .collect();
            let mapping = match &cut.participants.policy {
                ParticipantPolicy::Linear { participants } => AxisMapping::GridStride {
                    participants: Sym::param(&shape.tuning()[PlanParamId(participants.0)].name),
                },
                ParticipantPolicy::Serial
                | ParticipantPolicy::Cooperative { .. }
                | ParticipantPolicy::DynamicPull { .. }
                | ParticipantPolicy::GridCooperative { .. } => AxisMapping::Serialized,
            };
            let iteration = LinearIterationMap::from_axes(&extents, mapping, &|extent| facts.runtime_extent(extent))
                .expect("the test domains are static");
            blocks.push((
                id,
                RoutedBlock {
                    id,
                    interface: ClosedKernelInterface {
                        inputs: IdVec::new(inputs),
                        outputs: IdVec::new(outputs),
                        locals: IdVec::new(Vec::new()),
                        axes: IdVec::new(axes),
                        iteration,
                    },
                    pull_counter: PullCounter::None,
                },
            ));
        }
        RoutedStrategy::seal(
            shape,
            RouteTable::seal(router.routes),
            ResidenceGraph::seal(IdVec::new(router.residences), 0),
            IdVec::from_iter(blocks),
            IdVec::new(Vec::new()),
            0,
        )
    }

    fn form_block(facts: &OccurrenceFacts<'_>, strategy: &RoutedStrategy, block: BlockId) -> ClosedKernelBlock<NoIntrinsic> {
        KernelFormer::form::<TestDialect>(facts, strategy, block, &NoCatalog).unwrap_or_else(|defect| panic!("{defect}"))
    }

    fn collect(block: &ClosedKernelBlock<NoIntrinsic>) -> Vec<Op> {
        let mut ops = Vec::new();
        block.walk(&mut |op| ops.push(op.clone()));
        ops
    }

    fn stores(ops: &[Op]) -> Vec<(KernelPlaceRef, Vec<KernelValueRef>)> {
        ops.iter()
            .filter_map(|op| match op {
                KernelOp::Core(CoreKernelOp::Store { place, coords, .. }) => Some((*place, coords.clone())),
                _ => None,
            })
            .collect()
    }

    fn repeats(ops: &[Op]) -> Vec<KernelSsaId> {
        ops.iter()
            .filter_map(|op| match op {
                KernelOp::Core(CoreKernelOp::Repeat { binder, .. }) => Some(*binder),
                _ => None,
            })
            .collect()
    }

    fn assert_sealed(block: &ClosedKernelBlock<NoIntrinsic>) {
        // Every declared SSA is defined exactly once across the walk.
        let mut defined = BTreeMap::new();
        block.walk(&mut |op| {
            for id in op.references(&|intrinsic| TestDialect::intrinsic_references(intrinsic)).defines {
                *defined.entry(id).or_insert(0u32) += 1;
            }
        });
        assert_eq!(defined.len(), block.ssa().len(), "every SSA is defined");
        assert!(defined.values().all(|count| *count == 1), "no SSA is defined twice");
        // Status fields are dense per block, in guard order.
        for (index, (field, _)) in block.status_fields().iter().enumerate() {
            assert_eq!(*field, StatusFieldTemplateId(index as u32));
        }
    }

    const KERNEL: &str = "fn add[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    let mut output = result\n    parallel for row in 0..M:\n        for col in 0..N:\n            output[row, col] = x[row, col] + y[row, col]\n    return output\n\nfn linear[M, N](x: &tensor[M, N] f32, y: &tensor[M, N] f32, result: tensor[M, N] f32) -> tensor[M, N] f32:\n    return add(x, y, result)\n";

    /// The parallel-loop block of `add`: the outer binder is the block axis,
    /// the inner ordered loop is a `Repeat` with a serial binder SSA, and the
    /// one element write is one `Store` at `[Axis, Ssa(binder)]`.
    #[test]
    fn streaming_and_point_serial_blocks_store_each_element_at_axis_coordinates() {
        let logical = logical(KERNEL, "linear", &[("M", 4), ("N", 8)]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion validates the logical program");
        let facts = forest.facts();
        let profile = profile();
        let add = OccurrenceId(1);
        assert_eq!(facts.occurrence(add).interface.name, "add");
        for proposal in proposals(facts, &profile, add) {
            let shape = shape_of(facts, &profile, &proposal);
            let strategy = routed(facts, shape);
            let parallel = strategy
                .shape()
                .blocks()
                .entries()
                .find(|(_, cut)| !cut.participants.independent_axes.is_empty())
                .map(|(id, _)| id)
                .expect("the parallel loop is absorbed by one block with an axis");
            let block = form_block(facts, &strategy, parallel);
            assert_sealed(&block);
            let ops = collect(&block);
            let binders = repeats(&ops);
            assert_eq!(binders.len(), 1, "the inner ordered loop is the one `Repeat` ({:?})", proposal.rule);
            let stores = stores(&ops);
            assert_eq!(stores.len(), 1, "one `Store` per element write");
            let (place, coords) = &stores[0];
            assert!(matches!(place, KernelPlaceRef::Input(_)), "the write goes through the exclusive input view");
            assert_eq!(coords.len(), 2);
            assert!(matches!(coords[0], KernelValueRef::Axis(KernelAxisId(0))));
            assert_eq!(coords[1], KernelValueRef::Ssa(binders[0]));
            // Loads carry the same coordinates.
            let loads: Vec<Vec<KernelValueRef>> = ops
                .iter()
                .filter_map(|op| match op {
                    KernelOp::Core(CoreKernelOp::Load { coords, .. }) => Some(coords.clone()),
                    _ => None,
                })
                .collect();
            assert_eq!(loads.len(), 2);
            assert!(loads.iter().all(|load| *load == *coords));
            // No value crosses this block's cut, so nothing is published.
            assert!(!ops.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::Publish { .. }))));
            assert!(block.status_fields().is_empty(), "every index is proved statically");
            // Every block of the strategy forms.
            for (id, _) in strategy.shape().blocks().entries() {
                assert_sealed(&form_block(facts, &strategy, id));
            }
        }
    }

    const GUARDED: &str = "fn g[N](x: &tensor[N] f32, i: i32) -> f32:\n    return x[i]\n";

    /// A data-dependent index keeps its obligation as one `Check` wrapping
    /// the guarded `Load`; the scalar result is published exactly once.
    #[test]
    fn an_index_guard_wraps_the_load_and_the_result_is_published_once() {
        let logical = logical(GUARDED, "g", &[("N", 4)]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion validates the logical program");
        let facts = forest.facts();
        let profile = profile();
        let mut guarded_blocks = 0;
        for proposal in proposals(facts, &profile, facts.entry()) {
            let shape = shape_of(facts, &profile, &proposal);
            let strategy = routed(facts, shape);
            for (id, cut) in strategy.shape().blocks().entries() {
                let block = form_block(facts, &strategy, id);
                assert_sealed(&block);
                if cut.guards.is_empty() {
                    continue;
                }
                guarded_blocks += 1;
                assert_eq!(block.status_fields().len(), 1);
                let ops = collect(&block);
                let checks: Vec<&Op> = ops
                    .iter()
                    .filter(|op| matches!(op, KernelOp::Core(CoreKernelOp::Check { .. })))
                    .collect();
                assert_eq!(checks.len(), 1, "exactly one `Check` site");
                let KernelOp::Core(CoreKernelOp::Check {
                    predicate,
                    status,
                    guarded,
                    obligation,
                }) = checks[0]
                else {
                    unreachable!()
                };
                assert_eq!(*status, StatusFieldTemplateId(0));
                assert_eq!(block.status_fields()[0].1, *obligation);
                assert!(matches!(
                    predicate,
                    CheckPredicate::IndexInBounds {
                        index: KernelValueRef::Input(_),
                        extent: ExtentExpr::Static(4)
                    }
                ));
                assert!(guarded.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::Load { .. }))));
                let publishes = ops
                    .iter()
                    .filter(|op| matches!(op, KernelOp::Core(CoreKernelOp::Publish { .. })))
                    .count();
                assert_eq!(publishes, 1, "the returned scalar is published once");
            }
        }
        assert!(guarded_blocks > 0, "the data-dependent index reaches a kernel guard");
    }

    const PACKED: &str = "fn d[N](x: &tensor[N] q4g64) -> f32:\n    let v = decode(x)\n    return reduce(v, 0, sum)\n";

    /// A decode expands to plane reads and scalar ops (`PackedPlaneRead`,
    /// `Cast`, `Fma`): no op decodes. Streaming spills the decoded tensor
    /// into its output residence; point-serial fuses it into the reduction
    /// loop with a `Carry`.
    #[test]
    fn a_packed_decode_expands_into_plane_reads_and_fma() {
        let logical = logical(PACKED, "d", &[("N", 128)]);
        let forest = OccurrenceForest::expand(&logical).expect("expansion validates the logical program");
        let facts = forest.facts();
        let profile = profile();
        let mut spilled = 0;
        let mut fused = 0;
        for proposal in proposals(facts, &profile, facts.entry()) {
            let shape = shape_of(facts, &profile, &proposal);
            let strategy = routed(facts, shape);
            for (id, _) in strategy.shape().blocks().entries() {
                let block = form_block(facts, &strategy, id);
                assert_sealed(&block);
                let ops = collect(&block);
                let plane_reads: Vec<PlaneField> = ops
                    .iter()
                    .filter_map(|op| match op {
                        KernelOp::Core(CoreKernelOp::PackedPlaneRead { plane, repr, .. }) => {
                            assert_eq!(repr, "q4g64");
                            Some(*plane)
                        }
                        _ => None,
                    })
                    .collect();
                if plane_reads.is_empty() {
                    continue;
                }
                assert_eq!(plane_reads, vec![PlaneField::Words, PlaneField::Scale, PlaneField::Bias]);
                assert!(ops.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::Fma { dtype: DType::F32, .. }))));
                assert!(!ops.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::TableLookup { .. }))));
                let has_carry = ops.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::Carry { .. })));
                let has_output_store = stores(&ops).iter().any(|(place, _)| matches!(place, KernelPlaceRef::Output(_)));
                if has_carry {
                    fused += 1;
                    assert!(!ops.iter().any(|op| matches!(op, KernelOp::Core(CoreKernelOp::Fold { .. }))));
                    assert_eq!(
                        ops.iter().filter(|op| matches!(op, KernelOp::Core(CoreKernelOp::Publish { .. }))).count(),
                        1
                    );
                } else {
                    spilled += 1;
                    assert!(has_output_store, "the decoded tensor crosses the cut into its spill residence");
                }
            }
        }
        assert!(spilled > 0, "the streaming rule spills the decoded tensor");
        assert!(fused > 0, "the point-serial rule fuses the decode into the reduction");
    }
}
