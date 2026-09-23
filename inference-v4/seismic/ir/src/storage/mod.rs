//! Storage, lifetime, and allocation topology (spec §8).
//!
//! Global and launch-local storage are different types and are never
//! convertible. Native launch bindings accept only global buffer views;
//! local storage is declared inside a kernel and consumed by its native
//! compiler. Resource expressions are derived from this topology once; no
//! later phase recomputes them from schedule inspection.
//!
//! W4 owns the internals; the handle types and read surface are frozen.

mod layout;
mod local;

pub use layout::{
    addressed_span_u64, concrete_storage_units, representation_alignment, valid_concrete_view,
};
pub use local::{
    derive_launch_local_layout, LaunchAbiRequirement, LaunchLocalId, LaunchLocalKind,
    LaunchLocalLayout, LaunchScratchRequirements, LocalAllocation, LocalLayout,
    ScratchRequirement,
};

use crate::identity::OwnerToken;
use crate::repr::Representation;
use seismic_lang::expr::{CmpOp, AnyExpr, DecisionId, ExprArena, NatExpr, NodeView};
use seismic_lang::ids::{ParameterId, RepresentationId, SemanticValueId};
use seismic_lang::registry::{self, RepresentationKind};
use std::fmt;
use std::marker::PhantomData;

/// One global allocation, untyped by representation (an allocation is bytes;
/// views are typed).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalAllocationId {
    owner: OwnerToken,
    index: u32,
}

impl fmt::Debug for GlobalAllocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}/galloc#{}", self.owner, self.index)
    }
}

impl GlobalAllocationId {
    pub(crate) fn new(owner: OwnerToken, index: u32) -> Self {
        Self { owner, index }
    }
    pub fn index(self) -> u32 {
        self.index
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
}

/// One tensor parameter/result of a structured schedule region. Its owner is
/// the region product itself; it is not an allocation or an alias assertion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorValueId {
    owner: OwnerToken,
    index: u32,
}
impl TensorValueId {
    pub(crate) fn new(owner: OwnerToken, index: u32) -> Self {
        Self { owner, index }
    }
    pub fn index(self) -> u32 {
        self.index
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
}

/// The actual base of a view. Region values resolve through their structured
/// product; only allocation bases refer directly to physical storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ViewBase {
    Allocation(GlobalAllocationId),
    TensorValue(TensorValueId),
}

/// A typed view of one global allocation: `(base allocation, typed layout)`.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferViewId<R: Representation> {
    owner: OwnerToken,
    index: u32,
    repr: PhantomData<R>,
}

impl<R: Representation> Clone for BufferViewId<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: Representation> Copy for BufferViewId<R> {}
impl<R: Representation> fmt::Debug for BufferViewId<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}/view<{}>#{}", self.owner, R::NAME, self.index)
    }
}

/// A representation-erased view handle for tables and provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AnyBufferView {
    pub(crate) index: u32,
    pub(crate) representation: RepresentationId,
    owner: OwnerToken,
}
impl AnyBufferView {
    pub fn index(&self) -> u32 {
        self.index
    }
    pub fn representation(&self) -> RepresentationId {
        self.representation
    }
}

impl<R: Representation> BufferViewId<R> {
    pub fn erase(self) -> AnyBufferView {
        AnyBufferView {
            index: self.index,
            representation: R::id(),
            owner: self.owner,
        }
    }
    pub(crate) fn new(owner: OwnerToken, index: u32) -> Self {
        Self {
            owner,
            index,
            repr: PhantomData,
        }
    }
    pub fn index(self) -> u32 {
        self.index
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
}

impl AnyBufferView {
    pub(crate) fn new(owner: OwnerToken, index: u32, representation: RepresentationId) -> Self {
        Self {
            owner,
            index,
            representation,
        }
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
}

/// Kinds of global storage (§8.1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GlobalBufferKind {
    /// Bound from a call argument.
    Argument {
        value: SemanticValueId,
        abi: Option<ParameterId>,
    },
    /// Allocated by the runtime for a result leaf, by ordinal path.
    Result { value: SemanticValueId },
    /// A child-construction proxy for a caller-owned view. It is eliminated
    /// by the typed import operation and can never reach a root frozen plan.
    Imported {
        value: SemanticValueId,
        source: AnyBufferView,
    },
    /// Invocation-scoped scratch in the plan's arena.
    Arena,
    /// Outlives the invocation (kernel-owned persistent state).
    Persistent,
}

/// One concrete use site in the structured schedule. `region` is a path from
/// the root through structured control nodes; unlike a flattened position it
/// preserves mutual exclusion and loop scope.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScheduleUse {
    pub(crate) owner: OwnerToken,
    pub(crate) region: Vec<ScheduleRegionEdge>,
    pub(crate) ordinal: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScheduleRegionEdge {
    IfThen {
        node: u32,
        parent_ordinal: u32,
    },
    IfElse {
        node: u32,
        parent_ordinal: u32,
    },
    RepeatBody {
        node: u32,
        parent_ordinal: u32,
    },
    Imported {
        node: u32,
        parent_ordinal: u32,
    },
}

/// Exact structured use set of one global allocation. Empty means the
/// allocation is part of the ABI (argument/result/persistent) and therefore
/// lives for the invocation; arena allocations must have at least one use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AllocationLiveness {
    uses: Vec<ScheduleUse>,
}

impl AllocationLiveness {
    pub(crate) fn new(uses: Vec<ScheduleUse>) -> Self {
        Self { uses }
    }
    pub fn uses(&self) -> &[ScheduleUse] {
        &self.uses
    }
}

/// Allocation timing derived from the closed schedule's instance transitions
/// and root publications. It is not supplied by allocation-slot choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationAcquisition {
    Invocation,
    Reached,
}

/// One global allocation's facts.
#[derive(Clone, Debug)]
pub struct GlobalAllocation {
    pub geometry: Option<TensorGeometry>,
    pub acquisition: AllocationAcquisition,
    /// Pre-reserved backing instances derived from actual repeat products.
    pub instances: u32,
    /// Per-bank reservation lifted through the actual definition occurrence scope.
    pub reserved_bytes: NatExpr,
    pub kind: GlobalBufferKind,
    pub bytes: NatExpr,
    pub alignment: u64,
    pub liveness: AllocationLiveness,
    /// When `Some`, an arena slot decision: allocations sharing a slot value
    /// reuse space, and the topology's constraints forbid overlapping
    /// lifetimes in one slot.
    pub slot: Option<DecisionId>,
}

/// Exact root result publication. Results bind views rather than allocations:
/// an alias may publish a slice/strided view whose offset and shape are part of
/// the ABI result contract.
#[derive(Clone, Debug)]
pub struct ResultViewPublication {
    pub path: Vec<u32>,
    pub view: AnyBufferView,
    pub bytes: NatExpr,
}

/// Geometry owned by a tensor allocation. Its whole view shares these exact
/// operands; arbitrary views cannot manufacture this constructor relationship.
#[derive(Clone, Debug)]
pub struct TensorGeometry {
    pub representation: RepresentationId,
    pub extents: Vec<NatExpr>,
    pub strides: Vec<NatExpr>,
}

/// One typed view's layout.
/// The actual defining coordinate map. Cached layout fields are derived only
/// by these constructors; a transpose retains its source view through import.
#[derive(Clone, Debug)]
pub enum ViewMapping {
    /// Issued only together with allocation-owned tensor geometry.
    WholeAllocation,
    Direct,
    Transpose {
        source: AnyBufferView,
        permutation: Vec<u32>,
    },
    Slice {
        source: AnyBufferView,
        selections: Vec<ViewSelection>,
    },
}

/// Actual coordinate operands of a view map. These are not a bounds assertion:
/// closure must establish their range at each use from actual definitions/control.
#[derive(Clone, Copy, Debug)]
pub enum ViewSelection {
    Full,
    Point(NatExpr),
    Range { start: NatExpr, end: NatExpr },
}

#[derive(Clone, Debug)]
pub struct BufferViewLayout {
    pub mapping: ViewMapping,
    pub base: ViewBase,
    pub representation: RepresentationId,
    /// Byte offset within the allocation.
    pub offset: NatExpr,
    pub extents: Vec<NatExpr>,
    /// Element strides, one per axis; packed representations stride in
    /// packets along their packing axis.
    pub strides: Vec<NatExpr>,
    /// True only when construction established the canonical packed/dense
    /// row-major layout for these extents. Direct byte Copy/Fill commands are
    /// restricted to such views; transformed views use elementwise kernels.
    pub contiguous: bool,
}

/// The global allocation topology of one implementation.
#[derive(Debug)]
pub struct GlobalAllocationTopology {
    owner: OwnerToken,
    allocations: Vec<GlobalAllocation>,
    views: Vec<BufferViewLayout>,
    result_views: Vec<ResultViewPublication>,
    /// Pairs of argument parameters proved distinct by the call schema.
    disjoint_arguments: Vec<(ParameterId, ParameterId)>,
}

impl GlobalAllocationTopology {
    /// Returns the owned handle for an existing view, without exposing its scope token.
    pub fn view_id(&self, index: u32) -> AnyBufferView {
        let layout = self
            .views
            .get(index as usize)
            .expect("view is outside closed topology");
        AnyBufferView::new(self.owner, index, layout.representation)
    }

    pub fn retained_bytes(&self) -> usize {
        self.allocations.capacity() * std::mem::size_of::<GlobalAllocation>()
            + self
                .allocations
                .iter()
                .map(|allocation| {
                    allocation.geometry.as_ref().map_or(0, |geometry|
                        (geometry.extents.capacity() + geometry.strides.capacity()) * std::mem::size_of::<NatExpr>())
                        + allocation.liveness.uses.capacity() * std::mem::size_of::<ScheduleUse>()
                        + allocation
                            .liveness
                            .uses
                            .iter()
                            .map(|usage| {
                                usage.region.capacity() * std::mem::size_of::<ScheduleRegionEdge>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
            + self.views.capacity() * std::mem::size_of::<BufferViewLayout>()
            + self
                .views
                .iter()
                .map(|view| {
                    (view.extents.capacity() + view.strides.capacity())
                        * std::mem::size_of::<NatExpr>()
                        + match &view.mapping {
                            ViewMapping::Direct | ViewMapping::WholeAllocation => 0,
                            ViewMapping::Slice { selections, .. } => {
                                selections.capacity() * std::mem::size_of::<ViewSelection>()
                            }
                            ViewMapping::Transpose { permutation, .. } => {
                                permutation.capacity() * std::mem::size_of::<u32>()
                            }
                        }
                })
                .sum::<usize>()
            + self.result_views.capacity() * std::mem::size_of::<ResultViewPublication>()
            + self
                .result_views
                .iter()
                .map(|result| result.path.capacity() * std::mem::size_of::<u32>())
                .sum::<usize>()
            + self.disjoint_arguments.capacity() * std::mem::size_of::<(ParameterId, ParameterId)>()
    }
    pub(crate) fn new(
        owner: OwnerToken,
        allocations: Vec<GlobalAllocation>,
        views: Vec<BufferViewLayout>,
        result_views: Vec<ResultViewPublication>,
        disjoint_arguments: Vec<(ParameterId, ParameterId)>,
    ) -> Self {
        Self {
            owner,
            allocations,
            views,
            result_views,
            disjoint_arguments,
        }
    }
    pub(crate) fn derive_reservations<B: crate::target::PhysicalDialect>(
        &mut self,
        arena: &mut ExprArena,
        schedule: &crate::schedule::ParametricSchedule<B>,
    ) {
        for (ordinal, allocation) in self.allocations.iter_mut().enumerate() {
            allocation.reserved_bytes =
                schedule.allocation_reservation(arena, &self.views, ordinal, allocation.bytes);
        }
    }
    pub fn allocations(&self) -> &[GlobalAllocation] {
        &self.allocations
    }
    pub fn allocation(&self, id: GlobalAllocationId) -> &GlobalAllocation {
        self.assert_owner(id.owner());
        &self.allocations[id.index as usize]
    }
    pub fn views(&self) -> &[BufferViewLayout] {
        &self.views
    }
    pub fn view(&self, view: AnyBufferView) -> &BufferViewLayout {
        self.assert_owner(view.owner());
        &self.views[view.index as usize]
    }
    /// Closed consumers use the same allocation identity/entry alias law as
    /// construction. Different IDs alone never establish argument disjointness.
    pub fn allocations_may_overlap(&self, a: GlobalAllocationId, b: GlobalAllocationId) -> bool {
        self.assert_owner(a.owner());
        self.assert_owner(b.owner());
        a == b
            || allocation_overlap(
                &self.allocations[a.index() as usize].kind,
                &self.allocations[b.index() as usize].kind,
                &self.disjoint_arguments,
            )
    }
    pub fn disjoint_arguments(&self) -> &[(ParameterId, ParameterId)] {
        &self.disjoint_arguments
    }
    pub fn result_views(&self) -> &[ResultViewPublication] {
        &self.result_views
    }
    pub fn allocation_ids(&self) -> impl Iterator<Item = GlobalAllocationId> {
        let owner = self.owner;
        (0..self.allocations.len() as u32).map(move |index| GlobalAllocationId::new(owner, index))
    }
    pub(crate) fn owner(&self) -> OwnerToken {
        self.owner
    }
    fn assert_owner(&self, owner: OwnerToken) {
        assert_eq!(
            owner, self.owner,
            "storage handle belongs to another implementation"
        )
    }
}

/// The launch-local topology of one implementation, keyed by kernel.
#[derive(Debug)]
pub struct LocalAllocationTopology {
    owner: OwnerToken,
    /// `locals[kernel][index]`.
    locals: Vec<Vec<LocalAllocation>>,
}

impl LocalAllocationTopology {
    pub fn retained_bytes(&self) -> usize {
        self.locals.capacity() * std::mem::size_of::<Vec<LocalAllocation>>()
            + self
                .locals
                .iter()
                .map(|locals| {
                    locals.capacity() * std::mem::size_of::<LocalAllocation>()
                        + locals
                            .iter()
                            .map(|local| local.extents.capacity() * std::mem::size_of::<NatExpr>())
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub(crate) fn new(owner: OwnerToken, locals: Vec<Vec<LocalAllocation>>) -> Self {
        Self { owner, locals }
    }
    pub fn of_kernel(&self, kernel: u32) -> &[LocalAllocation] {
        &self.locals[kernel as usize]
    }
    pub fn into_locals(self) -> Vec<Vec<LocalAllocation>> {
        self.locals
    }
}

impl GlobalAllocationTopology {
    pub fn into_parts(
        self,
    ) -> (
        Vec<GlobalAllocation>,
        Vec<BufferViewLayout>,
        Vec<ResultViewPublication>,
        Vec<(ParameterId, ParameterId)>,
    ) {
        (
            self.allocations,
            self.views,
            self.result_views,
            self.disjoint_arguments,
        )
    }
}

// ---------------------------------------------------------------------------
// Construction (W4)
// ---------------------------------------------------------------------------

/// Byte size of a dense row-major tensor of `representation` over
/// `extents`. Packed representations pack along the last axis in groups;
/// a partial trailing group occupies a whole packet.
pub fn tensor_bytes(
    arena: &mut ExprArena,
    representation: RepresentationId,
    extents: &[NatExpr],
) -> NatExpr {
    match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => {
            let elements = arena.nat_product(extents);
            let bytes = arena.nat(dtype.bytes() as u64);
            arena.nat_mul(elements, bytes)
        }
        RepresentationKind::Packed(layout) => {
            let packets = packed_extents(arena, extents, layout.group);
            let count = arena.nat_product(&packets);
            let bytes = arena.nat(u64::from(layout.packet_size));
            arena.nat_mul(count, bytes)
        }
        RepresentationKind::External(layout) => {
            let packets = packed_extents(arena, extents, layout.logical_group);
            let count = arena.nat_product(&packets);
            let bytes = arena.nat(u64::from(layout.packet_size));
            arena.nat_mul(count, bytes)
        }
    }
}

/// Extents in packets: the packing axis (last) is divided into groups.
fn packed_extents(arena: &mut ExprArena, extents: &[NatExpr], group: u32) -> Vec<NatExpr> {
    let mut packets = extents.to_vec();
    if let Some(last) = packets.last_mut() {
        let g = arena.nat(group as u64);
        *last = arena.nat_ceil_div(*last, g);
    }
    packets
}

/// Row-major element strides (in packets along the packing axis of a packed
/// representation).
pub fn dense_strides(
    arena: &mut ExprArena,
    representation: RepresentationId,
    extents: &[NatExpr],
) -> Vec<NatExpr> {
    let units = match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(_) => extents.to_vec(),
        RepresentationKind::Packed(layout) => packed_extents(arena, extents, layout.group),
        RepresentationKind::External(layout) => {
            packed_extents(arena, extents, layout.logical_group)
        }
    };
    let mut strides = vec![arena.nat(1); units.len()];
    let mut acc = arena.nat(1);
    for axis in (0..units.len()).rev() {
        strides[axis] = acc;
        acc = arena.nat_mul(acc, units[axis]);
    }
    strides
}

/// Builds the allocation topology of one implementation. Lifetimes and reuse
/// decisions are attached at close from the schedule.
pub struct TopologyBuilder {
    owner: OwnerToken,
    imported_allowed: bool,
    allocations: Vec<PendingAllocation>,
    views: Vec<BufferViewLayout>,
    result_views: Vec<ResultViewPublication>,
    disjoint: Vec<(ParameterId, ParameterId)>,
}

struct PendingAllocation {
    geometry: Option<TensorGeometry>,
    kind: GlobalBufferKind,
    bytes: NatExpr,
    alignment: u64,
}

impl TopologyBuilder {
    pub(crate) fn new(
        owner: OwnerToken,
        disjoint: Vec<(ParameterId, ParameterId)>,
        imported_allowed: bool,
    ) -> Self {
        Self {
            owner,
            imported_allowed,
            allocations: Vec::new(),
            views: Vec::new(),
            result_views: Vec::new(),
            disjoint,
        }
    }

    pub(crate) fn owner(&self) -> OwnerToken {
        self.owner
    }

    pub fn allocation_count(&self) -> u32 {
        self.allocations.len() as u32
    }
    /// Uses actual construction storage identity and the entry alias contract.
    /// Unfinished region parameters intentionally cannot prove disjointness.
    pub fn may_overlap_views<B: crate::target::PhysicalDialect>(
        &self,
        schedule: &crate::schedule::ScheduleConstruction<B>,
        a: AnyBufferView,
        b: AnyBufferView,
    ) -> bool {
        self.assert_owner(a.owner());
        self.assert_owner(b.owner());
        let Some(left) = schedule.possible_backing_allocations(&self.views, a) else {
            return true;
        };
        let Some(right) = schedule.possible_backing_allocations(&self.views, b) else {
            return true;
        };
        left.iter()
            .any(|a| right.iter().any(|b| self.allocations_may_overlap(*a, *b)))
    }

    fn allocations_may_overlap(&self, a: GlobalAllocationId, b: GlobalAllocationId) -> bool {
        self.assert_owner(a.owner());
        self.assert_owner(b.owner());
        a == b
            || allocation_overlap(
                &self.allocations[a.index() as usize].kind,
                &self.allocations[b.index() as usize].kind,
                &self.disjoint,
            )
    }

    pub fn view_count(&self) -> u32 {
        self.views.len() as u32
    }
    pub fn views(&self) -> &[BufferViewLayout] {
        &self.views
    }
    pub fn view_layout(&self, view: AnyBufferView) -> &BufferViewLayout {
        self.assert_owner(view.owner());
        let layout = self
            .views
            .get(view.index as usize)
            .unwrap_or_else(|| panic!("unknown buffer view {view:?}"));
        assert_eq!(
            layout.representation, view.representation,
            "buffer view representation does not match its topology entry"
        );
        layout
    }
    pub fn allocation_kind(&self, id: GlobalAllocationId) -> &GlobalBufferKind {
        self.assert_owner(id.owner());
        &self.allocations[id.index() as usize].kind
    }
    pub fn allocation_bytes(&self, id: GlobalAllocationId) -> NatExpr {
        self.assert_owner(id.owner());
        self.allocations[id.index() as usize].bytes
    }
    pub fn allocation_alignment(&self, id: GlobalAllocationId) -> u64 {
        self.assert_owner(id.owner());
        self.allocations[id.index() as usize].alignment
    }
    pub fn result_views(&self) -> &[ResultViewPublication] {
        &self.result_views
    }

    pub fn allocate(
        &mut self,
        kind: GlobalBufferKind,
        bytes: NatExpr,
        alignment: u64,
    ) -> GlobalAllocationId {
        assert!(
            alignment.is_power_of_two(),
            "allocation alignment must be a nonzero power of two"
        );
        let id = GlobalAllocationId::new(self.owner, self.allocations.len() as u32);
        self.allocations.push(PendingAllocation {
            geometry: None,
            kind,
            bytes,
            alignment,
        });
        id
    }

    /// Capture the actual caller view as an alias. The proxy's origin is the
    /// caller view's origin, and its capacity is the reachable span, including
    /// holes in a strided layout. Import substitutes the exact source handle;
    /// this does not reserve a new tensor or reconstruct caller provenance.
    pub fn capture_view(
        &mut self,
        arena: &mut ExprArena,
        caller: &TopologyBuilder,
        source: AnyBufferView,
        parameter: SemanticValueId,
    ) -> AnyBufferView {
        assert_ne!(self.owner, caller.owner, "capture requires a child owner");
        let mut layout = caller.view_layout(source).clone();
        layout.offset = arena.nat(0);
        let bytes = if layout.contiguous {
            tensor_bytes(arena, layout.representation, &layout.extents)
        } else {
            addressed_bytes(arena, &layout)
        };
        let allocation = self.allocate(
            GlobalBufferKind::Imported { value: parameter, source },
            bytes,
            representation_alignment(source.representation()),
        );
        layout.base = ViewBase::Allocation(allocation);
        // The map lives in the caller. Imported.source is its actual alias;
        // copying parent-owned map handles into this child would be invalid.
        layout.mapping = ViewMapping::Direct;
        let index = self.push_view(layout);
        AnyBufferView::new(self.owner, index, source.representation())
    }

    /// Derived from reached publication operations by allocation analysis.
    pub(crate) fn set_result_publications(&mut self, publications: Vec<ResultViewPublication>) {
        for publication in &publications { self.assert_owner(publication.view.owner()); }
        self.result_views = publications;
    }

    /// An allocation sized for a dense tensor plus its whole-tensor view.
    pub fn tensor(
        &mut self,
        arena: &mut ExprArena,
        kind: GlobalBufferKind,
        representation: RepresentationId,
        extents: Vec<NatExpr>,
    ) -> (GlobalAllocationId, u32) {
        let bytes = tensor_bytes(arena, representation, &extents);
        let allocation = self.allocate(kind, bytes, representation_alignment(representation));
        let zero = arena.nat(0);
        let view = self.dense_view(arena, allocation, representation, zero, extents);
        let layout = &mut self.views[view as usize];
        layout.mapping = ViewMapping::WholeAllocation;
        self.allocations[allocation.index() as usize].geometry = Some(TensorGeometry {
            representation, extents: layout.extents.clone(), strides: layout.strides.clone(),
        });
        (allocation, view)
    }

    pub fn dense_view(
        &mut self,
        arena: &mut ExprArena,
        allocation: GlobalAllocationId,
        representation: RepresentationId,
        offset: NatExpr,
        extents: Vec<NatExpr>,
    ) -> u32 {
        self.assert_owner(allocation.owner());
        self.allocations[allocation.index() as usize].alignment = self.allocations
            [allocation.index() as usize]
            .alignment
            .max(representation_alignment(representation));
        let strides = dense_strides(arena, representation, &extents);
        self.push_view(BufferViewLayout {
            mapping: ViewMapping::Direct,
            base: ViewBase::Allocation(allocation),
            representation,
            offset,
            extents,
            strides,
            contiguous: true,
        })
    }

    /// Derive an affine descriptor only when the complete logical map is
    /// representable. Failure inserts no views; callers must retain the map.
    pub fn can_project_affine_view(
        &self,
        arena: &mut ExprArena,
        view: &crate::tensor_view::TensorView<AnyBufferView, NatExpr>,
    ) -> bool {
        use crate::tensor_view::{SliceAxis, ViewStep};
        let backing = *view.backing();
        if !view.steps().is_empty()
            && !matches!(registry::representation_info(backing.representation()).kind, RepresentationKind::Dense(_))
        {
            return false;
        }
        let mut extents = self.view_layout(backing).extents.clone();
        let mut strides = self.view_layout(backing).strides.clone();
        for step in view.steps() {
            match step {
                ViewStep::Transpose(permutation) => {
                    extents = permutation.iter().map(|axis| extents[*axis as usize]).collect();
                    strides = permutation.iter().map(|axis| strides[*axis as usize]).collect();
                }
                ViewStep::Slice(axes) => {
                    let mut next_extents = Vec::new();
                    let mut next_strides = Vec::new();
                    for (axis, selection) in axes.iter().enumerate() {
                        match selection {
                            SliceAxis::Point(_) => continue,
                            SliceAxis::Range { start, end } => next_extents.push(arena.nat_sub(*end,*start)),
                            SliceAxis::Full => next_extents.push(extents[axis]),
                        }
                        next_strides.push(strides[axis]);
                    }
                    extents = next_extents;
                    strides = next_strides;
                }
                ViewStep::Reshape { from, to } if from == to => {}
                ViewStep::Reshape { from, to } => {
                    if strides != dense_strides(arena,backing.representation(),from) { return false; }
                    extents = to.clone();
                    strides = dense_strides(arena,backing.representation(),to);
                }
                ViewStep::Plane { .. } => return false,
            }
        }
        true
    }

    pub fn affine_view(
        &mut self,
        arena: &mut ExprArena,
        view: &crate::tensor_view::TensorView<AnyBufferView, NatExpr>,
    ) -> Option<AnyBufferView> {
        use crate::tensor_view::{SliceAxis, ViewStep};
        if !self.can_project_affine_view(arena,view) { return None; }
        let backing=*view.backing();
        let mut actual = backing;
        for step in view.steps() {
            let index = match step {
                ViewStep::Transpose(permutation) => self.transpose_view(actual,permutation),
                ViewStep::Slice(axes) => {
                    let selections = axes.iter().map(|axis| match axis {
                        SliceAxis::Full => ViewSelection::Full,
                        SliceAxis::Point(value) => ViewSelection::Point(*value),
                        SliceAxis::Range { start, end } => ViewSelection::Range { start:*start,end:*end },
                    }).collect::<Vec<_>>();
                    self.slice_view(arena,actual,&selections)
                }
                ViewStep::Reshape { from, to } if from == to => continue,
                ViewStep::Reshape { to, .. } => {
                    let zero=arena.nat(0);
                    let strides=dense_strides(arena,backing.representation(),to);
                    self.subview(arena,actual.index(),zero,to.clone(),strides)
                }
                ViewStep::Plane { .. } => unreachable!("projection checked before mutation"),
            };
            actual=AnyBufferView::new(self.owner,index,backing.representation());
        }
        Some(actual)
    }

    pub fn transpose_view(&mut self, source: AnyBufferView, permutation: &[u32]) -> u32 {
        self.assert_owner(source.owner());
        let layout = self.view_layout(source).clone();
        assert!(
            matches!(
                registry::representation_info(layout.representation).kind,
                RepresentationKind::Dense(_)
            ),
            "packed transpose has no direct storage map"
        );
        let mut axes = permutation.to_vec();
        axes.sort_unstable();
        assert!(
            axes.iter().copied().eq(0..layout.extents.len() as u32),
            "transpose must be a complete coordinate permutation"
        );
        self.push_view(BufferViewLayout {
            mapping: ViewMapping::Transpose {
                source,
                permutation: permutation.to_vec(),
            },
            base: layout.base,
            representation: layout.representation,
            offset: layout.offset,
            extents: permutation
                .iter()
                .map(|axis| layout.extents[*axis as usize])
                .collect(),
            strides: permutation
                .iter()
                .map(|axis| layout.strides[*axis as usize])
                .collect(),
            contiguous: false,
        })
    }

    /// Derive a slice layout from its source and actual coordinate operands.
    /// This constructor retains, rather than discharges, the coordinate bounds.
    pub fn slice_view(
        &mut self,
        arena: &mut ExprArena,
        source: AnyBufferView,
        selections: &[ViewSelection],
    ) -> u32 {
        self.assert_owner(source.owner());
        let layout = self.view_layout(source).clone();
        let RepresentationKind::Dense(dtype) =
            registry::representation_info(layout.representation).kind
        else {
            panic!("stored packed slice requires its registered logical-coordinate map");
        };
        assert_eq!(
            selections.len(),
            layout.extents.len(),
            "slice must select every source axis"
        );
        let mut displacement = arena.nat(0);
        let mut extents = Vec::new();
        let mut strides = Vec::new();
        for (axis, selection) in selections.iter().enumerate() {
            let start = match *selection {
                ViewSelection::Full => {
                    extents.push(layout.extents[axis]);
                    strides.push(layout.strides[axis]);
                    continue;
                }
                ViewSelection::Point(point) => point,
                ViewSelection::Range { start, end } => {
                    extents.push(arena.nat_sub(end, start));
                    strides.push(layout.strides[axis]);
                    start
                }
            };
            let term = arena.nat_mul(start, layout.strides[axis]);
            displacement = arena.nat_add(displacement, term);
        }
        let unit = arena.nat(u64::from(dtype.bytes()));
        let displacement = arena.nat_mul(displacement, unit);
        let index = self.subview(arena, source.index(), displacement, extents, strides);
        self.views[index as usize].mapping = ViewMapping::Slice {
            source,
            selections: selections.to_vec(),
        };
        index
    }

    pub fn strided_view(
        &mut self,
        arena: &mut ExprArena,
        allocation: GlobalAllocationId,
        representation: RepresentationId,
        offset: NatExpr,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> u32 {
        self.assert_owner(allocation.owner());
        self.allocations[allocation.index() as usize].alignment = self.allocations
            [allocation.index() as usize]
            .alignment
            .max(representation_alignment(representation));
        assert_eq!(
            extents.len(),
            strides.len(),
            "view rank and stride count differ"
        );
        let contiguous = strides == dense_strides(arena, representation, &extents);
        self.push_view(BufferViewLayout {
            mapping: ViewMapping::Direct,
            base: ViewBase::Allocation(allocation),
            representation,
            offset,
            extents,
            strides,
            contiguous,
        })
    }

    pub(crate) fn instance_view(
        &mut self, arena: &mut ExprArena, source: AnyBufferView,
        extents: Vec<NatExpr>, strides: Vec<NatExpr>,
    ) -> AnyBufferView {
        let layout = self.view_layout(source);
        assert!(matches!(layout.mapping, ViewMapping::WholeAllocation));
        let representation = layout.representation;
        let value = self.region_view(arena, representation, extents, strides);
        // Begin captures precisely the canonical geometry of this allocation.
        self.views[value.index() as usize].contiguous = true;
        value
    }

    pub(crate) fn region_view(
        &mut self,
        arena: &mut ExprArena,
        representation: RepresentationId,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> AnyBufferView {
        let base = TensorValueId::new(self.owner, self.views.len() as u32);
        assert_eq!(extents.len(), strides.len(), "region view rank differs");
        let offset = arena.nat(0);
        let index = self.push_view(BufferViewLayout {
            mapping: ViewMapping::Direct,
            base: ViewBase::TensorValue(base),
            representation,
            offset,
            extents,
            strides,
            contiguous: false,
        });
        AnyBufferView::new(self.owner, index, representation)
    }

    /// A strided sub-view: `offset` is a byte offset added to the base's.
    pub fn subview(
        &mut self,
        arena: &mut ExprArena,
        base: u32,
        offset: NatExpr,
        extents: Vec<NatExpr>,
        strides: Vec<NatExpr>,
    ) -> u32 {
        assert_eq!(
            extents.len(),
            strides.len(),
            "subview rank and stride count differ"
        );
        let layout = &self.views[base as usize];
        let offset = arena.nat_add(layout.offset, offset);
        let (base, representation) = (layout.base, layout.representation);
        let contiguous = strides == dense_strides(arena, representation, &extents);
        self.push_view(BufferViewLayout {
            mapping: ViewMapping::Direct,
            base,
            representation,
            offset,
            extents,
            strides,
            contiguous,
        })
    }

    fn push_view(&mut self, layout: BufferViewLayout) -> u32 {
        let index = self.views.len() as u32;
        self.views.push(layout);
        index
    }

    /// Imports a closed child's storage into this owner. ABI allocations are
    /// rebound to caller-owned views; child-private allocations and views are
    /// reconstructed with fresh parent ordinals. The returned vector is the
    /// complete child-view -> parent-view remap and exists only for the
    /// duration of the import.
    pub(crate) fn import(
        &mut self,
        arena: &mut ExprArena,
        child: GlobalAllocationTopology,
        bindings: &[(SemanticValueId, AnyBufferView)],
    ) -> Vec<AnyBufferView> {
        assert_ne!(
            child.owner, self.owner,
            "child implementation must have a distinct owner"
        );
        enum Base {
            External(AnyBufferView),
            Imported(GlobalAllocationId),
        }
        let mut bases = Vec::with_capacity(child.allocations.len());
        for allocation in &child.allocations {
            if let GlobalBufferKind::Imported { source, .. } = &allocation.kind {
                self.assert_owner(source.owner());
                bases.push(Base::External(*source));
                continue;
            }
            let value = match &allocation.kind {
                GlobalBufferKind::Argument { value, .. } | GlobalBufferKind::Result { value } => {
                    Some(*value)
                }
                GlobalBufferKind::Imported { .. } => unreachable!("handled above"),
                GlobalBufferKind::Arena | GlobalBufferKind::Persistent => None,
            };
            if let Some(value) = value {
                let view = bindings
                    .iter()
                    .find_map(|(bound, view)| (*bound == value).then_some(*view))
                    .unwrap_or_else(|| {
                        panic!("spliced child storage value {value:?} has no caller binding")
                    });
                self.assert_owner(view.owner());
                bases.push(Base::External(view));
            } else {
                let imported = self.allocate(allocation.kind.clone(), allocation.bytes, allocation.alignment);
                self.allocations[imported.index() as usize].geometry = allocation.geometry.clone();
                bases.push(Base::Imported(imported));
            }
        }
        let mut remap = Vec::with_capacity(child.views.len());
        for (ordinal, layout) in child.views.into_iter().enumerate() {
            if let ViewMapping::Slice { source, selections } = &layout.mapping {
                let source = remap[source.index() as usize];
                let index = self.slice_view(arena, source, selections);
                remap.push(AnyBufferView::new(self.owner, index, layout.representation));
                continue;
            }
            if let ViewMapping::Transpose {
                source,
                permutation,
            } = &layout.mapping
            {
                let source = remap[source.index() as usize];
                let index = self.transpose_view(source, permutation);
                remap.push(AnyBufferView::new(self.owner, index, layout.representation));
                continue;
            }
            if let ViewBase::TensorValue(region) = layout.base {
                let view = if region.index() as usize == ordinal {
                    self.region_view(arena, layout.representation, layout.extents, layout.strides)
                } else {
                    let defining: AnyBufferView = remap[region.index() as usize];
                    let index = self.subview(
                        arena,
                        defining.index(),
                        layout.offset,
                        layout.extents,
                        layout.strides,
                    );
                    AnyBufferView::new(self.owner, index, layout.representation)
                };
                remap.push(view);
                continue;
            }
            let ViewBase::Allocation(allocation) = layout.base else {
                unreachable!()
            };
            let base = &bases[allocation.index() as usize];
            let index = match base {
                Base::External(view) => {
                    assert_eq!(
                        view.representation, layout.representation,
                        "spliced view representation mismatch"
                    );
                    let external = &self.views[view.index as usize];
                    let identity_view = matches!(
                        arena.view(AnyExpr::Nat(layout.offset)),
                        NodeView::NatConst(0)
                    ) && layout.extents == external.extents
                        && layout.strides == external.strides
                        && layout.contiguous == external.contiguous;
                    if identity_view {
                        // Preserve exact caller-view identity across semantic
                        // splicing. Reconstructing an identical whole view as
                        // a generic subview loses its contiguous provenance and
                        // manufactures an equivalent but noncanonical addressed-
                        // range predicate that universal closure cannot match to
                        // TargetDomain's canonical tensor-byte fact.
                        view.index
                    } else {
                        self.subview(
                            arena,
                            view.index,
                            layout.offset,
                            layout.extents,
                            layout.strides,
                        )
                    }
                }
                Base::Imported(allocation) => {
                    let whole = matches!(layout.mapping, ViewMapping::WholeAllocation);
                    let index = self.strided_view(arena, *allocation, layout.representation,
                        layout.offset, layout.extents, layout.strides);
                    if whole {
                        assert!(self.allocations[allocation.index() as usize].geometry.is_some());
                        self.views[index as usize].mapping = ViewMapping::WholeAllocation;
                    }
                    index
                },
            };
            remap.push(AnyBufferView::new(self.owner, index, layout.representation));
        }
        remap
    }

    /// Closes with exactly one derived structured use set and reuse decision
    /// per allocation. Count mismatches are construction bugs; explicit
    /// indexing prevents the silent truncation possible with `zip`.
    pub(crate) fn close(
        self,
        liveness: Vec<AllocationLiveness>,
        slots: Vec<Option<DecisionId>>,
        instances: Vec<u32>,
        acquisitions: Vec<AllocationAcquisition>,
    ) -> GlobalAllocationTopology {
        let count = self.allocations.len();
        assert_eq!(
            liveness.len(),
            count,
            "one liveness set is required per allocation"
        );
        assert_eq!(
            slots.len(),
            count,
            "one reuse decision is required per allocation"
        );
        assert!(
            self.imported_allowed
                || !self
                    .allocations
                    .iter()
                    .any(|allocation| matches!(allocation.kind, GlobalBufferKind::Imported { .. })),
            "caller-owned imported storage escaped into a root implementation"
        );
        for live in &liveness {
            for use_site in live.uses() {
                self.assert_owner(use_site.owner);
            }
        }
        assert_eq!(
            instances.len(),
            count,
            "one instance capacity per allocation"
        );
        assert_eq!(acquisitions.len(), count, "one acquisition timing per allocation");
        let mut acquisitions = acquisitions.into_iter();
        let mut instances = instances.into_iter();
        let mut pending = self.allocations.into_iter();
        let mut liveness = liveness.into_iter();
        let mut slots = slots.into_iter();
        let allocations = (0..count)
            .map(|_| {
                let pending = pending.next().expect("allocation count was checked");
                GlobalAllocation {
                    geometry: pending.geometry,
                    acquisition: acquisitions.next().expect("acquisition count checked"),
                    instances: instances.next().expect("instance count checked"),
                    kind: pending.kind,
                    bytes: pending.bytes,
                    reserved_bytes: pending.bytes,
                    alignment: pending.alignment,
                    liveness: liveness.next().expect("liveness count was checked"),
                    slot: slots.next().expect("slot count was checked"),
                }
            })
            .collect();
        GlobalAllocationTopology::new(
            self.owner,
            allocations,
            self.views,
            self.result_views,
            self.disjoint,
        )
    }

    fn assert_owner(&self, owner: OwnerToken) {
        assert_eq!(
            owner, self.owner,
            "storage handle belongs to another implementation"
        )
    }
}

impl ScheduleUse {
    pub fn region(&self) -> &[ScheduleRegionEdge] {
        &self.region
    }
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }
}

fn allocation_overlap(
    a: &GlobalBufferKind,
    b: &GlobalBufferKind,
    disjoint: &[(ParameterId, ParameterId)],
) -> bool {
    match (a, b) {
        (
            GlobalBufferKind::Argument { abi: Some(a), .. },
            GlobalBufferKind::Argument { abi: Some(b), .. },
        ) => !disjoint
            .iter()
            .any(|(x, y)| (x == a && y == b) || (x == b && y == a)),
        (
            GlobalBufferKind::Argument { .. } | GlobalBufferKind::Imported { .. },
            GlobalBufferKind::Argument { .. } | GlobalBufferKind::Imported { .. },
        ) => true,
        _ => false,
    }
}

pub fn addressed_axis_span(
    arena: &mut ExprArena,
    extent: NatExpr,
    stride: NatExpr,
) -> NatExpr {
    if matches!(arena.view(AnyExpr::Nat(stride)), NodeView::NatConst(0)) {
        return stride;
    }
    let one = arena.nat(1);
    let nonzero_extent = arena.nat_max(extent, one);
    let last = arena.nat_sub(nonzero_extent, one);
    arena.nat_mul(last, stride)
}

/// Exclusive byte end of the furthest element reachable through a view.
/// A zero-extent view addresses no bytes and therefore ends at its offset.
pub fn addressed_bytes(
    arena: &mut ExprArena,
    layout: &BufferViewLayout,
) -> NatExpr {
    assert_eq!(
        layout.extents.len(),
        layout.strides.len(),
        "closed view rank and stride count differ"
    );
    let (unit_bytes, logical_extents) =
        match &registry::representation_info(layout.representation).kind {
            registry::RepresentationKind::Dense(dtype) => {
                (dtype.bytes() as u64, layout.extents.clone())
            }
            registry::RepresentationKind::Packed(packet) => {
                let mut extents = layout.extents.clone();
                if let Some(last) = extents.last_mut() {
                    let group = arena.nat(u64::from(packet.group));
                    *last = arena.nat_ceil_div(*last, group);
                }
                (u64::from(packet.packet_size), extents)
            }
            registry::RepresentationKind::External(packet) => {
                let mut extents = layout.extents.clone();
                if let Some(last) = extents.last_mut() {
                    let group = arena.nat(u64::from(packet.logical_group));
                    *last = arena.nat_ceil_div(*last, group);
                }
                (u64::from(packet.packet_size), extents)
            }
        };
    let zero = arena.nat(0);
    let one = arena.nat(1);
    let mut span = zero;
    let mut empty_terms = Vec::with_capacity(logical_extents.len());
    for (extent, stride) in logical_extents
        .iter()
        .copied()
        .zip(layout.strides.iter().copied())
    {
        empty_terms.push(arena.nat_cmp(CmpOp::Eq, extent, zero));
        let axis = addressed_axis_span(arena, extent, stride);
        span = arena.nat_add(span, axis);
    }
    let element_bytes = arena.nat(unit_bytes);
    let span_with_element = arena.nat_add(span, one);
    let payload_end = arena.nat_mul(span_with_element, element_bytes);
    let end = arena.nat_add(layout.offset, payload_end);
    let empty = arena.any(&empty_terms);
    arena.nat_select(empty, layout.offset, end)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::DenseF32;

    #[test]
    fn concrete_view_footprint_handles_packed_units_and_empty_axes() {
        let packed = registry::representation("q8g32").unwrap();
        let (_, packet_bytes) = concrete_storage_units(packed, &[33]).unwrap();
        assert!(valid_concrete_view(packed, &[33], &[1], 0, 2 * packet_bytes));
        assert!(!valid_concrete_view(packed, &[33], &[1], 0, packet_bytes));
        assert!(valid_concrete_view(packed, &[0], &[u64::MAX], 0, 0));
        assert!(!valid_concrete_view(DenseF32::id(), &[3], &[u64::MAX], 0, 12));
    }

    #[test]
    fn only_tensor_constructor_issues_whole_geometry_and_import_preserves_it() {
        let mut arena = ExprArena::default();
        let mut child = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let representation = DenseF32::id();
        let three = arena.nat(3);
        let zero = arena.nat(0);
        let (allocation, whole) = child.tensor(&mut arena, GlobalBufferKind::Arena, representation, vec![three]);
        let arbitrary = child.dense_view(&mut arena, allocation, representation, zero, vec![three]);
        assert!(matches!(child.views[whole as usize].mapping, ViewMapping::WholeAllocation));
        assert!(matches!(child.views[arbitrary as usize].mapping, ViewMapping::Direct));
        let child = child.close(vec![AllocationLiveness::new(vec![])], vec![None], vec![1], vec![AllocationAcquisition::Reached]);
        let mut parent = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let remap = parent.import(&mut arena, child, &[]);
        let layout = parent.view_layout(remap[whole as usize]);
        assert!(matches!(layout.mapping, ViewMapping::WholeAllocation));
        let ViewBase::Allocation(id) = layout.base else { panic!("whole view lost allocation"); };
        let geometry = parent.allocations[id.index() as usize].geometry.as_ref().unwrap();
        assert_eq!(geometry.extents, layout.extents);
        assert_eq!(geometry.strides, layout.strides);
        assert!(matches!(parent.view_layout(remap[arbitrary as usize]).mapping, ViewMapping::Direct));
    }

    #[test]
    fn affine_projection_preserves_offset_and_rejects_nonaffine_reshape_atomically() {
        use crate::tensor_view::{SliceAxis, TensorView};
        let mut arena = ExprArena::default();
        let mut storage = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let representation = DenseF32::id();
        let zero = arena.nat(0);
        let one = arena.nat(1);
        let two = arena.nat(2);
        let three = arena.nat(3);
        let four = arena.nat(4);
        let six = arena.nat(6);
        let (_, index) = storage.tensor(&mut arena, GlobalBufferKind::Persistent, representation, vec![three, four]);
        let backing = AnyBufferView::new(storage.owner, index, representation);
        let map = TensorView::new(backing, vec![three, four]).slice(vec![
            SliceAxis::Range { start: one, end: three },
            SliceAxis::Range { start: zero, end: three },
        ], |end, start| arena.nat_sub(end, start)).transpose(vec![1, 0]);
        let actual = storage.affine_view(&mut arena, &map).unwrap();
        let layout = storage.view_layout(actual);
        assert_eq!(layout.offset, arena.nat(16));
        assert_eq!(layout.extents, vec![three, two]);
        assert_eq!(layout.strides, vec![one, four]);
        let nonaffine = map.reshape(vec![six]);
        let before = storage.views.len();
        assert!(storage.affine_view(&mut arena, &nonaffine).is_none());
        assert_eq!(storage.views.len(), before, "failed projection must not partially construct views");
    }

    #[test]
    fn captured_strided_view_keeps_span_and_exact_caller_lineage() {
        use seismic_lang::checked::{check_source, SourceFile, SourceSet};
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "capture.seismic".into(),
            text: "fn probe(x: &tensor[2,3] f32):\n    let y = x[0,0]\n".into(),
        }])).unwrap();
        let entry = module.entry(module.entry_named("probe").unwrap(), &Default::default()).unwrap();
        let parameter = entry.schema().parameters()[0].value;
        let mut arena = ExprArena::default();
        let mut parent = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let representation = DenseF32::id();
        let one = arena.nat(1);
        let three = arena.nat(3);
        let four = arena.nat(4);
        let (_, whole) = parent.tensor(&mut arena, GlobalBufferKind::Persistent, representation, vec![three, four]);
        let whole = AnyBufferView::new(parent.owner, whole, representation);
        let sliced = parent.slice_view(&mut arena, whole, &[
            ViewSelection::Range { start: one, end: three },
            ViewSelection::Range { start: one, end: four },
        ]);
        let sliced = AnyBufferView::new(parent.owner, sliced, representation);
        let mut child = TopologyBuilder::new(OwnerToken::fresh(), vec![], true);
        let proxy = child.capture_view(&mut arena, &parent, sliced, parameter);
        assert_eq!(child.allocations[0].bytes, arena.nat(28));
        assert_eq!(child.view_layout(proxy).offset, arena.nat(0));
        let transposed = child.transpose_view(proxy, &[1, 0]);
        let child = child.close(vec![AllocationLiveness::new(vec![])], vec![None], vec![1], vec![AllocationAcquisition::Invocation]);
        let remap = parent.import(&mut arena, child, &[]);
        assert_eq!(remap[proxy.index() as usize], sliced);
        let result = parent.view_layout(remap[transposed as usize]);
        assert_eq!(result.offset, arena.nat(20));
        assert_eq!(result.strides, vec![one, four]);
        assert!(matches!(&result.mapping, ViewMapping::Transpose { source, .. } if *source == sliced));
        assert!(matches!(parent.view_layout(sliced).mapping, ViewMapping::Slice { .. }));
    }

    #[test]
    fn slice_map_derives_rectangular_offset_and_survives_import() {
        let mut arena = ExprArena::default();
        let mut child = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let representation = DenseF32::id();
        let one = arena.nat(1);
        let two = arena.nat(2);
        let three = arena.nat(3);
        let four = arena.nat(4);
        let (_, whole) = child.tensor(
            &mut arena,
            GlobalBufferKind::Persistent,
            representation,
            vec![three, four],
        );
        let source = AnyBufferView::new(child.owner, whole, representation);
        let sliced = child.slice_view(
            &mut arena,
            source,
            &[
                ViewSelection::Range {
                    start: one,
                    end: three,
                },
                ViewSelection::Range {
                    start: one,
                    end: four,
                },
            ],
        );
        let sliced = AnyBufferView::new(child.owner, sliced, representation);
        let transposed = child.transpose_view(sliced, &[1, 0]);
        let child = child.close(vec![AllocationLiveness::new(vec![])], vec![None], vec![1], vec![AllocationAcquisition::Invocation]);
        let mut parent = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let remap = parent.import(&mut arena, child, &[]);
        let layout = parent.view_layout(remap[sliced.index() as usize]);
        assert_eq!(layout.offset, arena.nat(20));
        assert_eq!(layout.extents, vec![two, three]);
        assert_eq!(layout.strides, vec![four, one]);
        let ViewMapping::Slice { source, .. } = &layout.mapping else {
            panic!("slice map lost during import")
        };
        assert_eq!(*source, remap[whole as usize]);
        let layout = parent.view_layout(remap[transposed as usize]);
        assert_eq!(layout.extents, vec![three, two]);
        assert_eq!(layout.strides, vec![one, four]);
    }

    #[test]
    fn imported_views_derive_contiguity_without_losing_offsets_or_strides() {
        let mut arena = ExprArena::default();
        let mut child = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let representation = DenseF32::id();
        let zero = arena.nat(0);
        let one = arena.nat(1);
        let two = arena.nat(2);
        let four = arena.nat(4);
        let sixteen = arena.nat(16);
        let (allocation, whole) = child.tensor(
            &mut arena,
            GlobalBufferKind::Persistent,
            representation,
            vec![two, four],
        );
        let row = child.subview(&mut arena, whole, sixteen, vec![four], vec![one]);
        let transpose = child.strided_view(
            &mut arena,
            allocation,
            representation,
            zero,
            vec![four, two],
            vec![one, four],
        );
        let child = child.close(vec![AllocationLiveness::new(vec![])], vec![None], vec![1], vec![AllocationAcquisition::Invocation]);
        let mut parent = TopologyBuilder::new(OwnerToken::fresh(), vec![], false);
        let remap = parent.import(&mut arena, child, &[]);
        assert!(parent.view_layout(remap[whole as usize]).contiguous);
        let row_layout = parent.view_layout(remap[row as usize]);
        assert!(row_layout.contiguous);
        assert_eq!(row_layout.offset, sixteen);
        assert_eq!(row_layout.extents, vec![four]);
        let transposed = parent.view_layout(remap[transpose as usize]);
        assert!(!transposed.contiguous);
        assert_eq!(transposed.strides, vec![one, four]);
    }
}
