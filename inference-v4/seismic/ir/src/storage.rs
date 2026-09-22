//! Storage, lifetime, and allocation topology (spec §8).
//!
//! Global and launch-local storage are different types and are never
//! convertible. Native launch bindings accept only global buffer views;
//! local storage is declared inside a kernel and consumed by its native
//! compiler. Resource expressions are derived from this topology once; no
//! later phase recomputes them from schedule inspection.
//!
//! W4 owns the internals; the handle types and read surface are frozen.

use crate::identity::OwnerToken;
use crate::repr::Representation;
use seismic_lang::expr::{AnyExpr, DecisionId, ExprArena, NatExpr, NodeView};
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

/// A launch-local allocation declared by one kernel. Not convertible to a
/// global buffer.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LaunchLocalId<R: Representation> {
    owner: OwnerToken,
    kernel: u32,
    index: u32,
    repr: PhantomData<R>,
}

impl<R: Representation> Clone for LaunchLocalId<R> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R: Representation> Copy for LaunchLocalId<R> {}
impl<R: Representation> fmt::Debug for LaunchLocalId<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?}/local<{}>#{}.{}",
            self.owner,
            R::NAME,
            self.kernel,
            self.index
        )
    }
}

impl<R: Representation> LaunchLocalId<R> {
    pub(crate) fn new(owner: OwnerToken, kernel: u32, index: u32) -> Self {
        Self {
            owner,
            kernel,
            index,
            repr: PhantomData,
        }
    }
    pub(crate) fn kernel(self) -> u32 {
        self.kernel
    }
    pub fn index(self) -> u32 {
        self.index
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

/// Kinds of launch-local storage (§8.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchLocalKind {
    Workgroup,
    Participant,
    Register,
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
    ChooseOption {
        node: u32,
        parent_ordinal: u32,
        value: i64,
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

/// One global allocation's facts.
#[derive(Clone, Debug)]
pub struct GlobalAllocation {
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

/// One typed view's layout.
#[derive(Clone, Debug)]
pub struct BufferViewLayout {
    pub allocation: GlobalAllocationId,
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

/// One launch-local allocation's facts.
#[derive(Clone, Debug)]
pub struct LocalAllocation {
    pub kind: LaunchLocalKind,
    pub representation: RepresentationId,
    pub extents: Vec<NatExpr>,
    pub alignment: u64,
}

#[derive(Clone, Debug)]
pub struct LocalLayout {
    pub kind: LaunchLocalKind,
    pub representation: RepresentationId,
    pub offset: NatExpr,
    pub extents: Vec<NatExpr>,
    pub strides: Vec<NatExpr>,
    pub bytes: NatExpr,
    pub alignment: u64,
}

#[derive(Clone, Debug)]
pub struct LaunchLocalLayout {
    pub locals: Vec<LocalLayout>,
    pub workgroup_bytes: NatExpr,
    pub participant_bytes: NatExpr,
    pub register_bytes: NatExpr,
}

/// One compiler-owned physical scratch allocation required to realize a
/// launch-local address space on the selected target profile.
#[derive(Clone, Debug)]
pub struct ScratchRequirement {
    pub bytes: NatExpr,
    pub alignment: u64,
}

#[derive(Clone, Debug, Default)]
pub struct LaunchScratchRequirements {
    pub workgroup: Option<ScratchRequirement>,
    pub participant: Option<ScratchRequirement>,
    pub register: Option<ScratchRequirement>,
}

#[derive(Clone, Debug)]
pub struct LaunchAbiRequirement {
    pub role: crate::target::KernelAbiAllocationRole,
    pub bytes: NatExpr,
    pub alignment: u64,
}

pub fn derive_launch_local_layout(
    arena: &mut ExprArena,
    locals: &[LocalAllocation],
    intrinsic_resources: &[crate::kernel::ops::IntrinsicResources],
) -> LaunchLocalLayout {
    let zero = arena.nat(0);
    let mut cursors = [zero, zero, zero];
    let mut layouts = Vec::with_capacity(locals.len());
    for local in locals {
        assert!(
            local.alignment.is_power_of_two(),
            "launch-local alignment must be a nonzero power of two"
        );
        let class = match local.kind {
            LaunchLocalKind::Workgroup => 0,
            LaunchLocalKind::Participant => 1,
            LaunchLocalKind::Register => 2,
        };
        let alignment = arena.nat(local.alignment);
        let groups = arena.nat_ceil_div(cursors[class], alignment);
        let offset = arena.nat_mul(groups, alignment);
        let bytes = tensor_bytes(arena, local.representation, &local.extents);
        cursors[class] = arena.nat_add(offset, bytes);
        layouts.push(LocalLayout {
            kind: local.kind,
            representation: local.representation,
            offset,
            extents: local.extents.clone(),
            strides: dense_strides(arena, local.representation, &local.extents),
            bytes,
            alignment: local.alignment,
        });
    }
    for resources in intrinsic_resources {
        for (class, bytes) in [
            resources.workgroup_bytes,
            resources.participant_bytes,
            resources.register_bytes,
        ]
        .into_iter()
        .enumerate()
        {
            if let Some(bytes) = bytes {
                cursors[class] = arena.nat_add(cursors[class], bytes);
            }
        }
    }
    LaunchLocalLayout {
        locals: layouts,
        workgroup_bytes: cursors[0],
        participant_bytes: cursors[1],
        register_bytes: cursors[2],
    }
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
                    allocation.liveness.uses.capacity() * std::mem::size_of::<ScheduleUse>()
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

/// Minimum alignment of one element of `representation`.
pub fn representation_alignment(representation: RepresentationId) -> u64 {
    match &registry::representation_info(representation).kind {
        RepresentationKind::Dense(dtype) => dtype.bytes() as u64,
        RepresentationKind::Packed(layout) => u64::from(layout.packet_alignment),
        RepresentationKind::External(layout) => u64::from(layout.packet_alignment),
    }
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
            kind,
            bytes,
            alignment,
        });
        id
    }

    pub fn publish_result(&mut self, arena: &mut ExprArena, view: AnyBufferView, path: Vec<u32>) {
        let layout = self.view_layout(view);
        let bytes = tensor_bytes(arena, layout.representation, &layout.extents);
        assert!(
            !self
                .result_views
                .iter()
                .any(|publication| publication.path == path),
            "result path was published twice"
        );
        self.result_views
            .push(ResultViewPublication { path, view, bytes });
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
            allocation,
            representation,
            offset,
            extents,
            strides,
            contiguous: true,
        })
    }

    pub fn strided_view(
        &mut self,
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
        self.push_view(BufferViewLayout {
            allocation,
            representation,
            offset,
            extents,
            strides,
            contiguous: false,
        })
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
        let (allocation, representation) = (layout.allocation, layout.representation);
        self.push_view(BufferViewLayout {
            allocation,
            representation,
            offset,
            extents,
            strides,
            contiguous: false,
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
                bases.push(Base::Imported(self.allocate(
                    allocation.kind.clone(),
                    allocation.bytes,
                    allocation.alignment,
                )));
            }
        }
        let mut remap = Vec::with_capacity(child.views.len());
        for layout in child.views {
            let base = &bases[layout.allocation.index() as usize];
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
                Base::Imported(allocation) => self.strided_view(
                    *allocation,
                    layout.representation,
                    layout.offset,
                    layout.extents,
                    layout.strides,
                ),
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
        let mut pending = self.allocations.into_iter();
        let mut liveness = liveness.into_iter();
        let mut slots = slots.into_iter();
        let allocations = (0..count)
            .map(|_| {
                let pending = pending.next().expect("allocation count was checked");
                GlobalAllocation {
                    kind: pending.kind,
                    bytes: pending.bytes,
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
