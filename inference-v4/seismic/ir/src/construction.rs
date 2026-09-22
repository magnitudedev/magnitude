//! Owns construction identity and the coordinated storage/kernel/schedule tables.
//! Child import rewrites all scoped references together; callers never mint or
//! rebrand handles. Semantic refinement and target selection stay with the compiler.

use crate::identity::OwnerToken;
use crate::kernel::{internals, Kernel, KernelArena, KernelBuilder};
use crate::repr::Representation;
use crate::schedule::{
    AnyScalarSlot, ClosedSchedule, ImportedSchedule, LaunchId, ParametricSchedule, ScheduleBuilder,
    ScheduleConstruction,
};
use crate::storage::GlobalBufferKind;
use crate::storage::{
    AllocationLiveness, AnyBufferView, BufferViewId, GlobalAllocationId, GlobalAllocationTopology,
    LocalAllocationTopology, TopologyBuilder,
};
use crate::target::{AddressableResourceClass, KernelDialect, VectorSupport};
use seismic_lang::expr::{BoolExpr, DecisionId, ExprArena, NatExpr};
use seismic_lang::ids::{ParameterId, RepresentationId};

pub struct Construction<B: KernelDialect> {
    owner: OwnerToken,
    storage: TopologyBuilder,
    kernels: Vec<Kernel<B>>,
    kernel_state: internals::KernelState<B>,
    schedule: ScheduleConstruction<B>,
}

impl<B: KernelDialect> Construction<B> {
    pub fn new(
        arena: &mut ExprArena,
        disjoint: Vec<(ParameterId, ParameterId)>,
        imported_allowed: bool,
        resources: usize,
    ) -> Self {
        let owner = OwnerToken::fresh();
        let zero = arena.nat(0);
        Self {
            owner,
            storage: TopologyBuilder::new(owner, disjoint, imported_allowed),
            kernels: vec![],
            kernel_state: internals::KernelState::new(owner, 0, zero, resources),
            schedule: ScheduleConstruction::new(owner),
        }
    }
    pub fn storage(&self) -> &TopologyBuilder {
        &self.storage
    }
    pub fn storage_mut(&mut self) -> &mut TopologyBuilder {
        &mut self.storage
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        &self.kernels
    }
    pub fn schedule_state(&mut self) -> &mut ScheduleConstruction<B> {
        &mut self.schedule
    }
    pub fn view(&self, index: u32, representation: RepresentationId) -> AnyBufferView {
        let layout = self
            .storage
            .views()
            .get(index as usize)
            .expect("view is outside this construction");
        assert_eq!(
            layout.representation, representation,
            "view representation differs from its layout"
        );
        AnyBufferView::new(self.owner, index, representation)
    }
    pub fn typed_view<R: Representation>(&self, view: AnyBufferView) -> BufferViewId<R> {
        self.assert_view(view);
        assert_eq!(
            view.representation,
            R::id(),
            "typed view representation mismatch"
        );
        BufferViewId::new(self.owner, view.index)
    }
    pub fn allocation(&self, index: u32) -> GlobalAllocationId {
        assert!(
            index < self.storage.allocation_count(),
            "allocation is outside this construction"
        );
        GlobalAllocationId::new(self.owner, index)
    }
    pub fn assert_view(&self, view: AnyBufferView) {
        self.storage.view_layout(view);
    }
    pub fn assert_slot(&self, slot: AnyScalarSlot) {
        assert_eq!(
            slot.owner(),
            self.owner,
            "slot belongs to another construction"
        );
    }
    pub fn kernel<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        facts: &'a B::Facts,
        resources: &'a [AddressableResourceClass],
        vectors: &'a VectorSupport,
    ) -> KernelBuilder<'a, B> {
        internals::open(
            self.owner,
            arena,
            self.storage.views(),
            &mut self.kernels,
            &mut self.kernel_state,
            facts,
            resources,
            vectors,
        )
    }
    pub fn portable_kernel<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        facts: &'a B::Facts,
        resources: &'a [AddressableResourceClass],
        vectors: &'a VectorSupport,
    ) -> internals::PortableBuilder<'a, B> {
        internals::open_portable(
            self.owner,
            arena,
            self.storage.views(),
            &mut self.kernels,
            &mut self.kernel_state,
            facts,
            resources,
            vectors,
        )
    }
    pub fn schedule<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        region: u32,
    ) -> ScheduleBuilder<'a, B> {
        self.schedule
            .builder_at(arena, self.storage.views(), region)
    }
    pub fn import(
        &mut self,
        arena: &mut ExprArena,
        child: ImportableExecutableIr<B>,
        forced_slots: &[(u32, AnyScalarSlot)],
    ) -> ImportedSchedule {
        let child = child.0;
        let ExecutableIr {
            owner: _,
            storage,
            kernels,
            schedule,
            // Import reconstructs child-private allocations in the parent and
            // therefore invalidates the child's physical reuse mapping. The
            // parent derives fresh liveness, slots, and obligations after the
            // combined schedule closes.
            allocation_constraints: _,
        } = child;
        let view_map = self.storage.import(arena, storage, &[]);
        let mut child = internals::arena_into_kernels(kernels);
        let ids: Vec<_> = (0..child.len())
            .map(|offset| {
                crate::kernel::KernelId::new(
                    self.owner,
                    u32::try_from(self.kernels.len() + offset).expect("kernel ordinal overflow"),
                )
            })
            .collect();
        let imported = self
            .schedule
            .import(arena, schedule, &ids, &view_map, forced_slots);
        for (offset, kernel) in child.iter_mut().enumerate() {
            internals::data_mut(kernel).rebrand(
                self.owner,
                ids[offset].index(),
                |view| view_map[view.index as usize],
                |slot| imported.slots[slot.index as usize],
            );
        }
        self.kernels.extend(child);
        imported
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        LocalAllocationTopology::new(
            self.owner,
            self.kernels
                .iter()
                .map(|kernel| kernel.locals().to_vec())
                .collect(),
        )
    }
    /// Closes every mutable IR table as one consuming transition. The returned
    /// phase has no mutation API, so closed storage, kernels, and schedule
    /// cannot subsequently diverge.
    ///
    /// ```compile_fail
    /// use seismic_ir::{construction::ClosedConstruction, target::KernelDialect};
    /// fn cannot_mutate<B: KernelDialect>(mut closed: ClosedConstruction<B>) {
    ///     closed.storage_mut();
    /// }
    /// ```
    pub fn close(self, token: ClosedSchedule) -> ClosedConstruction<B> {
        let schedule = self.schedule.finish(token);
        ClosedConstruction {
            owner: self.owner,
            storage: self.storage,
            kernels: self.kernels,
            schedule,
        }
    }
}

/// Structurally closed executable tables, before schedule-derived allocation
/// facts have been calculated.
pub struct ClosedConstruction<B: KernelDialect> {
    owner: OwnerToken,
    storage: TopologyBuilder,
    kernels: Vec<Kernel<B>>,
    schedule: ParametricSchedule,
}

impl<B: KernelDialect> ClosedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        &self.storage
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        &self.kernels
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        &self.schedule
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        local_allocations(self.owner, &self.kernels)
    }

    /// Derives policy-neutral lifetime and compatibility facts. This consumes
    /// the closed construction so a later physical plan can only be applied to
    /// the exact tables from which these facts were derived.
    pub fn analyze_allocations(self) -> AnalyzedConstruction<B> {
        let mut uses = vec![Vec::new(); self.storage.allocation_count() as usize];
        for (view, at) in self.schedule.direct_view_uses() {
            let allocation = self.storage.view_layout(*view).allocation;
            uses[allocation.index() as usize].push(at.clone());
        }
        for (launch_id, at) in self.schedule.launch_uses() {
            let launch = self.schedule.launch(*launch_id);
            let kernel = self
                .kernels
                .get(launch.kernel.index() as usize)
                .expect("launch kernel was closed into this construction");
            for binding in &kernel.interface().bindings {
                let allocation = self.storage.view_layout(binding.view).allocation;
                uses[allocation.index() as usize].push(at.clone());
            }
        }
        let liveness = uses
            .into_iter()
            .map(AllocationLiveness::new)
            .collect::<Vec<_>>();
        let candidates: Vec<_> = (0..self.storage.allocation_count())
            .map(|index| GlobalAllocationId::new(self.owner, index))
            .filter(|id| matches!(self.storage.allocation_kind(*id), GlobalBufferKind::Arena))
            .collect();
        for candidate in &candidates {
            assert!(
                !liveness[candidate.index() as usize].uses().is_empty(),
                "arena allocation has no structured schedule use"
            );
        }
        let mut relations = Vec::new();
        for (right_ordinal, &right) in candidates.iter().enumerate() {
            for &left in &candidates[..right_ordinal] {
                relations.push(AllocationRelation {
                    left,
                    right,
                    storage_compatible: reuse_compatible(&self.storage, left, right),
                    lifetimes_interfere: lifetimes_interfere(
                        &liveness[left.index() as usize],
                        &liveness[right.index() as usize],
                    ),
                });
            }
        }
        AnalyzedConstruction {
            closed: self,
            liveness,
            arena_allocations: candidates,
            relations,
        }
    }
}

/// A pair of arena allocations and the complete structural facts needed to
/// decide whether a common physical slot is permitted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationRelation {
    left: GlobalAllocationId,
    right: GlobalAllocationId,
    storage_compatible: bool,
    lifetimes_interfere: bool,
}
impl AllocationRelation {
    pub fn left(&self) -> GlobalAllocationId {
        self.left
    }
    pub fn right(&self) -> GlobalAllocationId {
        self.right
    }
    pub fn storage_compatible(&self) -> bool {
        self.storage_compatible
    }
    pub fn lifetimes_interfere(&self) -> bool {
        self.lifetimes_interfere
    }
    pub fn may_share_slot(&self) -> bool {
        self.storage_compatible && !self.lifetimes_interfere
    }
}

/// Closed IR plus schedule-derived, policy-neutral allocation facts.
pub struct AnalyzedConstruction<B: KernelDialect> {
    closed: ClosedConstruction<B>,
    liveness: Vec<AllocationLiveness>,
    arena_allocations: Vec<GlobalAllocationId>,
    relations: Vec<AllocationRelation>,
}
impl<B: KernelDialect> AnalyzedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        self.closed.storage()
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        self.closed.kernels()
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        self.closed.schedule()
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.closed.local_allocations()
    }
    pub fn liveness(&self) -> &[AllocationLiveness] {
        &self.liveness
    }
    /// Arena-backed allocations that may be assigned a symbolic physical-slot
    /// choice by an optimization owner.
    pub fn arena_allocations(&self) -> &[GlobalAllocationId] {
        &self.arena_allocations
    }
    pub fn allocation_relations(&self) -> &[AllocationRelation] {
        &self.relations
    }

    /// Applies a supplied symbolic storage plan to this exact analysis.
    /// Allocations omitted from the plan remain physically distinct. Only
    /// arena allocations may receive symbolic slot choices.
    pub fn apply_allocation_plan(
        self,
        arena: &mut ExprArena,
        plan: AllocationPlan,
    ) -> StoragePlannedConstruction<B> {
        let mut slots = vec![None; self.closed.storage.allocation_count() as usize];
        for assignment in plan.slot_choices {
            let allocation = assignment.allocation;
            let decision = assignment.slot_choice;
            assert_eq!(
                allocation.owner(),
                self.closed.owner,
                "allocation plan belongs to another construction"
            );
            assert!(
                matches!(
                    self.closed.storage.allocation_kind(allocation),
                    GlobalBufferKind::Arena
                ),
                "only arena allocations can use symbolic reuse slots"
            );
            let slot = &mut slots[allocation.index() as usize];
            assert!(slot.is_none(), "allocation has multiple reuse assignments");
            *slot = Some(decision);
        }
        let mut mandatory_constraints = Vec::new();
        for relation in &self.relations {
            if relation.may_share_slot() {
                continue;
            }
            let Some(left) = slots[relation.left.index() as usize] else {
                continue;
            };
            let Some(right) = slots[relation.right.index() as usize] else {
                continue;
            };
            let left_values = arena.decision_domain(left).values().to_vec();
            let right_values = arena.decision_domain(right).values().to_vec();
            for value in left_values {
                if right_values.binary_search(&value).is_ok() {
                    let left_is = arena.decision_is(left, value);
                    let right_is = arena.decision_is(right, value);
                    let both = arena.all(&[left_is, right_is]);
                    mandatory_constraints.push(arena.not(both));
                }
            }
        }
        StoragePlannedConstruction {
            analyzed: self,
            slots,
            mandatory_constraints,
        }
    }
}

/// One arena allocation whose physical slot is selected by a finite decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationSlotChoice {
    allocation: GlobalAllocationId,
    slot_choice: DecisionId,
}
impl AllocationSlotChoice {
    pub fn new(allocation: GlobalAllocationId, slot_choice: DecisionId) -> Self {
        Self {
            allocation,
            slot_choice,
        }
    }
}

/// Compiler-supplied symbolic storage choices. An allocation absent from this
/// plan retains its own physical storage.
#[derive(Default)]
pub struct AllocationPlan {
    slot_choices: Vec<AllocationSlotChoice>,
}
impl AllocationPlan {
    pub fn distinct() -> Self {
        Self::default()
    }
    pub fn new(slot_choices: impl IntoIterator<Item = AllocationSlotChoice>) -> Self {
        Self {
            slot_choices: slot_choices.into_iter().collect(),
        }
    }
}

/// Fully analyzed construction with a supplied symbolic storage plan attached.
pub struct StoragePlannedConstruction<B: KernelDialect> {
    analyzed: AnalyzedConstruction<B>,
    slots: Vec<Option<DecisionId>>,
    mandatory_constraints: Vec<BoolExpr>,
}
impl<B: KernelDialect> StoragePlannedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        self.analyzed.storage()
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        self.analyzed.kernels()
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        self.analyzed.schedule()
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        self.analyzed.local_allocations()
    }
    pub fn liveness(&self) -> &[AllocationLiveness] {
        self.analyzed.liveness()
    }
    pub fn slots(&self) -> &[Option<DecisionId>] {
        &self.slots
    }
    pub fn mandatory_constraints(&self) -> &[BoolExpr] {
        &self.mandatory_constraints
    }

    pub fn finish(self) -> ExecutableIr<B> {
        let ClosedConstruction {
            owner,
            storage,
            kernels,
            schedule,
        } = self.analyzed.closed;
        let executable = ExecutableIr {
            owner,
            storage: storage.close(self.analyzed.liveness, self.slots),
            kernels: internals::arena_from_kernels(owner, kernels),
            schedule,
            allocation_constraints: self.mandatory_constraints,
        };
        executable.assert_coordinated();
        executable
    }
}

/// One authoritative executable IR. Its component tables are intentionally not
/// independently extractable, so schedules, kernels, and storage cannot be
/// recombined across owners after closure.
pub struct ExecutableIr<B: KernelDialect> {
    owner: OwnerToken,
    storage: GlobalAllocationTopology,
    kernels: KernelArena<B>,
    schedule: ParametricSchedule,
    allocation_constraints: Vec<BoolExpr>,
}

/// Why a closed executable cannot be structurally spliced into a parent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImportabilityError {
    /// Root ABI allocations require explicit semantic bindings that the
    /// structural splice operation does not accept.
    RootAllocation { allocation: u32 },
    /// Root result publications belong to the child's external ABI and cannot
    /// silently become publications of the parent.
    ResultPublications,
}

/// An executable whose storage and publication shape permits structural
/// splicing. Construction is only through [`ExecutableIr::into_importable`].
pub struct ImportableExecutableIr<B: KernelDialect>(ExecutableIr<B>);

impl<B: KernelDialect> ExecutableIr<B> {
    pub(crate) fn owner(&self) -> OwnerToken {
        self.owner
    }

    /// Derives launch layouts from this executable's exact kernels. No caller
    /// supplies a layout slice, so the resulting owner stamp certifies both
    /// provenance and launch/kernel correspondence.
    pub fn close_execution(self, arena: &mut ExprArena) -> crate::execution::ClosedExecutableIr<B> {
        let layouts = self
            .schedule
            .launches()
            .iter()
            .map(|launch| {
                let kernel = self.kernels.kernel(launch.kernel);
                crate::storage::derive_launch_local_layout(
                    arena,
                    kernel.locals(),
                    kernel.intrinsic_resources(),
                )
            })
            .collect();
        let closed = crate::execution::ClosedLaunchLayouts {
            owner: self.owner,
            layouts,
        };
        crate::execution::ClosedExecutableIr::new(self, closed)
    }
    pub fn storage(&self) -> &GlobalAllocationTopology {
        &self.storage
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        &self.kernels
    }
    pub fn schedule(&self) -> &ParametricSchedule {
        &self.schedule
    }
    /// Mandatory implications of the attached symbolic allocation slots.
    /// Planning must include these in the executable's hard constraints.
    pub fn allocation_constraints(&self) -> &[BoolExpr] {
        &self.allocation_constraints
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        LocalAllocationTopology::new(
            self.owner,
            self.kernels
                .kernels()
                .map(|(_, kernel)| kernel.locals().to_vec())
                .collect(),
        )
    }

    /// Converts a closed executable into the narrower artifact accepted by
    /// structural child import. Root arguments/results require a separate
    /// binding contract and root publications require ABI composition, neither
    /// of which this operation pretends to provide.
    pub fn into_importable(self) -> Result<ImportableExecutableIr<B>, ImportabilityError> {
        if let Some((allocation, _)) =
            self.storage
                .allocations()
                .iter()
                .enumerate()
                .find(|(_, allocation)| {
                    matches!(
                        allocation.kind,
                        GlobalBufferKind::Argument { .. } | GlobalBufferKind::Result { .. }
                    )
                })
        {
            return Err(ImportabilityError::RootAllocation {
                allocation: allocation as u32,
            });
        }
        if !self.storage.result_views().is_empty() {
            return Err(ImportabilityError::ResultPublications);
        }
        Ok(ImportableExecutableIr(self))
    }

    /// Applies the one post-close schedule transformation currently required
    /// by universal native qualification. The consuming API keeps all tables
    /// together, rejects duplicate rewrites, and revalidates every schedule,
    /// kernel, and storage reference before returning the new executable.
    ///
    /// This is deliberately narrower than exposing a mutable schedule. Once
    /// universal chunk selection moves before structural close, this method can
    /// be removed without changing the executable owner's shape.
    pub fn chunk_semantic_launches(
        mut self,
        arena: &mut ExprArena,
        chunks: impl IntoIterator<Item = (LaunchId, NatExpr)>,
    ) -> Self {
        let mut rewritten = vec![false; self.schedule.launches().len()];
        for (launch, maximum_grid_x) in chunks {
            let ordinal = launch.index() as usize;
            assert!(
                ordinal < rewritten.len(),
                "chunked launch is outside this executable"
            );
            assert!(!rewritten[ordinal], "launch was chunked more than once");
            rewritten[ordinal] = true;
            self.schedule
                .chunk_semantic_launch(arena, launch, maximum_grid_x);
        }
        self.assert_coordinated();
        self
    }

    fn assert_coordinated(&self) {
        assert_eq!(
            self.schedule.owner(),
            self.owner,
            "schedule owner differs from executable owner"
        );
        assert_eq!(
            self.storage.owner(),
            self.owner,
            "storage owner differs from executable owner"
        );
        assert_eq!(
            internals::arena_owner(&self.kernels),
            self.owner,
            "kernel owner differs from executable owner"
        );
        for launch in self.schedule.launches() {
            self.kernels.kernel(launch.kernel);
        }
        for (view, _) in self.schedule.direct_view_uses() {
            self.storage.view(*view);
        }
        for (_, kernel) in self.kernels.kernels() {
            for binding in &kernel.interface().bindings {
                self.storage.view(binding.view);
            }
        }
    }
}

fn local_allocations<B: KernelDialect>(
    owner: OwnerToken,
    kernels: &[Kernel<B>],
) -> LocalAllocationTopology {
    LocalAllocationTopology::new(
        owner,
        kernels
            .iter()
            .map(|kernel| kernel.locals().to_vec())
            .collect(),
    )
}

fn reuse_compatible(
    storage: &TopologyBuilder,
    left_id: GlobalAllocationId,
    right_id: GlobalAllocationId,
) -> bool {
    if storage.allocation_alignment(left_id) != storage.allocation_alignment(right_id) {
        return false;
    }
    let representations = |allocation: GlobalAllocationId| {
        let mut values: Vec<_> = storage
            .views()
            .iter()
            .filter_map(|view| (view.allocation == allocation).then_some(view.representation))
            .collect();
        values.sort();
        values.dedup();
        values
    };
    representations(left_id) == representations(right_id)
}
fn lifetimes_interfere(left: &AllocationLiveness, right: &AllocationLiveness) -> bool {
    if left
        .uses()
        .iter()
        .all(|a| right.uses().iter().all(|b| mutually_exclusive(a, b)))
    {
        return false;
    }
    let mut left_points: Vec<_> = left.uses().iter().map(schedule_point).collect();
    let mut right_points: Vec<_> = right.uses().iter().map(schedule_point).collect();
    left_points.sort();
    right_points.sort();
    !(left_points.last() < right_points.first() || right_points.last() < left_points.first())
}

fn schedule_point(at: &crate::storage::ScheduleUse) -> Vec<u32> {
    let mut point = Vec::with_capacity(at.region.len() + 1);
    for edge in &at.region {
        let parent_ordinal = match edge {
            crate::storage::ScheduleRegionEdge::IfThen { parent_ordinal, .. }
            | crate::storage::ScheduleRegionEdge::IfElse { parent_ordinal, .. }
            | crate::storage::ScheduleRegionEdge::RepeatBody { parent_ordinal, .. }
            | crate::storage::ScheduleRegionEdge::ChooseOption { parent_ordinal, .. }
            | crate::storage::ScheduleRegionEdge::Imported { parent_ordinal, .. } => {
                *parent_ordinal
            }
        };
        point.push(parent_ordinal);
    }
    point.push(at.ordinal);
    point
}

fn mutually_exclusive(a: &crate::storage::ScheduleUse, b: &crate::storage::ScheduleUse) -> bool {
    for (left, right) in a.region.iter().zip(&b.region) {
        use crate::storage::ScheduleRegionEdge::*;
        match (left, right) {
            (IfThen { node: a, .. }, IfElse { node: b, .. })
            | (IfElse { node: a, .. }, IfThen { node: b, .. })
                if a == b =>
            {
                return true;
            }
            (
                ChooseOption {
                    node: a, value: av, ..
                },
                ChooseOption {
                    node: b, value: bv, ..
                },
            ) if a == b && av != bv => return true,
            _ if left != right => return false,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{DenseF16, DenseF32};
    use crate::target::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    // Deliberately no Backend implementation, native handle, or executor.
    impl KernelDialect for Dialect {
        const NAME: seismic_lang::registry::BackendName = seismic_lang::registry::BackendName::Cpu;
        type Facts = ();
        type Intrinsic = NoIntrinsic;
        fn write_intrinsic_identity(op: &NoIntrinsic, _: &mut IntrinsicIdentityBuilder) {
            match *op {}
        }
        fn intrinsic_numerics(
            _: &(),
            _: &seismic_lang::registry::IntrinsicSignature,
            op: &NoIntrinsic,
        ) -> IntrinsicNumericalSemantics {
            match *op {}
        }
        fn intrinsic_addressable_resources(
            op: &NoIntrinsic,
        ) -> Vec<crate::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }

    fn filled(arena: &mut ExprArena) -> ExecutableIr<Dialect> {
        let mut c = Construction::<Dialect>::new(arena, vec![], false, 0);
        let n = arena.nat(8);
        let (_, index) =
            c.storage_mut()
                .tensor(arena, GlobalBufferKind::Arena, DenseF32::id(), vec![n]);
        let view = c.typed_view::<DenseF32>(c.view(index, DenseF32::id()));
        let mut schedule = c.schedule(arena, 0);
        schedule.fill_zero(view);
        let token = schedule.close();
        let analyzed = c.close(token).analyze_allocations();
        assert_eq!(analyzed.arena_allocations().len(), 1);
        analyzed
            .apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish()
    }
    #[test]
    fn importing_real_storage_preserves_schedule_uses_and_reuse_choices() {
        let mut arena = ExprArena::default();
        let child = filled(&mut arena).into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let imported = parent.import(&mut arena, child, &[]);
        let mut builder = parent.schedule(&mut arena, 0);
        builder.splice(None, vec![(0, imported)]);
        let token = builder.close();
        let analyzed = parent.close(token).analyze_allocations();
        assert_eq!(analyzed.arena_allocations().len(), 1);
        assert_eq!(analyzed.liveness()[0].uses().len(), 1);
        assert_eq!(analyzed.liveness()[0].uses()[0].region().len(), 1);
        let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
        let executable = planned.finish();
        assert_eq!(executable.storage().views().len(), 1);
    }
    #[test]
    fn allocation_analysis_exposes_policy_neutral_reuse_facts() {
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let n = arena.nat(8);
        let (_, first_index) = construction.storage_mut().tensor(
            &mut arena,
            GlobalBufferKind::Arena,
            DenseF32::id(),
            vec![n],
        );
        let (_, second_index) = construction.storage_mut().tensor(
            &mut arena,
            GlobalBufferKind::Arena,
            DenseF32::id(),
            vec![n],
        );
        let first =
            construction.typed_view::<DenseF32>(construction.view(first_index, DenseF32::id()));
        let second =
            construction.typed_view::<DenseF32>(construction.view(second_index, DenseF32::id()));
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_zero(first);
        schedule.fill_zero(second);
        let token = schedule.close();
        let analyzed = construction.close(token).analyze_allocations();

        assert_eq!(analyzed.arena_allocations().len(), 2);
        assert_eq!(analyzed.allocation_relations().len(), 1);
        let relation = analyzed.allocation_relations()[0];
        assert!(relation.storage_compatible());
        assert!(!relation.lifetimes_interfere());
        assert!(relation.may_share_slot());

        let allocations = analyzed.arena_allocations().to_vec();
        let domain = seismic_lang::expr::FiniteDomain::new(vec![0, 1])
            .expect("nonempty physical-slot domain");
        let first = arena.decision(domain.clone());
        let second = arena.decision(domain);
        let planned = analyzed.apply_allocation_plan(
            &mut arena,
            AllocationPlan::new([
                AllocationSlotChoice::new(allocations[0], first),
                AllocationSlotChoice::new(allocations[1], second),
            ]),
        );
        assert!(planned.mandatory_constraints().is_empty());
        assert_eq!(planned.slots().iter().flatten().count(), 2);
        let executable = planned.finish();
        assert_eq!(executable.storage().allocations().len(), 2);
    }
    #[test]
    fn external_reuse_mapping_retains_mandatory_legality_constraints() {
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let n = arena.nat(8);
        let (_, first_index) = construction.storage_mut().tensor(
            &mut arena,
            GlobalBufferKind::Arena,
            DenseF32::id(),
            vec![n],
        );
        let (_, second_index) = construction.storage_mut().tensor(
            &mut arena,
            GlobalBufferKind::Arena,
            DenseF16::id(),
            vec![n],
        );
        let first =
            construction.typed_view::<DenseF32>(construction.view(first_index, DenseF32::id()));
        let second =
            construction.typed_view::<DenseF16>(construction.view(second_index, DenseF16::id()));
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_zero(first);
        schedule.fill_zero(second);
        let token = schedule.close();
        let analyzed = construction.close(token).analyze_allocations();
        let candidates = analyzed.arena_allocations().to_vec();
        assert!(!analyzed.allocation_relations()[0].storage_compatible());

        let domain = seismic_lang::expr::FiniteDomain::new(vec![0, 1])
            .expect("nonempty physical-slot domain");
        let left = arena.decision(domain.clone());
        let right = arena.decision(domain);
        let planned = analyzed.apply_allocation_plan(
            &mut arena,
            AllocationPlan::new([
                AllocationSlotChoice::new(candidates[0], left),
                AllocationSlotChoice::new(candidates[1], right),
            ]),
        );
        assert_eq!(planned.mandatory_constraints().len(), 2);
        let executable = planned.finish();
        assert_eq!(executable.allocation_constraints().len(), 2);
    }
    #[test]
    fn imported_schedule_cannot_be_spliced_into_another_owner() {
        let mut arena = ExprArena::default();
        let child = filled(&mut arena).into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let imported = parent.import(&mut arena, child, &[]);
        let mut unrelated = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            unrelated
                .schedule(&mut arena, 0)
                .splice(None, vec![(0, imported)]);
        }))
        .is_err());
    }
    #[test]
    fn construction_rejects_foreign_views() {
        let mut arena = ExprArena::default();
        let executable = filled(&mut arena);
        let c = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            c.assert_view(executable.storage().view_id(0))
        }))
        .is_err());
    }
    #[test]
    fn result_publication_is_not_structurally_importable() {
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let n = arena.nat(8);
        let (_, index) = construction.storage_mut().tensor(
            &mut arena,
            GlobalBufferKind::Arena,
            DenseF32::id(),
            vec![n],
        );
        let view = construction.view(index, DenseF32::id());
        construction
            .storage_mut()
            .publish_result(&mut arena, view, vec![0]);
        let typed = construction.typed_view::<DenseF32>(view);
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_zero(typed);
        let token = schedule.close();
        let analyzed = construction.close(token).analyze_allocations();
        let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
        assert_eq!(
            planned.finish().into_importable().err(),
            Some(ImportabilityError::ResultPublications)
        );
    }
    #[test]
    fn dynamic_places_cannot_cross_kernel_boundaries() {
        let mut arena = ExprArena::default();
        let mut c = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let n = arena.nat(1);
        let (_, index) =
            c.storage_mut()
                .tensor(&mut arena, GlobalBufferKind::Arena, DenseF32::id(), vec![n]);
        let view = c.view(index, DenseF32::id());
        let vectors = VectorSupport::default();
        let mut first = c.portable_kernel(&mut arena, &(), &[], &vectors);
        let place = first.arg_view(view, false);
        first.close();
        assert!(catch_unwind(AssertUnwindSafe(|| {
            let mut second = c.portable_kernel(&mut arena, &(), &[], &vectors);
            let _ = second.arg_view(view, false);
            let index = second.index_constant(0);
            second.read(place, &[index]);
        }))
        .is_err());
    }
    #[test]
    fn intrinsic_scalar_metadata_preserves_boolean_and_index_categories() {
        use crate::kernel::ops::{ConstantValue, ValueType};
        use seismic_lang::types::DType;
        let mut arena = ExprArena::default();
        let mut c = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = c.portable_kernel(&mut arena, &(), &[], &vectors);
        let value = builder.constant(ConstantValue::Bool(true), ValueType::Bool);
        let scalar = builder.semantic_scalar(value, DType::Bool);
        assert_eq!(scalar.dtype(), DType::Bool);
        assert!(!scalar.index());
        let index = builder.index_constant(3);
        assert!(builder.semantic_scalar(index, DType::U32).index());
        assert!(catch_unwind(AssertUnwindSafe(
            || builder.semantic_scalar(index, DType::F32)
        ))
        .is_err());
    }
}
