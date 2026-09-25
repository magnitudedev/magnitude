//! Owns construction identity and the coordinated storage/kernel/schedule tables.
//! Child import rewrites all scoped references together; callers never mint or
//! rebrand handles. Semantic refinement and target selection stay with the compiler.

use crate::identity::OwnerToken;
use crate::kernel::{internals, Kernel, KernelArena};
use crate::schedule::{
    AnyScalarSlot, ClosedSchedule, ImportedBindings, ParametricSchedule, ScheduleBuilder,
    ScheduleConstruction,
};
use crate::storage::{AllocationAcquisition, GlobalBufferKind};
use crate::storage::{
    AllocationLiveness, AnyBufferView, GlobalAllocationId, GlobalAllocationTopology,
    LocalAllocationTopology, TopologyBuilder,
};
use crate::physical_target::{AddressableResourceClass, PhysicalDialect, VectorSupport};
use seismic_lang::expr::{BoolExpr, DecisionId, ExprArena, NatExpr};
use seismic_lang::ids::{ParameterId, RepresentationId};

/// An unfinished structured repeat. Its constructor owns all header/result
/// destinations; finishing consumes the actual body product exactly once.
pub struct RepeatConstruction {
    body: u32,
    binding: crate::schedule::LoopBinding,
    initial: crate::region::Product<crate::region::ValueOperand>,
    header: crate::region::Product<crate::region::ValueDestination>,
    result: crate::region::Product<crate::region::ValueDestination>,
}
impl RepeatConstruction {
    pub fn body(&self) -> u32 {
        self.body
    }
    pub fn binding(&self) -> crate::schedule::LoopBinding {
        self.binding
    }
    pub fn header(&self) -> &crate::region::Product<crate::region::ValueDestination> {
        &self.header
    }
}

pub struct Construction<B: PhysicalDialect> {
    owner: OwnerToken,
    storage: TopologyBuilder,
    kernels: Vec<Kernel<B>>,
    kernel_state: internals::KernelState<B>,
    schedule: ScheduleConstruction<B>,
}

impl<B: PhysicalDialect> Construction<B> {
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
            &self.storage,
            &self.schedule,
            &mut self.kernels,
            &mut self.kernel_state,
            facts,
            resources,
            vectors,
        )
    }
    /// Reborrow the same unfinished kernel after a construction pause.
    pub fn resume_portable_kernel<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        facts: &'a B::Facts,
        resources: &'a [AddressableResourceClass],
        vectors: &'a VectorSupport,
        cursor: internals::PortableCursor,
    ) -> internals::PortableBuilder<'a, B> {
        internals::resume_portable(
            self.owner,
            arena,
            &self.storage,
            &self.schedule,
            &mut self.kernels,
            &mut self.kernel_state,
            facts,
            resources,
            vectors,
            cursor,
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
    pub fn may_overlap_views(&self, a: AnyBufferView, b: AnyBufferView) -> bool {
        self.storage.may_overlap_views(&self.schedule, a, b)
    }

    pub fn bind_argument_tensor(&mut self, arena: &mut ExprArena, source: AnyBufferView) -> AnyBufferView {
        let layout = self.storage.view_layout(source).clone();
        assert!(matches!(layout.base, crate::storage::ViewBase::Allocation(id)
            if matches!(self.storage.allocation_kind(id), GlobalBufferKind::Argument { .. })));
        let mut values = |rank| (0..rank).map(|_| {
            let slot = self.schedule.quantity_slot(arena, crate::schedule::HostQuantityKind::Natural);
            arena.nat_symbol(slot.symbol())
        }).collect::<Vec<_>>();
        // Admission already validates argument axes against this instantiated contract.
        // Only the physical strides are discovered from the actual backing.
        let strides = values(layout.strides.len());
        let result = self.storage.region_view(arena, layout.representation, layout.extents, strides);
        self.schedule(arena, 0).bind_argument_tensor(source, result);
        result
    }

    /// Define one stored tensor value only on successful reached acquisition.
    pub fn begin_tensor_instance(&mut self, arena: &mut ExprArena, region: u32, source: AnyBufferView) -> AnyBufferView {
        assert!(matches!(self.storage.view_layout(source).mapping, crate::storage::ViewMapping::WholeAllocation),
            "reached tensor creation requires allocation-owned geometry");
        // The instance's geometry is its allocation's geometry: the same
        // extent and stride expressions, evaluated where the instance is
        // acquired. Every operand of those expressions is defined once per
        // visit of its region (call values, target constants, decisions,
        // loop binders, and single-assignment slots), and a tensor value only
        // leaves the visit that acquired it through a branch or repeat product,
        // whose destination owns fresh geometry slots. So the expressions keep
        // their acquired values for as long as this instance is in scope, and
        // every obligation over the instance is stated over its source.
        let result = self.storage.instance_view(arena, source);
        self.schedule(arena, region).begin_allocation_instance(source, result);
        result
    }

    fn value_destinations(&mut self, arena: &mut ExprArena, source: &crate::region::Product<crate::region::ValueOperand>) -> crate::region::Product<crate::region::ValueDestination> {
        use crate::region::{ValueDestination, ValueOperand};
        source.map(&mut |value| match *value {
                ValueOperand::Scalar(value) => {
                    ValueDestination::Scalar(self.schedule.slot_any(arena, value.kind()))
                }
                ValueOperand::Quantity(value) => {
                    ValueDestination::Quantity(self.schedule.quantity_slot(arena, value.kind()))
                }
                ValueOperand::Tensor(view) => {
                    let layout = self.storage.view_layout(view).clone();
                    let mut geometry = |rank: usize| (0..rank).map(|_| {
                        let slot = self.schedule.quantity_slot(arena, crate::schedule::HostQuantityKind::Natural);
                        arena.nat_symbol(slot.symbol())
                    }).collect::<Vec<_>>();
                    let extents = geometry(layout.extents.len());
                    let strides = geometry(layout.strides.len());
                    ValueDestination::Tensor(self.storage.region_view(
                        arena,
                        layout.representation,
                        extents,
                        strides,
                    ))
                }
            })
    }

    pub fn begin_value_repeat(
        &mut self,
        arena: &mut ExprArena,
        parent: u32,
        start: NatExpr,
        end: NatExpr,
        visits: crate::schedule::RepeatVisits,
        initial: crate::region::Product<crate::region::ValueOperand>,
    ) -> RepeatConstruction {
        if visits == crate::schedule::RepeatVisits::Independent {
            let mut carried = 0usize;
            initial.visit(&mut |_| carried += 1);
            assert_eq!(carried, 0, "independent visits carry no products");
        }
        let (body, binding) = self.schedule.begin_repeat(arena, parent, start, end, visits);
        self.schedule.open_repeat_products(body);
        let header = self.value_destinations(arena, &initial);
        let result = self.value_destinations(arena, &initial);
        RepeatConstruction {
            body,
            binding,
            initial,
            header,
            result,
        }
    }

    pub fn finish_value_branch(
        &mut self, arena: &mut ExprArena, parent: u32, then_region: u32,
        then_values: crate::region::Product<crate::region::ValueOperand>,
        else_values: crate::region::Product<crate::region::ValueOperand>,
    ) -> crate::region::Product<crate::region::ValueDestination> {
        use crate::region::{BranchResult, Product, ValueDestination, ValueOperand};
        fn combine(a: Product<ValueOperand>, b: Product<ValueOperand>, destination: Product<ValueDestination>, storage: &TopologyBuilder) -> Product<BranchResult> {
            match (a, b, destination) {
                (Product::Unit, Product::Unit, Product::Unit) => Product::Unit,
                (Product::Leaf(a), Product::Leaf(b), Product::Leaf(result)) => {
                    match (a,b) {
                        (ValueOperand::Tensor(a),ValueOperand::Tensor(b)) => {
                            assert_eq!(a.representation(),b.representation(),"branch tensor representation changed");
                            assert_eq!(storage.view_layout(a).extents.len(), storage.view_layout(b).extents.len(),"branch tensor rank changed");
                        }
                        (ValueOperand::Scalar(a),ValueOperand::Scalar(b)) => assert_eq!(a.kind(),b.kind(),"branch scalar kind changed"),
                        (ValueOperand::Quantity(a),ValueOperand::Quantity(b)) => assert_eq!(a.kind(),b.kind(),"branch quantity kind changed"),
                        _ => panic!("branch result leaf kind changed"),
                    }
                    Product::Leaf(BranchResult { then_value: a, else_value: b, result })
                }
                (Product::Range(a,b),Product::Range(c,d),Product::Range(e,f)) => Product::Range(Box::new(combine(*a,*c,*e,storage)),Box::new(combine(*b,*d,*f,storage))),
                (Product::Tuple(a),Product::Tuple(b),Product::Tuple(c)) => {
                    assert_eq!(a.len(),b.len()); assert_eq!(a.len(),c.len());
                    Product::Tuple(a.into_iter().zip(b).zip(c).map(|((a,b),c)|combine(a,b,c,storage)).collect())
                }
                _ => panic!("branch result product changed shape"),
            }
        }
        let destination = self.value_destinations(arena, &then_values);
        let products = combine(then_values, else_values, destination.clone(), &self.storage);
        self.schedule.finish_branch_products(parent, then_region, products);
        destination
    }

    pub fn finish_value_repeat(
        &mut self,
        repeat: RepeatConstruction,
        backedge: crate::region::Product<crate::region::ValueOperand>,
    ) -> crate::region::Product<crate::region::ValueDestination> {
        use crate::region::{Product, RepeatCarry, ValueDestination, ValueOperand};
        fn combine(
            initial: Product<ValueOperand>,
            header: Product<ValueDestination>,
            backedge: Product<ValueOperand>,
            result: Product<ValueDestination>,
            storage: &TopologyBuilder,
        ) -> Product<RepeatCarry> {
            match (initial, header, backedge, result) {
                (Product::Unit, Product::Unit, Product::Unit, Product::Unit) => Product::Unit,
                (
                    Product::Leaf(initial),
                    Product::Leaf(header),
                    Product::Leaf(backedge),
                    Product::Leaf(result),
                ) => {
                    match (initial, header, backedge, result) {
                        (
                            ValueOperand::Scalar(a),
                            ValueDestination::Scalar(h),
                            ValueOperand::Scalar(b),
                            ValueDestination::Scalar(r),
                        ) => {
                            assert_eq!(a.kind(), b.kind(), "repeat scalar kind changed");
                            assert_eq!(a.kind(), h.kind());
                            assert_eq!(a.kind(), r.kind());
                        }
                        (
                            ValueOperand::Quantity(a),
                            ValueDestination::Quantity(h),
                            ValueOperand::Quantity(b),
                            ValueDestination::Quantity(r),
                        ) => {
                            assert_eq!(a.kind(), b.kind(), "repeat quantity kind changed");
                            assert_eq!(a.kind(), h.kind());
                            assert_eq!(a.kind(), r.kind());
                        }
                        (
                            ValueOperand::Tensor(a),
                            ValueDestination::Tensor(_),
                            ValueOperand::Tensor(b),
                            ValueDestination::Tensor(_),
                        ) => {
                            assert_eq!(
                                a.representation(),
                                b.representation(),
                                "repeat tensor representation changed"
                            );
                            assert_eq!(
                                storage.view_layout(a).extents.len(),
                                storage.view_layout(b).extents.len(),
                                "repeat tensor rank changed"
                            );
                        }
                        _ => panic!("repeat leaf kind changed"),
                    }
                    Product::Leaf(RepeatCarry {
                        initial,
                        header,
                        backedge,
                        result,
                    })
                }
                (
                    Product::Range(a, b),
                    Product::Range(c, d),
                    Product::Range(e, f),
                    Product::Range(g, h),
                ) => Product::Range(
                    Box::new(combine(*a, *c, *e, *g, storage)),
                    Box::new(combine(*b, *d, *f, *h, storage)),
                ),
                (Product::Tuple(a), Product::Tuple(b), Product::Tuple(c), Product::Tuple(d)) => {
                    assert_eq!(a.len(), b.len());
                    assert_eq!(a.len(), c.len());
                    assert_eq!(a.len(), d.len());
                    Product::Tuple(
                        a.into_iter()
                            .zip(b)
                            .zip(c)
                            .zip(d)
                            .map(|(((a, b), c), d)| combine(a, b, c, d, storage))
                            .collect(),
                    )
                }
                _ => panic!("repeat result product changed shape"),
            }
        }
        let result = repeat.result.clone();
        let carries = combine(
            repeat.initial,
            repeat.header,
            backedge,
            repeat.result,
            &self.storage,
        );
        self.schedule.finish_repeat_products(repeat.body, carries);
        result
    }

    pub fn import(
        &mut self,
        arena: &mut ExprArena,
        region: u32,
        child: ImportableExecutableIr<B>,
        forced_slots: &[(u32, AnyScalarSlot)],
    ) -> ImportedBindings {
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
        self.schedule.append_imported_at(region, imported)
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        local_allocations(&self.kernels)
    }
    /// Closes every mutable IR table as one consuming transition. The returned
    /// phase has no mutation API, so closed storage, kernels, and schedule
    /// cannot subsequently diverge.
    ///
    /// ```compile_fail
    /// use seismic_ir::{construction::SealedConstruction, target::PhysicalDialect};
    /// fn cannot_mutate<B: PhysicalDialect>(mut closed: SealedConstruction<B>) {
    ///     closed.storage_mut();
    /// }
    /// ```
    pub fn close(self, token: ClosedSchedule) -> SealedConstruction<B> {
        self.kernel_state.assert_closed();
        let schedule = self.schedule.finish(token);
        SealedConstruction {
            owner: self.owner,
            storage: self.storage,
            kernels: self.kernels,
            schedule,
        }
    }
}

/// Structurally closed executable tables, before schedule-derived allocation
/// facts have been calculated.
pub struct SealedConstruction<B: PhysicalDialect> {
    owner: OwnerToken,
    storage: TopologyBuilder,
    kernels: Vec<Kernel<B>>,
    schedule: ParametricSchedule<B>,
}

impl<B: PhysicalDialect> SealedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        &self.storage
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        &self.kernels
    }
    pub fn schedule(&self) -> &ParametricSchedule<B> {
        &self.schedule
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        local_allocations(&self.kernels)
    }

    /// Establish bounded semantic launch scopes before deriving liveness,
    /// scratch allocation, resource predicates, or executable identity. Imported
    /// child launches already own their scopes and are never rewritten twice.
    pub fn normalize_launches(
        mut self,
        arena: &mut ExprArena,
        maximum_grid_x: u64,
        natural_bits: u32,
    ) -> Result<NormalizedConstruction<B>, crate::schedule::LaunchNormalizationError> {
        if maximum_grid_x == 0 || !(1..=64).contains(&natural_bits) {
            return Err(crate::schedule::LaunchNormalizationError::InvalidTargetLimits);
        }
        let launches = self
            .schedule
            .launches()
            .iter()
            .enumerate()
            .filter(|(_, launch)| launch.logical_base.is_some())
            .map(|(ordinal, _)| self.schedule.launch_id(ordinal as u32))
            .collect::<Vec<_>>();
        for launch in launches {
            self.schedule
                .chunk_semantic_launch(arena, launch, maximum_grid_x, natural_bits)?;
        }
        Ok(NormalizedConstruction { sealed: self })
    }
}

/// The same construction after launch normalization. Only this phase can derive
/// allocation facts; no subsequent phase can rewrite its schedule.
///
/// ```compile_fail
/// use seismic_ir::{construction::SealedConstruction, target::PhysicalDialect};
/// fn cannot_skip_normalization<B: PhysicalDialect>(sealed: SealedConstruction<B>) {
///     sealed.analyze_allocations();
/// }
/// ```
///
/// ```compile_fail
/// use seismic_ir::{construction::NormalizedConstruction, target::PhysicalDialect};
/// use seismic_lang::expr::ExprArena;
/// fn cannot_normalize_twice<B: PhysicalDialect>(normalized: NormalizedConstruction<B>, arena: &mut ExprArena) {
///     normalized.normalize_launches(arena, 8, 64).unwrap();
/// }
/// ```
pub struct NormalizedConstruction<B: PhysicalDialect> {
    sealed: SealedConstruction<B>,
}

impl<B: PhysicalDialect> NormalizedConstruction<B> {
    /// Derives policy-neutral lifetime and compatibility facts. This consumes
    /// the closed construction so a later physical plan can only be applied to
    /// the exact tables from which these facts were derived.
    pub fn analyze_allocations(self) -> AnalyzedConstruction<B> {
        let mut sealed = self.sealed;
        fn publications(steps: &[crate::schedule::ScheduleStep], output: &mut Vec<crate::storage::ResultViewPublication>) {
            use crate::schedule::ScheduleStep;
            for step in steps {
                match step {
                    ScheduleStep::PublishTensor { view, path, bytes, .. } => output.push(crate::storage::ResultViewPublication {
                        view: *view, path: path.clone(), bytes: *bytes,
                    }),
                    ScheduleStep::Imported { body, .. } | ScheduleStep::Repeat { body, .. } => publications(body, output),
                    ScheduleStep::If { then_steps, else_steps, .. } => { publications(then_steps, output); publications(else_steps, output); }
                    _ => {}
                }
            }
        }
        let mut result_views = Vec::new();
        publications(sealed.schedule.steps(), &mut result_views);
        sealed.storage.set_result_publications(result_views);
        let mut uses = vec![Vec::new(); sealed.storage.allocation_count() as usize];
        for (view, at) in sealed.schedule.direct_view_uses() {
            for allocation in sealed
                .schedule
                .backing_allocations(sealed.storage.views(), *view)
            {
                uses[allocation.index() as usize].push(at.clone());
            }
        }
        for (launch_id, at) in sealed.schedule.launch_uses() {
            let launch = sealed.schedule.launch(*launch_id);
            let kernel = sealed
                .kernels
                .get(launch.kernel.index() as usize)
                .expect("launch kernel was closed into this construction");
            for binding in &kernel.interface().bindings {
                for allocation in sealed
                    .schedule
                    .backing_allocations(sealed.storage.views(), binding.view)
                {
                    uses[allocation.index() as usize].push(at.clone());
                }
            }
        }
        let liveness = uses
            .into_iter()
            .map(AllocationLiveness::new)
            .collect::<Vec<_>>();
        let instances = sealed.schedule.allocation_instances(
            sealed.storage.views(),
            sealed.storage.allocation_count() as usize,
        );
        fn definitions(steps: &[crate::schedule::ScheduleStep], views: &[crate::storage::BufferViewLayout], reached: &mut std::collections::BTreeSet<u32>) {
            use crate::schedule::ScheduleStep;
            for step in steps {
                match step {
                    ScheduleStep::BeginAllocationInstance { source: view, .. } => {
                        if let crate::storage::ViewBase::Allocation(id) = views[view.index() as usize].base {
                            reached.insert(id.index());
                        }
                    }
                    ScheduleStep::Imported { body, .. } | ScheduleStep::Repeat { body, .. } => definitions(body, views, reached),
                    ScheduleStep::If { then_steps, else_steps, .. } => {
                        definitions(then_steps, views, reached); definitions(else_steps, views, reached);
                    }
                    _ => {}
                }
            }
        }
        let mut reached = std::collections::BTreeSet::new();
        definitions(sealed.schedule.steps(), sealed.storage.views(), &mut reached);
        let published = sealed.storage.result_views().iter().flat_map(|publication| {
            sealed.schedule.backing_allocations(sealed.storage.views(), publication.view)
        }).map(|allocation| allocation.index()).collect::<std::collections::BTreeSet<_>>();
        let acquisitions = (0..sealed.storage.allocation_count()).map(|index| {
            let allocation = GlobalAllocationId::new(sealed.owner, index);
            if matches!(sealed.storage.allocation_kind(allocation), GlobalBufferKind::Arena)
                && reached.contains(&index) {
                AllocationAcquisition::Reached
            } else {
                AllocationAcquisition::Invocation
            }
        }).collect::<Vec<_>>();
        let candidates: Vec<_> = (0..sealed.storage.allocation_count())
            .map(|index| GlobalAllocationId::new(sealed.owner, index))
            .filter(|id| {
                matches!(sealed.storage.allocation_kind(*id), GlobalBufferKind::Arena)
                    && instances[id.index() as usize] == 1
            })
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
                    storage_compatible: acquisitions[left.index() as usize] == acquisitions[right.index() as usize]
                        && reuse_compatible(&sealed.storage, left, right),
                    lifetimes_interfere: published.contains(&left.index()) || published.contains(&right.index()) || lifetimes_interfere(
                        &liveness[left.index() as usize],
                        &liveness[right.index() as usize],
                    ),
                });
            }
        }
        AnalyzedConstruction {
            closed: sealed,
            liveness,
            instances,
            acquisitions,
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
pub struct AnalyzedConstruction<B: PhysicalDialect> {
    closed: SealedConstruction<B>,
    liveness: Vec<AllocationLiveness>,
    instances: Vec<u32>,
    acquisitions: Vec<AllocationAcquisition>,
    arena_allocations: Vec<GlobalAllocationId>,
    relations: Vec<AllocationRelation>,
}
impl<B: PhysicalDialect> AnalyzedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        self.closed.storage()
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        self.closed.kernels()
    }
    pub fn schedule(&self) -> &ParametricSchedule<B> {
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
            assert_eq!(
                self.instances[allocation.index() as usize],
                1,
                "multi-instance allocation cannot share a reuse slot"
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
pub struct StoragePlannedConstruction<B: PhysicalDialect> {
    analyzed: AnalyzedConstruction<B>,
    slots: Vec<Option<DecisionId>>,
    mandatory_constraints: Vec<BoolExpr>,
}
impl<B: PhysicalDialect> StoragePlannedConstruction<B> {
    pub fn storage(&self) -> &TopologyBuilder {
        self.analyzed.storage()
    }
    pub fn kernels(&self) -> &[Kernel<B>] {
        self.analyzed.kernels()
    }
    pub fn schedule(&self) -> &ParametricSchedule<B> {
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
        let SealedConstruction {
            owner,
            storage,
            kernels,
            schedule,
        } = self.analyzed.closed;
        let executable = ExecutableIr {
            owner,
            storage: storage.close(self.analyzed.liveness, self.slots, self.analyzed.instances, self.analyzed.acquisitions),
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
pub struct ExecutableIr<B: PhysicalDialect> {
    owner: OwnerToken,
    storage: GlobalAllocationTopology,
    kernels: KernelArena<B>,
    schedule: ParametricSchedule<B>,
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
pub struct ImportableExecutableIr<B: PhysicalDialect>(ExecutableIr<B>);

impl<B: PhysicalDialect> ExecutableIr<B> {
    /// Derives all launch resources from this exact normalized structure and
    /// immutable target rules. No independently assembled resource tables enter closure.
    pub fn close_execution(
        mut self,
        arena: &mut ExprArena,
        policy: crate::physical_target::LocalRealizationPolicy,
        abi: &impl crate::physical_target::KernelAbiModel<B>,
    ) -> crate::execution::ClosedExecutableIr<B> {
        self.storage.derive_reservations(arena, &self.schedule);
        crate::execution::ClosedExecutableIr::new(self, arena, policy, abi)
    }

    pub fn storage(&self) -> &GlobalAllocationTopology {
        &self.storage
    }
    pub fn kernels(&self) -> &KernelArena<B> {
        &self.kernels
    }
    pub fn schedule(&self) -> &ParametricSchedule<B> {
        &self.schedule
    }
    /// Mandatory implications of the attached symbolic allocation slots.
    /// Planning must include these in the executable's hard constraints.
    pub fn allocation_constraints(&self) -> &[BoolExpr] {
        &self.allocation_constraints
    }
    pub fn local_allocations(&self) -> LocalAllocationTopology {
        LocalAllocationTopology::new(
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

fn local_allocations<B: PhysicalDialect>(kernels: &[Kernel<B>]) -> LocalAllocationTopology {
    LocalAllocationTopology::new(
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
            .filter_map(|view| {
                (view.base == crate::storage::ViewBase::Allocation(allocation))
                    .then_some(view.representation)
            })
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
            _ if left != right => return false,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repr::{DenseF16, DenseF32, Representation};
    use crate::physical_target::{IntrinsicIdentityBuilder, IntrinsicNumericalSemantics};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    // Deliberately no Backend implementation, native handle, or executor.
    impl PhysicalDialect for Dialect {
        type LaunchDescriptor = ();
        fn ordinary_launch() -> Self::LaunchDescriptor {
            ()
        }

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

    fn instance_tensor(construction: &mut Construction<Dialect>, arena: &mut ExprArena, extent: u64) -> AnyBufferView {
        let representation = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
        let extent = arena.nat(extent);
        let (_, index) = construction.storage_mut().tensor(arena, GlobalBufferKind::Arena, representation, vec![extent]);
        construction.view(index, representation)
    }

    fn carry_tensor(
        construction: &mut Construction<Dialect>,
        arena: &mut ExprArena,
        offset: u64,
    ) -> AnyBufferView {
        let bytes = arena.nat(64);
        let allocation = construction
            .storage_mut()
            .allocate(GlobalBufferKind::Arena, bytes, 4);
        let offset = arena.nat(offset);
        let extent = arena.nat(4);
        let stride = arena.nat(2);
        let representation = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
        let ordinal = construction.storage_mut().strided_view(
            arena,
            allocation,
            representation,
            offset,
            vec![extent],
            vec![stride],
        );
        construction.view(ordinal, representation)
    }

    #[test]
    fn begin_defines_an_instance_with_its_allocation_geometry() {
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let fixed = arena.nat(2);
        let local = construction.schedule(&mut arena, 0).quantity_slot(crate::schedule::HostQuantityKind::Natural);
        let dynamic = arena.nat_symbol(local.symbol());
        let representation = seismic_lang::registry::dense(seismic_lang::types::DType::F32);
        let (_, index) = construction.storage_mut().tensor(&mut arena, GlobalBufferKind::Arena, representation, vec![fixed, dynamic]);
        let source = construction.view(index, representation);
        let result = construction.begin_tensor_instance(&mut arena, 0, source);
        let (source, layout) = (construction.storage.view_layout(source), construction.storage.view_layout(result));
        // Invocation-fixed and execution-local axes alike: the instance's
        // extents and strides are its allocation's own expressions.
        assert_eq!(layout.extents, [fixed, dynamic]);
        assert_eq!(layout.strides, source.strides);
        assert!(layout.contiguous);
        assert!(matches!(layout.base, crate::storage::ViewBase::TensorValue(_)));
    }

    #[test]
    fn branch_product_retains_both_possible_origins_with_actual_geometry() {
        use crate::region::{Product,ValueOperand,ValueDestination};
        let mut arena=ExprArena::default();
        let mut construction=Construction::<Dialect>::new(&mut arena,vec![],false,0);
        let condition=arena.bool(true);
        let (then_region,else_region)=construction.schedule_state().begin_branch(0,condition);
        let a=instance_tensor(&mut construction,&mut arena,2);
        let b=instance_tensor(&mut construction,&mut arena,5);
        let a=construction.begin_tensor_instance(&mut arena,then_region,a);
        let b=construction.begin_tensor_instance(&mut arena,else_region,b);
        let result=construction.finish_value_branch(&mut arena,0,then_region,Product::Leaf(ValueOperand::Tensor(a)),Product::Leaf(ValueOperand::Tensor(b)));
        let Product::Leaf(ValueDestination::Tensor(result))=result else {panic!("tensor product")};
        assert_ne!(construction.storage.view_layout(a).extents,construction.storage.view_layout(result).extents);
        let close=construction.schedule(&mut arena,0).close();
        let closed=construction.close(close);
        let roots=closed.schedule.backing_allocations(closed.storage.views(),result);
        assert_eq!(roots.len(),2,"both actual arm origins feed the destination");
        let crate::schedule::ScheduleStep::If { results,.. }=&closed.schedule.steps()[0] else {panic!("branch")};
        assert!(matches!(results,Product::Leaf(_)));
    }

    #[test]
    fn repeat_product_preserves_actual_different_backings_and_views() {
        use crate::region::{Product, ValueDestination, ValueOperand};
        use crate::storage::ViewBase;
        for trips in [0, 3] {
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let initial = carry_tensor(&mut construction, &mut arena, 4);
            let next_source = instance_tensor(&mut construction, &mut arena, 4);
            let next = next_source;
            let zero = arena.nat(0);
            let end = arena.nat(trips);
            let repeat = construction.begin_value_repeat(
                &mut arena,
                0,
                zero,
                end,
                crate::schedule::RepeatVisits::Ordered,
                Product::Leaf(ValueOperand::Tensor(initial)),
            );
            let Product::Leaf(ValueDestination::Tensor(header)) = repeat.header() else {
                panic!("tensor header")
            };
            let header = *header;
            assert!(matches!(
                construction.storage().view_layout(header).base,
                ViewBase::TensorValue(_)
            ));
            assert!(
                construction.may_overlap_views(header, next),
                "unfinished recurrence cannot establish disjointness"
            );
            let next = construction.begin_tensor_instance(&mut arena, repeat.body(), next_source);
            let result =
                construction.finish_value_repeat(repeat, Product::Leaf(ValueOperand::Tensor(next)));
            let Product::Leaf(ValueDestination::Tensor(result)) = result else {
                panic!("tensor result")
            };
            let close = construction.schedule(&mut arena, 0).close();
            let closed = construction.close(close);
            let roots = closed
                .schedule
                .backing_allocations(closed.storage.views(), result);
            assert_eq!(roots.len(), 2);
            let instances = closed.schedule.allocation_instances(
                closed.storage.views(),
                closed.storage.allocation_count() as usize,
            );
            let ViewBase::Allocation(next_root) = closed.storage.view_layout(next_source).base else {
                unreachable!()
            };
            assert!(instances[next_root.index() as usize] >= 4,
                "all value locations and simultaneous transfer temporaries are bounded");
            assert!(
                closed
                    .schedule
                    .direct_view_uses()
                    .iter()
                    .any(|(view, at)| *view == next && !at.region.is_empty()),
                "backedge transport is a storage use"
            );

            let crate::schedule::ScheduleStep::Repeat {
                carries: Product::Leaf(carry),
                ..
            } = &closed.schedule.steps()[0]
            else {
                panic!("repeat owns its product")
            };
            assert!(matches!(carry.initial(),ValueOperand::Tensor(value) if value==initial));
            assert!(matches!(carry.backedge(),ValueOperand::Tensor(value) if value==next));
            assert_eq!(closed.storage.view_layout(initial).offset, arena.nat(4));
            assert_eq!(closed.storage.view_layout(next).offset, arena.nat(0));
        }
    }

    #[test]
    #[should_panic(expected = "unfinished repeat product cannot close")]
    fn unfinished_repeat_product_cannot_escape_construction() {
        use crate::region::{Product, ScalarOperand, ValueOperand};
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let zero = arena.nat(0);
        let one = arena.nat(1);
        let _repeat = construction.begin_value_repeat(
            &mut arena,
            0,
            zero,
            one,
            crate::schedule::RepeatVisits::Ordered,
            Product::Leaf(ValueOperand::Scalar(ScalarOperand::Natural(one))),
        );
        let close = construction.schedule(&mut arena, 0).close();
        construction.close(close);
    }

    #[test]
    fn repeat_transports_exact_integer_through_host_slots() {
        use crate::region::{Product, QuantityOperand, ValueDestination, ValueOperand};
        use crate::schedule::{HostEvaluation, HostQuantityKind, HostValueDestination, HostValueExpr, ScheduleStep};
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let max = arena.int(i64::MAX);
        let one = arena.int(1);
        let initial = arena.int_add(max, one);
        let start = arena.nat(0);
        let end = arena.nat(2);
        let repeat = construction.begin_value_repeat(
            &mut arena, 0, start, end, crate::schedule::RepeatVisits::Ordered,
            Product::Leaf(ValueOperand::Quantity(QuantityOperand::Integer(initial))),
        );
        let Product::Leaf(ValueDestination::Quantity(header)) = repeat.header() else { panic!("exact header") };
        assert_eq!(header.kind(), HostQuantityKind::Integer);
        let header_value = arena.int_symbol(header.symbol());
        let next = arena.int_add(header_value, one);
        let next_slot = construction.schedule(&mut arena, repeat.body()).quantity_slot(HostQuantityKind::Integer);
        construction.schedule(&mut arena, repeat.body()).evaluate_host(HostEvaluation {
            value: HostValueExpr::Integer(next),
            to: HostValueDestination::Quantity(next_slot),
            failure: None,
        });
        let backedge = arena.int_symbol(next_slot.symbol());
        let result = construction.finish_value_repeat(repeat, Product::Leaf(ValueOperand::Quantity(QuantityOperand::Integer(backedge))));
        let Product::Leaf(ValueDestination::Quantity(result)) = result else { panic!("exact result") };
        assert_eq!(result.kind(), HostQuantityKind::Integer);
        let close = construction.schedule(&mut arena, 0).close();
        let closed = construction.close(close);
        assert_eq!(closed.schedule.quantity_slots().len(), 3);
        assert!(matches!(&closed.schedule.steps()[0], ScheduleStep::Repeat { body, .. } if matches!(body.as_slice(), [ScheduleStep::EvaluateHost(_)])));
    }

    #[test]
    fn allocation_occurrence_retains_its_reached_definition_scope() {
        use crate::region::Product;
        use crate::schedule::ScheduleStep;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let outside = instance_tensor(&mut construction, &mut arena, 16);
        let inside = instance_tensor(&mut construction, &mut arena, 16);
        construction.begin_tensor_instance(&mut arena, 0, outside);
        let zero = arena.nat(0);
        let repeat = construction.begin_value_repeat(&mut arena, 0, zero, zero, crate::schedule::RepeatVisits::Ordered, Product::Unit);
        construction.begin_tensor_instance(&mut arena, repeat.body(), inside);
        construction.finish_value_repeat(repeat, Product::Unit);
        let close = construction.schedule(&mut arena, 0).close();
        let closed = construction.close(close);
        assert!(
            matches!(closed.schedule.steps()[0],ScheduleStep::BeginAllocationInstance { source: value, .. } if value==outside)
        );
        let ScheduleStep::Repeat {
            body, start, end, ..
        } = &closed.schedule.steps()[1]
        else {
            panic!("repeat")
        };
        assert_eq!(start, end);
        assert!(matches!(body[0],ScheduleStep::BeginAllocationInstance { source: value, .. } if value==inside));
        let views = closed.storage.views();
        let crate::storage::ViewBase::Allocation(outside_root) =
            views[outside.index() as usize].base
        else {
            unreachable!()
        };
        let crate::storage::ViewBase::Allocation(inside_root) = views[inside.index() as usize].base
        else {
            unreachable!()
        };
        let bytes = arena.nat(64);
        let outside_bytes = closed.schedule.allocation_reservation(
            &mut arena,
            views,
            outside_root.index() as usize,
            bytes,
        );
        let inside_bytes = closed.schedule.allocation_reservation(
            &mut arena,
            views,
            inside_root.index() as usize,
            bytes,
        );
        assert_eq!(
            arena
                .eval_nat_u64(outside_bytes, &seismic_lang::expr::Assignment::new())
                .unwrap(),
            64
        );
        assert_eq!(
            arena
                .eval_nat_u64(inside_bytes, &seismic_lang::expr::Assignment::new())
                .unwrap(),
            0
        );
    }

    #[test]
    fn suspended_kernel_preserves_lexical_values_and_branch_state() {
        use crate::kernel::ops::{ConstantValue, ValueType};
        use crate::repr::ScalarKind;
        use seismic_lang::types::DType;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let slot = construction
            .schedule(&mut arena, 0)
            .slot_any(ScalarKind::Scalar(DType::U32));
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let condition = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        let branch = kernel.begin_branch(condition);
        let left = kernel.constant(ConstantValue::U32(41), ValueType::Scalar(DType::U32));
        let cursor = kernel.suspend();
        // The expression owner is available while physical construction waits.
        arena.nat(987);
        let mut kernel =
            construction.resume_portable_kernel(&mut arena, &(), &[], &vectors, cursor);
        kernel.begin_otherwise(&branch);
        let right = kernel.constant(ConstantValue::U32(42), ValueType::Scalar(DType::U32));
        let joined = kernel.finish_branch(branch, vec![left], vec![right])[0];
        let destination = kernel.result_slot(slot);
        kernel.store_slot(destination, joined);
        let kernel_id = kernel.close();
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.launch_sequential(kernel_id);
        let closed = schedule.close();
        let closed = construction.close(closed);
        assert_eq!(closed.kernels.len(), 1);
        assert_eq!(closed.kernels[0].interface().result_slots, vec![slot]);
    }

    #[test]
    #[should_panic(expected = "construction still owns an unfinished kernel")]
    fn suspended_kernel_cannot_be_overwritten_by_a_new_open() {
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let _cursor = construction
            .portable_kernel(&mut arena, &(), &[], &vectors)
            .suspend();
        let _replacement = construction.portable_kernel(&mut arena, &(), &[], &vectors);
    }

    #[test]
    #[should_panic(expected = "kernel cursor belongs to another construction")]
    fn suspended_kernel_cannot_resume_in_a_different_owner() {
        let mut arena = ExprArena::default();
        let mut first = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let mut second = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let cursor = first
            .portable_kernel(&mut arena, &(), &[], &vectors)
            .suspend();
        let _wrong = second.resume_portable_kernel(&mut arena, &(), &[], &vectors, cursor);
    }

    #[test]
    #[should_panic(expected = "loop yielded value exceeds its recurrence uniformity")]
    fn repeat_rejects_initially_uniform_carry_that_becomes_varying() {
        use crate::kernel::ops::{CmpOp, ConstantValue, ValueType};
        use seismic_lang::registry::IntrinsicUniformity;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let zero = kernel.index_constant(0);
        let two = kernel.index_constant(2);
        let yes = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        kernel.repeat(
            zero,
            two,
            vec![yes],
            &[IntrinsicUniformity::Workgroup],
            |body, _, flag| {
                // This is legal on the first iteration only if the recurrence is
                // actually uniform. The backedge must not silently change its type.
                body.branch(
                    flag[0],
                    |body| {
                        body.workgroup_barrier();
                        vec![]
                    },
                    |_| vec![],
                );
                let lane = body.local_id(0);
                vec![body.cmp(CmpOp::Eq, lane, zero)]
            },
        );
        kernel.close();
    }

    #[test]
    #[should_panic(expected = "loop initial value exceeds its recurrence uniformity")]
    fn zero_trip_loop_still_checks_the_initial_carry_uniformity() {
        use seismic_lang::registry::IntrinsicUniformity;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let zero = kernel.index_constant(0);
        let varying = kernel.local_id(0);
        kernel.repeat(
            zero,
            zero,
            vec![varying],
            &[IntrinsicUniformity::Workgroup],
            |_, _, carry| carry,
        );
    }

    #[test]
    fn loop_carries_preserve_the_declared_inductive_uniformity() {
        use crate::kernel::ops::{BinaryOp, ConstantValue, ValueType};
        use seismic_lang::registry::IntrinsicUniformity;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let zero = kernel.index_constant(0);
        let three = kernel.index_constant(3);
        let uniform = kernel.repeat(
            zero,
            three,
            vec![zero],
            &[IntrinsicUniformity::Workgroup],
            |body, binder, carry| vec![body.binary(BinaryOp::Add, carry[0], binder)],
        );
        assert_eq!(
            kernel.uniformity(uniform[0]),
            IntrinsicUniformity::Workgroup
        );
        let yes = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        let varying = kernel.repeat(
            zero,
            three,
            vec![yes],
            &[IntrinsicUniformity::Varying],
            |body, _, _| {
                let lane = body.local_id(0);
                vec![body.cmp(crate::kernel::ops::CmpOp::Eq, lane, zero)]
            },
        );
        assert_eq!(kernel.uniformity(varying[0]), IntrinsicUniformity::Varying);
        kernel.close();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let zero = kernel.index_constant(0);
        let three = kernel.index_constant(3);
        let yes = kernel.constant(ConstantValue::Bool(true), ValueType::Bool);
        kernel.repeat(
            zero,
            three,
            vec![yes],
            &[IntrinsicUniformity::Workgroup],
            |body, _, carry| {
                body.branch(
                    carry[0],
                    |body| {
                        body.workgroup_barrier();
                        vec![]
                    },
                    |_| vec![],
                );
                vec![body.not(carry[0])]
            },
        );
        kernel.close();
    }

    #[test]
    fn private_storage_reads_cannot_admit_collective_control() {
        use crate::kernel::ops::{CmpOp, ConstantValue, ValueType};
        use crate::storage::LaunchLocalKind;
        use seismic_lang::{registry, types::DType};
        for kind in [LaunchLocalKind::Participant, LaunchLocalKind::Register] {
            let failure = catch_unwind(AssertUnwindSafe(|| {
                let mut arena = ExprArena::default();
                let one = arena.nat(1);
                let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
                let vectors = VectorSupport::default();
                let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
                let cell = kernel.local_tensor(kind, registry::dense(DType::U32), vec![one]);
                let zero = kernel.index_constant(0);
                let lane = kernel.local_id(0);
                let value = kernel.cast(lane, ValueType::Scalar(DType::U32));
                kernel.tensor_write(&cell, &[zero], value);
                let observed = kernel.tensor_read(&cell, &[zero]);
                let zero_word =
                    kernel.constant(ConstantValue::U32(0), ValueType::Scalar(DType::U32));
                let condition = kernel.cmp(CmpOp::Eq, observed, zero_word);
                kernel.branch(
                    condition,
                    |body| {
                        body.workgroup_barrier();
                        vec![]
                    },
                    |_| vec![],
                );
            }))
            .expect_err("private cell[0] is not a shared uniform observation");
            let message = failure
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| failure.downcast_ref::<&str>().copied())
                .unwrap();
            assert!(message.contains("divergent lexical control"), "{message}");
        }
    }

    #[test]
    fn branch_tensor_payload_joins_keep_launch_storage_and_reconverge() {
        use crate::kernel::dynamic::PortableSliceAxis;
        use crate::kernel::ops::{CmpOp, ConstantValue, ValueType};
        use crate::storage::LaunchLocalKind;
        use seismic_lang::{registry, types::DType};
        let mut arena = ExprArena::default();
        let four = arena.nat(4);
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let subgroup = kernel.subgroup_ordinal();
        let zero = kernel.index_constant(0);
        let first = kernel.cmp(CmpOp::Eq, subgroup, zero);
        let branch = kernel.begin_branch(first);
        let allocation = kernel.local_tensor(
            LaunchLocalKind::Participant,
            registry::dense(DType::U32),
            vec![four],
        );
        let one = kernel.index_constant(1);
        let two = kernel.index_constant(2);
        let value = kernel.constant(ConstantValue::U32(73), ValueType::Scalar(DType::U32));
        kernel.tensor_write(&allocation, &[one], value);
        let view = kernel.tensor_slice(
            allocation,
            vec![PortableSliceAxis::Range {
                start: one,
                end: two,
            }],
        );
        let fields = view.scalar_fields();
        kernel.begin_otherwise(&branch);
        let joined = kernel.finish_branch_products(
            branch,
            fields.into_iter().map(|v| (Some(v), None)).collect(),
        );
        let joined_view = kernel.tensor_with_scalar_fields(&view, &joined);
        kernel.branch(
            first,
            |body| {
                body.tensor_read(&joined_view, &[zero]);
                vec![]
            },
            |_| vec![],
        );
        kernel.workgroup_barrier();
        kernel.close();
    }

    #[test]
    fn actual_extent_values_survive_close_and_import_without_a_host_map() {
        use crate::kernel::dynamic::PortableSliceAxis;
        use crate::kernel::ops::Op;
        use crate::storage::LaunchLocalKind;
        use seismic_lang::{registry, types::DType};
        let mut arena = ExprArena::default();
        let four = arena.nat(4);
        let mut child = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = child.portable_kernel(&mut arena, &(), &[], &vectors);
        let representation = registry::dense(DType::F32);
        let backing = builder.local_tensor(LaunchLocalKind::Participant, representation, vec![four]);
        let zero = builder.index_constant(0);
        let one = builder.index_constant(1);
        let three = builder.index_constant(3);
        let lane = builder.local_id(0);
        let dynamic = builder.tensor_slice(backing.clone(), vec![PortableSliceAxis::Range {
            start: zero, end: lane,
        }]);
        // This constructor previously panicked while demanding a second NatExpr map.
        builder.semantic_place(dynamic, representation, 1, false);
        let host = builder.tensor_slice(backing, vec![PortableSliceAxis::Range {
            start: one, end: three,
        }]);
        let extent = builder.tensor_extents(&host)[0];
        builder.close();
        let kernel = &child.kernels()[0];
        assert_eq!(kernel.exact_nat(lane.raw), None);
        let exact = kernel.exact_nat(extent.raw).unwrap();
        assert_eq!(arena.eval_nat_u64(exact, &seismic_lang::expr::Assignment::new()).unwrap(), 2);
        let token = child.schedule(&mut arena, 0).close();
        let child = child.close(token).normalize_launches(&mut arena, 64, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        parent.import(&mut arena, 0, child, &[]);
        let kernel = &parent.kernels()[0];
        let imported = kernel.blocks().iter().flat_map(|block| &block.ops).find_map(|op| {
            match op {
                Op::Binary { out, .. } if out.ordinal() == extent.raw.ordinal() => Some(*out),
                _ => None,
            }
        }).unwrap();
        assert_eq!(kernel.exact_nat(imported), Some(exact));
        assert!(catch_unwind(AssertUnwindSafe(|| kernel.exact_nat(extent.raw))).is_err());
    }

    #[test]
    fn exact_native_index_facts_follow_word_wrap_before_host_projection() {
        use crate::kernel::ops::BinaryOp;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut builder = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let maximum = builder.index_constant(u64::MAX);
        let one = builder.index_constant(1);
        let two = builder.index_constant(2);
        let zero = builder.index_constant(0);
        let sum = builder.binary(BinaryOp::Add, maximum, one);
        let product = builder.binary(BinaryOp::Mul, maximum, two);
        let difference = builder.binary(BinaryOp::Sub, zero, one);
        builder.close();
        let kernel = &construction.kernels()[0];
        for (value, expected) in [(sum, 0), (product, u64::MAX - 1), (difference, u64::MAX)] {
            let expression = kernel.exact_nat(value.raw).unwrap();
            assert_eq!(arena.eval_nat_u64(expression, &seismic_lang::expr::Assignment::new()).unwrap(), expected);
        }
    }

    #[test]
    fn subgroup_ordinal_has_subgroup_uniform_origin() {
        use crate::kernel::ops::CmpOp;
        use seismic_lang::registry::IntrinsicUniformity;
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut portable = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let ordinal = portable.subgroup_ordinal();
        assert_eq!(portable.uniformity(ordinal), IntrinsicUniformity::Subgroup);
        let width = portable.subgroup_size();
        assert_eq!(portable.uniformity(width), IntrinsicUniformity::Workgroup);
        portable.close();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        let ordinal = kernel.subgroup_ordinal();
        let zero = kernel.index_constant(0);
        let first = kernel.cmp(CmpOp::Eq, ordinal, zero);
        kernel.branch(
            first,
            |body| {
                body.subgroup_barrier();
                vec![]
            },
            |_| vec![],
        );
        kernel.close();
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
            let ordinal = kernel.subgroup_ordinal();
            let zero = kernel.index_constant(0);
            let first = kernel.cmp(CmpOp::Eq, ordinal, zero);
            kernel.branch(
                first,
                |body| {
                    body.workgroup_barrier();
                    vec![]
                },
                |_| vec![],
            );
        }));
        assert!(rejected.is_err());
    }

    #[test]
    fn typed_constants_retain_payload_bits_in_the_closed_kernel() {
        use crate::kernel::ops::{ConstantValue, Op};
        use crate::repr::ScalarKind;
        use seismic_lang::reference_math::ReferenceScalar;

        let payloads = [
            ReferenceScalar::F16(0x7c01),
            ReferenceScalar::F16(0xfc23),
            ReferenceScalar::F16(0x8000),
            ReferenceScalar::BF16(0x7f81),
            ReferenceScalar::BF16(0xffe3),
            ReferenceScalar::BF16(0x8000),
            ReferenceScalar::F32(0x7f80_0001),
            ReferenceScalar::F32(0xffc0_1234),
            ReferenceScalar::F32(0x8000_0000),
            ReferenceScalar::I32(i32::MIN),
            ReferenceScalar::U32(u32::MAX),
            ReferenceScalar::Bool(true),
        ];
        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
        for value in payloads {
            kernel.constant(
                ConstantValue::from_scalar(value),
                ScalarKind::Scalar(value.dtype()).value_type(),
            );
        }
        kernel.close();

        {
            let kernel = &construction.kernels()[0];
            let actual: Vec<_> = kernel
                .blocks()
                .iter()
                .flat_map(|block| &block.ops)
                .map(|op| {
                    let Op::Constant { out, value } = op else {
                        panic!("constant transport introduced an operation")
                    };
                    let value = match *value {
                        ConstantValue::F16(bits) => ReferenceScalar::F16(bits),
                        ConstantValue::BF16(bits) => ReferenceScalar::BF16(bits),
                        ConstantValue::F32(value) => ReferenceScalar::F32(value.to_bits()),
                        ConstantValue::I32(value) => ReferenceScalar::I32(value),
                        ConstantValue::U32(value) => ReferenceScalar::U32(value),
                        ConstantValue::Bool(value) => ReferenceScalar::Bool(value),
                        ConstantValue::Index(_) => panic!("source constant became a natural"),
                    };
                    assert_eq!(
                        kernel.value_type(*out),
                        ScalarKind::Scalar(value.dtype()).value_type()
                    );
                    value
                })
                .collect();
            assert_eq!(actual, payloads);
        }
    }

    #[test]
    fn approximate_exponential_remains_an_explicit_physical_choice() {
        use crate::kernel::ops::{ConstantValue, MathPrecision, Op, ValueType};
        use seismic_lang::intrinsics::MathOp;
        use seismic_lang::types::DType;

        let mut arena = ExprArena::default();
        let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let vectors = VectorSupport::default();
        for approximate in [false, true] {
            let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
            let input = kernel.constant(ConstantValue::F32(1.0), ValueType::Scalar(DType::F32));
            if approximate {
                kernel.math_approximate(MathOp::Exp, input);
            } else {
                kernel.math(MathOp::Exp, input);
            }
            kernel.close();
        }
        let exact = &construction.kernels()[0];
        assert!(exact
            .blocks()
            .iter()
            .flat_map(|block| &block.ops)
            .all(|op| !matches!(op, Op::Math { .. })));

        let approximate = &construction.kernels()[1];
        assert_eq!(
            approximate
                .blocks()
                .iter()
                .flat_map(|block| &block.ops)
                .filter(|op| matches!(
                    op,
                    Op::Math {
                        op: MathOp::Exp,
                        precision: MathPrecision::Approximate,
                        ..
                    }
                ))
                .count(),
            1
        );
    }

    #[test]
    fn imported_normalization_checks_receiving_grid_and_natural_width() {
        use crate::schedule::LaunchNormalizationError;
        for (parent_cap, parent_bits, compatible) in [
            (8, 64, true),
            (16, 64, true),
            (4, 64, false),
            (8, 32, false),
        ] {
            let mut arena = ExprArena::default();
            let child = semantic_construction(&mut arena, 17, 1)
                .normalize_launches(&mut arena, 8, 64)
                .unwrap()
                .analyze_allocations()
                .apply_allocation_plan(&mut arena, AllocationPlan::distinct())
                .finish()
                .into_importable()
                .unwrap();
            let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            parent.import(&mut arena, 0, child, &[]);
            let schedule = parent.schedule(&mut arena, 0);
            let token = schedule.close();
            let result =
                parent
                    .close(token)
                    .normalize_launches(&mut arena, parent_cap, parent_bits);
            if compatible {
                assert!(result.is_ok());
            } else {
                assert!(matches!(
                    result,
                    Err(LaunchNormalizationError::IncompatibleLaunchNormalization { .. })
                ));
            }
        }
    }

    fn semantic_construction(arena: &mut ExprArena, n: u64, p: u64) -> SealedConstruction<Dialect> {
        use crate::kernel::ops::SegmentLaunchDomain;
        use crate::schedule::LaunchParticipation;
        let mut construction = Construction::<Dialect>::new(arena, vec![], false, 0);
        let one = arena.nat(1);
        let extent = arena.nat(n);
        let participants = arena.nat(p);
        let groups = arena.nat_ceil_div(extent, participants);
        let empty = arena.bool(n == 0);
        let vectors = VectorSupport::default();
        let mut kernel = construction.portable_kernel(arena, &(), &[], &vectors);
        let (_, binding) = kernel.logical_global_id(extent);
        let kernel = kernel.close();
        let mut schedule = construction.schedule(arena, 0);
        schedule.launch_semantic(
            kernel,
            SegmentLaunchDomain {
                mode: LaunchParticipation::Independent,
                grid: [groups, one, one],
                workgroup: [participants, one, one],
                empty,
                parallel_extent: extent,
            },
            Some(binding),
            (),
        );
        let token = schedule.close();
        construction.close(token)
    }

    #[test]
    fn normalized_chunks_obey_natural_width_at_tails_and_zero_extent() {
        use crate::schedule::ScheduleStep;
        use seismic_lang::expr::{Assignment, SymbolValue};
        for (extent, participants, grid_cap, bits, expected_chunks, expected_envelope) in [
            (255, 4, 100, 8, 2, 63),
            (17, 4, 2, 8, 3, 2),
            (0, 4, 2, 8, 0, 0),
            (u64::MAX, 4, u64::MAX, 64, 2, u64::MAX / 4),
        ] {
            let mut arena = ExprArena::default();
            let analyzed = semantic_construction(&mut arena, extent, participants)
                .normalize_launches(&mut arena, grid_cap, bits)
                .unwrap()
                .analyze_allocations();
            let schedule = analyzed.schedule();
            let ScheduleStep::Repeat { symbol, end, .. } = &schedule.steps()[0] else {
                panic!("expected normalized repeat")
            };
            assert_eq!(
                arena.eval_nat_u64(*end, &Assignment::new()).unwrap(),
                expected_chunks
            );
            let launch = &schedule.launches()[0];
            let envelope = schedule.launch_grid_envelope(&mut arena, launch);
            assert_eq!(
                arena.eval_nat_u64(envelope[0], &Assignment::new()).unwrap(),
                expected_envelope
            );
            let mut covered = 0;
            for chunk in 0..expected_chunks {
                let mut assignment = Assignment::new();
                assignment.bind(*symbol, SymbolValue::Nat((chunk).into()));
                let base = arena
                    .eval_nat_u64(launch.logical_base.unwrap().value(), &assignment)
                    .unwrap();
                let grid = arena.eval_nat_u64(launch.grid[0], &assignment).unwrap();
                assert_eq!(base, covered);
                assert!(grid <= grid_cap);
                let physical = grid.checked_mul(participants).unwrap();
                assert!(physical <= u64::MAX >> (64 - bits));
                covered += physical.min(extent - base);
            }
            assert_eq!(covered, extent);
        }
    }

    #[test]
    fn semantic_index_binding_rejects_foreign_kernel_extent_and_mapping() {
        use crate::kernel::ops::SegmentLaunchDomain;
        use crate::schedule::LaunchParticipation;
        for mismatch in 0..4 {
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let one = arena.nat(1);
            let two = arena.nat(2);
            let extent = arena.nat(17);
            let empty = arena.bool(false);
            let vectors = VectorSupport::default();
            let mut kernel = construction.portable_kernel(&mut arena, &(), &[], &vectors);
            let (_, binding) = kernel.logical_global_id(extent);
            let kernel = kernel.close();
            let other = construction
                .portable_kernel(&mut arena, &(), &[], &vectors)
                .close();
            let mut domain = SegmentLaunchDomain {
                mode: LaunchParticipation::Independent,
                grid: [extent, one, one],
                workgroup: [one; 3],
                empty,
                parallel_extent: extent,
            };
            match mismatch {
                0 => {}
                1 => domain.parallel_extent = two,
                2 => domain.grid[1] = two,
                3 => domain.workgroup[2] = two,
                _ => unreachable!(),
            }
            let mut schedule = construction.schedule(&mut arena, 0);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    schedule.launch_semantic(
                        if mismatch == 0 { other } else { kernel },
                        domain,
                        Some(binding),
                        (),
                    );
                }))
                .is_err(),
                "invalid logical index binding case {mismatch} was admitted"
            );
        }
    }

    #[test]
    fn semantic_bounds_precede_analysis_and_survive_import_without_rechunking() {
        use crate::kernel::ops::SegmentLaunchDomain;
        use crate::schedule::{LaunchParticipation, ScheduleStep};
        let mut arena = ExprArena::default();
        let mut child = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let one = arena.nat(1);
        let extent = arena.nat(17);
        let empty = arena.bool(false);
        let vectors = VectorSupport::default();
        let mut kernel = child.portable_kernel(&mut arena, &(), &[], &vectors);
        let (_, base) = kernel.logical_global_id(extent);
        let logical_base_value = kernel.logical_base(base);
        let kernel = kernel.close();
        assert_eq!(child.kernels()[0].exact_nat(logical_base_value.raw), None);
        let mut schedule = child.schedule(&mut arena, 0);
        schedule.launch_semantic(
            kernel,
            SegmentLaunchDomain {
                mode: LaunchParticipation::Independent,
                grid: [extent, one, one],
                workgroup: [one; 3],
                empty,
                parallel_extent: extent,
            },
            Some(base),
            (),
        );
        let closed = schedule.close();
        let analyzed = child
            .close(closed)
            .normalize_launches(&mut arena, 8, 64)
            .unwrap()
            .analyze_allocations();
        assert!(matches!(
            analyzed.schedule().steps(),
            [ScheduleStep::Repeat { .. }]
        ));
        assert!(matches!(
            analyzed.schedule().launch_uses()[0].1.region(),
            [crate::storage::ScheduleRegionEdge::RepeatBody { .. }]
        ));
        let child = analyzed
            .apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish()
            .into_importable()
            .unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        parent.import(&mut arena, 0, child, &[]);
        let imported_kernel = &parent.kernels()[0];
        for op in imported_kernel.blocks().iter().flat_map(|block| &block.ops) {
            if let crate::kernel::ops::Op::NatArg { out, index } = op {
                if *index == base.argument {
                    assert_eq!(imported_kernel.exact_nat(*out), None);
                }
            }
        }
        let schedule = parent.schedule(&mut arena, 0);
        let closed = schedule.close();
        let analyzed = parent
            .close(closed)
            .normalize_launches(&mut arena, 8, 64)
            .unwrap()
            .analyze_allocations();
        fn repeats(steps: &[ScheduleStep]) -> usize {
            steps
                .iter()
                .map(|step| match step {
                    ScheduleStep::Imported { body, .. } => repeats(body),
                    ScheduleStep::Repeat { body, .. } => 1 + repeats(body),
                    ScheduleStep::If {
                        then_steps,
                        else_steps,
                        ..
                    } => repeats(then_steps) + repeats(else_steps),
                    _ => 0,
                })
                .sum()
        }
        assert_eq!(repeats(analyzed.schedule().steps()), 1);
        assert!(analyzed.schedule().launches()[0]
            .logical_base
            .unwrap()
            .is_chunked());
    }

    fn filled(arena: &mut ExprArena) -> ExecutableIr<Dialect> {
        let mut c = Construction::<Dialect>::new(arena, vec![], false, 0);
        let n = arena.nat(8);
        let (_, index) =
            c.storage_mut()
                .tensor(arena, GlobalBufferKind::Arena, DenseF32::id(), vec![n]);
        let view = c.view(index, DenseF32::id());
        let mut schedule = c.schedule(arena, 0);
        schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
        let token = schedule.close();
        let analyzed = c
            .close(token)
            .normalize_launches(arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
        assert_eq!(analyzed.arena_allocations().len(), 1);
        analyzed
            .apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish()
    }
    #[test]
    fn importing_real_storage_preserves_schedule_uses_and_reuse_choices() {
        use crate::schedule::ScheduleStep;
        let mut arena = ExprArena::default();
        let child = filled(&mut arena);
        let ScheduleStep::Fill(fill) = &child.schedule().steps()[0] else {
            panic!("fixture has one fill");
        };
        let returned_view = fill.destination;
        let child = child.into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let imported = parent.import(&mut arena, 0, child, &[]);
        let returned_view = imported.remap_view(returned_view);
        parent.assert_view(returned_view);
        let builder = parent.schedule(&mut arena, 0);
        let token = builder.close();
        let analyzed = parent
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
        assert_eq!(analyzed.arena_allocations().len(), 1);
        assert_eq!(analyzed.liveness()[0].uses().len(), 1);
        assert_eq!(analyzed.liveness()[0].uses()[0].region().len(), 1);
        let planned = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct());
        let executable = planned.finish();
        assert_eq!(executable.storage().views().len(), 1);
    }
    #[test]
    fn allocation_timing_precedes_slot_reuse_and_publication_preserves_reached_timing() {
        for published in [false, true] {
            let mut arena = ExprArena::default();
            let mut construction = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            let n = arena.nat(8);
            let (first_id, first_index) = construction.storage_mut().tensor(
                &mut arena, GlobalBufferKind::Arena, DenseF32::id(), vec![n]);
            let (second_id, second_index) = construction.storage_mut().tensor(
                &mut arena, GlobalBufferKind::Arena, DenseF32::id(), vec![n]);
            let first = construction.view(first_index, DenseF32::id());
            let second = construction.view(second_index, DenseF32::id());
            let first_value = construction.begin_tensor_instance(&mut arena, 0, first);
            let mut schedule = construction.schedule(&mut arena, 0);
            schedule.fill_constant_any(first_value, seismic_lang::intrinsics::FillConstant::Zero);
            if published { schedule.publish_tensor(first_value, vec![0], vec![n]); }
            // Compiler-owned status-like storage has no source instance transition.
            schedule.fill_constant_any(second, seismic_lang::intrinsics::FillConstant::Zero);
            let closed = schedule.close();
            let analyzed = construction.close(closed).normalize_launches(&mut arena, u64::MAX, 64)
                .unwrap().analyze_allocations();
            let expected = AllocationAcquisition::Reached;
            assert_eq!(analyzed.acquisitions[first_id.index() as usize], expected);
            assert_eq!(analyzed.acquisitions[second_id.index() as usize], AllocationAcquisition::Invocation);
            assert_eq!(analyzed.allocation_relations()[0].storage_compatible(), false,
                "mixed acquisition timing must not share a physical slot");
            let executable = analyzed.apply_allocation_plan(&mut arena, AllocationPlan::distinct()).finish();
            assert_eq!(executable.storage().allocation(first_id).acquisition, expected);
        }
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
        let first = construction.view(first_index, DenseF32::id());
        let second = construction.view(second_index, DenseF32::id());
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_constant_any(first, seismic_lang::intrinsics::FillConstant::Zero);
        schedule.fill_constant_any(second, seismic_lang::intrinsics::FillConstant::Zero);
        let token = schedule.close();
        let analyzed = construction
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();

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
        let first = construction.view(first_index, DenseF32::id());
        let second = construction.view(second_index, DenseF16::id());
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_constant_any(first, seismic_lang::intrinsics::FillConstant::Zero);
        schedule.fill_constant_any(second, seismic_lang::intrinsics::FillConstant::Zero);
        let token = schedule.close();
        let analyzed = construction
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
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
    fn empty_import_scopes_remain_owned_and_nested_after_closure() {
        use crate::schedule::ScheduleStep;
        fn empty(arena: &mut ExprArena) -> ImportableExecutableIr<Dialect> {
            let mut construction = Construction::<Dialect>::new(arena, vec![], false, 0);
            let token = construction.schedule(arena, 0).close();
            construction
                .close(token)
                .normalize_launches(arena, u64::MAX, 64)
                .unwrap()
                .analyze_allocations()
                .apply_allocation_plan(arena, AllocationPlan::distinct())
                .finish()
                .into_importable()
                .unwrap()
        }
        let mut arena = ExprArena::default();
        let mut middle = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let empty = empty(&mut arena);
        let nested = middle.import(&mut arena, 0, empty, &[]).scope().clone();
        let token = middle.schedule(&mut arena, 0).close();
        let middle = middle
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations()
            .apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish()
            .into_importable()
            .unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let imported = parent.import(&mut arena, 0, middle, &[]);
        let nested = imported.remap_scope(&nested);
        let token = parent.schedule(&mut arena, 0).close();
        let closed = parent.close(token);
        let [ScheduleStep::Imported { scope, body }] = closed.schedule.steps() else {
            panic!("outer import was erased")
        };
        assert_eq!(scope, imported.scope());
        let [ScheduleStep::Imported { scope, body }] = body.as_slice() else {
            panic!("effect-only nested scope was erased")
        };
        assert_eq!(scope, &nested);
        assert!(body.is_empty());
        assert_ne!(scope, imported.scope());
    }

    #[test]
    fn imported_bindings_reject_foreign_child_views() {
        let mut arena = ExprArena::default();
        let child = filled(&mut arena).into_importable().unwrap();
        let foreign = filled(&mut arena);
        let foreign_view = foreign.storage().view_id(0);
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let imported = parent.import(&mut arena, 0, child, &[]);
        assert!(catch_unwind(AssertUnwindSafe(|| imported.remap_view(foreign_view))).is_err());
    }
    #[test]
    fn imported_host_quantity_keeps_its_expression_identity_and_remaps_its_slot() {
        use crate::schedule::{HostEvaluation, HostQuantityKind, HostValueDestination, HostValueExpr, ScheduleStep};
        let mut arena = ExprArena::default();
        let mut child = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let max = arena.int(i64::MAX);
        let one = arena.int(1);
        let exact = arena.int_add(max, one);
        let child_slot = child.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Integer);
        child.schedule(&mut arena, 0).evaluate_host(HostEvaluation {
            value: HostValueExpr::Integer(exact),
            to: HostValueDestination::Quantity(child_slot),
            failure: None,
        });
        let token = child.schedule(&mut arena, 0).close();
        let child = child.close(token).normalize_launches(&mut arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let bindings = parent.import(&mut arena, 0, child, &[]);
        let mapped = bindings.remap_quantity_slot(child_slot);
        assert_ne!(mapped, child_slot);
        assert_eq!(mapped.symbol(), child_slot.symbol());
        let token = parent.schedule(&mut arena, 0).close();
        let parent = parent.close(token);
        let [ScheduleStep::Imported { body, .. }] = parent.schedule.steps() else { panic!("import scope") };
        assert!(matches!(body.as_slice(), [ScheduleStep::EvaluateHost(value)] if matches!(value.to, HostValueDestination::Quantity(slot) if slot == mapped)));
    }
    fn exact_host_child(arena: &mut ExprArena) -> (ImportableExecutableIr<Dialect>, crate::schedule::HostQuantitySlot) {
        use crate::schedule::{HostEvaluation, HostQuantityKind, HostValueDestination, HostValueExpr};
        let mut child = Construction::<Dialect>::new(arena, vec![], false, 0);
        let value = arena.int(7);
        let slot = child.schedule(arena, 0).quantity_slot(HostQuantityKind::Integer);
        child.schedule(arena, 0).evaluate_host(HostEvaluation {
            value: HostValueExpr::Integer(value),
            to: HostValueDestination::Quantity(slot), failure: None,
        });
        let token = child.schedule(arena, 0).close();
        let child = child.close(token).normalize_launches(arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        (child, slot)
    }

    #[test]
    fn sibling_imports_receive_distinct_final_schedule_symbol_fingerprints() {
        use seismic_lang::expr::{RootName, SymbolKind};
        let mut arena = ExprArena::default();
        let (first, first_slot) = exact_host_child(&mut arena);
        let (second, second_slot) = exact_host_child(&mut arena);
        assert_eq!(arena.symbol_kind(first_slot.symbol()), arena.symbol_kind(second_slot.symbol()));
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let first = parent.import(&mut arena, 0, first, &[]).remap_quantity_slot(first_slot);
        let second = parent.import(&mut arena, 0, second, &[]).remap_quantity_slot(second_slot);
        assert_eq!(arena.symbol_kind(first.symbol()), SymbolKind::ScheduleSlot(0));
        assert_eq!(arena.symbol_kind(second.symbol()), SymbolKind::ScheduleSlot(1));
        let first_value = arena.int_symbol(first.symbol());
        let second_value = arena.int_symbol(second.symbol());
        let first_root = arena.root(RootName::HostEvaluation { step: 0 }, first_value.into());
        let second_root = arena.root(RootName::HostEvaluation { step: 0 }, second_value.into());
        assert_ne!(arena.canonical_digest(&[first_root]), arena.canonical_digest(&[second_root]));
    }

    #[test]
    fn borrowed_parent_quantity_reuses_its_owner_slot_when_child_is_spliced() {
        use crate::schedule::{HostEvaluation, HostQuantityKind, HostValueDestination, HostValueExpr};
        use seismic_lang::expr::SymbolKind;
        let mut arena = ExprArena::default();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let parent_slot = parent.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Integer);
        let mut child = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let borrowed = child.schedule_state().capture_quantity_slot(parent_slot);
        let result = child.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Integer);
        let captured = arena.int_symbol(borrowed.symbol());
        let one = arena.int(1);
        let next = arena.int_add(captured, one);
        child.schedule(&mut arena, 0).evaluate_host(HostEvaluation {
            value: HostValueExpr::Integer(next),
            to: HostValueDestination::Quantity(result), failure: None,
        });
        let token = child.schedule(&mut arena, 0).close();
        let child = child.close(token).normalize_launches(&mut arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let bindings = parent.import(&mut arena, 0, child, &[]);
        assert_eq!(bindings.remap_quantity_slot(borrowed), parent_slot);
        let mapped_result = bindings.remap_quantity_slot(result);
        assert_eq!(arena.symbol_kind(parent_slot.symbol()), SymbolKind::ScheduleSlot(0));
        assert_eq!(arena.symbol_kind(mapped_result.symbol()), SymbolKind::ScheduleSlot(1));
        let token = parent.schedule(&mut arena, 0).close();
        let parent = parent.close(token);
        assert_eq!(parent.schedule.quantity_slots().len(), 2);
    }

    #[test]
    fn nested_borrowed_quantity_resolves_to_one_ancestor_slot() {
        use crate::schedule::HostQuantityKind;
        let mut arena = ExprArena::default();
        let mut root = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let owned = root.schedule(&mut arena, 0).quantity_slot(HostQuantityKind::Integer);
        let mut middle = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let borrowed = middle.schedule_state().capture_quantity_slot(owned);
        let mut child = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        let twice_borrowed = child.schedule_state().capture_quantity_slot(borrowed);
        let token = child.schedule(&mut arena, 0).close();
        let child = child.close(token).normalize_launches(&mut arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let child_binding = middle.import(&mut arena, 0, child, &[]).remap_quantity_slot(twice_borrowed);
        assert_eq!(child_binding, borrowed);
        let token = middle.schedule(&mut arena, 0).close();
        let middle = middle.close(token).normalize_launches(&mut arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let root_binding = root.import(&mut arena, 0, middle, &[]).remap_quantity_slot(borrowed);
        assert_eq!(root_binding, owned);
        let token = root.schedule(&mut arena, 0).close();
        let root = root.close(token);
        assert_eq!(root.schedule.quantity_slots().len(), 1);
    }

    #[test]
    fn nested_import_rebases_slots_again_into_final_parent_order() {
        use seismic_lang::expr::SymbolKind;
        let mut arena = ExprArena::default();
        let (child, child_slot) = exact_host_child(&mut arena);
        let mut middle = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        middle.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
        let middle_slot = middle.import(&mut arena, 0, child, &[]).remap_quantity_slot(child_slot);
        assert_eq!(arena.symbol_kind(middle_slot.symbol()), SymbolKind::ScheduleSlot(1));
        let token = middle.schedule(&mut arena, 0).close();
        let middle = middle.close(token).normalize_launches(&mut arena, u64::MAX, 64).unwrap()
            .analyze_allocations().apply_allocation_plan(&mut arena, AllocationPlan::distinct())
            .finish().into_importable().unwrap();
        let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
        parent.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
        parent.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
        let final_slot = parent.import(&mut arena, 0, middle, &[]).remap_quantity_slot(middle_slot);
        assert_eq!(arena.symbol_kind(final_slot.symbol()), SymbolKind::ScheduleSlot(3));
    }

    #[test]
    fn final_import_fingerprint_ignores_unrelated_arena_materialization() {
        use seismic_lang::expr::RootName;
        fn fingerprint(unrelated: bool) -> seismic_lang::expr::ExprDigest {
            let mut arena = ExprArena::default();
            if unrelated {
                let mut other = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
                other.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
                other.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
            }
            let (child, child_slot) = exact_host_child(&mut arena);
            let mut parent = Construction::<Dialect>::new(&mut arena, vec![], false, 0);
            parent.schedule(&mut arena, 0).slot_any(crate::repr::ScalarKind::Nat64);
            let imported = parent.import(&mut arena, 0, child, &[]).remap_quantity_slot(child_slot);
            let value = arena.int_symbol(imported.symbol());
            let root = arena.root(RootName::HostEvaluation { step: 0 }, value.into());
            arena.canonical_digest(&[root])
        }
        assert_eq!(fingerprint(false), fingerprint(true));
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
        let mut schedule = construction.schedule(&mut arena, 0);
        schedule.fill_constant_any(view, seismic_lang::intrinsics::FillConstant::Zero);
        schedule.publish_tensor(view, vec![0], vec![n]);
        let token = schedule.close();
        let analyzed = construction
            .close(token)
            .normalize_launches(&mut arena, u64::MAX, 64)
            .unwrap()
            .analyze_allocations();
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
