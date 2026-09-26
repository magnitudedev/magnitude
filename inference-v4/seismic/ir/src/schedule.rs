//! Structured parametric schedules and typed data movement.
//!
//! Construction stores control as a region tree. A leaf command belongs to
//! exactly one lexical region, so allocation liveness and later command
//! grouping cannot accidentally cross `If` or `Repeat`.

use crate::identity::OwnerToken;
use crate::kernel::KernelId;
use crate::physical_target::PhysicalDialect;
use crate::region::{BranchResult, Product, RepeatCarry};
use crate::repr::ScalarKind;
use crate::storage::{AnyBufferView, BufferViewLayout, ScheduleRegionEdge, ScheduleUse};
use seismic_lang::expr::{BoolExpr, ExprArena, IntExpr, LoopBinderId, NatExpr, SymbolId};
use seismic_lang::types::DType;
use std::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnyScalarSlot {
    owner: OwnerToken,
    pub(crate) index: u32,
    pub(crate) kind: ScalarKind,
    pub(crate) symbol: SymbolId,
    capture: Option<SlotCapture>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SlotCapture {
    owner: OwnerToken,
    index: u32,
}
impl AnyScalarSlot {
    pub fn index(&self) -> u32 {
        self.index
    }
    pub fn kind(&self) -> ScalarKind {
        self.kind
    }
    pub fn symbol(&self) -> SymbolId {
        self.symbol
    }
    pub fn sort(&self) -> seismic_lang::expr::SymbolSort {
        self.kind.sort()
    }
}

/// Exact mathematical value held by the host schedule. It is deliberately
/// separate from the fixed-width `AnyScalarSlot` kernel ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HostQuantityKind {
    Integer,
    Natural,
}
impl HostQuantityKind {
    pub const fn sort(self) -> seismic_lang::expr::SymbolSort {
        match self {
            Self::Integer => seismic_lang::expr::SymbolSort::Int,
            Self::Natural => seismic_lang::expr::SymbolSort::Nat,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HostQuantitySlot {
    owner: OwnerToken,
    index: u32,
    kind: HostQuantityKind,
    symbol: SymbolId,
    capture: Option<SlotCapture>,
}

/// One value evaluated when its source operation is reached. The expression
/// must name the actual bound operand versions of that operation. A word
/// expression is already the language's typed conversion/operation recipe;
/// the schedule only projects its proven word-domain result to the ABI slot.
#[derive(Clone, Copy, Debug)]
pub enum HostValueExpr {
    Integer(IntExpr),
    Natural(NatExpr),
    Bool(BoolExpr),
    Word {
        dtype: DType,
        value: IntExpr,
    },
    /// The language's float conversion of an exact integer (C1-19: one RNE
    /// rounding of the mathematical value to `dtype`).
    Float {
        dtype: DType,
        value: IntExpr,
    },
}

#[derive(Clone, Copy, Debug)]
pub enum HostValueDestination {
    Quantity(HostQuantitySlot),
    Native(AnyScalarSlot),
}

#[derive(Clone, Debug)]
pub struct HostEvaluation {
    pub value: HostValueExpr,
    pub to: HostValueDestination,
    /// Present only for a reached source operation that owns an existing
    /// checked failure event, such as mathematical division by zero.
    pub failure: Option<crate::kernel::ops::CheckSite>,
}
impl HostQuantitySlot {
    pub fn kind(self) -> HostQuantityKind {
        self.kind
    }
    pub fn symbol(self) -> SymbolId {
        self.symbol
    }
    pub fn index(self) -> u32 {
        self.index
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub(crate) fn remap(self, owner: OwnerToken, index: u32) -> Self {
        Self {
            owner,
            index,
            ..self
        }
    }
}

impl AnyScalarSlot {
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub(crate) fn remap(self, owner: OwnerToken, index: u32) -> Self {
        Self {
            owner,
            index,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchId {
    owner: OwnerToken,
    index: u32,
}
impl LaunchId {
    pub(crate) fn new(owner: OwnerToken, index: u32) -> Self {
        Self { owner, index }
    }
    pub(crate) fn owner(self) -> OwnerToken {
        self.owner
    }
    pub fn index(self) -> u32 {
        self.index
    }
}

#[derive(Debug)]
pub struct Launch<B: PhysicalDialect> {
    pub kernel: KernelId,
    pub descriptor: B::LaunchDescriptor,
    pub grid: [NatExpr; 3],
    pub workgroup: [NatExpr; 3],
    pub empty: BoolExpr,
    /// Logical participants covered by a semantic launch. Native closure uses
    /// this with the reflected subgroup width; non-semantic launches omit it.
    pub parallel_extent: Option<NatExpr>,
    /// Compiler-owned logical indexing for a universal portable launch.
    /// Authored/native launches do not carry this contract.
    pub logical_base: Option<LogicalLaunchBase>,
}

impl<B: PhysicalDialect> Clone for Launch<B> {
    fn clone(&self) -> Self {
        Self {
            kernel: self.kernel,
            descriptor: self.descriptor.clone(),
            grid: self.grid,
            workgroup: self.workgroup,
            empty: self.empty,
            parallel_extent: self.parallel_extent,
            logical_base: self.logical_base,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalLaunchBase {
    binding: crate::kernel::internals::LogicalIndexBinding,
    /// Schedule expression supplied to that argument for this launch.
    value: NatExpr,
    /// Owned by schedule construction; imported launch scopes retain it.
    chunk_bounds: Option<SemanticChunkBounds>,
}
impl LogicalLaunchBase {
    pub fn argument(self) -> u32 {
        self.binding.argument
    }
    pub fn value(self) -> NatExpr {
        self.value
    }
    pub fn is_chunked(self) -> bool {
        self.chunk_bounds.is_some()
    }
}

/// Limits actually established by launch normalization, owned by the launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SemanticChunkBounds {
    grid_cap: u64,
    participants: u64,
    natural_bits: u32,
}

/// A normalized child cannot be transplanted under incompatible target limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaunchNormalizationError {
    InvalidTargetLimits,
    InvalidParticipantCount {
        launch: LaunchId,
    },
    IncompatibleLaunchNormalization {
        launch: LaunchId,
        child_grid_cap: u64,
        parent_grid_cap: u64,
        child_natural_bits: u32,
        parent_natural_bits: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Semantic cohort requirement consumed during construction, not a selected dispatch.
pub enum LaunchParticipation {
    Independent,
    CooperativeGrid,
}

#[derive(Clone, Debug)]
pub struct BufferCopy {
    pub source: AnyBufferView,
    pub destination: AnyBufferView,
    pub bytes: NatExpr,
}
#[derive(Clone, Debug)]
pub struct BufferFill {
    pub destination: AnyBufferView,
    pub value: FillValue,
    pub bytes: NatExpr,
}
/// Canonical little-endian element pattern. Construction encodes it from the
/// typed destination element; executors only repeat these exact bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FillValue {
    U8([u8; 1]),
    U16([u8; 2]),
    U32([u8; 4]),
}
impl FillValue {
    pub fn pattern(&self) -> &[u8] {
        match self {
            Self::U8(value) => value,
            Self::U16(value) => value,
            Self::U32(value) => value,
        }
    }
    pub const fn width(&self) -> u64 {
        match self {
            Self::U8(_) => 1,
            Self::U16(_) => 2,
            Self::U32(_) => 4,
        }
    }
}
#[derive(Clone, Debug)]
pub struct ScalarMove {
    pub from: AnyScalarSlot,
    pub to: AnyScalarSlot,
}
#[derive(Clone, Debug)]
pub struct ScalarRead {
    pub source: AnyBufferView,
    pub index: Vec<NatExpr>,
    /// Construction-owned logical bounds for every coordinate. These become
    /// implementation hard constraints before an executable is frozen.
    pub bounds: Vec<BoolExpr>,
    pub byte_offset: NatExpr,
    pub to: AnyScalarSlot,
}
#[derive(Clone, Debug)]
pub struct ScalarCheck {
    pub condition: AnyScalarSlot,
    pub expectation: ScalarCheckExpectation,
    pub site: crate::kernel::ops::CheckSite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarCheckExpectation {
    BoolTrue,
    U32Zero,
}

/// One actual lexical child import. Its location is schedule-owned and stays
/// distinct across static call sites; a runtime repeat revisits this same scope.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ImportScope {
    at: ScheduleUse,
    node: u32,
}
impl ImportScope {
    pub fn at(&self) -> &ScheduleUse {
        &self.at
    }
    fn body_path(&self) -> Vec<ScheduleRegionEdge> {
        let mut path = self.at.region.clone();
        path.push(ScheduleRegionEdge::Imported {
            node: self.node,
            parent_ordinal: self.at.ordinal,
        });
        path
    }
    fn remap(&self, child: &Self) -> Self {
        let mut region = self.body_path();
        region.extend(child.at.region.iter().cloned());
        Self {
            at: ScheduleUse {
                owner: self.at.owner,
                region,
                ordinal: child.at.ordinal,
            },
            node: child.node,
        }
    }
}

/// The lexical point that owns a physical-use proof obligation.
pub enum RequirementPoint<'a> {
    Step(&'a ScheduleStep),
    BranchResult(&'a Product<BranchResult>, bool),
    RepeatInitial(&'a Product<RepeatCarry>),
    RepeatBackedge(&'a Product<RepeatCarry>),
}

#[derive(Clone, Debug)]
pub enum ScheduleStep {
    /// Ordered lexical composition, retained for source/physical relation
    /// analysis even when the child produces no result value.
    Imported {
        scope: ImportScope,
        body: Vec<ScheduleStep>,
    },
    /// Reached construction of one stored value. Selects its actual backing
    /// instance and acquires storage here when the allocation plan is reached.
    BeginAllocationInstance {
        source: AnyBufferView,
        result: AnyBufferView,
    },
    BindArgumentTensor {
        source: AnyBufferView,
        result: AnyBufferView,
    },
    /// Capture the actual result descriptor at its successful source point.
    /// Visibility to the caller still requires successful terminal completion.
    PublishTensor {
        view: AnyBufferView,
        path: Vec<u32>,
        bytes: NatExpr,
        declared_axes: Vec<NatExpr>,
    },
    Launch(LaunchId),
    Copy(BufferCopy),
    Fill(BufferFill),
    ScalarMove(ScalarMove),
    EvaluateHost(HostEvaluation),
    ScalarRead(ScalarRead),
    Check(ScalarCheck),
    If {
        condition: BoolExpr,
        then_steps: Vec<ScheduleStep>,
        else_steps: Vec<ScheduleStep>,
        results: Product<BranchResult>,
    },
    Repeat {
        binder: LoopBinderId,
        symbol: SymbolId,
        start: NatExpr,
        end: NatExpr,
        visits: RepeatVisits,
        body: Vec<ScheduleStep>,
        carries: Product<RepeatCarry>,
    },
}

/// How a repeat's visits relate when one of them stops at a source failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RepeatVisits {
    /// Each visit continues the previous one; a failure ends the repeat.
    Ordered,
    /// The checked independent participants of a `parallel for`, visited in
    /// index order. A source failure ends only its own visit; every other
    /// visit still runs, and the repeat then fails with the lowest-index
    /// failure. Independent visits carry no products.
    Independent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopBinding {
    pub id: LoopBinderId,
    pub symbol: SymbolId,
    pub index: NatExpr,
}

#[derive(Debug)]
pub struct ParametricSchedule<B: PhysicalDialect> {
    owner: OwnerToken,
    next_control: u32,
    launches: Vec<Launch<B>>,
    slots: Vec<AnyScalarSlot>,
    quantity_slots: Vec<HostQuantitySlot>,
    steps: Vec<ScheduleStep>,
    direct_view_uses: Vec<(AnyBufferView, ScheduleUse)>,
    launch_uses: Vec<(LaunchId, ScheduleUse)>,
}
impl<B: PhysicalDialect> ParametricSchedule<B> {
    /// Finite backing set of an actual region value. Cyclic header/backedge
    /// references are resolved from the same product, never an alias registry.
    pub fn backing_allocations(
        &self,
        views: &[BufferViewLayout],
        view: AnyBufferView,
    ) -> Vec<crate::storage::GlobalAllocationId> {
        let mut products = Vec::new();
        crate::region::collect_carries(&self.steps, &mut products);
        let mut definitions = Vec::new();
        crate::region::collect_tensor_definitions(&self.steps, &mut definitions);
        crate::region::possible_backings(views, view, &products, &definitions)
            .expect("closed region view has no defining product")
    }
    /// Pool cardinality follows reached host allocation occurrences and actual
    /// transported descriptors. Host schedule visits are sequential; native
    /// participant-local allocation capacity is owned by the kernel layout.
    pub(crate) fn allocation_instances(
        &self,
        views: &[BufferViewLayout],
        count: usize,
    ) -> Vec<u32> {
        use crate::storage::ViewBase;
        let mut retained = vec![0u32; count];
        // Every finite tensor-value location can retain one distinct instance.
        // Count an additional copy for simultaneous old/next product capture.
        // These are bank identities only; reached bytes are never multiplied.
        for (ordinal, layout) in views.iter().enumerate() {
            if !matches!(layout.base, ViewBase::TensorValue(id) if id.index() as usize == ordinal) {
                continue;
            }
            let view = AnyBufferView::new(self.owner, ordinal as u32, layout.representation);
            for root in self.backing_allocations(views, view) {
                retained[root.index() as usize] = retained[root.index() as usize]
                    .checked_add(2)
                    .expect("tensor value pool count exceeds u32");
            }
        }
        fn visit(
            steps: &[ScheduleStep],
            views: &[BufferViewLayout],
            inside: bool,
            retained: &[u32],
            result: &mut [u32],
        ) {
            for step in steps {
                match step {
                    ScheduleStep::BeginAllocationInstance { source: view, .. } if inside => {
                        let ViewBase::Allocation(root) = views[view.index() as usize].base else {
                            panic!("allocation definition is not stored")
                        };
                        result[root.index() as usize] = retained[root.index() as usize]
                            .checked_add(1)
                            .expect("backing count exceeds u32");
                    }
                    ScheduleStep::Imported { body, .. } => {
                        visit(body, views, inside, retained, result)
                    }
                    ScheduleStep::Repeat { body, .. } => visit(body, views, true, retained, result),
                    ScheduleStep::If {
                        then_steps,
                        else_steps,
                        ..
                    } => {
                        visit(then_steps, views, inside, retained, result);
                        visit(else_steps, views, inside, retained, result);
                    }
                    _ => (),
                }
            }
        }
        let mut result = vec![1; count];
        visit(&self.steps, views, false, &retained, &mut result);
        result
    }
    pub fn launch_id(&self, ordinal: u32) -> LaunchId {
        assert!(
            (ordinal as usize) < self.launches.len(),
            "launch ordinal is outside schedule"
        );
        LaunchId::new(self.owner, ordinal)
    }

    /// Rewrites one universal semantic launch into exact block-capped chunks.
    ///
    /// The launch identity and launch-local layout remain stable. The repeat
    /// binder owns the logical base, and the final iteration derives its grid
    /// from the exact remaining logical extent rather than over-launching a
    /// full chunk.
    pub(crate) fn chunk_semantic_launch(
        &mut self,
        arena: &mut ExprArena,
        id: LaunchId,
        maximum_grid_x: u64,
        natural_bits: u32,
    ) -> Result<(), LaunchNormalizationError> {
        self.assert_owner(id.owner());
        if maximum_grid_x == 0 || !(1..=64).contains(&natural_bits) {
            return Err(LaunchNormalizationError::InvalidTargetLimits);
        }
        let launch = self.launch(id);
        let participants = arena.nat_product(&launch.workgroup);
        let Ok(participant_count) =
            arena.eval_nat_u64(participants, &seismic_lang::expr::Assignment::new())
        else {
            // Optimized launches may retain finite or invocation-dependent
            // geometry. They keep their ordinary geometry obligations; no
            // constant-geometry normalization authority is issued for them.
            assert!(!launch.logical_base.unwrap().is_chunked());
            return Ok(());
        };
        let maximum_natural = u64::MAX >> (64 - natural_bits);
        if participant_count == 0 || participant_count > maximum_natural {
            return Err(LaunchNormalizationError::InvalidParticipantCount { launch: id });
        }
        let grid_cap = maximum_grid_x.min(maximum_natural / participant_count);
        if let Some(bounds) = launch.logical_base.and_then(|base| base.chunk_bounds) {
            if bounds.participants != participant_count
                || bounds.grid_cap > grid_cap
                || bounds.natural_bits != natural_bits
            {
                return Err(LaunchNormalizationError::IncompatibleLaunchNormalization {
                    launch: id,
                    child_grid_cap: bounds.grid_cap,
                    parent_grid_cap: grid_cap,
                    child_natural_bits: bounds.natural_bits,
                    parent_natural_bits: natural_bits,
                });
            }
            return Ok(());
        }
        let max_grid_x = arena.nat(grid_cap);
        let control = self.next_control;
        self.next_control = self
            .next_control
            .checked_add(1)
            .expect("schedule control identity space exhausted");
        for (used, at) in &mut self.launch_uses {
            if *used == id {
                at.region.push(ScheduleRegionEdge::RepeatBody {
                    node: control,
                    parent_ordinal: at.ordinal,
                });
                at.ordinal = 0;
            }
        }
        let launch = self
            .launches
            .get_mut(id.index() as usize)
            .expect("chunked launch ordinal is in bounds");
        let extent = launch
            .parallel_extent
            .expect("only semantic launches can be chunked");
        let logical = launch
            .logical_base
            .as_mut()
            .expect("only universal portable launches can be chunked");
        assert!(
            !logical.is_chunked(),
            "semantic launch already has a bounded repeat scope"
        );
        logical.chunk_bounds = Some(SemanticChunkBounds {
            grid_cap,
            participants: participant_count,
            natural_bits,
        });
        let participants = arena.nat_product(&launch.workgroup);
        let capacity = arena.nat_mul(max_grid_x, participants);
        let zero = arena.nat(0);
        let blocks = arena.nat_ceil_div(extent, participants);
        let repeat_end = arena.nat_ceil_div(blocks, max_grid_x);
        let full_chunks = arena.nat_div(blocks, max_grid_x);
        let tail_blocks = arena.nat_rem(blocks, max_grid_x);
        let (binder, symbol, iteration) = arena.nat_loop_binder();
        let base = arena.nat_mul(iteration, capacity);
        // Quotient and remainder make the last launch exact without a
        // subtraction whose definedness depends on the enclosing repeat.
        let full = arena.nat_cmp(seismic_lang::expr::CmpOp::Lt, iteration, full_chunks);
        launch.grid[0] = arena.nat_select(full, max_grid_x, tail_blocks);
        logical.value = base;
        launch.empty = arena.bool(false);

        let mut replacements = 0usize;
        rewrite_launch_as_repeat(
            &mut self.steps,
            id,
            binder,
            symbol,
            zero,
            repeat_end,
            &mut replacements,
        );
        assert_eq!(
            replacements, 1,
            "a chunkable semantic launch must occur exactly once"
        );
        Ok(())
    }

    /// A construction-proven per-axis envelope for resources reserved across
    /// repeated chunks. Execution retains its exact tail grid; scratch addresses
    /// are physical per-launch participant indices, never logical-base indices.
    pub fn launch_grid_envelope(&self, arena: &mut ExprArena, launch: &Launch<B>) -> [NatExpr; 3] {
        let mut grid = launch.grid;
        if let Some(bounds) = launch.logical_base.and_then(|base| base.chunk_bounds) {
            let limit = arena.nat(bounds.grid_cap);
            let participants = arena.nat_product(&launch.workgroup);
            let groups = arena.nat_ceil_div(
                launch
                    .parallel_extent
                    .expect("chunked launch has semantic extent"),
                participants,
            );
            grid[0] = arena.nat_min(groups, limit);
        }
        grid
    }

    /// Maximum reservation for one launch over its enclosing lexical loops.
    /// Runtime branches reserve both possible arms. Empty loops do not evaluate
    /// the body requirement.
    fn occurrence_reservation(
        &self,
        arena: &mut ExprArena,
        occurrence: &impl Fn(&ScheduleStep) -> bool,
        bytes: NatExpr,
    ) -> Option<NatExpr> {
        fn visit(
            arena: &mut ExprArena,
            steps: &[ScheduleStep],
            occurrence: &impl Fn(&ScheduleStep) -> bool,
            bytes: NatExpr,
        ) -> Option<NatExpr> {
            let mut maximum = None;
            for step in steps {
                let value = match step {
                    step if occurrence(step) => Some(bytes),
                    ScheduleStep::Imported { body, .. } => visit(arena, body, occurrence, bytes),
                    ScheduleStep::If {
                        then_steps,
                        else_steps,
                        ..
                    } => {
                        match (
                            visit(arena, then_steps, occurrence, bytes),
                            visit(arena, else_steps, occurrence, bytes),
                        ) {
                            (Some(a), Some(b)) => Some(arena.nat_max(a, b)),
                            (a, b) => a.or(b),
                        }
                    }
                    ScheduleStep::Repeat {
                        binder,
                        symbol,
                        start,
                        end,
                        body,
                        ..
                    } => visit(arena, body, occurrence, bytes).map(|value| {
                        if matches!(
                            arena.view(value.into()),
                            seismic_lang::expr::NodeView::NatConst(0)
                        ) {
                            return value;
                        }
                        let zero = arena.nat(0);
                        let nonempty = arena.nat_cmp(seismic_lang::expr::CmpOp::Gt, *end, *start);
                        if !arena.free_symbols(value.into()).contains(symbol) {
                            return arena.nat_select(nonempty, value, zero);
                        }
                        let extent = arena.nat_sub(*end, *start);
                        let maximum = arena.nat_fold_range(
                            seismic_lang::expr::FoldOp::Max,
                            *binder,
                            *start,
                            extent,
                            value,
                        );
                        arena.nat_select(nonempty, maximum, zero)
                    }),
                    ScheduleStep::Launch(_)
                    | ScheduleStep::BeginAllocationInstance { .. }
                    | ScheduleStep::BindArgumentTensor { .. }
                    | ScheduleStep::PublishTensor { .. }
                    | ScheduleStep::Copy(_)
                    | ScheduleStep::Fill(_)
                    | ScheduleStep::ScalarMove(_)
                    | ScheduleStep::EvaluateHost(_)
                    | ScheduleStep::ScalarRead(_)
                    | ScheduleStep::Check(_) => None,
                };
                if let Some(value) = value {
                    maximum = Some(match maximum {
                        Some(previous) => arena.nat_max(previous, value),
                        None => value,
                    });
                }
            }
            maximum
        }
        visit(arena, &self.steps, occurrence, bytes)
    }

    pub(crate) fn launch_reservation(
        &self,
        arena: &mut ExprArena,
        ordinal: usize,
        bytes: NatExpr,
    ) -> NatExpr {
        self.occurrence_reservation(
            arena,
            &|step| matches!(step, ScheduleStep::Launch(id) if id.index() as usize == ordinal),
            bytes,
        )
        .expect("closed launch has no schedule occurrence")
    }

    pub(crate) fn allocation_reservation(
        &self,
        arena: &mut ExprArena,
        views: &[BufferViewLayout],
        ordinal: usize,
        bytes: NatExpr,
    ) -> NatExpr {
        self.occurrence_reservation(arena, &|step| match step {
            ScheduleStep::BeginAllocationInstance { source: view, .. } => matches!(views[view.index() as usize].base, crate::storage::ViewBase::Allocation(id) if id.index() as usize == ordinal),
            _ => false,
        }, bytes).unwrap_or(bytes)
    }

    /// Close obligations at their actual lexical use, including zero-trip
    /// repeats and the distinct initial/backedge scopes of carried products.
    /// `established` supplies only postconditions guaranteed by successful
    /// completion of that step. They justify later dominated uses, including
    /// uses inside later compound regions, and expire when a slot they
    /// reference may be written: by a leaf, anywhere inside a compound region
    /// (on entry, since a repeat revisits its body), or by a native launch
    /// (native scalar slots). A branch arm also knows its condition.
    pub fn scoped_requirements(
        &self,
        arena: &mut ExprArena,
        views: &[BufferViewLayout],
        requirement: &mut impl FnMut(&mut ExprArena, RequirementPoint<'_>) -> BoolExpr,
        established: &mut impl FnMut(&mut ExprArena, &ScheduleStep) -> BoolExpr,
    ) -> BoolExpr {
        use crate::region::{ValueDestination, ValueOperand};
        fn destination(value: ValueDestination, out: &mut Vec<SymbolId>) {
            match value {
                ValueDestination::Scalar(slot) => out.push(slot.symbol()),
                ValueDestination::Quantity(slot) => out.push(slot.symbol()),
                // Fresh geometry slots, created after every earlier fact.
                ValueDestination::Tensor(_) => {}
            }
        }
        /// Every slot a step may write.
        fn written(step: &ScheduleStep, native: &[SymbolId], out: &mut Vec<SymbolId>) {
            match step {
                ScheduleStep::ScalarMove(value) => out.push(value.to.symbol()),
                ScheduleStep::ScalarRead(value) => out.push(value.to.symbol()),
                ScheduleStep::EvaluateHost(value) => out.push(match value.to {
                    HostValueDestination::Quantity(slot) => slot.symbol(),
                    HostValueDestination::Native(slot) => slot.symbol(),
                }),
                ScheduleStep::Launch(_) => out.extend_from_slice(native),
                ScheduleStep::Imported { body, .. } => {
                    body.iter().for_each(|step| written(step, native, out))
                }
                ScheduleStep::If {
                    then_steps,
                    else_steps,
                    results,
                    ..
                } => {
                    then_steps
                        .iter()
                        .chain(else_steps)
                        .for_each(|step| written(step, native, out));
                    results.visit(&mut |result| destination(result.result(), out));
                }
                ScheduleStep::Repeat { body, carries, .. } => {
                    body.iter().for_each(|step| written(step, native, out));
                    carries.visit(&mut |carry| {
                        destination(carry.header(), out);
                        destination(carry.result(), out);
                    });
                }
                ScheduleStep::BeginAllocationInstance { .. }
                | ScheduleStep::BindArgumentTensor { .. }
                | ScheduleStep::PublishTensor { .. }
                | ScheduleStep::Copy(_)
                | ScheduleStep::Fill(_)
                | ScheduleStep::Check(_) => {}
            }
        }
        fn expire(arena: &ExprArena, facts: &mut Vec<BoolExpr>, written: &[SymbolId]) {
            facts.retain(|fact| {
                !arena
                    .free_symbols((*fact).into())
                    .iter()
                    .any(|symbol| written.contains(symbol))
            });
        }
        /// Product-level invariant: a tensor destination axis whose every
        /// incoming value has the same geometry expression, over operands the
        /// region does not write, equals that expression wherever the
        /// destination is in scope.
        fn product_axes(
            arena: &mut ExprArena,
            views: &[BufferViewLayout],
            sources: [ValueOperand; 2],
            destinations: &[ValueDestination],
            writes: &[SymbolId],
            equal: &[(SymbolId, NatExpr)],
            out: &mut Vec<(SymbolId, NatExpr)>,
        ) {
            let (ValueOperand::Tensor(a), ValueOperand::Tensor(b)) = (sources[0], sources[1])
            else {
                return;
            };
            let (a, b) = (&views[a.index() as usize], &views[b.index() as usize]);
            let axes = |layout: &BufferViewLayout| {
                layout
                    .extents
                    .iter()
                    .chain(&layout.strides)
                    .copied()
                    .collect::<Vec<_>>()
            };
            let (a, b) = (axes(a), axes(b));
            for (axis, (value, other)) in a.iter().zip(&b).enumerate() {
                if value != other
                    || arena
                        .free_symbols((*value).into())
                        .iter()
                        .any(|symbol| writes.contains(symbol))
                {
                    continue;
                }
                let value = arena.substitute_nat(*value, equal);
                for destination in destinations {
                    let ValueDestination::Tensor(view) = destination else {
                        continue;
                    };
                    let slot = axes(&views[view.index() as usize])[axis];
                    let seismic_lang::expr::NodeView::Symbol(symbol) = arena.view(slot.into())
                    else {
                        panic!("region product geometry is not its destination slot")
                    };
                    out.push((symbol, value));
                }
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn visit(
            arena: &mut ExprArena,
            views: &[BufferViewLayout],
            steps: &[ScheduleStep],
            mut established_facts: Vec<BoolExpr>,
            mut equal: Vec<(SymbolId, NatExpr)>,
            native: &[SymbolId],
            requirement: &mut impl FnMut(&mut ExprArena, RequirementPoint<'_>) -> BoolExpr,
            established: &mut impl FnMut(&mut ExprArena, &ScheduleStep) -> BoolExpr,
        ) -> BoolExpr {
            let mut terms = Vec::new();
            for step in steps {
                let mut writes = Vec::new();
                written(step, native, &mut writes);
                let compound = matches!(
                    step,
                    ScheduleStep::If { .. }
                        | ScheduleStep::Repeat { .. }
                        | ScheduleStep::Imported { .. }
                );
                if compound {
                    expire(arena, &mut established_facts, &writes);
                }
                let facts = arena.all(&established_facts);
                let require = |arena: &mut ExprArena,
                               requirement: &mut dyn FnMut(
                    &mut ExprArena,
                    RequirementPoint<'_>,
                ) -> BoolExpr,
                               point: RequirementPoint<'_>,
                               equal: &[(SymbolId, NatExpr)]| {
                    let term = requirement(arena, point);
                    arena.substitute_nat(term, equal)
                };
                let term = match step {
                    ScheduleStep::Imported { body, .. } => visit(
                        arena,
                        views,
                        body,
                        established_facts.clone(),
                        equal.clone(),
                        native,
                        requirement,
                        established,
                    ),
                    ScheduleStep::If {
                        condition,
                        then_steps,
                        else_steps,
                        results,
                    } => {
                        let condition = arena.substitute_nat(*condition, &equal);
                        let mut then_facts = established_facts.clone();
                        then_facts.push(condition);
                        let a = visit(
                            arena,
                            views,
                            then_steps,
                            then_facts,
                            equal.clone(),
                            native,
                            requirement,
                            established,
                        );
                        let yielded_a = require(
                            arena,
                            requirement,
                            RequirementPoint::BranchResult(results, true),
                            &equal,
                        );
                        let a = arena.and(a, yielded_a);
                        let mut else_facts = established_facts.clone();
                        else_facts.push(arena.not(condition));
                        let b = visit(
                            arena,
                            views,
                            else_steps,
                            else_facts,
                            equal.clone(),
                            native,
                            requirement,
                            established,
                        );
                        let yielded_b = require(
                            arena,
                            requirement,
                            RequirementPoint::BranchResult(results, false),
                            &equal,
                        );
                        let b = arena.and(b, yielded_b);
                        let mut joined = Vec::new();
                        results.visit(&mut |result| {
                            product_axes(
                                arena,
                                views,
                                [result.then_value(), result.else_value()],
                                &[result.result()],
                                &writes,
                                &equal,
                                &mut joined,
                            )
                        });
                        let term = arena.and(a, b);
                        equal.extend(joined);
                        term
                    }
                    ScheduleStep::Repeat {
                        binder,
                        symbol,
                        start,
                        end,
                        body,
                        carries,
                        ..
                    } => {
                        let initial = require(
                            arena,
                            requirement,
                            RequirementPoint::RepeatInitial(carries),
                            &equal,
                        );
                        terms.push(arena.implies(facts, initial));
                        let mut header = equal.clone();
                        let mut result = Vec::new();
                        carries.visit(&mut |carry| {
                            product_axes(
                                arena,
                                views,
                                [carry.initial(), carry.backedge()],
                                &[carry.header()],
                                &writes,
                                &equal,
                                &mut header,
                            );
                            product_axes(
                                arena,
                                views,
                                [carry.initial(), carry.backedge()],
                                &[carry.result()],
                                &writes,
                                &equal,
                                &mut result,
                            );
                        });
                        let body = visit(
                            arena,
                            views,
                            body,
                            established_facts.clone(),
                            header.clone(),
                            native,
                            requirement,
                            established,
                        );
                        let backedge = require(
                            arena,
                            requirement,
                            RequirementPoint::RepeatBackedge(carries),
                            &header,
                        );
                        let body = arena.and(body, backedge);
                        equal.extend(result);
                        if matches!(
                            arena.view(body.into()),
                            seismic_lang::expr::NodeView::BoolConst(true)
                        ) {
                            body
                        } else {
                            let zero = arena.nat(0);
                            let one = arena.nat(1);
                            let nonempty =
                                arena.nat_cmp(seismic_lang::expr::CmpOp::Gt, *end, *start);
                            if !arena.free_symbols(body.into()).contains(symbol) {
                                let body = arena.implies(nonempty, body);
                                terms.push(arena.implies(facts, body));
                                continue;
                            }
                            let extent = arena.nat_sub(*end, *start);
                            let violation = arena.nat_select(body, zero, one);
                            let worst = arena.nat_fold_range(
                                seismic_lang::expr::FoldOp::Max,
                                *binder,
                                *start,
                                extent,
                                violation,
                            );
                            let holds = arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, worst, zero);
                            arena.implies(nonempty, holds)
                        }
                    }
                    leaf => require(arena, requirement, RequirementPoint::Step(leaf), &equal),
                };
                terms.push(arena.implies(facts, term));
                // Only the completed leaf's own postcondition is exported.
                // Branch/loop-local facts never escape their lexical region.
                if !compound {
                    expire(arena, &mut established_facts, &writes);
                    let fact = established(arena, step);
                    established_facts.push(arena.substitute_nat(fact, &equal));
                }
            }
            arena.all(&terms)
        }
        let native = self
            .slots
            .iter()
            .map(|slot| slot.symbol())
            .collect::<Vec<_>>();
        visit(
            arena,
            views,
            &self.steps,
            Vec::new(),
            Vec::new(),
            &native,
            requirement,
            established,
        )
    }

    pub fn retained_bytes(&self) -> usize {
        fn steps(values: &[ScheduleStep]) -> usize {
            values
                .len()
                .saturating_mul(std::mem::size_of::<ScheduleStep>())
                .saturating_add(values.iter().fold(0usize, |bytes, step| {
                    bytes.saturating_add(match step {
                        ScheduleStep::If {
                            then_steps,
                            else_steps,
                            results,
                            ..
                        } => steps(then_steps)
                            .saturating_add(steps(else_steps))
                            .saturating_add(results.retained_heap_bytes(&|_| 0)),
                        ScheduleStep::Imported { scope, body } => steps(body).saturating_add(
                            scope.at.region.capacity() * std::mem::size_of::<ScheduleRegionEdge>(),
                        ),
                        ScheduleStep::Repeat { body, carries, .. } => {
                            steps(body).saturating_add(carries.retained_heap_bytes(&|_| 0))
                        }
                        ScheduleStep::ScalarRead(read) => read
                            .index
                            .capacity()
                            .saturating_mul(std::mem::size_of::<NatExpr>())
                            .saturating_add(
                                read.bounds
                                    .capacity()
                                    .saturating_mul(std::mem::size_of::<BoolExpr>()),
                            ),
                        ScheduleStep::EvaluateHost(value) => value
                            .failure
                            .as_ref()
                            .map_or(0, |site| site.path.capacity()),
                        _ => 0,
                    })
                }))
        }
        let use_paths = self
            .direct_view_uses
            .iter()
            .map(|(_, usage)| usage.region.capacity() * std::mem::size_of::<ScheduleRegionEdge>())
            .chain(self.launch_uses.iter().map(|(_, usage)| {
                usage.region.capacity() * std::mem::size_of::<ScheduleRegionEdge>()
            }))
            .sum::<usize>();
        self.launches.capacity() * std::mem::size_of::<Launch<B>>()
            + self.slots.capacity() * std::mem::size_of::<AnyScalarSlot>()
            + self.quantity_slots.capacity() * std::mem::size_of::<HostQuantitySlot>()
            + steps(&self.steps)
            + self.direct_view_uses.capacity() * std::mem::size_of::<(AnyBufferView, ScheduleUse)>()
            + self.launch_uses.capacity() * std::mem::size_of::<(LaunchId, ScheduleUse)>()
            + use_paths
    }
    pub fn launches(&self) -> &[Launch<B>] {
        &self.launches
    }
    pub fn launch(&self, id: LaunchId) -> &Launch<B> {
        self.assert_owner(id.owner());
        &self.launches[id.index() as usize]
    }
    pub fn slots(&self) -> &[AnyScalarSlot] {
        &self.slots
    }
    pub fn quantity_slots(&self) -> &[HostQuantitySlot] {
        &self.quantity_slots
    }
    pub fn steps(&self) -> &[ScheduleStep] {
        &self.steps
    }
    pub(crate) fn owner(&self) -> OwnerToken {
        self.owner
    }
    pub fn direct_view_uses(&self) -> &[(AnyBufferView, ScheduleUse)] {
        &self.direct_view_uses
    }
    pub fn launch_uses(&self) -> &[(LaunchId, ScheduleUse)] {
        &self.launch_uses
    }
    fn assert_owner(&self, owner: OwnerToken) {
        assert_eq!(
            owner, self.owner,
            "schedule handle belongs to another implementation"
        )
    }
}

fn rewrite_launch_as_repeat(
    steps: &mut Vec<ScheduleStep>,
    target: LaunchId,
    binder: LoopBinderId,
    symbol: SymbolId,
    start: NatExpr,
    end: NatExpr,
    replacements: &mut usize,
) {
    for step in steps {
        match step {
            ScheduleStep::Launch(id) if *id == target => {
                // Chunks of one launch; the body cannot stop at a source failure.
                *step = ScheduleStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    visits: RepeatVisits::Ordered,
                    body: vec![ScheduleStep::Launch(target)],
                    carries: Product::Unit,
                };
                *replacements += 1;
            }
            ScheduleStep::If {
                then_steps,
                else_steps,
                ..
            } => {
                rewrite_launch_as_repeat(
                    then_steps,
                    target,
                    binder,
                    symbol,
                    start,
                    end,
                    replacements,
                );
                rewrite_launch_as_repeat(
                    else_steps,
                    target,
                    binder,
                    symbol,
                    start,
                    end,
                    replacements,
                );
            }
            ScheduleStep::Imported { body, .. } | ScheduleStep::Repeat { body, .. } => {
                rewrite_launch_as_repeat(body, target, binder, symbol, start, end, replacements)
            }
            _ => {}
        }
    }
}

pub struct ScheduleConstruction<B: PhysicalDialect> {
    owner: OwnerToken,
    launches: Vec<Launch<B>>,
    slots: Vec<AnyScalarSlot>,
    quantity_slots: Vec<HostQuantitySlot>,
    next_slot_symbol: u32,
    regions: Vec<Region>,
    next_control: u32,
    closed: bool,
    imported_direct: Vec<(AnyBufferView, ScheduleUse)>,
    imported_launch: Vec<(LaunchId, ScheduleUse)>,
    marker: PhantomData<B>,
}
struct Region {
    path: Vec<ScheduleRegionEdge>,
    steps: Vec<RegionStep>,
}
enum RegionStep {
    Leaf(ScheduleStep),
    Imported {
        scope: ImportScope,
        body: Vec<ScheduleStep>,
    },
    If {
        condition: BoolExpr,
        then_region: u32,
        else_region: u32,
        results: Product<BranchResult>,
    },
    Repeat {
        binder: LoopBinderId,
        symbol: SymbolId,
        start: NatExpr,
        end: NatExpr,
        visits: RepeatVisits,
        body_region: u32,
        carries: Option<Product<RepeatCarry>>,
    },
}

impl<B: PhysicalDialect> ScheduleConstruction<B> {
    pub(crate) fn new(owner: OwnerToken) -> Self {
        Self {
            owner,
            launches: Vec::new(),
            slots: Vec::new(),
            quantity_slots: Vec::new(),
            next_slot_symbol: 0,
            regions: vec![Region {
                path: Vec::new(),
                steps: Vec::new(),
            }],
            next_control: 0,
            closed: false,
            imported_direct: Vec::new(),
            imported_launch: Vec::new(),
            marker: PhantomData,
        }
    }
    pub(crate) fn builder_at<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        views: &'a [crate::storage::BufferViewLayout],
        region: u32,
    ) -> ScheduleBuilder<'a, B> {
        assert!(!self.closed, "a closed schedule cannot be extended");
        assert!(
            (region as usize) < self.regions.len(),
            "schedule region is not owned by this construction"
        );
        ScheduleBuilder {
            inner: internals::Builder {
                arena,
                views,
                state: self,
                region,
            },
        }
    }
    pub fn begin_branch(&mut self, parent: u32, condition: BoolExpr) -> (u32, u32) {
        assert!(!self.closed, "a closed schedule cannot be extended");
        let control = self.next_control;
        self.next_control = self
            .next_control
            .checked_add(1)
            .unwrap_or_else(|| panic!("schedule control identity space exhausted"));
        let parent_ordinal = self.regions[parent as usize].steps.len() as u32;
        let mut then_path = self.regions[parent as usize].path.clone();
        then_path.push(ScheduleRegionEdge::IfThen {
            node: control,
            parent_ordinal,
        });
        let then_region = self.regions.len() as u32;
        self.regions.push(Region {
            path: then_path,
            steps: Vec::new(),
        });
        let mut else_path = self.regions[parent as usize].path.clone();
        else_path.push(ScheduleRegionEdge::IfElse {
            node: control,
            parent_ordinal,
        });
        let else_region = self.regions.len() as u32;
        self.regions.push(Region {
            path: else_path,
            steps: Vec::new(),
        });
        self.regions[parent as usize].steps.push(RegionStep::If {
            condition,
            then_region,
            else_region,
            results: Product::Unit,
        });
        (then_region, else_region)
    }
    pub(crate) fn finish_branch_products(
        &mut self,
        parent: u32,
        then_region: u32,
        products: Product<BranchResult>,
    ) {
        let step = self.regions[parent as usize]
            .steps
            .iter_mut()
            .find(|step| {
                matches!(step,
            RegionStep::If { then_region: region, .. } if *region == then_region)
            })
            .expect("branch belongs to parent region");
        let RegionStep::If { results, .. } = step else {
            unreachable!()
        };
        assert!(
            matches!(results, Product::Unit),
            "branch results already closed"
        );
        *results = products;
    }

    pub fn begin_repeat(
        &mut self,
        arena: &mut ExprArena,
        parent: u32,
        start: NatExpr,
        end: NatExpr,
        visits: RepeatVisits,
    ) -> (u32, LoopBinding) {
        assert!(!self.closed, "a closed schedule cannot be extended");
        let control = self.next_control;
        self.next_control = self
            .next_control
            .checked_add(1)
            .unwrap_or_else(|| panic!("schedule control identity space exhausted"));
        let parent_ordinal = self.regions[parent as usize].steps.len() as u32;
        let mut path = self.regions[parent as usize].path.clone();
        path.push(ScheduleRegionEdge::RepeatBody {
            node: control,
            parent_ordinal,
        });
        let body_region = self.regions.len() as u32;
        self.regions.push(Region {
            path,
            steps: Vec::new(),
        });
        let (binder, symbol, index) = arena.nat_loop_binder();
        self.regions[parent as usize]
            .steps
            .push(RegionStep::Repeat {
                binder,
                symbol,
                start,
                end,
                visits,
                body_region,
                carries: Some(Product::Unit),
            });
        (
            body_region,
            LoopBinding {
                id: binder,
                symbol,
                index,
            },
        )
    }
    pub fn possible_backing_allocations(
        &self,
        views: &[BufferViewLayout],
        view: AnyBufferView,
    ) -> Option<Vec<crate::storage::GlobalAllocationId>> {
        let mut products = Vec::new();
        let mut definitions = Vec::new();
        for region in &self.regions {
            for step in &region.steps {
                match step {
                    RegionStep::Leaf(
                        ScheduleStep::BeginAllocationInstance { source, result }
                        | ScheduleStep::BindArgumentTensor { source, result },
                    ) => definitions.push((*source, *result)),
                    RegionStep::If { results, .. } => results.visit(&mut |result| {
                        if let crate::region::ValueDestination::Tensor(destination) =
                            result.result()
                        {
                            for value in [result.then_value(), result.else_value()] {
                                let crate::region::ValueOperand::Tensor(source) = value else {
                                    unreachable!("branch product kind")
                                };
                                definitions.push((source, destination));
                            }
                        }
                    }),
                    RegionStep::Repeat {
                        carries: Some(carries),
                        ..
                    } => crate::region::carry_leaves(carries, &mut products),
                    RegionStep::Imported { body, .. } => {
                        crate::region::collect_carries(body, &mut products);
                        crate::region::collect_tensor_definitions(body, &mut definitions)
                    }
                    _ => (),
                }
            }
        }
        crate::region::possible_backings(views, view, &products, &definitions)
    }
    pub(crate) fn open_repeat_products(&mut self, body: u32) {
        for region in &mut self.regions {
            for step in &mut region.steps {
                if let RegionStep::Repeat {
                    body_region,
                    carries,
                    ..
                } = step
                {
                    if *body_region == body {
                        assert!(matches!(carries, Some(Product::Unit)));
                        *carries = None;
                        return;
                    }
                }
            }
        }
        panic!("repeat body belongs to another construction");
    }
    pub(crate) fn finish_repeat_products(&mut self, body: u32, carries: Product<RepeatCarry>) {
        for region in &mut self.regions {
            for step in &mut region.steps {
                if let RegionStep::Repeat {
                    body_region,
                    carries: current,
                    ..
                } = step
                {
                    if *body_region == body {
                        assert!(current.is_none(), "repeat product already finished");
                        *current = Some(carries);
                        return;
                    }
                }
            }
        }
        panic!("repeat body belongs to another construction");
    }
    pub fn slot_any(&mut self, arena: &mut ExprArena, kind: ScalarKind) -> AnyScalarSlot {
        assert!(
            !self.closed,
            "a closed schedule cannot allocate a result slot"
        );
        let index = self.slots.len() as u32;
        let symbol = arena.schedule_slot(self.next_slot_symbol, kind.sort());
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        let slot = AnyScalarSlot {
            owner: self.owner,
            index,
            kind,
            symbol,
            capture: None,
        };
        self.slots.push(slot);
        slot
    }
    pub fn quantity_slot(
        &mut self,
        arena: &mut ExprArena,
        kind: HostQuantityKind,
    ) -> HostQuantitySlot {
        assert!(
            !self.closed,
            "a closed schedule cannot allocate a quantity slot"
        );
        let index = self.quantity_slots.len() as u32;
        let symbol = arena.schedule_slot(self.next_slot_symbol, kind.sort());
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        let slot = HostQuantitySlot {
            owner: self.owner,
            index,
            kind,
            symbol,
            capture: None,
        };
        self.quantity_slots.push(slot);
        slot
    }
    /// Transfer a scalar publication into this construction while preserving its
    /// expression identity. Slot ordinals are local; symbols belong to the shared
    /// expression arena and are already referenced by kernels and control flow.
    /// Borrow an ancestor's already-bound slot into a call body. Its symbol
    /// remains owned by the ancestor and is recovered when that body is spliced.
    pub fn capture_slot(&mut self, source: AnyScalarSlot) -> AnyScalarSlot {
        assert!(
            !self.closed,
            "a closed schedule cannot capture a result slot"
        );
        assert_ne!(
            source.owner, self.owner,
            "capture must come from another construction"
        );
        let slot = AnyScalarSlot {
            owner: self.owner,
            index: self.slots.len() as u32,
            capture: Some(source.capture.unwrap_or(SlotCapture {
                owner: source.owner,
                index: source.index,
            })),
            ..source
        };
        self.slots.push(slot);
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        slot
    }
    pub fn capture_quantity_slot(&mut self, source: HostQuantitySlot) -> HostQuantitySlot {
        assert!(
            !self.closed,
            "a closed schedule cannot capture a quantity slot"
        );
        assert_ne!(
            source.owner, self.owner,
            "capture must come from another construction"
        );
        let mut slot = source.remap(self.owner, self.quantity_slots.len() as u32);
        slot.capture = Some(source.capture.unwrap_or(SlotCapture {
            owner: source.owner,
            index: source.index,
        }));
        self.quantity_slots.push(slot);
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        slot
    }
    fn splice_slot(&mut self, arena: &mut ExprArena, source: AnyScalarSlot) -> AnyScalarSlot {
        assert!(
            !self.closed,
            "a closed schedule cannot splice a result slot"
        );
        if let Some(capture) = source.capture {
            if capture.owner == self.owner {
                let original = self.slots[capture.index as usize];
                assert_eq!(
                    (original.symbol, original.kind),
                    (source.symbol, source.kind),
                    "captured slot changed before its owner was reached"
                );
                return original;
            }
            if let Some(existing) = self.slots.iter().find(|slot| slot.capture == Some(capture)) {
                assert_eq!(
                    (existing.symbol, existing.kind),
                    (source.symbol, source.kind),
                    "one borrowed slot has conflicting definitions"
                );
                return *existing;
            }
        } else {
            arena.rebase_schedule_slot(source.symbol, self.next_slot_symbol);
        }
        let slot = source.remap(self.owner, self.slots.len() as u32);
        self.slots.push(slot);
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        slot
    }
    fn splice_quantity_slot(
        &mut self,
        arena: &mut ExprArena,
        source: HostQuantitySlot,
    ) -> HostQuantitySlot {
        assert!(
            !self.closed,
            "a closed schedule cannot splice a quantity slot"
        );
        if let Some(capture) = source.capture {
            if capture.owner == self.owner {
                let original = self.quantity_slots[capture.index as usize];
                assert_eq!(
                    (original.symbol, original.kind),
                    (source.symbol, source.kind),
                    "captured quantity changed before its owner was reached"
                );
                return original;
            }
            if let Some(existing) = self
                .quantity_slots
                .iter()
                .find(|slot| slot.capture == Some(capture))
            {
                assert_eq!(
                    (existing.symbol, existing.kind),
                    (source.symbol, source.kind),
                    "one borrowed quantity has conflicting definitions"
                );
                return *existing;
            }
        } else {
            arena.rebase_schedule_slot(source.symbol, self.next_slot_symbol);
        }
        let slot = source.remap(self.owner, self.quantity_slots.len() as u32);
        self.quantity_slots.push(slot);
        self.next_slot_symbol = self
            .next_slot_symbol
            .checked_add(1)
            .expect("schedule slot symbol space exhausted");
        slot
    }

    pub(crate) fn finish(mut self, token: ClosedSchedule) -> ParametricSchedule<B> {
        assert_eq!(
            token.owner, self.owner,
            "closed schedule belongs to another implementation"
        );
        assert!(
            self.closed,
            "schedule close token was not produced by this builder"
        );
        let mut direct_view_uses = Vec::new();
        let mut launch_uses = Vec::new();
        let steps = self.lower_region(0, &mut direct_view_uses, &mut launch_uses);
        direct_view_uses.extend(std::mem::take(&mut self.imported_direct));
        launch_uses.extend(std::mem::take(&mut self.imported_launch));
        ParametricSchedule {
            owner: self.owner,
            next_control: self.next_control,
            launches: self.launches,
            slots: self.slots,
            quantity_slots: self.quantity_slots,
            steps,
            direct_view_uses,
            launch_uses,
        }
    }
    fn lower_region(
        &mut self,
        region: u32,
        direct: &mut Vec<(AnyBufferView, ScheduleUse)>,
        launches: &mut Vec<(LaunchId, ScheduleUse)>,
    ) -> Vec<ScheduleStep> {
        let path = self.regions[region as usize].path.clone();
        let raw = std::mem::take(&mut self.regions[region as usize].steps);
        let mut out = Vec::with_capacity(raw.len());
        for (ordinal, step) in raw.into_iter().enumerate() {
            let at = ScheduleUse {
                owner: self.owner,
                region: path.clone(),
                ordinal: ordinal as u32,
            };
            match step {
                RegionStep::Leaf(step) => {
                    match &step {
                        ScheduleStep::BeginAllocationInstance { source: view, .. }
                        | ScheduleStep::BindArgumentTensor { source: view, .. }
                        | ScheduleStep::PublishTensor { view, .. } => direct.push((*view, at)),
                        ScheduleStep::Launch(id) => launches.push((*id, at)),
                        ScheduleStep::Copy(copy) => {
                            direct.push((copy.source, at.clone()));
                            direct.push((copy.destination, at));
                        }
                        ScheduleStep::Fill(fill) => direct.push((fill.destination, at)),
                        ScheduleStep::ScalarRead(read) => direct.push((read.source, at)),
                        ScheduleStep::ScalarMove(_)
                        | ScheduleStep::EvaluateHost(_)
                        | ScheduleStep::Check(_) => {}
                        ScheduleStep::Imported { .. }
                        | ScheduleStep::If { .. }
                        | ScheduleStep::Repeat { .. } => {
                            panic!("structured control cannot be inserted as a leaf")
                        }
                    }
                    out.push(step);
                }
                RegionStep::Imported { scope, body } => {
                    out.push(ScheduleStep::Imported { scope, body })
                }
                RegionStep::If {
                    condition,
                    then_region,
                    else_region,
                    results,
                } => {
                    for (region, then) in [(then_region, true), (else_region, false)] {
                        let at = ScheduleUse {
                            owner: self.owner,
                            region: self.regions[region as usize].path.clone(),
                            ordinal: self.regions[region as usize].steps.len() as u32,
                        };
                        results.visit(&mut |result| {
                            if let crate::region::ValueOperand::Tensor(view) = if then {
                                result.then_value()
                            } else {
                                result.else_value()
                            } {
                                direct.push((view, at.clone()));
                            }
                        });
                    }
                    out.push(ScheduleStep::If {
                        condition,
                        then_steps: self.lower_region(then_region, direct, launches),
                        else_steps: self.lower_region(else_region, direct, launches),
                        results,
                    });
                }
                RegionStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    visits,
                    body_region,
                    carries,
                } => {
                    let carries = carries.expect("unfinished repeat product cannot close");
                    let backedge_use = ScheduleUse {
                        owner: self.owner,
                        region: self.regions[body_region as usize].path.clone(),
                        ordinal: self.regions[body_region as usize].steps.len() as u32,
                    };
                    // Transport itself retains storage across the entire recurrence,
                    // even when no body leaf explicitly reads the carried view.
                    carries.visit(&mut |carry| {
                        for value in [carry.initial(), carry.backedge()] {
                            if let crate::region::ValueOperand::Tensor(view) = value {
                                direct.push((view, at.clone()));
                                direct.push((view, backedge_use.clone()));
                            }
                        }
                        for value in [carry.header(), carry.result()] {
                            if let crate::region::ValueDestination::Tensor(view) = value {
                                direct.push((view, at.clone()));
                                direct.push((view, backedge_use.clone()));
                            }
                        }
                    });
                    out.push(ScheduleStep::Repeat {
                        binder,
                        symbol,
                        start,
                        end,
                        visits,
                        body: self.lower_region(body_region, direct, launches),
                        carries,
                    });
                }
            }
        }
        out
    }

    pub(crate) fn import(
        &mut self,
        arena: &mut ExprArena,
        child: ParametricSchedule<B>,
        kernels: &[KernelId],
        views: &[AnyBufferView],
        forced_slots: &[(u32, AnyScalarSlot)],
    ) -> ImportedSchedule {
        assert_ne!(
            child.owner, self.owner,
            "child schedule must have a distinct owner"
        );
        for (index, mapped) in forced_slots {
            assert!(
                (*index as usize) < child.slots.len(),
                "forced child slot is out of range"
            );
            assert_eq!(
                mapped.owner(),
                self.owner,
                "forced slot belongs to another construction"
            );
            assert_eq!(
                self.slots.get(mapped.index as usize),
                Some(mapped),
                "forced slot is not registered"
            );
            assert_eq!(
                mapped.symbol, child.slots[*index as usize].symbol,
                "callee publication must use the caller destination identity"
            );
        }
        let mut slots = Vec::with_capacity(child.slots.len());
        for child_slot in &child.slots {
            let mapped = forced_slots
                .iter()
                .find_map(|(index, slot)| (*index == child_slot.index).then_some(*slot))
                .unwrap_or_else(|| self.splice_slot(arena, *child_slot));
            assert_eq!(
                mapped.kind, child_slot.kind,
                "spliced scalar-slot kind mismatch"
            );
            slots.push(mapped);
        }
        let quantity_slots = child
            .quantity_slots
            .iter()
            .copied()
            .map(|slot| self.splice_quantity_slot(arena, slot))
            .collect::<Vec<_>>();
        let mut launches = Vec::with_capacity(child.launches.len());
        for launch in child.launches {
            let kernel = *kernels
                .get(launch.kernel.index() as usize)
                .expect("child kernel remap is complete");
            let id = LaunchId::new(self.owner, self.launches.len() as u32);
            let logical_base = launch.logical_base.map(|mut base| {
                base.binding.kernel = kernel;
                base
            });
            self.launches.push(Launch {
                kernel,
                logical_base,
                ..launch
            });
            launches.push(id);
        }
        let map_view = |view: AnyBufferView| {
            *views
                .get(view.index as usize)
                .expect("child view remap is complete")
        };
        let map_slot = |slot: AnyScalarSlot| slots[slot.index as usize];
        let map_quantity_slot = |slot: HostQuantitySlot| quantity_slots[slot.index as usize];
        let map_launch = |launch: LaunchId| launches[launch.index() as usize];
        let steps = remap_steps(
            child.steps,
            map_view,
            map_slot,
            map_quantity_slot,
            map_launch,
        );
        let direct = child
            .direct_view_uses
            .into_iter()
            .map(|(view, at)| (map_view(view), at))
            .collect();
        let launch_uses = child
            .launch_uses
            .into_iter()
            .map(|(launch, at)| (map_launch(launch), at))
            .collect();
        ImportedSchedule {
            owner: self.owner,
            source_owner: child.owner,
            views: views.to_vec(),
            steps,
            direct,
            launch_uses,
            slots,
            quantity_slots,
        }
    }

    pub(crate) fn append_imported_at(
        &mut self,
        parent: u32,
        imported: ImportedSchedule,
    ) -> ImportedBindings {
        assert_eq!(
            imported.owner, self.owner,
            "imported schedule belongs to another construction"
        );
        let node = self.next_control;
        self.next_control = self
            .next_control
            .checked_add(1)
            .unwrap_or_else(|| panic!("schedule control identity space exhausted"));
        let parent_ordinal = self.regions[parent as usize].steps.len() as u32;
        let scope = ImportScope {
            at: ScheduleUse {
                owner: self.owner,
                region: self.regions[parent as usize].path.clone(),
                ordinal: parent_ordinal,
            },
            node,
        };
        let prefix = scope.body_path();
        for (view, mut at) in imported.direct {
            at.owner = self.owner;
            let mut path = prefix.clone();
            path.extend(at.region);
            at.region = path;
            self.imported_direct.push((view, at));
        }
        for (launch, mut at) in imported.launch_uses {
            at.owner = self.owner;
            let mut path = prefix.clone();
            path.extend(at.region);
            at.region = path;
            self.imported_launch.push((launch, at));
        }
        let mut body = imported.steps;
        fn remap_scopes(steps: &mut [ScheduleStep], parent: &ImportScope) {
            for step in steps {
                match step {
                    ScheduleStep::Imported { scope, body } => {
                        *scope = parent.remap(scope);
                        remap_scopes(body, parent);
                    }
                    ScheduleStep::If {
                        then_steps,
                        else_steps,
                        ..
                    } => {
                        remap_scopes(then_steps, parent);
                        remap_scopes(else_steps, parent);
                    }
                    ScheduleStep::Repeat { body, .. } => remap_scopes(body, parent),
                    _ => {}
                }
            }
        }
        remap_scopes(&mut body, &scope);
        self.regions[parent as usize]
            .steps
            .push(RegionStep::Imported {
                scope: scope.clone(),
                body,
            });
        ImportedBindings {
            scope,
            source_owner: imported.source_owner,
            views: imported.views,
            slots: imported.slots,
            quantity_slots: imported.quantity_slots,
        }
    }
}

#[derive(Debug)]
pub(crate) struct ImportedSchedule {
    owner: OwnerToken,
    source_owner: OwnerToken,
    views: Vec<AnyBufferView>,
    pub(crate) steps: Vec<ScheduleStep>,
    pub(crate) direct: Vec<(AnyBufferView, ScheduleUse)>,
    pub(crate) launch_uses: Vec<(LaunchId, ScheduleUse)>,
    pub(crate) slots: Vec<AnyScalarSlot>,
    pub(crate) quantity_slots: Vec<HostQuantitySlot>,
}

/// Physical bindings and lexical scope returned by the atomic child import.
/// This projection owns no executable steps and cannot be spliced elsewhere.
#[derive(Debug)]
pub struct ImportedBindings {
    scope: ImportScope,
    source_owner: OwnerToken,
    views: Vec<AnyBufferView>,
    slots: Vec<AnyScalarSlot>,
    quantity_slots: Vec<HostQuantitySlot>,
}
impl ImportedBindings {
    pub fn scope(&self) -> &ImportScope {
        &self.scope
    }
    pub fn remap_scope(&self, child: &ImportScope) -> ImportScope {
        assert_eq!(
            child.at.owner, self.source_owner,
            "scope belongs to another imported child"
        );
        self.scope.remap(child)
    }
    /// Rebind an escaping child value using the same physical import that
    /// remapped its producing operations. No new allocation is introduced.
    pub fn remap_view(&self, source: AnyBufferView) -> AnyBufferView {
        assert_eq!(
            source.owner(),
            self.source_owner,
            "view belongs to another imported child"
        );
        self.views[source.index() as usize]
    }

    pub fn remap_slot(&self, source: AnyScalarSlot) -> AnyScalarSlot {
        assert_eq!(
            source.owner(),
            self.source_owner,
            "slot belongs to another imported child"
        );
        self.slots[source.index() as usize]
    }
    pub fn remap_quantity_slot(&self, source: HostQuantitySlot) -> HostQuantitySlot {
        assert_eq!(
            source.owner(),
            self.source_owner,
            "quantity slot belongs to another imported child"
        );
        self.quantity_slots[source.index() as usize]
    }
}

fn remap_steps(
    steps: Vec<ScheduleStep>,
    view: impl Fn(AnyBufferView) -> AnyBufferView + Copy,
    slot: impl Fn(AnyScalarSlot) -> AnyScalarSlot + Copy,
    quantity_slot: impl Fn(HostQuantitySlot) -> HostQuantitySlot + Copy,
    launch: impl Fn(LaunchId) -> LaunchId + Copy,
) -> Vec<ScheduleStep> {
    steps
        .into_iter()
        .map(|step| match step {
            ScheduleStep::Imported { scope, body } => ScheduleStep::Imported {
                scope,
                body: remap_steps(body, view, slot, quantity_slot, launch),
            },
            ScheduleStep::BeginAllocationInstance { source, result } => {
                ScheduleStep::BeginAllocationInstance {
                    source: view(source),
                    result: view(result),
                }
            }
            ScheduleStep::BindArgumentTensor { source, result } => {
                ScheduleStep::BindArgumentTensor {
                    source: view(source),
                    result: view(result),
                }
            }
            ScheduleStep::PublishTensor {
                view: value,
                path,
                bytes,
                declared_axes,
            } => ScheduleStep::PublishTensor {
                view: view(value),
                path,
                bytes,
                declared_axes,
            },
            ScheduleStep::Launch(id) => ScheduleStep::Launch(launch(id)),
            ScheduleStep::Copy(copy) => ScheduleStep::Copy(BufferCopy {
                source: view(copy.source),
                destination: view(copy.destination),
                bytes: copy.bytes,
            }),
            ScheduleStep::Fill(fill) => ScheduleStep::Fill(BufferFill {
                destination: view(fill.destination),
                value: fill.value,
                bytes: fill.bytes,
            }),
            ScheduleStep::ScalarMove(m) => ScheduleStep::ScalarMove(ScalarMove {
                from: slot(m.from),
                to: slot(m.to),
            }),
            ScheduleStep::EvaluateHost(mut evaluation) => {
                evaluation.to = match evaluation.to {
                    HostValueDestination::Quantity(to) => {
                        HostValueDestination::Quantity(quantity_slot(to))
                    }
                    HostValueDestination::Native(to) => HostValueDestination::Native(slot(to)),
                };
                ScheduleStep::EvaluateHost(evaluation)
            }
            ScheduleStep::ScalarRead(r) => ScheduleStep::ScalarRead(ScalarRead {
                source: view(r.source),
                index: r.index,
                bounds: r.bounds,
                byte_offset: r.byte_offset,
                to: slot(r.to),
            }),
            ScheduleStep::Check(c) => ScheduleStep::Check(ScalarCheck {
                condition: slot(c.condition),
                expectation: c.expectation,
                site: c.site,
            }),
            ScheduleStep::If {
                condition,
                then_steps,
                else_steps,
                results,
            } => ScheduleStep::If {
                condition,
                then_steps: remap_steps(then_steps, view, slot, quantity_slot, launch),
                else_steps: remap_steps(else_steps, view, slot, quantity_slot, launch),
                results: results.map(&mut |result| result.remap(view, slot, quantity_slot)),
            },
            ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                visits,
                body,
                carries,
            } => ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                visits,
                body: remap_steps(body, view, slot, quantity_slot, launch),
                carries: carries.map(&mut |carry| carry.remap(view, slot, quantity_slot)),
            },
        })
        .collect()
}

pub struct ScheduleBuilder<'a, B: PhysicalDialect> {
    inner: internals::Builder<'a, B>,
}
impl<'a, B: PhysicalDialect> ScheduleBuilder<'a, B> {
    pub fn launch(&mut self, launch: Launch<B>) -> LaunchId {
        self.inner.launch(launch)
    }
    pub fn step_launch(&mut self, launch: LaunchId) -> ScheduleUse {
        self.inner.step_launch(launch)
    }
    pub fn branch(
        &mut self,
        condition: BoolExpr,
        then: impl FnOnce(&mut ScheduleBuilder<'_, B>),
        otherwise: impl FnOnce(&mut ScheduleBuilder<'_, B>),
    ) {
        self.inner.branch(condition, then, otherwise)
    }
    pub fn close(self) -> ClosedSchedule {
        self.inner.close()
    }
    pub fn slot_any(&mut self, kind: ScalarKind) -> AnyScalarSlot {
        self.inner.state.slot_any(self.inner.arena, kind)
    }
    pub fn quantity_slot(&mut self, kind: HostQuantityKind) -> HostQuantitySlot {
        self.inner.state.quantity_slot(self.inner.arena, kind)
    }
    pub fn quantity_slot_symbol(&mut self, slot: HostQuantitySlot) -> SymbolId {
        assert_eq!(
            slot.owner(),
            self.inner.state.owner,
            "quantity slot belongs to another construction"
        );
        slot.symbol()
    }
    pub fn evaluate_host(&mut self, evaluation: HostEvaluation) -> ScheduleUse {
        if let Some(site) = &evaluation.failure {
            assert!(
                matches!(
                    site.failure.cause,
                    seismic_lang::failure::SourceFailureCause::Scalar(
                        seismic_lang::reference_math::ScalarFailure::IntegerDivisionByZero
                    )
                ),
                "host evaluation may own only its source integer division failure"
            );
        }
        match (evaluation.value, evaluation.to) {
            (HostValueExpr::Integer(_), HostValueDestination::Quantity(to)) => {
                assert_eq!(to.kind(), HostQuantityKind::Integer);
                assert_eq!(to.owner(), self.inner.state.owner);
            }
            (HostValueExpr::Natural(_), HostValueDestination::Quantity(to)) => {
                assert_eq!(to.kind(), HostQuantityKind::Natural);
                assert_eq!(to.owner(), self.inner.state.owner);
            }
            (HostValueExpr::Bool(_), HostValueDestination::Native(to)) => {
                assert_eq!(to.kind(), ScalarKind::Scalar(DType::Bool));
                assert_eq!(to.owner(), self.inner.state.owner);
            }
            (HostValueExpr::Word { dtype, .. }, HostValueDestination::Native(to)) => {
                assert!(
                    matches!(dtype, DType::I32 | DType::U32),
                    "host word projection needs an integer scalar dtype"
                );
                assert_eq!(to.kind(), ScalarKind::Scalar(dtype));
                assert_eq!(to.owner(), self.inner.state.owner);
            }
            (HostValueExpr::Float { dtype, .. }, HostValueDestination::Native(to)) => {
                assert!(
                    dtype.is_float(),
                    "host float conversion needs a float scalar dtype"
                );
                assert_eq!(to.kind(), ScalarKind::Scalar(dtype));
                assert_eq!(to.owner(), self.inner.state.owner);
            }
            _ => panic!("host evaluation result and destination kind differ"),
        }
        self.inner.leaf(ScheduleStep::EvaluateHost(evaluation))
    }
    pub fn temporary_bool(&mut self) -> AnyScalarSlot {
        self.inner
            .state
            .slot_any(self.inner.arena, ScalarKind::Scalar(DType::Bool))
    }
    pub fn slot_symbol_any(&mut self, slot: AnyScalarSlot) -> SymbolId {
        self.inner.slot_symbol(slot)
    }
    pub fn begin_allocation_instance(
        &mut self,
        view: AnyBufferView,
        result: AnyBufferView,
    ) -> ScheduleUse {
        self.inner.begin_allocation_instance(view, result)
    }
    pub(crate) fn bind_argument_tensor(
        &mut self,
        source: AnyBufferView,
        result: AnyBufferView,
    ) -> ScheduleUse {
        self.inner
            .leaf(ScheduleStep::BindArgumentTensor { source, result })
    }
    pub fn publish_tensor(
        &mut self,
        view: AnyBufferView,
        path: Vec<u32>,
        declared_axes: Vec<NatExpr>,
    ) -> ScheduleUse {
        self.inner.publish_tensor(view, path, declared_axes)
    }
    pub fn copy_any(&mut self, source: AnyBufferView, destination: AnyBufferView) -> ScheduleUse {
        self.inner.copy(source, destination)
    }
    pub fn fill_constant_any(
        &mut self,
        destination: AnyBufferView,
        value: seismic_lang::intrinsics::FillConstant,
    ) -> ScheduleUse {
        let pattern = crate::repr::fill_constant_for(destination.representation, value);
        self.inner.fill(destination, pattern)
    }
    pub fn scalar_move_any(&mut self, from: AnyScalarSlot, to: AnyScalarSlot) -> ScheduleUse {
        self.inner.scalar_move(from, to)
    }
    pub fn scalar_read_any(
        &mut self,
        source: AnyBufferView,
        index: Vec<NatExpr>,
        to: AnyScalarSlot,
    ) -> ScheduleUse {
        self.inner.scalar_read(source, index, to)
    }
    pub fn check_any(
        &mut self,
        condition: AnyScalarSlot,
        site: crate::kernel::ops::CheckSite,
    ) -> ScheduleUse {
        self.inner.check(condition, site)
    }
    pub fn check_zero_any(
        &mut self,
        condition: AnyScalarSlot,
        site: crate::kernel::ops::CheckSite,
    ) -> ScheduleUse {
        self.inner.check_zero(condition, site)
    }
    pub fn launch_sequential(&mut self, kernel: KernelId) -> ScheduleUse {
        let one = self.inner.arena.nat(1);
        let empty = self.inner.arena.bool(false);
        let launch = self.inner.launch(Launch {
            kernel,
            descriptor: B::ordinary_launch(),
            grid: [one; 3],
            workgroup: [one; 3],
            empty,
            parallel_extent: None,
            logical_base: None,
        });
        self.inner.step_launch(launch)
    }
    pub fn launch_semantic(
        &mut self,
        kernel: KernelId,
        contract: crate::kernel::ops::SegmentLaunchDomain,
        logical_index: Option<crate::kernel::internals::LogicalIndexBinding>,
        descriptor: B::LaunchDescriptor,
    ) -> ScheduleUse {
        let zero = self.inner.arena.nat(0);
        let launch = self.inner.launch(Launch {
            kernel,
            descriptor,
            grid: contract.grid,
            workgroup: contract.workgroup,
            empty: contract.empty,
            parallel_extent: Some(contract.parallel_extent),
            logical_base: logical_index.map(|binding| LogicalLaunchBase {
                binding,
                value: zero,
                chunk_bounds: None,
            }),
        });
        self.inner.step_launch(launch)
    }
}
#[derive(Debug)]
pub struct ClosedSchedule {
    pub(crate) owner: OwnerToken,
}

mod internals {
    use super::*;
    pub(super) struct Builder<'a, B: PhysicalDialect> {
        pub(super) arena: &'a mut ExprArena,
        pub(super) views: &'a [crate::storage::BufferViewLayout],
        pub(super) state: &'a mut ScheduleConstruction<B>,
        pub(super) region: u32,
    }
    impl<'a, B: PhysicalDialect> Builder<'a, B> {
        fn owner(&self) -> OwnerToken {
            self.state.owner
        }
        fn assert_owner(&self, owner: OwnerToken) {
            assert_eq!(
                owner,
                self.owner(),
                "schedule handle belongs to another implementation"
            )
        }
        fn use_site(&self) -> ScheduleUse {
            ScheduleUse {
                owner: self.owner(),
                region: self.state.regions[self.region as usize].path.clone(),
                ordinal: self.state.regions[self.region as usize].steps.len() as u32,
            }
        }
        pub(super) fn leaf(&mut self, step: ScheduleStep) -> ScheduleUse {
            let at = self.use_site();
            self.state.regions[self.region as usize]
                .steps
                .push(RegionStep::Leaf(step));
            at
        }
        pub(super) fn slot_symbol(&mut self, slot: AnyScalarSlot) -> SymbolId {
            self.assert_owner(slot.owner());
            let stored = self.state.slots[slot.index as usize];
            assert_eq!(stored, slot, "scalar slot identity changed");
            stored.symbol
        }
        pub(super) fn launch(&mut self, launch: Launch<B>) -> LaunchId {
            self.assert_owner(launch.kernel.owner());
            if let Some(base) = launch.logical_base {
                assert!(
                    !base.is_chunked(),
                    "normalized launches must use structural import"
                );
                assert_eq!(
                    base.binding.kernel, launch.kernel,
                    "logical index belongs to another kernel"
                );
                assert_eq!(
                    Some(base.binding.extent),
                    launch.parallel_extent,
                    "logical index extent differs from launch extent"
                );
                let one = self.arena.nat(1);
                assert_eq!(
                    &launch.grid[1..],
                    &[one, one],
                    "logical index requires a 1D grid"
                );
                assert_eq!(
                    &launch.workgroup[1..],
                    &[one, one],
                    "logical index requires a 1D workgroup"
                );
            }
            let id = LaunchId::new(self.owner(), self.state.launches.len() as u32);
            self.state.launches.push(launch);
            id
        }
        pub(super) fn step_launch(&mut self, launch: LaunchId) -> ScheduleUse {
            self.assert_owner(launch.owner());
            assert!(
                (launch.index() as usize) < self.state.launches.len(),
                "unknown launch handle"
            );
            self.leaf(ScheduleStep::Launch(launch))
        }
        pub(super) fn begin_allocation_instance(
            &mut self,
            view: AnyBufferView,
            result: AnyBufferView,
        ) -> ScheduleUse {
            self.assert_owner(view.owner());
            assert!(
                matches!(
                    self.views[view.index() as usize].base,
                    crate::storage::ViewBase::Allocation(_)
                ),
                "allocation occurrence must define actual storage"
            );
            self.leaf(ScheduleStep::BeginAllocationInstance {
                source: view,
                result,
            })
        }
        pub(super) fn publish_tensor(
            &mut self,
            view: AnyBufferView,
            path: Vec<u32>,
            declared_axes: Vec<NatExpr>,
        ) -> ScheduleUse {
            self.assert_owner(view.owner());
            let layout = &self.views[view.index() as usize];
            assert_eq!(
                layout.extents.len(),
                declared_axes.len(),
                "published tensor rank differs from declaration"
            );
            let end = crate::storage::addressed_bytes(self.arena, layout);
            let bytes = self.arena.nat_sub(end, layout.offset);
            self.leaf(ScheduleStep::PublishTensor {
                view,
                path,
                bytes,
                declared_axes,
            })
        }
        pub(super) fn copy(
            &mut self,
            source: AnyBufferView,
            destination: AnyBufferView,
        ) -> ScheduleUse {
            self.assert_owner(source.owner());
            self.assert_owner(destination.owner());
            assert_eq!(
                source.representation, destination.representation,
                "copy representation mismatch"
            );
            assert_eq!(
                self.views[source.index as usize].extents.len(),
                self.views[destination.index as usize].extents.len(),
                "copy rank mismatch"
            );
            let layout = &self.views[source.index as usize];
            assert!(
                layout.contiguous && self.views[destination.index as usize].contiguous,
                "direct byte copy requires canonical contiguous views"
            );
            let bytes =
                crate::storage::tensor_bytes(self.arena, layout.representation, &layout.extents);
            self.leaf(ScheduleStep::Copy(BufferCopy {
                source,
                destination,
                bytes,
            }))
        }
        pub(super) fn fill(&mut self, destination: AnyBufferView, value: FillValue) -> ScheduleUse {
            self.assert_owner(destination.owner());
            let layout = &self.views[destination.index as usize];
            assert!(
                layout.contiguous,
                "direct byte fill requires a canonical contiguous view"
            );
            let bytes =
                crate::storage::tensor_bytes(self.arena, layout.representation, &layout.extents);
            self.leaf(ScheduleStep::Fill(BufferFill {
                destination,
                value,
                bytes,
            }))
        }
        pub(super) fn scalar_move(
            &mut self,
            from: AnyScalarSlot,
            to: AnyScalarSlot,
        ) -> ScheduleUse {
            self.assert_owner(from.owner());
            self.assert_owner(to.owner());
            assert_eq!(from.kind, to.kind, "scalar move kind mismatch");
            self.leaf(ScheduleStep::ScalarMove(ScalarMove { from, to }))
        }
        pub(super) fn scalar_read(
            &mut self,
            source: AnyBufferView,
            index: Vec<NatExpr>,
            to: AnyScalarSlot,
        ) -> ScheduleUse {
            self.assert_owner(source.owner());
            self.assert_owner(to.owner());
            assert_eq!(
                self.views[source.index as usize].extents.len(),
                index.len(),
                "scalar read rank mismatch"
            );
            let dtype =
                match &seismic_lang::registry::representation_info(source.representation).kind {
                    seismic_lang::registry::RepresentationKind::Dense(dtype) => *dtype,
                    seismic_lang::registry::RepresentationKind::Packed(_)
                    | seismic_lang::registry::RepresentationKind::PackedRows(_) => DType::F32,
                    seismic_lang::registry::RepresentationKind::External(_) => {
                        panic!("external representations permit only sealed conversion")
                    }
                };
            assert_eq!(
                ScalarKind::Scalar(dtype),
                to.kind,
                "scalar read destination dtype mismatch"
            );
            let layout = &self.views[source.index as usize];
            let bounds = index
                .iter()
                .copied()
                .zip(layout.extents.iter().copied())
                .map(|(coordinate, extent)| {
                    self.arena
                        .nat_cmp(seismic_lang::expr::CmpOp::Lt, coordinate, extent)
                })
                .collect();
            let mut units = self.arena.nat(0);
            for (coordinate, stride) in index.iter().copied().zip(layout.strides.iter().copied()) {
                let term = self.arena.nat_mul(coordinate, stride);
                units = self.arena.nat_add(units, term);
            }
            let unit_bytes =
                match seismic_lang::registry::representation_info(source.representation).kind {
                    seismic_lang::registry::RepresentationKind::Dense(dtype) => {
                        u64::from(dtype.bytes())
                    }
                    seismic_lang::registry::RepresentationKind::Packed(_)
                    | seismic_lang::registry::RepresentationKind::PackedRows(_) => {
                        unreachable!("DenseRepresentation excludes packed scalar reads")
                    }
                    seismic_lang::registry::RepresentationKind::External(_) => {
                        unreachable!("external representations exclude scalar reads")
                    }
                };
            let unit_bytes = self.arena.nat(unit_bytes);
            let byte_offset = self.arena.nat_mul(units, unit_bytes);
            self.leaf(ScheduleStep::ScalarRead(ScalarRead {
                source,
                index,
                bounds,
                byte_offset,
                to,
            }))
        }
        pub(super) fn check(
            &mut self,
            condition: AnyScalarSlot,
            site: crate::kernel::ops::CheckSite,
        ) -> ScheduleUse {
            self.assert_owner(condition.owner());
            assert_eq!(
                condition.kind,
                ScalarKind::Scalar(DType::Bool),
                "schedule check condition must be boolean"
            );
            self.leaf(ScheduleStep::Check(ScalarCheck {
                condition,
                expectation: ScalarCheckExpectation::BoolTrue,
                site,
            }))
        }

        pub(super) fn check_zero(
            &mut self,
            condition: AnyScalarSlot,
            site: crate::kernel::ops::CheckSite,
        ) -> ScheduleUse {
            self.assert_owner(condition.owner());
            assert_eq!(
                condition.kind,
                ScalarKind::Scalar(DType::U32),
                "preflight status must be u32"
            );
            self.leaf(ScheduleStep::Check(ScalarCheck {
                condition,
                expectation: ScalarCheckExpectation::U32Zero,
                site,
            }))
        }
        pub(super) fn branch(
            &mut self,
            condition: BoolExpr,
            then: impl FnOnce(&mut ScheduleBuilder<'_, B>),
            otherwise: impl FnOnce(&mut ScheduleBuilder<'_, B>),
        ) {
            let (then_region, else_region) = self.state.begin_branch(self.region, condition);
            {
                let mut child = ScheduleBuilder {
                    inner: Builder {
                        arena: &mut *self.arena,
                        views: self.views,
                        state: &mut *self.state,
                        region: then_region,
                    },
                };
                then(&mut child);
            }
            {
                let mut child = ScheduleBuilder {
                    inner: Builder {
                        arena: &mut *self.arena,
                        views: self.views,
                        state: &mut *self.state,
                        region: else_region,
                    },
                };
                otherwise(&mut child);
            }
        }
        pub(super) fn close(self) -> ClosedSchedule {
            assert_eq!(
                self.region, 0,
                "only the root schedule builder can close construction"
            );
            self.state.closed = true;
            ClosedSchedule {
                owner: self.owner(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::expr::{Assignment, SymbolValue};

    /// Per-launch obligations lifted through their lexical scopes.
    fn launch_requirements<B: PhysicalDialect>(
        schedule: &ParametricSchedule<B>,
        arena: &mut ExprArena,
        requirements: &[BoolExpr],
    ) -> BoolExpr {
        schedule.scoped_requirements(
            arena,
            &[],
            &mut |arena, point| match point {
                RequirementPoint::Step(ScheduleStep::Launch(id)) => {
                    requirements[id.index() as usize]
                }
                _ => arena.bool(true),
            },
            &mut |arena, _| arena.bool(true),
        )
    }

    #[derive(Debug)]
    struct Dialect;
    #[derive(Clone, Debug)]
    enum NoIntrinsic {}
    impl PhysicalDialect for Dialect {
        const NAME: seismic_lang::registry::BackendName = seismic_lang::registry::BackendName::Cpu;
        type Facts = ();
        type Intrinsic = NoIntrinsic;
        type LaunchDescriptor = ();
        fn ordinary_launch() {}
        fn write_intrinsic_identity(
            op: &NoIntrinsic,
            _: &mut crate::physical_target::IntrinsicIdentityBuilder,
        ) {
            match *op {}
        }
        fn intrinsic_numerics(
            _: &(),
            _: &seismic_lang::registry::IntrinsicSignature,
            op: &NoIntrinsic,
        ) -> crate::physical_target::IntrinsicNumericalSemantics {
            match *op {}
        }
        fn intrinsic_addressable_resources(
            op: &NoIntrinsic,
        ) -> Vec<crate::kernel::ops::AddressableResourceHandle> {
            match *op {}
        }
    }
    #[test]
    fn successful_step_facts_require_dominance_and_expire_on_slot_writes() {
        let owner = OwnerToken::fresh();
        let mut arena = ExprArena::default();
        let symbol = arena.schedule_slot(0, seismic_lang::expr::SymbolSort::Int);
        let value = arena.int_symbol(symbol);
        let zero = arena.int(0);
        let fact = arena.int_cmp(seismic_lang::expr::CmpOp::Ge, value, zero);
        let slot = HostQuantitySlot {
            owner,
            index: 0,
            kind: HostQuantityKind::Integer,
            symbol,
            capture: None,
        };
        let view = AnyBufferView::new(owner, 0, seismic_lang::registry::dense(DType::I32));
        let begin = ScheduleStep::BeginAllocationInstance {
            source: view,
            result: view,
        };
        let usage = ScheduleStep::Launch(LaunchId::new(owner, 0));
        let mut schedule = ParametricSchedule::<Dialect> {
            owner,
            next_control: 0,
            launches: Vec::new(),
            slots: Vec::new(),
            quantity_slots: vec![slot],
            steps: Vec::new(),
            direct_view_uses: Vec::new(),
            launch_uses: Vec::new(),
        };
        let mut values = Assignment::new();
        values.bind(symbol, SymbolValue::Int((-1).into()));
        let check = |schedule: &ParametricSchedule<Dialect>, arena: &mut ExprArena| {
            let predicate = schedule.scoped_requirements(
                arena,
                &[],
                &mut |arena, point| match point {
                    RequirementPoint::Step(ScheduleStep::Launch(_)) => fact,
                    _ => arena.bool(true),
                },
                &mut |arena, step| match step {
                    ScheduleStep::BeginAllocationInstance { .. } => fact,
                    _ => arena.bool(true),
                },
            );
            arena.eval_bool(predicate, &values).unwrap()
        };
        schedule.steps = vec![begin.clone(), usage.clone()];
        assert!(check(&schedule, &mut arena));
        schedule.steps = vec![usage.clone(), begin.clone()];
        assert!(!check(&schedule, &mut arena));
        schedule.steps = vec![
            begin.clone(),
            ScheduleStep::EvaluateHost(HostEvaluation {
                value: HostValueExpr::Integer(arena.int(-1)),
                to: HostValueDestination::Quantity(slot),
                failure: None,
            }),
            usage.clone(),
        ];
        assert!(!check(&schedule, &mut arena));
        schedule.steps = vec![
            ScheduleStep::If {
                condition: fact,
                then_steps: vec![begin],
                else_steps: Vec::new(),
                results: Product::Unit,
            },
            usage,
        ];
        assert!(!check(&schedule, &mut arena));
    }

    #[test]
    fn semantic_launch_chunking_uses_exact_tail_grid_and_contiguous_logical_base() {
        let owner = OwnerToken::fresh();
        let mut arena = ExprArena::default();
        let one = arena.nat(1);
        let eight = arena.nat(8);
        let seventeen = arena.nat(17);
        let three = arena.nat(3);
        let zero = arena.nat(0);
        let never_empty = arena.bool(false);
        let kernel = KernelId::new(owner, 0);
        let launch_id = LaunchId::new(owner, 0);
        let mut schedule = ParametricSchedule::<Dialect> {
            owner,
            next_control: 0,
            launches: vec![Launch {
                kernel,
                descriptor: (),
                grid: [three, one, one],
                workgroup: [eight, one, one],
                empty: never_empty,
                parallel_extent: Some(seventeen),
                logical_base: Some(LogicalLaunchBase {
                    binding: crate::kernel::internals::LogicalIndexBinding {
                        kernel,
                        argument: 2,
                        extent: seventeen,
                    },
                    value: zero,
                    chunk_bounds: None,
                }),
            }],
            slots: Vec::new(),
            quantity_slots: Vec::new(),
            steps: vec![ScheduleStep::Launch(launch_id)],
            direct_view_uses: Vec::new(),
            launch_uses: Vec::new(),
        };

        schedule
            .chunk_semantic_launch(&mut arena, launch_id, 2, 64)
            .unwrap();

        let ScheduleStep::Repeat {
            symbol, end, body, ..
        } = &schedule.steps[0]
        else {
            panic!("semantic launch was not wrapped in a repeat")
        };
        assert_eq!(arena.eval_nat_u64(*end, &Assignment::new()).unwrap(), 2);
        assert!(matches!(body.as_slice(), [ScheduleStep::Launch(id)] if *id == launch_id));

        let envelope = schedule.launch_grid_envelope(&mut arena, schedule.launch(launch_id));
        let reserved_groups = arena.eval_nat_u64(envelope[0], &Assignment::new()).unwrap();
        assert_eq!(reserved_groups, 2);
        let launch = schedule.launch(launch_id);
        let logical = launch.logical_base.expect("logical base was discarded");
        assert_eq!(logical.argument(), 2);
        let mut covered = Vec::new();
        for iteration in 0u64..2 {
            let mut values = Assignment::new();
            values.bind(*symbol, SymbolValue::Nat((iteration).into()));
            let base = arena.eval_nat_u64(logical.value, &values).unwrap();
            let blocks = arena.eval_nat_u64(launch.grid[0], &values).unwrap();
            assert_eq!(blocks, if iteration == 0 { 2 } else { 1 });
            assert!(blocks <= reserved_groups);
            for physical in 0..blocks * 8 {
                // Scratch addressing uses this local physical participant,
                // independent of the logical base used for semantic indexing.
                assert!(physical < reserved_groups * 8);
                let logical_index = base + physical;
                if logical_index < 17 {
                    covered.push(logical_index);
                }
            }
        }
        assert_eq!(covered, (0..17).collect::<Vec<_>>());
        let local_symbol = *symbol;
        let base = schedule.launch(launch_id).logical_base.unwrap().value;
        let requirement = arena.nat_cmp(seismic_lang::expr::CmpOp::Lt, base, seventeen);
        let lifted = launch_requirements(&schedule, &mut arena, &[requirement]);
        assert!(!arena.free_symbols(lifted.into()).contains(&local_symbol));
        assert!(arena.eval_bool(lifted, &Assignment::new()).unwrap());
        let one = arena.nat(1);
        let too_small = arena.nat_cmp(
            seismic_lang::expr::CmpOp::Le,
            schedule.launch(launch_id).grid[0],
            one,
        );
        let lifted = launch_requirements(&schedule, &mut arena, &[too_small]);
        assert!(!arena.free_symbols(lifted.into()).contains(&local_symbol));
        assert!(!arena.eval_bool(lifted, &Assignment::new()).unwrap());

        let bytes = arena.nat_add(base, one);
        let reserved = schedule.launch_reservation(&mut arena, 0, bytes);
        assert!(arena.free_symbols(reserved.into()).is_empty());
        assert_eq!(
            arena.eval_nat_u64(reserved, &Assignment::new()).unwrap(),
            17
        );

        // An outer semantic loop and the generated chunk loop must both be
        // closed over before admission evaluates the reservation.
        let (outer, outer_symbol, outer_value) = arena.loop_binder();
        let outer_value = arena.nat_from_int(outer_value);
        let bytes = arena.nat_add(bytes, outer_value);
        let two = arena.nat(2);
        let four = arena.nat(4);
        let body = std::mem::take(&mut schedule.steps);
        schedule.steps = vec![ScheduleStep::Repeat {
            binder: outer,
            symbol: outer_symbol,
            start: two,
            end: four,
            visits: RepeatVisits::Ordered,
            body,
            carries: Product::Unit,
        }];
        let reserved = schedule.launch_reservation(&mut arena, 0, bytes);
        assert!(arena.free_symbols(reserved.into()).is_empty());
        assert_eq!(
            arena.eval_nat_u64(reserved, &Assignment::new()).unwrap(),
            20
        );
        let ScheduleStep::Repeat { end, .. } = &mut schedule.steps[0] else {
            unreachable!()
        };
        *end = one;
        let reserved = schedule.launch_reservation(&mut arena, 0, bytes);
        assert_eq!(arena.eval_nat_u64(reserved, &Assignment::new()).unwrap(), 0);
        // A rejected native descriptor in an unexecuted loop body must not
        // restrict the empty loop's applicability.
        let unsupported_descriptor = arena.bool(false);
        let lifted = launch_requirements(&schedule, &mut arena, &[unsupported_descriptor]);
        assert!(arena.eval_bool(lifted, &Assignment::new()).unwrap());
        // Use obligations in an empty body/backedge must preserve partial
        // expression laziness. Initial carried products are still evaluated.
        let zero = arena.nat(0);
        let partial = arena.nat_div(one, zero);
        let lifted = schedule.scoped_requirements(
            &mut arena,
            &[],
            &mut |arena, point| match point {
                RequirementPoint::Step(_) | RequirementPoint::RepeatBackedge(_) => {
                    arena.nat_cmp(seismic_lang::expr::CmpOp::Eq, partial, zero)
                }
                RequirementPoint::RepeatInitial(_) | RequirementPoint::BranchResult(_, _) => {
                    arena.bool(true)
                }
            },
            &mut |arena, _| arena.bool(true),
        );
        assert!(arena.eval_bool(lifted, &Assignment::new()).unwrap());
        let defined = arena.side_conditions(lifted.into());
        assert!(arena.eval_bool(defined, &Assignment::new()).unwrap());
        let invalid_initial = schedule.scoped_requirements(
            &mut arena,
            &[],
            &mut |arena, point| arena.bool(!matches!(point, RequirementPoint::RepeatInitial(_))),
            &mut |arena, _| arena.bool(true),
        );
        assert!(!arena
            .eval_bool(invalid_initial, &Assignment::new())
            .unwrap());
    }

    #[test]
    fn semantic_launch_at_exact_cap_uses_one_chunk() {
        let owner = OwnerToken::fresh();
        let mut arena = ExprArena::default();
        let one = arena.nat(1);
        let eight = arena.nat(8);
        let sixteen = arena.nat(16);
        let two = arena.nat(2);
        let zero = arena.nat(0);
        let launch_id = LaunchId::new(owner, 0);
        let mut schedule = ParametricSchedule::<Dialect> {
            owner,
            next_control: 0,
            launches: vec![Launch {
                kernel: KernelId::new(owner, 0),
                descriptor: (),
                grid: [two, one, one],
                workgroup: [eight, one, one],
                empty: arena.bool(false),
                parallel_extent: Some(sixteen),
                logical_base: Some(LogicalLaunchBase {
                    binding: crate::kernel::internals::LogicalIndexBinding {
                        kernel: KernelId::new(owner, 0),
                        argument: 0,
                        extent: sixteen,
                    },
                    value: zero,
                    chunk_bounds: None,
                }),
            }],
            slots: Vec::new(),
            quantity_slots: Vec::new(),
            steps: vec![ScheduleStep::Launch(launch_id)],
            direct_view_uses: Vec::new(),
            launch_uses: Vec::new(),
        };

        schedule
            .chunk_semantic_launch(&mut arena, launch_id, 2, 64)
            .unwrap();
        let ScheduleStep::Repeat { end, .. } = schedule.steps[0] else {
            panic!("semantic launch was not wrapped in a repeat")
        };
        assert_eq!(arena.eval_nat_u64(end, &Assignment::new()).unwrap(), 1);
    }
}
