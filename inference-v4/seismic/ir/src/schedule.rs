//! Structured parametric schedules and typed data movement.
//!
//! Construction stores control as a region tree. A leaf command belongs to
//! exactly one lexical region, so allocation liveness and later command
//! grouping cannot accidentally cross `If`, `Repeat`, or `Choose`.

use crate::identity::OwnerToken;
use crate::kernel::KernelId;
use crate::repr::{DenseRepresentation, Representation, ScalarType, WritableRepresentation};
use crate::storage::{AnyBufferView, BufferViewId, ScheduleRegionEdge, ScheduleUse};
use crate::target::KernelDialect;
use seismic_lang::expr::{BoolExpr, DecisionId, ExprArena, LoopBinderId, NatExpr, SymbolId};
use seismic_lang::types::DType;
use std::fmt;
use std::marker::PhantomData;

#[derive(PartialEq, Eq, Hash)]
pub struct ScalarSlotId<T: ScalarType> {
    owner: OwnerToken,
    index: u32,
    symbol: SymbolId,
    marker: PhantomData<T>,
}
impl<T: ScalarType> Clone for ScalarSlotId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ScalarType> Copy for ScalarSlotId<T> {}
impl<T: ScalarType> fmt::Debug for ScalarSlotId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot<{:?}>#{}", T::DTYPE, self.index)
    }
}
impl<T: ScalarType> ScalarSlotId<T> {
    pub(crate) fn new(owner: OwnerToken, index: u32, symbol: SymbolId) -> Self {
        Self {
            owner,
            index,
            symbol,
            marker: PhantomData,
        }
    }
    pub fn erase(self) -> AnyScalarSlot {
        AnyScalarSlot {
            owner: self.owner,
            index: self.index,
            dtype: T::DTYPE,
            symbol: self.symbol,
            sort: T::SYMBOL_SORT,
        }
    }
    pub fn from_any(slot: AnyScalarSlot) -> Self {
        assert_eq!(slot.dtype, T::DTYPE, "scalar result slot dtype mismatch");
        assert_eq!(
            slot.sort,
            T::SYMBOL_SORT,
            "scalar result slot sort mismatch"
        );
        Self::new(slot.owner, slot.index, slot.symbol)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AnyScalarSlot {
    owner: OwnerToken,
    pub(crate) index: u32,
    pub(crate) dtype: DType,
    pub(crate) symbol: SymbolId,
    pub(crate) sort: seismic_lang::expr::SymbolSort,
}
impl AnyScalarSlot {
    pub fn index(&self) -> u32 {
        self.index
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn symbol(&self) -> SymbolId {
        self.symbol
    }
    pub fn sort(&self) -> seismic_lang::expr::SymbolSort {
        self.sort
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

#[derive(Clone, Debug)]
pub struct Launch {
    pub kernel: KernelId,
    pub mode: LaunchMode,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LogicalLaunchBase {
    /// Ordinal of the natural kernel argument used as the logical base.
    pub argument: u32,
    /// Schedule expression supplied to that argument for this launch.
    pub value: NatExpr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LaunchMode {
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

#[derive(Clone, Debug)]
pub enum ScheduleStep {
    Launch(LaunchId),
    Copy(BufferCopy),
    Fill(BufferFill),
    ScalarMove(ScalarMove),
    ScalarRead(ScalarRead),
    Check(ScalarCheck),
    If {
        condition: BoolExpr,
        then_steps: Vec<ScheduleStep>,
        else_steps: Vec<ScheduleStep>,
    },
    Repeat {
        binder: LoopBinderId,
        symbol: SymbolId,
        start: NatExpr,
        end: NatExpr,
        body: Vec<ScheduleStep>,
    },
    Choose {
        decision: DecisionId,
        options: Vec<(i64, Vec<ScheduleStep>)>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoopBinding {
    pub id: LoopBinderId,
    pub symbol: SymbolId,
    pub index: NatExpr,
}

#[derive(Debug)]
pub struct ParametricSchedule {
    owner: OwnerToken,
    launches: Vec<Launch>,
    slots: Vec<AnyScalarSlot>,
    steps: Vec<ScheduleStep>,
    direct_view_uses: Vec<(AnyBufferView, ScheduleUse)>,
    launch_uses: Vec<(LaunchId, ScheduleUse)>,
}
impl ParametricSchedule {
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
        max_grid_x: NatExpr,
    ) {
        self.assert_owner(id.owner());
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
        let participants = arena.nat_product(&launch.workgroup);
        let capacity = arena.nat_mul(max_grid_x, participants);
        let zero = arena.nat(0);
        let repeat_end = arena.nat_ceil_div(extent, capacity);
        let (binder, symbol, iteration) = arena.nat_loop_binder();
        let base = arena.nat_mul(iteration, capacity);
        let remaining = arena.nat_sub(extent, base);
        let active = arena.nat_min(remaining, capacity);
        launch.grid[0] = arena.nat_ceil_div(active, participants);
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
                            ..
                        } => steps(then_steps).saturating_add(steps(else_steps)),
                        ScheduleStep::Repeat { body, .. } => steps(body),
                        ScheduleStep::Choose { options, .. } => options.iter().fold(
                            options
                                .capacity()
                                .saturating_mul(std::mem::size_of::<(i64, Vec<ScheduleStep>)>()),
                            |bytes, (_, option)| bytes.saturating_add(steps(option)),
                        ),
                        ScheduleStep::ScalarRead(read) => read
                            .index
                            .capacity()
                            .saturating_mul(std::mem::size_of::<NatExpr>())
                            .saturating_add(
                                read.bounds
                                    .capacity()
                                    .saturating_mul(std::mem::size_of::<BoolExpr>()),
                            ),
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
        self.launches.capacity() * std::mem::size_of::<Launch>()
            + self.slots.capacity() * std::mem::size_of::<AnyScalarSlot>()
            + steps(&self.steps)
            + self.direct_view_uses.capacity() * std::mem::size_of::<(AnyBufferView, ScheduleUse)>()
            + self.launch_uses.capacity() * std::mem::size_of::<(LaunchId, ScheduleUse)>()
            + use_paths
    }
    pub fn launches(&self) -> &[Launch] {
        &self.launches
    }
    pub fn launch(&self, id: LaunchId) -> &Launch {
        self.assert_owner(id.owner());
        &self.launches[id.index() as usize]
    }
    pub fn slots(&self) -> &[AnyScalarSlot] {
        &self.slots
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
                *step = ScheduleStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    body: vec![ScheduleStep::Launch(target)],
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
            ScheduleStep::Repeat { body, .. } => {
                rewrite_launch_as_repeat(body, target, binder, symbol, start, end, replacements)
            }
            ScheduleStep::Choose { options, .. } => {
                for (_, body) in options {
                    rewrite_launch_as_repeat(
                        body,
                        target,
                        binder,
                        symbol,
                        start,
                        end,
                        replacements,
                    );
                }
            }
            _ => {}
        }
    }
}

pub struct ScheduleConstruction<B: KernelDialect> {
    owner: OwnerToken,
    launches: Vec<Launch>,
    slots: Vec<AnyScalarSlot>,
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
    Imported(Vec<ScheduleStep>),
    If {
        condition: BoolExpr,
        then_region: u32,
        else_region: u32,
    },
    Repeat {
        binder: LoopBinderId,
        symbol: SymbolId,
        start: NatExpr,
        end: NatExpr,
        body_region: u32,
    },
    Choose {
        decision: DecisionId,
        options: Vec<(i64, u32)>,
    },
}

impl<B: KernelDialect> ScheduleConstruction<B> {
    pub(crate) fn new(owner: OwnerToken) -> Self {
        Self {
            owner,
            launches: Vec::new(),
            slots: Vec::new(),
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
    pub(crate) fn builder<'a>(
        &'a mut self,
        arena: &'a mut ExprArena,
        views: &'a [crate::storage::BufferViewLayout],
    ) -> ScheduleBuilder<'a, B> {
        self.builder_at(arena, views, 0)
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
        });
        (then_region, else_region)
    }
    pub fn begin_repeat(
        &mut self,
        arena: &mut ExprArena,
        parent: u32,
        start: NatExpr,
        end: NatExpr,
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
                body_region,
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
    pub fn slot_any(
        &mut self,
        arena: &mut ExprArena,
        dtype: DType,
        sort: seismic_lang::expr::SymbolSort,
    ) -> AnyScalarSlot {
        assert!(
            !self.closed,
            "a closed schedule cannot allocate a result slot"
        );
        assert!(
            match sort {
                seismic_lang::expr::SymbolSort::Nat => dtype == DType::U32,
                seismic_lang::expr::SymbolSort::Int => dtype == DType::I32,
                seismic_lang::expr::SymbolSort::Scalar(scalar) => dtype == scalar,
            },
            "scalar-slot dtype and expression sort differ"
        );
        let index = self.slots.len() as u32;
        let symbol = arena.schedule_slot(index, sort);
        let slot = AnyScalarSlot {
            owner: self.owner,
            index,
            dtype,
            symbol,
            sort,
        };
        self.slots.push(slot);
        slot
    }
    pub(crate) fn finish(mut self, token: ClosedSchedule) -> ParametricSchedule {
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
            launches: self.launches,
            slots: self.slots,
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
                        ScheduleStep::Launch(id) => launches.push((*id, at)),
                        ScheduleStep::Copy(copy) => {
                            direct.push((copy.source, at.clone()));
                            direct.push((copy.destination, at));
                        }
                        ScheduleStep::Fill(fill) => direct.push((fill.destination, at)),
                        ScheduleStep::ScalarRead(read) => direct.push((read.source, at)),
                        ScheduleStep::ScalarMove(_) | ScheduleStep::Check(_) => {}
                        ScheduleStep::If { .. }
                        | ScheduleStep::Repeat { .. }
                        | ScheduleStep::Choose { .. } => {
                            panic!("structured control cannot be inserted as a leaf")
                        }
                    }
                    out.push(step);
                }
                RegionStep::Imported(steps) => out.extend(steps),
                RegionStep::If {
                    condition,
                    then_region,
                    else_region,
                } => out.push(ScheduleStep::If {
                    condition,
                    then_steps: self.lower_region(then_region, direct, launches),
                    else_steps: self.lower_region(else_region, direct, launches),
                }),
                RegionStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    body_region,
                } => out.push(ScheduleStep::Repeat {
                    binder,
                    symbol,
                    start,
                    end,
                    body: self.lower_region(body_region, direct, launches),
                }),
                RegionStep::Choose { decision, options } => {
                    let options = options
                        .into_iter()
                        .map(|(value, child)| (value, self.lower_region(child, direct, launches)))
                        .collect();
                    out.push(ScheduleStep::Choose { decision, options });
                }
            }
        }
        out
    }

    pub(crate) fn import(
        &mut self,
        arena: &mut ExprArena,
        child: ParametricSchedule,
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
                mapped.sort, child.slots[*index as usize].sort,
                "spliced scalar-slot sort mismatch"
            );
        }
        let mut slots = Vec::with_capacity(child.slots.len());
        for child_slot in &child.slots {
            let mapped = forced_slots
                .iter()
                .find_map(|(index, slot)| (*index == child_slot.index).then_some(*slot))
                .unwrap_or_else(|| self.slot_any(arena, child_slot.dtype, child_slot.sort));
            assert_eq!(
                mapped.dtype, child_slot.dtype,
                "spliced scalar-slot dtype mismatch"
            );
            slots.push(mapped);
        }
        let mut launches = Vec::with_capacity(child.launches.len());
        for launch in child.launches {
            let kernel = *kernels
                .get(launch.kernel.index() as usize)
                .expect("child kernel remap is complete");
            let id = LaunchId::new(self.owner, self.launches.len() as u32);
            self.launches.push(Launch { kernel, ..launch });
            launches.push(id);
        }
        let map_view = |view: AnyBufferView| {
            *views
                .get(view.index as usize)
                .expect("child view remap is complete")
        };
        let map_slot = |slot: AnyScalarSlot| slots[slot.index as usize];
        let map_launch = |launch: LaunchId| launches[launch.index() as usize];
        let steps = remap_steps(child.steps, map_view, map_slot, map_launch);
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
            steps,
            direct,
            launch_uses,
            slots,
        }
    }

    fn append_choose_at(
        &mut self,
        parent: u32,
        decision: DecisionId,
        mut options: Vec<(i64, ImportedSchedule)>,
    ) {
        options.sort_by_key(|(value, _)| *value);
        let control = self.next_control;
        self.next_control = self
            .next_control
            .checked_add(1)
            .unwrap_or_else(|| panic!("schedule control identity space exhausted"));
        let parent_ordinal = self.regions[parent as usize].steps.len() as u32;
        let mut region_options = Vec::with_capacity(options.len());
        for (value, imported) in options {
            assert_eq!(
                imported.owner, self.owner,
                "imported schedule belongs to another construction"
            );
            let edge = ScheduleRegionEdge::ChooseOption {
                node: control,
                parent_ordinal,
                value,
            };
            let region = self.regions.len() as u32;
            let mut path = self.regions[parent as usize].path.clone();
            path.push(edge);
            self.regions.push(Region {
                path: path.clone(),
                steps: vec![RegionStep::Imported(imported.steps)],
            });
            for (view, mut at) in imported.direct {
                at.owner = self.owner;
                let mut prefixed = path.clone();
                prefixed.extend(at.region);
                at.region = prefixed;
                self.imported_direct.push((view, at));
            }
            for (launch, mut at) in imported.launch_uses {
                at.owner = self.owner;
                let mut prefixed = path.clone();
                prefixed.extend(at.region);
                at.region = prefixed;
                self.imported_launch.push((launch, at));
            }
            region_options.push((value, region));
        }
        self.regions[parent as usize]
            .steps
            .push(RegionStep::Choose {
                decision,
                options: region_options,
            });
    }

    fn append_imported_at(&mut self, parent: u32, imported: ImportedSchedule) {
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
        let mut prefix = self.regions[parent as usize].path.clone();
        prefix.push(ScheduleRegionEdge::Imported {
            node,
            parent_ordinal,
        });
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
        self.regions[parent as usize]
            .steps
            .push(RegionStep::Imported(imported.steps));
    }
}

#[derive(Debug)]
pub struct ImportedSchedule {
    owner: OwnerToken,
    pub(crate) steps: Vec<ScheduleStep>,
    pub(crate) direct: Vec<(AnyBufferView, ScheduleUse)>,
    pub(crate) launch_uses: Vec<(LaunchId, ScheduleUse)>,
    pub(crate) slots: Vec<AnyScalarSlot>,
}

fn remap_steps(
    steps: Vec<ScheduleStep>,
    view: impl Fn(AnyBufferView) -> AnyBufferView + Copy,
    slot: impl Fn(AnyScalarSlot) -> AnyScalarSlot + Copy,
    launch: impl Fn(LaunchId) -> LaunchId + Copy,
) -> Vec<ScheduleStep> {
    steps
        .into_iter()
        .map(|step| match step {
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
            } => ScheduleStep::If {
                condition,
                then_steps: remap_steps(then_steps, view, slot, launch),
                else_steps: remap_steps(else_steps, view, slot, launch),
            },
            ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                body,
            } => ScheduleStep::Repeat {
                binder,
                symbol,
                start,
                end,
                body: remap_steps(body, view, slot, launch),
            },
            ScheduleStep::Choose { decision, options } => ScheduleStep::Choose {
                decision,
                options: options
                    .into_iter()
                    .map(|(value, body)| (value, remap_steps(body, view, slot, launch)))
                    .collect(),
            },
        })
        .collect()
}

pub struct ScheduleBuilder<'a, B: KernelDialect> {
    inner: internals::Builder<'a, B>,
}
impl<'a, B: KernelDialect> ScheduleBuilder<'a, B> {
    pub fn slot<T: ScalarType>(&mut self) -> ScalarSlotId<T> {
        self.inner.slot::<T>()
    }
    pub fn slot_symbol<T: ScalarType>(&mut self, slot: ScalarSlotId<T>) -> SymbolId {
        self.inner.slot_symbol(slot.erase(), T::SYMBOL_SORT)
    }
    pub fn launch(&mut self, launch: Launch) -> LaunchId {
        self.inner.launch(launch)
    }
    pub fn step_launch(&mut self, launch: LaunchId) -> ScheduleUse {
        self.inner.step_launch(launch)
    }
    pub fn copy<R: Representation>(
        &mut self,
        source: BufferViewId<R>,
        destination: BufferViewId<R>,
    ) -> ScheduleUse {
        self.inner.copy(source.erase(), destination.erase())
    }
    pub fn fill_zero<R: WritableRepresentation>(
        &mut self,
        destination: BufferViewId<R>,
    ) -> ScheduleUse {
        self.inner.fill(
            destination.erase(),
            crate::repr::zero_fill_of::<R::Element>(),
        )
    }
    pub fn fill<R: WritableRepresentation>(
        &mut self,
        destination: BufferViewId<R>,
        value: <R::Element as ScalarType>::Value,
    ) -> ScheduleUse {
        self.inner.fill(
            destination.erase(),
            crate::repr::fill_value_of::<R::Element>(value),
        )
    }
    pub fn scalar_move<T: ScalarType>(
        &mut self,
        from: ScalarSlotId<T>,
        to: ScalarSlotId<T>,
    ) -> ScheduleUse {
        self.inner.scalar_move(from.erase(), to.erase())
    }
    pub fn scalar_read<R: DenseRepresentation>(
        &mut self,
        source: BufferViewId<R>,
        index: Vec<NatExpr>,
        to: ScalarSlotId<R::Element>,
    ) -> ScheduleUse {
        self.inner.scalar_read(source.erase(), index, to.erase())
    }
    pub fn branch(
        &mut self,
        condition: BoolExpr,
        then: impl FnOnce(&mut ScheduleBuilder<'_, B>),
        otherwise: impl FnOnce(&mut ScheduleBuilder<'_, B>),
    ) {
        self.inner.branch(condition, then, otherwise)
    }
    pub fn repeat(
        &mut self,
        start: NatExpr,
        end: NatExpr,
        body: impl FnOnce(&mut ScheduleBuilder<'_, B>, LoopBinding),
    ) {
        self.inner.repeat(start, end, body)
    }
    pub fn choose(
        &mut self,
        decision: DecisionId,
        options: impl FnOnce(&mut ChoiceBuilder<'_, B>),
    ) {
        self.inner.choose(decision, options)
    }
    pub fn close(self) -> ClosedSchedule {
        self.inner.close()
    }
    pub fn slot_any(
        &mut self,
        dtype: DType,
        sort: seismic_lang::expr::SymbolSort,
    ) -> AnyScalarSlot {
        self.inner.state.slot_any(self.inner.arena, dtype, sort)
    }
    pub fn temporary_bool(&mut self) -> AnyScalarSlot {
        self.inner.state.slot_any(
            self.inner.arena,
            DType::Bool,
            seismic_lang::expr::SymbolSort::Scalar(DType::Bool),
        )
    }
    pub fn slot_symbol_any(&mut self, slot: AnyScalarSlot) -> SymbolId {
        self.inner.slot_symbol(slot, slot.sort)
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
            mode: LaunchMode::Independent,
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
        logical_base_argument: Option<u32>,
    ) -> ScheduleUse {
        let zero = self.inner.arena.nat(0);
        let launch = self.inner.launch(Launch {
            kernel,
            mode: contract.mode,
            grid: contract.grid,
            workgroup: contract.workgroup,
            empty: contract.empty,
            parallel_extent: Some(contract.parallel_extent),
            logical_base: logical_base_argument.map(|argument| LogicalLaunchBase {
                argument,
                value: zero,
            }),
        });
        self.inner.step_launch(launch)
    }
    pub fn splice(
        &mut self,
        decision: Option<DecisionId>,
        mut alternatives: Vec<(i64, ImportedSchedule)>,
    ) {
        match decision {
            Some(decision) => {
                let mut values: Vec<_> = alternatives.iter().map(|(value, _)| *value).collect();
                values.sort_unstable();
                assert_eq!(
                    values,
                    self.inner.arena.decision_domain(decision).values(),
                    "splice alternatives must cover the exact decision domain"
                );
                self.inner
                    .state
                    .append_choose_at(self.inner.region, decision, alternatives)
            }
            None => {
                assert_eq!(
                    alternatives.len(),
                    1,
                    "decision-free splice has exactly one portable alternative"
                );
                self.inner.state.append_imported_at(
                    self.inner.region,
                    alternatives.pop().expect("length checked").1,
                );
            }
        }
    }
}
pub struct ChoiceBuilder<'a, B: KernelDialect> {
    inner: internals::Choice<'a, B>,
}
impl<'a, B: KernelDialect> ChoiceBuilder<'a, B> {
    pub fn option(&mut self, value: i64, body: impl FnOnce(&mut ScheduleBuilder<'_, B>)) {
        self.inner.option(value, body)
    }
}
#[derive(Debug)]
pub struct ClosedSchedule {
    pub(crate) owner: OwnerToken,
}

mod internals {
    use super::*;
    pub(super) struct Builder<'a, B: KernelDialect> {
        pub(super) arena: &'a mut ExprArena,
        pub(super) views: &'a [crate::storage::BufferViewLayout],
        pub(super) state: &'a mut ScheduleConstruction<B>,
        pub(super) region: u32,
    }
    pub(super) struct Choice<'a, B: KernelDialect> {
        arena: &'a mut ExprArena,
        views: &'a [crate::storage::BufferViewLayout],
        state: &'a mut ScheduleConstruction<B>,
        parent: u32,
        control: u32,
        decision: DecisionId,
        options: Vec<(i64, u32)>,
    }
    impl<'a, B: KernelDialect> Builder<'a, B> {
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
        fn leaf(&mut self, step: ScheduleStep) -> ScheduleUse {
            let at = self.use_site();
            self.state.regions[self.region as usize]
                .steps
                .push(RegionStep::Leaf(step));
            at
        }
        fn child_region(&mut self, edge: ScheduleRegionEdge) -> u32 {
            let mut path = self.state.regions[self.region as usize].path.clone();
            path.push(edge);
            let id = self.state.regions.len() as u32;
            self.state.regions.push(Region {
                path,
                steps: Vec::new(),
            });
            id
        }
        fn control(&mut self) -> u32 {
            let id = self.state.next_control;
            self.state.next_control = self
                .state
                .next_control
                .checked_add(1)
                .unwrap_or_else(|| panic!("schedule control identity space exhausted"));
            id
        }
        pub(super) fn slot<T: ScalarType>(&mut self) -> ScalarSlotId<T> {
            let any = self.state.slot_any(self.arena, T::DTYPE, T::SYMBOL_SORT);
            ScalarSlotId::new(self.owner(), any.index, any.symbol)
        }
        pub(super) fn slot_symbol(
            &mut self,
            slot: AnyScalarSlot,
            sort: seismic_lang::expr::SymbolSort,
        ) -> SymbolId {
            self.assert_owner(slot.owner());
            let stored = self.state.slots[slot.index as usize];
            assert_eq!(stored.dtype, slot.dtype, "scalar slot dtype changed");
            assert_eq!(stored.sort, sort, "scalar slot sort changed");
            stored.symbol
        }
        pub(super) fn launch(&mut self, launch: Launch) -> LaunchId {
            self.assert_owner(launch.kernel.owner());
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
            assert_eq!(from.dtype, to.dtype, "scalar move dtype mismatch");
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
                    seismic_lang::registry::RepresentationKind::Packed(_) => DType::F32,
                    seismic_lang::registry::RepresentationKind::External(_) => {
                        panic!("external representations permit only sealed conversion")
                    }
                };
            assert_eq!(dtype, to.dtype, "scalar read destination dtype mismatch");
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
                    seismic_lang::registry::RepresentationKind::Packed(_) => {
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
                condition.dtype,
                DType::Bool,
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
            assert_eq!(condition.dtype, DType::U32, "preflight status must be u32");
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
        pub(super) fn repeat(
            &mut self,
            start: NatExpr,
            end: NatExpr,
            body: impl FnOnce(&mut ScheduleBuilder<'_, B>, LoopBinding),
        ) {
            let (body_region, binding) =
                self.state.begin_repeat(self.arena, self.region, start, end);
            {
                let mut child = ScheduleBuilder {
                    inner: Builder {
                        arena: &mut *self.arena,
                        views: self.views,
                        state: &mut *self.state,
                        region: body_region,
                    },
                };
                body(&mut child, binding);
            }
        }
        pub(super) fn choose(
            &mut self,
            decision: DecisionId,
            options: impl FnOnce(&mut ChoiceBuilder<'_, B>),
        ) {
            let control = self.control();
            let mut choice = ChoiceBuilder {
                inner: Choice {
                    arena: &mut *self.arena,
                    views: self.views,
                    state: &mut *self.state,
                    parent: self.region,
                    control,
                    decision,
                    options: Vec::new(),
                },
            };
            options(&mut choice);
            let Choice {
                decision,
                mut options,
                ..
            } = choice.inner;
            options.sort_by_key(|(value, _)| *value);
            let actual: Vec<i64> = options.iter().map(|(value, _)| *value).collect();
            assert_eq!(
                actual.as_slice(),
                self.arena.decision_domain(decision).values(),
                "Choose must define exactly one option for every decision value"
            );
            self.state.regions[self.region as usize]
                .steps
                .push(RegionStep::Choose { decision, options });
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
    impl<'a, B: KernelDialect> Choice<'a, B> {
        pub(super) fn option(
            &mut self,
            value: i64,
            body: impl FnOnce(&mut ScheduleBuilder<'_, B>),
        ) {
            assert!(
                !self.options.iter().any(|(existing, _)| *existing == value),
                "duplicate Choose option {value}"
            );
            assert!(
                self.arena
                    .decision_domain(self.decision)
                    .values()
                    .contains(&value),
                "Choose option is outside the decision domain"
            );
            let parent_ordinal = self.state.regions[self.parent as usize].steps.len() as u32;
            let mut path = self.state.regions[self.parent as usize].path.clone();
            path.push(ScheduleRegionEdge::ChooseOption {
                node: self.control,
                parent_ordinal,
                value,
            });
            let region = self.state.regions.len() as u32;
            self.state.regions.push(Region {
                path,
                steps: Vec::new(),
            });
            {
                let mut child = ScheduleBuilder {
                    inner: Builder {
                        arena: &mut *self.arena,
                        views: self.views,
                        state: &mut *self.state,
                        region,
                    },
                };
                body(&mut child);
            }
            self.options.push((value, region));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic_lang::expr::{Assignment, SymbolValue};

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
        let mut schedule = ParametricSchedule {
            owner,
            launches: vec![Launch {
                kernel,
                mode: LaunchMode::Independent,
                grid: [three, one, one],
                workgroup: [eight, one, one],
                empty: never_empty,
                parallel_extent: Some(seventeen),
                logical_base: Some(LogicalLaunchBase {
                    argument: 2,
                    value: zero,
                }),
            }],
            slots: Vec::new(),
            steps: vec![ScheduleStep::Launch(launch_id)],
            direct_view_uses: Vec::new(),
            launch_uses: Vec::new(),
        };

        let two_blocks = arena.nat(2);
        schedule.chunk_semantic_launch(&mut arena, launch_id, two_blocks);

        let ScheduleStep::Repeat {
            symbol, end, body, ..
        } = &schedule.steps[0]
        else {
            panic!("semantic launch was not wrapped in a repeat")
        };
        assert_eq!(arena.eval_nat(*end, &Assignment::new()).unwrap(), 2);
        assert!(matches!(body.as_slice(), [ScheduleStep::Launch(id)] if *id == launch_id));

        let launch = schedule.launch(launch_id);
        let logical = launch.logical_base.expect("logical base was discarded");
        assert_eq!(logical.argument, 2);
        let mut covered = Vec::new();
        for iteration in 0..2 {
            let mut values = Assignment::new();
            values.bind(*symbol, SymbolValue::Nat(iteration));
            let base = arena.eval_nat(logical.value, &values).unwrap();
            let blocks = arena.eval_nat(launch.grid[0], &values).unwrap();
            assert_eq!(blocks, if iteration == 0 { 2 } else { 1 });
            for physical in 0..blocks * 8 {
                let logical_index = base + physical;
                if logical_index < 17 {
                    covered.push(logical_index);
                }
            }
        }
        assert_eq!(covered, (0..17).collect::<Vec<_>>());
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
        let mut schedule = ParametricSchedule {
            owner,
            launches: vec![Launch {
                kernel: KernelId::new(owner, 0),
                mode: LaunchMode::Independent,
                grid: [two, one, one],
                workgroup: [eight, one, one],
                empty: arena.bool(false),
                parallel_extent: Some(sixteen),
                logical_base: Some(LogicalLaunchBase {
                    argument: 0,
                    value: zero,
                }),
            }],
            slots: Vec::new(),
            steps: vec![ScheduleStep::Launch(launch_id)],
            direct_view_uses: Vec::new(),
            launch_uses: Vec::new(),
        };

        schedule.chunk_semantic_launch(&mut arena, launch_id, two);
        let ScheduleStep::Repeat { end, .. } = schedule.steps[0] else {
            panic!("semantic launch was not wrapped in a repeat")
        };
        assert_eq!(arena.eval_nat(end, &Assignment::new()).unwrap(), 1);
    }
}
