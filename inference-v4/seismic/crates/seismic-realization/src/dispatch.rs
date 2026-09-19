//! Dispatch and storage declarations shared by accounting and emission. These describe
//! declared work/storage, not register allocation, occupancy or memory service.
use seismic_lang::types::DType;
pub mod geometry;

/// Row-major coordinates assigned to a work item, with explicit tail extents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkMapping {
    axes: Vec<AxisMapping>,
    work_items: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisMapping {
    pub logical_extent: u64,
    /// Number of work coordinates along this axis, before applying the step.
    pub extent: u64,
    pub stride: u64,
    pub step: u64,
}

impl WorkMapping {
    pub fn new(extents: &[u64], steps: &[u64]) -> Result<Self, String> {
        if extents.len() != steps.len() {
            return Err("work mapping needs one step per axis".into());
        }
        let geometry = geometry::mapping(&mut geometry::Concrete, extents, steps)?;
        let axes = extents
            .iter()
            .zip(steps)
            .enumerate()
            .map(|(index, (&logical_extent, &step))| AxisMapping {
                logical_extent,
                extent: geometry.counts[index],
                stride: geometry.strides[index],
                step,
            })
            .collect();
        let work_items = geometry.work_items;
        Ok(Self { axes, work_items })
    }
    pub fn axes(&self) -> &[AxisMapping] {
        &self.axes
    }
    pub fn work_items(&self) -> u64 {
        self.work_items
    }
    /// Logical base coordinates for this work item. Empty domains and padding
    /// lanes have no coordinates and must not be evaluated by code generation.
    pub fn coordinates(&self, item: u64) -> Result<Vec<u64>, String> {
        if item >= self.work_items {
            return Err("work item is outside the mapped domain".into());
        }
        Ok(self
            .axes
            .iter()
            .map(|axis| (item / axis.stride % axis.extent) * axis.step)
            .collect())
    }
    pub fn extents(&self, item: u64) -> Result<Vec<u64>, String> {
        Ok(self
            .coordinates(item)?
            .into_iter()
            .zip(&self.axes)
            .map(|(base, axis)| axis.step.min(axis.logical_extent - base))
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupDispatch {
    pub work_items: u64,
    pub lanes_per_item: u64,
    pub items_per_group: u64,
    pub groups: u64,
    pub threads_per_group: u64,
}
impl GroupDispatch {
    pub fn new(work_items: u64, lanes_per_item: u64, items_per_group: u64) -> Result<Self, String> {
        if lanes_per_item == 0 || items_per_group == 0 {
            return Err("dispatch widths must be positive".into());
        }
        let geometry = geometry::dispatch(
            &mut geometry::Concrete,
            work_items,
            lanes_per_item,
            items_per_group,
        )?;
        let threads_per_group = geometry.threads_per_group;
        let groups = geometry.groups;
        Ok(Self {
            work_items,
            lanes_per_item,
            items_per_group,
            groups,
            threads_per_group,
        })
    }
    pub fn dispatched_lanes(&self) -> u64 {
        self.groups * self.threads_per_group
    }
    pub fn participating_lanes(&self) -> u64 {
        self.work_items * self.lanes_per_item
    }
    pub fn padding_lanes(&self) -> u64 {
        self.dispatched_lanes() - self.participating_lanes()
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TilePlacement {
    /// Every participating lane owns a complete declared array.
    Replicated,
    /// Each lane owns ceil(capacity / lanes_per_item) array slots, with a minimum of one.
    Distributed,
    /// Each work item owns a disjoint shared array within its group.
    GroupShared,
    /// One shared array per group, common to all its work items: a tile of the outer owner
    /// of a launch whose work items are the inner owners of one piece. Every lane of the
    /// group cooperates on its element loops.
    GroupWide,
}
impl TilePlacement {
    /// The tile is an array in the group's shared memory.
    pub fn group_memory(&self) -> bool {
        matches!(self, Self::GroupShared | Self::GroupWide)
    }
}
/// The element type a tile is held in inside a thread or threadgroup array. This is the one
/// rule for local storage types; declarations, byte accounting and emission all follow the
/// declaration it produces.
///
/// Half-width floats are held widened to `f32`: Apple's Metal compiler miscompiles
/// thread-address-space `bfloat` arrays (bugs/26-09-19/metal-thread-local-bfloat-arrays.md),
/// and `half` shares the path. Widening is exact; a value is rounded to its logical type
/// before it is stored, and narrowed again only when published to device memory. A
/// matrix-intrinsic operand keeps its native type: `simdgroup_load`/`store` address it as
/// the matrix element type.
pub fn local_storage_dtype(dtype: DType, native_operand: bool) -> DType {
    match dtype {
        DType::BF16 | DType::F16 if !native_operand => DType::F32,
        other => other,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileDeclaration {
    pub symbol: String,
    pub dtype: DType,
    pub capacity: u64,
    pub placement: TilePlacement,
}
/// Physical array geometry shared by code generation and storage accounting.
/// Empty logical tiles still require one element in a declared native array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLayout {
    pub private_elements_per_lane: u64,
    pub shared_elements_per_item: u64,
    pub private_bytes_per_lane: u64,
    pub shared_bytes_per_group: u64,
}
impl TileDeclaration {
    pub fn layout(&self, dispatch: &GroupDispatch) -> Result<TileLayout, String> {
        if dispatch.lanes_per_item == 0 || dispatch.items_per_group == 0 {
            return Err("storage layout requires positive dispatch widths".into());
        }
        let layout = geometry::storage(
            &mut geometry::Concrete,
            self.capacity,
            u64::from(self.dtype.bytes()),
            &self.placement,
            dispatch.lanes_per_item,
            dispatch.items_per_group,
        )?;
        Ok(TileLayout {
            private_elements_per_lane: layout.private_elements_per_lane,
            shared_elements_per_item: layout.shared_elements_per_item,
            private_bytes_per_lane: layout.private_bytes_per_lane,
            shared_bytes_per_group: layout.shared_bytes_per_group,
        })
    }

    /// Declared array storage. This is not native register bytes or a lifetime peak.
    pub fn bytes(&self, dispatch: &GroupDispatch) -> Result<(u64, u64), String> {
        let layout = self.layout(dispatch)?;
        Ok((layout.private_bytes_per_lane, layout.shared_bytes_per_group))
    }
}

/// Executable coordinate ownership within a logical work item. Replication is
/// private storage replication; publications still need a unique participant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ownership {
    Replicated,
    Cyclic { consecutive: u64 },
}
impl Ownership {
    pub fn period(self, lanes: u64) -> Result<u64, String> {
        if lanes == 0 {
            return Err("ownership requires positive participation".into());
        }
        match self {
            Self::Replicated => Ok(1),
            Self::Cyclic { consecutive } if consecutive > 0 => lanes
                .checked_mul(consecutive)
                .ok_or_else(|| "ownership period overflow".into()),
            _ => Err("cyclic ownership requires a positive consecutive extent".into()),
        }
    }
    pub fn owner(self, coordinate: u64, lanes: u64) -> Result<Option<u64>, String> {
        self.period(lanes)?;
        Ok(match self {
            Self::Replicated => None,
            Self::Cyclic { consecutive } => Some(coordinate / consecutive % lanes),
        })
    }
}

/// Source work item participation, private value placement, and publication
/// ownership retained by scalar instruction preparation and backend dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Participation {
    Thread,
    Subgroup { lanes: u32 },
}
impl Participation {
    pub fn lanes(self) -> u32 {
        match self {
            Self::Thread => 1,
            Self::Subgroup { lanes } => lanes,
        }
    }
}
