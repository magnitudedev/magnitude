//! Dispatch and storage declarations shared by accounting and emission. These describe
//! declared work/storage, not register allocation, occupancy or memory service.
use seismic_lang::types::DType;

/// Row-major coordinates assigned to a work item. A step covers consecutive
/// logical coordinates along an axis; partial steps require a different mapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkMapping {
    axes: Vec<AxisMapping>,
    work_items: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisMapping {
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
        let mut axes = extents
            .iter()
            .zip(steps)
            .map(|(&extent, &step)| {
                if step == 0 || !extent.is_multiple_of(step) {
                    return Err("work mapping step must be positive and divide its extent");
                }
                Ok(AxisMapping {
                    extent: extent / step,
                    stride: 0,
                    step,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Empty domains contain no coordinates; zero strides are never evaluated.
        let work_items = if axes.iter().any(|axis| axis.extent == 0) {
            0
        } else {
            let mut stride = 1u64;
            for axis in axes.iter_mut().rev() {
                axis.stride = stride;
                stride = stride
                    .checked_mul(axis.extent)
                    .ok_or("work mapping extent product overflow")?;
            }
            stride
        };
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
        let threads_per_group = lanes_per_item
            .checked_mul(items_per_group)
            .ok_or("thread count overflow")?;
        let groups = work_items.div_ceil(items_per_group);
        groups
            .checked_mul(threads_per_group)
            .ok_or("dispatch lane count overflow")?;
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
        let (private, shared) = match self.placement {
            TilePlacement::Replicated => (self.capacity.max(1), 0),
            TilePlacement::Distributed => {
                (self.capacity.div_ceil(dispatch.lanes_per_item).max(1), 0)
            }
            TilePlacement::GroupShared => (0, self.capacity.max(1)),
        };
        let width = u64::from(self.dtype.bytes());
        Ok(TileLayout {
            private_elements_per_lane: private,
            shared_elements_per_item: shared,
            private_bytes_per_lane: private
                .checked_mul(width)
                .ok_or("private tile byte overflow")?,
            shared_bytes_per_group: shared
                .checked_mul(width)
                .and_then(|n| n.checked_mul(dispatch.items_per_group))
                .ok_or("shared tile byte overflow")?,
        })
    }
    /// Declared array storage. This is not native register bytes or a lifetime peak.
    pub fn bytes(&self, dispatch: &GroupDispatch) -> Result<(u64, u64), String> {
        let layout = self.layout(dispatch)?;
        Ok((layout.private_bytes_per_lane, layout.shared_bytes_per_group))
    }
}
