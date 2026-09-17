//! Dispatch and storage facts emitted alongside native source. These describe
//! declared work/storage, not register allocation, occupancy or memory service.
use seismic_lang::types::DType;
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
    /// Each lane owns ceil(capacity / lanes_per_item) array slots.
    Distributed,
    /// Each work item owns a disjoint shared array within its group.
    GroupShared,
}
#[derive(Clone, Debug)]
pub struct TileDeclaration {
    pub symbol: String,
    pub dtype: DType,
    pub capacity: u64,
    pub placement: TilePlacement,
}
impl TileDeclaration {
    /// Declared array storage. This is not native register bytes or a lifetime peak.
    pub fn bytes(&self, dispatch: &GroupDispatch) -> Result<(u64, u64), String> {
        let width = u64::from(self.dtype.bytes());
        match self.placement {
            TilePlacement::Replicated => Ok((
                self.capacity
                    .max(1)
                    .checked_mul(width)
                    .ok_or("tile byte overflow")?,
                0,
            )),
            TilePlacement::Distributed => Ok((
                self.capacity
                    .div_ceil(dispatch.lanes_per_item)
                    .checked_mul(width)
                    .ok_or("tile byte overflow")?,
                0,
            )),
            TilePlacement::GroupShared => Ok((
                0,
                self.capacity
                    .max(1)
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(dispatch.items_per_group))
                    .ok_or("shared tile byte overflow")?,
            )),
        }
    }
}
