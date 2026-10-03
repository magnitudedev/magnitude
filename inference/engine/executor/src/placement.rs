//! Backend-neutral placement intent over one original model.
//!
//! These descriptors authorize no allocation or execution. A strategy must
//! project and admit them before opening devices. Seismic owns device discovery
//! and physical memory identity; selectors here name execution endpoints only.
use magnitude_family_contracts::WeightRole;
use seismic::DeviceSelector;
use std::ops::Range;

/// Ordered execution groups. Multiple partitions within a group describe shards
/// of the same logical work, rather than consecutive whole-layer stages.
#[derive(Clone, Debug)]
pub struct ModelPlacement {
    pub groups: Vec<PlacementGroup>,
}

#[derive(Clone, Debug)]
pub struct PlacementGroup {
    pub partitions: Vec<PartitionAssignment>,
}

#[derive(Clone, Debug)]
pub struct PartitionAssignment {
    pub device: DeviceSelector,
    pub region: ModelRegion,
}

#[derive(Clone, Debug)]
pub enum ModelRegion {
    /// Original/global decoder blocks; endpoint ownership is derived by the
    /// pipeline projection, never authored as independent flags.
    DecoderBlocks(Range<usize>),
    /// Logical portion of an original semantic weight tensor. Axis and range
    /// use decoded tensor coordinates, independent of packed storage layout.
    /// This describes intent only: tensor geometry, complete shard coverage,
    /// activation/state partitioning and collectives need a future admission.
    TensorSlice {
        role: WeightRole,
        axis: usize,
        elements: Range<u64>,
    },
}

impl ModelPlacement {
    /// Explicit whole-block placement, without backend or stage-count policy.
    pub fn pipeline(stages: impl IntoIterator<Item = (DeviceSelector, Range<usize>)>) -> Self {
        Self {
            groups: stages
                .into_iter()
                .map(|(device, blocks)| PlacementGroup {
                    partitions: vec![PartitionAssignment {
                        device,
                        region: ModelRegion::DecoderBlocks(blocks),
                    }],
                })
                .collect(),
        }
    }
}
