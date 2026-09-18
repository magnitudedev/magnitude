//! Selected allocation and memory-ordering contracts shared by backends and accounting.
//! These describe declared storage and required lifetime, not native registers or traffic.
use crate::{dispatch::TileDeclaration, execution::Multiplicity};
use seismic_lang::ir::{OperationId, VarId};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Purpose {
    Value,
    ReductionInput,
    Merge,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId {
    pub operation: OperationId,
    pub variable: VarId,
    pub purpose: Purpose,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    Body(OperationId),
    Then(OperationId),
    Else(OperationId),
}
#[derive(Clone, Debug)]
pub struct ArrayAllocation {
    pub id: AllocationId,
    pub declaration: TileDeclaration,
    pub scope: Vec<Scope>,
}
/// Memory accesses ordered among lanes of one SIMD group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemorySpace {
    Threadgroup,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BarrierPurpose {
    Snapshot(Purpose),
    Copy,
    Owned,
    Merge,
    IntrinsicStore,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BarrierSite {
    pub operation: OperationId,
    pub variable: VarId,
    pub purpose: BarrierPurpose,
}
/// References to structured IR, before scalar SSA or native emission.
#[derive(Clone, Debug)]
pub enum ControlValue {
    Integer(seismic_lang::sym::Sym),
    Predicate(Box<seismic_lang::ir::Expr>),
}
#[derive(Clone, Debug)]
pub struct Barrier {
    pub memory: MemorySpace,
    pub scope: Vec<Scope>,
    /// Per-work-item executions, conditional on valid collective participation.
    pub executions: Arc<Multiplicity<ControlValue>>,
}
#[derive(Clone, Debug)]
pub struct LaunchMemory {
    /// This launch must observe completion of its predecessor before it starts.
    pub predecessor: Option<usize>,
    pub arrays: Vec<ArrayAllocation>,
    pub barriers: BTreeMap<BarrierSite, Barrier>,
    /// Sum of declared private arrays per lane, not a simultaneous residency claim.
    pub declared_private_bytes_per_lane: u64,
    /// Shared arrays are hoisted to launch scope by the current emission contract.
    pub shared_bytes_per_group: u64,
    pub unmodeled_fragments: Vec<OperationId>,
}
/// Device storage carrying a split phase's partial values to its merge launch.
/// Buffers currently have separate allocations; producer/consumer identities
/// describe the required lifetime, not an assertion that reuse is implemented.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchAllocation {
    pub index: usize,
    pub phase: usize,
    pub variable: VarId,
    pub dtype: seismic_lang::types::DType,
    pub elements_per_item: u64,
    pub work_items: u64,
    pub parts: u64,
    pub bytes: usize,
    pub producer: usize,
    pub consumer: usize,
}
#[derive(Clone, Debug, Default)]
pub struct MemoryPlan {
    launches: Vec<LaunchMemory>,
    scratch: Vec<ScratchAllocation>,
}
impl MemoryPlan {
    pub fn new(
        launches: Vec<LaunchMemory>,
        scratch: Vec<ScratchAllocation>,
    ) -> Result<Self, String> {
        let mut ids = HashSet::new();
        for (index, launch) in launches.iter().enumerate() {
            if launch.predecessor.is_some_and(|p| p >= index) {
                return Err("memory launch predecessor must precede the launch".into());
            }
            for allocation in &launch.arrays {
                if !ids.insert(allocation.id) {
                    return Err("duplicate memory allocation identity".into());
                }
            }
        }
        let mut bindings = HashSet::new();
        for (index, allocation) in scratch.iter().enumerate() {
            if allocation.index != index
                || !bindings.insert((allocation.phase, allocation.variable))
            {
                return Err("scratch allocation identity is ambiguous".into());
            }
            if allocation.parts == 0
                || allocation.producer >= allocation.consumer
                || allocation.consumer >= launches.len()
            {
                return Err("invalid scratch producer/consumer domain".into());
            }
            let mut cursor = Some(allocation.consumer);
            while let Some(launch) = cursor {
                if launch == allocation.producer {
                    break;
                }
                cursor = launches[launch].predecessor;
            }
            if cursor != Some(allocation.producer) {
                return Err("scratch consumer is not ordered after its producer".into());
            }
            let bytes = allocation
                .work_items
                .checked_mul(allocation.parts)
                .and_then(|n| n.checked_mul(allocation.elements_per_item))
                .and_then(|n| n.checked_mul(u64::from(allocation.dtype.bytes())))
                .and_then(|n| usize::try_from(n).ok());
            if bytes != Some(allocation.bytes) {
                return Err("scratch byte size disagrees with its layout".into());
            }
        }
        Ok(Self { launches, scratch })
    }
    pub fn scratch(&self) -> &[ScratchAllocation] {
        &self.scratch
    }
    pub fn launches(&self) -> &[LaunchMemory] {
        &self.launches
    }
}
