//! Seismic graph slots for numerical execution and an owned import upload.

mod graph;
pub use graph::{
    GraphOutputOwner, GraphOutputTensor, NativeGraphOutputLease, NativeGraphPool,
    NativeGraphWorkspaceLease, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
};

use crate::{
    device_resources::RetentionClaim, ExecutionPlan, InvariantError, PreparedHeadGraphs,
    PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs, ResourceDomainId, WeightPlan,
};
use magnitude_model_batching::LaunchClass;
use seismic::{Device, Tensor};
use std::{
    cell::Cell,
    fmt,
    rc::Rc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolClass {
    Target(LaunchClass),
    Head(LaunchClass),
    Vision { patch_rows: usize },
    State { rows: usize },
    Import { bytes: u64 },
}

/// One exact, plan-backed upload tensor owned through import submission.
/// It is released when the completed submission drops the lease.
pub struct ImportWorkspaceLease {
    domain: ResourceDomainId,
    class: PoolClass,
    bytes: u64,
    upload: Tensor,
}

impl ImportWorkspaceLease {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }
    pub fn class(&self) -> PoolClass {
        self.class
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn upload(&self) -> &Tensor {
        &self.upload
    }
    pub fn upload_mut(&mut self) -> &mut Tensor {
        &mut self.upload
    }
}

impl fmt::Debug for ImportWorkspaceLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ImportWorkspaceLease")
            .field("domain", &self.domain)
            .field("class", &self.class())
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationError {
    Device(String),
    Plan(InvariantError),
}

impl fmt::Display for AllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(error) => write!(formatter, "resource allocation: {error}"),
            Self::Plan(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for AllocationError {}

/// Numerical graph slots authorized by one ResourcePlan. One-shot import
/// uploads are allocated separately during component materialization.
pub struct AllocatedResources {
    domain: ResourceDomainId,
    target_graph: NativeGraphPool,
    target_readout_graph: NativeGraphPool,
    head_graph: Option<NativeGraphPool>,
    vision_graph: Option<NativeGraphPool>,
    state_graph: NativeGraphPool,
}

impl AllocatedResources {
    pub fn domain(&self) -> &ResourceDomainId {
        &self.domain
    }

    pub fn target_graph(&self) -> &NativeGraphPool {
        &self.target_graph
    }

    pub fn target_readout_graph(&self) -> &NativeGraphPool {
        &self.target_readout_graph
    }
    pub fn head_graph(&self) -> Option<&NativeGraphPool> {
        self.head_graph.as_ref()
    }
    pub fn vision_graph(&self) -> Option<&NativeGraphPool> {
        self.vision_graph.as_ref()
    }
    pub fn state_graph(&self) -> &NativeGraphPool {
        &self.state_graph
    }
}

pub struct ResourceAllocator;

impl ResourceAllocator {
    /// Variable retained features use an exact plan-backed charge rather than
    /// a fixed-shape pool. The returned claim refunds the charge on final drop.
    pub(crate) fn retained_feature(
        execution: &ExecutionPlan,
        device: &Device,
        source: &Tensor,
        used: &Rc<Cell<u64>>,
    ) -> Result<(Tensor, Rc<RetentionClaim>), AllocationError> {
        let invalid = |detail: &str| {
            AllocationError::Plan(InvariantError {
                context: "retained feature allocation",
                detail: detail.into(),
            })
        };
        let bytes = source.byte_len();
        let next = used
            .get()
            .checked_add(bytes)
            .ok_or_else(|| invalid("byte charge overflows"))?;
        if next > execution.resources().bytes().retained_features {
            return Err(invalid("byte charge exceeds the admitted method budget"));
        }
        let tensor = Tensor::zeros(device, source.element(), source.extents())
            .map_err(|error| AllocationError::Device(error.to_string()))?;
        if tensor.storage_bytes() != bytes {
            return Err(invalid("storage differs from the preflight byte charge"));
        }
        used.set(next);
        Ok((tensor, Rc::new(RetentionClaim::new(used.clone(), bytes))))
    }

    /// Imports are serialized at startup or optional-component materialization.
    /// Each upload owns one exact source tensor until physical completion.
    pub fn import_workspace(
        execution: &ExecutionPlan,
        weight: &WeightPlan,
        device: &Device,
        domain: ResourceDomainId,
    ) -> Result<ImportWorkspaceLease, AllocationError> {
        let invalid = |detail: &str| {
            AllocationError::Plan(InvariantError {
                context: "import workspace",
                detail: detail.into(),
            })
        };
        if !execution.weights().any(|planned| planned == weight) {
            return Err(invalid("weight is absent from the execution plan"));
        }
        let count = weight
            .shape
            .iter()
            .try_fold(1u64, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| invalid("weight element count overflows"))?;
        if weight.source_bytes > execution.resources().qualification_peak_bytes() {
            return Err(invalid("source upload exceeds admitted startup transient bytes"));
        }
        let upload = Tensor::zeros(device, weight.source, &[count])
            .map_err(|error| AllocationError::Device(error.to_string()))?;
        if !upload.belongs_to(device)
            || upload.element() != weight.source
            || upload.extents() != [count]
            || upload.byte_len() != weight.source_bytes
            || upload.storage_bytes() != weight.source_bytes
        {
            return Err(invalid("upload tensor differs from the admitted source contract"));
        }
        Ok(ImportWorkspaceLease {
            domain,
            class: PoolClass::Import {
                bytes: weight.source_bytes,
            },
            bytes: weight.source_bytes,
            upload,
        })
    }

    pub fn allocate(
        execution: &ExecutionPlan,
        device: &Device,
        domain: ResourceDomainId,
        target_graphs: &PreparedTargetGraphs,
        target_readout_graphs: &PreparedTargetReadoutGraphs,
        head_graphs: Option<&PreparedHeadGraphs>,
        vision_graphs: Option<&PreparedVisionGraphs>,
        state_graphs: &PreparedStateCopyGraphs,
    ) -> Result<AllocatedResources, AllocationError> {
        if device.backend() != execution.device().backend()
            || device.info().name != execution.device().name()
        {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "opened device differs from the selected execution plan".into(),
            }));
        }
        let plan = execution.resources();
        // The admitted charge was derived from this family's footprint.
        let admitted = |family: &seismic::NativeGraphFamily, charge: crate::NativeGraphCharge| {
            family.workspace_bytes() == charge.workspace_bytes
                && family.output_bytes() == charge.output_bytes
                && family.upload_bytes() == charge.upload_bytes
        };
        let graph_charge = plan.target_graph();
        if !admitted(target_graphs.family(), graph_charge)
            || target_graphs.runs_per_step() != graph_charge.upload_regions
        {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "prepared target graph differs from admitted Seismic footprint".into(),
            }));
        }
        let target_graph =
            NativeGraphPool::new(domain.clone(), target_graphs.family(), graph_charge)?;
        let readout_charge = plan.target_readout_graph();
        if !admitted(target_readout_graphs.family(), readout_charge) {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "prepared target readout graph differs from admitted Seismic footprint"
                    .into(),
            }));
        }
        let target_readout_graph =
            NativeGraphPool::new(domain.clone(), target_readout_graphs.family(), readout_charge)?;
        let head_graph = match (head_graphs, plan.head_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => {
                Some(NativeGraphPool::new(domain.clone(), graphs.family(), charge)?)
            }
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared head graph differs from admitted Seismic footprint".into(),
                }))
            }
        };
        let vision_graph = match (vision_graphs, plan.vision_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => {
                Some(NativeGraphPool::new(domain.clone(), graphs.family(), charge)?)
            }
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared vision graph differs from admitted Seismic footprint".into(),
                }))
            }
        };
        let state_charge = plan.state_graph();
        if !admitted(state_graphs.family(), state_charge) {
            return Err(AllocationError::Plan(InvariantError {
                context: "resource allocator",
                detail: "prepared state graph differs from admitted Seismic footprint".into(),
            }));
        }
        let state_graph = NativeGraphPool::new(domain.clone(), state_graphs.family(), state_charge)?;
        let allocated = AllocatedResources {
            domain: domain.clone(),
            target_graph,
            target_readout_graph,
            head_graph,
            vision_graph,
            state_graph,
        };
        Ok(allocated)
    }
}
