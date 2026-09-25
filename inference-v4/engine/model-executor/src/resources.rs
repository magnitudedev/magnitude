//! Seismic graph slots for numerical execution and owned import sources.

mod graph;
pub use graph::{
    GraphOutputOwner, GraphOutputTensor, NativeGraphOutputLease, NativeGraphPool,
    NativeGraphWorkspaceLease, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
};

use crate::Stored;
use crate::{
    ExecutionPlan, InvariantError, PreparedHeadGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, ResourceDomainId,
    WeightPlan,
};
use magnitude_artifacts::FileSource;
use magnitude_model_batching::LaunchClass;
use seismic::{BackendName, Device, HostRegion, ReadOnlyMappedRegion, Tensor};
use std::fmt;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PoolClass {
    Target(LaunchClass),
    Head(LaunchClass),
    Vision { patch_rows: usize },
    Import { bytes: u64 },
}

/// One plan-backed source tensor owned through import completion. On Metal it
/// views an immutable artifact mapping; other backends own a staged upload.
pub struct ImportWorkspaceLease {
    domain: ResourceDomainId,
    class: PoolClass,
    bytes: u64,
    source: ImportSource,
}

enum ImportSource {
    Mapped(Tensor),
    Staged(Tensor),
}

/// One page-rounded artifact range charged once and shared by the source
/// tensor views of an ordered import batch.
pub(crate) struct ImportWindow {
    source: Arc<FileSource>,
    start: u64,
    end: u64,
    data_offset: u64,
    region: ReadOnlyMappedRegion,
}

impl ImportWindow {
    #[cfg(unix)]
    pub(crate) fn new(
        execution: &ExecutionPlan,
        device: &Device,
        source: Arc<FileSource>,
        start: u64,
        end: u64,
    ) -> Result<Self, AllocationError> {
        let invalid = |detail: &str| {
            AllocationError::Plan(InvariantError {
                context: "import window",
                detail: detail.into(),
            })
        };
        if device.backend() != BackendName::Metal || end <= start {
            return Err(invalid(
                "a mapped window requires Metal and a nonempty range",
            ));
        }
        let window = source
            .map_window(start, end - start)
            .map_err(|error| AllocationError::Device(error.to_string()))?;
        if window.mapped_len() as u64 > execution.resources().qualification_peak_bytes() {
            return Err(invalid("mapped source exceeds admitted import transient"));
        }
        let pointer = std::ptr::NonNull::new(window.as_ref().as_ref().as_ptr() as *mut u8)
            .ok_or_else(|| invalid("mapped artifact window has no address"))?;
        // SAFETY: the window owns the immutable mapped range through every
        // device view and every in-flight submission using it.
        let host = unsafe { HostRegion::new(pointer, window.mapped_len(), window.clone()) };
        let region = ReadOnlyMappedRegion::new(device, host)
            .map_err(|error| AllocationError::Device(error.to_string()))?;
        Ok(Self {
            source,
            start,
            end,
            data_offset: window.data_offset() as u64,
            region,
        })
    }

    #[cfg(not(unix))]
    pub(crate) fn new(
        _execution: &ExecutionPlan,
        _device: &Device,
        _source: Arc<FileSource>,
        _start: u64,
        _end: u64,
    ) -> Result<Self, AllocationError> {
        Err(AllocationError::Device(
            "mapped import windows require a Unix host".into(),
        ))
    }

    pub(crate) fn tensor(
        &self,
        stored: &Stored,
        weight: &WeightPlan,
        count: u64,
    ) -> Result<Tensor, AllocationError> {
        let (source, offset, length) = stored.file_range();
        let end = offset.checked_add(length).ok_or_else(|| {
            AllocationError::Plan(InvariantError {
                context: "import window",
                detail: "source range overflows".into(),
            })
        })?;
        if !Arc::ptr_eq(source, &self.source) || offset < self.start || end > self.end {
            return Err(AllocationError::Plan(InvariantError {
                context: "import window",
                detail: "source tensor lies outside its mapped window".into(),
            }));
        }
        self.region
            .tensor(
                weight.source,
                &[count],
                self.data_offset + offset - self.start,
            )
            .map_err(|error| AllocationError::Device(error.to_string()))
    }
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
    pub fn source(&self) -> &Tensor {
        match &self.source {
            ImportSource::Mapped(tensor) | ImportSource::Staged(tensor) => tensor,
        }
    }
    pub fn staged_mut(&mut self) -> Option<&mut Tensor> {
        match &mut self.source {
            ImportSource::Mapped(_) => None,
            ImportSource::Staged(tensor) => Some(tensor),
        }
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

/// Numerical graph slots authorized by one ResourcePlan. Import sources are
/// allocated separately during component materialization.
pub struct AllocatedResources {
    domain: ResourceDomainId,
    target_graph: NativeGraphPool,
    target_readout_graph: NativeGraphPool,
    head_graph: Option<NativeGraphPool>,
    vision_graph: Option<NativeGraphPool>,
    state_graph: NativeGraphPool,
}

impl AllocatedResources {
    /// Sealed graph arenas remain owned by these pools while their slots are
    /// lent to launches or output views.
    pub fn committed_bytes(&self) -> Result<u64, &'static str> {
        [
            Some(&self.target_graph),
            Some(&self.target_readout_graph),
            self.head_graph.as_ref(),
            self.vision_graph.as_ref(),
            Some(&self.state_graph),
        ]
        .into_iter()
        .flatten()
        .try_fold(0u64, |bytes, pool| {
            bytes
                .checked_add(pool.committed_bytes())
                .ok_or("graph pool charge overflows")
        })
    }

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
    /// Imports are serialized at startup or optional-component materialization.
    /// Each lease owns its source tensor until physical completion.
    pub(crate) fn import_workspace(
        execution: &ExecutionPlan,
        weight: &WeightPlan,
        stored: &Stored,
        window: Option<&ImportWindow>,
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
            return Err(invalid(
                "source upload exceeds admitted startup transient bytes",
            ));
        }
        let staged = || {
            // SAFETY: NativeImportProgram fills this exact tensor with
            // write_from_host before binding it as a kernel source.
            unsafe { Tensor::uninitialized(device, weight.source, &[count]) }
                .map(ImportSource::Staged)
                .map_err(|error| AllocationError::Device(error.to_string()))
        };
        let source = if let Some(window) = window {
            ImportSource::Mapped(window.tensor(stored, weight, count)?)
        } else if device.backend() == BackendName::Metal {
            #[cfg(unix)]
            {
                let (file, offset, length) = stored.file_range();
                let end = offset
                    .checked_add(length)
                    .ok_or_else(|| invalid("source range overflows"))?;
                let mapped = ImportWindow::new(execution, device, file.clone(), offset, end)?;
                ImportSource::Mapped(mapped.tensor(stored, weight, count)?)
            }
            #[cfg(not(unix))]
            {
                staged()?
            }
        } else {
            staged()?
        };
        let tensor = match &source {
            ImportSource::Mapped(tensor) | ImportSource::Staged(tensor) => tensor,
        };
        if !tensor.belongs_to(device)
            || tensor.element() != weight.source
            || tensor.extents() != [count]
            || tensor.byte_len() != weight.source_bytes
            || matches!(&source, ImportSource::Staged(_))
                && tensor.storage_bytes() != weight.source_bytes
        {
            return Err(invalid(
                "source tensor differs from the admitted import contract",
            ));
        }
        Ok(ImportWorkspaceLease {
            domain,
            class: PoolClass::Import {
                bytes: weight.source_bytes,
            },
            bytes: weight.source_bytes,
            source,
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
        let target_readout_graph = NativeGraphPool::new(
            domain.clone(),
            target_readout_graphs.family(),
            readout_charge,
        )?;
        let head_graph = match (head_graphs, plan.head_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => Some(
                NativeGraphPool::new(domain.clone(), graphs.family(), charge)?,
            ),
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared head graph differs from admitted Seismic footprint".into(),
                }))
            }
        };
        let vision_graph = match (vision_graphs, plan.vision_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => Some(
                NativeGraphPool::new(domain.clone(), graphs.family(), charge)?,
            ),
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
        let state_graph =
            NativeGraphPool::new(domain.clone(), state_graphs.family(), state_charge)?;
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
