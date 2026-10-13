//! Seismic graph slots for numerical execution and owned import sources.

mod graph;
pub(crate) use graph::VisionGrowth;
pub use graph::{
    GraphOutputOwner, GraphOutputTensor, NativeGraphOutputLease, NativeGraphPool,
    NativeGraphWorkspaceLease, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
    VisionGraphPool,
};

use crate::Stored;
use crate::{
    ExecutionPlan, InvariantError, PreparedDrafterGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, ResourceDomainId,
    WeightPlan,
};
use magnitude_artifacts::FileSource;
use magnitude_batching::LaunchClass;
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

/// The tensor one import entry reads a weight from: a view of the mapped
/// artifact, or an upload its importer fills.
pub(crate) enum ImportSource {
    Mapped(Tensor),
    Staged(Tensor),
}

impl ImportSource {
    /// The source of `weight`'s import on `device`, the one way every
    /// import takes it, a load's and a tuning unit's alike. A weight
    /// uploaded as stored is a view of `window` where one is given: the
    /// artifact's own pages, which the process's footprint does not count.
    /// Any other is an upload of the weight's source bytes on the device,
    /// not yet written
    /// ([`crate::programs::native_import::fill_staged_source`] writes it).
    pub(crate) fn acquire(
        weight: &WeightPlan,
        stored: &Stored,
        window: Option<&ImportWindow>,
        device: &Device,
    ) -> Result<Self, AllocationError> {
        let count = weight
            .shape
            .iter()
            .try_fold(1u64, |count, extent| count.checked_mul(*extent))
            .ok_or_else(|| {
                AllocationError::Plan(InvariantError {
                    context: "import source",
                    detail: "weight element count overflows".into(),
                })
            })?;
        match window {
            // A transformed or dequantized weight uploads the host's
            // prepared bytes; only an unprepared one can read its stored
            // range in place.
            Some(window) if !weight.host_prepared() => {
                Ok(Self::Mapped(window.tensor(stored, weight, count)?))
            }
            // SAFETY: the importer fills this exact tensor before binding
            // it as a kernel source.
            _ => unsafe { Tensor::uninitialized(device, weight.upload, &[count]) }
                .map(Self::Staged)
                .map_err(|error| AllocationError::Device(error.to_string())),
        }
    }

    pub(crate) fn tensor(&self) -> &Tensor {
        match self {
            Self::Mapped(tensor) | Self::Staged(tensor) => tensor,
        }
    }

    pub(crate) fn staged_mut(&mut self) -> Option<&mut Tensor> {
        match self {
            Self::Mapped(_) => None,
            Self::Staged(tensor) => Some(tensor),
        }
    }

    pub(crate) fn into_tensor(self) -> Tensor {
        match self {
            Self::Mapped(tensor) | Self::Staged(tensor) => tensor,
        }
    }
}

/// One page-rounded artifact range charged once and shared by the source
/// tensor views of an ordered import batch.
pub(crate) struct ImportWindow {
    source: Arc<FileSource>,
    start: u64,
    end: u64,
    data_offset: u64,
    /// The bytes mapped: the range, rounded out to whole pages.
    mapped_bytes: u64,
    region: ReadOnlyMappedRegion,
}

impl ImportWindow {
    /// The window of exactly `stored`'s range in its artifact.
    pub(crate) fn of(device: &Device, stored: &Stored) -> Result<Self, AllocationError> {
        let (file, start, end) = stored_range(stored)?;
        Self::map(device, file, start, end)
    }
}

/// `stored`'s file and the range of it the tensor occupies.
fn stored_range(stored: &Stored) -> Result<(Arc<FileSource>, u64, u64), AllocationError> {
    let (file, offset, length) = stored.file_range();
    let end = offset.checked_add(length).ok_or_else(|| {
        AllocationError::Plan(InvariantError {
            context: "import window",
            detail: "source range overflows".into(),
        })
    })?;
    Ok((file.clone(), offset, end))
}

impl ImportWindow {
    /// A window of a load's import, held to the import transient its
    /// execution plan admitted.
    pub(crate) fn new(
        execution: &ExecutionPlan,
        device: &Device,
        source: Arc<FileSource>,
        start: u64,
        end: u64,
    ) -> Result<Self, AllocationError> {
        let window = Self::map(device, source, start, end)?;
        if window.mapped_bytes > execution.resources().qualification_peak_bytes() {
            return Err(AllocationError::Plan(InvariantError {
                context: "import window",
                detail: "mapped source exceeds admitted import transient".into(),
            }));
        }
        Ok(window)
    }

    /// Whether `device` reads an artifact's mapped pages in place: where it
    /// does, a weight uploaded as stored is imported from a window and
    /// nowhere else.
    pub(crate) fn serves(device: &Device) -> bool {
        cfg!(unix) && device.backend() == BackendName::Metal
    }

    /// The page-rounded range `start..end` of `source`, mapped for `device`.
    #[cfg(unix)]
    pub(crate) fn map(
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
        let mapped_bytes = window.mapped_len() as u64;
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
            mapped_bytes,
            region,
        })
    }

    #[cfg(not(unix))]
    pub(crate) fn map(
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
        self.source.tensor()
    }
    pub fn staged_mut(&mut self) -> Option<&mut Tensor> {
        self.source.staged_mut()
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
    /// The device's one graph workspace, which every pool's activations bind.
    arena: seismic::NativeExecutionArena,
    target_graph: NativeGraphPool,
    target_readout_graph: NativeGraphPool,
    head_graph: Option<NativeGraphPool>,
    vision_graph: Option<VisionGraphPool>,
    state_graph: NativeGraphPool,
}

impl AllocatedResources {
    /// The workspace arena once, and the arenas these pools own while their
    /// slots are lent to launches or output views.
    pub fn committed_bytes(&self) -> Result<u64, &'static str> {
        [
            Some(&self.target_graph),
            Some(&self.target_readout_graph),
            self.head_graph.as_ref(),
            Some(&self.state_graph),
        ]
        .into_iter()
        .flatten()
        .try_fold(self.arena.bytes(), |bytes, pool| {
            bytes
                .checked_add(pool.committed_bytes())
                .ok_or("graph pool charge overflows")
        })?
        .checked_add(
            self.vision_graph
                .as_ref()
                .map_or(0, VisionGraphPool::committed_bytes),
        )
        .ok_or("graph pool charge overflows")
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
    pub fn vision_graph(&self) -> Option<&VisionGraphPool> {
        self.vision_graph.as_ref()
    }
    pub(crate) fn vision_graph_mut(&mut self) -> Option<&mut VisionGraphPool> {
        self.vision_graph.as_mut()
    }
    pub fn state_graph(&self) -> &NativeGraphPool {
        &self.state_graph
    }

    /// The pool whose output slots `lane`'s launches retain beyond their
    /// completion: target readout or head outputs. Encoded images hold
    /// slots of their own class ([`VisionGraphPool`]).
    pub(crate) fn retained_outputs_mut(
        &mut self,
        lane: crate::domain::ReservationLane,
    ) -> Option<&mut NativeGraphPool> {
        match lane {
            crate::domain::ReservationLane::Target => Some(&mut self.target_readout_graph),
            crate::domain::ReservationLane::Head => self.head_graph.as_mut(),
            crate::domain::ReservationLane::Vision => None,
        }
    }

    /// Release every pool's free output slots beyond its startup count and
    /// vision's free storage; returns how many slots were released.
    pub(crate) fn release_idle_slots(&mut self) -> usize {
        [
            Some(&mut self.target_graph),
            Some(&mut self.target_readout_graph),
            self.head_graph.as_mut(),
            Some(&mut self.state_graph),
        ]
        .into_iter()
        .flatten()
        .map(NativeGraphPool::release_idle_outputs)
        .sum::<usize>()
            + self
                .vision_graph
                .as_mut()
                .map_or(0, VisionGraphPool::release_idle)
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
        // A weight imported alone, where the device reads mapped pages in
        // place, has a window of its own range; its source view keeps the
        // mapping.
        let own = match window {
            None if ImportWindow::serves(device) && !weight.host_prepared() => {
                let (file, start, end) = stored_range(stored)?;
                Some(ImportWindow::new(execution, device, file, start, end)?)
            }
            _ => None,
        };
        let source = ImportSource::acquire(weight, stored, window.or(own.as_ref()), device)?;
        let tensor = source.tensor();
        if !tensor.belongs_to(device)
            || tensor.element() != weight.upload
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
        head_graphs: Option<&PreparedDrafterGraphs>,
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
        let arena = device
            .execution_arena(plan.arena_bytes())
            .map_err(|error| AllocationError::Device(error.to_string()))?;
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
            NativeGraphPool::new(domain.clone(), target_graphs.family(), graph_charge, &arena)?;
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
            &arena,
        )?;
        let head_graph = match (head_graphs, plan.head_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => Some(
                NativeGraphPool::new(domain.clone(), graphs.family(), charge, &arena)?,
            ),
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared head graph differs from admitted Seismic footprint".into(),
                }));
            }
        };
        let vision_graph = match (vision_graphs, plan.vision_graph()) {
            (Some(graphs), Some(charge)) if admitted(graphs.family(), charge) => {
                Some(VisionGraphPool::new(
                    domain.clone(),
                    charge.upload_regions,
                    graphs.class_families(),
                ))
            }
            (None, None) => None,
            _ => {
                return Err(AllocationError::Plan(InvariantError {
                    context: "resource allocator",
                    detail: "prepared vision graph differs from admitted Seismic footprint".into(),
                }));
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
            NativeGraphPool::new(domain.clone(), state_graphs.family(), state_charge, &arena)?;
        let allocated = AllocatedResources {
            domain: domain.clone(),
            arena,
            target_graph,
            target_readout_graph,
            head_graph,
            vision_graph,
            state_graph,
        };
        Ok(allocated)
    }
}
