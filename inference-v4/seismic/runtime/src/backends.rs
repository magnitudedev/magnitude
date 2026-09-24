//! The runtime's one closed sum over supported backends.
//!
//! Backend crates remain generic and do not depend on the runtime.  This
//! module is the sole composition root which turns their target profiles,
//! device services, executors, and prepared kernels into the public device
//! API.  It does not reproduce target facts or planning policy.

use crate::api::{
    device::DeviceInner,
    kernel::{
        DecodedResults, EncodedArgs, EncodedWorkflowArgs, PendingWorkflowResults,
        WorkflowCompletionAny,
    },
    CallError, WorkflowError,
};
use crate::devices::{
    Availability, CapacityBasis, DeviceInfo, DeviceKind, DeviceMeasurements, DeviceMemoryStatus,
    DeviceSelector, DiscoveryDiagnostic, LedgerKey, ObservationError, OpenError,
};
use crate::driver::{self, Opened, PreparedHandle};
use crate::memory::MemoryDomain;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::NativeExecutor;
use seismic_compiler::feedback::PreparationOptions;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_target::TargetFamily;
use std::sync::Arc;

type CpuExecutor = seismic_cpu::Executor;
pub(crate) type CpuOpened = Opened<seismic_cpu::Cpu, CpuExecutor>;
type CpuPrepared = PreparedHandle<seismic_cpu::Cpu, CpuExecutor>;
type CpuWorkflowDraft = driver::WorkflowGraphDraft<seismic_cpu::Cpu, CpuExecutor>;
type CpuBoundWorkflow = driver::BoundWorkflowGraph<seismic_cpu::Cpu, CpuExecutor>;
type CpuAdmittedRun = crate::execution::AdmittedRun<seismic_cpu::Cpu, CpuExecutor>;

#[cfg(target_os = "macos")]
type MetalExecutor = seismic_metal::MetalExecutor;
#[cfg(target_os = "macos")]
pub(crate) type MetalOpened = Opened<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalPrepared = PreparedHandle<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalWorkflowDraft = driver::WorkflowGraphDraft<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalBoundWorkflow = driver::BoundWorkflowGraph<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalAdmittedRun = crate::execution::AdmittedRun<seismic_metal::Metal, MetalExecutor>;

type CudaExecutor = seismic_cuda::Executor;
pub(crate) type CudaOpened = Opened<seismic_cuda::Cuda, CudaExecutor>;
type CudaPrepared = PreparedHandle<seismic_cuda::Cuda, CudaExecutor>;
type CudaWorkflowDraft = driver::WorkflowGraphDraft<seismic_cuda::Cuda, CudaExecutor>;
type CudaBoundWorkflow = driver::BoundWorkflowGraph<seismic_cuda::Cuda, CudaExecutor>;
type CudaAdmittedRun = crate::execution::AdmittedRun<seismic_cuda::Cuda, CudaExecutor>;

/// A cheap, unopened physical-device descriptor. No variant contains a
/// profile, queue, worker pool, context, stream, or compiled probe.
pub(crate) enum Descriptor {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal {
        handle: seismic_metal::DeviceHandle,
    },
    Cuda {
        ordinal: u32,
        uuid: [u8; 16],
    },
}

/// One backend device as enumerated, before the catalog assigns snapshot
/// identifiers and pools.
pub(crate) struct DiscoveredDevice {
    pub(crate) selector: DeviceSelector,
    pub(crate) name: String,
    pub(crate) kind: DeviceKind,
    pub(crate) backend: seismic_lang::registry::BackendName,
    pub(crate) availability: Availability,
    pub(crate) memory: DiscoveredMemory,
    pub(crate) descriptor: Descriptor,
}

/// The backing a backend establishes from its native facts.
pub(crate) enum DiscoveredMemory {
    /// Allocations are host RAM (CPU; qualified unified-memory GPU).
    Host { max_allocation_bytes: u64 },
    /// Allocations consume a device-local pool with its own capacity.
    Dedicated {
        capacity_bytes: u64,
        basis: CapacityBasis,
        ledger: LedgerKey,
        max_allocation_bytes: u64,
    },
    Unsupported { reason: String },
}

pub(crate) struct BackendDiscovery {
    pub(crate) devices: Vec<DiscoveredDevice>,
    pub(crate) diagnostics: Vec<DiscoveryDiagnostic>,
}

/// One opened backend.  This is deliberately closed: callers cannot inject
/// an executor with a profile from a different device.
pub(crate) enum OpenedKind {
    Cpu(Arc<CpuOpened>),
    #[cfg(target_os = "macos")]
    Metal(Arc<MetalOpened>),
    Cuda(Arc<CudaOpened>),
}

#[derive(Clone)]
pub(crate) enum PreparedKind {
    Cpu(Arc<CpuPrepared>),
    #[cfg(target_os = "macos")]
    Metal(Arc<MetalPrepared>),
    Cuda(Arc<CudaPrepared>),
}

pub(crate) enum FeedbackKind<'a> {
    Cpu(
        driver::FeedbackCampaign<'a, seismic_cpu::Cpu, CpuExecutor, seismic_cpu::CpuNativeCompiler>,
    ),
    #[cfg(target_os = "macos")]
    Metal(
        driver::FeedbackCampaign<
            'a,
            seismic_metal::Metal,
            MetalExecutor,
            seismic_metal::MetalNativeCompiler,
        >,
    ),
    Cuda(
        driver::FeedbackCampaign<
            'a,
            seismic_cuda::Cuda,
            CudaExecutor,
            seismic_cuda::CudaNativeCompiler,
        >,
    ),
}

impl FeedbackKind<'_> {
    pub(crate) fn continue_for(
        &mut self,
        additional: std::time::Duration,
    ) -> Result<PreparedKind, crate::api::kernel::PrepareError> {
        match self {
            Self::Cpu(campaign) => campaign.continue_for(additional).map(PreparedKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(campaign) => campaign.continue_for(additional).map(PreparedKind::Metal),
            Self::Cuda(campaign) => campaign.continue_for(additional).map(PreparedKind::Cuda),
        }
    }
    pub(crate) fn report(&self) -> &seismic_compiler::feedback::FeedbackReport {
        match self {
            Self::Cpu(campaign) => campaign.report(),
            #[cfg(target_os = "macos")]
            Self::Metal(campaign) => campaign.report(),
            Self::Cuda(campaign) => campaign.report(),
        }
    }
}

impl PreparedKind {
    pub(crate) fn feedback_report(&self) -> Option<&seismic_compiler::feedback::FeedbackReport> {
        match self {
            Self::Cpu(kernel) => kernel.prepared.feedback_report.as_ref(),
            #[cfg(target_os = "macos")]
            Self::Metal(kernel) => kernel.prepared.feedback_report.as_ref(),
            Self::Cuda(kernel) => kernel.prepared.feedback_report.as_ref(),
        }
    }

    pub(crate) fn call(&self, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        match self {
            Self::Cpu(kernel) => kernel.call(args),
            #[cfg(target_os = "macos")]
            Self::Metal(kernel) => kernel.call(args),
            Self::Cuda(kernel) => kernel.call(args),
        }
    }
}

pub(crate) enum WorkflowDraftKind {
    Cpu(CpuWorkflowDraft),
    #[cfg(target_os = "macos")]
    Metal(MetalWorkflowDraft),
    Cuda(CudaWorkflowDraft),
}

pub(crate) enum BoundWorkflowKind {
    Cpu(CpuBoundWorkflow),
    #[cfg(target_os = "macos")]
    Metal(MetalBoundWorkflow),
    Cuda(CudaBoundWorkflow),
}

pub(crate) enum AdmittedWorkflowKind {
    Cpu(CpuAdmittedRun),
    #[cfg(target_os = "macos")]
    Metal(MetalAdmittedRun),
    Cuda(CudaAdmittedRun),
}

impl WorkflowDraftKind {
    pub(crate) fn enqueue(
        &mut self,
        kernel: &PreparedKind,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, WorkflowError> {
        match (self, kernel) {
            (Self::Cpu(workflow), PreparedKind::Cpu(kernel)) => {
                workflow.enqueue(kernel.clone(), args)
            }
            #[cfg(target_os = "macos")]
            (Self::Metal(workflow), PreparedKind::Metal(kernel)) => {
                workflow.enqueue(kernel.clone(), args)
            }
            (Self::Cuda(workflow), PreparedKind::Cuda(kernel)) => {
                workflow.enqueue(kernel.clone(), args)
            }
            _ => Err(WorkflowError::CrossWorkflowResult),
        }
    }

    pub(crate) fn bind(self) -> Result<BoundWorkflowKind, CallError> {
        match self {
            Self::Cpu(workflow) => workflow.bind().map(BoundWorkflowKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(workflow) => workflow.bind().map(BoundWorkflowKind::Metal),
            Self::Cuda(workflow) => workflow.bind().map(BoundWorkflowKind::Cuda),
        }
    }
}

impl BoundWorkflowKind {
    pub(crate) fn set_allocation_limit(&mut self, limit: u64) {
        match self {
            Self::Cpu(workflow) => workflow.set_allocation_limit(limit),
            #[cfg(target_os = "macos")]
            Self::Metal(workflow) => workflow.set_allocation_limit(limit),
            Self::Cuda(workflow) => workflow.set_allocation_limit(limit),
        }
    }
    pub(crate) fn initial_allocation_bytes(&self)->u64 {match self {
        Self::Cpu(w)=>w.initial_allocation_bytes(),
        #[cfg(target_os="macos")] Self::Metal(w)=>w.initial_allocation_bytes(),
        Self::Cuda(w)=>w.initial_allocation_bytes(),
    }}

    pub(crate) fn admit(self) -> Result<AdmittedWorkflowKind, CallError> {
        match self {
            Self::Cpu(workflow) => workflow.admit().map(AdmittedWorkflowKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(workflow) => workflow.admit().map(AdmittedWorkflowKind::Metal),
            Self::Cuda(workflow) => workflow.admit().map(AdmittedWorkflowKind::Cuda),
        }
    }
}

impl AdmittedWorkflowKind {
    pub(crate) fn submit(self) -> Result<WorkflowCompletionAny, CallError> {
        match self {
            Self::Cpu(workflow) => workflow.submit().map(|run| run.into_completion()),
            #[cfg(target_os = "macos")]
            Self::Metal(workflow) => workflow.submit().map(|run| run.into_completion()),
            Self::Cuda(workflow) => workflow.submit().map(|run| run.into_completion()),
        }
    }
}

/// Enumerates every enabled backend. Only cheap native queries run here:
/// no context, queue, stream, worker pool, probe or profile is created.
pub(crate) fn discover() -> BackendDiscovery {
    use seismic_lang::registry::BackendName;
    let mut devices = Vec::new();
    let mut diagnostics = Vec::new();

    devices.push(DiscoveredDevice {
        selector: DeviceSelector::HostCpu,
        name: format!("Host CPU ({})", std::env::consts::ARCH),
        kind: DeviceKind::Cpu,
        backend: BackendName::Cpu,
        availability: Availability::Available,
        memory: DiscoveredMemory::Host {
            max_allocation_bytes: seismic_cpu::MAX_ALLOCATION_BYTES,
        },
        descriptor: Descriptor::Cpu,
    });

    #[cfg(target_os = "macos")]
    for handle in seismic_metal::DeviceHandle::discover() {
        // Shared host backing is established only for Apple silicon, where
        // unified memory is the host RAM pool. Intel integrated carveouts
        // and discrete Metal GPUs are not qualified.
        let memory = if cfg!(target_arch = "aarch64") && handle.has_unified_memory() {
            DiscoveredMemory::Host {
                max_allocation_bytes: handle.max_allocation_bytes(),
            }
        } else {
            DiscoveredMemory::Unsupported {
                reason: "Metal memory backing is qualified only for unified memory on Apple silicon"
                    .into(),
            }
        };
        devices.push(DiscoveredDevice {
            selector: DeviceSelector::Metal {
                registry_id: handle.registry_id(),
            },
            name: handle.name(),
            kind: DeviceKind::Gpu,
            backend: BackendName::Metal,
            availability: Availability::Available,
            memory,
            descriptor: Descriptor::Metal { handle },
        });
    }

    match seismic_cuda::device_count() {
        Err(error) => diagnostics.push(DiscoveryDiagnostic {
            backend: BackendName::Cuda,
            message: error.to_string(),
        }),
        Ok(count) => {
            for ordinal in 0..count {
                let descriptor = match seismic_cuda::describe(ordinal) {
                    Ok(descriptor) => descriptor,
                    Err(error) => {
                        diagnostics.push(DiscoveryDiagnostic {
                            backend: BackendName::Cuda,
                            message: format!("device ordinal {ordinal}: {error}"),
                        });
                        continue;
                    }
                };
                // An integrated GPU (GB10) allocates from the host RAM pool,
                // like Apple silicon: it shares the host memory domain and
                // budget instead of contributing a device pool.
                let memory = if descriptor.integrated {
                    DiscoveredMemory::Host {
                        max_allocation_bytes: seismic_cuda::max_allocation_bytes(
                            descriptor.total_memory_bytes,
                        ),
                    }
                } else {
                    DiscoveredMemory::Dedicated {
                        capacity_bytes: descriptor.total_memory_bytes,
                        basis: CapacityBasis::CudaDeviceTotal,
                        ledger: LedgerKey::Cuda(descriptor.uuid),
                        max_allocation_bytes: seismic_cuda::max_allocation_bytes(
                            descriptor.total_memory_bytes,
                        ),
                    }
                };
                devices.push(DiscoveredDevice {
                    selector: DeviceSelector::Cuda {
                        uuid: descriptor.uuid,
                    },
                    name: descriptor.name,
                    kind: DeviceKind::Gpu,
                    backend: BackendName::Cuda,
                    availability: Availability::Available,
                    memory,
                    descriptor: Descriptor::Cuda {
                        ordinal,
                        uuid: descriptor.uuid,
                    },
                });
            }
        }
    }

    // `vulkan` is a registered backend name (native declarations may target
    // it), but no Vulkan runtime is built yet: a request for it is answered
    // by this diagnostic instead of an empty result.
    diagnostics.push(DiscoveryDiagnostic {
        backend: BackendName::Vulkan,
        message: "this build has no Vulkan runtime".into(),
    });

    BackendDiscovery {
        devices,
        diagnostics,
    }
}

/// Opens one catalog device, revalidating its native identity, with its
/// allocations charged to `memory`.
pub(crate) fn open(
    info: DeviceInfo,
    memory: Arc<MemoryDomain>,
    options: crate::artifacts::DeviceOptions,
) -> Result<Arc<DeviceInner>, OpenError> {
    let descriptor = info.descriptor.clone();
    let kind = match descriptor.as_ref() {
        Descriptor::Cpu => {
            let seismic_cpu::OpenedCpu {
                service,
                executor,
                device,
            } = seismic_cpu::open_host().map_err(OpenError::Backend)?;
            OpenedKind::Cpu(Arc::new(Opened::new(
                service,
                executor,
                seismic_cpu::registry(),
                device,
                |_, executor, device| executor.analytical(device),
                memory,
            )))
        }
        #[cfg(target_os = "macos")]
        Descriptor::Metal { handle } => {
            let service =
                seismic_metal::MetalDevice::open(handle.clone()).map_err(OpenError::Backend)?;
            let device =
                seismic_metal::profile::open_device(&service).map_err(OpenError::Backend)?;
            let executor = seismic_metal::MetalExecutor::new(service.clone());
            OpenedKind::Metal(Arc::new(Opened::new(
                service,
                executor,
                seismic_metal::profile::registry(),
                device,
                |service, _, device| seismic_metal::profile::open_analytical(service, device),
                memory,
            )))
        }
        Descriptor::Cuda { ordinal, uuid } => {
            // Ordinals are enumeration positions; the exposed device behind
            // one must still be the discovered device.
            let current = seismic_cuda::describe(*ordinal).map_err(OpenError::Backend)?;
            if current.uuid != *uuid {
                return Err(OpenError::IdentityChanged(info.selector));
            }
            let seismic_cuda::OpenedCuda { service, device } =
                seismic_cuda::open(*ordinal).map_err(open_error)?;
            let executor = seismic_cuda::Executor::new(service.clone());
            OpenedKind::Cuda(Arc::new(Opened::new(
                service,
                executor,
                seismic_cuda::registry(),
                device,
                |service, _, device| seismic_cuda::open_analytical(service, device),
                memory,
            )))
        }
    };
    Ok(Arc::new(DeviceInner {
        info,
        capabilities: std::sync::OnceLock::new(),
        kind,
        trace: std::sync::Mutex::new(None),
        artifacts: options.artifacts,
        native: crate::native::NativeQueue::default(),
    }))
}

fn open_error(error: ExecutionError) -> OpenError {
    OpenError::Backend(TargetError::DeviceUnavailable(error.to_string()))
}

impl OpenedKind {
    /// Samples this opened device's backend observation. A CUDA sample
    /// requires the opened context; Metal and host samples do not.
    pub(crate) fn memory_status(&self) -> Result<DeviceMemoryStatus, ObservationError> {
        let measurements = match self {
            Self::Cpu(_) => DeviceMeasurements::Host,
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                let handle = device.service().handle();
                DeviceMeasurements::Metal {
                    recommended_working_set_bytes: handle.recommended_working_set_bytes(),
                    current_allocated_bytes: handle.current_allocated_bytes(),
                }
            }
            Self::Cuda(device) => {
                let info = device
                    .service()
                    .memory_info()
                    .map_err(|error| ObservationError::Failed(error.to_string()))?;
                DeviceMeasurements::Cuda {
                    free_bytes: info.free_bytes,
                    total_bytes: info.total_bytes,
                }
            }
        };
        Ok(DeviceMemoryStatus {
            sampled_at: std::time::SystemTime::now(),
            measurements,
        })
    }

    pub(crate) fn workflow(&self) -> WorkflowDraftKind {
        match self {
            Self::Cpu(device) => {
                WorkflowDraftKind::Cpu(driver::WorkflowGraphDraft::new(device.clone()))
            }
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                WorkflowDraftKind::Metal(driver::WorkflowGraphDraft::new(device.clone()))
            }
            Self::Cuda(device) => {
                WorkflowDraftKind::Cuda(driver::WorkflowGraphDraft::new(device.clone()))
            }
        }
    }
    pub(crate) fn capabilities(&self) -> Vec<String> {
        match self {
            Self::Cpu(device) => driver::opened_capability_summaries(device),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => driver::opened_capability_summaries(device),
            Self::Cuda(device) => driver::opened_capability_summaries(device),
        }
    }

    pub(crate) fn memory_usage(&self) -> crate::memory::MemoryUsage {
        match self {
            Self::Cpu(device) => device.memory_usage(),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.memory_usage(),
            Self::Cuda(device) => device.memory_usage(),
        }
    }

    pub(crate) fn set_memory_limit(
        &self,
        limit: Option<u64>,
    ) -> Result<(), crate::memory::MemoryLimitError> {
        match self {
            Self::Cpu(device) => device.set_memory_limit(limit),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.set_memory_limit(limit),
            Self::Cuda(device) => device.set_memory_limit(limit),
        }
    }

    pub(crate) fn identity(&self) -> seismic_compiler::prepared::DeviceIdentity {
        match self {
            Self::Cpu(device) => device.identity(),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.identity(),
            Self::Cuda(device) => device.identity(),
        }
    }

    /// Facts that distinguish performance behavior, for keying per-device
    /// native tuning, including the toolchain and driver versions. Equal
    /// identities denote the same device model and configuration; there is
    /// no ordering or closeness.
    pub(crate) fn tuning_identity(&self) -> String {
        match self {
            Self::Cpu(device) => {
                let facts = device.device_description().facts();
                format!(
                    "cpu;{};workers {};simd {:?}",
                    std::env::consts::ARCH,
                    facts.workers,
                    facts.simd
                )
            }
            // The OS build ships the Metal compiler, and the NVRTC release
            // and driver form and run CUDA code: an update can change which
            // configuration wins.
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                let facts = device.device_description().facts();
                format!("{};os {}", facts.tuning_material(), facts.operating_system())
            }
            Self::Cuda(device) => {
                let facts = device.device_description().facts();
                let nvrtc = seismic_cuda::nvrtc::release().map_or_else(
                    |error| format!("unavailable ({error})"),
                    |(major, minor)| format!("{major}.{minor}"),
                );
                format!(
                    "cuda;sm {};multiprocessors {};driver {};nvrtc {nvrtc}",
                    facts.compute_capability, facts.multiprocessors, facts.driver_api
                )
            }
        }
    }

    pub(crate) fn supports_representation(&self, representation: RepresentationId) -> bool {
        match self {
            Self::Cpu(device) => device
                .device_description()
                .dtypes()
                .representations
                .contains(&representation),
            #[cfg(target_os = "macos")]
            // Direct Metal owns raw shared buffers; representation-specific
            // interpretation remains in the authored kernel. A normal
            // compiler preparation profiles and validates its narrower
            // representation support before planning.
            Self::Metal(_) => {
                let _ = seismic_lang::registry::representation_info(representation);
                true
            }
            // Row layouts (`mma16`, the CUDA resident layout) are storage for
            // direct native kernels only; compiled construction refuses them
            // through the profile's representation set.
            Self::Cuda(device) => {
                seismic_lang::registry::representation_info(representation).layout
                    != seismic_lang::registry::Layout::Packet
                    || device
                        .device_description()
                        .dtypes()
                        .representations
                        .contains(&representation)
            }
        }
    }

    pub(crate) fn allocate(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_storage(bytes, alignment),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_storage(bytes, alignment),
            Self::Cuda(device) => device.allocate_storage(bytes, alignment),
        }
    }

    /// Storage for inputs the host writes before each submission. Host
    /// writes to it are plain memory writes that never wait for queued
    /// device work: CPU and Metal storage is host-visible already, CUDA uses
    /// mapped pinned host memory.
    pub(crate) fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_storage(bytes, alignment),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_storage(bytes, alignment),
            Self::Cuda(device) => {
                device.allocate_storage_with(bytes, alignment, |service| {
                    service.allocate_mapped(bytes)
                })
            }
        }
    }

    pub(crate) fn prepare(
        &self,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        public_device: &Arc<DeviceInner>,
        options: PreparationOptions,
    ) -> Result<PreparedKind, crate::api::kernel::PrepareError> {
        match self {
            Self::Cpu(device) => prepare(
                device,
                seismic_cpu::native_compiler(),
                &(),
                module,
                entry,
                bindings,
                public_device,
                options,
            )
            .map(PreparedKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                let context = device.service_arc().handle().clone();
                prepare(
                    device,
                    seismic_metal::native_compiler(),
                    &context,
                    module,
                    entry,
                    bindings,
                    public_device,
                    options,
                )
                .map(PreparedKind::Metal)
            }
            Self::Cuda(device) => prepare(
                device,
                seismic_cuda::native_compiler(),
                &(),
                module,
                entry,
                bindings,
                public_device,
                options,
            )
            .map(PreparedKind::Cuda),
        }
    }

    pub(crate) fn start_feedback<'a>(
        &'a self,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        public_device: &Arc<DeviceInner>,
        precision: seismic_lang::precision::PrecisionPolicy,
        options: seismic_compiler::feedback::FeedbackOptions,
    ) -> Result<(FeedbackKind<'a>, PreparedKind), crate::api::kernel::PrepareError> {
        match self {
            Self::Cpu(device) => driver::FeedbackCampaign::start(
                device,
                seismic_cpu::native_compiler(),
                &(),
                module,
                entry,
                bindings,
                public_device,
                precision,
                options,
            )
            .map(|(campaign, kernel)| (FeedbackKind::Cpu(campaign), PreparedKind::Cpu(kernel))),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => driver::FeedbackCampaign::start(
                device,
                seismic_metal::native_compiler(),
                device.service().handle(),
                module,
                entry,
                bindings,
                public_device,
                precision,
                options,
            )
            .map(|(campaign, kernel)| (FeedbackKind::Metal(campaign), PreparedKind::Metal(kernel))),
            Self::Cuda(device) => driver::FeedbackCampaign::start(
                device,
                seismic_cuda::native_compiler(),
                &(),
                module,
                entry,
                bindings,
                public_device,
                precision,
                options,
            )
            .map(|(campaign, kernel)| (FeedbackKind::Cuda(campaign), PreparedKind::Cuda(kernel))),
        }
    }
}

fn prepare<T, E, C>(
    device: &Arc<Opened<T, E>>,
    compiler: &C,
    native_context: &C::Context,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    public_device: &Arc<DeviceInner>,
    options: PreparationOptions,
) -> Result<Arc<PreparedHandle<T, E>>, crate::api::kernel::PrepareError>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_target::NativeCompiler<T, Handle = E::Handle>,
{
    let prepared = driver::prepare(
        device,
        compiler,
        native_context,
        module,
        entry,
        bindings,
        public_device,
        options,
    )?;
    Ok(Arc::new(PreparedHandle {
        prepared,
        device: public_device.clone(),
    }))
}
