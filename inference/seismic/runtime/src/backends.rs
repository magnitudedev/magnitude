//! The runtime's one closed sum over supported backends.
//!
//! Backend crates remain generic and do not depend on the runtime.  This
//! module is the sole composition root which turns their target profiles,
//! device services, executors, and prepared kernels into the public device
//! API.  It does not reproduce target facts or planning policy.

use crate::api::{
    CallError, WorkflowError,
    device::DeviceInner,
    kernel::{
        DecodedResults, EncodedArgs, EncodedWorkflowArgs, PendingWorkflowResults,
        WorkflowCompletionAny,
    },
};
use crate::devices::OpenError;
use crate::driver::{self, Opened, PreparedHandle};
use crate::formation::ProgramFormer;
use crate::memory::ExecutorMemory;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::{DeviceService, NativeExecutor};
use seismic_compiler::feedback::PreparationOptions;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{ElementBindings, LogicalEntry};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_native_target::TargetFamily;
use seismic_topology::{DeviceTopology, Executor, ExecutorSelector, ObservationError, StorageKind};
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

#[cfg(not(target_os = "macos"))]
mod vulkan;
#[cfg(not(target_os = "macos"))]
pub(crate) use vulkan::{VulkanOpened, vulkan_buffer};

type CudaExecutor = seismic_cuda::Executor;
pub(crate) type CudaOpened = Opened<seismic_cuda::Cuda, CudaExecutor>;
type CudaPrepared = PreparedHandle<seismic_cuda::Cuda, CudaExecutor>;
type CudaWorkflowDraft = driver::WorkflowGraphDraft<seismic_cuda::Cuda, CudaExecutor>;
type CudaBoundWorkflow = driver::BoundWorkflowGraph<seismic_cuda::Cuda, CudaExecutor>;
type CudaAdmittedRun = crate::execution::AdmittedRun<seismic_cuda::Cuda, CudaExecutor>;

/// One opened backend.  This is deliberately closed: callers cannot inject
/// an executor with a profile from a different device.
pub(crate) enum OpenedKind {
    Cpu(Arc<CpuOpened>),
    #[cfg(target_os = "macos")]
    Metal(Arc<MetalOpened>),
    Cuda(Arc<CudaOpened>),
    /// Native-only: no planned route (Vulkan backend spec §9).
    #[cfg(not(target_os = "macos"))]
    Vulkan(Arc<VulkanOpened>),
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
    pub(crate) fn initial_allocation_bytes(&self) -> u64 {
        match self {
            Self::Cpu(w) => w.initial_allocation_bytes(),
            #[cfg(target_os = "macos")]
            Self::Metal(w) => w.initial_allocation_bytes(),
            Self::Cuda(w) => w.initial_allocation_bytes(),
        }
    }

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

/// Opens one executor of `topology`, revalidating its native identity, with
/// its allocations charged to `memory`.
pub(crate) fn open(
    info: Executor,
    topology: Arc<DeviceTopology>,
    memory: Arc<ExecutorMemory>,
    options: crate::artifacts::DeviceOptions,
) -> Result<Arc<DeviceInner>, OpenError> {
    let selector = info.selector();
    let mut programs = DevicePrograms::Cpu;
    let kind = match selector {
        ExecutorSelector::HostCpu => {
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
                memory.clone(),
            )))
        }
        #[cfg(target_os = "macos")]
        ExecutorSelector::Metal { registry_id } => {
            let handle = seismic_metal::DeviceHandle::discover()
                .into_iter()
                .find(|handle| handle.registry_id() == registry_id)
                .ok_or(OpenError::IdentityChanged(selector))?;
            let service =
                seismic_metal::MetalDevice::open(handle.clone()).map_err(OpenError::Backend)?;
            let device =
                seismic_metal::profile::open_device(&service).map_err(OpenError::Backend)?;
            let executor = seismic_metal::MetalExecutor::new(service.clone());
            programs = DevicePrograms::Metal(ProgramFormer::new(
                seismic_metal::toolchain::MetalToolchain::new(service.clone(), device.facts()),
                options.artifacts.clone(),
            ));
            OpenedKind::Metal(Arc::new(Opened::new(
                service,
                executor,
                seismic_metal::profile::registry(),
                device,
                |service, _, device| seismic_metal::profile::open_analytical(service, device),
                memory.clone(),
            )))
        }
        ExecutorSelector::Cuda { uuid } => {
            let mut ordinal = None;
            for index in 0..seismic_cuda::device_count().map_err(OpenError::Backend)? {
                if seismic_cuda::describe(index).map_err(OpenError::Backend)?.uuid == uuid {
                    ordinal = Some(index);
                    break;
                }
            }
            let ordinal = ordinal.ok_or(OpenError::IdentityChanged(selector))?;
            let seismic_cuda::OpenedCuda { service, device } =
                seismic_cuda::open(ordinal).map_err(open_error)?;
            let account = |domain| Arc::new(crate::memory::MemoryAccount(domain));
            service.set_host_memory_account(account(memory.staging.clone()));
            service.set_allocation_accounts(
                account(memory.device.physical()),
                account(memory.upload.physical()),
            );
            let executor = seismic_cuda::Executor::new(service.clone());
            let capability = device.facts().compute_capability;
            programs = DevicePrograms::Cuda(ProgramFormer::new(
                seismic_cuda::toolchain::CudaToolchain::new(
                    service.clone(),
                    u32::from(capability.major) * 10 + u32::from(capability.minor),
                ),
                options.artifacts.clone(),
            ));
            OpenedKind::Cuda(Arc::new(Opened::new(
                service,
                executor,
                seismic_cuda::registry(),
                device,
                |service, _, device| seismic_cuda::open_analytical(service, device),
                memory.clone(),
            )))
        }
        #[cfg(not(target_os = "macos"))]
        ExecutorSelector::Vulkan { uuid } => {
            let opened = VulkanOpened::open(uuid, &info, &topology, memory.clone())?;
            programs = DevicePrograms::Vulkan(ProgramFormer::new(
                seismic_vulkan::toolchain::VulkanToolchain::new(opened.service().clone()),
                options.artifacts.clone(),
            ));
            OpenedKind::Vulkan(Arc::new(opened))
        }
        // `info` is an executor of this process's topology, whose discovery
        // reports no executor of a backend the build lacks.
        #[cfg(not(target_os = "macos"))]
        ExecutorSelector::Metal { .. } => unreachable!("this build discovers no Metal executor"),
        #[cfg(target_os = "macos")]
        ExecutorSelector::Vulkan { .. } => unreachable!("this build discovers no Vulkan executor"),
    };
    Ok(Arc::new(DeviceInner {
        info,
        topology,
        memory,
        capabilities: std::sync::OnceLock::new(),
        kind,
        programs,
        trace: std::sync::Mutex::new(None),
        artifacts: options.artifacts,
        native: crate::native::NativeQueue::default(),
        chosen: std::sync::Mutex::new(std::collections::HashMap::new()),
    }))
}

/// The program former of an opened device, by backend. The CPU backend
/// renders no programs.
pub(crate) enum DevicePrograms {
    Cpu,
    #[cfg(target_os = "macos")]
    Metal(ProgramFormer<seismic_metal::toolchain::MetalToolchain>),
    Cuda(ProgramFormer<seismic_cuda::toolchain::CudaToolchain>),
    #[cfg(not(target_os = "macos"))]
    Vulkan(ProgramFormer<seismic_vulkan::toolchain::VulkanToolchain>),
}

impl DevicePrograms {
    /// Lend the device's former `workers`, or take them back
    /// ([`ProgramFormer::staff`]).
    pub(crate) fn staff(
        &self,
        workers: Option<Arc<seismic_native_target::workers::Workers>>,
    ) {
        match self {
            Self::Cpu => {}
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.staff(workers),
            Self::Cuda(former) => former.staff(workers),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.staff(workers),
        }
    }

    /// The threads lent to the device's former, if any.
    pub(crate) fn workers(&self) -> Option<Arc<seismic_native_target::workers::Workers>> {
        match self {
            Self::Cpu => None,
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.workers(),
            Self::Cuda(former) => former.workers(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.workers(),
        }
    }

    /// How many native programs this device's toolchain has formed.
    pub(crate) fn formed(&self) -> usize {
        match self {
            Self::Cpu => 0,
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.formed(),
            Self::Cuda(former) => former.formed(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.formed(),
        }
    }

    /// What compiling `source` takes this device's toolchain when nothing
    /// serves it ([`ProgramFormer::cold`]); nothing on the CPU backend,
    /// which compiles nothing at run time.
    pub(crate) fn cold(
        &self,
        source: &seismic_native_target::ProgramSource,
    ) -> Result<std::time::Duration, seismic_native_target::NativeCompilationError> {
        match self {
            Self::Cpu => Ok(std::time::Duration::ZERO),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.cold(source),
            Self::Cuda(former) => former.cold(source),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.cold(source),
        }
    }

    /// `sources` as this device's former reads them
    /// ([`ProgramFormer::keyed`]). The CPU backend forms no programs.
    pub(crate) fn keyed(
        &self,
        sources: Vec<seismic_native_target::ProgramSource>,
    ) -> Vec<crate::formation::Keyed> {
        let keyed = |key: &dyn Fn(seismic_native_target::ProgramSource) -> crate::formation::Keyed| {
            sources.into_iter().map(key).collect()
        };
        match self {
            Self::Cpu => Vec::new(),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => keyed(&|source| former.keyed(source)),
            Self::Cuda(former) => keyed(&|source| former.keyed(source)),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => keyed(&|source| former.keyed(source)),
        }
    }

    /// What forming `sources` together on this device does, known before
    /// they are formed, and the most entries one source's compile forms.
    pub(crate) fn class(
        &self,
        sources: &[crate::formation::Keyed],
    ) -> (crate::formation::FormationClass, usize) {
        match self {
            Self::Cpu => (crate::formation::FormationClass::Kept, 0),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.class(sources),
            Self::Cuda(former) => former.class(sources),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.class(sources),
        }
    }

    /// What publishing `sources` on this device does, known before it runs
    /// ([`ProgramFormer::publication`]).
    pub(crate) fn publication(
        &self,
        sources: &[crate::formation::Keyed],
    ) -> (crate::formation::FormationClass, usize) {
        match self {
            Self::Cpu => (crate::formation::FormationClass::Kept, 0),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.publication(sources),
            Self::Cuda(former) => former.publication(sources),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.publication(sources),
        }
    }

    /// How many of `sources` publishing them on this device compiles
    /// ([`ProgramFormer::publishing_sources`]).
    pub(crate) fn publishing_sources(&self, sources: &[crate::formation::Keyed]) -> usize {
        match self {
            Self::Cpu => 0,
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.publishing_sources(sources),
            Self::Cuda(former) => former.publishing_sources(sources),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.publishing_sources(sources),
        }
    }

    /// Publish `sources` as a later process will request them
    /// ([`ProgramFormer::publish`]).
    pub(crate) fn publish(
        &self,
        sources: &[seismic_native_target::ProgramSource],
    ) -> Result<(), seismic_native_target::NativeCompilationError> {
        match self {
            Self::Cpu => Ok(()),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.publish(sources),
            Self::Cuda(former) => former.publish(sources),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.publish(sources),
        }
    }

    /// What this device's toolchain's compiles have taken in this process,
    /// by the entries each formed.
    pub(crate) fn compiled(&self) -> crate::native::Linear {
        match self {
            Self::Cpu => crate::native::Linear::default(),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.compiled(),
            Self::Cuda(former) => former.compiled(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.compiled(),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal(&self) -> &ProgramFormer<seismic_metal::toolchain::MetalToolchain> {
        let Self::Metal(former) = self else {
            unreachable!("a Metal device forms Metal programs");
        };
        former
    }

    pub(crate) fn cuda(&self) -> &ProgramFormer<seismic_cuda::toolchain::CudaToolchain> {
        let Self::Cuda(former) = self else {
            unreachable!("a CUDA device forms CUDA programs");
        };
        former
    }

    #[cfg(not(target_os = "macos"))]
    pub(crate) fn vulkan(&self) -> &ProgramFormer<seismic_vulkan::toolchain::VulkanToolchain> {
        let Self::Vulkan(former) = self else {
            unreachable!("a Vulkan device forms Vulkan programs");
        };
        former
    }
}

fn open_error(error: ExecutionError) -> OpenError {
    OpenError::Backend(TargetError::DeviceUnavailable(error.to_string()))
}

impl OpenedKind {
    /// A CUDA executor's free memory (`cuMemGetInfo`); `None` on every other
    /// backend.
    pub(crate) fn cuda_free(&self) -> Option<Result<u64, ObservationError>> {
        match self {
            Self::Cuda(device) => Some(
                device
                    .service()
                    .memory_info()
                    .map(|info| info.free_bytes)
                    .map_err(|error| ObservationError::Failed(error.to_string())),
            ),
            _ => None,
        }
    }

    pub(crate) fn workflow(&self) -> Result<WorkflowDraftKind, WorkflowError> {
        Ok(match self {
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
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => {
                return Err(WorkflowError::PlannedRouteUnavailable {
                    backend: seismic_lang::registry::BackendName::Vulkan,
                });
            }
        })
    }
    /// Portable capability namespaces; a native-only backend has none.
    pub(crate) fn capabilities(&self) -> Vec<String> {
        match self {
            Self::Cpu(device) => driver::opened_capability_summaries(device),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => driver::opened_capability_summaries(device),
            Self::Cuda(device) => driver::opened_capability_summaries(device),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => Vec::new(),
        }
    }

    /// The usage of `kind` storage: its domain's charges, with what the
    /// backend holds for it beyond them.
    pub(crate) fn usage(&self, kind: StorageKind) -> crate::memory::MemoryUsage {
        match (self, kind) {
            (Self::Cpu(device), StorageKind::Device) => device.memory_usage(),
            #[cfg(target_os = "macos")]
            (Self::Metal(device), StorageKind::Device) => device.memory_usage(),
            (Self::Cuda(device), StorageKind::Device) => device.memory_usage(),
            (Self::Cpu(device), StorageKind::Upload) => device.upload_usage(),
            #[cfg(target_os = "macos")]
            (Self::Metal(device), StorageKind::Upload) => device.upload_usage(),
            (Self::Cuda(device), StorageKind::Upload) => device.upload_usage(),
            #[cfg(not(target_os = "macos"))]
            (Self::Vulkan(device), kind) => device.usage(match kind {
                StorageKind::Device => seismic_vulkan::PlacedStorage::Device,
                StorageKind::Upload => seismic_vulkan::PlacedStorage::Upload,
            }),
        }
    }

    /// Hold `kind` storage's domain to `limit`. Charges already made count
    /// against it; they are never charged twice.
    pub(crate) fn set_limit(&self, kind: StorageKind, limit: Option<u64>) {
        match self {
            Self::Cpu(device) => device.set_limit(kind, limit),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.set_limit(kind, limit),
            Self::Cuda(device) => device.set_limit(kind, limit),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.set_limit(kind, limit),
        }
    }

    pub(crate) fn restart_memory_high_water(&self) {
        match self {
            Self::Cpu(device) => device.restart_memory_high_water(),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.restart_memory_high_water(),
            Self::Cuda(device) => device.restart_memory_high_water(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.restart_memory_high_water(),
        }
    }

    pub(crate) fn identity(&self) -> seismic_compiler::prepared::DeviceIdentity {
        match self {
            Self::Cpu(device) => device.identity(),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.identity(),
            Self::Cuda(device) => device.identity(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.identity(),
        }
    }

    /// Facts that distinguish performance behavior, for keying per-device
    /// native tuning, including the toolchain and driver versions. Equal
    /// identities denote the same device model and configuration; there is
    /// no ordering or closeness.
    pub(crate) fn tuning_identity(&self) -> String {
        match self {
            // The detected tier fixes which native forms exist, and the CPU
            // library version their machine code.
            Self::Cpu(device) => {
                let facts = device.device_description().facts();
                format!(
                    "cpu;{};workers {};simd {:?};tier {};library {}",
                    std::env::consts::ARCH,
                    facts.workers,
                    facts.simd,
                    seismic_native_cpu::Tier::detected().map_or("none", |tier| tier.name()),
                    seismic_native_cpu::VERSION
                )
            }
            // The OS build ships the Metal compiler, and the NVRTC release
            // and driver form and run CUDA code: an update can change which
            // configuration wins.
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                let facts = device.device_description().facts();
                format!(
                    "{};os {}",
                    facts.tuning_material(),
                    facts.operating_system()
                )
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
            // A driver update changes the key (§7.5).
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.service().facts().tuning_identity(),
        }
    }

    pub(crate) fn supports_representation(&self, representation: RepresentationId) -> bool {
        match self {
            // Row layouts (`rows16`, the CPU weight layout) are storage for
            // direct native kernels only, whose weight components read them;
            // compiled construction refuses them through the profile's
            // representation set.
            Self::Cpu(device) => {
                seismic_lang::registry::representation_info(representation).layout
                    != seismic_lang::registry::Layout::Packet
                    || device
                        .device_description()
                        .dtypes()
                        .representations
                        .contains(&representation)
            }
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
            // Direct native kernels own raw storage, as on Metal.
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => {
                let _ = seismic_lang::registry::representation_info(representation);
                true
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
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate(bytes, alignment),
        }
    }

    /// Storage for inputs the host writes before each submission. Host
    /// writes to it are plain memory writes that never wait for queued
    /// device work: CPU and Metal storage is host-visible already, CUDA uses
    /// mapped pinned host memory, Vulkan persistently mapped host-visible
    /// memory (device-local with ReBAR or unified memory).
    pub(crate) fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_upload_with(bytes, alignment, |service| {
                service.allocate(bytes, alignment)
            }),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_upload_with(bytes, alignment, |service| {
                service.allocate(bytes, alignment)
            }),
            Self::Cuda(device) => device
                .allocate_upload_with(bytes, alignment, |service| service.allocate_mapped(bytes)),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_upload(bytes, alignment),
        }
    }

    pub(crate) fn map_read_only_host_region(
        &self,
        region: crate::api::HostRegion,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                let bytes = region.len() as u64;
                device.allocate_read_only_storage_with(bytes, alignment, |service| unsafe {
                    service.wrap_read_only_host_mapping(
                        region.pointer(),
                        region.len(),
                        region.owner(),
                    )
                })
            }
            _ => Err(ExecutionError::AllocationFailed(
                "read-only host mapping is unavailable on this backend".into(),
            )),
        }
    }

    /// Zero-filled storage of a reserved tensor: `committed` bytes backed of
    /// a `reserved`-byte logical length. CUDA with virtual memory management
    /// reserves the whole address range, so [`OpenedKind::recommit_in_place`]
    /// keeps its address; elsewhere only the committed bytes are allocated
    /// and recommitting reallocates.
    pub(crate) fn allocate_reserved_tensor(
        &self,
        committed: u64,
        reserved: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        if let Self::Cuda(device) = self {
            if device.service().supports_reservation() {
                return device.allocate_storage_with(committed, alignment, |service| {
                    service.allocate_reserved(committed, reserved)
                });
            }
        }
        let allocation = self.allocate(committed, alignment)?;
        driver::write_zeros(allocation.storage(), committed)?;
        Ok(allocation)
    }

    /// Whether `allocation` is a reserved address range the backend resizes
    /// in place (CUDA virtual memory management).
    pub(crate) fn reserves_address(&self, allocation: &Arc<driver::Allocation>) -> bool {
        let Self::Cuda(_) = self else {
            return false;
        };
        driver::typed_buffer::<seismic_cuda::Cuda, seismic_cuda::Executor>(allocation).is_reserved()
    }

    /// Device address of an ordinary allocation's first byte. Only Seismic
    /// uses this to build slab tables; lifetime still follows the allocation.
    pub(crate) fn allocation_address(&self, allocation: &Arc<driver::Allocation>) -> u64 {
        match self {
            Self::Cpu(_) => driver::typed_buffer::<seismic_cpu::Cpu, CpuExecutor>(allocation)
                .data_pointer() as u64,
            #[cfg(target_os = "macos")]
            Self::Metal(_) => {
                driver::typed_buffer::<seismic_metal::Metal, MetalExecutor>(allocation)
                    .gpu_address()
            }
            Self::Cuda(_) => {
                driver::typed_buffer::<seismic_cuda::Cuda, CudaExecutor>(allocation).pointer()
            }
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => vulkan_buffer(allocation).address(),
        }
    }

    /// The storage of `allocation`'s reserved tensor with `committed` bytes
    /// backed, resized in place when the backend reserved its address range
    /// (the new allocation shares the address and the kept bytes; bytes past
    /// the old commitment are zero). It first waits for every device use of
    /// `allocation`, so a shrink never releases bytes in use. `None` when
    /// the storage has to be reallocated.
    pub(crate) fn recommit_in_place(
        &self,
        allocation: &Arc<driver::Allocation>,
        committed: u64,
        alignment: u64,
        exclusive_view: bool,
    ) -> Result<Option<Arc<driver::Allocation>>, ExecutionError> {
        let Self::Cuda(device) = self else {
            return Ok(None);
        };
        let buffer = driver::typed_buffer::<seismic_cuda::Cuda, seismic_cuda::Executor>(allocation);
        if !buffer.is_reserved()
            || !exclusive_view
            || Arc::strong_count(allocation) != 1
            || allocation.has_live_predecessor()
        {
            return Ok(None);
        }
        let _exclusive = allocation.acquire(true);
        device
            .recommit_storage_with(
                allocation,
                committed,
                alignment,
                |service| service.recommit(buffer, committed),
                |service, current, old_bytes| service.recommit(current, old_bytes).map(|_| ()),
            )
            .map(Some)
    }

    pub(crate) fn prepare(
        &self,
        logical: LogicalEntry,
        bindings: ElementBindings,
        public_device: &Arc<DeviceInner>,
        options: PreparationOptions,
    ) -> Result<PreparedKind, crate::api::kernel::PrepareError> {
        match self {
            Self::Cpu(device) => prepare(
                device,
                seismic_cpu::native_compiler(),
                &(),
                logical,
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
                    logical,
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
                logical,
                bindings,
                public_device,
                options,
            )
            .map(PreparedKind::Cuda),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => Err(planned_route_unavailable()),
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
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => Err(planned_route_unavailable()),
        }
    }
}

/// The typed preparation-time refusal of planned operations on a
/// native-only backend (§9.2).
#[cfg(not(target_os = "macos"))]
fn planned_route_unavailable() -> crate::api::kernel::PrepareError {
    crate::api::kernel::PrepareError::Preparation(
        seismic_compiler::errors::PreparationError::PlannedRouteUnavailable {
            backend: seismic_lang::registry::BackendName::Vulkan,
        },
    )
}

fn prepare<T, E, C>(
    device: &Arc<Opened<T, E>>,
    compiler: &C,
    native_context: &C::Context,
    logical: LogicalEntry,
    bindings: ElementBindings,
    public_device: &Arc<DeviceInner>,
    options: PreparationOptions,
) -> Result<Arc<PreparedHandle<T, E>>, crate::api::kernel::PrepareError>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_native_target::NativeCompiler<T, Handle = E::Handle>,
{
    let prepared = driver::prepare(
        device,
        compiler,
        native_context,
        logical,
        bindings,
        public_device,
        options,
    )?;
    Ok(Arc::new(PreparedHandle {
        prepared,
        device: public_device.clone(),
    }))
}
