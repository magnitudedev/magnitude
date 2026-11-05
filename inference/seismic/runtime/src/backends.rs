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
use crate::memory::DeviceLedgers;
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::{DeviceService, NativeExecutor};
use seismic_compiler::feedback::PreparationOptions;
use seismic_compiler::graph_cost::GraphCost;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{ElementBindings, LogicalEntry};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_reservation as host;
use seismic_native_target::{TargetFamily, Toolchain};
use seismic_discovery::ObservationError;
use seismic_topology::{Contract, DeviceTopology, Executor, ExecutorSelector, Placing};
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
pub(crate) mod vulkan;
#[cfg(not(target_os = "macos"))]
pub(crate) use vulkan::{VulkanOpened, vulkan_address, vulkan_buffer, vulkan_transfer_end};

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

/// What a device of `backend` is charged per recorded graph before its own
/// cost is measured: its backend's bound (`seismic_cuda::UNMEASURED_GRAPH_COST`,
/// `seismic_vulkan::UNMEASURED_GRAPH_COST`), none where recording holds no
/// device memory. Opens nothing.
pub fn unmeasured_graph_cost(backend: seismic_lang::registry::BackendName) -> GraphCost {
    use seismic_lang::registry::BackendName;
    match backend {
        BackendName::Cuda => seismic_cuda::UNMEASURED_GRAPH_COST,
        #[cfg(not(target_os = "macos"))]
        BackendName::Vulkan => seismic_vulkan::UNMEASURED_GRAPH_COST,
        #[cfg(target_os = "macos")]
        BackendName::Vulkan => unreachable!("no Vulkan executor is discovered on macOS"),
        BackendName::Cpu | BackendName::Metal => GraphCost::None,
    }
}

/// Opens one executor of `topology`, revalidating its native identity, with
/// its allocations charged to `memory`.
pub(crate) fn open(
    info: Executor,
    topology: Arc<DeviceTopology>,
    memory: Arc<DeviceLedgers>,
    options: crate::artifacts::DeviceOptions,
) -> Result<Arc<DeviceInner>, OpenError> {
    let selector = info.selector();
    let mut programs = DevicePrograms::Cpu;
    // The CPU records no graphs, and a Metal schedule is host data: only
    // CUDA's and Vulkan's graphs hold device memory, measured on opening
    // unless the opener has it.
    let mut graph_cost = GraphCost::None;
    let given = options.graph_cost;
    let kind = match selector {
        ExecutorSelector::HostCpu => {
            let seismic_cpu::OpenedCpu {
                service,
                executor,
                device,
            } = seismic_cpu::open_host(host_cores().map_err(|reason| OpenError::Unavailable { selector, reason })?)
                .map_err(OpenError::Backend)?;
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
            let account = |placement| -> Arc<dyn seismic_compiler::executable::MemoryAccount> {
                Arc::new(crate::memory::MemoryAccount(placement))
            };
            let accounts = seismic_cuda::MemoryAccounts {
                device: account(memory.device().physical()),
                upload: account(memory.upload().physical()),
                host: account(memory.staging.clone()),
            };
            let seismic_cuda::OpenedCuda { service, device } =
                seismic_cuda::open(ordinal, accounts).map_err(open_error)?;
            graph_cost = given.map_or_else(|| seismic_cuda::graph_cost(&service).map_err(open_error), Ok)?;
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
            // What the measurement job stored of its probes, keyed by the
            // executor's identity.
            let identity = topology.executor_identity(info.id()).expect("the executor is of its topology");
            let probed = crate::probes::stored(options.artifacts.as_deref(), &identity, seismic_vulkan::probe::REVISION)
                .map_err(|reason| OpenError::Unavailable { selector, reason })?;
            let opened = VulkanOpened::open(uuid, &info, memory.clone(), &probed)?;
            graph_cost = given.map_or_else(|| seismic_vulkan::graph_cost(opened.service()).map_err(open_error), Ok)?;
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
    let properties = match &kind {
        OpenedKind::Cpu(opened) => seismic_lang::device::DeviceProperties::Cpu(*opened.executor().properties()),
        #[cfg(target_os = "macos")]
        OpenedKind::Metal(opened) => {
            seismic_lang::device::DeviceProperties::Metal(*opened.device_description().facts().properties())
        }
        OpenedKind::Cuda(opened) => {
            seismic_lang::device::DeviceProperties::Cuda(opened.device_description().facts().properties)
        }
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(opened) => seismic_lang::device::DeviceProperties::Vulkan(*opened.service().properties()),
    };
    Ok(Arc::new(DeviceInner {
        info,
        topology,
        memory,
        capabilities: std::sync::OnceLock::new(),
        kind,
        properties,
        programs,
        trace: std::sync::Mutex::new(None),
        artifacts: options.artifacts,
        native: crate::native::NativeState::default(),
        graph_cost,
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

    /// What besides the source determines the programs this device's
    /// toolchain forms ([`seismic_native_target::ToolchainIdentity`]): its
    /// compiler, target and options. The CPU backend forms none: its
    /// kernels are compiled into the binary.
    pub(crate) fn toolchain(&self) -> Option<&str> {
        match self {
            Self::Cpu => None,
            #[cfg(target_os = "macos")]
            Self::Metal(former) => Some(&former.toolchain().identity().material),
            Self::Cuda(former) => Some(&former.toolchain().identity().material),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => Some(&former.toolchain().identity().material),
        }
    }

    /// The native programs this device's toolchain has formed, and the
    /// time its calls took.
    pub(crate) fn programs_formed(&self) -> crate::formation::FormedPrograms {
        match self {
            Self::Cpu => crate::formation::FormedPrograms::default(),
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.programs_formed(),
            Self::Cuda(former) => former.programs_formed(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.programs_formed(),
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

    /// Whether this device's toolchain loads a stored program through a
    /// platform cache ([`seismic_native_target::Load::PlatformCached`]):
    /// a load the platform no longer serves takes a compile's time.
    pub(crate) fn platform_cached(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(target_os = "macos")]
            Self::Metal(former) => former.platform_cached(),
            Self::Cuda(former) => former.platform_cached(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(former) => former.platform_cached(),
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

/// The performance cores the process may run on, as the host's processors
/// report them: the CPU device's pool (`seismic_cpu::open_host`).
pub(crate) fn host_cores() -> Result<std::num::NonZeroUsize, String> {
    let processors = seismic_discovery::Processors::discover().map_err(|error| format!("the host's processors: {error}"))?;
    std::num::NonZeroUsize::new(processors.performance_cores()).ok_or_else(|| "the process may run on no performance core".to_owned())
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

    /// What the backend holds for fixed buffers of `contract` beyond their
    /// allocations.
    pub(crate) fn backend_held(&self, kind: Contract) -> crate::devices::BackendHeld {
        match self {
            Self::Cpu(device) => device.backend_held(kind),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.backend_held(kind),
            Self::Cuda(device) => device.backend_held(kind),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.backend_held(kind),
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

    /// The device's last-level cache in bytes, where its backend reports
    /// one: CUDA's L2. Metal, Vulkan and the CPU route report none.
    pub(crate) fn last_level_cache_bytes(&self) -> Option<u64> {
        match self {
            Self::Cuda(device) => Some(device.device_description().facts().properties.l2_bytes.get()),
            _ => None,
        }
    }

    /// Facts that distinguish performance behavior, for keying per-device
    /// native tuning: the device and its driver. The toolchain that forms
    /// its programs is its program former's to name
    /// ([`DevicePrograms::toolchain`]). Equal identities denote the same
    /// device model and configuration; there is no ordering or closeness.
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
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.device_description().facts().tuning_material(),
            // The driver runs CUDA code: an update can change which
            // configuration wins.
            Self::Cuda(device) => {
                let facts = device.device_description().facts();
                format!(
                    "cuda;sm {};multiprocessors {};driver {}",
                    facts.compute_capability, facts.properties.units.get(), facts.driver_api
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
        self.allocate_in(bytes, alignment, None)
    }

    pub(crate) fn allocate_in(
        &self,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_storage_in(bytes, alignment, access),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_storage_in(bytes, alignment, access),
            Self::Cuda(device) => device.allocate_storage_in(bytes, alignment, access),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_in(bytes, alignment, access),
        }
    }

    /// Storage that reads zero, backed ([`Self::zeroed`]).
    pub(crate) fn allocate_zeroed(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        self.zeroed(self.allocate(bytes, alignment)?, bytes)
    }

    /// A fixed buffer of a mapped `contract` ([`Self::allocate_mapped`]) that
    /// reads zero, backed as [`Self::zeroed`] says.
    pub(crate) fn allocate_mapped_zeroed(
        &self,
        contract: Contract,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        self.zeroed(self.allocate_mapped(contract, bytes, alignment)?, bytes)
    }

    /// A fixed buffer of `contract`, which the host reaches in place, in the
    /// kinds its placement names, charged to it. CPU and Metal have one kind.
    pub(crate) fn allocate_mapped(
        &self,
        contract: Contract,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_visible_with(contract, bytes, alignment, |service| {
                service.allocate(bytes, alignment)
            }),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_visible_with(contract, bytes, alignment, |service| {
                service.allocate(bytes, alignment)
            }),
            Self::Cuda(device) => {
                let kind = device.memory().kinds(Placing::Fixed(contract))[0];
                device.allocate_visible_with(contract, bytes, alignment, |service| {
                    service.allocate_kind(kind, bytes, alignment, None)
                })
            }
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_mapped(contract, bytes, alignment),
        }
    }

    /// `allocation`, fresh storage of `bytes`, reading zero. The CPU's
    /// allocations are zero-filled (`alloc_zeroed`) and Metal's are cleared
    /// by Metal (`newBufferWithLength:options:`), each backed as it is
    /// wired, so they are written no second time; CUDA's and Vulkan's memory
    /// holds whatever it held, so it is written with zeros.
    pub(crate) fn zeroed(
        &self,
        allocation: Arc<driver::Allocation>,
        bytes: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(_) => {}
            #[cfg(target_os = "macos")]
            Self::Metal(_) => {}
            _ => driver::write_zeros(allocation.storage(), bytes)?,
        }
        Ok(allocation)
    }

    /// Storage for inputs the host only writes before each submission. Host
    /// writes to it are plain memory writes that never wait for queued
    /// device work. The host never reads it: what it reads is `Shared`.
    pub(crate) fn allocate_upload(
        &self,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        self.allocate_mapped(Contract::Upload, bytes, alignment)
    }

    pub(crate) fn map_read_only_host_region(
        &self,
        region: crate::api::HostRegion,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            // The CPU reads host memory in place: a buffer over the region's
            // bytes, its owner held with it (`HostRegion::new`'s contract).
            Self::Cpu(device) => {
                let bytes = region.len() as u64;
                device.allocate_read_only_storage_with(bytes, alignment, |_| {
                    Ok(unsafe { seismic_cpu::Buffer::borrowed(region.pointer(), bytes, region.owner()) })
                })
            }
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

    /// Device address of byte `offset` of an allocation (of chunked
    /// storage, a byte of a backed chunk). Only Seismic uses this to build
    /// slab and record tables; lifetime still follows the allocation.
    pub(crate) fn allocation_address(&self, allocation: &Arc<driver::Allocation>, offset: u64) -> u64 {
        match self {
            Self::Cpu(_) => driver::typed_buffer::<seismic_cpu::Cpu, CpuExecutor>(allocation)
                .data_pointer() as u64 + offset,
            #[cfg(target_os = "macos")]
            Self::Metal(_) => match allocation.storage().as_any().downcast_ref::<MetalReserved>() {
                Some(MetalReserved(reserved)) => reserved.gpu_address(offset),
                None => driver::typed_buffer::<seismic_metal::Metal, MetalExecutor>(allocation)
                    .gpu_address(offset),
            },
            Self::Cuda(_) => {
                driver::typed_buffer::<seismic_cuda::Cuda, CudaExecutor>(allocation).pointer() + offset
            }
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(_) => vulkan_address(allocation, offset),
        }
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

impl OpenedKind {
    /// Zero-filled storage of a reserved tensor of `reservation`'s records,
    /// its leading `committed` rows backed and charged for what backs them,
    /// and the resizer that resizes it in place, under a contract the device
    /// reserves ([`seismic_topology::Executor::reserving`]).
    pub(crate) fn allocate_reserved_tensor(
        &self,
        reservation: driver::ReservedRange,
        committed: u64,
        alignment: u64,
    ) -> Result<driver::Resizer, crate::api::TensorError> {
        let bytes = reservation.committed_bytes(committed)?;
        let (reserved, granule) = (reservation.bytes, reservation.span.granule);
        Ok(match self {
            Self::Cpu(device) => device.allocate_reservation_with(reservation, bytes, alignment, |cpu| {
                cpu.allocate_reserved(bytes, reserved, granule).map(|(buffer, resizer)| {
                    let storage = driver::buffer_storage::<seismic_cpu::Cpu, CpuExecutor>(cpu, buffer);
                    (storage, ReservedResizer::Cpu(resizer), None)
                })
            }),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_reservation_with(reservation, bytes, alignment, |metal| {
                metal
                    .allocate_reserved(bytes, reserved, granule, reservation.record.get())
                    .map(|(buffer, resizer)| {
                        let table = buffer.table();
                        let storage: Box<dyn driver::Storage> = Box::new(MetalReserved(buffer));
                        (storage, ReservedResizer::Metal(resizer), Some(table))
                    })
            }),
            Self::Cuda(device) => device.allocate_reservation_with(reservation, bytes, alignment, |cuda| {
                let (kind, memory) = (device.memory().kinds(Placing::Reserved(reservation.contract))[0], device.memory());
                let range = seismic_cuda::RangeMemory::of(kind).expect("the rule reserves only in kinds that reserve");
                let account: Arc<dyn seismic_compiler::executable::MemoryAccount> =
                    Arc::new(crate::memory::MemoryAccount(memory.placement(Placing::Reserved(reservation.contract)).physical()));
                cuda.allocate_reserved(bytes, reserved, granule, range, account)
                    .map(|(buffer, resizer)| {
                        let storage =
                            driver::buffer_storage::<seismic_cuda::Cuda, CudaExecutor>(cuda, buffer);
                        (storage, ReservedResizer::Cuda(resizer), None)
                    })
            }),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_reservation(reservation, bytes, alignment),
        }?)
    }
}

/// Metal's reserved storage: chunks the host reads and writes in place
/// and device work reaches only through record tables. It is no
/// `MetalBuffer`, so no binding path can bind it.
#[cfg(target_os = "macos")]
struct MetalReserved(seismic_metal::ReservedBuffer);

#[cfg(target_os = "macos")]
impl driver::Storage for MetalReserved {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.0.read_bytes(offset, into);
        Ok(())
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.0.write_bytes(offset, bytes);
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        Some(self.0.host_base())
    }
}

/// A reserving backend's resizer (`driver::Resizer`): the one reservation's
/// over the backend's backing (the CPU backend's pages, Metal's granules,
/// CUDA's reserved memory, Vulkan's chunks).
pub(crate) enum ReservedResizer {
    Cpu(host::Resizer<host::Pages>),
    #[cfg(target_os = "macos")]
    Metal(host::Resizer<seismic_metal::Granules>),
    Cuda(seismic_cuda::CudaResizer),
    #[cfg(not(target_os = "macos"))]
    Vulkan(seismic_vulkan::VulkanResizer),
}

/// A reserving backend's prepared commitment.
pub(crate) enum ReservedPrepared {
    Cpu(host::Prepared<host::Pages>),
    #[cfg(target_os = "macos")]
    Metal(host::Prepared<seismic_metal::Granules>),
    Cuda(host::Prepared<seismic_cuda::ReservedMemory>),
    #[cfg(not(target_os = "macos"))]
    Vulkan(host::Prepared<seismic_vulkan::reservation::Chunks>),
}

/// A reserving backend's installed commitment, until its release.
pub(crate) enum ReservedPending {
    Cpu(host::Pending<host::Pages>),
    #[cfg(target_os = "macos")]
    Metal(host::Pending<seismic_metal::Granules>),
    Cuda(host::Pending<seismic_cuda::ReservedMemory>),
    #[cfg(not(target_os = "macos"))]
    Vulkan(host::Pending<seismic_vulkan::reservation::Chunks>),
}

impl ReservedResizer {
    /// The backend's prepare of exactly `bytes`; a refusal returns the
    /// resizer.
    pub(crate) fn prepare(self, bytes: u64) -> Result<ReservedPrepared, (Self, ExecutionError)> {
        match self {
            Self::Cpu(resizer) => resizer.prepare(bytes).map(ReservedPrepared::Cpu).map_err(
                |(resizer, failure)| {
                    let error = ExecutionError::AllocationFailed(failure.to_string());
                    (Self::Cpu(resizer), error)
                },
            ),
            #[cfg(target_os = "macos")]
            Self::Metal(resizer) => resizer
                .prepare(bytes)
                .map(ReservedPrepared::Metal)
                .map_err(|(resizer, error)| (Self::Metal(resizer), error)),
            Self::Cuda(resizer) => resizer
                .prepare(bytes)
                .map(ReservedPrepared::Cuda)
                .map_err(|(resizer, error)| (Self::Cuda(resizer), error)),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(resizer) => resizer
                .prepare(bytes)
                .map(ReservedPrepared::Vulkan)
                .map_err(|(resizer, error)| (Self::Vulkan(resizer), error)),
        }
    }
}

impl ReservedPrepared {
    pub(crate) fn install(self) -> ReservedPending {
        match self {
            Self::Cpu(prepared) => ReservedPending::Cpu(prepared.install()),
            #[cfg(target_os = "macos")]
            Self::Metal(prepared) => ReservedPending::Metal(prepared.install()),
            Self::Cuda(prepared) => ReservedPending::Cuda(prepared.install()),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(prepared) => ReservedPending::Vulkan(prepared.install()),
        }
    }

    pub(crate) fn abandon(self) -> ReservedResizer {
        match self {
            Self::Cpu(prepared) => ReservedResizer::Cpu(prepared.abandon()),
            #[cfg(target_os = "macos")]
            Self::Metal(prepared) => ReservedResizer::Metal(prepared.abandon()),
            Self::Cuda(prepared) => ReservedResizer::Cuda(prepared.abandon()),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(prepared) => ReservedResizer::Vulkan(prepared.abandon()),
        }
    }
}

impl ReservedPending {
    pub(crate) fn release(self) -> ReservedResizer {
        match self {
            Self::Cpu(pending) => ReservedResizer::Cpu(pending.release()),
            #[cfg(target_os = "macos")]
            Self::Metal(pending) => ReservedResizer::Metal(pending.release()),
            Self::Cuda(pending) => ReservedResizer::Cuda(pending.release()),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(pending) => ReservedResizer::Vulkan(pending.release()),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use seismic_lang::registry::BackendName;

    /// A stored tuning result is valid only for the programs it measured:
    /// the toolchain that formed them (Metal's: the OS build and the
    /// language version) is part of the identity it is keyed by.
    #[test]
    fn the_tuning_identity_names_the_toolchain() {
        let device = crate::devices::Devices::discover(&[])
            .expect("device discovery")
            .open_backend(BackendName::Metal)
            .expect("Metal device");
        let super::OpenedKind::Metal(metal) = &device.kind else {
            panic!("a Metal device opens as Metal");
        };
        let os = metal.device_description().facts().operating_system().to_owned();
        let toolchain = device.programs.toolchain().expect("Metal forms programs");
        assert!(toolchain.starts_with(&format!("metal;{os};msl-")));
        assert!(device.tuning_identity().contains(&format!(";toolchain {toolchain}")));
    }
}
