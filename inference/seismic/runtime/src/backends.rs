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
use crate::memory::{AllocationCharge, Charges, DeviceStores};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::NativeExecutor;
use seismic_compiler::feedback::PreparationOptions;
use seismic_compiler::graph_cost::GraphCost;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{ElementBindings, LogicalEntry};
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_native_target::{Programs, TargetFamily, Toolchain};
use seismic_discovery::ObservationError;
use seismic_topology::{Contract, DeviceTopology, Executor, ExecutorSelector};
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

    pub(crate) fn call(&self, charges: &Charges, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        match self {
            Self::Cpu(kernel) => kernel.call(charges, args),
            #[cfg(target_os = "macos")]
            Self::Metal(kernel) => kernel.call(charges, args),
            Self::Cuda(kernel) => kernel.call(charges, args),
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
    memory: Arc<DeviceStores>,
    options: crate::artifacts::DeviceOptions,
) -> Result<Arc<DeviceInner>, OpenError> {
    let selector = info.selector();
    let store = options.artifacts.clone().unwrap_or_else(crate::artifacts::ProgramStore::user);
    // Probe answers are the machine's, whatever store the opener names for
    // programs (`crate::artifacts`).
    let probes = crate::artifacts::ProgramStore::user();
    // The CPU records no graphs, and a Metal schedule is host data: only
    // CUDA's and Vulkan's graphs hold device memory, measured on opening
    // unless the opener has it.
    let mut graph_cost = GraphCost::None;
    let given = options.graph_cost;
    // Each arm builds the device's formers before anything it opens forms
    // a program (graph cost, the analytical context), which forms through
    // them.
    let (kind, programs) = match selector {
        ExecutorSelector::HostCpu => {
            let seismic_cpu::OpenedCpu {
                service,
                executor,
                device,
            } = seismic_cpu::open_host(
                topology.host().processors.cores().map_err(|reason| OpenError::Unavailable { selector, reason })?,
                memory.store(Contract::DeviceOnly),
            )
            .map_err(OpenError::Backend)?;
            let cpu = DeviceFormer::new(seismic_cpu::toolchain::CpuToolchain::new(&device), &store);
            let kind = OpenedKind::Cpu(Arc::new(Opened::new(
                service,
                executor,
                seismic_cpu::registry(),
                device,
                |_, executor, device| executor.analytical(device),
                memory.clone(),
            )));
            (kind, DevicePrograms::Cpu { cpu })
        }
        #[cfg(target_os = "macos")]
        ExecutorSelector::Metal { registry_id } => {
            let handle = seismic_metal::DeviceHandle::discover()
                .into_iter()
                .find(|handle| handle.registry_id() == registry_id)
                .ok_or(OpenError::IdentityChanged(selector))?;
            let service =
                seismic_metal::MetalDevice::open(handle.clone(), memory.store(Contract::DeviceOnly))
                    .map_err(OpenError::Backend)?;
            let device =
                seismic_metal::profile::open_device(&service, &*probes).map_err(OpenError::Backend)?;
            let executor = seismic_metal::MetalExecutor::new(service.clone());
            let metal = DeviceFormer::new(
                seismic_metal::toolchain::MetalToolchain::new(service.clone(), device.device_info()),
                &store,
            );
            let analytical = metal.programs().clone();
            let kind = OpenedKind::Metal(Arc::new(Opened::new(
                service,
                executor,
                seismic_metal::profile::registry(),
                device,
                move |service, _, device| seismic_metal::profile::open_analytical(service, device, &analytical),
                memory.clone(),
            )));
            (kind, DevicePrograms::Metal { metal })
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
            // One store twice on unified memory.
            let stores = seismic_cuda::CudaStores {
                device: memory.store(Contract::DeviceOnly),
                host: memory.host_store(),
            };
            let seismic_cuda::OpenedCuda { service, device } =
                seismic_cuda::open(ordinal, stores).map_err(OpenError::Backend)?;
            let capability = device.device_info().compute_capability;
            let architecture = u32::from(capability.major) * 10 + u32::from(capability.minor);
            let cuda = DeviceFormer::new(seismic_cuda::toolchain::CudaToolchain::new(service.clone(), architecture), &store);
            let cuda_ptx =
                DeviceFormer::new(seismic_cuda::toolchain::CudaPtxToolchain::new(service.clone(), architecture), &store);
            graph_cost = given
                .map_or_else(|| seismic_cuda::graph_cost(&service, cuda_ptx.programs()).map_err(open_error), Ok)?;
            let executor = seismic_cuda::Executor::new(service.clone());
            let analytical = cuda_ptx.programs().clone();
            let kind = OpenedKind::Cuda(Arc::new(Opened::new(
                service,
                executor,
                seismic_cuda::registry(),
                device,
                move |service, _, device| seismic_cuda::open_analytical(service, device, &analytical),
                memory.clone(),
            )));
            (kind, DevicePrograms::Cuda { cuda, cuda_ptx })
        }
        #[cfg(not(target_os = "macos"))]
        ExecutorSelector::Vulkan { uuid } => {
            // What the measurement job stored of its probes, keyed by the
            // executor's identity, in the user's store.
            let identity = topology.executor_identity(info.id()).expect("the executor is of its topology");
            let probed = crate::probes::stored(&*probes, &identity, seismic_vulkan::probe::REVISION);
            let opened = VulkanOpened::open(uuid, &info, memory.clone(), &probed)?;
            let vulkan =
                DeviceFormer::new(seismic_vulkan::toolchain::VulkanToolchain::new(opened.service().clone()), &store);
            graph_cost = given.map_or_else(
                || seismic_vulkan::graph_cost(opened.service(), vulkan.programs()).map_err(open_error),
                Ok,
            )?;
            (OpenedKind::Vulkan(Arc::new(opened)), DevicePrograms::Vulkan { vulkan })
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
            seismic_lang::device::DeviceProperties::Metal(*opened.device_description().device_info().properties())
        }
        OpenedKind::Cuda(opened) => {
            seismic_lang::device::DeviceProperties::Cuda(opened.device_description().device_info().properties)
        }
        #[cfg(not(target_os = "macos"))]
        OpenedKind::Vulkan(opened) => seismic_lang::device::DeviceProperties::Vulkan(*opened.service().properties()),
    };
    let test = seismic_lang::registry::TestDevice::opened(info.backend());
    Ok(Arc::new(DeviceInner {
        test,
        info,
        topology,
        memory,
        capabilities: std::sync::OnceLock::new(),
        kind,
        properties,
        programs,
        trace: std::sync::Mutex::new(None),
        artifacts: store,
        native: crate::native::NativeState::default(),
        graph_cost,
        chosen: std::sync::Mutex::new(std::collections::HashMap::new()),
    }))
}

/// One toolchain's former on an opened device, and the same former as the
/// backend code that forms through it takes it ([`seismic_native_target::Programs`]).
pub(crate) struct DeviceFormer<T: Toolchain> {
    former: Arc<ProgramFormer<T>>,
    programs: Arc<dyn Programs<T>>,
}

impl<T: Toolchain> DeviceFormer<T> {
    /// The former of `toolchain`, forming through `store`.
    fn new(toolchain: T, store: &Arc<crate::artifacts::ProgramStore>) -> Self {
        let former = Arc::new(ProgramFormer::new(toolchain, store.clone()));
        Self { programs: former.clone(), former }
    }

    /// The former as backend code forms through it.
    pub(crate) fn programs(&self) -> &Arc<dyn Programs<T>> {
        &self.programs
    }
}

impl<T: Toolchain> std::ops::Deref for DeviceFormer<T> {
    type Target = ProgramFormer<T>;

    fn deref(&self) -> &ProgramFormer<T> {
        &self.former
    }
}

/// The program formers of an opened device, one for each toolchain its
/// backend forms programs with, all over the device's store: what the
/// device's native entries, its Seismic-IR kernels, its graph-cost kernel
/// and its analytical-service probes form through. The native entries'
/// toolchain is Metal's, CUDA's (`cuda`) and Vulkan's; the CPU's native
/// entries are compiled into the binary, so it forms only Seismic-IR
/// kernels (`cpu`), as CUDA forms them and its device kernels from PTX
/// (`cuda_ptx`).
pub(crate) enum DevicePrograms {
    Cpu {
        cpu: DeviceFormer<seismic_cpu::toolchain::CpuToolchain>,
    },
    #[cfg(target_os = "macos")]
    Metal {
        metal: DeviceFormer<seismic_metal::toolchain::MetalToolchain>,
    },
    Cuda {
        cuda: DeviceFormer<seismic_cuda::toolchain::CudaToolchain>,
        cuda_ptx: DeviceFormer<seismic_cuda::toolchain::CudaPtxToolchain>,
    },
    #[cfg(not(target_os = "macos"))]
    Vulkan {
        vulkan: DeviceFormer<seismic_vulkan::toolchain::VulkanToolchain>,
    },
}

impl DevicePrograms {
    /// Lend the former of the device's native entries `workers`, or take
    /// them back ([`ProgramFormer::staff`]). The CPU forms no native
    /// entries; every other former forms on the thread that asks.
    pub(crate) fn staff(
        &self,
        workers: Option<Arc<seismic_native_target::workers::Workers>>,
    ) {
        match self {
            Self::Cpu { .. } => {}
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => metal.staff(workers),
            Self::Cuda { cuda, .. } => cuda.staff(workers),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => vulkan.staff(workers),
        }
    }

    /// The threads lent to the former of the device's native entries, if
    /// any.
    pub(crate) fn workers(&self) -> Option<Arc<seismic_native_target::workers::Workers>> {
        match self {
            Self::Cpu { .. } => None,
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => metal.workers(),
            Self::Cuda { cuda, .. } => cuda.workers(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => vulkan.workers(),
        }
    }

    /// What besides the source determines the native entries this device's
    /// toolchain forms ([`seismic_native_target::ToolchainIdentity`]). The
    /// CPU backend forms none: its native entries are compiled into the
    /// binary.
    pub(crate) fn toolchain(&self) -> Option<&seismic_native_target::ToolchainIdentity> {
        match self {
            Self::Cpu { .. } => None,
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => Some(metal.toolchain().identity()),
            Self::Cuda { cuda, .. } => Some(cuda.toolchain().identity()),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => Some(vulkan.toolchain().identity()),
        }
    }

    /// The programs every toolchain of this device has formed, and the
    /// time their calls took.
    pub(crate) fn programs_formed(&self) -> crate::formation::FormedPrograms {
        match self {
            Self::Cpu { cpu } => cpu.programs_formed(),
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => metal.programs_formed(),
            Self::Cuda { cuda, cuda_ptx } => [cuda.programs_formed(), cuda_ptx.programs_formed()].into_iter().sum(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => vulkan.programs_formed(),
        }
    }

    /// `sources` as the former of this device's native entries reads them
    /// ([`ProgramFormer::keyed`]). The CPU backend forms no native entries.
    pub(crate) fn keyed(
        &self,
        sources: Vec<seismic_native_target::ProgramSource>,
    ) -> Vec<crate::formation::Keyed> {
        let keyed = |key: &dyn Fn(seismic_native_target::ProgramSource) -> crate::formation::Keyed| {
            sources.into_iter().map(key).collect()
        };
        match self {
            Self::Cpu { .. } => Vec::new(),
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => keyed(&|source| metal.keyed(source)),
            Self::Cuda { cuda, .. } => keyed(&|source| cuda.keyed(source)),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => keyed(&|source| vulkan.keyed(source)),
        }
    }

    /// What forming native entries' `sources` together on this device
    /// does, known before they are formed, and the most entries one
    /// source's compile forms.
    pub(crate) fn class(
        &self,
        sources: &[crate::formation::Keyed],
    ) -> (crate::formation::FormationClass, usize) {
        match self {
            Self::Cpu { .. } => (crate::formation::FormationClass::Kept, 0),
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => metal.class(sources),
            Self::Cuda { cuda, .. } => cuda.class(sources),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => vulkan.class(sources),
        }
    }

    /// What the compiles of this device's native entries have taken in
    /// this process, by the entries each formed.
    pub(crate) fn compiled(&self) -> crate::native::Linear {
        match self {
            Self::Cpu { .. } => crate::native::Linear::default(),
            #[cfg(target_os = "macos")]
            Self::Metal { metal } => metal.compiled(),
            Self::Cuda { cuda, .. } => cuda.compiled(),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan { vulkan } => vulkan.compiled(),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal(&self) -> &ProgramFormer<seismic_metal::toolchain::MetalToolchain> {
        let Self::Metal { metal } = self else {
            unreachable!("a Metal device forms Metal programs");
        };
        metal
    }

    /// The programs a Metal device's Seismic-IR kernels form through: its
    /// native entries' former.
    #[cfg(target_os = "macos")]
    fn metal_programs(&self) -> &seismic_metal::MetalPrograms {
        let Self::Metal { metal } = self else {
            unreachable!("a Metal device forms Metal programs");
        };
        metal.programs()
    }

    pub(crate) fn cuda(&self) -> &ProgramFormer<seismic_cuda::toolchain::CudaToolchain> {
        let Self::Cuda { cuda, .. } = self else {
            unreachable!("a CUDA device forms CUDA programs");
        };
        cuda
    }

    #[cfg(not(target_os = "macos"))]
    pub(crate) fn vulkan(&self) -> &ProgramFormer<seismic_vulkan::toolchain::VulkanToolchain> {
        let Self::Vulkan { vulkan } = self else {
            unreachable!("a Vulkan device forms Vulkan programs");
        };
        vulkan
    }

    /// The programs a CPU device's Seismic-IR kernels form through.
    fn cpu_programs(&self) -> &seismic_cpu::CpuPrograms {
        let Self::Cpu { cpu } = self else {
            unreachable!("a CPU device forms CPU programs");
        };
        cpu.programs()
    }

    /// The programs a CUDA device's Seismic-IR kernels form through.
    fn cuda_ptx_programs(&self) -> &seismic_cuda::CudaPtxPrograms {
        let Self::Cuda { cuda_ptx, .. } = self else {
            unreachable!("a CUDA device forms CUDA programs");
        };
        cuda_ptx.programs()
    }
}

/// The machine's cores as this process sees them (`Processors::cores`): the
/// CPU device's pool (`seismic_cpu::open_host`). Never the process's thread
/// share: the pool's size is part of the CPU device's identity, which every
/// process of the machine, serving or measuring, must open alike.
/// A device's memory comes only through these, so each allocation is work its
/// test did on it ([`DeviceInner::ran_work`]).
impl DeviceInner {
    /// [`OpenedKind::allocate_contract`] on this device.
    pub(crate) fn allocate_contract(
        &self,
        charge: AllocationCharge<'_>,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        seismic_memory::assert_blocking_allocation_allowed("DeviceInner::allocate_contract");
        self.ran_work();
        self.kind.allocate_contract(charge, bytes, alignment, access)
    }

    /// [`OpenedKind::allocate_block`] on this device.
    pub(crate) fn allocate_block(
        &self,
        kinds: &[u32],
        bytes: u64,
        entry: &Arc<dyn crate::memory::Entry>,
        access: Arc<driver::AccessGroup>,
        fill: seismic_memory::SlabFill,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        seismic_memory::assert_blocking_allocation_allowed("DeviceInner::allocate_block");
        self.ran_work();
        self.kind.allocate_block(kinds, bytes, entry, access, fill)
    }

    /// [`OpenedKind::map_read_only_host_region`] on this device.
    pub(crate) fn map_read_only_host_region(
        &self,
        region: crate::api::HostRegion,
        alignment: u64,
    ) -> Result<Arc<driver::Allocation>, crate::api::TensorError> {
        seismic_memory::assert_blocking_allocation_allowed("DeviceInner::map_read_only_host_region");
        self.ran_work();
        self.kind.map_read_only_host_region(region, alignment).map_err(crate::api::TensorError::Execution)
    }
}

pub(crate) fn host_cores() -> Result<std::num::NonZeroUsize, String> {
    seismic_discovery::discover_processors().cores()
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

    pub(crate) fn workflow(&self, charges: &Charges) -> Result<WorkflowDraftKind, WorkflowError> {
        Ok(match self {
            Self::Cpu(device) => {
                WorkflowDraftKind::Cpu(driver::WorkflowGraphDraft::new(device.clone(), charges.clone()))
            }
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                WorkflowDraftKind::Metal(driver::WorkflowGraphDraft::new(device.clone(), charges.clone()))
            }
            Self::Cuda(device) => {
                WorkflowDraftKind::Cuda(driver::WorkflowGraphDraft::new(device.clone(), charges.clone()))
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
            Self::Cuda(device) => Some(device.device_description().device_info().properties.l2_bytes.get()),
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
                let info = device.device_description().device_info();
                format!(
                    "cpu;{};workers {};simd {:?};tier {};library {}",
                    std::env::consts::ARCH,
                    info.workers,
                    info.simd,
                    seismic_native_cpu::Tier::detected().map_or("none", |tier| tier.name()),
                    seismic_native_cpu::VERSION
                )
            }
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.device_description().device_info().tuning_material(),
            // The driver runs CUDA code: an update can change which
            // configuration wins.
            Self::Cuda(device) => {
                let info = device.device_description().device_info();
                format!(
                    "cuda;sm {};multiprocessors {};driver {}",
                    info.compute_capability, info.properties.units.get(), info.driver_api
                )
            }
            // A driver update changes the key (§7.5).
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.service().info().tuning_identity(),
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

    /// A fixed buffer charged as `charge` says, landing in the first of its
    /// kinds with room and lying along that kind's path, ordered by
    /// `access`'s hazard domain where given (a slab store's address table,
    /// its pool's).
    fn allocate_contract(
        &self,
        charge: AllocationCharge<'_>,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<driver::AccessGroup>>,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        match self {
            Self::Cpu(device) => device.allocate_contract(charge, bytes, alignment, access),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.allocate_contract(charge, bytes, alignment, access),
            Self::Cuda(device) => device.allocate_contract(charge, bytes, alignment, access),
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_contract(charge, bytes, alignment, access),
        }
    }

    /// One block of the granule pool of `contract`'s store: one native
    /// allocation of `bytes` of the first of `contract`'s kinds that admits
    /// it, in `kinds` (those of the pool's store, in placement order),
    /// charged exactly to `entry`, in `access`'s hazard domain, lying along
    /// the path of the kind it landed in (`driver::Opened::allocate_block`).
    /// A refusal by `entry` is the last kind's refusal: every kind charges
    /// the same store.
    fn allocate_block(
        &self,
        kinds: &[u32],
        bytes: u64,
        entry: &Arc<dyn crate::memory::Entry>,
        access: Arc<driver::AccessGroup>,
        fill: seismic_memory::SlabFill,
    ) -> Result<Arc<driver::Allocation>, ExecutionError> {
        let allocate = |kind: u32| match self {
            Self::Cpu(device) => {
                device.allocate_block(kind, bytes, access.clone(), |service| service.allocate_block(kind, bytes, entry, fill))
            }
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                device.allocate_block(kind, bytes, access.clone(), |service| service.allocate_block(kind, bytes, entry, fill))
            }
            Self::Cuda(device) => {
                device.allocate_block(kind, bytes, access.clone(), |service| service.allocate_block(kind, bytes, entry, fill))
            }
            #[cfg(not(target_os = "macos"))]
            Self::Vulkan(device) => device.allocate_block(kind, bytes, entry, access.clone(), fill),
        };
        allocate_across_kinds(kinds, allocate)
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
        seismic_memory::assert_blocking_allocation_allowed("OpenedKind::zeroed");
        match self {
            Self::Cpu(_) => {}
            #[cfg(target_os = "macos")]
            Self::Metal(_) => {}
            _ => driver::write_zeros(allocation.storage(), 0, bytes)?,
        }
        Ok(allocation)
    }

    fn map_read_only_host_region(
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

    /// Device address of byte `offset` of an allocation. Only Seismic uses
    /// this to build slab and record tables; lifetime still follows the
    /// allocation.
    pub(crate) fn allocation_address(&self, allocation: &Arc<driver::Allocation>, offset: u64) -> u64 {
        match self {
            Self::Cpu(_) => driver::typed_buffer::<seismic_cpu::Cpu, CpuExecutor>(allocation)
                .data_pointer() as u64 + offset,
            #[cfg(target_os = "macos")]
            Self::Metal(_) => {
                driver::typed_buffer::<seismic_metal::Metal, MetalExecutor>(allocation).gpu_address(offset)
            }
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
        charges: &Charges,
        options: PreparationOptions,
    ) -> Result<PreparedKind, crate::api::kernel::PrepareError> {
        let public_device = charges.device();
        match self {
            Self::Cpu(device) => prepare(
                device,
                seismic_cpu::native_compiler(),
                public_device.programs.cpu_programs(),
                logical,
                bindings,
                charges,
                options,
            )
            .map(PreparedKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => prepare(
                device,
                seismic_metal::native_compiler(),
                public_device.programs.metal_programs(),
                logical,
                bindings,
                charges,
                options,
            )
            .map(PreparedKind::Metal),
            Self::Cuda(device) => prepare(
                device,
                seismic_cuda::native_compiler(),
                public_device.programs.cuda_ptx_programs(),
                logical,
                bindings,
                charges,
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
        charges: &'a Charges,
        precision: seismic_lang::precision::PrecisionPolicy,
        options: seismic_compiler::feedback::FeedbackOptions,
    ) -> Result<(FeedbackKind<'a>, PreparedKind), crate::api::kernel::PrepareError> {
        let public_device = charges.device();
        match self {
            Self::Cpu(device) => driver::FeedbackCampaign::start(
                device,
                seismic_cpu::native_compiler(),
                public_device.programs.cpu_programs(),
                module,
                entry,
                bindings,
                charges,
                precision,
                options,
            )
            .map(|(campaign, kernel)| (FeedbackKind::Cpu(campaign), PreparedKind::Cpu(kernel))),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => driver::FeedbackCampaign::start(
                device,
                seismic_metal::native_compiler(),
                public_device.programs.metal_programs(),
                module,
                entry,
                bindings,
                charges,
                precision,
                options,
            )
            .map(|(campaign, kernel)| (FeedbackKind::Metal(campaign), PreparedKind::Metal(kernel))),
            Self::Cuda(device) => driver::FeedbackCampaign::start(
                device,
                seismic_cuda::native_compiler(),
                public_device.programs.cuda_ptx_programs(),
                module,
                entry,
                bindings,
                charges,
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
    charges: &Charges,
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
        charges,
        options,
    )?;
    Ok(Arc::new(PreparedHandle {
        prepared,
        device: charges.device().clone(),
    }))
}

/// Allocate in the first kind of `kinds` that admits it: a kind short of room
/// (any capacity refusal: [`ExecutionError::capacity_refusal`]) passes the
/// block to the next; any other failure returns immediately.
fn allocate_across_kinds<T>(
    kinds: &[u32],
    mut allocate: impl FnMut(u32) -> Result<T, ExecutionError>,
) -> Result<T, ExecutionError> {
    let (last, earlier) = kinds.split_last().expect("a pool's store has a kind its contract places in");
    for &kind in earlier {
        match allocate(kind) {
            Err(ref error) if error.capacity_refusal().is_some() => continue,
            done => return done,
        }
    }
    allocate(*last)
}

#[cfg(test)]
mod tests {
    use super::allocate_across_kinds;
    use seismic_compiler::errors::{ExecutionError, RefusalReading};
    use seismic_lang::registry::BackendName;

    #[test]
    fn allocate_across_kinds_falls_back_on_capacity_and_device_refused() {
        let mut tried = Vec::new();
        let landed = allocate_across_kinds(&[1, 2, 3], |kind| {
            tried.push(kind);
            match kind {
                1 => Err(ExecutionError::AllocationCapacity { available: 0, required: 64u64.into() }),
                2 => Err(ExecutionError::DeviceRefused { required: 64, reading: RefusalReading::Host }),
                3 => Ok(kind),
                _ => unreachable!(),
            }
        })
        .expect("third kind admits");
        assert_eq!(landed, 3);
        assert_eq!(tried, vec![1, 2, 3]);
    }

    #[test]
    fn allocate_across_kinds_stops_on_non_capacity_error_and_returns_last_refusal() {
        let mut tried = Vec::new();
        let failed = allocate_across_kinds::<u32>(&[1, 2], |kind| {
            tried.push(kind);
            Err(ExecutionError::AllocationFailed(format!("kind {kind}")))
        });
        assert!(matches!(failed, Err(ExecutionError::AllocationFailed(_))));
        assert_eq!(tried, vec![1]);

        let mut tried = Vec::new();
        let lost = allocate_across_kinds::<u32>(&[1, 2], |kind| {
            tried.push(kind);
            Err(ExecutionError::DeviceLost(format!("kind {kind}")))
        });
        assert!(matches!(lost, Err(ExecutionError::DeviceLost(_))));
        assert_eq!(tried, vec![1]);

        let mut tried = Vec::new();
        let refused = allocate_across_kinds::<u32>(&[1, 2], |kind| {
            tried.push(kind);
            Err(ExecutionError::DeviceRefused { required: u64::from(kind), reading: RefusalReading::Host })
        });
        assert!(matches!(refused, Err(ExecutionError::DeviceRefused { required: 2, .. })));
        assert_eq!(tried, vec![1, 2]);
    }

    /// A stored tuning result is valid only for the programs it measured:
    /// the toolchain's material that formed them (Metal's: the OS build and
    /// the language version) is part of the identity it is keyed by. The
    /// build of the backend's formation code is not, so a rebuild keeps
    /// every tuning record while its stored programs miss
    /// (`formation::tests::a_rebuilt_formation_misses_the_programs_stored_before`).
    /// Checked on each accelerator the host opens; Metal is on every Mac.
    #[test]
    fn the_tuning_identity_names_the_toolchain_and_not_its_build() {
        let devices = crate::devices::Devices::discover(&[]).expect("device discovery");
        for backend in [BackendName::Metal, BackendName::Cuda, BackendName::Vulkan] {
            let device = match devices.open_backend(backend) {
                Ok(device) => device,
                Err(_) if !(cfg!(target_os = "macos") && backend == BackendName::Metal) => continue,
                Err(error) => panic!("a Mac opens Metal: {error:?}"),
            };
            #[cfg(target_os = "macos")]
            if let super::OpenedKind::Metal(metal) = &device.kind {
                let os = metal.device_description().device_info().operating_system().to_owned();
                let toolchain = device.programs.toolchain().expect("Metal forms programs");
                assert!(toolchain.material.starts_with(&format!("metal;{os};msl-")));
            }
            let toolchain = device.programs.toolchain().expect("an accelerator forms programs");
            let tuning = device.tuning_identity();
            assert!(tuning.contains(&format!(";toolchain {}", toolchain.material)), "{tuning}");
            assert!(!tuning.contains(&toolchain.stored), "{tuning}");
        }
    }
}
