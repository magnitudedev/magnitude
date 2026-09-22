//! The runtime's one closed sum over supported backends.
//!
//! Backend crates remain generic and do not depend on the runtime.  This
//! module is the sole composition root which turns their target profiles,
//! device services, executors, and prepared kernels into the public device
//! API.  It does not reproduce target facts or planning policy.

use crate::api::{
    device::DeviceInner,
    kernel::{
        DecodedResults, EncodedArgs, EncodedWorkflowArgs, NativeDefinition, PendingWorkflowResults,
        WorkflowCompletionAny,
    },
    CallError, DeviceId, DeviceInfo, WorkflowError,
};
use crate::driver::{self, Opened, PreparedHandle};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::executable::NativeExecutor;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_target::TargetFamily;
use std::sync::Arc;

type CpuExecutor = seismic_cpu::Executor;
type CpuOpened = Opened<seismic_cpu::Cpu, CpuExecutor>;
type CpuPrepared = PreparedHandle<seismic_cpu::Cpu, CpuExecutor>;
type CpuWorkflowDraft = driver::WorkflowGraphDraft<seismic_cpu::Cpu, CpuExecutor>;
type CpuBoundWorkflow = driver::BoundWorkflowGraph<seismic_cpu::Cpu, CpuExecutor>;
type CpuAdmittedRun = crate::execution::AdmittedRun<seismic_cpu::Cpu, CpuExecutor>;

#[cfg(target_os = "macos")]
type MetalExecutor = seismic_metal::MetalExecutor;
#[cfg(target_os = "macos")]
type MetalOpened = Opened<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalPrepared = PreparedHandle<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalWorkflowDraft = driver::WorkflowGraphDraft<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalBoundWorkflow = driver::BoundWorkflowGraph<seismic_metal::Metal, MetalExecutor>;
#[cfg(target_os = "macos")]
type MetalAdmittedRun = crate::execution::AdmittedRun<seismic_metal::Metal, MetalExecutor>;

type CudaExecutor = seismic_cuda::Executor;
type CudaOpened = Opened<seismic_cuda::Cuda, CudaExecutor>;
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
    },
}

/// One opened backend.  This is deliberately closed: callers cannot inject
/// an executor with a profile from a different device.
pub(crate) enum DeviceKind {
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

pub(crate) enum NativePreparedKind {
    #[cfg(target_os = "macos")]
    Metal(Arc<driver::NativePreparedMetal>),
    #[cfg(not(target_os = "macos"))]
    Unsupported,
}

impl NativePreparedKind {
    pub(crate) fn call(&self, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(kernel) => kernel.call(args),
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => unreachable!("unsupported native kernel cannot be prepared"),
        }
    }
}

impl PreparedKind {
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

pub(crate) fn discover() -> Result<Vec<DeviceInfo>, TargetError> {
    let mut infos = Vec::new();

    // CPU discovery does not create the production worker pool. The address
    // limit is cheap identity information; the exact pool and its complete
    // profile are constructed atomically by `open`.
    infos.push(info(
        seismic_lang::registry::BackendName::Cpu,
        format!("Host CPU ({})", std::env::consts::ARCH),
        isize::MAX as u64,
        infos.len(),
        Descriptor::Cpu,
    ));

    #[cfg(target_os = "macos")]
    for handle in seismic_metal::DeviceHandle::discover() {
        infos.push(info(
            seismic_lang::registry::BackendName::Metal,
            handle.name(),
            handle.memory_bytes(),
            infos.len(),
            Descriptor::Metal { handle },
        ));
    }

    // Absence of a CUDA driver/device is not catalog failure. Description
    // performs only driver enumeration/name/memory queries; context, stream,
    // probes, and profile assembly remain in `open`.
    if let Ok(count) = seismic_cuda::device_count() {
        for ordinal in 0..count {
            let descriptor = seismic_cuda::describe(ordinal)?;
            infos.push(info(
                seismic_lang::registry::BackendName::Cuda,
                descriptor.name,
                descriptor.memory_bytes,
                infos.len(),
                Descriptor::Cuda { ordinal },
            ));
        }
    }

    Ok(infos)
}

fn info(
    backend: seismic_lang::registry::BackendName,
    name: String,
    memory_bytes: u64,
    ordinal: usize,
    descriptor: Descriptor,
) -> DeviceInfo {
    DeviceInfo {
        id: DeviceId(ordinal),
        backend,
        name,
        memory_bytes,
        descriptor: Arc::new(descriptor),
    }
}

pub(crate) fn open(infos: &[DeviceInfo], id: DeviceId) -> Result<Arc<DeviceInner>, TargetError> {
    let info = infos
        .get(id.0)
        .filter(|info| info.id == id)
        .cloned()
        .ok_or_else(|| TargetError::DeviceUnavailable(format!("unknown device id {}", id.0)))?;
    let descriptor = info.descriptor.clone();

    let kind = match descriptor.as_ref() {
        Descriptor::Cpu => {
            let seismic_cpu::OpenedCpu {
                service,
                executor,
                device,
                analytical,
            } = seismic_cpu::open_host()?;
            DeviceKind::Cpu(Arc::new(Opened::new(
                service,
                executor,
                seismic_cpu::registry(),
                device,
                analytical,
            )))
        }
        #[cfg(target_os = "macos")]
        Descriptor::Metal { handle } => {
            let service = seismic_metal::MetalDevice::open(handle.clone())?;
            let device = seismic_metal::profile::open_device(&service)?;
            let executor = seismic_metal::MetalExecutor::new(service.clone());
            DeviceKind::Metal(Arc::new(Opened::new_lazy(
                service,
                executor,
                seismic_metal::profile::registry(),
                device,
                seismic_metal::profile::open_analytical,
            )))
        }
        Descriptor::Cuda { ordinal } => {
            let seismic_cuda::OpenedCuda {
                service,
                device,
                analytical,
            } = seismic_cuda::open(*ordinal).map_err(open_error)?;
            let executor = seismic_cuda::Executor::new(service.clone());
            DeviceKind::Cuda(Arc::new(Opened::new(
                service,
                executor,
                seismic_cuda::registry(),
                device,
                analytical,
            )))
        }
    };
    Ok(Arc::new(DeviceInner {
        info,
        capabilities: std::sync::OnceLock::new(),
        kind,
    }))
}

fn open_error(error: ExecutionError) -> TargetError {
    TargetError::DeviceUnavailable(error.to_string())
}

impl DeviceKind {
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
    ) -> Result<(), crate::api::MemoryLimitError> {
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
            Self::Cuda(device) => device
                .device_description()
                .dtypes()
                .representations
                .contains(&representation),
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

    pub(crate) fn prepare(
        &self,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        public_device: &Arc<DeviceInner>,
        precision: PrecisionPolicy,
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
                precision,
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
                    precision,
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
                precision,
            )
            .map(PreparedKind::Cuda),
        }
    }

    pub(crate) fn prepare_native(
        &self,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        public_device: &Arc<DeviceInner>,
        definition: NativeDefinition,
    ) -> Result<NativePreparedKind, crate::api::kernel::PrepareError> {
        match self {
            #[cfg(target_os = "macos")]
            Self::Metal(device) => driver::prepare_native_metal(
                device,
                module,
                entry,
                bindings,
                public_device,
                definition,
            )
            .map(NativePreparedKind::Metal),
            Self::Cpu(_) | Self::Cuda(_) => Err(crate::api::kernel::PrepareError::Preparation(
                seismic_compiler::errors::PreparationError::NoApplicableImplementation(
                    seismic_compiler::errors::NoApplicableReport {
                        entry: definition.entry.to_owned(),
                        declined: vec![(
                            "native.metal".to_owned(),
                            "the selected device is not a Metal device".to_owned(),
                        )],
                    },
                ),
            )),
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
    precision: PrecisionPolicy,
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
        precision,
    )?;
    Ok(Arc::new(PreparedHandle {
        prepared,
        device: public_device.clone(),
    }))
}
