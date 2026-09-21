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
        WorkflowCompletionAny, WorkflowResultRef,
    },
    CallError, DeviceId, DeviceInfo, WorkflowError,
};
use crate::driver::{self, Opened, PreparedHandle};
use seismic_compiler::errors::{ExecutionError, TargetError};
use seismic_compiler::target::Backend;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::ElementBindings;
use seismic_lang::ids::{EntryId, RepresentationId};
use seismic_lang::precision::PrecisionPolicy;
use std::sync::Arc;

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
    Cpu(Arc<Opened<seismic_cpu::Cpu>>),
    #[cfg(target_os = "macos")]
    Metal(Arc<Opened<seismic_metal::Metal>>),
    Cuda(Arc<Opened<seismic_cuda::Cuda>>),
}

#[derive(Clone)]
pub(crate) enum PreparedKind {
    Cpu(Arc<PreparedHandle<seismic_cpu::Cpu>>),
    #[cfg(target_os = "macos")]
    Metal(Arc<PreparedHandle<seismic_metal::Metal>>),
    Cuda(Arc<PreparedHandle<seismic_cuda::Cuda>>),
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
    Cpu(driver::WorkflowGraphDraft<seismic_cpu::Cpu>),
    #[cfg(target_os = "macos")]
    Metal(driver::WorkflowGraphDraft<seismic_metal::Metal>),
    Cuda(driver::WorkflowGraphDraft<seismic_cuda::Cuda>),
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

    pub(crate) fn run(self, outputs: Vec<WorkflowResultRef>) -> Result<DecodedResults, CallError> {
        self.submit()?.resolve(outputs)
    }

    pub(crate) fn submit(self) -> Result<WorkflowCompletionAny, CallError> {
        match self {
            Self::Cpu(workflow) => Ok(workflow.admit()?.submit()?.into_completion()),
            #[cfg(target_os = "macos")]
            Self::Metal(workflow) => Ok(workflow.admit()?.submit()?.into_completion()),
            Self::Cuda(workflow) => Ok(workflow.admit()?.submit()?.into_completion()),
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
            let executor = seismic_cpu::Executor::host()?;
            let service = *executor.device();
            let contract = executor.contract().clone();
            let execution = executor.execution_profile().clone();
            DeviceKind::Cpu(Arc::new(Opened::new(
                service, executor, contract, execution,
            )))
        }
        #[cfg(target_os = "macos")]
        Descriptor::Metal { handle } => {
            let service = seismic_metal::MetalDevice::open(handle.clone())?;
            let executor = seismic_metal::MetalExecutor::new(service.clone());
            DeviceKind::Metal(Arc::new(Opened::new_lazy(
                service,
                executor,
                seismic_metal::profile::open,
            )))
        }
        Descriptor::Cuda { ordinal } => {
            let service = seismic_cuda::Device::open(*ordinal).map_err(open_error)?;
            let contract = service.contract().clone();
            let execution = service.execution_profile().clone();
            let executor = seismic_cuda::Executor::new(service.clone());
            DeviceKind::Cuda(Arc::new(Opened::new(
                service, executor, contract, execution,
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
    pub(crate) fn workflow(&self, public_device: &Arc<DeviceInner>) -> WorkflowDraftKind {
        match self {
            Self::Cpu(device) => WorkflowDraftKind::Cpu(driver::WorkflowGraphDraft::new(
                device.clone(),
                public_device.clone(),
            )),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => WorkflowDraftKind::Metal(driver::WorkflowGraphDraft::new(
                device.clone(),
                public_device.clone(),
            )),
            Self::Cuda(device) => WorkflowDraftKind::Cuda(driver::WorkflowGraphDraft::new(
                device.clone(),
                public_device.clone(),
            )),
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

    pub(crate) fn memory_usage(&self) -> driver::MemoryUsage {
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
                .contract()
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
                .contract()
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
            Self::Cpu(device) => prepare(device, module, entry, bindings, public_device, precision)
                .map(PreparedKind::Cpu),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => {
                prepare(device, module, entry, bindings, public_device, precision)
                    .map(PreparedKind::Metal)
            }
            Self::Cuda(device) => {
                prepare(device, module, entry, bindings, public_device, precision)
                    .map(PreparedKind::Cuda)
            }
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

fn prepare<B: Backend>(
    device: &Arc<Opened<B>>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    public_device: &Arc<DeviceInner>,
    precision: PrecisionPolicy,
) -> Result<Arc<PreparedHandle<B>>, crate::api::kernel::PrepareError> {
    let prepared = driver::prepare(device, module, entry, bindings, precision)?;
    Ok(Arc::new(PreparedHandle {
        prepared,
        device: public_device.clone(),
    }))
}
