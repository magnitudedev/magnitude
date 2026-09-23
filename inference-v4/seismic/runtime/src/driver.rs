//! The single backend-generic preparation and synchronous invocation driver.
//! Planning ends at `PreparedKernel`; this module validates public arguments,
//! evaluates a selected executable, owns storage, and executes its typed schedule.

use crate::api::kernel::{
    DecodedResults, DecodedValue, EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument,
    EncodedOutputs, NativeDefinition, NativeExpr, PendingWorkflowResults, PrepareError, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::{CallError, OutputError};
use crate::memory::{MemoryCharge, MemoryDomain, MemoryReservation, MemoryUsage};
use crate::resources::{AdmissionDomain, AdmittedResources, PersistentTable};
use crate::telemetry::{self, hex, key_bool, key_str, key_u64, Timed};
use opentelemetry::KeyValue;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::evaluation::AnalyticalEvaluationContext;
use seismic_compiler::executable::{
    execute_variant, DeviceService, ExecutableResultBinding, ExecutableVariant, NativeExecutor,
    RuntimeBuffer,
};
use seismic_compiler::executable::{ExecutableAllocationKind, ExecutableGlobalAllocationKind};
use seismic_compiler::feedback::{
    EvaluationMethod, FeedbackPreparation, FeedbackReport, PreparationOptions,
};
use seismic_compiler::numerics::PolicyIdentity;
use seismic_compiler::prepared::{
    validate_invocation, ArgumentValue, DeviceIdentity, InvocationContract, PreparedKernel,
};
use seismic_compiler::target::CompilerRegistry;
use seismic_compiler::{
    prepare_analytically, OptimizationCompletion, PlanningBudget, PreparationBudget,
};
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{
    CallSchema, ElementBindings, LogicalEntry, ParameterKind, ResultKind, TensorAccess,
};
use seismic_lang::expr::compiled::{CompiledNat, InvocationValues};
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::{EntryId, ModuleHash, RepresentationId, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry;
use seismic_target::{DeviceDescription, TargetFamily};
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

pub(crate) type Service<T, E> = <E as NativeExecutor<T>>::Device;
pub(crate) type Buffer<T, E> = <Service<T, E> as DeviceService<T>>::Buffer;

static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);
static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);
static NEXT_PREPARED: AtomicU64 = AtomicU64::new(1);

fn fresh_allocation_identity() -> u64 {
    NEXT_ALLOCATION.fetch_add(1, Ordering::Relaxed)
}

fn admitted<T>(value: Result<T, seismic_lang::expr::EvalError>) -> T {
    value.unwrap_or_else(|error| panic!("PreparedKernel coverage invariant violated: admitted evaluator was not total: {error:?}"))
}

pub(crate) struct Opened<T, E>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
{
    identity: DeviceIdentity,
    service: Arc<Service<T, E>>,
    device: Arc<DeviceDescription<T>>,
    executor: E,
    compiler_registry: &'static CompilerRegistry<T>,
    analytical: std::sync::OnceLock<
        Result<AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError>,
    >,
    analytical_loader:
        fn(
            &Service<T, E>,
            &E,
            Arc<DeviceDescription<T>>,
        )
            -> Result<AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError>,
    cache: Mutex<HashMap<PreparationKey, Weak<Prepared<T, E>>>>,
    memory: Arc<MemoryDomain>,
    admission: AdmissionDomain,
}

#[derive(PartialEq, Eq, Hash)]
struct PreparationKey {
    module: ModuleHash,
    entry: StableEntryId,
    bindings: Vec<(String, RepresentationId)>,
    policy: PolicyIdentity,
    evaluation: [u8; 32],
}

impl<T, E> Opened<T, E>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
{
    pub(crate) fn new(
        service: Service<T, E>,
        executor: E,
        compiler_registry: &'static CompilerRegistry<T>,
        device: Arc<DeviceDescription<T>>,
        analytical_loader: fn(
            &Service<T, E>,
            &E,
            Arc<DeviceDescription<T>>,
        ) -> Result<
            AnalyticalEvaluationContext<T>,
            seismic_compiler::errors::TargetError,
        >,
        memory: Arc<MemoryDomain>,
    ) -> Self {
        Self {
            identity: DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)),
            service: Arc::new(service),
            device,
            executor,
            compiler_registry,
            analytical: std::sync::OnceLock::new(),
            analytical_loader,
            cache: Mutex::new(HashMap::new()),
            memory,
            admission: AdmissionDomain::new(),
        }
    }
    fn analytical(
        &self,
    ) -> Result<&AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError> {
        self.analytical
            .get_or_init(|| {
                let context =
                    (self.analytical_loader)(&self.service, &self.executor, self.device.clone())?;
                assert!(
                    context.is_bound_to(&self.device),
                    "analytical context belongs to another device"
                );
                Ok(context)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
    pub(crate) fn identity(&self) -> DeviceIdentity {
        self.identity
    }
    pub(crate) fn device_description(&self) -> &DeviceDescription<T> {
        &self.device
    }
    pub(crate) fn begin_submission(&self) -> Result<E::Submission, ExecutionError> {
        self.executor.begin_submission()
    }
    pub(crate) fn service(&self) -> &Service<T, E> {
        &self.service
    }
    pub(crate) fn service_arc(&self) -> Arc<Service<T, E>> {
        self.service.clone()
    }
    fn cache(&self) -> MutexGuard<'_, HashMap<PreparationKey, Weak<Prepared<T, E>>>> {
        self.cache
            .lock()
            .expect("Opened preparation-cache lock poisoned while mutating private cache state")
    }
    pub(crate) fn memory_usage(&self) -> MemoryUsage {
        self.memory.usage()
    }
    pub(crate) fn set_memory_limit(
        &self,
        limit: Option<u64>,
    ) -> Result<(), crate::memory::MemoryLimitError> {
        self.memory.set_limit(limit)
    }
    pub(crate) fn allocate_storage(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let mut reservation = self.memory.reserve(bytes).map_err(|capacity| {
            ExecutionError::AllocationCapacity { required: capacity.required.into(), available: capacity.available }
        })?;
        self.allocate_reserved(bytes, alignment, &mut reservation)
    }

    fn allocate_reserved(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let limits = self.device_description().limits();
        let natural_max = if limits.max_index_bits >= 64 { u64::MAX } else { (1u64 << limits.max_index_bits) - 1 };
        let maximum = limits.max_allocation_bytes.min(natural_max);
        if bytes > maximum {
            return Err(ExecutionError::AllocationCapacity { required: bytes.into(), available: maximum });
        }
        if !alignment.is_power_of_two() || alignment > limits.max_allocation_alignment {
            return Err(ExecutionError::ConstructionContradiction(format!(
                "allocation alignment {alignment} exceeds target contract {}", limits.max_allocation_alignment)));
        }
        let buffer = self.service.allocate(bytes, alignment)?;
        Ok(Allocation::new(
            fresh_allocation_identity(),
            bytes,
            reservation.take(bytes),
            Box::new(TypedStorage::<T, E> {
                service: self.service.clone(),
                buffer,
            }),
        ))
    }
}

pub(crate) fn capability_summaries<T: TargetFamily>(device: &DeviceDescription<T>) -> Vec<String> {
    registry::capabilities(T::NAME)
        .iter()
        .filter(|capability| device.supports_capability(capability.id))
        .map(|capability| format!("{}.{}", T::NAME.as_str(), capability.name))
        .collect()
}

pub(crate) fn opened_capability_summaries<T, E>(opened: &Opened<T, E>) -> Vec<String>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
{
    capability_summaries(opened.device_description())
}

pub(crate) trait Storage: Send + Sync {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError>;
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError>;
    fn as_any(&self) -> &dyn Any;
}

struct TypedStorage<T: TargetFamily, E: NativeExecutor<T>> {
    service: Arc<Service<T, E>>,
    buffer: Buffer<T, E>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> Storage for TypedStorage<T, E> {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.service.read(&self.buffer, offset, into)
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.service.write(&self.buffer, offset, bytes)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One physical allocation. Views share it, so access exclusion is allocation-wide.
pub(crate) struct Allocation {
    identity: u64,
    bytes: u64,
    _charge: MemoryCharge,
    storage: Box<dyn Storage>,
    access: Mutex<AllocationAccess>,
    access_changed: Condvar,
}

#[derive(Default)]
struct AllocationAccess {
    readers: u64,
    writer: bool,
}

impl Allocation {
    fn new(
        identity: u64,
        bytes: u64,
        charge: MemoryCharge,
        storage: Box<dyn Storage>,
    ) -> Arc<Self> {
        Arc::new(Self {
            identity,
            bytes,
            _charge: charge,
            storage,
            access: Mutex::new(AllocationAccess::default()),
            access_changed: Condvar::new(),
        })
    }
    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) fn storage(&self) -> &dyn Storage {
        &*self.storage
    }
    pub(crate) fn acquire(self: &Arc<Self>, write: bool) -> AllocationPermit {
        let mut state = self
            .access
            .lock()
            .expect("tensor allocation access lock poisoned");
        while state.writer || (write && state.readers != 0) {
            state = self
                .access_changed
                .wait(state)
                .expect("tensor allocation access lock poisoned while waiting");
        }
        if write {
            state.writer = true;
        } else {
            state.readers = state
                .readers
                .checked_add(1)
                .expect("tensor allocation reader count overflowed");
        }
        drop(state);
        AllocationPermit {
            allocation: self.clone(),
            write,
        }
    }

    /// Non-blocking acquisition for a freshly allocated, unpublished buffer.
    /// Admission uses this to assert that materialization never introduces a
    /// hidden wait after the graph transaction has claimed resources.
    pub(crate) fn try_acquire(self: &Arc<Self>, write: bool) -> Option<AllocationPermit> {
        let mut state = self
            .access
            .lock()
            .expect("tensor allocation access lock poisoned");
        if state.writer || (write && state.readers != 0) {
            return None;
        }
        if write {
            state.writer = true;
        } else {
            state.readers = state
                .readers
                .checked_add(1)
                .expect("tensor allocation reader count overflowed");
        }
        Some(AllocationPermit {
            allocation: self.clone(),
            write,
        })
    }
}

pub(crate) struct AllocationPermit {
    allocation: Arc<Allocation>,
    write: bool,
}

impl AllocationPermit {
    pub(crate) fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }
    pub(crate) fn owns(&self, allocation: &Arc<Allocation>) -> bool {
        Arc::ptr_eq(&self.allocation, allocation)
    }
    pub(crate) fn read(&self, allocation: &Arc<Allocation>, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        assert!(self.owns(allocation), "allocation read uses a different allocation's permit");
        assert!(offset.checked_add(into.len() as u64).is_some_and(|end| end <= allocation.bytes()),
            "allocation read exceeds its admitted backing");
        self.allocation.storage().read(offset, into)
    }
}

impl Drop for AllocationPermit {
    fn drop(&mut self) {
        let mut state = self
            .allocation
            .access
            .lock()
            .expect("tensor allocation access lock poisoned while releasing");
        if self.write {
            assert!(
                state.writer,
                "write permit released without an active writer"
            );
            state.writer = false;
        } else {
            state.readers = state
                .readers
                .checked_sub(1)
                .expect("read permit released without an active reader");
        }
        self.allocation.access_changed.notify_all();
    }
}

fn typed_buffer<T: TargetFamily, E: NativeExecutor<T>>(allocation: &Allocation) -> Buffer<T, E> {
    allocation.storage.as_any().downcast_ref::<TypedStorage<T, E>>()
        .unwrap_or_else(|| panic!("Tensor allocation backend invariant violated after successful WrongDevice validation"))
        .buffer.clone()
}

pub(crate) fn write_zeros(storage: &dyn Storage, byte_len: u64) -> Result<(), ExecutionError> {
    const CHUNK: usize = 1 << 20;
    let zeros = vec![0u8; CHUNK.min(usize::try_from(byte_len).unwrap_or(CHUNK))];
    let mut offset = 0u64;
    while offset < byte_len {
        let length = usize::try_from((byte_len - offset).min(CHUNK as u64)).unwrap_or(CHUNK);
        storage.write(offset, &zeros[..length])?;
        offset += length as u64;
    }
    Ok(())
}

pub(crate) struct Prepared<T: TargetFamily, E: NativeExecutor<T>> {
    identity: u64,
    device: Arc<Opened<T, E>>,
    kernel: PreparedKernel<T, E::Handle>,
    pub(crate) feedback_report: Option<FeedbackReport>,
    persistent: Arc<PersistentTable>,
}

pub(crate) fn prepare<T, E, C>(
    opened: &Arc<Opened<T, E>>,
    compiler: &C,
    native_context: &C::Context,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    public_device: &Arc<crate::api::device::DeviceInner>,
    options: PreparationOptions,
) -> Result<Arc<Prepared<T, E>>, PrepareError>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_target::NativeCompiler<T, Handle = E::Handle>,
{
    let logical = module
        .entry(entry, &bindings)
        .map_err(PrepareError::Source)?;
    let key = PreparationKey {
        module: logical.module_hash(),
        entry: logical.identity(),
        bindings: bindings
            .iter()
            .map(|(name, representation)| (name.to_owned(), representation))
            .collect(),
        policy: PolicyIdentity::of(&options.precision),
        evaluation: options.evaluation.fingerprint(),
    };
    if let Some(hit) = opened.cache().get(&key).and_then(Weak::upgrade) {
        let compatible = match &options.evaluation {
            EvaluationMethod::Analytical => true,
            EvaluationMethod::Feedback(_) => {
                use seismic_compiler::feedback::ControlledObserver;
                let environment = feedback::Observer::new(opened.clone(), public_device.clone())
                    .environment()
                    .map_err(|error| {
                        PrepareError::Preparation(
                            seismic_compiler::errors::PreparationError::Feedback(
                                seismic_compiler::feedback::FeedbackError::Observation(error),
                            ),
                        )
                    })?;
                hit.feedback_report
                    .as_ref()
                    .is_some_and(|report| report.measurement_environment == Some(environment))
            }
        };
        if compatible {
            return Ok(hit);
        }
    }
    let attributes = vec![
        key_str("seismic.module", hex(key.module.digest())),
        key_str("seismic.entry", hex(key.entry.digest())),
        key_str("seismic.backend", T::NAME.as_str()),
        key_str(
            "seismic.target.hardware",
            opened
                .device_description()
                .compatibility_identity()
                .hardware
                .clone(),
        ),
        key_str(
            "seismic.target.fingerprint",
            hex(&opened.device_description().identity().fingerprint),
        ),
        key_str("seismic.policy", hex(&key.policy.0)),
    ];
    let mut span = Timed::start("seismic.prepare", attributes.clone());
    let mut preparation_budget = PreparationBudget::default();
    if let EvaluationMethod::Feedback(feedback) = &options.evaluation {
        preparation_budget.construction_wall_time = feedback.search_time;
        preparation_budget.native_compile_wall_time = feedback.search_time;
    }
    let planning_budget = PlanningBudget::default();
    let (kernel, feedback_report) = match options.evaluation {
        EvaluationMethod::Analytical => {
            let analytical = opened.analytical().map_err(|error| {
                PrepareError::Preparation(
                    seismic_compiler::errors::PreparationError::NativeCompilation(
                        seismic_target::NativeCompilationError::ToolchainFailure(error.to_string()),
                    ),
                )
            })?;
            let kernel = prepare_analytically(
                logical,
                analytical,
                opened.compiler_registry,
                compiler,
                native_context,
                &options.precision,
                &preparation_budget,
                &planning_budget,
            )
            .map_err(PrepareError::Preparation)?;
            (kernel, None)
        }
        EvaluationMethod::Feedback(feedback_options) => {
            let observer = feedback::Observer::new(opened.clone(), public_device.clone());
            let (campaign, kernel) = FeedbackPreparation::start(
                logical,
                opened.device_description(),
                opened.compiler_registry,
                compiler,
                native_context,
                &options.precision,
                &preparation_budget,
                &planning_budget,
                observer,
                feedback_options,
            )
            .map_err(PrepareError::Preparation)?;
            (kernel, Some(campaign.report().clone()))
        }
    };
    span.attribute(key_u64("seismic.variants", kernel.variants().len() as u64));
    let planning_report = kernel.planning_report();
    span.attribute(key_str(
        "seismic.planning.optimization_completion",
        match &planning_report.optimization {
            OptimizationCompletion::Complete => "complete",
            OptimizationCompletion::Limited(_) => "limited",
        },
    ));
    if let OptimizationCompletion::Limited(limit) = &planning_report.optimization {
        span.attribute(key_str(
            "seismic.planning.optimization_limit",
            format!("{limit:?}"),
        ));
    }
    span.attribute(key_u64(
        "seismic.planning.solver_work_units",
        planning_report.budget.solver_work_units,
    ));
    span.attribute(key_u64(
        "seismic.planning.solver_elapsed_ms",
        planning_report.budget.solver_elapsed_ms,
    ));
    span.attribute(key_u64(
        "seismic.planning.solver_memory_bytes",
        planning_report.budget.solver_memory_bytes,
    ));
    span.attribute(key_u64(
        "seismic.planning.optimized_assignments",
        planning_report.budget.optimized_assignments,
    ));
    span.attribute(key_u64(
        "seismic.planning.executable_variants",
        planning_report.budget.executable_variants,
    ));
    span.attribute(key_u64(
        "seismic.planning.retained_metadata_bytes",
        planning_report.budget.retained_metadata_bytes,
    ));
    telemetry::record_preparation(span.elapsed_ms(), &attributes);
    let prepared = Arc::new(Prepared {
        identity: NEXT_PREPARED.fetch_add(1, Ordering::Relaxed),
        device: opened.clone(),
        kernel,
        feedback_report,
        persistent: Arc::new(PersistentTable::new()),
    });
    opened.cache().insert(key, Arc::downgrade(&prepared));
    Ok(prepared)
}

/// Explicit mutable search ownership, separate from every returned kernel.
pub(crate) struct FeedbackCampaign<'a, T, E, C>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_target::NativeCompiler<T, Handle = E::Handle>,
{
    opened: Arc<Opened<T, E>>,
    public_device: Arc<crate::api::device::DeviceInner>,
    campaign: FeedbackPreparation<'a, T, C, feedback::Observer<T, E>>,
}

impl<'a, T, E, C> FeedbackCampaign<'a, T, E, C>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_target::NativeCompiler<T, Handle = E::Handle>,
{
    pub(crate) fn start(
        opened: &'a Arc<Opened<T, E>>,
        compiler: &'a C,
        native_context: &'a C::Context,
        module: &CheckedModule,
        entry: EntryId,
        bindings: ElementBindings,
        public_device: &Arc<crate::api::device::DeviceInner>,
        precision: PrecisionPolicy,
        options: seismic_compiler::feedback::FeedbackOptions,
    ) -> Result<(Self, Arc<PreparedHandle<T, E>>), PrepareError> {
        let logical = module
            .entry(entry, &bindings)
            .map_err(PrepareError::Source)?;
        let budget = PreparationBudget {
            construction_wall_time: options.search_time,
            native_compile_wall_time: options.search_time,
            ..Default::default()
        };
        let observer = feedback::Observer::new(opened.clone(), public_device.clone());
        let (campaign, kernel) = FeedbackPreparation::start(
            logical,
            opened.device_description(),
            opened.compiler_registry,
            compiler,
            native_context,
            &precision,
            &budget,
            &PlanningBudget::default(),
            observer,
            options,
        )
        .map_err(PrepareError::Preparation)?;
        let preparation = Self {
            opened: opened.clone(),
            public_device: public_device.clone(),
            campaign,
        };
        let kernel = preparation.snapshot(kernel);
        Ok((preparation, kernel))
    }

    pub(crate) fn continue_for(
        &mut self,
        additional: std::time::Duration,
    ) -> Result<Arc<PreparedHandle<T, E>>, PrepareError> {
        let kernel = self
            .campaign
            .continue_for(additional)
            .map_err(PrepareError::Preparation)?;
        Ok(self.snapshot(kernel))
    }

    pub(crate) fn report(&self) -> &seismic_compiler::feedback::FeedbackReport {
        self.campaign.report()
    }

    fn snapshot(&self, kernel: PreparedKernel<T, E::Handle>) -> Arc<PreparedHandle<T, E>> {
        Arc::new(PreparedHandle {
            prepared: Arc::new(Prepared {
                identity: NEXT_PREPARED.fetch_add(1, Ordering::Relaxed),
                device: self.opened.clone(),
                kernel,
                feedback_report: Some(self.report().clone()),
                persistent: Arc::new(PersistentTable::new()),
            }),
            device: self.public_device.clone(),
        })
    }
}

#[cfg(target_os = "macos")]
enum NativeResult {
    Tensor {
        representation: RepresentationId,
        axes: Vec<CompiledNat>,
    },
    Scalar(seismic_lang::types::DType),
    Index,
    Range,
}

#[derive(Clone)]
pub(crate) struct NativeTensorSpec {
    pub(crate) representation: RepresentationId,
    pub(crate) extents: Vec<u64>,
    pub(crate) strides: Vec<u64>,
    pub(crate) byte_len: u64,
    pub(crate) alignment: u64,
}

/// A direct Metal entry point. It deliberately has no `PreparedKernel`,
/// plan space, precision policy, portfolio, or workflow representation.
#[cfg(target_os = "macos")]
pub(crate) struct NativePreparedMetal {
    opened: Arc<Opened<seismic_metal::Metal, seismic_metal::MetalExecutor>>,
    public_device: Arc<crate::api::device::DeviceInner>,
    logical: LogicalEntry,
    invocation: InvocationContract,
    pipeline: seismic_metal::DirectPipeline,
    definition: NativeDefinition,
    results: Vec<NativeResult>,
    invocation_storage: Mutex<NativeInvocationStorage>,
    invocation_workspace_bytes: u64,
}

#[cfg(target_os = "macos")]
struct NativeInvocationStorage {
    words: Arc<Allocation>,
    scalars: Arc<Allocation>,
}

#[cfg(target_os = "macos")]
pub(crate) struct NativeBoundCall {
    kernel: Arc<NativePreparedMetal>,
    args: EncodedArgs,
    tensor_results: Vec<Arc<TensorInner>>,
    word_bytes: Vec<u8>,
    threadgroups: [u64; 3],
    threads: [u64; 3],
}

#[cfg(target_os = "macos")]
impl NativeBoundCall {
    pub(crate) fn run(self) -> Result<DecodedResults, CallError> {
        let kernel = self.kernel.clone();
        kernel.execute_bound(self)
    }
}

/// Run already-bound graph nodes in one ordered Metal command buffer. The
/// union permit excludes host mutation for the complete dependency chain.
#[cfg(target_os = "macos")]
pub(crate) fn run_native_graph_batch(calls: Vec<NativeBoundCall>) -> Result<(), CallError> {
    let Some(first) = calls.first() else {
        return Ok(());
    };
    let mut kernels = BTreeMap::new();
    let mut access = Vec::new();
    for call in &calls {
        kernels.insert(Arc::as_ptr(&call.kernel) as usize, call.kernel.clone());
        access.extend(collect_native_access(call.kernel.logical.schema(), &call.args));
        access.extend(
            call.tensor_results
                .iter()
                .map(|tensor| (tensor.allocation().clone(), true)),
        );
    }
    // The standalone native route also locks invocation storage before tensor
    // access. Lock each distinct prepared kernel once, in a stable order.
    let guards = kernels
        .values()
        .map(|kernel| {
            kernel
                .invocation_storage
                .lock()
                .expect("native invocation workspace lock poisoned")
        })
        .collect::<Vec<_>>();
    let scalar_allocations = kernels
        .keys()
        .copied()
        .zip(guards.iter().map(|guard| guard.scalars.clone()))
        .collect::<BTreeMap<_, _>>();
    let _permits = acquire_access(&access);
    let mut batch = seismic_metal::DirectBatch::new(&first.kernel.opened.service)
        .map_err(CallError::Execution)?;
    for call in &calls {
        let mut buffers = Vec::new();
        for (ordinal, parameter) in call.kernel.logical.schema().parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = call
                    .args
                    .tensor(ordinal)
                    .expect("validated native graph tensor argument disappeared");
                buffers.push((
                    typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(tensor.allocation()),
                    tensor.byte_offset(),
                ));
            }
        }
        for tensor in &call.tensor_results {
            buffers.push((
                typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(tensor.allocation()),
                tensor.byte_offset(),
            ));
        }
        let borrowed = buffers
            .iter()
            .map(|(buffer, offset)| (buffer, *offset))
            .collect::<Vec<_>>();
        let scalar = scalar_allocations
            .get(&(Arc::as_ptr(&call.kernel) as usize))
            .expect("native graph scalar workspace is absent");
        let scalar_buffer = typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(scalar);
        batch
            .encode(
                &call.kernel.pipeline,
                &borrowed,
                &call.word_bytes,
                &scalar_buffer,
                call.threadgroups,
                call.threads,
            )
            .map_err(CallError::Execution)?;
    }
    batch.commit_wait().map_err(CallError::Execution)
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_native_metal(
    opened: &Arc<Opened<seismic_metal::Metal, seismic_metal::MetalExecutor>>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    public_device: &Arc<crate::api::device::DeviceInner>,
    definition: NativeDefinition,
) -> Result<Arc<NativePreparedMetal>, PrepareError> {
    let logical = module
        .entry(entry, &bindings)
        .map_err(PrepareError::Source)?;
    let invocation = InvocationContract::compile_entry(&logical);
    let results = logical
        .schema()
        .results()
        .iter()
        .map(|result| match &result.kind {
            ResultKind::Tensor {
                representation,
                axes,
            } => NativeResult::Tensor {
                representation: *representation,
                axes: axes
                    .iter()
                    .map(|axis| logical.arena().compile_nat(*axis))
                    .collect(),
            },
            ResultKind::Scalar(dtype) => NativeResult::Scalar(*dtype),
            ResultKind::Index { .. } => NativeResult::Index,
            ResultKind::Range { .. } => NativeResult::Range,
        })
        .collect();
    let word_count = logical.schema().dimensions().len()
        + logical.schema().parameters().iter().map(|parameter| match &parameter.kind {
            ParameterKind::Tensor { axes, .. } => axes.len() * 2,
            ParameterKind::Range { .. } => 2,
            ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => 1,
        }).sum::<usize>()
        + logical.schema().results().iter().map(|result| match &result.kind {
            ResultKind::Tensor { axes, .. } => axes.len() * 2,
            ResultKind::Range { .. } | ResultKind::Scalar(_) | ResultKind::Index { .. } => 0,
        }).sum::<usize>();
    let scalar_count = logical.schema().results().iter().map(|result| match &result.kind {
        ResultKind::Range { .. } => 2,
        ResultKind::Scalar(_) | ResultKind::Index { .. } => 1,
        ResultKind::Tensor { .. } => 0,
    }).sum::<usize>();
    let word_bytes = u64::try_from(word_count).ok().and_then(|count| count.checked_mul(8)).ok_or_else(|| {
        PrepareError::Preparation(seismic_compiler::errors::PreparationError::NativeWorkspaceAllocation("native argument word count overflow".into()))
    })?.max(1);
    let scalar_bytes = u64::try_from(scalar_count).ok().and_then(|count| count.checked_mul(8)).ok_or_else(|| {
        PrepareError::Preparation(seismic_compiler::errors::PreparationError::NativeWorkspaceAllocation("native scalar result count overflow".into()))
    })?.max(1);
    let source = render_native_source(logical.schema(), &bindings, &definition.source);
    let pipeline =
        seismic_metal::DirectPipeline::compile(&opened.service, &source, &definition.entry)
            .map_err(|error| {
                PrepareError::Preparation(
                    seismic_compiler::errors::PreparationError::NativeCompilation(error),
                )
            })?;
    let allocation_error = |error: ExecutionError| PrepareError::Preparation(
        seismic_compiler::errors::PreparationError::NativeWorkspaceAllocation(error.to_string())
    );
    let words = opened.allocate_storage(word_bytes, 8).map_err(allocation_error)?;
    let scalars = opened.allocate_storage(scalar_bytes, 8).map_err(allocation_error)?;
    Ok(Arc::new(NativePreparedMetal {
        opened: opened.clone(),
        public_device: public_device.clone(),
        logical,
        invocation,
        pipeline,
        definition,
        results,
        invocation_storage: Mutex::new(NativeInvocationStorage { words, scalars }),
        invocation_workspace_bytes: word_bytes + scalar_bytes,
    }))
}

#[cfg(target_os = "macos")]
impl NativePreparedMetal {
    pub(crate) fn call_with_commit(
        &self,
        args: EncodedArgs,
        commit: impl FnOnce(),
    ) -> Result<DecodedResults, CallError> {
        let schema = self.logical.schema();
        let arguments = args.values();
        let values = validate_invocation(&self.invocation, self.opened.identity(), &arguments)
            .map_err(CallError::Invocation)?;

        let mut tensor_results = Vec::new();
        for result in &self.results {
            match result {
                NativeResult::Tensor {
                    representation,
                    axes,
                } => {
                    let extents = axes
                        .iter()
                        .map(|axis| native_eval_compiled(axis, &values))
                        .collect::<Result<Vec<_>, _>>()?;
                    let tensor = Arc::new(
                        TensorInner::zeros(&self.public_device, *representation, &extents)
                            .map_err(native_tensor_error)?,
                    );
                    tensor_results.push(tensor);
                }
                NativeResult::Scalar(_) | NativeResult::Index | NativeResult::Range => {
                    // Filled from the scalar-result buffer after dispatch.
                }
            }
        }

        let words = native_words(schema, &args, &tensor_results, &values)?;
        let word_bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let word_allocation = self
            .opened
            .allocate_storage(word_bytes.len().max(1) as u64, 8)
            .map_err(CallError::Execution)?;
        word_allocation
            .storage()
            .write(0, &word_bytes)
            .map_err(CallError::Execution)?;
        let scalar_words = self
            .results
            .iter()
            .map(|result| match result {
                NativeResult::Range => 2usize,
                NativeResult::Scalar(_) | NativeResult::Index => 1,
                NativeResult::Tensor { .. } => 0,
            })
            .sum::<usize>();
        let scalar_allocation = self
            .opened
            .allocate_storage((scalar_words * 8).max(1) as u64, 8)
            .map_err(CallError::Execution)?;
        write_zeros(
            scalar_allocation.storage(),
            (scalar_words * 8).max(1) as u64,
        )
        .map_err(CallError::Execution)?;

        let mut access = collect_native_access(schema, &args);
        access.extend(
            tensor_results
                .iter()
                .map(|tensor| (tensor.allocation().clone(), true)),
        );
        let _permits = acquire_access(&access);

        let mut buffers = Vec::new();
        for (ordinal, parameter) in schema.parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = args
                    .tensor(ordinal)
                    .expect("validated tensor argument disappeared");
                buffers.push((
                    typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(
                        tensor.allocation(),
                    ),
                    tensor.byte_offset(),
                ));
            }
        }
        for tensor in &tensor_results {
            buffers.push((
                typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(
                    tensor.allocation(),
                ),
                tensor.byte_offset(),
            ));
        }
        buffers.push((
            typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(&word_allocation),
            0,
        ));
        buffers.push((
            typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(&scalar_allocation),
            0,
        ));
        let borrowed = buffers
            .iter()
            .map(|(buffer, offset)| (buffer, *offset))
            .collect::<Vec<_>>();
        let threadgroups = self
            .definition
            .threadgroups
            .each_ref()
            .map(|expression| native_eval_launch(expression, schema, &values));
        let threads = self
            .definition
            .threads_per_threadgroup
            .each_ref()
            .map(|expression| native_eval_launch(expression, schema, &values));
        let threadgroups = collect_native_geometry(threadgroups)?;
        let threads = collect_native_geometry(threads)?;
        commit();
        self.pipeline
            .dispatch(&self.opened.service, &borrowed, threadgroups, threads)
            .map_err(CallError::Execution)?;

        let mut scalar_bytes = vec![0u8; scalar_words * 8];
        scalar_allocation
            .storage()
            .read(0, &mut scalar_bytes)
            .map_err(CallError::Execution)?;
        let mut scalar_offset = 0usize;
        let mut tensors = tensor_results.into_iter();
        let mut final_values = Vec::with_capacity(self.results.len());
        for result in &self.results {
            match result {
                NativeResult::Tensor { .. } => final_values.push(DecodedValue::Tensor(
                    tensors.next().expect("native result tensor count changed"),
                )),
                NativeResult::Scalar(dtype) => {
                    let word = read_native_word(&scalar_bytes, scalar_offset);
                    scalar_offset += 1;
                    final_values.push(DecodedValue::Scalar(native_scalar(*dtype, word)));
                }
                NativeResult::Index => {
                    let word = read_native_word(&scalar_bytes, scalar_offset);
                    scalar_offset += 1;
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Index(word.into())));
                }
                NativeResult::Range => {
                    let start = read_native_word(&scalar_bytes, scalar_offset);
                    let end = read_native_word(&scalar_bytes, scalar_offset + 1);
                    scalar_offset += 2;
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Range { start: start.into(), end: end.into() }));
                }
            }
        }
        Ok(DecodedResults::new(final_values))
    }
}

#[cfg(target_os = "macos")]
impl NativePreparedMetal {
    pub(crate) fn validate_graph_batch(&self, device: DeviceIdentity) -> Result<(), CallError> {
        if self.opened.identity() != device {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::NativeGraphSlotMismatch,
            ));
        }
        if self
            .results
            .iter()
            .any(|result| !matches!(result, NativeResult::Tensor { .. }))
        {
            return Err(CallError::Workflow(
                crate::api::WorkflowError::HostBoundaryRequired,
            ));
        }
        let words = self
            .invocation_storage
            .lock()
            .expect("native invocation workspace lock poisoned")
            .words
            .bytes();
        if words > 4096 {
            return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                "native graph ABI words exceed Metal setBytes limit".into(),
            )));
        }
        Ok(())
    }

    pub(crate) fn result_count(&self) -> u32 {
        u32::try_from(self.results.len()).expect("native result ordinal space exhausted")
    }

    pub(crate) fn tensor_parameter_spec(
        &self,
        name: &str,
        dimensions: &[(&str, u64)],
    ) -> Result<NativeTensorSpec, CallError> {
        let schema = self.logical.schema();
        let parameter = schema
            .parameters()
            .iter()
            .find(|parameter| parameter.name == name)
            .ok_or_else(|| {
                CallError::Execution(ExecutionError::SubmissionFailed(format!(
                    "checked native entry has no tensor parameter `{name}`"
                )))
            })?;
        let ParameterKind::Tensor {
            representation,
            axes,
            ..
        } = &parameter.kind
        else {
            return Err(CallError::Execution(ExecutionError::SubmissionFailed(
                format!("checked native parameter `{name}` is not a tensor"),
            )));
        };
        let mut values = InvocationValues::new();
        for dimension in schema.dimensions() {
            let value = dimensions
                .iter()
                .find(|(candidate, _)| *candidate == dimension.name)
                .map(|(_, value)| *value)
                .ok_or_else(|| {
                    CallError::Execution(ExecutionError::SubmissionFailed(format!(
                        "native graph omitted dimension `{}` for `{name}`",
                        dimension.name
                    )))
                })?;
            values.bind(dimension.symbol, SymbolValue::Nat(value.into()));
        }
        let extents = axes
            .iter()
            .map(|axis| native_eval_compiled(&self.logical.arena().compile_nat(*axis), &values))
            .collect::<Result<Vec<_>, _>>()?;
        let layout =
            crate::layout::canonical(*representation, &extents).map_err(CallError::Execution)?;
        Ok(NativeTensorSpec {
            representation: *representation,
            extents,
            strides: layout.strides,
            byte_len: layout.byte_len,
            alignment: layout.alignment,
        })
    }
    pub(crate) fn invocation_workspace_bytes(&self) -> u64 {
        self.invocation_workspace_bytes
    }

    /// Describe a native node without allocating or dispatching it. The
    /// checked entry schema alone determines every result representation and
    /// extent. Graph planning feeds descriptors, including virtual result
    /// edges, through the same invocation contract as a direct call.
    pub(crate) fn describe_results(
        &self,
        arguments: &[ArgumentValue],
    ) -> Result<Vec<Option<NativeTensorSpec>>, CallError> {
        let values = validate_invocation(
            &self.invocation,
            self.opened.identity(),
            arguments,
        )
        .map_err(CallError::Invocation)?;
        collect_native_geometry(
            self.definition
                .threadgroups
                .each_ref()
                .map(|expr| native_eval_launch(expr, self.logical.schema(), &values)),
        )?;
        collect_native_geometry(
            self.definition
                .threads_per_threadgroup
                .each_ref()
                .map(|expr| native_eval_launch(expr, self.logical.schema(), &values)),
        )?;
        let mut results = Vec::with_capacity(self.results.len());
        for result in &self.results {
            match result {
                NativeResult::Tensor {
                    representation,
                    axes,
                } => {
                    let extents = axes
                        .iter()
                        .map(|axis| native_eval_compiled(axis, &values))
                        .collect::<Result<Vec<_>, _>>()?;
                    let layout = crate::layout::canonical(*representation, &extents)
                        .map_err(CallError::Execution)?;
                    results.push(Some(NativeTensorSpec {
                        representation: *representation,
                        extents,
                        strides: layout.strides,
                        byte_len: layout.byte_len,
                        alignment: layout.alignment,
                    }));
                }
                NativeResult::Scalar(_) | NativeResult::Index | NativeResult::Range => {
                    results.push(None);
                }
            }
        }
        Ok(results)
    }

    /// Check a fully attached graph node before it becomes executable. No
    /// command is encoded or submitted here; a later native call sees these
    /// same immutable tensor descriptors and result storage.
    pub(crate) fn validate_bound(
        &self,
        args: &EncodedArgs,
        outputs: &[Arc<TensorInner>],
    ) -> Result<(), CallError> {
        let arguments = args.values();
        let values = validate_invocation(
            &self.invocation,
            self.opened.identity(),
            &arguments,
        )
        .map_err(CallError::Invocation)?;
        let expected = self.results.iter().filter_map(|result| match result {
            NativeResult::Tensor {
                representation,
                axes,
            } => Some((representation, axes)),
            _ => None,
        });
        let mut prior = Vec::with_capacity(outputs.len());
        let mut count = 0;
        for (representation, axes) in expected {
            let extents = axes
                .iter()
                .map(|axis| native_eval_compiled(axis, &values))
                .collect::<Result<Vec<_>, _>>()?;
            let tensor = outputs
                .get(count)
                .ok_or(CallError::Output(OutputError::Count {
                    expected: count + 1,
                    actual: outputs.len(),
                }))?;
            validate_native_output(
                count,
                tensor,
                self.opened.identity(),
                *representation,
                &extents,
                args,
                &prior,
            )?;
            prior.push(tensor.clone());
            count += 1;
        }
        if count != outputs.len() {
            return Err(CallError::Output(OutputError::Count {
                expected: count,
                actual: outputs.len(),
            }));
        }
        native_words(self.logical.schema(), args, outputs, &values)?;
        Ok(())
    }

    /// Produce a call whose ABI words, launch geometry, tensor descriptors,
    /// and aliases have all been checked before the execution boundary.
    pub(crate) fn bind(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Vec<Arc<TensorInner>>,
    ) -> Result<NativeBoundCall, CallError> {
        self.validate_bound(&args, &outputs)?;
        let values = validate_invocation(
            &self.invocation,
            self.opened.identity(),
            &args.values(),
        )
        .map_err(CallError::Invocation)?;
        self.seal_bound(args, outputs, &values)
    }

    fn seal_bound(
        self: &Arc<Self>,
        args: EncodedArgs,
        tensor_results: Vec<Arc<TensorInner>>,
        values: &InvocationValues,
    ) -> Result<NativeBoundCall, CallError> {
        let schema = self.logical.schema();
        let words = native_words(schema, &args, &tensor_results, values)?;
        let word_bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let threadgroups = collect_native_geometry(
            self.definition
                .threadgroups
                .each_ref()
                .map(|expression| native_eval_launch(expression, schema, values)),
        )?;
        let threads = collect_native_geometry(
            self.definition
                .threads_per_threadgroup
                .each_ref()
                .map(|expression| native_eval_launch(expression, schema, values)),
        )?;
        Ok(NativeBoundCall {
            kernel: self.clone(),
            args,
            tensor_results,
            word_bytes,
            threadgroups,
            threads,
        })
    }

    pub(crate) fn call(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        self.call_with_outputs(args, None)
    }

    pub(crate) fn call_into(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: EncodedOutputs,
    ) -> Result<DecodedResults, CallError> {
        self.call_with_outputs(args, Some(outputs))
    }

    fn call_with_outputs(
        self: &Arc<Self>,
        args: EncodedArgs,
        outputs: Option<EncodedOutputs>,
    ) -> Result<DecodedResults, CallError> {
        let schema = self.logical.schema();
        let arguments = args.values();
        let values =
            validate_invocation(&self.invocation, self.opened.identity(), &arguments)
                .map_err(CallError::Invocation)?;

        let supplied = outputs.map(EncodedOutputs::into_tensors);
        let expected_count = self
            .results
            .iter()
            .filter(|result| matches!(result, NativeResult::Tensor { .. }))
            .count();
        if let Some(outputs) = &supplied {
            if outputs.len() != expected_count {
                return Err(CallError::Output(OutputError::Count {
                    expected: expected_count,
                    actual: outputs.len(),
                }));
            }
        }
        let mut tensor_results = Vec::with_capacity(expected_count);
        for result in &self.results {
            match result {
                NativeResult::Tensor {
                    representation,
                    axes,
                } => {
                    let extents = axes
                        .iter()
                        .map(|axis| native_eval_compiled(axis, &values))
                        .collect::<Result<Vec<_>, _>>()?;
                    let tensor = if let Some(outputs) = &supplied {
                        let index = tensor_results.len();
                        let tensor = outputs[index].clone();
                        validate_native_output(
                            index,
                            &tensor,
                            self.opened.identity(),
                            *representation,
                            &extents,
                            &args,
                            &tensor_results,
                        )?;
                        tensor
                    } else {
                        Arc::new(
                            TensorInner::zeros(&self.public_device, *representation, &extents)
                                .map_err(native_tensor_error)?,
                        )
                    };
                    tensor_results.push(tensor);
                }
                NativeResult::Scalar(_) | NativeResult::Index | NativeResult::Range => {
                    // Filled from the scalar-result buffer after dispatch.
                }
            }
        }

        self.seal_bound(args, tensor_results, &values)?.run()
    }
    fn execute_bound(&self, bound: NativeBoundCall) -> Result<DecodedResults, CallError> {
        let NativeBoundCall {
            kernel: _,
            args,
            tensor_results,
            word_bytes,
            threadgroups,
            threads,
        } = bound;
        let schema = self.logical.schema();
        // This lock covers upload, dispatch and readback. The prepared entry
        // owns its fixed-size native ABI storage for its entire callable life.
        let invocation = self
            .invocation_storage
            .lock()
            .expect("native invocation workspace lock poisoned");
        let word_allocation = &invocation.words;
        word_allocation
            .storage()
            .write(0, &word_bytes)
            .map_err(CallError::Execution)?;
        let scalar_words = self
            .results
            .iter()
            .map(|result| match result {
                NativeResult::Range => 2usize,
                NativeResult::Scalar(_) | NativeResult::Index => 1,
                NativeResult::Tensor { .. } => 0,
            })
            .sum::<usize>();
        let scalar_allocation = &invocation.scalars;
        write_zeros(
            scalar_allocation.storage(),
            (scalar_words * 8).max(1) as u64,
        )
        .map_err(CallError::Execution)?;

        let mut access = collect_native_access(schema, &args);
        access.extend(
            tensor_results
                .iter()
                .map(|tensor| (tensor.allocation().clone(), true)),
        );
        let _permits = acquire_access(&access);

        let mut buffers = Vec::new();
        for (ordinal, parameter) in schema.parameters().iter().enumerate() {
            if matches!(parameter.kind, ParameterKind::Tensor { .. }) {
                let tensor = args
                    .tensor(ordinal)
                    .expect("validated tensor argument disappeared");
                buffers.push((
                    typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(tensor.allocation()),
                    tensor.byte_offset(),
                ));
            }
        }
        for tensor in &tensor_results {
            buffers.push((
                typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(tensor.allocation()),
                tensor.byte_offset(),
            ));
        }
        buffers.push((typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(&word_allocation), 0));
        buffers.push((typed_buffer::<seismic_metal::Metal, seismic_metal::MetalExecutor>(&scalar_allocation), 0));
        let borrowed = buffers
            .iter()
            .map(|(buffer, offset)| (buffer, *offset))
            .collect::<Vec<_>>();
        self.pipeline
            .dispatch(&self.opened.service, &borrowed, threadgroups, threads)
            .map_err(CallError::Execution)?;

        let mut scalar_bytes = vec![0u8; scalar_words * 8];
        scalar_allocation
            .storage()
            .read(0, &mut scalar_bytes)
            .map_err(CallError::Execution)?;
        let mut scalar_offset = 0usize;
        let mut tensors = tensor_results.into_iter();
        let mut final_values = Vec::with_capacity(self.results.len());
        for result in &self.results {
            match result {
                NativeResult::Tensor { .. } => final_values.push(DecodedValue::Tensor(
                    tensors.next().expect("native result tensor count changed"),
                )),
                NativeResult::Scalar(dtype) => {
                    let word = read_native_word(&scalar_bytes, scalar_offset);
                    scalar_offset += 1;
                    final_values.push(DecodedValue::Scalar(native_scalar(*dtype, word)));
                }
                NativeResult::Index => {
                    let word = read_native_word(&scalar_bytes, scalar_offset);
                    scalar_offset += 1;
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Index(word.into())));
                }
                NativeResult::Range => {
                    let start = read_native_word(&scalar_bytes, scalar_offset);
                    let end = read_native_word(&scalar_bytes, scalar_offset + 1);
                    scalar_offset += 2;
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Range { start: start.into(), end: end.into() }));
                }
            }
        }
        Ok(DecodedResults::new(final_values))
    }
}

#[cfg(target_os = "macos")]
fn validate_native_output(
    result: usize,
    tensor: &Arc<TensorInner>,
    device: DeviceIdentity,
    representation: RepresentationId,
    extents: &[u64],
    args: &EncodedArgs,
    prior_outputs: &[Arc<TensorInner>],
) -> Result<(), CallError> {
    let descriptor = tensor.descriptor();
    if descriptor.device != device {
        return Err(CallError::Output(OutputError::WrongDevice { result }));
    }
    if descriptor.representation != representation {
        return Err(CallError::Output(OutputError::WrongRepresentation {
            result,
        }));
    }
    if descriptor.extents.len() != extents.len() {
        return Err(CallError::Output(OutputError::ShapeMismatch {
            result,
            axis: descriptor.extents.len().min(extents.len()),
        }));
    }
    for (axis, (actual, expected)) in descriptor.extents.iter().zip(extents).enumerate() {
        if actual != expected {
            return Err(CallError::Output(OutputError::ShapeMismatch {
                result,
                axis,
            }));
        }
    }
    let layout = crate::layout::canonical(representation, extents).map_err(CallError::Execution)?;
    if descriptor.strides != layout.strides || descriptor.byte_len != layout.byte_len {
        return Err(CallError::Output(OutputError::NoncanonicalLayout {
            result,
        }));
    }
    let overlaps = |other: &Arc<TensorInner>| {
        let other = other.descriptor();
        descriptor.allocation == other.allocation
            && descriptor.byte_offset < other.byte_offset.saturating_add(other.byte_len)
            && other.byte_offset < descriptor.byte_offset.saturating_add(descriptor.byte_len)
    };
    if args.tensors().flatten().any(|other| overlaps(other))
        || prior_outputs.iter().any(|other| overlaps(other))
    {
        return Err(CallError::Output(OutputError::IllegalAliasing { result }));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn native_tensor_error(error: crate::api::TensorError) -> CallError {
    match error {
        crate::api::TensorError::Execution(error) => CallError::Execution(error),
        other => CallError::Execution(ExecutionError::AllocationFailed(other.to_string())),
    }
}

#[cfg(target_os = "macos")]
fn native_eval_compiled(
    expression: &CompiledNat,
    values: &InvocationValues,
) -> Result<u64, CallError> {
    expression.evaluate_u64(values).map_err(|error| {
        CallError::Execution(ExecutionError::SubmissionFailed(format!(
            "native ABI expression failed after invocation validation: {error:?}"
        )))
    })
}

#[cfg(target_os = "macos")]
fn native_eval_launch(
    expression: &NativeExpr,
    schema: &CallSchema,
    values: &InvocationValues,
) -> Result<u64, CallError> {
    let binary = |left: &NativeExpr, right: &NativeExpr, operation: fn(u64, u64) -> Option<u64>| {
        let left = native_eval_launch(left, schema, values)?;
        let right = native_eval_launch(right, schema, values)?;
        operation(left, right).ok_or_else(|| {
            CallError::Execution(ExecutionError::SubmissionFailed(
                "native launch expression overflowed or divided by zero".to_owned(),
            ))
        })
    };
    match expression {
        NativeExpr::Constant(value) => Ok(*value),
        NativeExpr::Dimension(name) => {
            let dimension = schema
                .dimensions()
                .iter()
                .find(|dimension| dimension.name == *name)
                .expect("checked native launch expression names an absent dimension");
            match values.get(dimension.symbol) {
                Some(SymbolValue::Nat(value)) => native_symbol(SymbolValue::Nat(value)),
                _ => panic!("validated invocation omitted a native launch dimension"),
            }
        }
        NativeExpr::Add(left, right) => binary(left, right, u64::checked_add),
        NativeExpr::Sub(left, right) => binary(left, right, u64::checked_sub),
        NativeExpr::Mul(left, right) => binary(left, right, u64::checked_mul),
        NativeExpr::Div(left, right) => binary(left, right, u64::checked_div),
        NativeExpr::Rem(left, right) => binary(left, right, u64::checked_rem),
        NativeExpr::CeilDiv(left, right) => {
            let left = native_eval_launch(left, schema, values)?;
            let right = native_eval_launch(right, schema, values)?;
            left.checked_add(right.saturating_sub(1))
                .and_then(|value| value.checked_div(right))
                .ok_or_else(|| {
                    CallError::Execution(ExecutionError::SubmissionFailed(
                        "native ceil_div launch expression overflowed or divided by zero"
                            .to_owned(),
                    ))
                })
        }
    }
}

#[cfg(target_os = "macos")]
fn collect_native_geometry(values: [Result<u64, CallError>; 3]) -> Result<[u64; 3], CallError> {
    let [x, y, z] = values;
    Ok([x?, y?, z?])
}

#[cfg(target_os = "macos")]
fn native_words(
    schema: &CallSchema,
    args: &EncodedArgs,
    results: &[Arc<TensorInner>],
    values: &InvocationValues,
) -> Result<Vec<u64>, CallError> {
    let mut words = Vec::new();
    for dimension in schema.dimensions() {
        match values.get(dimension.symbol) {
            Some(SymbolValue::Nat(value)) => words.push(native_symbol(SymbolValue::Nat(value))?),
            _ => panic!("validated invocation omitted a native ABI dimension"),
        }
    }
    for (ordinal, parameter) in schema.parameters().iter().enumerate() {
        match &parameter.kind {
            ParameterKind::Tensor { .. } => {
                let tensor = args.tensor(ordinal).expect("validated tensor disappeared");
                words.extend_from_slice(tensor.extents());
                words.extend_from_slice(tensor.strides());
            }
            ParameterKind::Scalar { symbol, .. } | ParameterKind::Index { symbol, .. } => {
                words.push(native_symbol(
                    values.get(*symbol).expect("validated scalar disappeared"),
                )?);
            }
            ParameterKind::Range { start, end, .. } => {
                words.push(native_symbol(
                    values
                        .get(*start)
                        .expect("validated range start disappeared"),
                )?);
                words.push(native_symbol(
                    values.get(*end).expect("validated range end disappeared"),
                )?);
            }
        }
    }
    for tensor in results {
        words.extend_from_slice(tensor.extents());
        words.extend_from_slice(tensor.strides());
    }
    Ok(words)
}

#[cfg(target_os = "macos")]
fn native_symbol(value: SymbolValue) -> Result<u64, CallError> {
    value.try_word64().map_err(|error| CallError::Execution(ExecutionError::ConstructionContradiction(
        format!("native ABI quantity does not fit its word: {error:?}"))))
}

#[cfg(target_os = "macos")]
fn native_scalar(dtype: seismic_lang::types::DType, value: u64) -> ArgumentValue {
    match dtype {
        seismic_lang::types::DType::F32 => ArgumentValue::F32(f32::from_bits(value as u32)),
        seismic_lang::types::DType::F16 => ArgumentValue::F16(value as u16),
        seismic_lang::types::DType::BF16 => ArgumentValue::BF16(value as u16),
        seismic_lang::types::DType::I32 => ArgumentValue::I32(value as u32 as i32),
        seismic_lang::types::DType::U32 => ArgumentValue::U32(value as u32),
        seismic_lang::types::DType::Bool => ArgumentValue::Bool(value != 0),
    }
}

#[cfg(target_os = "macos")]
fn read_native_word(bytes: &[u8], word: usize) -> u64 {
    let start = word * 8;
    u64::from_le_bytes(
        bytes[start..start + 8]
            .try_into()
            .expect("native scalar word"),
    )
}

fn render_native_source(schema: &CallSchema, bindings: &ElementBindings, source: &str) -> String {
    let mut prefix = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    for (name, representation) in bindings.iter() {
        render_native_representation(
            &mut prefix,
            &format!("SEISMIC_ELEMENT_{}", native_macro(name)),
            representation,
        );
    }
    let mut buffer = 0usize;
    let mut word = 0usize;
    for dimension in schema.dimensions() {
        prefix.push_str(&format!(
            "#define SEISMIC_DIM_{} (seismic_words[{}])\n",
            native_macro(&dimension.name),
            word
        ));
        word += 1;
    }
    for (ordinal, parameter) in schema.parameters().iter().enumerate() {
        let name = native_macro(&parameter.name);
        let named = schema
            .parameters()
            .iter()
            .filter(|candidate| candidate.name == parameter.name)
            .count()
            == 1;
        match &parameter.kind {
            ParameterKind::Tensor {
                axes,
                representation,
                ..
            } => {
                if named {
                    prefix.push_str(&format!("#define SEISMIC_BUFFER_{name} {buffer}\n"));
                }
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_BUFFER {buffer}\n"
                ));
                render_native_representation(
                    &mut prefix,
                    &format!("SEISMIC_PARAM_{ordinal}"),
                    *representation,
                );
                if named {
                    render_native_representation(
                        &mut prefix,
                        &format!("SEISMIC_{name}"),
                        *representation,
                    );
                }
                buffer += 1;
                for axis in 0..axes.len() {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{ordinal}_EXTENT_{axis} (seismic_words[{}])\n",
                        word + axis
                    ));
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{ordinal}_STRIDE_{axis} (seismic_words[{}])\n",
                        word + axes.len() + axis
                    ));
                    if named {
                        prefix.push_str(&format!(
                            "#define SEISMIC_{name}_EXTENT_{axis} SEISMIC_PARAM_{ordinal}_EXTENT_{axis}\n#define SEISMIC_{name}_STRIDE_{axis} SEISMIC_PARAM_{ordinal}_STRIDE_{axis}\n"
                        ));
                    }
                }
                word += axes.len() * 2;
            }
            ParameterKind::Scalar { .. } | ParameterKind::Index { .. } => {
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal} (seismic_words[{word}])\n"
                ));
                if named {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{name} SEISMIC_PARAM_{ordinal}\n"
                    ));
                }
                word += 1;
            }
            ParameterKind::Range { .. } => {
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_START (seismic_words[{word}])\n"
                ));
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_END (seismic_words[{}])\n",
                    word + 1
                ));
                if named {
                    prefix.push_str(&format!(
                        "#define SEISMIC_PARAM_{name}_START SEISMIC_PARAM_{ordinal}_START\n#define SEISMIC_PARAM_{name}_END SEISMIC_PARAM_{ordinal}_END\n"
                    ));
                }
                word += 2;
            }
        }
    }
    let mut scalar_word = 0usize;
    for (ordinal, result) in schema.results().iter().enumerate() {
        if let ResultKind::Tensor {
            axes,
            representation,
        } = &result.kind
        {
            prefix.push_str(&format!(
                "#define SEISMIC_RESULT_{ordinal}_BUFFER {buffer}\n"
            ));
            render_native_representation(
                &mut prefix,
                &format!("SEISMIC_RESULT_{ordinal}"),
                *representation,
            );
            buffer += 1;
            for axis in 0..axes.len() {
                prefix.push_str(&format!(
                    "#define SEISMIC_RESULT_{ordinal}_EXTENT_{axis} (seismic_words[{}])\n",
                    word + axis
                ));
                prefix.push_str(&format!(
                    "#define SEISMIC_RESULT_{ordinal}_STRIDE_{axis} (seismic_words[{}])\n",
                    word + axes.len() + axis
                ));
            }
            word += axes.len() * 2;
        } else {
            prefix.push_str(&format!(
                "#define SEISMIC_RESULT_{ordinal}_WORD {scalar_word}\n"
            ));
            scalar_word += if matches!(result.kind, ResultKind::Range { .. }) {
                2
            } else {
                1
            };
        }
    }
    prefix.push_str(&format!("#define SEISMIC_BUFFER_WORDS {buffer}\n"));
    prefix.push_str(&format!(
        "#define SEISMIC_BUFFER_SCALAR_RESULTS {}\n",
        buffer + 1
    ));
    prefix.push_str(source);
    prefix
}

fn native_macro(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn render_native_representation(out: &mut String, prefix: &str, representation: RepresentationId) {
    let info = registry::representation_info(representation);
    out.push_str(&format!(
        "#define {prefix}_REPRESENTATION_{} 1\n",
        native_macro(info.name)
    ));
    out.push_str(&format!(
        "#define {prefix}_DECODED_{} 1\n",
        native_macro(info.decoded.name())
    ));
    match &info.kind {
        registry::RepresentationKind::Dense(dtype) => {
            out.push_str(&format!("#define {prefix}_KIND_DENSE 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP 1\n#define {prefix}_PLANE_COUNT 0\n",
                dtype.bytes(),
                dtype.bytes()
            ));
        }
        registry::RepresentationKind::Packed(layout) => {
            out.push_str(&format!("#define {prefix}_KIND_PACKED 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_PLANE_COUNT {}\n",
                layout.packet_size,
                layout.packet_alignment,
                layout.group,
                layout.planes.len()
            ));
            for (ordinal, plane) in layout.planes.iter().enumerate() {
                let plane_prefix = format!("{prefix}_PLANE_{ordinal}");
                out.push_str(&format!(
                    "#define {plane_prefix}_NAME_{} 1\n#define {plane_prefix}_OFFSET {}\n#define {plane_prefix}_BYTES_PER_GROUP {}\n#define {plane_prefix}_ALIGNMENT {}\n#define {plane_prefix}_GROUP {}\n#define {plane_prefix}_FIELDS {}\n#define {plane_prefix}_ENTRY_BITS {}\n#define {plane_prefix}_STORAGE_{} 1\n",
                    native_macro(plane.name),
                    plane.offset,
                    plane.bytes_per_group,
                    plane.alignment,
                    plane.group,
                    plane.fields,
                    plane.entry_bits,
                    native_macro(plane.storage_dtype.name())
                ));
                render_native_plane_encoding(out, &plane_prefix, &plane.encoding);
            }
        }
        registry::RepresentationKind::External(layout) => {
            out.push_str(&format!("#define {prefix}_KIND_EXTERNAL 1\n"));
            out.push_str(&format!(
                "#define {prefix}_PACKET_SIZE {}\n#define {prefix}_PACKET_ALIGNMENT {}\n#define {prefix}_LOGICAL_GROUP {}\n#define {prefix}_PLANE_COUNT 0\n",
                layout.packet_size, layout.packet_alignment, layout.logical_group
            ));
        }
    }
}

fn render_native_plane_encoding(
    out: &mut String,
    prefix: &str,
    encoding: &registry::PlaneEncoding,
) {
    match encoding {
        registry::PlaneEncoding::Dense(dtype) => {
            out.push_str(&format!(
                "#define {prefix}_ENCODING_DENSE 1\n#define {prefix}_ENCODING_DTYPE_{} 1\n",
                native_macro(dtype.name())
            ));
        }
        registry::PlaneEncoding::Packed {
            bits,
            interpretation,
        } => {
            out.push_str(&format!(
                "#define {prefix}_ENCODING_PACKED 1\n#define {prefix}_ENCODING_BITS {bits}\n"
            ));
            render_native_code_interpretation(out, prefix, interpretation);
        }
        registry::PlaneEncoding::FloatCode { format } => {
            let name = match format {
                registry::FloatCodeFormat::E2M1 => "E2M1",
                registry::FloatCodeFormat::E4M3 => "E4M3",
                registry::FloatCodeFormat::UE4M3 => "UE4M3",
            };
            out.push_str(&format!(
                "#define {prefix}_ENCODING_FLOAT_CODE 1\n#define {prefix}_ENCODING_FLOAT_CODE_{name} 1\n#define {prefix}_ENCODING_BITS {}\n",
                format.bits()
            ));
        }
    }
}

fn render_native_code_interpretation(
    out: &mut String,
    prefix: &str,
    interpretation: &registry::CodeInterpretation,
) {
    match interpretation {
        registry::CodeInterpretation::Unsigned => {
            out.push_str(&format!("#define {prefix}_CODE_UNSIGNED 1\n"));
        }
        registry::CodeInterpretation::TwosComplement => {
            out.push_str(&format!("#define {prefix}_CODE_TWOS_COMPLEMENT 1\n"));
        }
        registry::CodeInterpretation::Offset(offset) => {
            out.push_str(&format!(
                "#define {prefix}_CODE_OFFSET 1\n#define {prefix}_CODE_OFFSET_VALUE {offset}\n"
            ));
        }
        registry::CodeInterpretation::Table(values) => {
            out.push_str(&format!(
                "#define {prefix}_CODE_TABLE 1\n#define {prefix}_CODE_TABLE_COUNT {}\n",
                values.len()
            ));
            for (ordinal, value) in values.iter().enumerate() {
                out.push_str(&format!("#define {prefix}_CODE_TABLE_{ordinal} {value}\n"));
            }
        }
    }
}

#[cfg(test)]
mod native_abi_tests {
    use super::*;
    use seismic_lang::checked::{check_source, SourceFile, SourceSet};

    fn render_for(representation: &str) -> String {
        let module = check_source(SourceSet::new(vec![SourceFile {
            path: "probe.seismic".to_owned(),
            text: "fn probe[N](x: &tensor[N] E) -> tensor[1] f32:\n    let mut output = tensor[1] f32\n    for i in 0..1:\n        output[i] = f32(x[0])\n    return output\n"
                .to_owned(),
        }]))
        .expect("probe source checks");
        let binding = registry::representation(representation).expect("registered representation");
        let bindings = ElementBindings::new().bind("E", binding);
        let logical = module
            .entry(module.entries()[0].id, &bindings)
            .expect("probe entry monomorphizes");
        render_native_source(logical.schema(), &bindings, "\nkernel void probe() {}\n")
    }

    #[test]
    fn native_prefix_describes_dense_element_parameter_and_tensor_abi() {
        let source = render_for("f16");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_DENSE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_DECODED_F16 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 1\n"));
        assert!(source.contains("#define SEISMIC_X_REPRESENTATION_F16 1\n"));
        assert!(source.contains("#define SEISMIC_RESULT_0_REPRESENTATION_F32 1\n"));
    }

    #[test]
    fn native_prefix_describes_packed_planes_and_encoding() {
        let source = render_for("q8g32");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_Q8G32 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_PACKED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 36\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 2\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_NAME_WORDS 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_BYTES_PER_GROUP 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_ENCODING_PACKED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_0_CODE_UNSIGNED 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_NAME_SCALE 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_OFFSET 32\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_STORAGE_F32 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_1_ENCODING_DENSE 1\n"));
    }

    #[test]
    fn native_prefix_describes_external_packets() {
        let source = render_for("gguf_q4_k");
        assert!(source.contains("#define SEISMIC_ELEMENT_E_REPRESENTATION_GGUF_Q4_K 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_KIND_EXTERNAL 1\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PACKET_SIZE 144\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_LOGICAL_GROUP 256\n"));
        assert!(source.contains("#define SEISMIC_ELEMENT_E_PLANE_COUNT 0\n"));
    }
}

pub(crate) struct PreparedHandle<T: TargetFamily, E: NativeExecutor<T>> {
    pub(crate) prepared: Arc<Prepared<T, E>>,
    pub(crate) device: Arc<crate::api::device::DeviceInner>,
}

/// Executable allocation binding into the run-owned physical resource table.
/// This carries no backing ownership; aliases share the same admitted slot.
pub(crate) enum PhysicalBufferBinding {
    Bound { allocation: u64, base_offset: u64, accessible_bytes: u64, tensor: Option<seismic_compiler::executable::RuntimeTensorGeometry> },
    Reached { slot: u64, alignment: u64 },
}

struct IssuedResources<'a, T: TargetFamily, E: NativeExecutor<T>> {
    owner: &'a mut AdmittedResources,
    device: &'a Arc<Opened<T, E>>,
    bindings: &'a [PhysicalBufferBinding],
    buffers: Vec<Option<RuntimeBuffer<Buffer<T, E>>>>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> seismic_compiler::executable::ExecutionResources<Buffer<T, E>> for IssuedResources<'_, T, E> {
    fn buffer(&self, allocation: seismic_compiler::executable::ExecutableAllocationId) -> &RuntimeBuffer<Buffer<T, E>> {
        self.buffers[allocation.ordinal()].as_ref().expect("planned allocation used before its reached acquisition")
    }
    fn retire_completed_instances(&mut self, allocations: &[seismic_compiler::executable::ExecutableAllocationId]) {
        for allocation in allocations {
            let index = allocation.ordinal();
            if let PhysicalBufferBinding::Reached { slot, .. } = self.bindings[index] {
                // Drop the derived native handle before releasing backing and
                // its charge from the one run-owned physical slot.
                self.buffers[index] = None;
                self.owner.retire_private(slot);
            }
        }
    }
    fn acquire_instance(&mut self, allocation: seismic_compiler::executable::ExecutableAllocationId, bytes: u64, alignment: u64) -> Result<(), ExecutionError> {
        let index = allocation.ordinal();
        let PhysicalBufferBinding::Reached { slot, alignment: planned_alignment } = self.bindings[index] else {
            assert!(self.buffers[index].as_ref().expect("initial backing absent").accessible_bytes >= bytes,
                "reached instance exceeds initial backing");
            return Ok(());
        };
        if self.owner.private_backing(slot).is_some_and(|backing| backing.bytes() >= bytes) {
            return Ok(());
        }
        // The schedule has completed all prior users and excluded retained
        // region products before requesting replacement of this bank.
        self.buffers[index] = None;
        self.owner.retire_private(slot);
        self.owner.check_reached_capacity(bytes)?;
        let backing = self.device.allocate_storage(bytes, alignment.max(planned_alignment))?;
        let permit = backing.try_acquire(true).expect("fresh private backing cannot have an access owner");
        let buffer = RuntimeBuffer { tensor: None, buffer: typed_buffer::<T, E>(&backing), base_offset: 0, accessible_bytes: bytes };
        self.owner.install_private(slot, permit);
        self.buffers[index] = Some(buffer);
        Ok(())
    }
}

pub(crate) struct Staged {
    pub(crate) values: InvocationValues,
    pub(crate) buffers: Vec<PhysicalBufferBinding>,
    pub(crate) allocated_bytes: u64,
}

/// Opaque native command produced by resource admission.
///
/// Submission can issue this command and inspect only its completed scalar
/// slots and output device. It cannot reach the prepared policy, selected
/// executable, allocation plan, or layout expressions retained inside it.
/// One already selected executable and its runtime resource namespace.
/// Both policy dispatch and controlled preparation trials enter admission here.
pub(crate) struct SelectedExecutable<T: TargetFamily, E: NativeExecutor<T>> {
    executable: ExecutableVariant<T, E::Handle>,
    owner: u64,
    variant: usize,
    persistent: Arc<PersistentTable>,
    output_device: Arc<crate::api::device::DeviceInner>,
}

pub(crate) struct AdmittedCommand<T: TargetFamily, E: NativeExecutor<T>> {
    selected: Arc<SelectedExecutable<T, E>>,
    staged: Staged,
    published: Vec<seismic_compiler::executable::ExecutedTensorPublication>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> AdmittedCommand<T, E> {
    pub(crate) fn new(selected: Arc<SelectedExecutable<T, E>>, staged: Staged) -> Self {
        Self { selected, staged, published: Vec::new() }
    }
    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.staged.allocated_bytes
    }
    pub(crate) fn values(&self) -> &InvocationValues {
        &self.staged.values
    }
    pub(crate) fn output_device(&self) -> &Arc<crate::api::device::DeviceInner> {
        &self.selected.output_device
    }
    pub(crate) fn published_allocation(&self, path: &[u32]) -> Option<usize> {
        self.published.iter().find(|publication| publication.path == path)
            .map(|publication| publication.allocation.ordinal())
    }
    pub(crate) fn allocation_slot(&self, index: usize) -> u64 {
        match self.staged.buffers[index] {
            PhysicalBufferBinding::Bound { allocation, .. } => allocation,
            PhysicalBufferBinding::Reached { slot, .. } => slot,
        }
    }
    pub(crate) fn published_tensors(
        &self,
        resources: &AdmittedResources,
    ) -> Result<Vec<(Vec<u32>, crate::execution::AdmittedOutput)>, ExecutionError> {
        self.published.iter().map(|publication| {
            let key = match self.staged.buffers[publication.allocation.ordinal()] {
                PhysicalBufferBinding::Bound { allocation, .. } => allocation,
                PhysicalBufferBinding::Reached { slot, .. } => slot,
            };
            let allocation = resources.allocation(key).clone();
            if !publication.byte_offset.checked_add(publication.bytes)
                .is_some_and(|end| end <= allocation.bytes()) {
                return Err(ExecutionError::ConstructionContradiction(
                    "published tensor exceeds its actual backing".into(),
                ));
            }
            Ok((publication.path.clone(), crate::execution::AdmittedOutput::Tensor {
                allocation,
                byte_offset: publication.byte_offset,
                byte_len: publication.bytes,
                representation: publication.representation,
                extents: publication.extents.clone(),
                strides: publication.strides.clone(),
            }))
        }).collect()
    }
    pub(crate) fn issue(
        &mut self,
        submission: &mut E::Submission,
        device: &Service<T, E>,
        resources: &mut AdmittedResources,
        opened: &Arc<Opened<T, E>>,
    ) -> Result<(), ExecutionError> {
        let buffers = self.staged.buffers.iter().map(|binding| match binding {
            PhysicalBufferBinding::Bound { allocation, base_offset, accessible_bytes, tensor } => {
                let allocation = resources.allocation(*allocation);
                assert!(base_offset.checked_add(*accessible_bytes).is_some_and(|end| end <= allocation.bytes()),
                    "staged binding exceeds its admitted physical slot");
                Some(RuntimeBuffer { tensor: tensor.clone(), buffer: typed_buffer::<T, E>(allocation), base_offset: *base_offset, accessible_bytes: *accessible_bytes })
            }
            PhysicalBufferBinding::Reached { slot, .. } => {
                resources.declare_private(*slot);
                resources.private_backing(*slot).map(|allocation| RuntimeBuffer {
                    tensor: None, buffer: typed_buffer::<T, E>(allocation), base_offset: 0, accessible_bytes: allocation.bytes(),
                })
            }
        }).collect();
        let mut resources = IssuedResources { owner: resources, device: opened, bindings: &self.staged.buffers, buffers };
        self.published = execute_variant(
            &self.selected.executable,
            submission,
            device,
            &mut resources,
            &mut self.staged.values,
        )?;
        Ok(())
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> Prepared<T, E> {
    fn attributes(&self) -> Vec<KeyValue> {
        vec![
            key_str("seismic.entry", hex(self.kernel.entry().digest())),
            key_str("seismic.backend", T::NAME.as_str()),
        ]
    }
}

impl<T: TargetFamily, E: NativeExecutor<T>> PreparedHandle<T, E> {
    pub(crate) fn call(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError>
    where
        T: 'static,
    {
        let attributes = self.prepared.attributes();
        let mut span = Timed::start("seismic.call", attributes.clone());
        let (results, allocated_bytes) = workflow::native::call_one(self.clone(), args)?;
        span.attribute(key_bool("seismic.ok", true));
        telemetry::record_call(span.elapsed_ms(), allocated_bytes, &attributes);
        Ok(results)
    }
}

fn acquire_access(access: &[(Arc<Allocation>, bool)]) -> Vec<AllocationPermit> {
    access
        .iter()
        .map(|(allocation, write)| allocation.acquire(*write))
        .collect()
}

/// Access collection for the deliberately separate authored-native route.
fn collect_native_access(schema: &CallSchema, args: &EncodedArgs) -> Vec<(Arc<Allocation>, bool)> {
    let mut merged: BTreeMap<u64, (Arc<Allocation>, bool)> = BTreeMap::new();
    for (parameter, tensor) in schema.parameters().iter().zip(args.tensors()) {
        let (ParameterKind::Tensor { access, .. }, Some(tensor)) = (&parameter.kind, tensor) else {
            continue;
        };
        let write = matches!(access, TensorAccess::Owned | TensorAccess::Mutable);
        merged
            .entry(tensor.allocation().identity())
            .and_modify(|(_, current)| *current |= write)
            .or_insert_with(|| (tensor.allocation().clone(), write));
    }
    merged.into_values().collect()
}

fn copy_between<T: TargetFamily, E: NativeExecutor<T>>(
    service: &Service<T, E>,
    source: &Buffer<T, E>,
    destination: &Buffer<T, E>,
    bytes: u64,
) -> Result<(), ExecutionError> {
    const CHUNK: usize = 1 << 20;
    let mut scratch = vec![0u8; CHUNK.min(usize::try_from(bytes).unwrap_or(CHUNK))];
    let mut offset = 0u64;
    while offset < bytes {
        let length = usize::try_from((bytes - offset).min(CHUNK as u64)).unwrap_or(CHUNK);
        service.read(source, offset, &mut scratch[..length])?;
        service.write(destination, offset, &scratch[..length])?;
        offset += length as u64;
    }
    Ok(())
}

#[path = "workflow/mod.rs"]
pub(crate) mod workflow;
pub(crate) use workflow::native::{BoundWorkflowGraph, WorkflowGraphDraft};

#[path = "feedback.rs"]
mod feedback;

#[cfg(test)]
mod physical_slot_tests {
    use super::*;

    struct EmptyStorage;
    impl Storage for EmptyStorage {
        fn read(&self, _: u64, _: &mut [u8]) -> Result<(), ExecutionError> { Ok(()) }
        fn write(&self, _: u64, _: &[u8]) -> Result<(), ExecutionError> { Ok(()) }
        fn as_any(&self) -> &dyn Any { self }
    }

    #[test]
    fn admitted_slot_owns_backing_charge_and_exclusive_access() {
        let memory = MemoryDomain::new(crate::memory::PoolLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let allocation = Allocation::new(1, 16, reservation.take(16), Box::new(EmptyStorage));
        let weak = Arc::downgrade(&allocation);
        let permit = allocation.acquire(true);
        let resources = AdmittedResources::new(reservation, vec![permit]);
        drop(allocation);
        assert_eq!(memory.usage().charged, 16);
        assert!(resources.allocation(1).try_acquire(false).is_none());
        assert!(resources.access(resources.allocation(1)).owns(resources.allocation(1)));
        drop(resources);
        assert!(weak.upgrade().is_none());
        assert_eq!(memory.usage().charged, 0);
    }

    #[test]
    fn native_allocation_owner_enforces_target_limits_without_leaking_reservation() {
        let catalog = crate::devices::Catalog::discover().unwrap();
        let device = catalog.open_backend(registry::BackendName::Cpu).unwrap();
        let crate::backends::OpenedKind::Cpu(opened) = &device.kind else { unreachable!() };
        let baseline = opened.memory_usage().charged;
        let maximum = opened.device_description().limits().max_allocation_bytes;
        let error = opened.allocate_storage(maximum.checked_add(1).unwrap(), 4).err().expect("target limit is enforced before allocating");
        assert!(matches!(error, ExecutionError::AllocationCapacity { required, available } if required == (maximum + 1).into() && available == maximum));
        assert_eq!(opened.memory_usage().charged, baseline);
        assert!(matches!(opened.allocate_storage(4, 0), Err(ExecutionError::ConstructionContradiction(_))));
        assert_eq!(opened.memory_usage().charged, baseline);
    }

    #[test]
    fn reached_slots_release_dead_capacity_and_preserve_live_permits_on_refusal() {
        let memory = MemoryDomain::new(crate::memory::PoolLedger::new());
        let mut resources = AdmittedResources::new(memory.reserve(0).unwrap(), vec![]);
        resources.set_reached_budget(24);
        resources.declare_private(100);
        resources.declare_private(101);
        let make = |id, bytes| {
            let mut reservation = memory.reserve(bytes).unwrap();
            Allocation::new(id, bytes, reservation.take(bytes), Box::new(EmptyStorage))
        };
        let first = make(1, 16);
        resources.check_reached_capacity(16).unwrap();
        resources.install_private(100, first.acquire(true));
        drop(first);
        assert_eq!(resources.check_reached_capacity(16), Err(ExecutionError::AllocationCapacity { required: 16u64.into(), available: 8 }));
        assert!(resources.private_backing(100).unwrap().try_acquire(false).is_none());
        resources.retire_private(100);
        assert_eq!(memory.usage().charged, 0);
        resources.check_reached_capacity(24).unwrap();
        let second = make(2, 24);
        resources.install_private(101, second.acquire(true));
        drop(second);
        assert_eq!(memory.usage().charged, 24);
        assert_eq!(resources.reached_allocated(), 40);
        drop(resources);
        assert_eq!(memory.usage().charged, 0);
    }

    #[test]
    fn completed_dead_bank_releases_device_capacity_without_releasing_live_bank() {
        let memory = MemoryDomain::new(crate::memory::PoolLedger::new());
        memory.set_limit(Some(24)).unwrap();
        let mut resources = AdmittedResources::new(memory.reserve(0).unwrap(), vec![]);
        resources.set_reached_budget(24);
        for slot in [100, 101, 102] { resources.declare_private(slot); }
        let make = |id, bytes| {
            let mut reservation = memory.reserve(bytes).unwrap();
            Allocation::new(id, bytes, reservation.take(bytes), Box::new(EmptyStorage))
        };
        let dead = make(1, 16);
        let live = make(2, 8);
        resources.install_private(100, dead.acquire(true));
        resources.install_private(101, live.acquire(true));
        let pending = dead.clone();
        drop(dead);
        drop(live);
        assert!(memory.reserve(16).is_err());
        assert!(resources.check_reached_capacity(16).is_err());
        assert_eq!(memory.usage().charged, 24);
        // The compiler's successful prefix completion precedes this release;
        // backend references and the slot's permit must both be gone.
        drop(pending);
        resources.retire_private(100);
        assert_eq!(memory.usage().charged, 8);
        resources.check_reached_capacity(16).unwrap();
        let replacement = make(3, 16);
        resources.install_private(102, replacement.acquire(true));
        drop(replacement);
        assert_eq!(memory.usage().charged, 24);
        assert!(resources.private_backing(101).unwrap().try_acquire(false).is_none());
        assert!(memory.reserve(1).is_err());
        drop(resources);
        assert_eq!(memory.usage().charged, 0);
    }

    #[test]
    fn published_backing_survives_slot_release_without_retaining_its_permit() {
        let memory = MemoryDomain::new(crate::memory::PoolLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let allocation = Allocation::new(1, 16, reservation.take(16), Box::new(EmptyStorage));
        let resources = AdmittedResources::new(reservation, vec![allocation.acquire(true)]);
        drop(allocation);
        let published = resources.allocation(1).clone();
        drop(resources);
        assert_eq!(memory.usage().charged, 16);
        let read = published.try_acquire(false).expect("completed run released its exclusive access");
        drop(read);
        drop(published);
        assert_eq!(memory.usage().charged, 0);
    }
}
