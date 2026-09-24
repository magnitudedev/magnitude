//! The single backend-generic preparation and synchronous invocation driver.
//! Planning ends at `PreparedKernel`; this module validates public arguments,
//! evaluates a selected executable, owns storage, and executes its typed schedule.

use crate::api::kernel::{
    DecodedResults, DecodedValue, EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument,
    PendingWorkflowResults, PrepareError, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::CallError;
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
    validate_invocation, ArgumentValue, DeviceIdentity, PreparedKernel,
};
use seismic_compiler::target::CompilerRegistry;
use seismic_compiler::{
    prepare_analytically, OptimizationCompletion, PlanningBudget, PreparationBudget,
};
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{
    CallSchema, ElementBindings, ParameterKind, TensorAccess,
};
use seismic_lang::expr::compiled::InvocationValues;
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

/// The identity of a newly opened device, unique in the process.
pub(crate) fn fresh_device_identity() -> DeviceIdentity {
    DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed))
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
            identity: fresh_device_identity(),
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
    pub(crate) fn executor(&self) -> &E {
        &self.executor
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
        self.allocate_storage_with(bytes, alignment, |service| service.allocate(bytes, alignment))
    }

    /// Charge and allocate storage whose buffer `make` forms, for backends
    /// with more than one kind of memory.
    pub(crate) fn allocate_storage_with(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        make: impl FnOnce(&Service<T, E>) -> Result<Buffer<T, E>, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let mut reservation = reserve(&self.memory, bytes)?;
        self.allocate_reserved_with(bytes, alignment, &mut reservation, make)
    }

    pub(crate) fn allocate_reserved(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_reserved_with(bytes, alignment, reservation, |service| {
            service.allocate(bytes, alignment)
        })
    }

    fn allocate_reserved_with(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
        make: impl FnOnce(&Service<T, E>) -> Result<Buffer<T, E>, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let limits = self.device_description().limits();
        let natural_max = if limits.max_index_bits >= 64 { u64::MAX } else { (1u64 << limits.max_index_bits) - 1 };
        let limits = AllocationLimits {
            max_allocation_bytes: limits.max_allocation_bytes.min(natural_max),
            max_allocation_alignment: limits.max_allocation_alignment,
        };
        allocate_reserved(limits, bytes, alignment, reservation, || {
            let buffer = make(&self.service)?;
            Ok(Box::new(TypedStorage::<T, E> {
                service: self.service.clone(),
                buffer,
            }))
        })
    }
}

/// What one device admits for a single allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AllocationLimits {
    pub(crate) max_allocation_bytes: u64,
    pub(crate) max_allocation_alignment: u64,
}

/// Reserve `bytes` in `memory` for an allocation.
pub(crate) fn reserve(memory: &Arc<MemoryDomain>, bytes: u64) -> Result<MemoryReservation, ExecutionError> {
    memory.reserve(bytes).map_err(|capacity| ExecutionError::AllocationCapacity {
        required: capacity.required.into(),
        available: capacity.available,
    })
}

/// The backend-neutral allocation core: check `bytes` and `alignment`
/// against `limits`, form the storage with `make`, and charge it to
/// `reservation`.
pub(crate) fn allocate_reserved(
    limits: AllocationLimits,
    bytes: u64,
    alignment: u64,
    reservation: &mut MemoryReservation,
    make: impl FnOnce() -> Result<Box<dyn Storage>, ExecutionError>,
) -> Result<Arc<Allocation>, ExecutionError> {
    if bytes > limits.max_allocation_bytes {
        return Err(ExecutionError::AllocationCapacity { required: bytes.into(), available: limits.max_allocation_bytes });
    }
    if !alignment.is_power_of_two() || alignment > limits.max_allocation_alignment {
        return Err(ExecutionError::ConstructionContradiction(format!(
            "allocation alignment {alignment} exceeds target contract {}", limits.max_allocation_alignment)));
    }
    let storage = make()?;
    Ok(Allocation::new(fresh_allocation_identity(), bytes, reservation.take(bytes), storage))
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
    /// The newest submitted device work that uses this allocation, and the
    /// newest that writes it, until they complete. An allocation belongs to
    /// one device, whose native submissions all run on its one queue in
    /// submission order, so the newest fence of a kind completes after every
    /// earlier one: it is the only one host access must wait for.
    last_use: Option<Arc<dyn DeviceCompletion>>,
    last_write: Option<Arc<dyn DeviceCompletion>>,
}

/// Completion of submitted device work, as allocation fences observe it.
/// Reaching completion is independent of whether the work succeeded; its
/// submitter reports failures.
pub(crate) trait DeviceCompletion: Send + Sync {
    fn is_complete(&self) -> bool;
    /// Block until the work has finished executing.
    fn wait_complete(&self);
}

impl AllocationAccess {
    /// Forget completed fences. The newest use completes last, so once it
    /// has, no device work uses the allocation.
    fn prune(&mut self) {
        if self.last_write.as_ref().is_some_and(|fence| fence.is_complete()) {
            self.last_write = None;
        }
        if self.last_use.as_ref().is_some_and(|fence| fence.is_complete()) {
            self.last_use = None;
            self.last_write = None;
        }
    }
    /// The device fence host access of the given kind must wait for: a host
    /// write waits for every device use, a host read for device writes.
    fn conflicting_device(&mut self, write: bool) -> Option<Arc<dyn DeviceCompletion>> {
        self.prune();
        if write {
            self.last_use.clone()
        } else {
            self.last_write.clone()
        }
    }
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
    /// Host access: waits for conflicting host permits and for submitted
    /// device work that conflicts with the access (a host read waits for
    /// device writes; a host write waits for every device use).
    pub(crate) fn acquire(self: &Arc<Self>, write: bool) -> AllocationPermit {
        let mut state = self
            .access
            .lock()
            .expect("tensor allocation access lock poisoned");
        loop {
            if state.writer || (write && state.readers != 0) {
                state = self
                    .access_changed
                    .wait(state)
                    .expect("tensor allocation access lock poisoned while waiting");
                continue;
            }
            let Some(fence) = state.conflicting_device(write) else {
                break;
            };
            drop(state);
            fence.wait_complete();
            state = self
                .access
                .lock()
                .expect("tensor allocation access lock poisoned");
        }
        self.grant(state, write)
    }

    /// Access by device work being encoded for one queue. It excludes host
    /// access for the duration of encoding but never waits for other device
    /// work: the queue orders it. The submitter records its completion with
    /// [`Allocation::record_device_use`] before releasing the permit.
    pub(crate) fn acquire_for_device(self: &Arc<Self>, write: bool) -> AllocationPermit {
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
        self.grant(state, write)
    }

    /// Record submitted device work so later host access orders after it.
    /// Called in submission order: `completion` completes after every fence
    /// recorded before it.
    pub(crate) fn record_device_use(&self, completion: Arc<dyn DeviceCompletion>, write: bool) {
        let mut state = self
            .access
            .lock()
            .expect("tensor allocation access lock poisoned");
        if write {
            state.last_write = Some(completion.clone());
        }
        state.last_use = Some(completion);
    }

    /// No submitted device work still uses this allocation.
    pub(crate) fn device_idle(&self) -> bool {
        let mut state = self
            .access
            .lock()
            .expect("tensor allocation access lock poisoned");
        state.prune();
        state.last_use.is_none()
    }

    fn grant(
        self: &Arc<Self>,
        mut state: MutexGuard<'_, AllocationAccess>,
        write: bool,
    ) -> AllocationPermit {
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
        if state.writer || (write && state.readers != 0) || state.conflicting_device(write).is_some() {
            return None;
        }
        Some(self.grant(state, write))
    }
}

impl Drop for Allocation {
    /// Storage is released only after every submitted device use finishes.
    fn drop(&mut self) {
        let state = self
            .access
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(fence) = state.last_use.take() {
            fence.wait_complete();
        }
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

pub(crate) fn typed_buffer<T: TargetFamily, E: NativeExecutor<T>>(allocation: &Allocation) -> &Buffer<T, E> {
    &allocation.storage.as_any().downcast_ref::<TypedStorage<T, E>>()
        .unwrap_or_else(|| panic!("Tensor allocation backend invariant violated after successful WrongDevice validation"))
        .buffer
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
        let buffer = RuntimeBuffer { tensor: None, buffer: typed_buffer::<T, E>(&backing).clone(), base_offset: 0, accessible_bytes: bytes };
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
                Some(RuntimeBuffer { tensor: tensor.clone(), buffer: typed_buffer::<T, E>(allocation).clone(), base_offset: *base_offset, accessible_bytes: *accessible_bytes })
            }
            PhysicalBufferBinding::Reached { slot, .. } => {
                resources.declare_private(*slot);
                resources.private_backing(*slot).map(|allocation| RuntimeBuffer {
                    tensor: None, buffer: typed_buffer::<T, E>(allocation).clone(), base_offset: 0, accessible_bytes: allocation.bytes(),
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

/// Access collection for the deliberately separate authored-native route.
pub(crate) fn collect_native_access(schema: &CallSchema, args: &EncodedArgs) -> Vec<(Arc<Allocation>, bool)> {
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
