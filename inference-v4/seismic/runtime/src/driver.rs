//! The single backend-generic preparation and synchronous invocation driver.
//! Planning ends at `PreparedKernel`; this module validates public arguments,
//! evaluates a selected executable, owns storage, and executes its typed schedule.

use crate::api::kernel::{
    DecodedResults, DecodedValue, EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument,
    NativeDefinition, NativeExpr, PendingWorkflowResults, PrepareError, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::CallError;
use crate::memory::{MemoryCharge, MemoryDomain, MemoryReservation, MemoryUsage};
use crate::resources::{AdmissionDomain, PersistentTable};
use crate::telemetry::{self, hex, key_bool, key_str, key_u64, Timed};
use opentelemetry::KeyValue;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::evaluation::AnalyticalEvaluationContext;
use seismic_compiler::executable::{
    execute_variant, DeviceService, ExecutableResultBinding, NativeExecutor, RuntimeBuffer,
};
use seismic_compiler::executable::{
    ExecutableAllocationKind, ExecutableGlobalAllocationKind, ExecutableScalarResultKind,
};
use seismic_compiler::numerics::{EvidenceCatalog, PolicyIdentity};
use seismic_compiler::prepared::{
    validate_invocation, ArgumentValue, DeviceIdentity, InvocationContract, PreparedKernel,
};
use seismic_compiler::target::CompilerRegistry;
use seismic_compiler::{
    prepare_analytically, OptimizationCompletion, PlanningBudget, PreparationBudget, TargetCoverage,
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
    analytical_loader: Option<
        fn(
            &Service<T, E>,
            Arc<DeviceDescription<T>>,
        )
            -> Result<AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError>,
    >,
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
        analytical: Result<AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError>,
    ) -> Self {
        if let Ok(analytical) = &analytical {
            assert!(
                analytical.is_bound_to(&device),
                "runtime composition paired an analytical context with another device description"
            );
        }
        Self {
            identity: DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)),
            service: Arc::new(service),
            device,
            executor,
            compiler_registry,
            analytical: std::sync::OnceLock::from(analytical),
            analytical_loader: None,
            cache: Mutex::new(HashMap::new()),
            memory: MemoryDomain::new(),
            admission: AdmissionDomain::new(),
        }
    }
    pub(crate) fn new_lazy(
        service: Service<T, E>,
        executor: E,
        compiler_registry: &'static CompilerRegistry<T>,
        device: Arc<DeviceDescription<T>>,
        analytical_loader: fn(
            &Service<T, E>,
            Arc<DeviceDescription<T>>,
        ) -> Result<
            AnalyticalEvaluationContext<T>,
            seismic_compiler::errors::TargetError,
        >,
    ) -> Self {
        Self {
            identity: DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)),
            service: Arc::new(service),
            device,
            executor,
            compiler_registry,
            analytical: std::sync::OnceLock::new(),
            analytical_loader: Some(analytical_loader),
            cache: Mutex::new(HashMap::new()),
            memory: MemoryDomain::new(),
            admission: AdmissionDomain::new(),
        }
    }
    fn analytical(
        &self,
    ) -> Result<&AnalyticalEvaluationContext<T>, seismic_compiler::errors::TargetError> {
        self.analytical
            .get_or_init(|| {
                let loader = self
                    .analytical_loader
                    .expect("an unopened device analytical context has no loader");
                loader(&self.service, self.device.clone())
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
    ) -> Result<(), crate::api::MemoryLimitError> {
        self.memory.set_limit(limit)
    }
    pub(crate) fn allocate_storage(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let mut reservation = self.memory.reserve(bytes).map_err(|capacity| {
            ExecutionError::AllocationFailed(format!(
                "allocation requires {} bytes; {} bytes remain in the configured device limit",
                capacity.required, capacity.available
            ))
        })?;
        self.allocate_reserved(bytes, alignment, &mut reservation)
    }

    fn allocate_reserved(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
    ) -> Result<Arc<Allocation>, ExecutionError> {
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
    persistent: Arc<PersistentTable>,
}

pub(crate) fn prepare<T, E, C>(
    opened: &Arc<Opened<T, E>>,
    compiler: &C,
    native_context: &C::Context,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    precision: PrecisionPolicy,
) -> Result<Arc<Prepared<T, E>>, PrepareError>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_target::NativeCompiler<T, Handle = E::Handle>,
{
    let analytical = opened.analytical().map_err(|error| {
        PrepareError::Preparation(
            seismic_compiler::errors::PreparationError::NativeCompilation(
                seismic_target::NativeCompilationError::ToolchainFailure(error.to_string()),
            ),
        )
    })?;
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
        policy: PolicyIdentity::of(&precision),
    };
    if let Some(hit) = opened.cache().get(&key).and_then(Weak::upgrade) {
        return Ok(hit);
    }
    let attributes = vec![
        key_str("seismic.module", hex(key.module.digest())),
        key_str("seismic.entry", hex(key.entry.digest())),
        key_str("seismic.backend", T::NAME.as_str()),
        key_str(
            "seismic.target.hardware",
            analytical
                .device()
                .compatibility_identity()
                .hardware
                .clone(),
        ),
        key_str(
            "seismic.target.fingerprint",
            hex(&analytical.device().identity().fingerprint),
        ),
        key_str("seismic.policy", hex(&key.policy.0)),
    ];
    let mut span = Timed::start("seismic.prepare", attributes.clone());
    let preparation_budget = PreparationBudget::default();
    let planning_budget = PlanningBudget::default();
    let kernel = prepare_analytically(
        logical,
        analytical,
        opened.compiler_registry,
        compiler,
        native_context,
        &precision,
        &EvidenceCatalog::default(),
        &preparation_budget,
        &planning_budget,
    )
    .map_err(PrepareError::Preparation)?;
    span.attribute(key_u64("seismic.variants", kernel.variants().len() as u64));
    let coverage = kernel.planning_coverage();
    span.attribute(key_str(
        "seismic.planning.target_coverage",
        match coverage.target {
            TargetCoverage::Exhaustive => "exhaustive",
        },
    ));
    span.attribute(key_str(
        "seismic.planning.optimization_completion",
        match &coverage.optimization {
            OptimizationCompletion::Complete => "complete",
            OptimizationCompletion::Limited(_) => "limited",
        },
    ));
    if let OptimizationCompletion::Limited(limit) = &coverage.optimization {
        span.attribute(key_str(
            "seismic.planning.optimization_limit",
            format!("{limit:?}"),
        ));
    }
    span.attribute(key_u64(
        "seismic.planning.solver_work_units",
        coverage.budget.solver_work_units,
    ));
    span.attribute(key_u64(
        "seismic.planning.solver_elapsed_ms",
        coverage.budget.solver_elapsed_ms,
    ));
    span.attribute(key_u64(
        "seismic.planning.solver_memory_bytes",
        coverage.budget.solver_memory_bytes,
    ));
    span.attribute(key_u64(
        "seismic.planning.optimized_assignments",
        coverage.budget.optimized_assignments,
    ));
    span.attribute(key_u64(
        "seismic.planning.executable_variants",
        coverage.budget.executable_variants,
    ));
    span.attribute(key_u64(
        "seismic.planning.retained_metadata_bytes",
        coverage.budget.retained_metadata_bytes,
    ));
    telemetry::record_preparation(span.elapsed_ms(), &attributes);
    let prepared = Arc::new(Prepared {
        identity: NEXT_PREPARED.fetch_add(1, Ordering::Relaxed),
        device: opened.clone(),
        kernel,
        persistent: Arc::new(PersistentTable::new()),
    });
    opened.cache().insert(key, Arc::downgrade(&prepared));
    Ok(prepared)
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
    let source = render_native_source(logical.schema(), &bindings, definition.source);
    let pipeline =
        seismic_metal::DirectPipeline::compile(&opened.service, &source, definition.entry)
            .map_err(|error| {
                PrepareError::Preparation(
                    seismic_compiler::errors::PreparationError::NativeCompilation(error),
                )
            })?;
    Ok(Arc::new(NativePreparedMetal {
        opened: opened.clone(),
        public_device: public_device.clone(),
        logical,
        invocation,
        pipeline,
        definition,
        results,
    }))
}

#[cfg(target_os = "macos")]
impl NativePreparedMetal {
    pub(crate) fn call(&self, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        let schema = self.logical.schema();
        let arguments = args.values();
        let values =
            validate_invocation(schema, &self.invocation, self.opened.identity(), &arguments)
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
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Index(word)));
                }
                NativeResult::Range => {
                    let start = read_native_word(&scalar_bytes, scalar_offset);
                    let end = read_native_word(&scalar_bytes, scalar_offset + 1);
                    scalar_offset += 2;
                    final_values.push(DecodedValue::Scalar(ArgumentValue::Range { start, end }));
                }
            }
        }
        Ok(DecodedResults::new(final_values))
    }
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
    expression.evaluate(values).map_err(|error| {
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
                Some(SymbolValue::Nat(value)) => Ok(value),
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
            Some(SymbolValue::Nat(value)) => words.push(value),
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
                ));
            }
            ParameterKind::Range { start, end, .. } => {
                words.push(native_symbol(
                    values
                        .get(*start)
                        .expect("validated range start disappeared"),
                ));
                words.push(native_symbol(
                    values.get(*end).expect("validated range end disappeared"),
                ));
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
fn native_symbol(value: SymbolValue) -> u64 {
    match value {
        SymbolValue::Nat(value) => value,
        SymbolValue::Int(value) => value as u64,
        SymbolValue::F32(value) => u64::from(value.to_bits()),
        SymbolValue::F16(value) | SymbolValue::BF16(value) => u64::from(value),
        SymbolValue::I32(value) => u64::from(value as u32),
        SymbolValue::U32(value) => u64::from(value),
        SymbolValue::Bool(value) => u64::from(value),
    }
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

pub(crate) struct Staged<T: TargetFamily, E: NativeExecutor<T>> {
    pub(crate) variant: usize,
    pub(crate) values: InvocationValues,
    pub(crate) buffers: Vec<RuntimeBuffer<Buffer<T, E>>>,
    // Keeps every staged allocation alive through native completion, including
    // non-result scratch and external arguments whose typed buffers borrow the
    // backend storage.
    pub(crate) _allocations: Vec<Arc<Allocation>>,
    pub(crate) allocated_bytes: u64,
}

/// Opaque native command produced by resource admission.
///
/// Submission can issue this command and inspect only its completed scalar
/// slots and output device. It cannot reach the prepared policy, selected
/// executable, allocation plan, or layout expressions retained inside it.
pub(crate) struct AdmittedCommand<T: TargetFamily, E: NativeExecutor<T>> {
    kernel: Arc<PreparedHandle<T, E>>,
    staged: Staged<T, E>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> AdmittedCommand<T, E> {
    pub(crate) fn new(kernel: Arc<PreparedHandle<T, E>>, staged: Staged<T, E>) -> Self {
        Self { kernel, staged }
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.staged.allocated_bytes
    }

    pub(crate) fn values(&self) -> &InvocationValues {
        &self.staged.values
    }

    pub(crate) fn output_device(&self) -> &Arc<crate::api::device::DeviceInner> {
        &self.kernel.device
    }

    pub(crate) fn issue(
        &mut self,
        submission: &mut E::Submission,
        device: &Service<T, E>,
    ) -> Result<(), ExecutionError> {
        self.kernel
            .issue_admitted(submission, device, &mut self.staged)
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
    pub(crate) fn issue_admitted(
        &self,
        submission: &mut E::Submission,
        device: &Service<T, E>,
        staged: &mut Staged<T, E>,
    ) -> Result<(), ExecutionError> {
        let variant = &self.prepared.kernel.variants().as_slice()[staged.variant];
        execute_variant(
            variant,
            submission,
            device,
            &staged.buffers,
            &mut staged.values,
        )
    }

    pub(crate) fn call(self: &Arc<Self>, args: EncodedArgs) -> Result<DecodedResults, CallError>
    where
        T: 'static,
    {
        let attributes = self.prepared.attributes();
        let mut span = Timed::start("seismic.call", attributes.clone());
        let (results, allocated_bytes) = workflow_native::call_one(self.clone(), args)?;
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

#[path = "workflow/native.rs"]
mod workflow_native;
pub(crate) use workflow_native::{BoundWorkflowGraph, WorkflowGraphDraft};
