//! The single backend-generic preparation and synchronous invocation driver.
//! Planning ends at `PreparedKernel`; this module validates public arguments,
//! evaluates a selected executable, owns storage, and executes its typed schedule.

use crate::api::kernel::{
    DecodedResults, DecodedValue, EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument,
    NativeDefinition, NativeExpr, PendingWorkflowResults, PrepareError, WorkflowCompletionAny,
    WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::CallError;
use crate::telemetry::{self, hex, key_bool, key_str, key_u64, Timed};
use opentelemetry::KeyValue;
use seismic_compiler::errors::{ExecutionError, InvocationError};
use seismic_compiler::executable::{
    execute_schedule, DeviceService, ExecutableResultBinding, ExecutionEnvironment,
    NativeExecution, NativeExecutor, NativeSubmission, RuntimeBuffer,
};
use seismic_compiler::executable::{
    ExecutableAllocationKind, ExecutableGlobalAllocationKind, ExecutableScalarResultKind,
};
use seismic_compiler::numerics::{EvidenceCatalog, PolicyIdentity};
use seismic_compiler::plan_space::plan_space;
use seismic_compiler::portfolio::prepare_kernel;
use seismic_compiler::prepared::{
    validate_invocation, ArgumentValue, DeviceIdentity, InvocationContract, PreparedKernel,
};
use seismic_compiler::target::{Backend, DeviceContract, ExecutionProfile};
use seismic_compiler::PreparationBudget;
use seismic_lang::checked::CheckedModule;
use seismic_lang::entry::{
    CallSchema, ElementBindings, LogicalEntry, ParameterKind, ResultKind, TensorAccess,
};
use seismic_lang::expr::compiled::{CompiledNat, InvocationValues};
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::{EntryId, ModuleHash, RepresentationId, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry;
use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

pub(crate) type Service<B> = <<B as Backend>::Executor as NativeExecutor<B>>::Device;
pub(crate) type Buffer<B> = <Service<B> as DeviceService<B>>::Buffer;

static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);
static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);
static NEXT_WORKFLOW: AtomicU64 = AtomicU64::new(1);

fn fresh_allocation_identity() -> u64 {
    NEXT_ALLOCATION.fetch_add(1, Ordering::Relaxed)
}

fn admitted<T>(value: Result<T, seismic_lang::expr::EvalError>) -> T {
    value.unwrap_or_else(|error| panic!("PreparedKernel coverage invariant violated: admitted evaluator was not total: {error:?}"))
}

pub(crate) struct Opened<B: Backend> {
    identity: DeviceIdentity,
    service: Arc<Service<B>>,
    executor: B::Executor,
    profile: std::sync::OnceLock<Result<OpenedProfile<B>, seismic_compiler::errors::TargetError>>,
    profile_loader: Option<
        fn(
            &Service<B>,
        ) -> Result<
            (DeviceContract<B>, ExecutionProfile<B>),
            seismic_compiler::errors::TargetError,
        >,
    >,
    cache: Mutex<HashMap<PreparationKey, Weak<Prepared<B>>>>,
    memory: Arc<MemoryDomain>,
}

struct OpenedProfile<B: Backend> {
    contract: Arc<DeviceContract<B>>,
    execution: Arc<ExecutionProfile<B>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemoryUsage {
    pub(crate) charged: u64,
    pub(crate) limit: Option<u64>,
}

struct MemoryState {
    charged: u64,
    limit: Option<u64>,
}

struct MemoryDomain {
    state: Mutex<MemoryState>,
}

impl MemoryDomain {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(MemoryState {
                charged: 0,
                limit: None,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, MemoryState> {
        self.state
            .lock()
            .expect("device memory-accounting lock poisoned")
    }

    fn usage(&self) -> MemoryUsage {
        let state = self.state();
        MemoryUsage {
            charged: state.charged,
            limit: state.limit,
        }
    }

    fn set_limit(&self, limit: Option<u64>) -> Result<(), crate::api::MemoryLimitError> {
        let mut state = self.state();
        if let Some(limit) = limit.filter(|limit| *limit < state.charged) {
            return Err(crate::api::MemoryLimitError {
                limit,
                charged: state.charged,
            });
        }
        state.limit = limit;
        Ok(())
    }

    fn reserve(self: &Arc<Self>, bytes: u64) -> Result<MemoryReservation, MemoryCapacity> {
        let mut state = self.state();
        let next = state.charged.checked_add(bytes).ok_or(MemoryCapacity {
            required: bytes,
            available: 0,
        })?;
        if let Some(limit) = state.limit {
            if next > limit {
                return Err(MemoryCapacity {
                    required: bytes,
                    available: limit.saturating_sub(state.charged),
                });
            }
        }
        state.charged = next;
        Ok(MemoryReservation {
            domain: self.clone(),
            remaining: bytes,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct MemoryCapacity {
    required: u64,
    available: u64,
}

/// One atomic reservation for a complete invocation. It is split into the
/// allocation-owned charges as physical allocations are created; any unused
/// tail is released on drop. This prevents concurrent public allocations
/// from invalidating a successful invocation-capacity check.
struct MemoryReservation {
    domain: Arc<MemoryDomain>,
    remaining: u64,
}

/// One indivisible workflow admission result. Members are acquired before
/// submission and are released together if any later binding step fails.
struct ReservationSet {
    _members: Vec<MemoryReservation>,
}

impl MemoryReservation {
    fn take(&mut self, bytes: u64) -> MemoryCharge {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .expect("invocation allocated more memory than it atomically reserved");
        MemoryCharge {
            domain: self.domain.clone(),
            bytes,
        }
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        if self.remaining == 0 {
            return;
        }
        let mut state = self.domain.state();
        state.charged = state
            .charged
            .checked_sub(self.remaining)
            .expect("unused memory reservation exceeded the accounting total");
    }
}

struct MemoryCharge {
    domain: Arc<MemoryDomain>,
    bytes: u64,
}

impl Drop for MemoryCharge {
    fn drop(&mut self) {
        let mut state = self.domain.state();
        state.charged = state
            .charged
            .checked_sub(self.bytes)
            .expect("device allocation charge exceeded the accounting total");
    }
}

#[derive(PartialEq, Eq, Hash)]
struct PreparationKey {
    module: ModuleHash,
    entry: StableEntryId,
    bindings: Vec<(String, RepresentationId)>,
    policy: PolicyIdentity,
}

impl<B: Backend> Opened<B> {
    pub(crate) fn new(
        service: Service<B>,
        executor: B::Executor,
        contract: Arc<DeviceContract<B>>,
        execution: Arc<ExecutionProfile<B>>,
    ) -> Self {
        Self {
            identity: DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)),
            service: Arc::new(service),
            executor,
            profile: std::sync::OnceLock::from(Ok(OpenedProfile {
                contract,
                execution,
            })),
            profile_loader: None,
            cache: Mutex::new(HashMap::new()),
            memory: MemoryDomain::new(),
        }
    }
    pub(crate) fn new_lazy(
        service: Service<B>,
        executor: B::Executor,
        profile_loader: fn(
            &Service<B>,
        ) -> Result<
            (DeviceContract<B>, ExecutionProfile<B>),
            seismic_compiler::errors::TargetError,
        >,
    ) -> Self {
        Self {
            identity: DeviceIdentity(NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)),
            service: Arc::new(service),
            executor,
            profile: std::sync::OnceLock::new(),
            profile_loader: Some(profile_loader),
            cache: Mutex::new(HashMap::new()),
            memory: MemoryDomain::new(),
        }
    }
    fn profile(&self) -> Result<&OpenedProfile<B>, seismic_compiler::errors::TargetError> {
        self.profile
            .get_or_init(|| {
                let loader = self
                    .profile_loader
                    .expect("an unopened device profile has no loader");
                loader(&self.service).map(|(contract, execution)| OpenedProfile {
                    contract: Arc::new(contract),
                    execution: Arc::new(execution),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }
    pub(crate) fn identity(&self) -> DeviceIdentity {
        self.identity
    }
    pub(crate) fn contract(&self) -> &DeviceContract<B> {
        &self
            .profile()
            .expect("device profile acquisition failed after opening")
            .contract
    }
    fn executor(&self) -> &B::Executor {
        &self.executor
    }
    fn cache(&self) -> MutexGuard<'_, HashMap<PreparationKey, Weak<Prepared<B>>>> {
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
            Box::new(TypedStorage::<B> {
                service: self.service.clone(),
                buffer,
            }),
        ))
    }
}

pub(crate) fn capability_summaries<B: Backend>(contract: &DeviceContract<B>) -> Vec<String> {
    registry::capabilities(B::NAME)
        .iter()
        .filter(|capability| contract.supports_capability(capability.id))
        .map(|capability| format!("{}.{}", B::NAME.as_str(), capability.name))
        .collect()
}

pub(crate) fn opened_capability_summaries<B: Backend>(opened: &Opened<B>) -> Vec<String> {
    opened
        .profile()
        .map(|profile| capability_summaries(&profile.contract))
        .unwrap_or_default()
}

pub(crate) trait Storage: Send + Sync {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError>;
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError>;
    fn as_any(&self) -> &dyn Any;
}

struct TypedStorage<B: Backend> {
    service: Arc<Service<B>>,
    buffer: Buffer<B>,
}

impl<B: Backend> Storage for TypedStorage<B> {
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

fn typed_buffer<B: Backend>(allocation: &Allocation) -> Buffer<B> {
    allocation.storage.as_any().downcast_ref::<TypedStorage<B>>()
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

pub(crate) struct Prepared<B: Backend> {
    device: Arc<Opened<B>>,
    kernel: PreparedKernel<B>,
    persistent: Arc<PersistentTable<B>>,
}

struct Persistent<B: Backend> {
    buffer: Buffer<B>,
    allocation: Arc<Allocation>,
    capacity: u64,
}

impl<B: Backend> Clone for Persistent<B> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            allocation: self.allocation.clone(),
            capacity: self.capacity,
        }
    }
}

enum PersistentSlot<B: Backend> {
    Ready {
        allocation: Persistent<B>,
        leases: u64,
    },
    Growing,
}

struct PersistentTable<B: Backend> {
    slots: Mutex<HashMap<(usize, usize), PersistentSlot<B>>>,
    changed: Condvar,
}

struct PersistentLease<B: Backend> {
    table: Arc<PersistentTable<B>>,
    key: (usize, usize),
}

impl<B: Backend> Drop for PersistentLease<B> {
    fn drop(&mut self) {
        let mut slots = self
            .table
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(PersistentSlot::Ready { leases, .. }) = slots.get_mut(&self.key) else {
            panic!("leased persistent allocation is not ready")
        };
        *leases = leases
            .checked_sub(1)
            .expect("persistent allocation lease count underflow");
        if *leases == 0 {
            self.table.changed.notify_all();
        }
    }
}

impl<B: Backend> PersistentTable<B> {
    fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
            changed: Condvar::new(),
        }
    }
}

pub(crate) fn prepare<B: Backend>(
    opened: &Arc<Opened<B>>,
    module: &CheckedModule,
    entry: EntryId,
    bindings: ElementBindings,
    precision: PrecisionPolicy,
) -> Result<Arc<Prepared<B>>, PrepareError> {
    let profile = opened.profile().map_err(|error| {
        PrepareError::Preparation(
            seismic_compiler::errors::PreparationError::NativeCompilation(
                seismic_compiler::errors::NativeCompilationError::ToolchainFailure(
                    error.to_string(),
                ),
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
        key_str("seismic.backend", B::NAME.as_str()),
        key_str(
            "seismic.target.hardware",
            profile.contract.compatibility_identity().hardware.clone(),
        ),
        key_str(
            "seismic.target.fingerprint",
            hex(&profile.contract.identity().fingerprint),
        ),
        key_str("seismic.policy", hex(&key.policy.0)),
    ];
    let mut span = Timed::start("seismic.prepare", attributes.clone());
    let budget = PreparationBudget::default();
    let machine = profile.contract.planning_with(&profile.execution);
    let space = plan_space(
        logical,
        machine,
        &precision,
        &EvidenceCatalog::default(),
        &budget,
    )
    .map_err(PrepareError::Preparation)?;
    let kernel = prepare_kernel(space, machine).map_err(PrepareError::Preparation)?;
    span.attribute(key_u64("seismic.variants", kernel.variants().len() as u64));
    span.attribute(key_bool("seismic.optimal", kernel.optimal()));
    telemetry::record_preparation(span.elapsed_ms(), &attributes);
    let prepared = Arc::new(Prepared {
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
    opened: Arc<Opened<seismic_metal::Metal>>,
    public_device: Arc<crate::api::device::DeviceInner>,
    logical: LogicalEntry,
    invocation: InvocationContract,
    pipeline: seismic_metal::DirectPipeline,
    definition: NativeDefinition,
    results: Vec<NativeResult>,
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_native_metal(
    opened: &Arc<Opened<seismic_metal::Metal>>,
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
    let source = render_native_source(logical.schema(), definition.source);
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

        let mut access = collect_access(schema, &args);
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
                    typed_buffer::<seismic_metal::Metal>(tensor.allocation()),
                    tensor.byte_offset(),
                ));
            }
        }
        for tensor in &tensor_results {
            buffers.push((
                typed_buffer::<seismic_metal::Metal>(tensor.allocation()),
                tensor.byte_offset(),
            ));
        }
        buffers.push((typed_buffer::<seismic_metal::Metal>(&word_allocation), 0));
        buffers.push((typed_buffer::<seismic_metal::Metal>(&scalar_allocation), 0));
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

#[cfg(target_os = "macos")]
fn render_native_source(schema: &CallSchema, source: &str) -> String {
    let mut prefix = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
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
            ParameterKind::Tensor { axes, .. } => {
                if named {
                    prefix.push_str(&format!("#define SEISMIC_BUFFER_{name} {buffer}\n"));
                }
                prefix.push_str(&format!(
                    "#define SEISMIC_PARAM_{ordinal}_BUFFER {buffer}\n"
                ));
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
        if let ResultKind::Tensor { axes, .. } = &result.kind {
            prefix.push_str(&format!(
                "#define SEISMIC_RESULT_{ordinal}_BUFFER {buffer}\n"
            ));
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

#[cfg(target_os = "macos")]
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

pub(crate) struct PreparedHandle<B: Backend> {
    pub(crate) prepared: Arc<Prepared<B>>,
    pub(crate) device: Arc<crate::api::device::DeviceInner>,
}

type Submission<B> = <<B as Backend>::Executor as NativeExecutor<B>>::Submission;
type SubmittedExecution<B> = <Submission<B> as NativeSubmission<B>>::Execution;

/// Open workflow topology. Only this draft accepts nodes; closing consumes it
/// so raw nodes can never reach backend submission.
struct WorkflowDraft<'a, B: Backend> {
    node: &'a PreparedHandle<B>,
}

/// Dependency-closed reusable topology. The one-node constructor is the
/// synchronous convenience path; multi-node generated workflows use the same
/// bind/admit/submit states.
struct PreparedWorkflow<'a, B: Backend> {
    node: &'a PreparedHandle<B>,
}

struct BoundWorkflow<'a, B: Backend> {
    node: &'a PreparedHandle<B>,
    args: EncodedArgs,
    values: InvocationValues,
    variant: usize,
    access: Vec<(Arc<Allocation>, bool)>,
}

struct AdmittedWorkflowRun<'a, B: Backend> {
    node: &'a PreparedHandle<B>,
    staged: Staged<B>,
    _reservation: MemoryReservation,
    _access: Vec<AllocationPermit>,
    submission: Submission<B>,
}

struct WorkflowExecution<'a, B: Backend> {
    node: &'a PreparedHandle<B>,
    staged: Staged<B>,
    _reservation: MemoryReservation,
    _access: Vec<AllocationPermit>,
    native: SubmittedExecution<B>,
}

impl<'a, B: Backend> WorkflowDraft<'a, B> {
    fn one(node: &'a PreparedHandle<B>) -> Self {
        Self { node }
    }

    fn close(self) -> PreparedWorkflow<'a, B> {
        PreparedWorkflow { node: self.node }
    }
}

impl<'a, B: Backend> PreparedWorkflow<'a, B> {
    fn bind(self, args: EncodedArgs) -> Result<BoundWorkflow<'a, B>, CallError> {
        let arguments = args.values();
        let values = validate_invocation(
            self.node.prepared.kernel.schema(),
            self.node.prepared.kernel.invocation_contract(),
            self.node.prepared.device.identity(),
            &arguments,
        )
        .map_err(CallError::Invocation)?;
        let variant = self.node.prepared.kernel.select(&values).index;
        let access = collect_access(self.node.prepared.kernel.schema(), &args);
        Ok(BoundWorkflow {
            node: self.node,
            args,
            values,
            variant,
            access,
        })
    }
}

impl<'a, B: Backend> BoundWorkflow<'a, B> {
    fn admit(self, span: &mut Timed) -> Result<AdmittedWorkflowRun<'a, B>, CallError> {
        let required = self
            .node
            .prepared
            .allocation_requirement(&self.values, self.variant);
        let mut reservation =
            self.node
                .prepared
                .device
                .memory
                .reserve(required)
                .map_err(|capacity| {
                    CallError::Invocation(InvocationError::AllocationCapacity {
                        required: capacity.required,
                        available: capacity.available,
                    })
                })?;
        let staged = self
            .node
            .prepared
            .stage(self.args, self.values, self.variant, span, &mut reservation)
            .map_err(CallError::Execution)?;
        let mut access: BTreeMap<u64, (Arc<Allocation>, bool)> = self
            .access
            .into_iter()
            .map(|(allocation, write)| (allocation.identity(), (allocation, write)))
            .collect();
        for allocation in &staged.persistent_allocations {
            access
                .entry(allocation.identity())
                .and_modify(|(_, write)| *write = true)
                .or_insert_with(|| (allocation.clone(), true));
        }
        let access = acquire_access(&access.into_values().collect::<Vec<_>>());
        let submission = self
            .node
            .prepared
            .device
            .executor()
            .begin_submission()
            .map_err(CallError::Execution)?;
        Ok(AdmittedWorkflowRun {
            node: self.node,
            staged,
            _reservation: reservation,
            _access: access,
            submission,
        })
    }
}

impl<'a, B: Backend> AdmittedWorkflowRun<'a, B> {
    fn submit(mut self) -> Result<WorkflowExecution<'a, B>, CallError> {
        let variant = &self.node.prepared.kernel.variants().as_slice()[self.staged.variant];
        let mut environment = ExecutionEnvironment {
            device: &*self.node.prepared.device.service,
            buffers: &self.staged.buffers,
            values: &mut self.staged.values,
            kernels: variant.kernels(),
        };
        let issued = execute_schedule(variant.schedule(), &mut self.submission, &mut environment);
        let native = self.submission.submit().map_err(CallError::Execution)?;
        if let Err(error) = issued {
            native.complete().map_err(CallError::Execution)?;
            return Err(CallError::Execution(error));
        }
        Ok(WorkflowExecution {
            node: self.node,
            staged: self.staged,
            _reservation: self._reservation,
            _access: self._access,
            native,
        })
    }
}

impl<B: Backend> WorkflowExecution<'_, B> {
    fn complete(self) -> Result<DecodedResults, CallError> {
        self.native.complete().map_err(CallError::Execution)?;
        Ok(self.node.prepared.decode(&self.staged, &self.node.device))
    }
}

struct WorkflowGraphNode<B: Backend> {
    kernel: Arc<PreparedHandle<B>>,
    args: EncodedWorkflowArgs,
}

pub(crate) struct WorkflowGraphDraft<B: Backend> {
    identity: u64,
    device: Arc<Opened<B>>,
    public_device: Arc<crate::api::device::DeviceInner>,
    nodes: Vec<WorkflowGraphNode<B>>,
}

pub(crate) struct AdmittedWorkflowGraph<B: Backend> {
    identity: u64,
    device: Arc<Opened<B>>,
    nodes: Vec<(Arc<PreparedHandle<B>>, Staged<B>)>,
    _reservations: ReservationSet,
    _access: Vec<AllocationPermit>,
    submission: Submission<B>,
}

pub(crate) struct SubmittedWorkflowGraph<B: Backend> {
    identity: u64,
    nodes: Vec<(Arc<PreparedHandle<B>>, Staged<B>)>,
    _reservations: ReservationSet,
    _access: Vec<AllocationPermit>,
    native: SubmittedExecution<B>,
}

impl<B: Backend> WorkflowGraphDraft<B> {
    pub(crate) fn new(
        device: Arc<Opened<B>>,
        public_device: Arc<crate::api::device::DeviceInner>,
    ) -> Self {
        Self {
            identity: NEXT_WORKFLOW.fetch_add(1, Ordering::Relaxed),
            device,
            public_device,
            nodes: Vec::new(),
        }
    }

    pub(crate) fn enqueue(
        &mut self,
        kernel: Arc<PreparedHandle<B>>,
        args: EncodedWorkflowArgs,
    ) -> Result<PendingWorkflowResults, crate::api::WorkflowError> {
        if !Arc::ptr_eq(&kernel.prepared.device, &self.device) {
            return Err(crate::api::WorkflowError::CrossWorkflowResult);
        }
        let node = u32::try_from(self.nodes.len()).expect("workflow node ordinal space exhausted");
        for argument in args.arguments() {
            let reference = match argument {
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference))
                | EncodedWorkflowArgument::ScalarResult(reference) => Some(reference),
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::ResultLeadingSlice {
                    result,
                    ..
                }) => Some(result),
                EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(_))
                | EncodedWorkflowArgument::Scalar(_) => None,
            };
            if reference.is_some_and(|reference| {
                reference.workflow != self.identity || reference.node >= node
            }) {
                return Err(crate::api::WorkflowError::CrossWorkflowResult);
            }
        }
        let count = u32::try_from(kernel.prepared.kernel.schema().results().len())
            .expect("workflow result ordinal space exhausted");
        self.nodes.push(WorkflowGraphNode { kernel, args });
        Ok(PendingWorkflowResults::new(self.identity, node, count))
    }

    pub(crate) fn admit(self) -> Result<AdmittedWorkflowGraph<B>, CallError> {
        if self.nodes.is_empty() {
            return Err(CallError::Workflow(crate::api::WorkflowError::Empty));
        }
        let mut staged_nodes = Vec::with_capacity(self.nodes.len());
        let mut tensor_results: Vec<Vec<Option<Arc<TensorInner>>>> = Vec::new();
        let mut reservations = Vec::with_capacity(self.nodes.len());
        let mut access: BTreeMap<u64, (Arc<Allocation>, bool)> = BTreeMap::new();
        let mut span = Timed::start("seismic.workflow.admit", Vec::new());

        for node in self.nodes {
            let mut args = EncodedArgs::new();
            for argument in node.args.into_arguments() {
                match argument {
                    EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::External(tensor)) => {
                        args.push_tensor(tensor);
                    }
                    EncodedWorkflowArgument::Tensor(WorkflowTensorArgument::Result(reference)) => {
                        let tensor = tensor_results
                            .get(reference.node as usize)
                            .and_then(|results| results.get(reference.result as usize))
                            .and_then(Clone::clone)
                            .ok_or(CallError::Workflow(
                                crate::api::WorkflowError::MissingProducerResult,
                            ))?;
                        args.push_tensor(tensor);
                    }
                    EncodedWorkflowArgument::Tensor(
                        WorkflowTensorArgument::ResultLeadingSlice { result, start, end },
                    ) => {
                        let tensor = tensor_results
                            .get(result.node as usize)
                            .and_then(|results| results.get(result.result as usize))
                            .and_then(Clone::clone)
                            .ok_or(CallError::Workflow(
                                crate::api::WorkflowError::MissingProducerResult,
                            ))?;
                        let view = tensor
                            .slice_leading(start, end)
                            .map_err(crate::api::WorkflowError::TensorView)
                            .map_err(CallError::Workflow)?;
                        args.push_tensor(Arc::new(view));
                    }
                    EncodedWorkflowArgument::Scalar(value) => args.push_scalar(value),
                    EncodedWorkflowArgument::ScalarResult(_) => {
                        return Err(CallError::Workflow(
                            crate::api::WorkflowError::HostBoundaryRequired,
                        ));
                    }
                }
            }
            for (allocation, write) in collect_access(node.kernel.prepared.kernel.schema(), &args) {
                access
                    .entry(allocation.identity())
                    .and_modify(|(_, existing)| *existing |= write)
                    .or_insert((allocation, write));
            }
            let arguments = args.values();
            let values = validate_invocation(
                node.kernel.prepared.kernel.schema(),
                node.kernel.prepared.kernel.invocation_contract(),
                self.device.identity(),
                &arguments,
            )
            .map_err(CallError::Invocation)?;
            let variant = node.kernel.prepared.kernel.select(&values).index;
            let required = node
                .kernel
                .prepared
                .allocation_requirement(&values, variant);
            let mut reservation = self.device.memory.reserve(required).map_err(|capacity| {
                CallError::Invocation(InvocationError::AllocationCapacity {
                    required: capacity.required,
                    available: capacity.available,
                })
            })?;
            let staged = node
                .kernel
                .prepared
                .stage(args, values, variant, &mut span, &mut reservation)
                .map_err(CallError::Execution)?;
            for allocation in &staged.persistent_allocations {
                access
                    .entry(allocation.identity())
                    .and_modify(|(_, write)| *write = true)
                    .or_insert_with(|| (allocation.clone(), true));
            }
            tensor_results.push(
                node.kernel
                    .prepared
                    .workflow_tensor_results(&staged, &self.public_device),
            );
            reservations.push(reservation);
            staged_nodes.push((node.kernel, staged));
        }
        let access = acquire_access(&access.into_values().collect::<Vec<_>>());
        let submission = self
            .device
            .executor()
            .begin_submission()
            .map_err(CallError::Execution)?;
        Ok(AdmittedWorkflowGraph {
            identity: self.identity,
            device: self.device,
            nodes: staged_nodes,
            _reservations: ReservationSet {
                _members: reservations,
            },
            _access: access,
            submission,
        })
    }
}

impl<B: Backend> AdmittedWorkflowGraph<B> {
    pub(crate) fn submit(mut self) -> Result<SubmittedWorkflowGraph<B>, CallError> {
        let mut issued = Ok(());
        for (kernel, staged) in &mut self.nodes {
            let variant = &kernel.prepared.kernel.variants().as_slice()[staged.variant];
            let mut environment = ExecutionEnvironment {
                device: &*self.device.service,
                buffers: &staged.buffers,
                values: &mut staged.values,
                kernels: variant.kernels(),
            };
            if let Err(error) =
                execute_schedule(variant.schedule(), &mut self.submission, &mut environment)
            {
                issued = Err(error);
                break;
            }
        }
        let native = self.submission.submit().map_err(CallError::Execution)?;
        if let Err(error) = issued {
            native.complete().map_err(CallError::Execution)?;
            return Err(CallError::Execution(error));
        }
        Ok(SubmittedWorkflowGraph {
            identity: self.identity,
            nodes: self.nodes,
            _reservations: self._reservations,
            _access: self._access,
            native,
        })
    }
}

impl<B: Backend> SubmittedWorkflowGraph<B> {
    fn complete_values(self) -> Result<Vec<Vec<DecodedValue>>, CallError> {
        self.native.complete().map_err(CallError::Execution)?;
        Ok(self
            .nodes
            .iter()
            .map(|(kernel, staged)| kernel.prepared.decode_values(staged, &kernel.device))
            .collect())
    }

    pub(crate) fn into_completion(self) -> WorkflowCompletionAny
    where
        B: 'static,
    {
        let identity = self.identity;
        WorkflowCompletionAny::pending(identity, move || self.complete_values())
    }
}

struct Staged<B: Backend> {
    variant: usize,
    values: InvocationValues,
    buffers: Vec<RuntimeBuffer<Buffer<B>>>,
    allocations: Vec<Arc<Allocation>>,
    persistent_allocations: Vec<Arc<Allocation>>,
    _persistent_leases: Vec<PersistentLease<B>>,
    _arguments: Vec<Arc<TensorInner>>,
    allocated_bytes: u64,
}

impl<B: Backend> Prepared<B> {
    fn attributes(&self) -> Vec<KeyValue> {
        vec![
            key_str("seismic.entry", hex(self.kernel.entry().digest())),
            key_str("seismic.backend", B::NAME.as_str()),
        ]
    }
    fn stage(
        &self,
        args: EncodedArgs,
        values: InvocationValues,
        variant_index: usize,
        span: &mut Timed,
        reservation: &mut MemoryReservation,
    ) -> Result<Staged<B>, ExecutionError> {
        let variant = self
            .kernel
            .variants()
            .as_slice()
            .get(variant_index)
            .expect("bound workflow selected a variant outside its prepared portfolio");
        span.attribute(key_str(
            "seismic.variant.factory",
            variant.identity().implementation.factory.name,
        ));
        span.attribute(key_str(
            "seismic.variant.assignment",
            hex(&variant.identity().assignment),
        ));
        let mut buffers = Vec::with_capacity(variant.allocations().len());
        let mut allocations = Vec::with_capacity(variant.allocations().len());
        let mut persistent_allocations = Vec::new();
        let mut persistent_leases = Vec::new();
        let mut allocated_bytes = 0u64;
        for (index, plan) in variant.allocations().iter().enumerate() {
            let candidates = plan.byte_candidates();
            let required = candidates.rest().iter().fold(
                admitted(candidates.first().evaluate(&values)),
                |required, bytes| required.max(admitted(bytes.evaluate(&values))),
            );
            match &plan.kind {
                ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Argument(
                    parameter,
                )) => {
                    let ordinal = self.kernel.schema().parameter_ordinal(*parameter);
                    let tensor = args
                        .tensor(ordinal)
                        .unwrap_or_else(|| panic!("generated tensor argument keepalive is absent"));
                    let allocation = tensor.allocation().clone();
                    buffers.push(RuntimeBuffer {
                        buffer: typed_buffer::<B>(&allocation),
                        base_offset: tensor.byte_offset(),
                        accessible_bytes: tensor.byte_len(),
                    });
                    allocations.push(allocation);
                }
                ExecutableAllocationKind::Global(
                    ExecutableGlobalAllocationKind::Result | ExecutableGlobalAllocationKind::Arena,
                )
                | ExecutableAllocationKind::LaunchScratch { .. }
                | ExecutableAllocationKind::KernelAbi { .. } => {
                    let allocation =
                        self.device
                            .allocate_reserved(required, plan.alignment, reservation)?;
                    let buffer = typed_buffer::<B>(&allocation);
                    buffers.push(RuntimeBuffer {
                        buffer,
                        base_offset: 0,
                        accessible_bytes: required,
                    });
                    allocations.push(allocation);
                    allocated_bytes = allocated_bytes.saturating_add(required);
                }
                ExecutableAllocationKind::Global(ExecutableGlobalAllocationKind::Persistent) => {
                    let key = (variant_index, index);
                    let (grow, old, selected) = loop {
                        let mut slots = self
                            .persistent
                            .slots
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        match slots.get_mut(&key) {
                            Some(PersistentSlot::Ready { allocation, leases })
                                if allocation.capacity >= required =>
                            {
                                *leases = leases
                                    .checked_add(1)
                                    .expect("persistent allocation lease count overflow");
                                break (false, None, Some(allocation.clone()));
                            }
                            Some(PersistentSlot::Ready { allocation, leases }) if *leases == 0 => {
                                let old = allocation.clone();
                                slots.insert(key, PersistentSlot::Growing);
                                break (true, Some(old), None);
                            }
                            Some(PersistentSlot::Ready { .. } | PersistentSlot::Growing) => {
                                drop(
                                    self.persistent
                                        .changed
                                        .wait(slots)
                                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                                );
                            }
                            None => {
                                slots.insert(key, PersistentSlot::Growing);
                                break (true, None, None);
                            }
                        }
                    };
                    let entry = if grow {
                        let grown = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let allocation = self.device.allocate_reserved(
                                required,
                                plan.alignment,
                                reservation,
                            )?;
                            let new_buffer = typed_buffer::<B>(&allocation);
                            if let Some(old) = &old {
                                let _old_read = old.allocation.acquire(false);
                                let _new_write = allocation.acquire(true);
                                copy_between::<B>(
                                    &*self.device.service,
                                    &old.buffer,
                                    &new_buffer,
                                    old.capacity,
                                )?;
                            }
                            Ok::<_, ExecutionError>(Persistent {
                                buffer: new_buffer,
                                allocation,
                                capacity: required,
                            })
                        }));
                        let mut slots = self
                            .persistent
                            .slots
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        match grown {
                            Ok(Ok(entry)) => {
                                slots.insert(
                                    key,
                                    PersistentSlot::Ready {
                                        allocation: entry.clone(),
                                        leases: 1,
                                    },
                                );
                                allocated_bytes = allocated_bytes.saturating_add(required);
                                self.persistent.changed.notify_all();
                                entry
                            }
                            Ok(Err(error)) => {
                                if let Some(old) = old {
                                    slots.insert(
                                        key,
                                        PersistentSlot::Ready {
                                            allocation: old,
                                            leases: 0,
                                        },
                                    );
                                } else {
                                    slots.remove(&key);
                                }
                                self.persistent.changed.notify_all();
                                return Err(error);
                            }
                            Err(payload) => {
                                if let Some(old) = old {
                                    slots.insert(
                                        key,
                                        PersistentSlot::Ready {
                                            allocation: old,
                                            leases: 0,
                                        },
                                    );
                                } else {
                                    slots.remove(&key);
                                }
                                self.persistent.changed.notify_all();
                                drop(slots);
                                std::panic::resume_unwind(payload);
                            }
                        }
                    } else {
                        selected.expect("ready persistent allocation was not selected")
                    };
                    persistent_leases.push(PersistentLease {
                        table: self.persistent.clone(),
                        key,
                    });
                    persistent_allocations.push(entry.allocation.clone());
                    buffers.push(RuntimeBuffer {
                        buffer: entry.buffer.clone(),
                        base_offset: 0,
                        accessible_bytes: entry.capacity,
                    });
                    allocations.push(entry.allocation.clone());
                }
            }
        }
        span.attribute(key_u64("seismic.allocated_bytes", allocated_bytes));
        Ok(Staged {
            variant: variant_index,
            values,
            buffers,
            allocations,
            persistent_allocations,
            _persistent_leases: persistent_leases,
            _arguments: args.into_tensors(),
            allocated_bytes,
        })
    }
    fn allocation_requirement(&self, values: &InvocationValues, variant_index: usize) -> u64 {
        let variant = self
            .kernel
            .variants()
            .as_slice()
            .get(variant_index)
            .expect("bound workflow selected a variant outside its prepared portfolio");
        loop {
            let persistent = self
                .persistent
                .slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let unavailable = variant
                .allocations()
                .iter()
                .enumerate()
                .any(|(index, plan)| {
                    if !matches!(
                        &plan.kind,
                        ExecutableAllocationKind::Global(
                            ExecutableGlobalAllocationKind::Persistent
                        )
                    ) {
                        return false;
                    }
                    let candidates = plan.byte_candidates();
                    let required = candidates.rest().iter().fold(
                        admitted(candidates.first().evaluate(values)),
                        |required, bytes| required.max(admitted(bytes.evaluate(values))),
                    );
                    match persistent.get(&(variant_index, index)) {
                        Some(PersistentSlot::Growing) => true,
                        Some(PersistentSlot::Ready { allocation, leases }) => {
                            allocation.capacity < required && *leases > 0
                        }
                        None => false,
                    }
                });
            if unavailable {
                drop(
                    self.persistent
                        .changed
                        .wait(persistent)
                        .unwrap_or_else(std::sync::PoisonError::into_inner),
                );
                continue;
            }
            return variant
                .allocations()
                .iter()
                .enumerate()
                .map(|(index, plan)| {
                    let candidates = plan.byte_candidates();
                    let required = candidates.rest().iter().fold(
                        admitted(candidates.first().evaluate(values)),
                        |required, bytes| required.max(admitted(bytes.evaluate(values))),
                    );
                    match &plan.kind {
                        ExecutableAllocationKind::Global(
                            ExecutableGlobalAllocationKind::Argument(_),
                        ) => 0,
                        ExecutableAllocationKind::Global(
                            ExecutableGlobalAllocationKind::Persistent,
                        ) if matches!(
                            persistent.get(&(variant_index, index)),
                            Some(PersistentSlot::Ready { allocation, .. })
                                if allocation.capacity >= required
                        ) =>
                        {
                            0
                        }
                        ExecutableAllocationKind::Global(_)
                        | ExecutableAllocationKind::LaunchScratch { .. }
                        | ExecutableAllocationKind::KernelAbi { .. } => required,
                    }
                })
                .try_fold(0u64, u64::checked_add)
                .unwrap_or_else(|| panic!("closed invocation allocation total overflowed u64"));
        }
    }
    fn decode_values(
        &self,
        staged: &Staged<B>,
        device: &Arc<crate::api::device::DeviceInner>,
    ) -> Vec<DecodedValue> {
        let variant = &self.kernel.variants().as_slice()[staged.variant];
        let mut results = Vec::with_capacity(variant.bindings().results.len());
        for (_path, binding) in &variant.bindings().results {
            match binding {
                ExecutableResultBinding::Buffer { view, bytes } => {
                    let runtime = &staged.buffers[view.allocation_index()];
                    let relative = admitted(view.byte_offset.evaluate(&staged.values));
                    let byte_offset = runtime
                        .base_offset
                        .checked_add(relative)
                        .expect("guarded result view offset overflowed");
                    let byte_len = admitted(bytes.evaluate(&staged.values));
                    let extents = view
                        .extents
                        .iter()
                        .map(|v| admitted(v.evaluate(&staged.values)))
                        .collect();
                    let strides = view
                        .strides
                        .iter()
                        .map(|v| admitted(v.evaluate(&staged.values)))
                        .collect();
                    results.push(DecodedValue::Tensor(Arc::new(TensorInner::new_view(
                        device.clone(),
                        staged.allocations[view.allocation_index()].clone(),
                        byte_offset,
                        byte_len,
                        view.representation,
                        extents,
                        strides,
                    ))));
                }
                ExecutableResultBinding::Scalar {
                    slot,
                    kind: ExecutableScalarResultKind::Value(dtype),
                } => results.push(DecodedValue::Scalar(scalar_result(
                    *dtype,
                    slot.symbol,
                    &staged.values,
                ))),
                ExecutableResultBinding::Scalar {
                    slot,
                    kind: ExecutableScalarResultKind::Index,
                } => {
                    let Some(SymbolValue::Nat(value)) = staged.values.get(slot.symbol) else {
                        panic!("closed index result slot has the wrong scalar sort")
                    };
                    results.push(DecodedValue::Scalar(ArgumentValue::Index(value)));
                }
                ExecutableResultBinding::Range { start, end } => {
                    let (Some(SymbolValue::Nat(start)), Some(SymbolValue::Nat(end))) = (
                        staged.values.get(start.symbol),
                        staged.values.get(end.symbol),
                    ) else {
                        panic!("closed range result slots have the wrong scalar sort")
                    };
                    results.push(DecodedValue::Scalar(ArgumentValue::Range { start, end }));
                }
            }
        }
        results
    }

    fn decode(
        &self,
        staged: &Staged<B>,
        device: &Arc<crate::api::device::DeviceInner>,
    ) -> DecodedResults {
        DecodedResults::new(self.decode_values(staged, device))
    }

    fn workflow_tensor_results(
        &self,
        staged: &Staged<B>,
        device: &Arc<crate::api::device::DeviceInner>,
    ) -> Vec<Option<Arc<TensorInner>>> {
        let variant = &self.kernel.variants().as_slice()[staged.variant];
        variant
            .bindings()
            .results
            .iter()
            .map(|(_, binding)| match binding {
                ExecutableResultBinding::Buffer { view, bytes } => {
                    let runtime = &staged.buffers[view.allocation_index()];
                    let relative = admitted(view.byte_offset.evaluate(&staged.values));
                    let byte_offset = runtime
                        .base_offset
                        .checked_add(relative)
                        .expect("guarded workflow result view offset overflowed");
                    let byte_len = admitted(bytes.evaluate(&staged.values));
                    let extents = view
                        .extents
                        .iter()
                        .map(|value| admitted(value.evaluate(&staged.values)))
                        .collect();
                    let strides = view
                        .strides
                        .iter()
                        .map(|value| admitted(value.evaluate(&staged.values)))
                        .collect();
                    Some(Arc::new(TensorInner::new_view(
                        device.clone(),
                        staged.allocations[view.allocation_index()].clone(),
                        byte_offset,
                        byte_len,
                        view.representation,
                        extents,
                        strides,
                    )))
                }
                ExecutableResultBinding::Scalar { .. } | ExecutableResultBinding::Range { .. } => {
                    None
                }
            })
            .collect()
    }
}

impl<B: Backend> PreparedHandle<B> {
    pub(crate) fn call(&self, args: EncodedArgs) -> Result<DecodedResults, CallError> {
        let attributes = self.prepared.attributes();
        let mut span = Timed::start("seismic.call", attributes.clone());
        let bound = WorkflowDraft::one(self).close().bind(args)?;
        for dimension in self.prepared.kernel.schema().dimensions() {
            if let Some(SymbolValue::Nat(extent)) = bound.values.get(dimension.symbol) {
                span.attribute(key_u64(format!("seismic.dim.{}", dimension.name), extent));
            }
        }
        let admitted = bound.admit(&mut span)?;
        let allocated_bytes = admitted.staged.allocated_bytes;
        let results = admitted.submit()?.complete()?;
        span.attribute(key_bool("seismic.ok", true));
        telemetry::record_call(span.elapsed_ms(), allocated_bytes, &attributes);
        Ok(results)
    }
}

fn collect_access(schema: &CallSchema, args: &EncodedArgs) -> Vec<(Arc<Allocation>, bool)> {
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

fn acquire_access(access: &[(Arc<Allocation>, bool)]) -> Vec<AllocationPermit> {
    access
        .iter()
        .map(|(allocation, write)| allocation.acquire(*write))
        .collect()
}

fn copy_between<B: Backend>(
    service: &Service<B>,
    source: &Buffer<B>,
    destination: &Buffer<B>,
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

fn scalar_result(
    dtype: seismic_lang::types::DType,
    symbol: seismic_lang::expr::SymbolId,
    values: &InvocationValues,
) -> ArgumentValue {
    match (dtype, values.get(symbol)) {
        (seismic_lang::types::DType::F32, Some(SymbolValue::F32(v))) => ArgumentValue::F32(v),
        (seismic_lang::types::DType::F16, Some(SymbolValue::F16(v))) => ArgumentValue::F16(v),
        (seismic_lang::types::DType::BF16, Some(SymbolValue::BF16(v))) => ArgumentValue::BF16(v),
        (seismic_lang::types::DType::I32, Some(SymbolValue::I32(v))) => ArgumentValue::I32(v),
        (seismic_lang::types::DType::U32, Some(SymbolValue::U32(v))) => ArgumentValue::U32(v),
        (seismic_lang::types::DType::Bool, Some(SymbolValue::Bool(v))) => ArgumentValue::Bool(v),
        _ => panic!("closed scalar result slot has the wrong scalar sort"),
    }
}
