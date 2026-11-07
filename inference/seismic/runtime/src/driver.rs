//! The single backend-generic preparation and synchronous invocation driver.
//! Planning ends at `PreparedKernel`; this module validates public arguments,
//! evaluates a selected executable, owns storage, and executes its typed schedule.

use crate::completion::DeviceCompletion;
use crate::api::kernel::{
    DecodedResults, DecodedValue, EncodedArgs, EncodedWorkflowArgs, EncodedWorkflowArgument,
    PendingWorkflowResults, PrepareError, WorkflowTensorArgument,
};
use crate::api::tensor::TensorInner;
use crate::api::CallError;
use crate::memory::{DeviceLedgers, MemoryCharge, MemoryReservation, Placement};
use seismic_reservation::Forming;
use crate::native::NativeQueue;
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
use seismic_lang::entry::{CallSchema, ElementBindings, LogicalEntry, ParameterKind, TensorAccess};
use seismic_lang::expr::compiled::InvocationValues;
use seismic_lang::expr::SymbolValue;
use seismic_lang::ids::{EntryId, ModuleHash, RepresentationId, StableEntryId};
use seismic_lang::precision::PrecisionPolicy;
use seismic_lang::registry;
use seismic_native_target::{DeviceDescription, TargetFamily};
use std::any::Any;
use std::borrow::Borrow;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

pub(crate) type Service<T, E> = <E as NativeExecutor<T>>::Device;
pub(crate) type Buffer<T, E> = <Service<T, E> as DeviceService<T>>::Buffer;
pub(crate) type Writable<T, E> = <Service<T, E> as DeviceService<T>>::Writable;

static NEXT_DEVICE: AtomicU64 = AtomicU64::new(1);
static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);
static NEXT_PREPARED: AtomicU64 = AtomicU64::new(1);

pub(crate) fn fresh_allocation_identity() -> u64 {
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
    /// Its ledgers.
    memory: Arc<DeviceLedgers>,
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
        memory: Arc<DeviceLedgers>,
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
    /// What the backend holds for fixed buffers of `contract` beyond their
    /// allocations: for `DeviceOnly`, its blocks' free space and its driver
    /// objects (such as instantiated CUDA graphs). These backends allocate
    /// the host-mapped contracts whole, one native allocation per tensor,
    /// holding nothing beyond their charges.
    pub(crate) fn backend_held(&self, contract: seismic_topology::Contract) -> crate::devices::BackendHeld {
        match contract {
            seismic_topology::Contract::DeviceOnly => crate::devices::BackendHeld {
                unallocated: self.service.unallocated(),
                beyond: self.service.held_beyond_allocations(),
            },
            seismic_topology::Contract::Upload | seismic_topology::Contract::Shared | seismic_topology::Contract::Words => {
                crate::devices::BackendHeld::default()
            }
        }
    }

    /// The ledgers its allocations are charged to.
    pub(crate) fn memory(&self) -> &Arc<DeviceLedgers> {
        &self.memory
    }
    pub(crate) fn allocate_storage(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        self.allocate_storage_in(bytes, alignment, None)
    }

    /// Charge and allocate device storage, ordered by `access`'s hazard
    /// domain where given (a slab its address table's, or a table that of
    /// the storage its slabs view), otherwise by a domain of its own.
    pub(crate) fn allocate_storage_in(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        access: Option<Arc<AccessGroup>>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        // The device memory this allocation may add: what the ledgers let
        // the backend hold beyond what it holds now.
        let room = Some(self.memory.device().physical().available());
        let mut reservation = reserve(self.memory.device(), bytes)?;
        allocate_reserved_in(self.allocation_limits(), bytes, alignment, &mut reservation, access, || {
            Ok(buffer_storage::<T, E>(&self.service, self.service.allocate_within(bytes, alignment, room)?))
        })
    }

    /// Charge `bytes` of a host-visible `contract` and allocate it by
    /// `make`.
    pub(crate) fn allocate_visible_with(
        self: &Arc<Self>,
        contract: seismic_topology::Contract,
        bytes: u64,
        alignment: u64,
        make: impl FnOnce(&Service<T, E>) -> Result<Writable<T, E>, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let mut reservation = reserve(self.memory.placement(seismic_topology::Placing::Fixed(contract)), bytes)?;
        self.allocate_reserved_with(bytes, alignment, &mut reservation, make)
    }

    /// Charge and allocate the storage of a reserved tensor, one row per
    /// record of `reservation`: the backend reserved `reservation` and
    /// backs its leading `committed` bytes (`make`, given chunked storage's
    /// record table, a charged allocation of its own), and the returned
    /// [`Resizer`] resizes it in place. Only the backed granules are charged,
    /// to the placement of the reservation's contract, and only they are
    /// allocated: the device's single-allocation limit bounds each native
    /// allocation backing them (all of them, or one chunk of chunked
    /// storage), never the address range.
    pub(crate) fn allocate_reservation_with(
        self: &Arc<Self>,
        reservation: ReservedRange,
        committed: u64,
        alignment: u64,
        make: impl FnOnce(
            &Arc<Service<T, E>>,
        ) -> Result<
            (Box<dyn Storage>, crate::backends::ReservedResizer, Option<Writable<T, E>>),
            ExecutionError,
        >,
    ) -> Result<Resizer, ExecutionError> {
        allocate_reservation(self.allocation_limits(), &self.memory, reservation, committed, alignment, || {
            let (storage, backend, table) = make(&self.service)?;
            let table = table.map(|table| {
                (self.service.buffer_len(table.borrow()), buffer_storage::<T, E>(&self.service, table))
            });
            Ok((storage, backend, table))
        })
    }

    fn allocation_limits(&self) -> AllocationLimits {
        let limits = self.device_description().limits();
        let natural_max = if limits.max_index_bits >= 64 {
            u64::MAX
        } else {
            (1u64 << limits.max_index_bits) - 1
        };
        AllocationLimits {
            max_allocation_bytes: limits.max_allocation_bytes.min(natural_max),
            max_allocation_alignment: limits.max_allocation_alignment,
        }
    }

    pub(crate) fn allocate_read_only_storage_with(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        make: impl FnOnce(&Service<T, E>) -> Result<Buffer<T, E>, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        let mut reservation = reserve(self.memory.device(), bytes)?;
        allocate_reserved(self.allocation_limits(), bytes, alignment, &mut reservation, || {
            let buffer = make(&self.service)?;
            Ok(Box::new(ReadOnlyStorage::<T, E> {
                service: self.service.clone(),
                buffer,
            }))
        })
    }

    pub(crate) fn allocate_reserved(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        // What the ledgers let the backend hold beyond what it holds now,
        // the reservation's untaken bytes among it.
        let room = Some(self.memory.device().physical().available());
        self.allocate_reserved_with(bytes, alignment, reservation, |service| {
            service.allocate_within(bytes, alignment, room)
        })
    }

    fn allocate_reserved_with(
        self: &Arc<Self>,
        bytes: u64,
        alignment: u64,
        reservation: &mut MemoryReservation,
        make: impl FnOnce(&Service<T, E>) -> Result<Writable<T, E>, ExecutionError>,
    ) -> Result<Arc<Allocation>, ExecutionError> {
        allocate_reserved(self.allocation_limits(), bytes, alignment, reservation, || {
            Ok(buffer_storage::<T, E>(&self.service, make(&self.service)?))
        })
    }
}

/// What one device admits for a single allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AllocationLimits {
    pub(crate) max_allocation_bytes: u64,
    pub(crate) max_allocation_alignment: u64,
}

impl AllocationLimits {
    /// Refuse an allocation spanning `bytes` at `alignment` beyond these
    /// limits.
    fn admit(&self, bytes: u64, alignment: u64) -> Result<(), ExecutionError> {
        self.admit_bytes(bytes)?;
        if !alignment.is_power_of_two() || alignment > self.max_allocation_alignment {
            return Err(ExecutionError::ConstructionContradiction(format!(
                "allocation alignment {alignment} exceeds target contract {}",
                self.max_allocation_alignment
            )));
        }
        Ok(())
    }

    /// Refuse an allocation of `bytes` beyond the single-allocation limit.
    fn admit_bytes(&self, bytes: u64) -> Result<(), ExecutionError> {
        if bytes > self.max_allocation_bytes {
            return Err(ExecutionError::AllocationCapacity {
                required: bytes.into(),
                available: self.max_allocation_bytes,
            });
        }
        Ok(())
    }
}

/// Reserve `bytes` in `memory` for an allocation.
pub(crate) fn reserve(
    memory: &Arc<Placement>,
    bytes: u64,
) -> Result<MemoryReservation, ExecutionError> {
    memory
        .reserve(bytes)
        .map_err(|capacity| ExecutionError::AllocationCapacity {
            required: capacity.required.into(),
            available: capacity.available,
        })
}

/// What a backend forms for a reserved tensor: its storage, its resizer,
/// and on chunked storage its record table's bytes and storage.
pub(crate) type ReservedParts = (
    Box<dyn Storage>,
    crate::backends::ReservedResizer,
    Option<(u64, Box<dyn Storage>)>,
);

/// The backend-neutral core of a reserved tensor's allocation
/// (`Opened::allocate_reservation_with`): its committed backing is admitted
/// against `limits` (the largest native allocation backing it: all of it,
/// or one chunk of chunked storage), as is its table. Each is charged to
/// the placement of its own storage kind; a table in the backing's own
/// placement shares the backing's grant, a table placed elsewhere gets one
/// of its own. Every grant is taken before `make` forms storage, and the
/// returned [`Resizer`] resizes it in place.
pub(crate) fn allocate_reservation(
    limits: AllocationLimits,
    memory: &DeviceLedgers,
    reservation: ReservedRange,
    committed: u64,
    alignment: u64,
    make: impl FnOnce() -> Result<ReservedParts, ExecutionError>,
) -> Result<Resizer, ExecutionError> {
    let rows = reservation.rows();
    let overflow = || ExecutionError::AllocationFailed(format!("the charge of a {rows}-record reservation overflows"));
    let charges = reservation.span.charges(reservation.contract, reservation.records(), rows, committed)
        .ok_or_else(overflow)?;
    limits.admit(reservation.largest_allocation(charges.backing), alignment)?;
    if let Some(table) = charges.table {
        limits.admit(table.bytes(), seismic_lang::registry::RECORD_TABLE_ENTRY_BYTES)?;
    }
    let backing_memory = memory.placement(seismic_topology::Placing::Reserved(reservation.contract));
    // A table charged with the backing takes the backing's grant; one
    // placed elsewhere a grant of its own.
    let (backing_request, mut table_grant) = match charges.table {
        Some(seismic_topology::TableCharge::Separate { placement, bytes }) =>
            (charges.backing, Some(reserve(memory.placement(seismic_topology::Placing::Fixed(placement)), bytes)?)),
        Some(seismic_topology::TableCharge::WithBacking { .. }) | None =>
            (charges.total().ok_or_else(overflow)?, None),
    };
    let mut backing_grant = reserve(backing_memory, backing_request)?;
    let (storage, backend, table) = make()?;
    // The table and the backing it addresses share one hazard domain.
    let access = Arc::new(AccessGroup::default());
    let table = table.map(|(bytes, storage)| {
        let grant = table_grant.as_mut().unwrap_or(&mut backing_grant);
        Allocation::new_in(fresh_allocation_identity(), bytes, grant.take(bytes), storage, Some(access.clone()))
    });
    let allocation = Allocation::new_in(fresh_allocation_identity(), committed,
        backing_grant.take(charges.backing), storage, Some(access))
        .reserving(reservation, table);
    Ok(Resizer::new(allocation, backing_memory.clone(), limits, backend))
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
    allocate_reserved_in(limits, bytes, alignment, reservation, None, make)
}

/// [`allocate_reserved`], ordered by `access`'s hazard domain where given.
pub(crate) fn allocate_reserved_in(
    limits: AllocationLimits,
    bytes: u64,
    alignment: u64,
    reservation: &mut MemoryReservation,
    access: Option<Arc<AccessGroup>>,
    make: impl FnOnce() -> Result<Box<dyn Storage>, ExecutionError>,
) -> Result<Arc<Allocation>, ExecutionError> {
    limits.admit(bytes, alignment)?;
    let storage = make()?;
    Ok(Allocation::new_in(
        fresh_allocation_identity(),
        bytes,
        reservation.take(bytes),
        storage,
        access,
    ))
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
    fn read_only(&self) -> bool {
        false
    }
    /// The host address of the storage's first byte where the host reads
    /// and writes it in place (`DeviceService::host_address`, and never for
    /// read-only storage), valid while the storage lives.
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>>;
}

/// Storage the service allocated, which the host reads and writes.
struct TypedStorage<T: TargetFamily, E: NativeExecutor<T>> {
    service: Arc<Service<T, E>>,
    buffer: Writable<T, E>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> Storage for TypedStorage<T, E> {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.service.read(self.buffer.borrow(), offset, into)
    }
    fn write(&self, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.service.write(&self.buffer, offset, bytes)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        self.service.host_address(&self.buffer)
    }
}

/// Read-only mapped host memory: a buffer device work binds and the host
/// only reads, so it holds no writable form.
struct ReadOnlyStorage<T: TargetFamily, E: NativeExecutor<T>> {
    service: Arc<Service<T, E>>,
    buffer: Buffer<T, E>,
}

impl<T: TargetFamily, E: NativeExecutor<T>> Storage for ReadOnlyStorage<T, E> {
    fn read(&self, offset: u64, into: &mut [u8]) -> Result<(), ExecutionError> {
        self.service.read(&self.buffer, offset, into)
    }
    fn write(&self, _: u64, _: &[u8]) -> Result<(), ExecutionError> {
        Err(ExecutionError::SubmissionFailed(
            "write to read-only mapped storage".into(),
        ))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn read_only(&self) -> bool {
        true
    }
    fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
        None
    }
}

/// One physical allocation. Views share it, so access exclusion is allocation-wide.
pub(crate) struct Allocation {
    identity: u64,
    /// Backed bytes. Only an installed resize of a reservation changes them,
    /// under exclusive access.
    bytes: AtomicU64,
    /// The address range the backend reserved for an in-place recommit.
    reservation: Option<ReservedRange>,
    /// A chunked reservation's record table: one device address per record
    /// of the range, written by its backing as chunks are formed, in this
    /// allocation's access group.
    records: Option<Arc<Allocation>>,
    /// The record tables of the views record ports bind
    /// ([`Allocation::view_table`]), one for each record size and phase,
    /// formed once and held with the storage they address.
    view_tables: Mutex<Vec<ViewTable>>,
    /// The record tables graph workspaces and pinned views of reserved
    /// storage formed over this allocation ([`Allocation::hold_table`]),
    /// each part of its storage while it lives: charged and observed with
    /// it.
    tables: Mutex<Vec<Weak<Allocation>>>,
    /// What the backing is charged: its bytes, or a reservation's backed
    /// granules.
    charge: Mutex<MemoryCharge>,
    /// The end of each live region of a reservation's backed bytes
    /// ([`Allocation::pin`]): a recommit backs no fewer.
    pins: Mutex<Vec<u64>>,
    storage: Box<dyn Storage>,
    access: Arc<AccessGroup>,
}

/// The right to resize one reserved tensor's committed rows in place
/// (`Tensor::reserved` hands it out with the tensor). A resize consumes it;
/// the resize's release, abandonment or refusal returns it, so the tensor's
/// resizes follow one another. It and each step of a resize are `Send`: a
/// resize is prepared and released off the thread that submits device work,
/// which installs a shrink between submissions; a growth installs on any
/// thread ([`PreparedRecommit::install`]). Once installed the tensor, every
/// view of it, bound work and sealed graphs stay valid, at the same address
/// on CUDA and the CPU backend (on Metal and Vulkan, through the record
/// table its backing keeps as chunks change);
/// kept rows keep their contents and rows past the old commitment read zero.
pub struct Resizer {
    allocation: Arc<Allocation>,
    /// The device's ledger and single-allocation limit, which a growth is
    /// admitted and charged against.
    memory: Arc<Placement>,
    limits: AllocationLimits,
    backend: crate::backends::ReservedResizer,
}

/// A refused resize: why, and the resizer back.
pub struct RecommitRefusal {
    pub resizer: Resizer,
    pub error: crate::api::TensorError,
}

impl std::fmt::Debug for RecommitRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecommitRefusal").field("error", &self.error).finish()
    }
}

impl Resizer {
    pub(crate) fn new(
        allocation: Arc<Allocation>,
        memory: Arc<Placement>,
        limits: AllocationLimits,
        backend: crate::backends::ReservedResizer,
    ) -> Self {
        Self {
            allocation,
            memory,
            limits,
            backend,
        }
    }

    pub(crate) fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }

    /// Prepare backing the tensor's leading `committed` rows in place,
    /// beside device work that uses it: nothing any view, bound work or the
    /// host sees changes until [`PreparedRecommit::install`]. A growth is
    /// admitted and granted here and its memory backed (on Metal, its new
    /// chunks' buffers formed). Committing no rows, more than the
    /// reservation spans, beyond the device's single-allocation limit or
    /// fewer than a live region pins is refused, the tensor as it was.
    /// Where the backend binds a range whole (Metal), the view's form
    /// changes as [`Self::admit`] chooses, within the grant.
    pub fn prepare(self, committed: u64) -> Result<PreparedRecommit, RecommitRefusal> {
        let Admitted { bytes, backing, grant, form, forming } = match self.admit(committed) {
            Ok(admitted) => admitted,
            Err(error) => return Err(RecommitRefusal { resizer: self, error }),
        };
        let Self {
            allocation,
            memory,
            limits,
            backend,
        } = self;
        match backend.prepare(bytes, form, forming) {
            Ok(prepared) => {
                // Work submitted from here on is committed with what the
                // prepare formed in the queue's residency set; what was
                // submitted before is read once no encoding holds a permit.
                let before = (prepared.rebinding() == seismic_reservation::Rebinding::Beside).then(|| {
                    let _encodings = allocation.acquire_exclusive();
                    allocation.last_device_uses()
                });
                Ok(PreparedRecommit {
                    next: Some(Next {
                        resizer: Parts {
                            allocation,
                            memory,
                            limits,
                        },
                        bytes,
                        backing,
                        grant,
                        backend: prepared,
                        before,
                    }),
                })
            }
            Err((backend, error)) => Err(RecommitRefusal {
                resizer: Parts {
                    allocation,
                    memory,
                    limits,
                }
                .with(backend),
                error: error.into(),
            }),
        }
    }

    /// Back the tensor's leading `committed` rows in place at once:
    /// [`Self::prepare`], its install, then its release.
    pub fn recommit(self, committed: u64) -> Result<Self, RecommitRefusal> {
        Ok(self.prepare(committed)?.install()?.release())
    }

    /// The bytes of `committed` rows, their backing, the grant a resize to
    /// them needs and the view's form, admitted. The bytes are counted once
    /// the rows are known to lie within the reservation, whose bytes were
    /// counted. A growth is granted its new backing. A change between a
    /// view's forms, where the backend binds a range whole, holds the next
    /// form's buffers beside the view's from its prepare to its release,
    /// which its device counts: the lesser form's backing more, granted for
    /// that window. A growth to every granule takes the whole form only
    /// where that is granted, and otherwise stays in chunks, which bind
    /// nothing directly ([`crate::api::Tensor::bindable`]); a shrink from
    /// the whole form, which needs no grant, forms its kept chunks beside
    /// it where that is granted, and otherwise in its place at its install
    /// ([`seismic_reservation::Forming::InPlace`]), which waits for all the
    /// device work using the tensor ([`PreparedRecommit::install`]).
    fn admit(&self, committed: u64) -> Result<Admitted, crate::api::TensorError> {
        let reservation = self
            .allocation
            .reservation()
            .expect("a resizer's allocation is a reservation");
        let bytes = reservation.committed_bytes(committed)?;
        let charge = self
            .allocation
            .charge
            .lock()
            .expect("allocation charge lock poisoned");
        self.allocation.unpinned(bytes)?;
        let backing = reservation.backing(bytes)?;
        let installed = reservation.backing(self.allocation.bytes())?;
        let growth = backing.saturating_sub(charge.bytes());
        let beside = backing.min(installed);
        let admitted = |grant, form, forming| Admitted { bytes, backing, grant, form, forming };
        if self.backend.whole() {
            if backing >= installed {
                return Ok(admitted(None, ReservedForm::Whole, Forming::Beside));
            }
            // `reserve` refuses only for capacity.
            return Ok(match reserve(&self.memory, beside) {
                Ok(grant) => admitted(Some(grant), ReservedForm::Chunks, Forming::Beside),
                Err(_) => admitted(None, ReservedForm::Chunks, Forming::InPlace),
            });
        }
        if growth == 0 {
            return Ok(admitted(None, ReservedForm::Chunks, Forming::Beside));
        }
        self.limits.admit_bytes(reservation.largest_allocation(backing))?;
        if backing > installed && reservation.whole_at(backing)? {
            if let Ok(grant) = reserve(&self.memory, growth + beside) {
                return Ok(admitted(Some(grant), ReservedForm::Whole, Forming::Beside));
            }
        }
        Ok(admitted(Some(reserve(&self.memory, growth)?), ReservedForm::Chunks, Forming::Beside))
    }
}

/// What [`Resizer::admit`] admits.
struct Admitted {
    bytes: u64,
    backing: u64,
    /// The growth's grant, and a change of form's beside it for its window.
    grant: Option<MemoryReservation>,
    form: ReservedForm,
    /// Where a change of form forms its next buffers: in place only for a
    /// shrink from the whole form the grant cannot hold its chunks beside.
    forming: Forming,
}

/// The form a reserved tensor's view takes where its backend binds a range
/// whole (`seismic_metal::ViewForm`); every other backend's view is of one
/// form, [`Self::Chunks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReservedForm {
    /// Its backed granules in their own buffers.
    Chunks,
    /// One buffer over every granule of the range.
    Whole,
}

/// A resizer's parts but its backend's, which each step of a resize changes.
struct Parts {
    allocation: Arc<Allocation>,
    memory: Arc<Placement>,
    limits: AllocationLimits,
}

impl Parts {
    fn with(self, backend: crate::backends::ReservedResizer) -> Resizer {
        Resizer::new(self.allocation, self.memory, self.limits, backend)
    }
}

/// A resize of a reserved tensor prepared beside device work that uses it
/// ([`Resizer::prepare`]): its bytes, a growth's grant, and its backend's
/// prepared commitment. Dropped, it is abandoned, its resizer with it.
pub struct PreparedRecommit {
    next: Option<Next>,
}

struct Next {
    resizer: Parts,
    bytes: u64,
    /// The backing of `bytes`, as admitted.
    backing: u64,
    grant: Option<MemoryReservation>,
    backend: crate::backends::ReservedPrepared,
    /// Where the prepare formed what the install moves rows device work
    /// already reaches to (`seismic_reservation::Rebinding::Beside`): the
    /// device work submitted before, which the install waits for.
    before: Option<DeviceUses>,
}

impl PreparedRecommit {
    /// Publish the resize: device work encoded from now sees the new
    /// commitment, while work submitted before keeps the old one, whose
    /// bytes beyond the new stay backed until the returned release. It waits
    /// for host access and encoding of the tensor, never for device work. A
    /// growth's grant becomes its charge. A region pinned since the prepare
    /// refuses a shrink below it, abandoning it. A shrink installs between
    /// submissions (never between queuing a sequence that binds the tensor
    /// and submitting it), and the caller guarantees that no device work
    /// submitted afterwards touches rows it released. A growth installs on
    /// any thread, beside submission: its prepare formed everything device
    /// work reaches its new rows through (CUDA's mapped granules, the CPU
    /// backend's pages, Metal's chunks wired into the queue's residency set,
    /// Vulkan's chunks with their record table entries), and its install
    /// only publishes the length (on Metal it also appends the formed chunks
    /// to the view). Its new rows are the backing's for device work
    /// submitted after the install: an encoding holds its permit until its
    /// work is committed and recorded as a use, and the install waits for
    /// every permit, so work submitted before it is among the uses its
    /// release waits for. On Metal a command buffer takes the queue's residency sets
    /// as they are when it commits, and whether a set's later commit reaches
    /// one already committed Metal does not state. Every resize's release
    /// returns once the device work submitted before its install completes,
    /// so a holder that hands a growth's new rows to device work only after
    /// its release (`magnitude_executor`'s growth joins) lets no earlier
    /// work reach them, on any backend. A resize that moves rows device work
    /// already reaches (Metal's change between its chunks and one buffer
    /// over a whole range) first waits, before taking the tensor, for the
    /// device work submitted before its prepare, which may not reach where
    /// they move; one that forms them in place of the view's
    /// ([`Forming::InPlace`]) waits, holding the tensor, for all the device
    /// work using it. At a boundary between flights there is none; a resize
    /// within a flight (a cut of a running chain) stalls there until the
    /// running work ends, which is taken only because it happens only where
    /// the grant refused the chunks beside the view.
    pub fn install(mut self) -> Result<RecommitRelease, RecommitRefusal> {
        let next = self.next.take().expect("a resize installs once");
        if let Some(before) = &next.before {
            before.retire();
        }
        let allocation = next.resizer.allocation.clone();
        let exclusive = allocation.acquire_exclusive();
        if next.backend.rebinding() == seismic_reservation::Rebinding::InPlace {
            allocation.last_device_uses().retire();
        }
        let mut charge = allocation
            .charge
            .lock()
            .expect("allocation charge lock poisoned");
        if let Err(error) = allocation.unpinned(next.bytes) {
            drop((charge, exclusive));
            return Err(RecommitRefusal {
                resizer: next.resizer.with(next.backend.abandon()),
                error,
            });
        }
        let backend = next.backend.install();
        let backing = next.backing;
        let mut beside = next.grant;
        if let Some(grant) = beside.as_mut() {
            let growth = backing.saturating_sub(charge.bytes());
            grant.extend(&mut charge, growth);
        }
        allocation.bytes.store(next.bytes, Ordering::Release);
        let submitted = allocation.last_device_uses();
        drop((charge, exclusive));
        Ok(RecommitRelease {
            released: Some(Released {
                resizer: next.resizer,
                backing,
                submitted,
                backend,
                beside,
            }),
        })
    }

    /// Abandon the resize: the backend releases what it backed and the grant
    /// returns. It returns the resizer.
    pub fn abandon(mut self) -> Resizer {
        let next = self.next.take().expect("a resize is abandoned once");
        next.resizer.with(next.backend.abandon())
    }
}

impl Drop for PreparedRecommit {
    fn drop(&mut self) {
        if let Some(next) = self.next.take() {
            drop(next.backend.abandon());
        }
    }
}

/// An installed resize's release ([`PreparedRecommit::install`]): what the
/// reservation backs beyond its commitment. Dropped, it is released, its
/// resizer with it.
pub struct RecommitRelease {
    released: Option<Released>,
}

struct Released {
    resizer: Parts,
    /// What the commitment's backing is charged.
    backing: u64,
    /// The last device work of each queue submitted before the install,
    /// which may still use what the release frees.
    submitted: DeviceUses,
    backend: crate::backends::ReservedPending,
    /// What a change of form's buffers held beside the view's are granted,
    /// until the release drops the ones it retired.
    beside: Option<MemoryReservation>,
}

impl RecommitRelease {
    /// Once the device work submitted before the install completes, free
    /// what the reservation backs beyond the tensor's rows; its charge
    /// follows. It returns the resizer.
    pub fn release(mut self) -> Resizer {
        Self::unback(self.released.take().expect("a resize is released once"))
    }

    fn unback(released: Released) -> Resizer {
        released.submitted.retire();
        let backend = released.backend.release();
        let mut charge = released
            .resizer
            .allocation
            .charge
            .lock()
            .expect("allocation charge lock poisoned");
        if charge.bytes() > released.backing {
            charge.shrink(released.backing);
        }
        drop((charge, released.beside));
        released.resizer.with(backend)
    }
}

impl Drop for RecommitRelease {
    fn drop(&mut self) {
        if let Some(released) = self.released.take() {
            drop(Self::unback(released));
        }
    }
}

/// A region's hold on its reservation's backed bytes up to `end`, released
/// when the last view of the region drops.
pub(crate) struct RegionPin {
    allocation: Arc<Allocation>,
    end: u64,
}

impl Drop for RegionPin {
    fn drop(&mut self) {
        let mut pins = self.allocation.pins();
        let index = pins
            .iter()
            .position(|&end| end == self.end)
            .expect("a live pin is listed");
        pins.swap_remove(index);
    }
}

/// The record table of an allocation's `records` `record`-byte records from
/// byte `phase` ([`Allocation::view_table`]).
struct ViewTable {
    record: u64,
    phase: u64,
    records: u64,
    table: Arc<Allocation>,
}

/// An allocation's reserved address range: `bytes` of address space of
/// `record`-byte records, backed over a leading prefix as its device
/// reserves (`span`, of its records), in `kind` storage: device storage, or
/// host storage the host addresses in place.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReservedRange {
    pub(crate) bytes: u64,
    pub(crate) record: std::num::NonZeroU64,
    /// The largest view its records carry beside their own
    /// (`seismic_topology::ReservedRecords`).
    pub(crate) carries: Option<std::num::NonZeroU64>,
    pub(crate) span: seismic_topology::Reservation,
    pub(crate) contract: seismic_topology::ReservedContract,
}

impl ReservedRange {
    /// The range of `rows` records of `records` under `contract`, reserved
    /// as `span` says (its device's reservation, as discovery reports it),
    /// on a device whose largest native allocation is `max_allocation`: a
    /// range past it is never bound as one buffer
    /// (`seismic_topology::Reservation::binds_whole`).
    pub(crate) fn new(
        contract: seismic_topology::ReservedContract,
        span: seismic_topology::Reservation,
        records: seismic_topology::ReservedRecords,
        rows: u64,
        max_allocation: u64,
    ) -> Self {
        let bytes = records.record.get() * rows;
        let mut span = span.of_records(records);
        let whole = span.binds_whole(bytes, max_allocation);
        if let Some(chunks) = span.chunks.as_mut() {
            chunks.whole = whole;
        }
        Self {
            bytes,
            record: records.record,
            carries: records.carries,
            span,
            contract,
        }
    }

    /// What its records are.
    pub(crate) fn records(&self) -> seismic_topology::ReservedRecords {
        seismic_topology::ReservedRecords { record: self.record, carries: self.carries }
    }

    /// The records the range spans.
    fn rows(&self) -> u64 {
        self.bytes / self.record
    }

    /// The bytes of `committed` leading records, admitted: a reserved
    /// tensor commits at least one record and at most the range's.
    pub(crate) fn committed_bytes(&self, committed: u64) -> Result<u64, crate::api::TensorError> {
        let rows = self.rows();
        if committed == 0 || committed > rows {
            return Err(crate::api::TensorError::Uncommittable { committed, rows });
        }
        Ok(self.record.get() * committed)
    }

    /// What backing `bytes` of the range holds, and is charged; refused
    /// past the bytes a `u64` counts.
    fn backing(&self, bytes: u64) -> Result<u64, ExecutionError> {
        self.span.backing(bytes).ok_or_else(|| {
            ExecutionError::AllocationFailed(format!("the backing of {bytes} reserved bytes overflows"))
        })
    }

    /// The largest single native allocation of the range's `backing`.
    fn largest_allocation(&self, backing: u64) -> u64 {
        self.span.largest_allocation(backing)
    }

    /// Whether `backing` may be one buffer over the range: its backend binds
    /// the range whole (`ReservationChunks::whole`, within the device's
    /// largest allocation) and it backs every granule.
    fn whole_at(&self, backing: u64) -> Result<bool, ExecutionError> {
        Ok(self.span.chunks.is_some_and(|chunks| chunks.whole) && backing == self.backing(self.bytes)?)
    }
}

#[derive(Default)]
pub(crate) struct AccessGroup {
    state: Mutex<AllocationAccess>,
    changed: Condvar,
}

/// A queue of one device on which work can use an allocation. Each runs
/// its work in submission order; they run beside each other.
///
/// The native queues order against each other by what their submissions
/// touch ([`Allocation::queue_order`]); the transfer queue is ordered by its
/// caller (`native::transfer`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DeviceQueue {
    Native(NativeQueue),
    /// Host-to-device copies (`native::transfer`).
    Transfer,
}

impl DeviceQueue {
    const COUNT: usize = NativeQueue::ALL.len() + 1;

    fn ordinal(self) -> usize {
        match self {
            Self::Native(queue) => queue as usize,
            Self::Transfer => NativeQueue::ALL.len(),
        }
    }
}

/// The newest submitted work of one queue that uses an allocation, until
/// it ends, and the newest that writes it, until it completes. A queue runs
/// its work in submission order, so the newest fence of a kind ends after
/// every earlier one of that queue: it is the only one of that queue access
/// must wait for. A write that faulted stays, so what reads the allocation
/// after it receives its fault, until a later write of the queue replaces it.
#[derive(Default)]
struct QueueFences {
    last_use: Option<Arc<dyn DeviceCompletion>>,
    last_write: Option<Arc<dyn DeviceCompletion>>,
}

impl QueueFences {
    /// Forget ended uses and completed writes. The newest use ends last, so
    /// once it has, no work of this queue uses the allocation; a faulted
    /// use ends it as completion does.
    fn prune(&mut self) {
        if self.last_use.as_ref().is_some_and(|fence| !matches!(fence.completed(), Ok(false))) {
            self.last_use = None;
        }
        if self.last_write.as_ref().is_some_and(|fence| matches!(fence.completed(), Ok(true))) {
            self.last_write = None;
        }
    }
}

/// The newest work of each queue still using an allocation: every device
/// use ends once these have.
#[derive(Default)]
pub(crate) struct DeviceUses([Option<Arc<dyn DeviceCompletion>>; DeviceQueue::COUNT]);

impl DeviceUses {
    /// Block until every use ends, to release what they use
    /// ([`DeviceCompletion::retire`]).
    pub(crate) fn retire(&self) {
        for fence in self.0.iter().flatten() {
            fence.retire();
        }
    }

    fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }
}

/// Who holds an allocation's permits, by kind: host access, and device
/// work being encoded.
#[derive(Default)]
struct Holders {
    readers: u64,
    writers: u64,
}

impl Holders {
    fn any(&self) -> bool {
        self.readers != 0 || self.writers != 0
    }

    /// Whether access `write` conflicts with these holders: a write with
    /// any, a read with a writer.
    fn conflict(&self, write: bool) -> bool {
        self.writers != 0 || (write && self.readers != 0)
    }
}

/// An allocation's access: its permits' holders and each queue's fences.
/// Host access excludes conflicting host access and the encoding of device
/// work; device encodings exclude conflicting host access but not each
/// other, each queue ordering its own work, native queues ordering against
/// each other by their fences and the caller ordering transfers
/// ([`Allocation::acquire_for_device`]). A resize's
/// install excludes every holder ([`Allocation::acquire_exclusive`]).
#[derive(Default)]
struct AllocationAccess {
    host: Holders,
    device: Holders,
    /// Per [`DeviceQueue`], by its ordinal.
    queues: [QueueFences; DeviceQueue::COUNT],
    /// Whether the protocol of the work that uses it orders its accesses
    /// across device queues ([`Allocation::order_by_protocol`]).
    protocol: bool,
}

impl AllocationAccess {
    fn holders(&mut self, holder: Holder) -> &mut Holders {
        match holder {
            Holder::Host => &mut self.host,
            Holder::Device => &mut self.device,
        }
    }

    fn queue(&mut self, queue: DeviceQueue) -> &mut QueueFences {
        &mut self.queues[queue.ordinal()]
    }

    fn prune(&mut self) {
        self.queues.iter_mut().for_each(QueueFences::prune);
    }

    /// A device fence host access of the given kind must wait for: a host
    /// write waits for every device use to end, a host read for device
    /// writes to complete, of every queue.
    fn conflicting_device(&mut self, write: bool) -> Option<Arc<dyn DeviceCompletion>> {
        self.prune();
        self.queues.iter().find_map(|fences| match write {
            true => fences.last_use.clone(),
            false => fences.last_write.clone(),
        })
    }

    /// The newest use of each queue.
    fn uses(&mut self) -> DeviceUses {
        self.prune();
        DeviceUses(std::array::from_fn(|queue| self.queues[queue].last_use.clone()))
    }
}

impl Allocation {
    /// The bytes the allocation's own storage, one of `memory`'s executor,
    /// charges along each of its access paths, in the order its bytes lie
    /// on them: what a view's byte range is attributed from.
    pub(crate) fn storage_path_charges(
        &self,
        memory: &DeviceLedgers,
    ) -> Vec<(seismic_topology::AccessPathId, u64)> {
        memory.path_charges(&self.charge.lock().expect("allocation charge lock poisoned"))
    }

    /// The bytes the allocation charges along each of its access paths: its
    /// storage's ([`Self::storage_path_charges`]) and its record tables',
    /// part of its storage, held and released with it.
    pub(crate) fn path_charges(
        &self,
        memory: &DeviceLedgers,
    ) -> Vec<(seismic_topology::AccessPathId, u64)> {
        let mut charges = self.storage_path_charges(memory);
        for (path, bytes) in self.record_tables().iter().flat_map(|table| table.path_charges(memory)) {
            match charges.iter_mut().find(|(charged, _)| *charged == path) {
                Some((_, charged)) => {
                    *charged = charged
                        .checked_add(bytes)
                        .expect("an allocation's charges along a path sum within u64")
                }
                None => charges.push((path, bytes)),
            }
        }
        charges
    }
    #[cfg(test)]
    fn new(
        identity: u64,
        bytes: u64,
        charge: MemoryCharge,
        storage: Box<dyn Storage>,
    ) -> Arc<Self> {
        Self::new_in(identity, bytes, charge, storage, None)
    }

    /// A fresh allocation in `access`'s hazard domain, or one of its own:
    /// the domain is fixed at construction, before anything can use it.
    fn new_in(
        identity: u64,
        bytes: u64,
        charge: MemoryCharge,
        storage: Box<dyn Storage>,
        access: Option<Arc<AccessGroup>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            identity,
            bytes: AtomicU64::new(bytes),
            reservation: None,
            records: None,
            view_tables: Mutex::new(Vec::new()),
            tables: Mutex::new(Vec::new()),
            charge: Mutex::new(charge),
            storage,
            access: access.unwrap_or_default(),
            pins: Mutex::new(Vec::new()),
        })
    }

    /// This fresh allocation as the backed prefix of `reservation`, with its
    /// record table where its storage is chunked.
    fn reserving(
        mut self: Arc<Self>,
        reservation: ReservedRange,
        records: Option<Arc<Allocation>>,
    ) -> Arc<Self> {
        debug_assert!(records.as_ref().is_none_or(|table| table.shares_access_with(&self)));
        let fresh = Arc::get_mut(&mut self).expect("a fresh allocation is unshared");
        fresh.reservation = Some(reservation);
        fresh.records = records;
        self
    }

    /// A chunked reservation's record table ([`Allocation::records`]).
    pub(crate) fn record_table(&self) -> Option<&Arc<Allocation>> {
        self.records.as_ref()
    }
    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }
    /// Backed bytes; a reservation's change only under exclusive access.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Acquire)
    }
    /// Pin a reservation's backed bytes up to `end` while the pin lives;
    /// `None` when `end` reaches past them. The charge lock orders it with
    /// recommits.
    pub(crate) fn pin(self: &Arc<Self>, end: u64) -> Option<RegionPin> {
        let _charge = self.charge.lock().expect("allocation charge lock poisoned");
        (end <= self.bytes()).then(|| {
            self.pins().push(end);
            RegionPin {
                allocation: self.clone(),
                end,
            }
        })
    }
    fn pins(&self) -> std::sync::MutexGuard<'_, Vec<u64>> {
        self.pins.lock().expect("allocation pins lock poisoned")
    }
    /// Refuse backing fewer than `bytes` than a live region pins. The caller
    /// holds the charge lock, which orders it with pinning.
    fn unpinned(&self, bytes: u64) -> Result<(), crate::api::TensorError> {
        match self.pins().iter().max() {
            Some(&pinned) if bytes < pinned => Err(crate::api::TensorError::Region(
                crate::api::RegionError::Pinned { pinned, bytes },
            )),
            _ => Ok(()),
        }
    }
    pub(crate) fn reservation(&self) -> Option<ReservedRange> {
        self.reservation
    }
    /// Bytes a view of the allocation may span: a reservation's address
    /// range, otherwise the backed bytes.
    pub(crate) fn span(&self) -> u64 {
        self.reservation
            .map_or_else(|| self.bytes(), |reservation| reservation.bytes)
    }
    /// What the allocation is charged, its record tables included: part of
    /// its storage, held and released with it.
    pub(crate) fn charged_bytes(&self) -> u64 {
        let storage = self
            .charge
            .lock()
            .expect("allocation charge lock poisoned")
            .bytes();
        self.record_tables()
            .iter()
            .try_fold(storage, |charged, table| charged.checked_add(table.charged_bytes()))
            .expect("an allocation's charges sum within u64")
    }

    /// Hold `table`, a record table a graph workspace or a pinned view of
    /// reserved storage formed over this allocation's records, as part of
    /// its storage while the table lives.
    pub(crate) fn hold_table(&self, table: &Arc<Allocation>) {
        let mut tables = self.tables.lock().expect("allocation tables lock poisoned");
        tables.retain(|held| held.strong_count() > 0);
        tables.push(Arc::downgrade(table));
    }

    /// The table a record port binds `count` of this fixed allocation's
    /// `record`-byte records from byte `offset` through, and the byte of
    /// its first entry: a table the storage holds whose entries are those
    /// records' addresses in turn. That is the table of records of their
    /// size and phase (`offset % record`), entry `i` record `i`'s; a single
    /// record is also found in any held table with an entry at its address.
    /// Formed by `form` (given the phase and the records from it) where
    /// none is held, then held with the storage. So the tables a storage
    /// holds are those of the record sizes and phases its views are bound
    /// at, and a storage whose first table is of its finest records (a
    /// streamed table's slots, held from its first by `hold_records`) holds
    /// no other for single records at their boundaries. A view formed per
    /// run (a landed read's rows) allocates nothing, and no table outlives
    /// its storage's use: a table a run's view alone held would be freed
    /// after the run's submission, waiting for every device use of the
    /// storage it shares its access with, the run's own included.
    pub(crate) fn view_table<E>(
        &self,
        record: u64,
        offset: u64,
        count: u64,
        form: impl FnOnce(u64, u64) -> Result<Arc<Allocation>, E>,
    ) -> Result<(Arc<Allocation>, u64), E> {
        let entry = |held: &ViewTable| {
            seismic_lang::registry::record_table_entry((offset - held.phase) / held.record)
        };
        let phase = offset % record;
        let mut tables = self.view_tables.lock().expect("allocation view tables lock poisoned");
        let spaced = tables
            .iter()
            .find(|held| held.record == record && held.phase == phase);
        let single = || {
            tables.iter().find(|held| {
                count == 1
                    && offset >= held.phase
                    && (offset - held.phase) % held.record == 0
                    && (offset - held.phase) / held.record < held.records
            })
        };
        if let Some(held) = spaced.or_else(single) {
            return Ok((held.table.clone(), entry(held)));
        }
        let records = (self.bytes() - phase) / record;
        let held = ViewTable {
            record,
            phase,
            records,
            table: form(phase, records)?,
        };
        let found = (held.table.clone(), entry(&held));
        tables.push(held);
        Ok(found)
    }

    /// The record tables over this allocation's records that live: a
    /// chunked reservation's own, its views', and those graph workspaces
    /// and pinned views of reserved storage formed.
    fn record_tables(&self) -> Vec<Arc<Allocation>> {
        let views = self.view_tables.lock().expect("allocation view tables lock poisoned");
        let tables = self.tables.lock().expect("allocation tables lock poisoned");
        self.records
            .iter()
            .cloned()
            .chain(views.iter().map(|held| held.table.clone()))
            .chain(tables.iter().filter_map(Weak::upgrade))
            .collect()
    }
    pub(crate) fn storage(&self) -> &dyn Storage {
        &*self.storage
    }
    pub(crate) fn access_group(&self) -> &Arc<AccessGroup> {
        &self.access
    }

    pub(crate) fn access_identity(&self) -> usize {
        Arc::as_ptr(self.access_group()) as usize
    }

    pub(crate) fn shares_access_with(&self, other: &Arc<Allocation>) -> bool {
        Arc::ptr_eq(self.access_group(), other.access_group())
    }

    /// Host access: waits for conflicting host permits and for submitted
    /// device work that conflicts with the access (a host read waits for
    /// device writes; a host write waits for every device use). A read
    /// after a device write that faulted reports its fault.
    pub(crate) fn acquire(self: &Arc<Self>, write: bool) -> Result<AllocationPermit, ExecutionError> {
        let group = self.access_group();
        let mut state = group
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        loop {
            if state.host.conflict(write) || state.device.conflict(write) {
                state = group
                    .changed
                    .wait(state)
                    .expect("tensor allocation access lock poisoned while waiting");
                continue;
            }
            let Some(fence) = state.conflicting_device(write) else {
                break;
            };
            drop(state);
            match write {
                true => fence.retire(),
                false => fence.ended()?,
            }
            state = group
                .state
                .lock()
                .expect("tensor allocation access lock poisoned");
        }
        Ok(self.grant(state, Holder::Host, write))
    }

    /// Access by device work being encoded for one queue. It excludes host
    /// access for the duration of encoding but never waits for other device
    /// work, nor for another encoding (of its queue or another): the queue
    /// orders its own work, the device orders native work of different
    /// queues by their fences ([`Self::queue_order`]), and the caller orders
    /// transfers.
    /// An encoding can block in the driver behind device work it follows (a
    /// parked kernel), so no other queue's encoding may wait on it: a
    /// transfer that answers the park would never be queued. The submitter
    /// records its completion with [`Allocation::record_device_use`] before
    /// releasing the permit.
    pub(crate) fn acquire_for_device(self: &Arc<Self>, write: bool) -> AllocationPermit {
        let group = self.access_group();
        let mut state = group
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        while state.host.conflict(write) {
            state = group
                .changed
                .wait(state)
                .expect("tensor allocation access lock poisoned while waiting");
        }
        self.grant(state, Holder::Device, write)
    }

    /// Access that excludes every other: host access and the encoding of
    /// device work, never waiting for device work (a resize's install,
    /// between submissions).
    pub(crate) fn acquire_exclusive(self: &Arc<Self>) -> AllocationPermit {
        let group = self.access_group();
        let mut state = group
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        while state.host.any() || state.device.any() {
            state = group
                .changed
                .wait(state)
                .expect("tensor allocation access lock poisoned while waiting");
        }
        self.grant(state, Holder::Host, true)
    }

    /// Record work submitted to `queue` so later host access, and the
    /// storage's release, order after it. Called in the queue's submission
    /// order: `completion` completes after every fence of that queue
    /// recorded before it.
    pub(crate) fn record_device_use(&self, queue: DeviceQueue, completion: Arc<dyn DeviceCompletion>, write: bool) {
        let mut state = self
            .access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        let fences = state.queue(queue);
        if write {
            fences.last_write = Some(completion.clone());
        }
        fences.last_use = Some(completion);
    }

    /// The submitted work of the other native queue that native work on
    /// `queue` using this allocation must follow: its newest use where the
    /// work writes it, its newest write where the work reads it. None of
    /// the transfer queue, which its caller orders (`native::transfer`),
    /// nor of an allocation whose users' protocol orders it
    /// ([`Self::order_by_protocol`]). A queue runs its work in order, so its
    /// newest fence of a kind follows every earlier one. Read under the
    /// device's submission order, in which native fences are recorded.
    /// Work that reads after another queue's write that faulted receives
    /// its fault.
    pub(crate) fn queue_order(
        &self,
        queue: NativeQueue,
        write: bool,
    ) -> Result<Vec<Arc<dyn DeviceCompletion>>, ExecutionError> {
        let mut state = self
            .access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        if state.protocol {
            return Ok(Vec::new());
        }
        state.prune();
        NativeQueue::ALL
            .into_iter()
            .filter(|&other| other != queue)
            .filter_map(|other| {
                let fences = &state.queues[DeviceQueue::Native(other).ordinal()];
                match write {
                    true => fences.last_use.clone(),
                    false => fences.last_write.clone(),
                }
            })
            .map(|fence| fence.completed().map(|_| fence))
            .collect()
    }

    /// Order this fresh allocation's device accesses across queues by the
    /// protocol of the work that uses it, not by its fences: no native
    /// queue waits for another's use of it (host access and its release
    /// still wait for every use). For words the host and device work
    /// exchange whose every cross-queue order the exchange gives: a slot
    /// one queue's work requests and another's waits on. Set before
    /// anything uses it.
    pub(crate) fn order_by_protocol(&self) {
        let mut state = self
            .access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        assert!(state.uses().is_empty() && !state.host.any() && !state.device.any(), "a fresh allocation is unused");
        state.protocol = true;
    }

    /// No submitted device work still uses this allocation.
    pub(crate) fn device_idle(&self) -> bool {
        let mut state = self
            .access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        state.uses().is_empty()
    }

    /// The newest submitted work of each queue that still uses this
    /// allocation: every use completes once these have.
    fn last_device_uses(&self) -> DeviceUses {
        self.access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned")
            .uses()
    }

    fn grant(
        self: &Arc<Self>,
        mut state: MutexGuard<'_, AllocationAccess>,
        holder: Holder,
        write: bool,
    ) -> AllocationPermit {
        let holders = state.holders(holder);
        let count = if write { &mut holders.writers } else { &mut holders.readers };
        *count = count.checked_add(1).expect("tensor allocation permit count overflowed");
        drop(state);
        AllocationPermit {
            allocation: self.clone(),
            holder,
            write,
        }
    }

    /// Non-blocking acquisition for a freshly allocated, unpublished buffer.
    /// Admission uses this to assert that materialization never introduces a
    /// hidden wait after the graph transaction has claimed resources. None
    /// where access would wait, or a read would report a write's fault.
    pub(crate) fn try_acquire(self: &Arc<Self>, write: bool) -> Option<AllocationPermit> {
        let mut state = self
            .access_group()
            .state
            .lock()
            .expect("tensor allocation access lock poisoned");
        if state.host.conflict(write) || state.device.conflict(write) || state.conflicting_device(write).is_some() {
            return None;
        }
        Some(self.grant(state, Holder::Host, write))
    }
}

impl Drop for Allocation {
    /// Storage is released only after every submitted device use finishes.
    fn drop(&mut self) {
        let uses = self
            .access_group()
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .uses();
        uses.retire();
    }
}

/// Whose permit: host access, or device work being encoded.
#[derive(Clone, Copy)]
enum Holder {
    Host,
    Device,
}

pub(crate) struct AllocationPermit {
    allocation: Arc<Allocation>,
    holder: Holder,
    write: bool,
}

impl AllocationPermit {
    pub(crate) fn allocation(&self) -> &Arc<Allocation> {
        &self.allocation
    }
    pub(crate) fn owns(&self, allocation: &Arc<Allocation>) -> bool {
        Arc::ptr_eq(&self.allocation, allocation)
    }
    pub(crate) fn read(
        &self,
        allocation: &Arc<Allocation>,
        offset: u64,
        into: &mut [u8],
    ) -> Result<(), ExecutionError> {
        assert!(
            self.owns(allocation),
            "allocation read uses a different allocation's permit"
        );
        assert!(
            offset
                .checked_add(into.len() as u64)
                .is_some_and(|end| end <= allocation.bytes()),
            "allocation read exceeds its admitted backing"
        );
        self.allocation.storage().read(offset, into)
    }
}

impl Drop for AllocationPermit {
    fn drop(&mut self) {
        let group = self.allocation.access_group();
        let mut state = group
            .state
            .lock()
            .expect("tensor allocation access lock poisoned while releasing");
        let holders = state.holders(self.holder);
        let count = if self.write { &mut holders.writers } else { &mut holders.readers };
        *count = count.checked_sub(1).expect("a permit released that was never granted");
        group.changed.notify_all();
    }
}

pub(crate) fn typed_buffer<T: TargetFamily, E: NativeExecutor<T>>(
    allocation: &Allocation,
) -> &Buffer<T, E> {
    let storage = allocation.storage.as_any();
    storage
        .downcast_ref::<TypedStorage<T, E>>()
        .map(|storage| storage.buffer.borrow())
        .or_else(|| storage.downcast_ref::<ReadOnlyStorage<T, E>>().map(|storage| &storage.buffer))
        .unwrap_or_else(|| panic!("Tensor allocation backend invariant violated after successful WrongDevice validation"))
}

/// Writable storage of `buffer`, which `service` reads and writes.
pub(crate) fn buffer_storage<T: TargetFamily, E: NativeExecutor<T>>(
    service: &Arc<Service<T, E>>,
    buffer: Writable<T, E>,
) -> Box<dyn Storage> {
    Box::new(TypedStorage::<T, E> {
        service: service.clone(),
        buffer,
    })
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
    logical: LogicalEntry,
    bindings: ElementBindings,
    public_device: &Arc<crate::api::device::DeviceInner>,
    options: PreparationOptions,
) -> Result<Arc<Prepared<T, E>>, PrepareError>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_native_target::NativeCompiler<T, Handle = E::Handle>,
{
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
    // Past its deadline, preparation stops at its next step boundary; what it
    // had constructed is dropped, and only the typed expiry is returned.
    let deadline = options.deadline;
    let evaluated = seismic_lang::cancellation::within(deadline, || {
        Ok::<_, PrepareError>(match options.evaluation {
            EvaluationMethod::Analytical => {
                let analytical = opened.analytical().map_err(|error| {
                    PrepareError::Preparation(
                        seismic_compiler::errors::PreparationError::NativeCompilation(
                            seismic_native_target::NativeCompilationError::ToolchainFailure(
                                error.to_string(),
                            ),
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
        })
    });
    let (kernel, feedback_report) = evaluated.map_err(|_| {
        PrepareError::Preparation(seismic_compiler::errors::PreparationError::Expired)
    })??;
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
    C: seismic_native_target::NativeCompiler<T, Handle = E::Handle>,
{
    opened: Arc<Opened<T, E>>,
    public_device: Arc<crate::api::device::DeviceInner>,
    campaign: FeedbackPreparation<'a, T, C, feedback::Observer<T, E>>,
}

impl<'a, T, E, C> FeedbackCampaign<'a, T, E, C>
where
    T: TargetFamily,
    E: NativeExecutor<T>,
    C: seismic_native_target::NativeCompiler<T, Handle = E::Handle>,
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
    Bound {
        allocation: u64,
        base_offset: u64,
        accessible_bytes: u64,
        tensor: Option<seismic_compiler::executable::RuntimeTensorGeometry>,
    },
    Reached {
        slot: u64,
        alignment: u64,
    },
}

struct IssuedResources<'a, T: TargetFamily, E: NativeExecutor<T>> {
    owner: &'a mut AdmittedResources,
    device: &'a Arc<Opened<T, E>>,
    bindings: &'a [PhysicalBufferBinding],
    buffers: Vec<Option<RuntimeBuffer<Buffer<T, E>>>>,
}

impl<T: TargetFamily, E: NativeExecutor<T>>
    seismic_compiler::executable::ExecutionResources<Buffer<T, E>> for IssuedResources<'_, T, E>
{
    fn buffer(
        &self,
        allocation: seismic_compiler::executable::ExecutableAllocationId,
    ) -> &RuntimeBuffer<Buffer<T, E>> {
        self.buffers[allocation.ordinal()]
            .as_ref()
            .expect("planned allocation used before its reached acquisition")
    }
    fn retire_completed_instances(
        &mut self,
        allocations: &[seismic_compiler::executable::ExecutableAllocationId],
    ) {
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
    fn acquire_instance(
        &mut self,
        allocation: seismic_compiler::executable::ExecutableAllocationId,
        bytes: u64,
        alignment: u64,
    ) -> Result<(), ExecutionError> {
        let index = allocation.ordinal();
        let PhysicalBufferBinding::Reached {
            slot,
            alignment: planned_alignment,
        } = self.bindings[index]
        else {
            assert!(
                self.buffers[index]
                    .as_ref()
                    .expect("initial backing absent")
                    .accessible_bytes
                    >= bytes,
                "reached instance exceeds initial backing"
            );
            return Ok(());
        };
        if self
            .owner
            .private_backing(slot)
            .is_some_and(|backing| backing.bytes() >= bytes)
        {
            return Ok(());
        }
        // The schedule has completed all prior users and excluded retained
        // region products before requesting replacement of this bank.
        self.buffers[index] = None;
        self.owner.retire_private(slot);
        self.owner.check_reached_capacity(bytes)?;
        let backing = self
            .device
            .allocate_storage(bytes, alignment.max(planned_alignment))?;
        let permit = backing
            .try_acquire(true)
            .expect("fresh private backing cannot have an access owner");
        let buffer = RuntimeBuffer {
            tensor: None,
            buffer: typed_buffer::<T, E>(&backing).clone(),
            base_offset: 0,
            accessible_bytes: bytes,
        };
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
        Self {
            selected,
            staged,
            published: Vec::new(),
        }
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
        self.published
            .iter()
            .find(|publication| publication.path == path)
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
        self.published
            .iter()
            .map(|publication| {
                let key = match self.staged.buffers[publication.allocation.ordinal()] {
                    PhysicalBufferBinding::Bound { allocation, .. } => allocation,
                    PhysicalBufferBinding::Reached { slot, .. } => slot,
                };
                let allocation = resources.allocation(key).clone();
                if !publication
                    .byte_offset
                    .checked_add(publication.bytes)
                    .is_some_and(|end| end <= allocation.bytes())
                {
                    return Err(ExecutionError::ConstructionContradiction(
                        "published tensor exceeds its actual backing".into(),
                    ));
                }
                Ok((
                    publication.path.clone(),
                    crate::execution::AdmittedOutput::Tensor {
                        allocation,
                        byte_offset: publication.byte_offset,
                        byte_len: publication.bytes,
                        representation: publication.representation,
                        extents: publication.extents.clone(),
                        strides: publication.strides.clone(),
                    },
                ))
            })
            .collect()
    }
    pub(crate) fn issue(
        &mut self,
        submission: &mut E::Submission,
        device: &Service<T, E>,
        resources: &mut AdmittedResources,
        opened: &Arc<Opened<T, E>>,
    ) -> Result<(), ExecutionError> {
        let buffers = self
            .staged
            .buffers
            .iter()
            .map(|binding| match binding {
                PhysicalBufferBinding::Bound {
                    allocation,
                    base_offset,
                    accessible_bytes,
                    tensor,
                } => {
                    let allocation = resources.allocation(*allocation);
                    assert!(
                        base_offset
                            .checked_add(*accessible_bytes)
                            .is_some_and(|end| end <= allocation.bytes()),
                        "staged binding exceeds its admitted physical slot"
                    );
                    Some(RuntimeBuffer {
                        tensor: tensor.clone(),
                        buffer: typed_buffer::<T, E>(allocation).clone(),
                        base_offset: *base_offset,
                        accessible_bytes: *accessible_bytes,
                    })
                }
                PhysicalBufferBinding::Reached { slot, .. } => {
                    resources.declare_private(*slot);
                    resources
                        .private_backing(*slot)
                        .map(|allocation| RuntimeBuffer {
                            tensor: None,
                            buffer: typed_buffer::<T, E>(allocation).clone(),
                            base_offset: 0,
                            accessible_bytes: allocation.bytes(),
                        })
                }
            })
            .collect();
        let mut resources = IssuedResources {
            owner: resources,
            device: opened,
            bindings: &self.staged.buffers,
            buffers,
        };
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
pub(crate) fn collect_native_access(
    schema: &CallSchema,
    args: &EncodedArgs,
) -> Vec<(Arc<Allocation>, bool)> {
    let mut merged: BTreeMap<usize, (Arc<Allocation>, bool)> = BTreeMap::new();
    for (parameter, tensor) in schema.parameters().iter().zip(args.tensors()) {
        let (ParameterKind::Tensor { access, .. }, Some(tensor)) = (&parameter.kind, tensor) else {
            continue;
        };
        let write = matches!(access, TensorAccess::Owned | TensorAccess::Mutable);
        for allocation in tensor.bound_allocations() {
            merged
                .entry(allocation.access_identity())
                .and_modify(|(_, current)| *current |= write)
                .or_insert((allocation, write));
        }
    }
    merged.into_values().collect()
}

fn copy_between(source: &Allocation, destination: &Allocation, bytes: u64) -> Result<(), ExecutionError> {
    const CHUNK: usize = 1 << 20;
    let mut scratch = vec![0u8; CHUNK.min(usize::try_from(bytes).unwrap_or(CHUNK))];
    let mut offset = 0u64;
    while offset < bytes {
        let length = usize::try_from((bytes - offset).min(CHUNK as u64)).unwrap_or(CHUNK);
        source.storage.read(offset, &mut scratch[..length])?;
        destination.storage.write(offset, &scratch[..length])?;
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
    use std::sync::atomic::{AtomicBool, Ordering};

    struct EmptyStorage;
    impl Storage for EmptyStorage {
        fn read(&self, _: u64, _: &mut [u8]) -> Result<(), ExecutionError> {
            Ok(())
        }
        fn write(&self, _: u64, _: &[u8]) -> Result<(), ExecutionError> {
            Ok(())
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn host_address(&self) -> Option<std::ptr::NonNull<u8>> {
            None
        }
    }

    struct PendingUse(Arc<AtomicBool>);
    impl DeviceCompletion for PendingUse {
        fn completed(&self) -> Result<bool, ExecutionError> {
            Ok(self.0.load(Ordering::SeqCst))
        }
        fn ended(&self) -> Result<(), ExecutionError> {
            assert!(self.completed()?, "test releases the fence before storage");
            Ok(())
        }
    }

    fn split_memory() -> DeviceLedgers {
        let placement = || Placement::on(crate::memory::StoreLedger::new());
        DeviceLedgers::split(placement(), placement(), placement(), placement())
    }

    fn split_span(granule: u64) -> seismic_topology::Reservation {
        seismic_topology::Reservation {
            granule,
            host_addressed: true,
            chunks: Some(seismic_topology::ReservationChunks {
                unit: granule,
                table: seismic_topology::Contract::Upload,
                whole: false,
            }),
        }
    }

    fn split_reservation() -> ReservedRange {
        ReservedRange::new(seismic_topology::ReservedContract::Shared, split_span(4096),
            seismic_topology::ReservedRecords::of(std::num::NonZeroU64::new(4096).unwrap()), 8, u64::MAX)
    }

    #[test]
    fn reservation_placement_refuses_each_placement_and_cleans_up_before_make() {
        let memory = split_memory();
        let limits = AllocationLimits { max_allocation_bytes: 4096, max_allocation_alignment: 4096 };
        for (host, upload) in [(4095, 64), (4096, 63)] {
            memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().set_bound(Some(host));
            memory.upload().ledger().set_bound(Some(upload));
            assert!(allocate_reservation(limits, &memory, split_reservation(), 4096, 4096,
                || panic!("refused before allocating native storage")).is_err());
            assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 0);
            assert_eq!(memory.upload().ledger().view().charged, 0);
        }
        memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().set_bound(Some(4096));
        memory.upload().ledger().set_bound(Some(64));
        assert!(allocate_reservation(limits, &memory, split_reservation(), 4096, 4096, || {
            assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 4096);
            assert_eq!(memory.upload().ledger().view().charged, 64);
            Err(ExecutionError::AllocationFailed("native allocation refused".into()))
        }).is_err());
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 0);
        assert_eq!(memory.upload().ledger().view().charged, 0);
        let mut large_table = split_reservation();
        large_table.bytes = 4096 * 1024;
        assert!(allocate_reservation(limits, &memory, large_table, 4096, 4096,
            || panic!("table size refused before allocation")).is_err());
    }

    #[test]
    fn reservation_placement_shared_source_admits_combined_charge() {
        let memory = split_memory();
        let limits = AllocationLimits { max_allocation_bytes: 4096, max_allocation_alignment: 4096 };
        let mut reservation = split_reservation();
        reservation.contract = seismic_topology::ReservedContract::DeviceOnly;
        reservation.span.chunks.as_mut().unwrap().table = seismic_topology::Contract::DeviceOnly;
        memory.device().ledger().set_bound(Some(4096 + 63));
        assert!(allocate_reservation(limits, &memory, reservation, 4096, 4096,
            || panic!("combined backing and table exceed their shared placement")).is_err());
        assert_eq!(memory.device().ledger().view().charged, 0);
        memory.device().ledger().set_bound(Some(4096 + 64));
        assert!(allocate_reservation(limits, &memory, reservation, 4096, 4096, || {
            assert_eq!(memory.device().ledger().view().charged, 4096 + 64);
            Err(ExecutionError::AllocationFailed("native allocation refused".into()))
        }).is_err());
        assert_eq!(memory.device().ledger().view().charged, 0);
    }

    #[test]
    fn reservation_placement_keeps_table_charge_when_backing_shrinks() {
        let memory = split_memory();
        let granule = seismic_reservation::host_page().unwrap();
        let reservation = ReservedRange::new(seismic_topology::ReservedContract::Shared, split_span(granule),
            seismic_topology::ReservedRecords::of(std::num::NonZeroU64::new(granule).unwrap()), 8, u64::MAX);
        let limits = AllocationLimits { max_allocation_bytes: u64::MAX, max_allocation_alignment: granule };
        let service = Arc::new(seismic_cpu::Device);
        let resizer = allocate_reservation(limits, &memory, reservation, 2 * granule, granule, || {
            let (buffer, resizer) = service.allocate_reserved(2 * granule, 8 * granule, granule)?;
            Ok((buffer_storage::<seismic_cpu::Cpu, seismic_cpu::Executor>(&service, buffer),
                crate::backends::ReservedResizer::Cpu(resizer), Some((64, Box::new(EmptyStorage)))))
        }).unwrap();
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 2 * granule);
        assert_eq!(memory.upload().ledger().view().charged, 64);
        assert_eq!(memory.device().ledger().view().charged, 0);
        let pending = resizer.prepare(1).unwrap().install().unwrap();
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 2 * granule);
        let resizer = pending.release();
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, granule);
        assert_eq!(memory.upload().ledger().view().charged, 64);
        let resizer = resizer.recommit(2).unwrap();
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 2 * granule);
        assert_eq!(memory.upload().ledger().view().charged, 64);
        drop(resizer);
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 0);
        assert_eq!(memory.upload().ledger().view().charged, 0);
    }

    /// Transfer-queue work, pending until a waiter completes it.
    #[derive(Default)]
    struct TransferBatch {
        done: AtomicBool,
        waited: AtomicBool,
    }
    struct TransferFence(Arc<TransferBatch>);
    impl DeviceCompletion for TransferFence {
        fn completed(&self) -> Result<bool, ExecutionError> {
            Ok(self.0.done.load(Ordering::SeqCst))
        }
        fn ended(&self) -> Result<(), ExecutionError> {
            self.0.waited.store(true, Ordering::SeqCst);
            self.0.done.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    fn transfer_beside_finished_compute(allocation: &Allocation, write: bool) -> Arc<TransferBatch> {
        allocation.record_device_use(DeviceQueue::Native(NativeQueue::Compute), Arc::new(PendingUse(Arc::new(AtomicBool::new(true)))), true);
        let batch = Arc::new(TransferBatch::default());
        allocation.record_device_use(DeviceQueue::Transfer, Arc::new(TransferFence(batch.clone())), write);
        batch
    }

    #[test]
    fn dropping_an_allocation_waits_for_its_in_flight_transfer() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(8).unwrap();
        let allocation = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        let batch = transfer_beside_finished_compute(&allocation, true);
        assert!(!allocation.device_idle(), "a transfer still uses it");
        drop(allocation);
        assert!(batch.waited.load(Ordering::SeqCst), "storage outlives the transfer");
    }

    #[test]
    fn host_access_waits_for_an_in_flight_transfer() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let source = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        let destination = Allocation::new(2, 8, reservation.take(8), Box::new(EmptyStorage));
        let read = transfer_beside_finished_compute(&source, false);
        let written = transfer_beside_finished_compute(&destination, true);
        // A host read of the source does not wait for the copy reading it.
        drop(source.acquire(false).unwrap());
        assert!(!read.waited.load(Ordering::SeqCst));
        drop(source.acquire(true).unwrap());
        assert!(read.waited.load(Ordering::SeqCst), "a host write waits for the copy's read");
        drop(destination.acquire(false).unwrap());
        assert!(written.waited.load(Ordering::SeqCst), "a host read waits for the copy's write");
    }

    /// Device work that ended with a fault.
    struct FaultedUse;
    impl DeviceCompletion for FaultedUse {
        fn completed(&self) -> Result<bool, ExecutionError> {
            Err(ExecutionError::SubmissionFailed("faulted".into()))
        }
        fn ended(&self) -> Result<(), ExecutionError> {
            self.completed().map(drop)
        }
    }

    #[test]
    fn a_faulted_write_ends_its_use_and_fails_what_reads_after_it() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(8).unwrap();
        let allocation = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        allocation.record_device_use(DeviceQueue::Native(NativeQueue::Fill), Arc::new(FaultedUse), true);
        assert!(allocation.device_idle(), "a fault ends the use");
        assert!(allocation.acquire(false).is_err(), "a host read receives the fault");
        assert!(allocation.queue_order(NativeQueue::Compute, false).is_err(), "a device read receives the fault");
        assert_eq!(allocation.queue_order(NativeQueue::Compute, true).unwrap().len(), 0, "a write follows no ended use");
        drop(allocation.acquire(true).unwrap());
    }

    /// Native work orders against the other native queue by what it
    /// touches: a write follows the other queue's newest use, a read its
    /// newest write; completed work, its own queue's, the transfer queue's
    /// (its caller's to order) and a protocol-ordered allocation's impose
    /// nothing.
    #[test]
    fn native_work_follows_the_other_native_queues_conflicting_fences() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let allocation = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        let pending = || Arc::new(PendingUse(Arc::new(AtomicBool::new(false))));
        let order = |queue, write| allocation.queue_order(queue, write).unwrap().len();
        // Compute reads it; a fill read follows nothing, a fill write the read.
        let read = pending();
        allocation.record_device_use(DeviceQueue::Native(NativeQueue::Compute), read.clone(), false);
        assert_eq!(order(NativeQueue::Fill, false), 0, "reads do not conflict");
        assert_eq!(order(NativeQueue::Fill, true), 1, "a write follows the other queue's read");
        assert_eq!(order(NativeQueue::Compute, true), 0, "a queue orders its own work");
        // A fill writes it; compute reads and writes follow it.
        allocation.record_device_use(DeviceQueue::Native(NativeQueue::Fill), pending(), true);
        assert_eq!(order(NativeQueue::Compute, false), 1, "a read follows the other queue's write");
        assert_eq!(order(NativeQueue::Compute, true), 1);
        // The transfer queue's work is its caller's to order.
        allocation.record_device_use(DeviceQueue::Transfer, pending(), true);
        assert_eq!(order(NativeQueue::Compute, true), 1, "no transfer fence orders native work");
        // Completed work orders nothing.
        read.0.store(true, Ordering::SeqCst);
        assert_eq!(order(NativeQueue::Fill, true), 0, "the compute read completed");
        // Words whose users' protocol orders them order no queue.
        let words = Allocation::new(2, 8, reservation.take(8), Box::new(EmptyStorage));
        words.order_by_protocol();
        words.record_device_use(DeviceQueue::Native(NativeQueue::Fill), pending(), true);
        assert!(words.queue_order(NativeQueue::Compute, true).unwrap().is_empty());
        assert!(!words.device_idle(), "host access still waits for every use");
        // Pending test fences are forgotten before the storage is released.
        for held in [&allocation, &words] {
            let mut state = held.access_group().state.lock().unwrap();
            state.queues.iter_mut().for_each(|fences| *fences = QueueFences::default());
        }
    }

    /// A transfer is encoded while compute work is being encoded into the
    /// same allocation (a step's submission, which can block in the driver
    /// behind a parked kernel the transfer answers): device encodings never
    /// exclude each other. Host access still excludes both, and an install
    /// excludes every holder.
    #[test]
    fn a_device_encoding_never_waits_for_another_queues_encoding() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(8).unwrap();
        let allocation = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        let compute = allocation.acquire_for_device(true);
        let transfer = allocation.acquire_for_device(true);
        assert!(allocation.try_acquire(false).is_none(), "host access waits for an encoding");
        drop(compute);
        assert!(allocation.try_acquire(false).is_none(), "host access waits for every encoding");
        drop(transfer);
        let host = allocation.try_acquire(false).expect("host access once no encoding holds");
        let waiting = {
            let allocation = allocation.clone();
            std::thread::spawn(move || drop(allocation.acquire_for_device(true)))
        };
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!waiting.is_finished(), "a device write waits for a host read");
        drop(host);
        waiting.join().unwrap();
        let exclusive = allocation.acquire_exclusive();
        assert!(allocation.try_acquire(false).is_none(), "an install excludes host access");
        drop(exclusive);
    }

    #[test]
    fn shrinking_a_reservation_waits_for_its_in_flight_transfer() {
        let memory = split_memory();
        let granule = seismic_reservation::host_page().unwrap();
        let reservation = ReservedRange::new(seismic_topology::ReservedContract::Shared, split_span(granule),
            seismic_topology::ReservedRecords::of(std::num::NonZeroU64::new(granule).unwrap()), 8, u64::MAX);
        let limits = AllocationLimits { max_allocation_bytes: u64::MAX, max_allocation_alignment: granule };
        let service = Arc::new(seismic_cpu::Device);
        let resizer = allocate_reservation(limits, &memory, reservation, 2 * granule, granule, || {
            let (buffer, resizer) = service.allocate_reserved(2 * granule, 8 * granule, granule)?;
            Ok((buffer_storage::<seismic_cpu::Cpu, seismic_cpu::Executor>(&service, buffer),
                crate::backends::ReservedResizer::Cpu(resizer), Some((64, Box::new(EmptyStorage)))))
        }).unwrap();
        let batch = transfer_beside_finished_compute(&resizer.allocation, true);
        let pending = resizer.prepare(1).unwrap().install().unwrap();
        assert!(!batch.waited.load(Ordering::SeqCst));
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, 2 * granule, "nothing is released before the copy completes");
        let resizer = pending.release();
        assert!(batch.waited.load(Ordering::SeqCst), "the shrink unbacks after the copy");
        assert_eq!(memory.placement(seismic_topology::Placing::Fixed(seismic_topology::Contract::Shared)).ledger().view().charged, granule);
        drop(resizer);
    }

    #[test]
    fn slab_and_table_share_one_hazard_group() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let table = Allocation::new(1, 8, reservation.take(8), Box::new(EmptyStorage));
        let slab = Allocation::new_in(2, 8, reservation.take(8), Box::new(EmptyStorage), Some(table.access_group().clone()));
        assert_eq!(table.access_identity(), slab.access_identity());
        let merged = crate::native::merge_access([(table.clone(), false), (slab.clone(), true)]);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].1);
        drop(merged);
        let done = Arc::new(AtomicBool::new(false));
        table.record_device_use(DeviceQueue::Native(NativeQueue::Compute), Arc::new(PendingUse(done.clone())), false);
        assert!(slab.try_acquire(true).is_none());
        assert!(slab.try_acquire(false).is_some());
        done.store(true, Ordering::SeqCst);
        assert!(slab.try_acquire(true).is_some());
        drop(slab);
        drop(table);
        assert_eq!(memory.ledger().view().charged, 0);
    }

    #[test]
    fn admitted_slot_owns_backing_charge_and_exclusive_access() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let allocation = Allocation::new(1, 16, reservation.take(16), Box::new(EmptyStorage));
        let weak = Arc::downgrade(&allocation);
        let permit = allocation.acquire(true).unwrap();
        let resources = AdmittedResources::new(reservation, vec![permit]);
        drop(allocation);
        assert_eq!(memory.ledger().view().charged, 16);
        assert!(resources.allocation(1).try_acquire(false).is_none());
        assert!(resources
            .access(resources.allocation(1))
            .owns(resources.allocation(1)));
        drop(resources);
        assert!(weak.upgrade().is_none());
        assert_eq!(memory.ledger().view().charged, 0);
    }

    #[test]
    fn native_allocation_owner_enforces_target_limits_without_leaking_reservation() {
        let devices = crate::devices::Devices::discover(&[]).unwrap();
        let device = devices.open_backend(registry::BackendName::Cpu).unwrap();
        let crate::backends::OpenedKind::Cpu(opened) = &device.kind else {
            unreachable!()
        };
        let baseline = opened.memory.device_ledger().view().charged;
        let maximum = opened.device_description().limits().max_allocation_bytes;
        let host_ram = device.topology().host_ram().capacity().value;
        // The CPU's largest allocation is its address space, beyond host
        // RAM, so host RAM's ledger refuses the request before the target
        // limit is reached: what it could take is within host RAM. (Its
        // exact room moves with the other tests charging host RAM in this
        // process.) Nothing is allocated or kept.
        assert!(host_ram < maximum);
        let error = opened
            .allocate_storage(maximum.checked_add(1).unwrap(), 4)
            .err()
            .expect("an allocation beyond every bound is refused before allocating");
        assert!(matches!(
            error,
            ExecutionError::AllocationCapacity { required, available }
                if required == (maximum + 1).into() && available <= host_ram
        ));
        assert_eq!(opened.memory.device_ledger().view().charged, baseline);
        assert!(matches!(
            opened.allocate_storage(4, 0),
            Err(ExecutionError::ConstructionContradiction(_))
        ));
        assert_eq!(opened.memory.device_ledger().view().charged, baseline);
    }

    #[test]
    fn reached_slots_release_dead_capacity_and_preserve_live_permits_on_refusal() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
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
        resources.install_private(100, first.acquire(true).unwrap());
        drop(first);
        assert_eq!(
            resources.check_reached_capacity(16),
            Err(ExecutionError::AllocationCapacity {
                required: 16u64.into(),
                available: 8
            })
        );
        assert!(resources
            .private_backing(100)
            .unwrap()
            .try_acquire(false)
            .is_none());
        resources.retire_private(100);
        assert_eq!(memory.ledger().view().charged, 0);
        resources.check_reached_capacity(24).unwrap();
        let second = make(2, 24);
        resources.install_private(101, second.acquire(true).unwrap());
        drop(second);
        assert_eq!(memory.ledger().view().charged, 24);
        assert_eq!(resources.reached_allocated(), 40);
        drop(resources);
        assert_eq!(memory.ledger().view().charged, 0);
    }

    #[test]
    fn completed_dead_bank_releases_device_capacity_without_releasing_live_bank() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        memory.ledger().set_bound(Some(24));
        let mut resources = AdmittedResources::new(memory.reserve(0).unwrap(), vec![]);
        resources.set_reached_budget(24);
        for slot in [100, 101, 102] {
            resources.declare_private(slot);
        }
        let make = |id, bytes| {
            let mut reservation = memory.reserve(bytes).unwrap();
            Allocation::new(id, bytes, reservation.take(bytes), Box::new(EmptyStorage))
        };
        let dead = make(1, 16);
        let live = make(2, 8);
        resources.install_private(100, dead.acquire(true).unwrap());
        resources.install_private(101, live.acquire(true).unwrap());
        let pending = dead.clone();
        drop(dead);
        drop(live);
        assert!(memory.reserve(16).is_err());
        assert!(resources.check_reached_capacity(16).is_err());
        assert_eq!(memory.ledger().view().charged, 24);
        // The compiler's successful prefix completion precedes this release;
        // backend references and the slot's permit must both be gone.
        drop(pending);
        resources.retire_private(100);
        assert_eq!(memory.ledger().view().charged, 8);
        resources.check_reached_capacity(16).unwrap();
        let replacement = make(3, 16);
        resources.install_private(102, replacement.acquire(true).unwrap());
        drop(replacement);
        assert_eq!(memory.ledger().view().charged, 24);
        assert!(resources
            .private_backing(101)
            .unwrap()
            .try_acquire(false)
            .is_none());
        assert!(memory.reserve(1).is_err());
        drop(resources);
        assert_eq!(memory.ledger().view().charged, 0);
    }

    #[test]
    fn published_backing_survives_slot_release_without_retaining_its_permit() {
        let memory = Placement::on(crate::memory::StoreLedger::new());
        let mut reservation = memory.reserve(16).unwrap();
        let allocation = Allocation::new(1, 16, reservation.take(16), Box::new(EmptyStorage));
        let resources = AdmittedResources::new(reservation, vec![allocation.acquire(true).unwrap()]);
        drop(allocation);
        let published = resources.allocation(1).clone();
        drop(resources);
        assert_eq!(memory.ledger().view().charged, 16);
        let read = published
            .try_acquire(false)
            .expect("completed run released its exclusive access");
        drop(read);
        drop(published);
        assert_eq!(memory.ledger().view().charged, 0);
    }
}
