//! Unified data-movement & contention plane (`IoPlane`):
//!
//! - One [`CoupledSetController`] admission owner per [`CoupledSetId`] (short
//!   mutex, NEVER held across OS/driver calls).
//! - Four-state physical operation lifecycle:
//!   `Queued -> Submitting -> Issued -> Retired | Quarantined`.
//! - Strict separation of logical subscribers ([`TransferHandle`]) from
//!   physical operations ([`PhysicalOpId`]), with strict [`ContentKey`]
//!   join-in-flight preconditions.
//! - Residual speculative service-time bounding (`τ_hol`), knee cap, and
//!   strict `Blocking` preemption across coupled `SharedBus`es.
//! - Zero-wait work-conserving batch coalescing on issue.
//! - Multi-hop pipeline support with opaque CPU [`TransformTask`], end-to-end
//!   priority inheritance, and [`IntermediateSlotReserve`] deadlock prevention.

use crate::fabric::{
    BusCoupling, BusDirection, CoupledSetId, CoupledSetSpec, DeviceId, DomainId, Fabric,
    FabricSpec, LinkCurve, LinkId, LinkKind, LinkSpec, NumaNodeId, RepresentationId,
    RootComplexId, SharedBusId, SharedBusSpec, SocId, VolumeId,
};
use crate::fence::{TimelineObserver, TimelinePoint};
use crate::lease::BufferLease;
use crate::metrics::{
    BusLedger, BusMetrics, BusReservation, FabricMetricsStore, FenceBusReservation,
    FenceReservationId, LinkLedger, LinkMetrics, TransferQuote,
};
use crate::schedule::{BatchCompatibility, ContentKey, PlacementRequirement, PriorityClass, ScheduleKey};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Standard well-known `SharedBusId`s and `LinkId`s for the process-wide
/// [`global_plane`] unifying Storage (`NvmeDirect`/`UnifiedDirect`) and
/// CopyEngine (`CopyEngineH2D`/`D2H`) under one `CoupledSet`.
pub const BUS_HOST_DDR: SharedBusId = SharedBusId::HostDdr(NumaNodeId(0));
pub const BUS_PCIE_ROOT: SharedBusId = SharedBusId::PcieRoot(RootComplexId(0));
pub const BUS_UNIFIED_FABRIC: SharedBusId = SharedBusId::UnifiedFabric(SocId(0));
pub const DOMAIN_STORAGE: DomainId = DomainId::Storage(VolumeId(0));
pub const DOMAIN_HOST_RAM: DomainId = DomainId::HostRam(NumaNodeId(0));
pub const DOMAIN_DEVICE_VRAM: DomainId = DomainId::DeviceVram(DeviceId(0));
pub const LINK_COPY_H2D: LinkId = LinkId(0);
pub const LINK_COPY_D2H: LinkId = LinkId(1);
pub const LINK_STORAGE_NVME: LinkId = LinkId(2);

/// Unique identifier of one physical transfer operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalOpId(pub u64);

/// Unique identifier of one logical subscriber attached to a [`PhysicalOpId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubscriberId(pub u64);

/// Unique identifier of one issuer-claimed batch in `Submitting` / `Issued`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BatchId(pub u64);

/// Four-state physical operation lifecycle (+ terminal `Quarantined`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpState {
    Queued,
    Submitting,
    Issued,
    Retired,
    Quarantined,
}

/// Terminal outcome observed by each [`TransferHandle`] subscriber.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransferOutcome {
    Completed { fence: Option<TimelinePoint> },
    Cancelled,
    RefusedBeforeIssue(String),
    Quarantined(String),
}

/// Request submitted to [`IoPlane::submit`].
pub struct TransferRequest {
    pub link: LinkId,
    pub key: ScheduleKey,
    pub bytes: u64,
    pub content_key: Option<ContentKey>,
    pub placement: PlacementRequirement,
    pub compatibility: BatchCompatibility,
    pub src_lease: Option<BufferLease>,
    pub dst_lease: BufferLease,
    pub direct: bool,
}

/// Optional owner discriminator tag attached to a physical op so multiple
/// independent issuers sharing one `IoPlane` (e.g. two `Copies` instances on
/// `global_plane()`) never claim each other's ops while still sharing
/// `CoupledSet` bus credits and `Blocking` preemption.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OwnerTag(pub u64);

/// Why [`IoPlane::submit`] failed synchronously before enqueueing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SubmitError {
    UnknownLink(LinkId),
    DomainMismatch { expected: DomainId, got: DomainId },
}

impl fmt::Display for SubmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownLink(link) => write!(f, "unknown link {link:?}"),
            Self::DomainMismatch { expected, got } => {
                write!(f, "destination lease domain {got:?} does not match link dst {expected:?}")
            }
        }
    }
}

impl std::error::Error for SubmitError {}

/// Opaque CPU transform stage scheduled by `seismic-io` between hops without
/// `seismic-io` knowing quantization formats.
pub trait TransformTask: Send + 'static {
    fn input_repr(&self) -> RepresentationId;
    fn output_repr(&self) -> RepresentationId;
    /// Total bytes read + written on host memory (debited at `weight = 2.0` on `HostDdr`).
    fn cpu_ddr_bytes(&self) -> u64;
    fn execute(&mut self, input: &BufferLease, output: &mut BufferLease) -> Result<(), String>;
}

/// Forward-progress intermediate slot pool preventing speculative (`Prefetch` /
/// `Background`) Hop-0 reads from exhausting intermediate `HostStaging` /
/// `HostRam` slots needed by `Blocking` pipelines.
pub struct IntermediateSlotReserve {
    total_slots: usize,
    blocking_reserve: usize,
    state: Mutex<ReserveState>,
}

struct ReserveState {
    in_use: usize,
    speculative_in_use: usize,
}

/// RAII token for one slot acquired from [`IntermediateSlotReserve`].
pub struct IntermediateSlotGuard {
    reserve: Arc<IntermediateSlotReserve>,
    speculative: bool,
}

impl Drop for IntermediateSlotGuard {
    fn drop(&mut self) {
        let mut s = self.reserve.state.lock().expect("intermediate slot reserve");
        s.in_use = s.in_use.saturating_sub(1);
        if self.speculative {
            s.speculative_in_use = s.speculative_in_use.saturating_sub(1);
        }
    }
}

impl IntermediateSlotReserve {
    /// Create a reserve of `total_slots` with `blocking_reserve` slots
    /// strictly reserved for `Blocking` / `Streaming` pipelines.
    pub fn new(total_slots: usize, blocking_reserve: usize) -> Arc<Self> {
        assert!(total_slots > 0, "intermediate pool must have at least 1 slot");
        assert!(
            blocking_reserve <= total_slots,
            "blocking_reserve ({blocking_reserve}) cannot exceed total_slots ({total_slots})"
        );
        Arc::new(Self {
            total_slots,
            blocking_reserve,
            state: Mutex::new(ReserveState {
                in_use: 0,
                speculative_in_use: 0,
            }),
        })
    }

    /// Try to acquire an intermediate slot for `class`.
    /// Speculative (`Prefetch` / `Background`) requests are refused whenever
    /// remaining free slots would drop below `blocking_reserve`.
    pub fn try_acquire(self: &Arc<Self>, class: PriorityClass) -> Option<IntermediateSlotGuard> {
        let mut s = self.state.lock().expect("intermediate slot reserve");
        let free = self.total_slots.saturating_sub(s.in_use);
        if class.is_speculative() {
            if free <= self.blocking_reserve {
                return None;
            }
            s.in_use += 1;
            s.speculative_in_use += 1;
            Some(IntermediateSlotGuard {
                reserve: Arc::clone(self),
                speculative: true,
            })
        } else {
            if free == 0 {
                return None;
            }
            s.in_use += 1;
            Some(IntermediateSlotGuard {
                reserve: Arc::clone(self),
                speculative: false,
            })
        }
    }

    pub fn in_use(&self) -> usize {
        self.state.lock().expect("intermediate slot reserve").in_use
    }
}

// ---------------------------------------------------------------------------
// Indexed min-heap for O(log N) push / pop / reprice / remove with zero
// steady-state allocations.
// ---------------------------------------------------------------------------

struct HeapEntry {
    key: ScheduleKey,
    op_slot: usize,
}

#[derive(Default)]
struct IndexedMinHeap {
    items: Vec<HeapEntry>,
    /// `pos[op_slot] == Some(heap_idx)` while `op_slot` is in the heap.
    pos: Vec<Option<usize>>,
}

impl IndexedMinHeap {
    fn ensure_slot(&mut self, op_slot: usize) {
        if op_slot >= self.pos.len() {
            self.pos.resize(op_slot + 1, None);
        }
    }

    #[inline]
    fn peek(&self) -> Option<(ScheduleKey, usize)> {
        self.items.first().map(|e| (e.key, e.op_slot))
    }

    fn push(&mut self, op_slot: usize, key: ScheduleKey) {
        self.ensure_slot(op_slot);
        debug_assert!(self.pos[op_slot].is_none(), "op_slot already in heap");
        let idx = self.items.len();
        self.items.push(HeapEntry { key, op_slot });
        self.pos[op_slot] = Some(idx);
        self.sift_up(idx);
    }

    fn pop(&mut self) -> Option<(ScheduleKey, usize)> {
        if self.items.is_empty() {
            return None;
        }
        Some(self.remove_at(0))
    }

    fn remove(&mut self, op_slot: usize) -> bool {
        let Some(idx) = self.pos.get(op_slot).copied().flatten() else {
            return false;
        };
        self.remove_at(idx);
        true
    }

    fn update_key(&mut self, op_slot: usize, new_key: ScheduleKey) {
        let Some(idx) = self.pos.get(op_slot).copied().flatten() else {
            return;
        };
        let old_key = self.items[idx].key;
        if old_key == new_key {
            return;
        }
        self.items[idx].key = new_key;
        if new_key < old_key {
            self.sift_up(idx);
        } else {
            self.sift_down(idx);
        }
    }

    fn remove_at(&mut self, idx: usize) -> (ScheduleKey, usize) {
        let last = self.items.len() - 1;
        let removed = self.items.swap_remove(idx);
        self.pos[removed.op_slot] = None;
        if idx < last {
            let moved_slot = self.items[idx].op_slot;
            self.pos[moved_slot] = Some(idx);
            // Restoring heap invariant may need either sift_up or sift_down.
            let parent = idx.wrapping_sub(1) / 2;
            if idx > 0 && self.items[idx].key < self.items[parent].key {
                self.sift_up(idx);
            } else {
                self.sift_down(idx);
            }
        }
        (removed.key, removed.op_slot)
    }

    fn sift_up(&mut self, mut idx: usize) {
        while idx > 0 {
            let parent = (idx - 1) / 2;
            if self.items[idx].key < self.items[parent].key {
                self.swap(idx, parent);
                idx = parent;
            } else {
                break;
            }
        }
    }

    fn sift_down(&mut self, mut idx: usize) {
        let len = self.items.len();
        loop {
            let left = 2 * idx + 1;
            let right = left + 1;
            let mut smallest = idx;
            if left < len && self.items[left].key < self.items[smallest].key {
                smallest = left;
            }
            if right < len && self.items[right].key < self.items[smallest].key {
                smallest = right;
            }
            if smallest != idx {
                self.swap(idx, smallest);
                idx = smallest;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn swap(&mut self, a: usize, b: usize) {
        self.items.swap(a, b);
        let sa = self.items[a].op_slot;
        let sb = self.items[b].op_slot;
        self.pos[sa] = Some(a);
        self.pos[sb] = Some(b);
    }
}

// ---------------------------------------------------------------------------
// Subscriber completion state & Handle
// ---------------------------------------------------------------------------

struct SubscriberCompletion {
    outcome: Mutex<Option<TransferOutcome>>,
    changed: Condvar,
}

impl SubscriberCompletion {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            outcome: Mutex::new(None),
            changed: Condvar::new(),
        })
    }

    fn set(&self, outcome: TransferOutcome) {
        let mut guard = self.outcome.lock().expect("subscriber outcome");
        if guard.is_none() {
            *guard = Some(outcome);
            drop(guard);
            self.changed.notify_all();
        }
    }

    fn try_get(&self) -> Option<TransferOutcome> {
        self.outcome.lock().expect("subscriber outcome").clone()
    }

    fn wait(&self) -> TransferOutcome {
        let mut guard = self.outcome.lock().expect("subscriber outcome");
        loop {
            if let Some(outcome) = guard.as_ref() {
                return outcome.clone();
            }
            guard = self.changed.wait(guard).expect("subscriber outcome");
        }
    }

    fn wait_timeout(&self, timeout: Duration) -> Option<TransferOutcome> {
        let deadline = Instant::now() + timeout;
        let mut guard = self.outcome.lock().expect("subscriber outcome");
        loop {
            if let Some(outcome) = guard.as_ref() {
                return Some(outcome.clone());
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let (next, _) = self
                .changed
                .wait_timeout(guard, remaining)
                .expect("subscriber outcome");
            guard = next;
        }
    }
}

/// Logical subscriber handle attached to a [`PhysicalOpId`].
///
/// - [`TransferHandle::reprice`] updates this subscriber's key and recomputes
///   the physical op's effective `min(subscriber_keys)` in O(log N) while `Queued`.
/// - `Drop` detaches ONLY this subscriber:
///   - If the physical op is still `Queued` and 0 subscribers remain, it cancels
///     immediately with zero bus debit and releases leases outside locks.
///   - If `Submitting` or `Issued`, or other subscribers remain, the physical op
///     completes normally and backing [`BufferLease`]s remain pinned until BOTH
///     hardware completion AND detachment of the last subscriber.
pub struct TransferHandle {
    pub op_id: PhysicalOpId,
    pub subscriber_id: SubscriberId,
    pub joined_existing: bool,
    pub dst_domain: DomainId,
    pub dst_slot: u32,
    pub dst_generation: u64,
    op_slot: usize,
    sub_slot: usize,
    controller: Arc<CoupledSetController>,
    completion: Arc<SubscriberCompletion>,
    detached: bool,
}

impl TransferHandle {
    /// Update this subscriber's [`ScheduleKey`].
    pub fn reprice(&self, new_key: ScheduleKey) {
        self.controller.reprice_subscriber(self.op_slot, self.sub_slot, new_key);
    }

    /// Non-blocking check for terminal outcome.
    pub fn outcome(&self) -> Option<TransferOutcome> {
        self.completion.try_get()
    }

    /// Block until this transfer reaches a terminal outcome.
    pub fn wait(&self) -> TransferOutcome {
        self.completion.wait()
    }

    /// Block up to `timeout` for a terminal outcome.
    pub fn wait_timeout(&self, timeout: Duration) -> Option<TransferOutcome> {
        self.completion.wait_timeout(timeout)
    }

    /// Explicitly detach / cancel this subscriber (equivalent to `drop(self)`).
    pub fn detach(mut self) {
        self.detach_inner();
    }

    fn detach_inner(&mut self) {
        if !self.detached {
            self.detached = true;
            self.controller.detach_subscriber(self.op_slot, self.sub_slot);
        }
    }
}

impl Drop for TransferHandle {
    fn drop(&mut self) {
        self.detach_inner();
    }
}

/// Multi-hop pipeline handle supporting end-to-end priority inheritance:
/// calling [`PipelineHandle::reprice`] promotes whichever stage is currently
/// queued (`Hop 0`, `TransformTask`, or `Hop 1`).
pub struct PipelineHandle {
    stages: Mutex<Vec<TransferHandle>>,
}

impl PipelineHandle {
    pub fn new(stages: Vec<TransferHandle>) -> Self {
        Self {
            stages: Mutex::new(stages),
        }
    }

    /// Attach a subsequent stage handle once it is enqueued.
    pub fn push_stage(&self, handle: TransferHandle) {
        self.stages.lock().expect("pipeline stages").push(handle);
    }

    /// End-to-end priority inheritance across all active pipeline stages.
    pub fn reprice(&self, new_key: ScheduleKey) {
        let stages = self.stages.lock().expect("pipeline stages");
        for stage in stages.iter() {
            stage.reprice(new_key);
        }
    }
}

// ---------------------------------------------------------------------------
// Issuer-side ClaimedBatch / ClaimedOp
// ---------------------------------------------------------------------------

/// Metadata for one physical op inside a [`ClaimedBatch`] in `Submitting`.
#[derive(Clone, Copy, Debug)]
pub struct ClaimedOp {
    pub op_id: PhysicalOpId,
    pub bytes: u64,
    pub class: PriorityClass,
    pub dst_domain: DomainId,
    pub dst_slot: u32,
    pub dst_generation: u64,
    pub direct: bool,
}

/// One work-conserving batch claimed by a Link's issuer (`Queued -> Submitting`).
///
/// The `CoupledSet` lock is already released when this value is returned, so
/// the issuer invokes OS/driver calls with ZERO locks held, then transitions
/// the batch via exactly one of:
/// - [`IoPlane::confirm_issued`] (`Submitting -> Issued`)
/// - [`IoPlane::rollback_transient`] (`Submitting -> Queued`)
/// - [`IoPlane::refuse_before_issue`] (`Submitting -> Retired`)
/// - [`IoPlane::quarantine_batch`] (`Submitting -> Quarantined`)
#[derive(Clone, Debug)]
pub struct ClaimedBatch {
    pub batch_id: BatchId,
    pub link: LinkId,
    pub ops: Vec<ClaimedOp>,
    pub total_bytes: u64,
    pub compatibility: BatchCompatibility,
}

// ---------------------------------------------------------------------------
// CoupledSetController (Single Admission Owner)
// ---------------------------------------------------------------------------

struct SubscriberRecord {
    key: ScheduleKey,
    completion: Arc<SubscriberCompletion>,
}

struct PhysicalOpRecord {
    id: PhysicalOpId,
    owner: OwnerTag,
    link_local_idx: usize,
    state: OpState,
    effective_key: ScheduleKey,
    admitted_class: PriorityClass,
    bytes: u64,
    content_key: Option<ContentKey>,
    compatibility: BatchCompatibility,
    direct: bool,
    src_lease: Option<BufferLease>,
    dst_lease: Option<BufferLease>,
    dst_domain: DomainId,
    dst_slot: u32,
    dst_generation: u64,
    /// Active subscribers attached to this physical op (`sub_slot`s in arena).
    subscribers: Vec<usize>,
    /// Set once hardware retirement has occurred (`Issued -> Retired`).
    hw_retired: bool,
    /// Estimated completion time for residual speculative service-time tracking.
    est_service_s: f64,
    started_at: Option<Instant>,
    fence: Option<TimelinePoint>,
    /// Index in `CoupledSetState::in_flight_speculative` while this op is
    /// `Submitting | Issued` with a speculative `admitted_class`.
    spec_idx: Option<usize>,
}

struct LinkState {
    id: LinkId,
    fabric_idx: usize,
    ready: IndexedMinHeap,
    ledger: LinkLedger,
    last_integral_at: Instant,
    /// Minimum speculative quantum (seconds) used in `tau_hol = max(2λ, min_quantum_s)`.
    min_quantum_s: f64,
}

impl LinkState {
    #[inline]
    fn advance_integrals(&mut self, now: Instant) {
        self.ledger.advance_integrals(&mut self.last_integral_at, now);
    }
}

struct BusState {
    id: SharedBusId,
    fabric_idx: usize,
    ledger: BusLedger,
}

struct BatchRecord {
    link_local_idx: usize,
    op_slots: Vec<usize>,
    state: OpState,
}

struct CoupledSetState {
    links: Vec<LinkState>,
    link_map: HashMap<LinkId, usize>,
    buses: Vec<BusState>,
    bus_map: HashMap<SharedBusId, usize>,
    /// Pre-allocated arena of `PhysicalOpRecord`s + free list (zero steady-state malloc).
    ops: Vec<Option<PhysicalOpRecord>>,
    free_ops: Vec<usize>,
    /// Pre-allocated arena of `SubscriberRecord`s + free list.
    subs: Vec<Option<SubscriberRecord>>,
    free_subs: Vec<usize>,
    /// Active joinable ops indexed by `ContentKey`.
    join_index: HashMap<ContentKey, usize>,
    /// Active `Submitting` / `Issued` batches.
    batches: HashMap<BatchId, BatchRecord>,
    /// Quarantined leases retained permanently until `teardown_quarantined()`.
    quarantined_leases: Vec<BufferLease>,
    /// Active GPU-fence-bound non-IO bus reservations.
    fence_reservations: VecDeque<FenceBusReservation>,
    /// Arena slots of currently `Submitting | Issued` speculative ops, so
    /// `residual_speculative_service_time_s` scans only active speculative
    /// flight rather than the full op arena.
    in_flight_speculative: Vec<usize>,
}

impl CoupledSetState {
    fn track_in_flight_speculative(&mut self, op_slot: usize) {
        let should_track = self.ops[op_slot]
            .as_ref()
            .is_some_and(|op| op.admitted_class.is_speculative() && op.spec_idx.is_none());
        if should_track {
            let idx = self.in_flight_speculative.len();
            self.in_flight_speculative.push(op_slot);
            if let Some(op) = self.ops[op_slot].as_mut() {
                op.spec_idx = Some(idx);
            }
        }
    }

    fn untrack_in_flight_speculative(&mut self, op_slot: usize) {
        let Some(idx) = self.ops[op_slot].as_mut().and_then(|op| op.spec_idx.take()) else {
            return;
        };
        let removed = self.in_flight_speculative.swap_remove(idx);
        debug_assert_eq!(removed, op_slot);
        if idx < self.in_flight_speculative.len() {
            let moved_slot = self.in_flight_speculative[idx];
            if let Some(moved_op) = self.ops[moved_slot].as_mut() {
                moved_op.spec_idx = Some(idx);
            }
        }
    }
}

/// Hook invoked strictly outside locks whenever a link's
/// `queued_ops_by_class[PriorityClass::Blocking]` transitions from `> 0` to `0`,
/// waking coupled links (e.g., storage workers or `Copies::issue`) that yielded
/// speculative work on a saturated shared bus.
pub type BlockingClearedHook = Arc<dyn Fn() + Send + Sync>;

/// RAII guard that deregisters a [`BlockingClearedHook`] from its
/// `CoupledSetController` when dropped, preventing closure accumulation across
/// model reloads.
pub struct BlockingClearedHookGuard {
    set: Option<Arc<CoupledSetController>>,
    hook_id: u64,
}

impl Drop for BlockingClearedHookGuard {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            set.unregister_blocking_cleared_hook(self.hook_id);
        }
    }
}

pub(crate) struct CoupledSetController {
    #[allow(dead_code)]
    id: CoupledSetId,
    fabric: Arc<Fabric>,
    metrics: Arc<FabricMetricsStore>,
    next_op_id: Arc<AtomicU64>,
    next_sub_id: Arc<AtomicU64>,
    next_batch_id: Arc<AtomicU64>,
    next_res_id: Arc<AtomicU64>,
    state: Mutex<CoupledSetState>,
    next_hook_id: AtomicU64,
    blocking_cleared_hooks: Mutex<Vec<(u64, LinkId, BlockingClearedHook)>>,
}

impl CoupledSetController {
    #[allow(clippy::too_many_arguments)]
    fn new(
        spec: &CoupledSetSpec,
        fabric: Arc<Fabric>,
        metrics: Arc<FabricMetricsStore>,
        next_op_id: Arc<AtomicU64>,
        next_sub_id: Arc<AtomicU64>,
        next_batch_id: Arc<AtomicU64>,
        next_res_id: Arc<AtomicU64>,
    ) -> Arc<Self> {
        let now = Instant::now();
        let mut links = Vec::with_capacity(spec.links.len());
        let mut link_map = HashMap::with_capacity(spec.links.len());
        for &link_id in &spec.links {
            let fabric_idx = fabric.link_index(link_id).expect("valid link in coupled set");
            let local_idx = links.len();
            link_map.insert(link_id, local_idx);
            links.push(LinkState {
                id: link_id,
                fabric_idx,
                ready: IndexedMinHeap::default(),
                ledger: LinkLedger::default(),
                last_integral_at: now,
                min_quantum_s: 100e-6, // 100 µs default floor quantum
            });
        }
        let mut buses = Vec::with_capacity(spec.buses.len());
        let mut bus_map = HashMap::with_capacity(spec.buses.len());
        for &bus_id in &spec.buses {
            let fabric_idx = fabric.bus_index(bus_id).expect("valid bus in coupled set");
            let local_idx = buses.len();
            bus_map.insert(bus_id, local_idx);
            buses.push(BusState {
                id: bus_id,
                fabric_idx,
                ledger: BusLedger::default(),
            });
        }

        // Pre-allocate arenas to avoid heap allocations on the steady-state hot path.
        let initial_cap = 256;
        let mut ops = Vec::with_capacity(initial_cap);
        let mut free_ops = Vec::with_capacity(initial_cap);
        for i in (0..initial_cap).rev() {
            ops.push(None);
            free_ops.push(i);
        }
        let mut subs = Vec::with_capacity(initial_cap);
        let mut free_subs = Vec::with_capacity(initial_cap);
        for i in (0..initial_cap).rev() {
            subs.push(None);
            free_subs.push(i);
        }

        Arc::new(Self {
            id: spec.id,
            fabric,
            metrics,
            next_op_id,
            next_sub_id,
            next_batch_id,
            next_res_id,
            state: Mutex::new(CoupledSetState {
                links,
                link_map,
                buses,
                bus_map,
                ops,
                free_ops,
                subs,
                free_subs,
                join_index: HashMap::with_capacity(128),
                batches: HashMap::with_capacity(32),
                quarantined_leases: Vec::new(),
                fence_reservations: VecDeque::new(),
                in_flight_speculative: Vec::new(),
            }),
            next_hook_id: AtomicU64::new(1),
            blocking_cleared_hooks: Mutex::new(Vec::new()),
        })
    }

    fn register_blocking_cleared_hook(
        self: &Arc<Self>,
        listener_link: LinkId,
        hook: BlockingClearedHook,
    ) -> BlockingClearedHookGuard {
        let hook_id = self.next_hook_id.fetch_add(1, Ordering::Relaxed);
        self.blocking_cleared_hooks
            .lock()
            .expect("blocking_cleared_hooks")
            .push((hook_id, listener_link, hook));
        BlockingClearedHookGuard {
            set: Some(Arc::clone(self)),
            hook_id,
        }
    }

    fn unregister_blocking_cleared_hook(&self, hook_id: u64) {
        self.blocking_cleared_hooks
            .lock()
            .expect("blocking_cleared_hooks")
            .retain(|(id, _, _)| *id != hook_id);
    }

    /// Invoke all registered `Blocking`-cleared hooks for *other* links
    /// (`listener_link != cleared_link`) strictly OUTSIDE `self.state`.
    /// Skipping `listener_link == cleared_link` eliminates same-link lock
    /// re-entrancy by construction (e.g., `Queue` holding `engine.queue()`
    /// while clearing `Blocking` on `LINK_STORAGE_NVME`).
    fn notify_blocking_cleared(&self, cleared_link: LinkId) {
        let hooks: Vec<BlockingClearedHook> = self
            .blocking_cleared_hooks
            .lock()
            .expect("blocking_cleared_hooks")
            .iter()
            .filter_map(|(_, listener_link, hook)| {
                (*listener_link != cleared_link).then(|| Arc::clone(hook))
            })
            .collect();
        for hook in hooks {
            hook();
        }
    }

    fn alloc_sub(state: &mut CoupledSetState, record: SubscriberRecord) -> usize {
        if let Some(slot) = state.free_subs.pop() {
            state.subs[slot] = Some(record);
            slot
        } else {
            let slot = state.subs.len();
            state.subs.push(Some(record));
            slot
        }
    }

    fn alloc_op(state: &mut CoupledSetState, record: PhysicalOpRecord) -> usize {
        if let Some(slot) = state.free_ops.pop() {
            state.ops[slot] = Some(record);
            slot
        } else {
            let slot = state.ops.len();
            state.ops.push(Some(record));
            slot
        }
    }

    fn submit(
        self: &Arc<Self>,
        req: TransferRequest,
        owner: OwnerTag,
    ) -> Result<TransferHandle, SubmitError> {
        let link_spec = self
            .fabric
            .link(req.link)
            .ok_or(SubmitError::UnknownLink(req.link))?;
        if req.dst_lease.domain != link_spec.dst {
            return Err(SubmitError::DomainMismatch {
                expected: link_spec.dst,
                got: req.dst_lease.domain,
            });
        }

        let sub_id = SubscriberId(self.next_sub_id.fetch_add(1, Ordering::Relaxed));
        let completion = SubscriberCompletion::new();

        // Any dropped leases (e.g. redundant leases when joining an existing op)
        // MUST be dropped strictly outside the CoupledSet mutex.
        let mut dropped_leases: Vec<BufferLease> = Vec::new();
        let handle = {
            let mut state = self.state.lock().expect("coupled set state");
            let link_local_idx = *state
                .link_map
                .get(&req.link)
                .expect("link belongs to this coupled set");

            // Check strict join-in-flight preconditions:
            // 1. content_key present (immutable content + source_gen + representation_id)
            // 2. joiner accepts AliasInDomain matching existing op's dst_domain
            // 3. existing op is Queued | Submitting | Issued on the same link.
            let join_target = req.content_key.and_then(|ck| {
                let &op_slot = state.join_index.get(&ck)?;
                let op = state.ops[op_slot].as_ref()?;
                let can_join = matches!(op.state, OpState::Queued | OpState::Submitting | OpState::Issued)
                    && op.link_local_idx == link_local_idx
                    && req.placement == PlacementRequirement::AliasInDomain(op.dst_domain);
                can_join.then_some(op_slot)
            });

            if let Some(op_slot) = join_target {
                let sub_slot = Self::alloc_sub(
                    &mut state,
                    SubscriberRecord {
                        key: req.key,
                        completion: Arc::clone(&completion),
                    },
                );
                // Recompute effective key if still Queued.
                let (op_id, dst_domain, dst_slot, dst_generation, old_class, new_key, reheap, bytes) = {
                    let op = state.ops[op_slot].as_mut().expect("join target exists");
                    op.subscribers.push(sub_slot);
                    let old_class = op.effective_key.class;
                    let reheap = op.state == OpState::Queued && req.key < op.effective_key;
                    if reheap {
                        op.effective_key = req.key;
                    }
                    (
                        op.id,
                        op.dst_domain,
                        op.dst_slot,
                        op.dst_generation,
                        old_class,
                        op.effective_key,
                        reheap,
                        op.bytes,
                    )
                };
                if reheap {
                    let link_state = &mut state.links[link_local_idx];
                    if old_class != new_key.class {
                        link_state.ledger.queued_ops_by_class[old_class.index()] -= 1;
                        link_state.ledger.queued_bytes_by_class[old_class.index()] -= bytes;
                        link_state.ledger.queued_ops_by_class[new_key.class.index()] += 1;
                        link_state.ledger.queued_bytes_by_class[new_key.class.index()] += bytes;
                    }
                    link_state.ready.update_key(op_slot, new_key);
                    self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
                }
                // Drop caller's unused leases outside lock.
                if let Some(src) = req.src_lease {
                    dropped_leases.push(src);
                }
                dropped_leases.push(req.dst_lease);

                TransferHandle {
                    op_id,
                    subscriber_id: sub_id,
                    joined_existing: true,
                    dst_domain,
                    dst_slot,
                    dst_generation,
                    op_slot,
                    sub_slot,
                    controller: Arc::clone(self),
                    completion,
                    detached: false,
                }
            } else {
                let op_id = PhysicalOpId(self.next_op_id.fetch_add(1, Ordering::Relaxed));
                let sub_slot = Self::alloc_sub(
                    &mut state,
                    SubscriberRecord {
                        key: req.key,
                        completion: Arc::clone(&completion),
                    },
                );
                let dst_domain = req.dst_lease.domain;
                let dst_slot = req.dst_lease.slot();
                let dst_generation = req.dst_lease.generation;
                let op_slot = Self::alloc_op(
                    &mut state,
                    PhysicalOpRecord {
                        id: op_id,
                        owner,
                        link_local_idx,
                        state: OpState::Queued,
                        effective_key: req.key,
                        admitted_class: req.key.class,
                        bytes: req.bytes,
                        content_key: req.content_key,
                        compatibility: req.compatibility,
                        direct: req.direct,
                        src_lease: req.src_lease,
                        dst_lease: Some(req.dst_lease),
                        dst_domain,
                        dst_slot,
                        dst_generation,
                        subscribers: vec![sub_slot],
                        hw_retired: false,
                        est_service_s: 0.0,
                        started_at: None,
                        fence: None,
                        spec_idx: None,
                    },
                );
                if let Some(ck) = req.content_key {
                    if req.placement == PlacementRequirement::AliasInDomain(dst_domain) {
                        state.join_index.entry(ck).or_insert(op_slot);
                    }
                }
                let class_idx = req.key.class.index();
                let link_state = &mut state.links[link_local_idx];
                link_state.ledger.queued_ops_by_class[class_idx] += 1;
                link_state.ledger.queued_bytes_by_class[class_idx] += req.bytes;
                link_state.ready.push(op_slot, req.key);
                self.publish_link_metrics(&state.links[link_local_idx], &state.buses);

                TransferHandle {
                    op_id,
                    subscriber_id: sub_id,
                    joined_existing: false,
                    dst_domain,
                    dst_slot,
                    dst_generation,
                    op_slot,
                    sub_slot,
                    controller: Arc::clone(self),
                    completion,
                    detached: false,
                }
            }
        };
        drop(dropped_leases);
        Ok(handle)
    }

    fn reprice_subscriber(&self, op_slot: usize, sub_slot: usize, new_key: ScheduleKey) {
        let mut state = self.state.lock().expect("coupled set state");
        if let Some(sub) = state.subs[sub_slot].as_mut() {
            sub.key = new_key;
        } else {
            return;
        }
        let Some(op) = state.ops[op_slot].as_ref() else {
            return;
        };
        if op.state != OpState::Queued {
            return;
        }
        let min_key = op
            .subscribers
            .iter()
            .filter_map(|&s| state.subs[s].as_ref().map(|r| r.key))
            .min()
            .unwrap_or(new_key);

        let mut blocking_cleared = false;
        let (link_local_idx, old_key, bytes) = {
            let op = state.ops[op_slot].as_mut().expect("op exists");
            let old = op.effective_key;
            if old == min_key {
                return;
            }
            op.effective_key = min_key;
            (op.link_local_idx, old, op.bytes)
        };
        let cleared_link = state.links[link_local_idx].id;
        let link_state = &mut state.links[link_local_idx];
        if old_key.class != min_key.class {
            link_state.ledger.queued_ops_by_class[old_key.class.index()] -= 1;
            link_state.ledger.queued_bytes_by_class[old_key.class.index()] -= bytes;
            link_state.ledger.queued_ops_by_class[min_key.class.index()] += 1;
            link_state.ledger.queued_bytes_by_class[min_key.class.index()] += bytes;
            if old_key.class == PriorityClass::Blocking
                && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0
            {
                blocking_cleared = true;
            }
        }
        link_state.ready.update_key(op_slot, min_key);
        self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
        drop(state);
        if blocking_cleared {
            self.notify_blocking_cleared(cleared_link);
        }
    }

    fn detach_subscriber(&self, op_slot: usize, sub_slot: usize) {
        let mut dropped_leases: Vec<BufferLease> = Vec::new();
        let mut cancelled_completion: Option<Arc<SubscriberCompletion>> = None;
        let mut blocking_cleared_link: Option<LinkId> = None;

        {
            let mut state = self.state.lock().expect("coupled set state");
            let removed_sub = state.subs[sub_slot].take();
            state.free_subs.push(sub_slot);

            let Some(op_state) = state.ops[op_slot].as_ref().map(|o| o.state) else {
                return;
            };

            let remaining_subs = {
                let op = state.ops[op_slot].as_mut().expect("op exists");
                op.subscribers.retain(|&s| s != sub_slot);
                op.subscribers.len()
            };

            match op_state {
                OpState::Queued => {
                    if remaining_subs == 0 {
                        // Cancel immediately with zero bus debit; drop leases outside lock.
                        let mut op = state.ops[op_slot].take().expect("op exists");
                        state.free_ops.push(op_slot);
                        if let Some(ck) = op.content_key {
                            if state.join_index.get(&ck) == Some(&op_slot) {
                                state.join_index.remove(&ck);
                            }
                        }
                        let link_state = &mut state.links[op.link_local_idx];
                        link_state.ready.remove(op_slot);
                        let class_idx = op.effective_key.class.index();
                        link_state.ledger.queued_ops_by_class[class_idx] -= 1;
                        link_state.ledger.queued_bytes_by_class[class_idx] -= op.bytes;
                        if op.effective_key.class == PriorityClass::Blocking
                            && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0
                        {
                            blocking_cleared_link = Some(link_state.id);
                        }
                        self.publish_link_metrics(&state.links[op.link_local_idx], &state.buses);

                        if let Some(src) = op.src_lease.take() {
                            dropped_leases.push(src);
                        }
                        if let Some(dst) = op.dst_lease.take() {
                            dropped_leases.push(dst);
                        }
                        if let Some(sub) = removed_sub {
                            cancelled_completion = Some(sub.completion);
                        }
                    } else {
                        // Recompute effective key among remaining subscribers.
                        let min_key = state.ops[op_slot]
                            .as_ref()
                            .expect("op exists")
                            .subscribers
                            .iter()
                            .filter_map(|&s| state.subs[s].as_ref().map(|r| r.key))
                            .min()
                            .expect("at least one subscriber remains");
                        let (link_local_idx, old_key, bytes) = {
                            let op = state.ops[op_slot].as_mut().expect("op exists");
                            let old = op.effective_key;
                            op.effective_key = min_key;
                            (op.link_local_idx, old, op.bytes)
                        };
                        if old_key != min_key {
                            let link_state = &mut state.links[link_local_idx];
                            if old_key.class != min_key.class {
                                link_state.ledger.queued_ops_by_class[old_key.class.index()] -= 1;
                                link_state.ledger.queued_bytes_by_class[old_key.class.index()] -= bytes;
                                link_state.ledger.queued_ops_by_class[min_key.class.index()] += 1;
                                link_state.ledger.queued_bytes_by_class[min_key.class.index()] += bytes;
                                if old_key.class == PriorityClass::Blocking
                                    && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0
                                {
                                    blocking_cleared_link = Some(link_state.id);
                                }
                            }
                            link_state.ready.update_key(op_slot, min_key);
                            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
                        }
                        if let Some(sub) = removed_sub {
                            cancelled_completion = Some(sub.completion);
                        }
                    }
                }
                OpState::Submitting | OpState::Issued => {
                    // Physical op MUST run to completion; backing leases stay pinned
                    // until BOTH hardware completion AND last subscriber detachment.
                    if let Some(sub) = removed_sub {
                        cancelled_completion = Some(sub.completion);
                    }
                }
                OpState::Retired => {
                    if remaining_subs == 0 {
                        // Hardware already retired AND last subscriber just detached ->
                        // now safe to release backing leases outside lock.
                        if let Some(mut op) = state.ops[op_slot].take() {
                            state.free_ops.push(op_slot);
                            if let Some(src) = op.src_lease.take() {
                                dropped_leases.push(src);
                            }
                            if let Some(dst) = op.dst_lease.take() {
                                dropped_leases.push(dst);
                            }
                        }
                    }
                }
                OpState::Quarantined => {
                    // Backing leases are already moved to `quarantined_leases` and
                    // permanently pinned until teardown.
                    if remaining_subs == 0 {
                        state.ops[op_slot].take();
                        state.free_ops.push(op_slot);
                    }
                }
            }
        }

        if let Some(comp) = cancelled_completion {
            comp.set(TransferOutcome::Cancelled);
        }
        drop(dropped_leases);
        if let Some(cleared_link) = blocking_cleared_link {
            self.notify_blocking_cleared(cleared_link);
        }
    }

    /// Reclaim any GPU-fence-bound bus reservations whose `TimelinePoint` has been reached.
    fn reclaim_reached_fence_reservations(&self, state: &mut CoupledSetState) {
        let mut idx = 0;
        while idx < state.fence_reservations.len() {
            let reached = state.fence_reservations[idx]
                .observer
                .is_reached(state.fence_reservations[idx].point)
                .unwrap_or(true);
            if reached {
                let res = state.fence_reservations.swap_remove_back(idx).expect("valid index");
                if let Some(&bus_local_idx) = state.bus_map.get(&res.bus) {
                    let bus = &mut state.buses[bus_local_idx];
                    bus.ledger.reserved_bps = (bus.ledger.reserved_bps - res.rate_bps).max(0.0);
                    bus.ledger.active_reservations = bus.ledger.active_reservations.saturating_sub(1);
                    self.publish_bus_metrics(bus);
                }
            } else {
                idx += 1;
            }
        }
    }

    /// Compute total residual speculative service time (seconds) currently in
    /// flight across this entire `CoupledSet` at `now`.
    fn residual_speculative_service_time_s(state: &CoupledSetState, now: Instant) -> f64 {
        let mut total = 0.0f64;
        for &op_slot in &state.in_flight_speculative {
            #[cfg(test)]
            SPEC_OP_VISITS.with(|c| c.set(c.get() + 1));
            let op = state.ops[op_slot]
                .as_ref()
                .expect("tracked in-flight speculative op exists");
            debug_assert!(
                matches!(op.state, OpState::Submitting | OpState::Issued)
                    && op.admitted_class.is_speculative()
            );
            let elapsed = op
                .started_at
                .map_or(0.0, |t| now.saturating_duration_since(t).as_secs_f64());
            total += (op.est_service_s - elapsed).max(0.0);
        }
        total
    }

    /// Whether two distinct links (`link_spec`, `other_spec`) share a
    /// `SharedBusId` that is actually saturated / binding (combined demand +
    /// non-IO `reserved_bps` >= bus `peak_bps`, excluding opposite full-duplex
    /// PCIe directions that do not contend).
    fn shares_saturated_bus(
        &self,
        state: &CoupledSetState,
        link_spec: &LinkSpec,
        other_spec: &LinkSpec,
    ) -> bool {
        for c1 in &link_spec.couplings {
            for c2 in &other_spec.couplings {
                if c1.bus != c2.bus {
                    continue;
                }
                // Opposite full-duplex PCIe directions (H2D vs D2H) run concurrently
                // without sharing the same simplex lane budget.
                if matches!(
                    (c1.direction, c2.direction),
                    (BusDirection::FullDuplexH2D, BusDirection::FullDuplexD2H)
                        | (BusDirection::FullDuplexD2H, BusDirection::FullDuplexH2D)
                ) {
                    continue;
                }
                if let Some(&bus_local_idx) = state.bus_map.get(&c1.bus) {
                    let bus_state = &state.buses[bus_local_idx];
                    let bus_spec = &self.fabric.buses()[bus_state.fabric_idx];
                    let combined_demand = bus_state.ledger.reserved_bps
                        + link_spec.curve.peak_bps * (c1.weight as f64)
                        + other_spec.curve.peak_bps * (c2.weight as f64);
                    if combined_demand >= bus_spec.peak_bps {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check whether a `Blocking` op is queued on this link or on another link
    /// in this `CoupledSet` sharing a **saturated** `SharedBus` with `link_local_idx`.
    fn coupled_blocking_waiting(&self, state: &CoupledSetState, link_local_idx: usize) -> bool {
        let link_spec = &self.fabric.links()[state.links[link_local_idx].fabric_idx];
        for other in &state.links {
            if other.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0 {
                continue;
            }
            if other.id == link_spec.id {
                return true;
            }
            let other_spec = &self.fabric.links()[other.fabric_idx];
            if self.shares_saturated_bus(state, link_spec, other_spec) {
                return true;
            }
        }
        false
    }

    fn effective_bps_for_link(&self, state: &CoupledSetState, link_local_idx: usize) -> f64 {
        let link_spec = &self.fabric.links()[state.links[link_local_idx].fabric_idx];
        let mut cap = link_spec.curve.peak_bps;
        for coupling in &link_spec.couplings {
            if let Some(&bus_local_idx) = state.bus_map.get(&coupling.bus) {
                let bus_spec = &self.fabric.buses()[state.buses[bus_local_idx].fabric_idx];
                let rem = (bus_spec.peak_bps - state.buses[bus_local_idx].ledger.reserved_bps)
                    .max(bus_spec.peak_bps * 0.05);
                cap = cap.min(rem / (coupling.weight as f64));
            }
        }
        cap.max(1.0)
    }

    /// Check the three simultaneous admission invariants for `op_bytes` of `class`
    /// on `link_local_idx`.
    fn can_admit(
        &self,
        state: &CoupledSetState,
        link_local_idx: usize,
        class: PriorityClass,
        op_bytes: u64,
        now: Instant,
    ) -> bool {
        let link_state = &state.links[link_local_idx];
        let curve = &self.fabric.links()[link_state.fabric_idx].curve;
        let in_flight_ops = link_state.ledger.total_in_flight_ops();
        if in_flight_ops >= curve.max_in_flight_ops {
            return false;
        }
        let in_flight_bytes = link_state.ledger.total_in_flight_bytes();

        if class.is_speculative() {
            // Invariant 1: Strict Blocking preemption on this link and any coupled bus.
            if self.coupled_blocking_waiting(state, link_local_idx) {
                return false;
            }
            // Invariant 2: Knee cap.
            if in_flight_bytes.saturating_add(op_bytes) > curve.knee_bytes && in_flight_bytes > 0 {
                return false;
            }
            // Invariant 3: Residual speculative service-time bound <= tau_hol.
            let b_eff = self.effective_bps_for_link(state, link_local_idx);
            let op_service_s = (op_bytes as f64) / b_eff;
            let residual_s = Self::residual_speculative_service_time_s(state, now);
            let tau_hol = curve.tau_hol_s(link_state.min_quantum_s);
            if residual_s > 0.0 && residual_s + op_service_s > tau_hol {
                return false;
            }
            true
        } else {
            // Blocking / Streaming: admit up to `due_depth_bytes` (or always if idle).
            let due_in_flight = link_state.ledger.due_in_flight_bytes();
            due_in_flight == 0 || due_in_flight.saturating_add(op_bytes) <= curve.due_depth_bytes
        }
    }

    /// Work-conserving batch claim (`Queued -> Submitting`):
    ///
    /// - Zero artificial wait: if a credit is available and the head op is
    ///   admissible, claims it immediately.
    /// - If the head op is `Blocking`, opportunistically drains additional
    ///   currently-ready `Blocking` ops that are [`BatchCompatibility`]-compatible
    ///   up to `(max_batch_bytes, max_batch_service_time_s)`.
    /// - Atomically debits Link + `CoupledSet` bus credits and releases the lock
    ///   BEFORE the caller invokes any OS/driver call.
    fn claim_next_batch(
        &self,
        link: LinkId,
        owner_filter: Option<OwnerTag>,
        max_batch_bytes: u64,
        max_batch_service_time_s: f64,
        now: Instant,
    ) -> Option<ClaimedBatch> {
        let mut state = self.state.lock().expect("coupled set state");
        self.reclaim_reached_fence_reservations(&mut state);

        let link_local_idx = *state.link_map.get(&link)?;
        let (head_key, head_slot) = match owner_filter {
            None => state.links[link_local_idx].ready.peek()?,
            Some(tag) => {
                // Fast path: heap head belongs to `tag`.
                if let Some((k, s)) = state.links[link_local_idx].ready.peek() {
                    if state.ops[s].as_ref().is_some_and(|op| op.owner == tag) {
                        (k, s)
                    } else {
                        // Scan for lowest-key ready op owned by `tag`.
                        let best = state.links[link_local_idx]
                            .ready
                            .items
                            .iter()
                            .filter(|e| state.ops[e.op_slot].as_ref().is_some_and(|op| op.owner == tag))
                            .min_by_key(|e| e.key)?;
                        (best.key, best.op_slot)
                    }
                } else {
                    return None;
                }
            }
        };
        let head_bytes = state.ops[head_slot].as_ref()?.bytes;
        let head_compat = state.ops[head_slot].as_ref()?.compatibility;

        if !self.can_admit(&state, link_local_idx, head_key.class, head_bytes, now) {
            return None;
        }

        let b_eff = self.effective_bps_for_link(&state, link_local_idx);
        let link_fabric_idx = state.links[link_local_idx].fabric_idx;
        let max_ops = self.fabric.links()[link_fabric_idx].curve.max_in_flight_ops;

        let mut claimed_slots: Vec<usize> = Vec::new();
        let mut batch_bytes = 0u64;

        // Remove the chosen head op.
        assert!(state.links[link_local_idx].ready.remove(head_slot));
        claimed_slots.push(head_slot);
        batch_bytes += head_bytes;

        // Work-conserving opportunistic drain for compatible Blocking ops.
        if head_key.class == PriorityClass::Blocking && head_compat != BatchCompatibility::Isolated {
            while let Some((next_key, next_slot)) = state.links[link_local_idx].ready.peek() {
                if next_key.class != PriorityClass::Blocking {
                    break;
                }
                if let Some(tag) = owner_filter {
                    if state.ops[next_slot].as_ref().is_none_or(|op| op.owner != tag) {
                        break;
                    }
                }
                let (next_bytes, next_compat) = {
                    let op = state.ops[next_slot].as_ref().expect("queued op exists");
                    (op.bytes, op.compatibility)
                };
                if !head_compat.compatible_with(next_compat) {
                    break;
                }
                let projected_bytes = batch_bytes.saturating_add(next_bytes);
                if projected_bytes > max_batch_bytes {
                    break;
                }
                if (projected_bytes as f64) / b_eff > max_batch_service_time_s {
                    break;
                }
                if state.links[link_local_idx].ledger.total_in_flight_ops() + (claimed_slots.len() as u32)
                    >= max_ops
                {
                    break;
                }
                let (_, popped) = state.links[link_local_idx].ready.pop().expect("peeked");
                claimed_slots.push(popped);
                batch_bytes = projected_bytes;
            }
        }

        // Atomically transition all claimed ops `Queued -> Submitting` and debit
        // Link + CoupledSet SharedBus ledgers.
        state.links[link_local_idx].advance_integrals(now);

        let mut claimed_ops = Vec::with_capacity(claimed_slots.len());
        let mut total_bytes = 0u64;

        for &op_slot in &claimed_slots {
            let (op_id, class, bytes, dst_domain, dst_slot, dst_generation, direct) = {
                let op = state.ops[op_slot].as_mut().expect("claimed op exists");
                let class = op.effective_key.class;
                op.state = OpState::Submitting;
                op.admitted_class = class;
                op.est_service_s = (op.bytes as f64) / b_eff;
                op.started_at = Some(now);
                (
                    op.id,
                    class,
                    op.bytes,
                    op.dst_domain,
                    op.dst_slot,
                    op.dst_generation,
                    op.direct,
                )
            };
            if class.is_speculative() {
                state.track_in_flight_speculative(op_slot);
            }

            let link_state = &mut state.links[link_local_idx];
            link_state.ledger.queued_ops_by_class[class.index()] -= 1;
            link_state.ledger.queued_bytes_by_class[class.index()] -= bytes;
            link_state.ledger.submitting_ops += 1;
            link_state.ledger.submitting_bytes += bytes;
            link_state.ledger.in_flight_ops_by_class[class.index()] += 1;
            link_state.ledger.in_flight_bytes_by_class[class.index()] += bytes;

            total_bytes += bytes;
            claimed_ops.push(ClaimedOp {
                op_id,
                bytes,
                class,
                dst_domain,
                dst_slot,
                dst_generation,
                direct,
            });
        }

        // Debit coupled SharedBuses atomically.
        let blocking_cleared = head_key.class == PriorityClass::Blocking
            && state.links[link_local_idx].ledger.queued_ops_by_class
                [PriorityClass::Blocking.index()]
                == 0;
        self.adjust_coupled_buses(&mut state, link_fabric_idx, total_bytes, true);
        self.publish_link_metrics(&state.links[link_local_idx], &state.buses);

        let batch_id = BatchId(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        state.batches.insert(
            batch_id,
            BatchRecord {
                link_local_idx,
                op_slots: claimed_slots,
                state: OpState::Submitting,
            },
        );
        drop(state);
        if blocking_cleared {
            self.notify_blocking_cleared(link);
        }

        Some(ClaimedBatch {
            batch_id,
            link,
            ops: claimed_ops,
            total_bytes,
            compatibility: head_compat,
        })
    }

    /// Atomically submit and claim a single `PriorityClass::Blocking` op
    /// directly in `OpState::Submitting` under the `CoupledSet` lock (zero
    /// race with background link issuers). Debits Link + coupled `SharedBus`
    /// ledgers immediately so concurrent speculative work yields; returns
    /// `(TransferHandle, BatchId)` with ZERO locks held.
    fn claim_direct_blocking(
        self: &Arc<Self>,
        link: LinkId,
        bytes: u64,
        direct: bool,
        now: Instant,
    ) -> Option<(TransferHandle, BatchId)> {
        let mut state = self.state.lock().expect("coupled set state");
        self.reclaim_reached_fence_reservations(&mut state);

        let link_local_idx = *state.link_map.get(&link)?;
        let link_fabric_idx = state.links[link_local_idx].fabric_idx;
        let dst_domain = self.fabric.links()[link_fabric_idx].dst;
        let b_eff = self.effective_bps_for_link(&state, link_local_idx);

        let op_id = PhysicalOpId(self.next_op_id.fetch_add(1, Ordering::Relaxed));
        let sub_id = SubscriberId(self.next_sub_id.fetch_add(1, Ordering::Relaxed));
        let key = ScheduleKey::blocking(crate::schedule::GlobalUrgency(0), 0, 0, op_id.0);
        let completion = SubscriberCompletion::new();

        let sub_slot = Self::alloc_sub(
            &mut state,
            SubscriberRecord {
                key,
                completion: Arc::clone(&completion),
            },
        );
        let dst_lease = BufferLease::unmanaged(
            dst_domain,
            std::ptr::NonNull::dangling(),
            0,
            0,
            bytes as usize,
            1,
            0,
        );
        let op_slot = Self::alloc_op(
            &mut state,
            PhysicalOpRecord {
                id: op_id,
                owner: OwnerTag(0),
                link_local_idx,
                state: OpState::Submitting,
                effective_key: key,
                admitted_class: PriorityClass::Blocking,
                bytes,
                content_key: None,
                compatibility: BatchCompatibility::Isolated,
                direct,
                src_lease: None,
                dst_lease: Some(dst_lease),
                dst_domain,
                dst_slot: 0,
                dst_generation: 0,
                subscribers: vec![sub_slot],
                hw_retired: false,
                est_service_s: (bytes as f64) / b_eff,
                started_at: Some(now),
                fence: None,
                spec_idx: None,
            },
        );

        let link_state = &mut state.links[link_local_idx];
        link_state.advance_integrals(now);
        link_state.ledger.submitting_ops += 1;
        link_state.ledger.submitting_bytes += bytes;
        link_state.ledger.in_flight_ops_by_class[PriorityClass::Blocking.index()] += 1;
        link_state.ledger.in_flight_bytes_by_class[PriorityClass::Blocking.index()] += bytes;

        self.adjust_coupled_buses(&mut state, link_fabric_idx, bytes, true);
        self.publish_link_metrics(&state.links[link_local_idx], &state.buses);

        let batch_id = BatchId(self.next_batch_id.fetch_add(1, Ordering::Relaxed));
        state.batches.insert(
            batch_id,
            BatchRecord {
                link_local_idx,
                op_slots: vec![op_slot],
                state: OpState::Submitting,
            },
        );

        let handle = TransferHandle {
            op_id,
            subscriber_id: sub_id,
            joined_existing: false,
            dst_domain,
            dst_slot: 0,
            dst_generation: 0,
            op_slot,
            sub_slot,
            controller: Arc::clone(self),
            completion,
            detached: false,
        };
        Some((handle, batch_id))
    }

    /// O(1), zero-allocation direct link accounting for self-scheduled links
    /// (e.g., `seismic-io` storage read workers): enqueue `count` pieces
    /// totaling `bytes` in `class`.
    fn direct_enqueue(&self, link: LinkId, class: PriorityClass, count: u32, bytes: u64) {
        let mut state = self.state.lock().expect("coupled set state");
        let Some(&link_local_idx) = state.link_map.get(&link) else {
            return;
        };
        let link_state = &mut state.links[link_local_idx];
        link_state.ledger.queued_ops_by_class[class.index()] += count;
        link_state.ledger.queued_bytes_by_class[class.index()] += bytes;
        self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
    }

    /// O(1) direct reprice of `count` pieces totaling `bytes` from `old_class` to `new_class`.
    fn direct_reprice(
        &self,
        link: LinkId,
        old_class: PriorityClass,
        new_class: PriorityClass,
        count: u32,
        bytes: u64,
    ) {
        if old_class == new_class || count == 0 {
            return;
        }
        let blocking_cleared = {
            let mut state = self.state.lock().expect("coupled set state");
            let Some(&link_local_idx) = state.link_map.get(&link) else {
                return;
            };
            let link_state = &mut state.links[link_local_idx];
            link_state.ledger.queued_ops_by_class[old_class.index()] = link_state
                .ledger
                .queued_ops_by_class[old_class.index()]
                .saturating_sub(count);
            link_state.ledger.queued_bytes_by_class[old_class.index()] = link_state
                .ledger
                .queued_bytes_by_class[old_class.index()]
                .saturating_sub(bytes);
            link_state.ledger.queued_ops_by_class[new_class.index()] += count;
            link_state.ledger.queued_bytes_by_class[new_class.index()] += bytes;
            let cleared = old_class == PriorityClass::Blocking
                && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0;
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
            cleared
        };
        if blocking_cleared {
            self.notify_blocking_cleared(link);
        }
    }

    /// O(1) direct cancel of `count` queued pieces totaling `bytes` in `class`.
    fn direct_cancel(&self, link: LinkId, class: PriorityClass, count: u32, bytes: u64) {
        if count == 0 {
            return;
        }
        let blocking_cleared = {
            let mut state = self.state.lock().expect("coupled set state");
            let Some(&link_local_idx) = state.link_map.get(&link) else {
                return;
            };
            let link_state = &mut state.links[link_local_idx];
            link_state.ledger.queued_ops_by_class[class.index()] = link_state
                .ledger
                .queued_ops_by_class[class.index()]
                .saturating_sub(count);
            link_state.ledger.queued_bytes_by_class[class.index()] = link_state
                .ledger
                .queued_bytes_by_class[class.index()]
                .saturating_sub(bytes);
            let cleared = class == PriorityClass::Blocking
                && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0;
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
            cleared
        };
        if blocking_cleared {
            self.notify_blocking_cleared(link);
        }
    }

    /// O(1) direct transition of one queued piece (`bytes`, `class`) to in-flight,
    /// debiting Link + coupled `SharedBus` ledgers in a single lock acquisition.
    fn direct_begin(&self, link: LinkId, class: PriorityClass, bytes: u64, now: Instant) {
        let blocking_cleared = {
            let mut state = self.state.lock().expect("coupled set state");
            let Some(&link_local_idx) = state.link_map.get(&link) else {
                return;
            };
            let link_fabric_idx = state.links[link_local_idx].fabric_idx;
            let link_state = &mut state.links[link_local_idx];
            link_state.advance_integrals(now);
            link_state.ledger.queued_ops_by_class[class.index()] = link_state
                .ledger
                .queued_ops_by_class[class.index()]
                .saturating_sub(1);
            link_state.ledger.queued_bytes_by_class[class.index()] = link_state
                .ledger
                .queued_bytes_by_class[class.index()]
                .saturating_sub(bytes);
            link_state.ledger.in_flight_ops_by_class[class.index()] += 1;
            link_state.ledger.in_flight_bytes_by_class[class.index()] += bytes;
            let cleared = class == PriorityClass::Blocking
                && link_state.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0;
            self.adjust_coupled_buses(&mut state, link_fabric_idx, bytes, true);
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
            cleared
        };
        if blocking_cleared {
            self.notify_blocking_cleared(link);
        }
    }

    /// O(1) direct retirement of one in-flight piece (`bytes`, `class`), crediting
    /// Link + coupled `SharedBus` ledgers in a single lock acquisition.
    fn direct_complete(
        &self,
        link: LinkId,
        class: PriorityClass,
        bytes: u64,
        direct: bool,
        now: Instant,
    ) {
        let mut state = self.state.lock().expect("coupled set state");
        let Some(&link_local_idx) = state.link_map.get(&link) else {
            return;
        };
        let link_fabric_idx = state.links[link_local_idx].fabric_idx;
        let link_state = &mut state.links[link_local_idx];
        link_state.advance_integrals(now);
        link_state.ledger.in_flight_ops_by_class[class.index()] = link_state
            .ledger
            .in_flight_ops_by_class[class.index()]
            .saturating_sub(1);
        link_state.ledger.in_flight_bytes_by_class[class.index()] = link_state
            .ledger
            .in_flight_bytes_by_class[class.index()]
            .saturating_sub(bytes);
        link_state.ledger.completed_ops_by_class[class.index()] += 1;
        link_state.ledger.completed_bytes_by_class[class.index()] += bytes;
        if direct {
            link_state.ledger.direct_bytes += bytes;
        } else {
            link_state.ledger.bounced_bytes += bytes;
        }
        self.adjust_coupled_buses(&mut state, link_fabric_idx, bytes, false);
        self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
    }

    /// Whether any *other* link sharing a **saturated** coupled `SharedBusId`
    /// with `link` has a `PriorityClass::Blocking` op queued (cross-link preemption).
    fn has_other_coupled_blocking_waiting(&self, link: LinkId) -> bool {
        let state = self.state.lock().expect("coupled set state");
        let Some(&link_local_idx) = state.link_map.get(&link) else {
            return false;
        };
        let link_spec = &self.fabric.links()[state.links[link_local_idx].fabric_idx];
        for other in &state.links {
            if other.id == link_spec.id {
                continue;
            }
            if other.ledger.queued_ops_by_class[PriorityClass::Blocking.index()] == 0 {
                continue;
            }
            let other_spec = &self.fabric.links()[other.fabric_idx];
            if self.shares_saturated_bus(&state, link_spec, other_spec) {
                return true;
            }
        }
        false
    }

    /// `Submitting -> Issued` after driver/OS accepts the batch and records its fence.
    fn confirm_issued(&self, batch_id: BatchId, fence: Option<TimelinePoint>) {
        let mut state = self.state.lock().expect("coupled set state");
        let Some(batch) = state.batches.get_mut(&batch_id) else {
            panic!("confirm_issued on unknown batch {batch_id:?}");
        };
        assert_eq!(
            batch.state,
            OpState::Submitting,
            "confirm_issued requires Submitting state, got {:?}",
            batch.state
        );
        batch.state = OpState::Issued;
        let link_local_idx = batch.link_local_idx;
        let op_slots = batch.op_slots.clone();
        for op_slot in op_slots {
            let bytes = {
                let op = state.ops[op_slot].as_mut().expect("batch op exists");
                op.state = OpState::Issued;
                op.fence = fence;
                op.bytes
            };
            let link_state = &mut state.links[link_local_idx];
            link_state.ledger.submitting_ops -= 1;
            link_state.ledger.submitting_bytes -= bytes;
        }
    }

    /// `Submitting -> Queued` on transient driver backpressure (credits rolled back).
    fn rollback_transient(&self, batch_id: BatchId, now: Instant) {
        let mut dropped_leases = Vec::new();
        {
            let mut state = self.state.lock().expect("coupled set state");
            let batch = state
                .batches
                .remove(&batch_id)
                .expect("rollback_transient on unknown batch");
            assert_eq!(batch.state, OpState::Submitting);

            let link_local_idx = batch.link_local_idx;
            let link_fabric_idx = state.links[link_local_idx].fabric_idx;
            state.links[link_local_idx].advance_integrals(now);

            let mut total_bytes = 0u64;
            for op_slot in batch.op_slots {
                state.untrack_in_flight_speculative(op_slot);
                let (class, bytes, no_subs) = {
                    let op = state.ops[op_slot].as_ref().expect("batch op exists");
                    (op.admitted_class, op.bytes, op.subscribers.is_empty())
                };
                total_bytes += bytes;

                let link_state = &mut state.links[link_local_idx];
                link_state.ledger.submitting_ops -= 1;
                link_state.ledger.submitting_bytes -= bytes;
                link_state.ledger.in_flight_ops_by_class[class.index()] -= 1;
                link_state.ledger.in_flight_bytes_by_class[class.index()] -= bytes;

                if no_subs {
                    // All subscribers detached while Submitting -> retire as cancelled now.
                    let mut op = state.ops[op_slot].take().expect("op exists");
                    state.free_ops.push(op_slot);
                    if let Some(ck) = op.content_key {
                        if state.join_index.get(&ck) == Some(&op_slot) {
                            state.join_index.remove(&ck);
                        }
                    }
                    if let Some(src) = op.src_lease.take() {
                        dropped_leases.push(src);
                    }
                    if let Some(dst) = op.dst_lease.take() {
                        dropped_leases.push(dst);
                    }
                } else {
                    let key = {
                        let op = state.ops[op_slot].as_mut().expect("batch op exists");
                        op.state = OpState::Queued;
                        op.started_at = None;
                        op.effective_key
                    };
                    let link_state = &mut state.links[link_local_idx];
                    link_state.ledger.queued_ops_by_class[key.class.index()] += 1;
                    link_state.ledger.queued_bytes_by_class[key.class.index()] += bytes;
                    link_state.ready.push(op_slot, key);
                }
            }

            self.adjust_coupled_buses(&mut state, link_fabric_idx, total_bytes, false);
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
        }
        drop(dropped_leases);
    }

    /// `Submitting -> Retired` on definite pre-issue refusal: credits & pins
    /// rolled back, callers notified `RefusedBeforeIssue`, leases dropped outside lock.
    fn refuse_before_issue(&self, batch_id: BatchId, reason: String, now: Instant) {
        let mut dropped_leases = Vec::new();
        let mut notify = Vec::new();
        {
            let mut state = self.state.lock().expect("coupled set state");
            let batch = state
                .batches
                .remove(&batch_id)
                .expect("refuse_before_issue on unknown batch");
            assert_eq!(batch.state, OpState::Submitting);

            let link_local_idx = batch.link_local_idx;
            let link_fabric_idx = state.links[link_local_idx].fabric_idx;
            state.links[link_local_idx].advance_integrals(now);

            let mut total_bytes = 0u64;
            for op_slot in batch.op_slots {
                state.untrack_in_flight_speculative(op_slot);
                let mut op = state.ops[op_slot].take().expect("batch op exists");
                state.free_ops.push(op_slot);
                let class = op.admitted_class;
                total_bytes += op.bytes;

                if let Some(ck) = op.content_key {
                    if state.join_index.get(&ck) == Some(&op_slot) {
                        state.join_index.remove(&ck);
                    }
                }

                let link_state = &mut state.links[link_local_idx];
                link_state.ledger.submitting_ops -= 1;
                link_state.ledger.submitting_bytes -= op.bytes;
                link_state.ledger.in_flight_ops_by_class[class.index()] -= 1;
                link_state.ledger.in_flight_bytes_by_class[class.index()] -= op.bytes;

                for &sub_slot in &op.subscribers {
                    if let Some(sub) = state.subs[sub_slot].as_ref() {
                        notify.push(Arc::clone(&sub.completion));
                    }
                }
                if let Some(src) = op.src_lease.take() {
                    dropped_leases.push(src);
                }
                if let Some(dst) = op.dst_lease.take() {
                    dropped_leases.push(dst);
                }
            }

            self.adjust_coupled_buses(&mut state, link_fabric_idx, total_bytes, false);
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
        }

        let outcome = TransferOutcome::RefusedBeforeIssue(reason);
        for comp in notify {
            comp.set(outcome.clone());
        }
        drop(dropped_leases);
    }

    /// `Submitting | Issued -> Quarantined` on ambiguous/partial driver failure
    /// or device fault. Backing [`BufferLease`]s are quarantined and retained
    /// permanently until [`CoupledSetController::teardown_quarantined`].
    fn quarantine_batch(&self, batch_id: BatchId, reason: String, now: Instant) {
        let mut notify = Vec::new();
        {
            let mut state = self.state.lock().expect("coupled set state");
            let batch = state
                .batches
                .remove(&batch_id)
                .expect("quarantine_batch on unknown batch");
            assert!(
                matches!(batch.state, OpState::Submitting | OpState::Issued),
                "quarantine_batch requires Submitting or Issued, got {:?}",
                batch.state
            );
            let was_submitting = batch.state == OpState::Submitting;
            let link_local_idx = batch.link_local_idx;
            let link_fabric_idx = state.links[link_local_idx].fabric_idx;
            state.links[link_local_idx].advance_integrals(now);

            let mut total_bytes = 0u64;
            for op_slot in batch.op_slots {
                state.untrack_in_flight_speculative(op_slot);
                let (ck, class, bytes, src_lease, dst_lease, sub_slots, no_subs) = {
                    let op = state.ops[op_slot].as_mut().expect("batch op exists");
                    op.state = OpState::Quarantined;
                    let src = op.src_lease.take();
                    let dst = op.dst_lease.take();
                    (
                        op.content_key,
                        op.admitted_class,
                        op.bytes,
                        src,
                        dst,
                        op.subscribers.clone(),
                        op.subscribers.is_empty(),
                    )
                };
                total_bytes += bytes;

                if let Some(ck) = ck {
                    if state.join_index.get(&ck) == Some(&op_slot) {
                        state.join_index.remove(&ck);
                    }
                }
                if let Some(mut src) = src_lease {
                    src.quarantine();
                    state.quarantined_leases.push(src);
                }
                if let Some(mut dst) = dst_lease {
                    dst.quarantine();
                    state.quarantined_leases.push(dst);
                }
                for sub_slot in sub_slots {
                    if let Some(sub) = state.subs[sub_slot].as_ref() {
                        notify.push(Arc::clone(&sub.completion));
                    }
                }
                if no_subs {
                    state.ops[op_slot].take();
                    state.free_ops.push(op_slot);
                }

                let link_state = &mut state.links[link_local_idx];
                if was_submitting {
                    link_state.ledger.submitting_ops -= 1;
                    link_state.ledger.submitting_bytes -= bytes;
                }
                link_state.ledger.in_flight_ops_by_class[class.index()] -= 1;
                link_state.ledger.in_flight_bytes_by_class[class.index()] -= bytes;
                link_state.ledger.quarantined_ops += 1;
            }

            self.adjust_coupled_buses(&mut state, link_fabric_idx, total_bytes, false);
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
        }

        let outcome = TransferOutcome::Quarantined(reason);
        for comp in notify {
            comp.set(outcome.clone());
        }
    }

    /// `Issued -> Retired` on hardware/OS completion: releases Link + `CoupledSet`
    /// credits atomically, updates exact ledger & seqlock metrics, notifies
    /// subscribers `Completed`, and drops leases outside locks ONLY for ops
    /// whose last subscriber has also detached.
    fn retire_issued(&self, batch_id: BatchId, now: Instant) {
        let mut dropped_leases = Vec::new();
        let mut notify: Vec<(Arc<SubscriberCompletion>, Option<TimelinePoint>)> = Vec::new();

        {
            let mut state = self.state.lock().expect("coupled set state");
            let batch = state
                .batches
                .remove(&batch_id)
                .expect("retire_issued on unknown batch");
            assert_eq!(batch.state, OpState::Issued, "retire_issued requires Issued state");

            let link_local_idx = batch.link_local_idx;
            let link_fabric_idx = state.links[link_local_idx].fabric_idx;
            state.links[link_local_idx].advance_integrals(now);

            let mut total_bytes = 0u64;
            for op_slot in batch.op_slots {
                state.untrack_in_flight_speculative(op_slot);
                let (ck, class, bytes, direct, fence, sub_slots, no_subs) = {
                    let op = state.ops[op_slot].as_mut().expect("batch op exists");
                    op.state = OpState::Retired;
                    op.hw_retired = true;
                    (
                        op.content_key,
                        op.admitted_class,
                        op.bytes,
                        op.direct,
                        op.fence,
                        op.subscribers.clone(),
                        op.subscribers.is_empty(),
                    )
                };
                total_bytes += bytes;

                if let Some(ck) = ck {
                    if state.join_index.get(&ck) == Some(&op_slot) {
                        state.join_index.remove(&ck);
                    }
                }
                for sub_slot in sub_slots {
                    if let Some(sub) = state.subs[sub_slot].as_ref() {
                        notify.push((Arc::clone(&sub.completion), fence));
                    }
                }

                let link_state = &mut state.links[link_local_idx];
                link_state.ledger.in_flight_ops_by_class[class.index()] -= 1;
                link_state.ledger.in_flight_bytes_by_class[class.index()] -= bytes;
                link_state.ledger.completed_ops_by_class[class.index()] += 1;
                link_state.ledger.completed_bytes_by_class[class.index()] += bytes;
                if direct {
                    link_state.ledger.direct_bytes += bytes;
                } else {
                    link_state.ledger.bounced_bytes += bytes;
                }

                if no_subs {
                    // All subscribers already detached while in flight -> safe to drop leases now.
                    if let Some(mut op) = state.ops[op_slot].take() {
                        state.free_ops.push(op_slot);
                        if let Some(src) = op.src_lease.take() {
                            dropped_leases.push(src);
                        }
                        if let Some(dst) = op.dst_lease.take() {
                            dropped_leases.push(dst);
                        }
                    }
                }
            }

            self.adjust_coupled_buses(&mut state, link_fabric_idx, total_bytes, false);
            self.publish_link_metrics(&state.links[link_local_idx], &state.buses);
        }

        for (comp, fence) in notify {
            comp.set(TransferOutcome::Completed { fence });
        }
        drop(dropped_leases);
    }

    /// Release all quarantined leases on explicit device teardown.
    fn teardown_quarantined(&self) -> usize {
        let leases = {
            let mut state = self.state.lock().expect("coupled set state");
            std::mem::take(&mut state.quarantined_leases)
        };
        let count = leases.len();
        for lease in leases {
            lease.release_on_teardown();
        }
        count
    }

    fn quarantined_lease_count(&self) -> usize {
        self.state
            .lock()
            .expect("coupled set state")
            .quarantined_leases
            .len()
    }

    fn reserve_bus_scoped(self: &Arc<Self>, bus: SharedBusId, rate_bps: f64) -> BusReservation {
        {
            let mut state = self.state.lock().expect("coupled set state");
            let bus_local_idx = *state
                .bus_map
                .get(&bus)
                .expect("bus belongs to this coupled set");
            let bus_state = &mut state.buses[bus_local_idx];
            bus_state.ledger.reserved_bps += rate_bps;
            bus_state.ledger.active_reservations += 1;
            self.publish_bus_metrics(bus_state);
        }
        let controller = Arc::clone(self);
        BusReservation::host_scoped(
            bus,
            rate_bps,
            Box::new(move || {
                let mut state = controller.state.lock().expect("coupled set state");
                if let Some(&bus_local_idx) = state.bus_map.get(&bus) {
                    let bus_state = &mut state.buses[bus_local_idx];
                    bus_state.ledger.reserved_bps =
                        (bus_state.ledger.reserved_bps - rate_bps).max(0.0);
                    bus_state.ledger.active_reservations =
                        bus_state.ledger.active_reservations.saturating_sub(1);
                    controller.publish_bus_metrics(bus_state);
                }
            }),
        )
    }

    fn reserve_bus_until_fence(
        &self,
        bus: SharedBusId,
        rate_bps: f64,
        point: TimelinePoint,
        observer: Arc<dyn TimelineObserver>,
    ) -> FenceReservationId {
        let id = FenceReservationId(self.next_res_id.fetch_add(1, Ordering::Relaxed));
        let mut state = self.state.lock().expect("coupled set state");
        let bus_local_idx = *state
            .bus_map
            .get(&bus)
            .expect("bus belongs to this coupled set");
        let bus_state = &mut state.buses[bus_local_idx];
        bus_state.ledger.reserved_bps += rate_bps;
        bus_state.ledger.active_reservations += 1;
        self.publish_bus_metrics(bus_state);

        state.fence_reservations.push_back(FenceBusReservation {
            bus,
            rate_bps,
            point,
            observer,
        });
        id
    }

    fn poll_fence_reservations(&self) {
        let mut state = self.state.lock().expect("coupled set state");
        self.reclaim_reached_fence_reservations(&mut state);
    }

    fn link_ledger(&self, link: LinkId) -> Option<LinkLedger> {
        let state = self.state.lock().expect("coupled set state");
        let &idx = state.link_map.get(&link)?;
        Some(state.links[idx].ledger.clone())
    }

    fn bus_ledger(&self, bus: SharedBusId) -> Option<BusLedger> {
        let state = self.state.lock().expect("coupled set state");
        let &idx = state.bus_map.get(&bus)?;
        Some(state.buses[idx].ledger.clone())
    }

    fn adjust_coupled_buses(
        &self,
        state: &mut CoupledSetState,
        link_fabric_idx: usize,
        bytes: u64,
        debit: bool,
    ) {
        let couplings = &self.fabric.links()[link_fabric_idx].couplings;
        for coupling in couplings {
            let weighted = ((bytes as f64) * (coupling.weight as f64)).round() as u64;
            if let Some(&bus_local_idx) = state.bus_map.get(&coupling.bus) {
                let bus = &mut state.buses[bus_local_idx];
                if debit {
                    bus.ledger.in_flight_weighted_bytes =
                        bus.ledger.in_flight_weighted_bytes.saturating_add(weighted);
                } else {
                    bus.ledger.in_flight_weighted_bytes =
                        bus.ledger.in_flight_weighted_bytes.saturating_sub(weighted);
                }
                self.publish_bus_metrics(bus);
            }
        }
    }

    fn publish_link_metrics(&self, link_state: &LinkState, buses: &[BusState]) {
        let link_spec = &self.fabric.links()[link_state.fabric_idx];
        let mut eff_bps = link_spec.curve.peak_bps;
        for coupling in &link_spec.couplings {
            if let Some(bus_state) = buses.iter().find(|b| b.id == coupling.bus) {
                let bus_spec = &self.fabric.buses()[bus_state.fabric_idx];
                let rem = (bus_spec.peak_bps - bus_state.ledger.reserved_bps)
                    .max(bus_spec.peak_bps * 0.05);
                eff_bps = eff_bps.min(rem / (coupling.weight as f64));
            }
        }
        self.metrics.publish_link(
            link_state.fabric_idx,
            LinkMetrics {
                effective_bps: eff_bps.max(1.0),
                in_flight_bytes: link_state.ledger.total_in_flight_bytes(),
                queued_blocking_ops: link_state.ledger.queued_ops_by_class
                    [PriorityClass::Blocking.index()],
                in_flight_ops: link_state.ledger.total_in_flight_ops(),
            },
        );
    }

    fn publish_bus_metrics(&self, bus_state: &BusState) {
        let bus_spec = &self.fabric.buses()[bus_state.fabric_idx];
        let util = (bus_state.ledger.reserved_bps / bus_spec.peak_bps).clamp(0.0, 2.0);
        self.metrics.publish_bus(
            bus_state.fabric_idx,
            BusMetrics {
                utilization: util,
                reserved_bps: bus_state.ledger.reserved_bps,
                in_flight_weighted_bytes: bus_state.ledger.in_flight_weighted_bytes,
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Top-level IoPlane façade
// ---------------------------------------------------------------------------

/// Top-level `seismic-io` v2 unified data-movement and contention plane.
pub struct IoPlane {
    fabric: Arc<Fabric>,
    metrics: Arc<FabricMetricsStore>,
    sets: Vec<Arc<CoupledSetController>>,
    batch_to_set: Mutex<HashMap<BatchId, usize>>,
}

impl IoPlane {
    /// Build the default unified [`Fabric`] coupling `CopyEngineH2D`,
    /// `CopyEngineD2H`, and `NvmeDirect` (`LINK_STORAGE_NVME`) onto shared
    /// `BUS_HOST_DDR` + `BUS_PCIE_ROOT` (one `CoupledSet`), plus
    /// `BUS_UNIFIED_FABRIC`.
    pub fn default_fabric() -> Fabric {
        let copy_curve = LinkCurve {
            latency_s: 5.1e-6,
            peak_bps: 25e9,
            knee_bytes: 8 * 1024 * 1024,
            due_depth_bytes: 64 * 1024 * 1024,
            max_in_flight_ops: 8,
            alignment_bytes: 256,
            uncertainty_rel: 0.05,
        };
        let storage_curve = LinkCurve {
            latency_s: 200e-6,
            peak_bps: 6e9,
            knee_bytes: 8 * 1024 * 1024,
            due_depth_bytes: 64 * 1024 * 1024,
            max_in_flight_ops: 32,
            alignment_bytes: 4096,
            uncertainty_rel: 0.05,
        };
        Fabric::build(FabricSpec {
            revision: 1,
            buses: vec![
                SharedBusSpec {
                    id: BUS_HOST_DDR,
                    peak_bps: 60e9,
                    uncertainty_rel: 0.05,
                },
                SharedBusSpec {
                    id: BUS_PCIE_ROOT,
                    peak_bps: 32e9,
                    uncertainty_rel: 0.05,
                },
                SharedBusSpec {
                    id: BUS_UNIFIED_FABRIC,
                    peak_bps: 250e9,
                    uncertainty_rel: 0.05,
                },
            ],
            links: vec![
                LinkSpec {
                    id: LINK_COPY_H2D,
                    kind: LinkKind::CopyEngineH2D,
                    src: DOMAIN_HOST_RAM,
                    dst: DOMAIN_DEVICE_VRAM,
                    curve: copy_curve,
                    couplings: vec![
                        BusCoupling {
                            bus: BUS_HOST_DDR,
                            direction: BusDirection::Read,
                            weight: 1.0,
                        },
                        BusCoupling {
                            bus: BUS_PCIE_ROOT,
                            direction: BusDirection::FullDuplexH2D,
                            weight: 1.0,
                        },
                    ],
                },
                LinkSpec {
                    id: LINK_COPY_D2H,
                    kind: LinkKind::CopyEngineD2H,
                    src: DOMAIN_DEVICE_VRAM,
                    dst: DOMAIN_HOST_RAM,
                    curve: copy_curve,
                    couplings: vec![
                        BusCoupling {
                            bus: BUS_HOST_DDR,
                            direction: BusDirection::Write,
                            weight: 1.0,
                        },
                        BusCoupling {
                            bus: BUS_PCIE_ROOT,
                            direction: BusDirection::FullDuplexD2H,
                            weight: 1.0,
                        },
                    ],
                },
                LinkSpec {
                    id: LINK_STORAGE_NVME,
                    kind: LinkKind::NvmeDirect,
                    src: DOMAIN_STORAGE,
                    dst: DOMAIN_HOST_RAM,
                    curve: storage_curve,
                    couplings: vec![
                        BusCoupling {
                            bus: BUS_HOST_DDR,
                            direction: BusDirection::Write,
                            weight: 1.0,
                        },
                        BusCoupling {
                            bus: BUS_PCIE_ROOT,
                            direction: BusDirection::FullDuplexD2H,
                            weight: 1.0,
                        },
                    ],
                },
            ],
        })
        .expect("default unified fabric spec is valid")
    }

    /// Create an isolated [`IoPlane`] with [`Self::default_fabric`] (useful for
    /// unit tests that should not share process-global state).
    pub fn new_default() -> Arc<Self> {
        Self::new(Self::default_fabric())
    }

    /// Construct a new [`IoPlane`] from an immutable [`Fabric`] graph.
    pub fn new(fabric: Fabric) -> Arc<Self> {
        let fabric = Arc::new(fabric);
        let metrics = Arc::new(FabricMetricsStore::new(&fabric));
        let next_op_id = Arc::new(AtomicU64::new(1));
        let next_sub_id = Arc::new(AtomicU64::new(1));
        let next_batch_id = Arc::new(AtomicU64::new(1));
        let next_res_id = Arc::new(AtomicU64::new(1));

        let sets = fabric
            .coupled_sets()
            .iter()
            .map(|spec| {
                CoupledSetController::new(
                    spec,
                    Arc::clone(&fabric),
                    Arc::clone(&metrics),
                    Arc::clone(&next_op_id),
                    Arc::clone(&next_sub_id),
                    Arc::clone(&next_batch_id),
                    Arc::clone(&next_res_id),
                )
            })
            .collect();

        Arc::new(Self {
            fabric,
            metrics,
            sets,
            batch_to_set: Mutex::new(HashMap::new()),
        })
    }

    pub fn fabric(&self) -> &Fabric {
        &self.fabric
    }

    /// Lock-free seqlock quote (~15 ns) for `bytes` on `link`.
    #[inline]
    pub fn quote(&self, link: LinkId, bytes: u64) -> Option<TransferQuote> {
        self.metrics.quote(&self.fabric, link, bytes)
    }

    /// Lock-free read of one link's live metrics.
    #[inline]
    pub fn link_metrics(&self, link: LinkId) -> Option<LinkMetrics> {
        let idx = self.fabric.link_index(link)?;
        Some(self.metrics.read_link(idx))
    }

    /// Lock-free read of one shared bus's live metrics.
    #[inline]
    pub fn bus_metrics(&self, bus: SharedBusId) -> Option<BusMetrics> {
        let idx = self.fabric.bus_index(bus)?;
        Some(self.metrics.read_bus(idx))
    }

    /// Submit a transfer request (`Queued`, zero bus debit). Automatically
    /// joins an existing in-flight op when strict [`ContentKey`] +
    /// [`PlacementRequirement::AliasInDomain`] preconditions hold.
    pub fn submit(&self, req: TransferRequest) -> Result<TransferHandle, SubmitError> {
        self.submit_owned(req, OwnerTag(0))
    }

    /// Submit a transfer request tagged with `owner` so multiple independent
    /// issuers sharing one [`IoPlane`] only claim their own ops.
    pub fn submit_owned(
        &self,
        req: TransferRequest,
        owner: OwnerTag,
    ) -> Result<TransferHandle, SubmitError> {
        let set_id = self
            .fabric
            .coupled_set_of_link(req.link)
            .ok_or(SubmitError::UnknownLink(req.link))?;
        self.sets[set_id.0 as usize].submit(req, owner)
    }

    /// Issuer-only: work-conservingly claim the next admissible batch on `link`
    /// (`Queued -> Submitting`). Returns with ZERO locks held.
    pub fn claim_next_batch(
        &self,
        link: LinkId,
        max_batch_bytes: u64,
        max_batch_service_time_s: f64,
    ) -> Option<ClaimedBatch> {
        self.claim_next_batch_at(link, max_batch_bytes, max_batch_service_time_s, Instant::now())
    }

    /// Issuer-only: claim the next admissible batch on `link` belonging
    /// strictly to `owner`.
    pub fn claim_next_batch_for_owner(
        &self,
        link: LinkId,
        owner: OwnerTag,
        max_batch_bytes: u64,
        max_batch_service_time_s: f64,
    ) -> Option<ClaimedBatch> {
        let set_id = self.fabric.coupled_set_of_link(link)?;
        let set_idx = set_id.0 as usize;
        let batch = self.sets[set_idx].claim_next_batch(
            link,
            Some(owner),
            max_batch_bytes,
            max_batch_service_time_s,
            Instant::now(),
        )?;
        self.batch_to_set
            .lock()
            .expect("batch_to_set")
            .insert(batch.batch_id, set_idx);
        Some(batch)
    }

    /// Deterministic timestamp variant of [`Self::claim_next_batch`] for replay tests.
    pub fn claim_next_batch_at(
        &self,
        link: LinkId,
        max_batch_bytes: u64,
        max_batch_service_time_s: f64,
        now: Instant,
    ) -> Option<ClaimedBatch> {
        let set_id = self.fabric.coupled_set_of_link(link)?;
        let set_idx = set_id.0 as usize;
        let batch = self.sets[set_idx].claim_next_batch(
            link,
            None,
            max_batch_bytes,
            max_batch_service_time_s,
            now,
        )?;
        self.batch_to_set
            .lock()
            .expect("batch_to_set")
            .insert(batch.batch_id, set_idx);
        Some(batch)
    }

    /// Atomically submit and claim a synchronous `PriorityClass::Blocking` op
    /// directly in `OpState::Submitting` (for demand callers such as
    /// `land_decode_misses` that invoke the driver directly with borrowed runs
    /// while sharing the exact same `CoupledSet` wire/bus ledger).
    pub fn claim_direct_blocking(
        &self,
        link: LinkId,
        bytes: u64,
        direct: bool,
    ) -> Option<(TransferHandle, BatchId)> {
        let set_id = self.fabric.coupled_set_of_link(link)?;
        let set_idx = set_id.0 as usize;
        let (handle, batch_id) =
            self.sets[set_idx].claim_direct_blocking(link, bytes, direct, Instant::now())?;
        self.batch_to_set
            .lock()
            .expect("batch_to_set")
            .insert(batch_id, set_idx);
        Some((handle, batch_id))
    }

    /// Register a wakeup hook invoked strictly outside locks whenever any link
    /// in `link`'s `CoupledSet` transitions `queued_ops_by_class[Blocking]`
    /// from `> 0` to `0`. Dropping the returned [`BlockingClearedHookGuard`]
    /// deregisters the hook.
    #[must_use]
    pub fn register_blocking_cleared_hook(
        &self,
        link: LinkId,
        hook: BlockingClearedHook,
    ) -> BlockingClearedHookGuard {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].register_blocking_cleared_hook(link, hook)
        } else {
            BlockingClearedHookGuard {
                set: None,
                hook_id: 0,
            }
        }
    }

    /// O(1), zero-allocation direct link accounting for self-scheduled storage
    /// workers (`seismic_io::scheduler::Queue`).
    pub fn direct_enqueue(&self, link: LinkId, class: PriorityClass, count: u32, bytes: u64) {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].direct_enqueue(link, class, count, bytes);
        }
    }

    pub fn direct_reprice(
        &self,
        link: LinkId,
        old_class: PriorityClass,
        new_class: PriorityClass,
        count: u32,
        bytes: u64,
    ) {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].direct_reprice(link, old_class, new_class, count, bytes);
        }
    }

    pub fn direct_cancel(&self, link: LinkId, class: PriorityClass, count: u32, bytes: u64) {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].direct_cancel(link, class, count, bytes);
        }
    }

    pub fn direct_begin(&self, link: LinkId, class: PriorityClass, bytes: u64) {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].direct_begin(link, class, bytes, Instant::now());
        }
    }

    pub fn direct_complete(&self, link: LinkId, class: PriorityClass, bytes: u64, direct: bool) {
        if let Some(set_id) = self.fabric.coupled_set_of_link(link) {
            self.sets[set_id.0 as usize].direct_complete(link, class, bytes, direct, Instant::now());
        }
    }

    /// Whether any *other* link sharing a **saturated** coupled `SharedBusId`
    /// with `link` has a `PriorityClass::Blocking` op queued.
    pub fn has_coupled_blocking_waiting(&self, link: LinkId) -> bool {
        let Some(set_id) = self.fabric.coupled_set_of_link(link) else {
            return false;
        };
        self.sets[set_id.0 as usize].has_other_coupled_blocking_waiting(link)
    }

    /// `Submitting -> Issued`.
    pub fn confirm_issued(&self, batch_id: BatchId, fence: Option<TimelinePoint>) {
        let set_idx = *self
            .batch_to_set
            .lock()
            .expect("batch_to_set")
            .get(&batch_id)
            .expect("unknown batch_id");
        self.sets[set_idx].confirm_issued(batch_id, fence);
    }

    /// `Submitting -> Queued` on transient backpressure.
    pub fn rollback_transient(&self, batch_id: BatchId) {
        self.rollback_transient_at(batch_id, Instant::now());
    }

    pub fn rollback_transient_at(&self, batch_id: BatchId, now: Instant) {
        let set_idx = self
            .batch_to_set
            .lock()
            .expect("batch_to_set")
            .remove(&batch_id)
            .expect("unknown batch_id");
        self.sets[set_idx].rollback_transient(batch_id, now);
    }

    /// `Submitting -> Retired` on definite pre-issue refusal.
    pub fn refuse_before_issue(&self, batch_id: BatchId, reason: impl Into<String>) {
        self.refuse_before_issue_at(batch_id, reason.into(), Instant::now());
    }

    pub fn refuse_before_issue_at(&self, batch_id: BatchId, reason: String, now: Instant) {
        let set_idx = self
            .batch_to_set
            .lock()
            .expect("batch_to_set")
            .remove(&batch_id)
            .expect("unknown batch_id");
        self.sets[set_idx].refuse_before_issue(batch_id, reason, now);
    }

    /// `Submitting | Issued -> Quarantined` on ambiguous/partial driver failure.
    pub fn quarantine_batch(&self, batch_id: BatchId, reason: impl Into<String>) {
        self.quarantine_batch_at(batch_id, reason.into(), Instant::now());
    }

    pub fn quarantine_batch_at(&self, batch_id: BatchId, reason: String, now: Instant) {
        let set_idx = self
            .batch_to_set
            .lock()
            .expect("batch_to_set")
            .remove(&batch_id)
            .expect("unknown batch_id");
        self.sets[set_idx].quarantine_batch(batch_id, reason, now);
    }

    /// `Issued -> Retired` on hardware/OS completion.
    pub fn retire_issued(&self, batch_id: BatchId) {
        self.retire_issued_at(batch_id, Instant::now());
    }

    pub fn retire_issued_at(&self, batch_id: BatchId, now: Instant) {
        let set_idx = self
            .batch_to_set
            .lock()
            .expect("batch_to_set")
            .remove(&batch_id)
            .expect("unknown batch_id");
        self.sets[set_idx].retire_issued(batch_id, now);
    }

    /// Host-scoped non-IO bus reservation (debited on creation, credited on `Drop`).
    pub fn reserve_bus_scoped(&self, bus: SharedBusId, rate_bps: f64) -> Option<BusReservation> {
        let set_id = self.fabric.coupled_set_of_bus(bus)?;
        Some(self.sets[set_id.0 as usize].reserve_bus_scoped(bus, rate_bps))
    }

    /// GPU-fence-bound non-IO bus reservation (credited ONLY when `point` is reached).
    pub fn reserve_bus_until_fence(
        &self,
        bus: SharedBusId,
        rate_bps: f64,
        point: TimelinePoint,
        observer: Arc<dyn TimelineObserver>,
    ) -> Option<FenceReservationId> {
        let set_id = self.fabric.coupled_set_of_bus(bus)?;
        Some(self.sets[set_id.0 as usize].reserve_bus_until_fence(bus, rate_bps, point, observer))
    }

    /// Explicitly poll and reclaim any reached GPU-fence-bound bus reservations.
    pub fn poll_fence_reservations(&self) {
        for set in &self.sets {
            set.poll_fence_reservations();
        }
    }

    /// Release all quarantined backing leases across all coupled sets on device teardown.
    pub fn teardown_quarantined(&self) -> usize {
        self.sets.iter().map(|s| s.teardown_quarantined()).sum()
    }

    pub fn quarantined_lease_count(&self) -> usize {
        self.sets.iter().map(|s| s.quarantined_lease_count()).sum()
    }

    /// Exact ledger snapshot for `link`.
    pub fn link_ledger(&self, link: LinkId) -> Option<LinkLedger> {
        let set_id = self.fabric.coupled_set_of_link(link)?;
        self.sets[set_id.0 as usize].link_ledger(link)
    }

    /// Exact ledger snapshot for `bus`.
    pub fn bus_ledger(&self, bus: SharedBusId) -> Option<BusLedger> {
        let set_id = self.fabric.coupled_set_of_bus(bus)?;
        self.sets[set_id.0 as usize].bus_ledger(bus)
    }
}

/// Process-wide unified [`IoPlane`] shared across Storage (`NvmeDirect`) and
/// Device Copy Engine (`CopyEngineH2D`/`D2H`) transfers.
pub fn global_plane() -> &'static Arc<IoPlane> {
    static PLANE: OnceLock<Arc<IoPlane>> = OnceLock::new();
    PLANE.get_or_init(IoPlane::new_default)
}

#[cfg(test)]
thread_local! {
    static SPEC_OP_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod work_bound_tests {
    use super::*;
    use crate::fabric::DeviceId;
    use crate::schedule::GlobalUrgency;

    fn submit_speculative(plane: &Arc<IoPlane>, seq: u64, bytes: u64) -> TransferHandle {
        let key = ScheduleKey::prefetch(GlobalUrgency(seq), 0, 0, 0.5, seq);
        let dst_lease = BufferLease::unmanaged(
            DomainId::DeviceVram(DeviceId(0)),
            std::ptr::NonNull::dangling(),
            0,
            0,
            bytes as usize,
            1,
            0,
        );
        plane
            .submit(TransferRequest {
                link: LINK_COPY_H2D,
                key,
                bytes,
                content_key: None,
                placement: PlacementRequirement::FixedDestination,
                compatibility: BatchCompatibility::Isolated,
                direct: true,
                src_lease: None,
                dst_lease,
            })
            .expect("submit speculative")
    }

    #[test]
    fn work_bound_residual_speculative_service_time_independent_of_arena_size() {
        let measure = |unrelated_queued: u64| -> usize {
            let plane = IoPlane::new_default();
            let now = Instant::now();
            let _active = submit_speculative(&plane, 1, 4096);
            let batch = plane
                .claim_next_batch(LINK_COPY_H2D, 1024 * 1024, 1.0)
                .expect("claim active speculative op");
            let mut _extra = Vec::with_capacity(unrelated_queued as usize);
            for i in 0..unrelated_queued {
                _extra.push(submit_speculative(&plane, 10 + i, 4096));
            }
            let set_id = plane.fabric.coupled_set_of_link(LINK_COPY_H2D).expect("set");
            let set = &plane.sets[set_id.0 as usize];
            SPEC_OP_VISITS.with(|c| c.set(0));
            {
                let state = set.state.lock().expect("state");
                assert_eq!(state.in_flight_speculative.len(), 1);
                let residual = CoupledSetController::residual_speculative_service_time_s(&state, now);
                assert!(residual >= 0.0);
            }
            let visits = SPEC_OP_VISITS.with(|c| c.get());
            // Verify retire_issued untracks the speculative op cleanly.
            plane.confirm_issued(batch.batch_id, None);
            plane.retire_issued(batch.batch_id);
            {
                let state = set.state.lock().expect("state");
                assert_eq!(state.in_flight_speculative.len(), 0);
            }
            visits
        };

        // Must visit exactly 1 active in-flight speculative op whether 0 or 256
        // unrelated ops sit in the arena.
        assert_eq!(measure(0), 1);
        assert_eq!(measure(256), 1);
    }
}
