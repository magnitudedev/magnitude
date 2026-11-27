//! Unified device transfer plane (`Copies`) built on `seismic_io::IoPlane`:
//!
//! Owns **every** `CopyEngineH2D` and `CopyEngineD2H` submission on the
//! device's transfer queue (`seismic::Transfer`) — both background/prefill
//! batches (`LinkMover`, `Scratch`) AND synchronous `Blocking` decode demand
//! misses (`ExpertTransport::land_decode_misses`) — through ONE `IoPlane`
//! `CoupledSet` admission owner:
//!
//! - **Zero competing authorities:** `Copies`'s old private `waiting` queue
//!   and hardcoded `MAX_RUNNING = 2` are deleted. `IoPlane` owns priority
//!   ordering (`ScheduleKey`), `reprice`, `cancel`, `Blocking` preemption
//!   across coupled `HostDdr` + `PcieRoot`, knee cap, and residual speculative
//!   service-time bounding (`τ_hol`).
//! - **Four-state issue boundary (`Queued -> Submitting -> Issued -> Retired | Quarantined`):**
//!   All `IoPlane` and `Copies` locks are released before entering
//!   `Transfer::copy` / `Transfer::copy_synced`. Definite pre-issue refusals
//!   roll back credits immediately; ambiguous/partial failures transition to
//!   `Quarantined` and retain backing tensors until explicit teardown.
//! - **Shadow quote telemetry:** `Copies::shadow_quote_h2d()` reads
//!   `IoPlane::quote()` lock-free (~35 ns) for read-only contention telemetry.

use super::events::Events;
use super::transport::{ExpertTransport, ScratchGenId, TransferId};
use seismic::{Tensor, Transfer, TransferError, TransferItem, TransferRun, TransferSync};
use seismic_io::{
    global_plane, BatchCompatibility, BatchId, BufferLease, GlobalUrgency, IoPlane, LinkId,
    OwnerTag, PhysicalOpId, PlacementRequirement, PriorityClass, ScheduleKey, TransferHandle,
    TransferQuote, TransferRequest, BUS_HOST_DDR, BUS_PCIE_ROOT, DOMAIN_DEVICE_VRAM,
    DOMAIN_HOST_RAM, LINK_COPY_D2H, LINK_COPY_H2D, LINK_STORAGE_NVME,
};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// `IoPlane` Link ID for Host RAM -> Device VRAM DMA (`PrefillFill`, `Stage`,
/// `Promote`, and synchronous `Blocking` decode demand).
pub(in crate::experts) const LINK_H2D: LinkId = LINK_COPY_H2D;

/// `IoPlane` Link ID for Device VRAM -> Host RAM write-back DMA (`WriteBack`).
pub(in crate::experts) const LINK_D2H: LinkId = LINK_COPY_D2H;

const HOST_DOMAIN: seismic_io::DomainId = DOMAIN_HOST_RAM;
const VRAM_DOMAIN: seismic_io::DomainId = DOMAIN_DEVICE_VRAM;

/// The longest one wait of the watch on the queue lasts before it looks
/// whether the load stopped: a bound, not a latency, as the batch's
/// completion ends the wait.
const BOUND: Duration = Duration::from_millis(50);

/// Wait timeout used by `Copies::watch` when any running batch was issued with
/// `copy_synced` (which intentionally skips `cuLaunchHostFunc` wakeups so
/// compute waits via `cuStreamWaitEvent`). Keeps `running` draining promptly.
const SYNCED_BOUND: Duration = Duration::from_millis(1);

/// Maximum batch bytes drained per issuer turn on `IoPlane::claim_next_batch`.
const MAX_ISSUER_BATCH_BYTES: u64 = 64 * 1024 * 1024;

/// Maximum batch service-time budget per issuer turn (seconds).
const MAX_ISSUER_BATCH_SERVICE_S: f64 = 1.0;

/// One run of a batch (`seismic::TransferRun`): `bytes` from `source` at
/// `source_offset` to `destination` at `destination_offset`, held by the
/// queue until its batch completes.
pub(in crate::experts) struct Run {
    pub source: Tensor,
    pub source_offset: u64,
    pub destination: Tensor,
    pub destination_offset: u64,
    pub bytes: u64,
}

/// One 32-bit signal written on the transfer stream (`TransferItem::Signal`)
/// once every run before it in the same batch has landed.
pub(in crate::experts) struct Signal {
    pub word: Tensor,
    pub index: u64,
    pub value: i32,
}

/// One step of a [`Copies`] batch, in order: a DMA copy run or a stream signal.
pub(in crate::experts) enum BatchItem {
    Run(Run),
    Signal(Signal),
}

fn batch_bytes(items: &[BatchItem]) -> u64 {
    let total: u64 = items
        .iter()
        .map(|item| match item {
            BatchItem::Run(run) => run.bytes,
            BatchItem::Signal(_) => 4,
        })
        .sum();
    total.max(1)
}

/// Dispatch priority for queued DMA batches on [`Copies`], mapped directly onto
/// `seismic_io::ScheduleKey` (`Blocking` / `Prefetch` / `Background`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::experts) enum TransferPriority {
    PrefillFill,
    WriteBack,
    Promote,
    Stage { step: u64, layer: usize, p: f64 },
}

impl TransferPriority {
    /// Target `IoPlane` link (`LINK_D2H` for `WriteBack`, `LINK_H2D` otherwise).
    fn link(self) -> LinkId {
        match self {
            Self::WriteBack => LINK_D2H,
            Self::PrefillFill | Self::Promote | Self::Stage { .. } => LINK_H2D,
        }
    }

    /// Convert to a total-ordered, NaN-free `seismic_io::ScheduleKey`.
    fn to_schedule_key(self, seq: u64) -> ScheduleKey {
        match self {
            Self::PrefillFill => ScheduleKey::blocking(GlobalUrgency(0), 0, 0, seq),
            // Background VRAM->Host eviction (`LINK_D2H`) is high urgency on D2H
            // without tripping cross-link `Blocking` preemption on `LINK_STORAGE_NVME`.
            Self::WriteBack => ScheduleKey::streaming(GlobalUrgency(0), 0, 0, seq),
            Self::Stage { step, layer, p } => ScheduleKey::prefetch(
                GlobalUrgency(0),
                step,
                u16::try_from(layer).unwrap_or(u16::MAX),
                p,
                seq,
            ),
            Self::Promote => ScheduleKey::background(GlobalUrgency(0), seq),
        }
    }
}

/// Metadata attached to EVERY prefill scratch-fill batch (`PrefillFill`).
#[derive(Clone, Copy)]
pub(in crate::experts) struct PrefillBatch {
    pub layer: usize,
    pub sequence: u64,
    pub window: u64,
    pub generation: ScratchGenId,
    pub publish_sync: bool,
}

/// RAII guard for one synchronous `Blocking` decode demand DMA batch issued
/// through [`Copies::issue_decode_demand`] on the shared [`IoPlane`].
///
/// - Calling [`Self::retire`] transitions `Issued -> Retired` on `IoPlane`,
///   crediting exact wire/bus bytes once `DecodeBack` retires (or after DMA
///   wait on abort).
/// - Calling [`Self::quarantine`] (or dropping without explicit retirement)
///   transitions `Issued -> Quarantined` on `IoPlane`.
pub(in crate::experts) struct DemandFlightGuard {
    plane: Arc<IoPlane>,
    batch_id: BatchId,
    _handle: TransferHandle,
    settled: bool,
}

impl DemandFlightGuard {
    pub(in crate::experts) fn retire(mut self) {
        if !self.settled {
            self.settled = true;
            self.plane.retire_issued(self.batch_id);
        }
    }

    pub(in crate::experts) fn quarantine(mut self, reason: &str) {
        if !self.settled {
            self.settled = true;
            self.plane.quarantine_batch(self.batch_id, reason);
        }
    }
}

impl Drop for DemandFlightGuard {
    fn drop(&mut self) {
        if !self.settled {
            self.settled = true;
            self.plane
                .quarantine_batch(self.batch_id, "DemandFlightGuard dropped without explicit retirement");
        }
    }
}

struct QueuedPayload {
    ticket: u64,
    items: Vec<BatchItem>,
    stage_transfer: Option<TransferId>,
    prefill: Option<PrefillBatch>,
    handle: TransferHandle,
}

/// In-flight batch metadata: ticket, completion value, `IoPlane` batch id,
/// subscriber handle, held items, optional tracked `ExpertTransport` transfer
/// id, and whether it was issued synced.
struct RunningBatch {
    ticket: u64,
    value: u64,
    batch_id: BatchId,
    _handle: TransferHandle,
    items: Vec<BatchItem>,
    stage_transfer: Option<TransferId>,
    synced: bool,
}

/// The device's unified transfer plane, shared by `LinkMover`, `Scratch`, and
/// `ExpertTransport::land_decode_misses`.
#[derive(Clone)]
pub(in crate::experts) struct Copies {
    inner: Arc<Inner>,
}

/// Process-unique owner tag generator so multiple `Copies` instances sharing
/// `seismic_io::global_plane()` never claim each other's ops.
static NEXT_COPIES_OWNER: AtomicU64 = AtomicU64::new(1);

struct Inner {
    owner: OwnerTag,
    plane: Arc<IoPlane>,
    _blocking_cleared_guard: seismic_io::BlockingClearedHookGuard,
    transfer: Transfer,
    events: Arc<Events>,
    transport: Option<Arc<ExpertTransport>>,
    next_ticket: AtomicU64,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Highest ticket posted so far (for diagnostics/tests).
    posted: u64,
    /// Queued payloads keyed by `PhysicalOpId` (scheduling/order/preemption is
    /// owned exclusively by `IoPlane`).
    queued_by_op: HashMap<PhysicalOpId, QueuedPayload>,
    /// Index `ticket -> PhysicalOpId` for `reprice` / `cancel` / `is_begun`.
    ticket_to_op: HashMap<u64, PhysicalOpId>,
    /// Currently submitting batch `(ticket, started_at)` for diagnostics.
    issuing: Option<(u64, std::time::Instant)>,
    /// Highest ticket issued so far (for diagnostics).
    issued: u64,
    /// Issued batches awaiting hardware completion in stream order.
    running: VecDeque<RunningBatch>,
    /// Batches refused before issue, by ticket.
    refused: BTreeMap<u64, String>,
    /// Poisoned prefill generations (refused partial/final batch).
    poisoned_generations: HashMap<ScratchGenId, String>,
    /// Cancelled batches' items deferred to drop on the issuer thread (outside
    /// `residency` and `NoBlockingAllocationGuard`).
    cancelled: Vec<Vec<BatchItem>>,
    /// Backing items of `Quarantined` batches retained permanently until teardown.
    quarantined_items: Vec<Vec<BatchItem>>,
    /// Terminal queue fault (after ambiguous/partial failure or wait fault).
    fault: Option<String>,
}

impl State {
    fn flight(&self) -> usize {
        self.queued_by_op.len() + usize::from(self.issuing.is_some()) + self.running.len()
    }

    fn is_waiting(&self, ticket: u64) -> bool {
        self.ticket_to_op.contains_key(&ticket)
    }

    fn is_in_flight(&self, ticket: u64) -> bool {
        self.is_waiting(ticket)
            || self.issuing.is_some_and(|(t, _)| t == ticket)
            || self.running.iter().any(|b| b.ticket == ticket)
    }
}

impl Copies {
    pub fn new(transfer: Transfer, events: Arc<Events>) -> Self {
        Self::with_plane(IoPlane::new_default(), transfer, events, None)
    }

    /// Production constructor: shares the process-wide `seismic_io::global_plane()`
    /// so Storage (`NvmeDirect`) and CopyEngine (`CopyEngineH2D`/`D2H`) share
    /// the exact same `CoupledSet` (`HostDdr` + `PcieRoot`) admission owner.
    pub fn with_transport(transfer: Transfer, events: Arc<Events>, transport: Arc<ExpertTransport>) -> Self {
        Self::with_plane(Arc::clone(global_plane()), transfer, events, Some(transport))
    }

    pub fn with_plane(
        plane: Arc<IoPlane>,
        transfer: Transfer,
        events: Arc<Events>,
        transport: Option<Arc<ExpertTransport>>,
    ) -> Self {
        let owner = OwnerTag(NEXT_COPIES_OWNER.fetch_add(1, Ordering::Relaxed));
        let weak_events = Arc::downgrade(&events);
        let blocking_cleared_guard = plane.register_blocking_cleared_hook(
            LINK_H2D,
            Arc::new(move || {
                if let Some(ev) = weak_events.upgrade() {
                    ev.notify();
                }
            }),
        );
        Self {
            inner: Arc::new(Inner {
                owner,
                plane,
                _blocking_cleared_guard: blocking_cleared_guard,
                transfer,
                events,
                transport,
                next_ticket: AtomicU64::new(1),
                state: Mutex::default(),
            }),
        }
    }

    /// Underlying `seismic_io::IoPlane` single admission & contention owner.
    pub fn plane(&self) -> &Arc<IoPlane> {
        &self.inner.plane
    }

    /// Lock-free read-only shadow quote for `bytes` on `CopyEngineH2D` (~35 ns).
    pub fn shadow_quote_h2d(&self, bytes: u64) -> TransferQuote {
        self.inner
            .plane
            .quote(LINK_H2D, bytes)
            .expect("LINK_H2D exists in copy fabric")
    }

    /// Lock-free read-only shadow quote for `bytes` on `NvmeDirect` (`LINK_STORAGE_NVME`).
    pub fn shadow_quote_storage(&self, bytes: u64) -> TransferQuote {
        self.inner
            .plane
            .quote(LINK_STORAGE_NVME, bytes)
            .expect("LINK_STORAGE_NVME exists in unified fabric")
    }

    /// Underlying hardware transfer queue (for waiting on a submitted demand
    /// timeline value during abort).
    pub fn transfer(&self) -> &Transfer {
        &self.inner.transfer
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.state.lock().expect("the transfer queue's batches")
    }

    /// Single-path synchronous `PriorityClass::Blocking` decode demand issue
    /// through the shared [`IoPlane`] `CopyEngineH2D` `CoupledSet`:
    ///
    /// 1. Atomically claims `Submitting` credits on `IoPlane` (debits
    ///    `CopyEngineH2D` + `HostDdr` + `PcieRoot`, preempting speculative work).
    /// 2. Releases all locks before invoking `Transfer::copy_synced(items)`.
    /// 3. Transitions `Submitting -> Issued` (returning [`DemandFlightGuard`]),
    ///    `Submitting -> Retired` on `RefusedBeforeIssue`, or
    ///    `Submitting -> Quarantined` on ambiguous/partial failure.
    pub fn issue_decode_demand(
        &self,
        items: &[TransferItem<'_>],
        bytes: u64,
    ) -> Result<(u64, TransferSync, DemandFlightGuard), TransferError> {
        let (handle, batch_id) = self
            .inner
            .plane
            .claim_direct_blocking(LINK_H2D, bytes.max(1), true)
            .expect("LINK_H2D exists in copy fabric");

        // Zero locks held across the hardware driver call.
        let result = self.inner.transfer.copy_synced(items);
        match result {
            Ok((value, sync)) => {
                self.inner
                    .plane
                    .confirm_issued(batch_id, Some(sync.timeline_point()));
                let quote = self.shadow_quote_h2d(bytes);
                tracing::trace!(
                    bytes,
                    est_us = quote.estimated_s * 1e6,
                    eff_gbps = quote.effective_bps / 1e9,
                    bus_util = quote.bus_utilization,
                    "seismic-io v2 shadow quote (decode demand H2D)"
                );
                let guard = DemandFlightGuard {
                    plane: Arc::clone(&self.inner.plane),
                    batch_id,
                    _handle: handle,
                    settled: false,
                };
                Ok((value, sync, guard))
            }
            Err(error) => {
                if error.refused_before_issue() {
                    self.inner
                        .plane
                        .refuse_before_issue(batch_id, error.to_string());
                } else {
                    self.inner
                        .plane
                        .quarantine_batch(batch_id, error.to_string());
                }
                Err(error)
            }
        }
    }

    fn post(
        &self,
        priority: TransferPriority,
        items: Vec<BatchItem>,
        stage_transfer: Option<TransferId>,
        prefill: Option<PrefillBatch>,
    ) -> u64 {
        let ticket = self.inner.next_ticket.fetch_add(1, Ordering::Relaxed);
        let bytes = batch_bytes(&items);
        let link = priority.link();
        let dst_domain = if link == LINK_D2H { HOST_DOMAIN } else { VRAM_DOMAIN };
        let key = priority.to_schedule_key(ticket);

        // Hold `self.state()` across `plane.submit_owned` + `queued_by_op.insert`
        // (strict `Copies::state -> IoPlane` lock order) so `claim_next_from_plane`
        // can NEVER observe an op whose payload has not yet been inserted.
        {
            let mut state = self.state();
            let handle = self
                .inner
                .plane
                .submit_owned(
                    TransferRequest {
                        link,
                        key,
                        bytes,
                        content_key: None,
                        placement: PlacementRequirement::FixedDestination,
                        compatibility: BatchCompatibility::Isolated,
                        src_lease: None,
                        dst_lease: BufferLease::unmanaged(
                            dst_domain,
                            NonNull::dangling(),
                            0,
                            0,
                            bytes as usize,
                            1,
                            0,
                        ),
                        direct: true,
                    },
                    self.inner.owner,
                )
                .expect("LINK_H2D/LINK_D2H submit succeeds");
            let op_id = handle.op_id;
            state.posted = state.posted.max(ticket);
            state.ticket_to_op.insert(ticket, op_id);
            state.queued_by_op.insert(
                op_id,
                QueuedPayload {
                    ticket,
                    items,
                    stage_transfer,
                    prefill,
                    handle,
                },
            );
        }
        self.inner.events.notify();
        ticket
    }

    /// Post a prefill scratch-fill DMA batch at `PriorityClass::Blocking`.
    pub fn post_prefill(&self, items: Vec<BatchItem>, prefill: PrefillBatch) -> Result<u64, String> {
        {
            let state = self.state();
            if let Some(fault) = &state.fault {
                return Err(fault.clone());
            }
            if let Some(reason) = state.poisoned_generations.get(&prefill.generation) {
                return Err(reason.clone());
            }
        }
        Ok(self.post(TransferPriority::PrefillFill, items, None, Some(prefill)))
    }

    pub fn post_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<Run>, T),
    ) -> Result<Option<(T, u64)>, String> {
        self.post_items_within(depth, priority, || {
            let (runs, formed) = form();
            (runs.into_iter().map(BatchItem::Run).collect(), formed)
        })
    }

    pub fn post_items_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<BatchItem>, T),
    ) -> Result<Option<(T, u64)>, String> {
        self.post_tracked_within(depth, priority, || {
            let (items, formed) = form();
            (items, formed, None)
        })
    }

    pub fn post_tracked_within<T>(
        &self,
        depth: usize,
        priority: TransferPriority,
        form: impl FnOnce() -> (Vec<BatchItem>, T, Option<TransferId>),
    ) -> Result<Option<(T, u64)>, String> {
        assert!(depth > 0, "a transfer queue keeps a batch in flight");
        {
            let state = self.state();
            if let Some(fault) = &state.fault {
                return Err(fault.clone());
            }
            if state.flight() >= depth {
                return Ok(None);
            }
        }
        let (items, formed, stage_transfer) = form();
        Ok(Some((formed, self.post(priority, items, stage_transfer, None))))
    }

    /// Reprice an unbegun queued batch via `IoPlane`'s `TransferHandle::reprice`.
    pub fn reprice(&self, ticket: u64, priority: TransferPriority) -> bool {
        let state = self.state();
        let Some(&op_id) = state.ticket_to_op.get(&ticket) else {
            return false;
        };
        let Some(payload) = state.queued_by_op.get(&op_id) else {
            return false;
        };
        payload.handle.reprice(priority.to_schedule_key(ticket));
        true
    }

    /// Cancel an unbegun queued batch via `IoPlane` (zero bus debit; defers
    /// dropping tensor items to the issuer thread outside residency locks).
    pub fn cancel(&self, ticket: u64) -> bool {
        let stage_transfer = {
            let mut state = self.state();
            let Some(op_id) = state.ticket_to_op.remove(&ticket) else {
                return false;
            };
            let Some(payload) = state.queued_by_op.remove(&op_id) else {
                return false;
            };
            // Detach from `IoPlane` while still holding `self.state()` (strict
            // `Copies::state -> IoPlane` lock order) so `claim_next_from_plane`
            // can never claim a cancelled op whose payload was already removed.
            // Backing tensor items are deferred to `state.cancelled` to drop
            // outside residency locks.
            payload.handle.detach();
            state.cancelled.push(payload.items);
            payload.stage_transfer
        };
        if let (Some(transport), Some(id)) = (&self.inner.transport, stage_transfer) {
            transport.cancel_unbegun_hint(id);
        }
        self.inner.events.notify();
        true
    }

    /// Whether `ticket` has already begun issuing (not waiting in `IoPlane`).
    pub fn is_begun(&self, ticket: u64) -> bool {
        !self.state().is_waiting(ticket)
    }

    /// Total batches posted and not yet complete.
    pub fn flight(&self) -> usize {
        self.state().flight()
    }

    /// Whether no batch is posted and not complete.
    pub fn idle(&self) -> bool {
        self.flight() == 0
    }

    #[cfg(test)]
    pub fn posted(&self) -> u64 {
        self.state().posted
    }

    /// Whether `ticket` is complete, or why it failed.
    pub fn completed(&self, ticket: u64) -> Result<bool, String> {
        let state = self.state();
        if let Some(reason) = state.refused.get(&ticket) {
            return Err(reason.clone());
        }
        if state.is_in_flight(ticket) {
            return match &state.fault {
                Some(fault) => Err(fault.clone()),
                None => Ok(false),
            };
        }
        Ok(true)
    }

    /// Diagnostic summary including live `IoPlane` coupled-bus ledger & shadow quotes.
    pub fn describe(&self) -> String {
        let state = self.state();
        let h2d = self.inner.plane.link_ledger(LINK_H2D).unwrap_or_default();
        let d2h = self.inner.plane.link_ledger(LINK_D2H).unwrap_or_default();
        let nvme = self.inner.plane.link_ledger(LINK_STORAGE_NVME).unwrap_or_default();
        let ddr = self.inner.plane.bus_ledger(BUS_HOST_DDR).unwrap_or_default();
        let pcie = self.inner.plane.bus_ledger(BUS_PCIE_ROOT).unwrap_or_default();
        let q_h2d = self.shadow_quote_h2d(2 * 1024 * 1024);
        let q_nvme = self.shadow_quote_storage(2 * 1024 * 1024);
        format!(
            "posted through {}, {} waiting, issuing {:?}, issued through {}, running {:?}, \
             io_plane(h2d={}B, d2h={}B, nvme={}B, ddr={}B, pcie={}B, quarantined={}, \
             quote_h2d={:.2}GB/s, quote_nvme={:.2}GB/s), refused {:?}, fault {:?}",
            state.posted,
            state.queued_by_op.len(),
            state.issuing.map(|(ticket, since)| (ticket, since.elapsed())),
            state.issued,
            state
                .running
                .iter()
                .map(|b| (b.ticket, b.value))
                .collect::<Vec<_>>(),
            h2d.total_in_flight_bytes(),
            d2h.total_in_flight_bytes(),
            nvme.total_in_flight_bytes(),
            ddr.in_flight_weighted_bytes,
            pcie.in_flight_weighted_bytes,
            h2d.quarantined_ops + d2h.quarantined_ops + nvme.quarantined_ops,
            q_h2d.effective_bps / 1e9,
            q_nvme.effective_bps / 1e9,
            state.refused,
            state.fault,
        )
    }

    /// Release quarantined backing items & `IoPlane` leases on device teardown.
    pub fn teardown_quarantined(&self) -> usize {
        let items = {
            let mut state = self.state();
            std::mem::take(&mut state.quarantined_items)
        };
        drop(items);
        self.inner.plane.teardown_quarantined()
    }

    /// Start the issuer and completion-watch threads until `stopped` returns true.
    pub fn start(&self, stopped: impl Fn() -> bool + Send + Sync + 'static) -> Result<Vec<JoinHandle<()>>, String> {
        let stopped = Arc::new(stopped);
        let issuer = {
            let (copies, stopped) = (self.clone(), stopped.clone());
            seismic::threads::spawn("expert-copies-issue", move || {
                seismic_io::waited_on();
                copies.issue(&*stopped)
            })
            .map_err(|error| format!("the transfer queue's issuer: {error}"))?
        };
        let copies = self.clone();
        let watch = seismic::threads::spawn("expert-copies", move || {
            seismic_io::waited_on();
            copies.watch(&*stopped)
        })
        .map_err(|error| format!("the transfer queue's watch: {error}"))?;
        Ok(vec![issuer, watch])
    }

    fn drain_stopped(&self) {
        let (remaining, cancelled) = {
            let mut state = self.state();
            state.ticket_to_op.clear();
            let remaining: Vec<(Vec<BatchItem>, Option<TransferId>, Option<PrefillBatch>)> = state
                .queued_by_op
                .drain()
                .map(|(_, payload)| {
                    payload.handle.detach();
                    (payload.items, payload.stage_transfer, payload.prefill)
                })
                .collect();
            let cancelled = std::mem::take(&mut state.cancelled);
            (remaining, cancelled)
        };
        drop(cancelled);
        for (items, stage_transfer, prefill) in remaining {
            if let Some(transport) = &self.inner.transport {
                if let Some(prefill) = prefill {
                    transport.poison_prefill_generation(
                        prefill.generation,
                        prefill.sequence,
                        "copy issuer stopped".to_string(),
                    );
                }
                if let Some(id) = stage_transfer {
                    transport.fault_transfer(id, "copy issuer stopped".to_string());
                }
            }
            drop(items);
        }
        if let Some(transport) = &self.inner.transport {
            transport.drain_deferred_syncs();
        }
    }

    /// Claim the next admissible batch from `IoPlane`: checks `Blocking`
    /// `LINK_H2D` (`PrefillFill`), then `LINK_D2H` (`WriteBack`), then
    /// remaining `LINK_H2D` (`Stage` / `Promote`). All preemption, knee-cap,
    /// and residual speculative service-time bounds are enforced by `IoPlane`.
    fn claim_next_from_plane(&self) -> Option<(BatchId, QueuedPayload)> {
        // Hold `self.state()` across `plane.claim_next_batch_for_owner` +
        // `queued_by_op.remove` (strict `Copies::state -> IoPlane` lock order),
        // eliminating both publish-before-insert and cancel-before-detach races
        // AND cross-instance claims on `global_plane()` by construction.
        let mut state = self.state();
        if state.queued_by_op.is_empty() {
            return None;
        }
        let h2d_has_blocking = self
            .inner
            .plane
            .link_ledger(LINK_H2D)
            .is_some_and(|l| l.queued_ops_by_class[PriorityClass::Blocking.index()] > 0);
        let order = if h2d_has_blocking {
            [LINK_H2D, LINK_D2H]
        } else {
            [LINK_D2H, LINK_H2D]
        };
        for link in order {
            if let Some(claimed) = self.inner.plane.claim_next_batch_for_owner(
                link,
                self.inner.owner,
                MAX_ISSUER_BATCH_BYTES,
                MAX_ISSUER_BATCH_SERVICE_S,
            ) {
                let op_id = claimed.ops[0].op_id;
                let payload = state
                    .queued_by_op
                    .remove(&op_id)
                    .expect("claimed op has payload in Copies");
                state.ticket_to_op.remove(&payload.ticket);
                state.issuing = Some((payload.ticket, std::time::Instant::now()));
                return Some((claimed.batch_id, payload));
            }
        }
        None
    }

    /// The single issuer loop driven by `IoPlane`.
    fn issue(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                self.drain_stopped();
                return;
            }
            let (cancelled, has_queued) = {
                let mut state = self.state();
                (std::mem::take(&mut state.cancelled), !state.queued_by_op.is_empty())
            };
            drop(cancelled);

            let Some((batch_id, payload)) = self.claim_next_from_plane() else {
                if has_queued {
                    // Queued speculative work yielded to a coupled link on
                    // `IoPlane` (e.g. Blocking NVMe or another `Copies` on
                    // `global_plane()`), which does not hold our `Events`.
                    events.wait_at_most(seen, SYNCED_BOUND);
                } else {
                    events.wait(seen);
                }
                continue;
            };
            let QueuedPayload {
                ticket,
                items: batch,
                stage_transfer,
                prefill,
                handle,
            } = payload;

            // If an earlier partial batch for this prefill generation was
            // refused, refuse before touching the hardware transfer stream.
            if let Some(prefill) = prefill {
                let poisoned = {
                    let state = self.state();
                    state.poisoned_generations.get(&prefill.generation).cloned()
                };
                if let Some(reason) = poisoned {
                    self.inner.plane.refuse_before_issue(batch_id, reason.clone());
                    if let Some(transport) = &self.inner.transport {
                        transport.poison_prefill_generation(
                            prefill.generation,
                            prefill.sequence,
                            reason.clone(),
                        );
                    }
                    let mut state = self.state();
                    state.issuing = None;
                    state.issued = state.issued.max(ticket);
                    state.refused.insert(ticket, reason);
                    drop(state);
                    drop(handle);
                    events.notify();
                    continue;
                }
            }

            let items = batch
                .iter()
                .map(|item| match item {
                    BatchItem::Run(run) => TransferItem::Run(TransferRun {
                        source: &run.source,
                        source_offset: run.source_offset,
                        destination: &run.destination,
                        destination_offset: run.destination_offset,
                        bytes: run.bytes,
                    }),
                    BatchItem::Signal(signal) => TransferItem::Signal {
                        word: &signal.word,
                        index: signal.index,
                        value: signal.value,
                    },
                })
                .collect::<Vec<_>>();
            let need_sync = stage_transfer.is_some() || prefill.is_some_and(|p| p.publish_sync);

            // ZERO locks held across the driver call.
            let issued = if need_sync {
                self.inner
                    .transfer
                    .copy_synced(&items)
                    .map(|(value, sync)| (value, Some(sync)))
            } else {
                self.inner.transfer.copy(&items).map(|value| (value, None))
            };
            drop(items);

            match issued {
                Ok((value, sync)) => {
                    let point = sync.as_ref().map(TransferSync::timeline_point);
                    self.inner.plane.confirm_issued(batch_id, point);

                    if let (Some(transport), Some(id), Some(sync)) =
                        (&self.inner.transport, stage_transfer, sync.as_ref())
                    {
                        transport.attach_sync(id, sync.clone());
                        transport.drain_deferred_syncs();
                    }
                    if let (Some(transport), Some(prefill), Some(sync)) =
                        (&self.inner.transport, prefill.filter(|p| p.publish_sync), sync)
                    {
                        transport.publish_prefill_sync(
                            prefill.layer,
                            prefill.sequence,
                            prefill.window,
                            prefill.generation,
                            Some(sync),
                        );
                    }
                    let mut state = self.state();
                    state.issuing = None;
                    state.issued = state.issued.max(ticket);
                    state.running.push_back(RunningBatch {
                        ticket,
                        value,
                        batch_id,
                        _handle: handle,
                        items: batch,
                        stage_transfer,
                        synced: need_sync,
                    });
                }
                Err(error) => {
                    let msg = error.to_string();
                    if error.refused_before_issue() {
                        self.inner.plane.refuse_before_issue(batch_id, msg.clone());
                    } else {
                        // Ambiguous/partial driver failure -> Quarantined!
                        // Backing items stay pinned until explicit teardown.
                        self.inner.plane.quarantine_batch(batch_id, msg.clone());
                    }
                    if let (Some(transport), Some(id)) = (&self.inner.transport, stage_transfer) {
                        transport.fault_transfer(id, msg.clone());
                        transport.drain_deferred_syncs();
                    }
                    if let Some(prefill) = prefill {
                        if let Some(transport) = &self.inner.transport {
                            transport.poison_prefill_generation(
                                prefill.generation,
                                prefill.sequence,
                                msg.clone(),
                            );
                        }
                    }
                    let mut state = self.state();
                    if !error.refused_before_issue() {
                        state.quarantined_items.push(batch);
                    }
                    if let Some(prefill) = prefill {
                        state
                            .poisoned_generations
                            .entry(prefill.generation)
                            .or_insert_with(|| msg.clone());
                    }
                    state.issuing = None;
                    state.issued = state.issued.max(ticket);
                    state.refused.insert(ticket, msg);
                    drop(state);
                    drop(handle);
                }
            }
            events.notify();
        }
    }

    /// Completion watch loop: retires `Issued -> Retired` (or `Quarantined` on
    /// fault) on `IoPlane` and releases backing tensors outside locks.
    fn watch(&self, stopped: &dyn Fn() -> bool) {
        let events = &self.inner.events;
        loop {
            let seen = events.seen();
            if stopped() {
                return;
            }
            let earliest = {
                let state = self.state();
                state
                    .fault
                    .is_none()
                    .then(|| state.running.front().map(|b| (b.value, b.synced)))
                    .flatten()
            };
            let Some((value, synced)) = earliest else {
                events.wait(seen);
                continue;
            };
            let timeout = if synced { SYNCED_BOUND } else { BOUND };
            match self.inner.transfer.wait(value, timeout) {
                Ok(false) => continue,
                Ok(true) => {
                    // Mark `Landed` in `ExpertTransport` BEFORE popping from
                    // `running` so `Copies::completed(batch) == Ok(true)`
                    // strictly implies `TransferState::Landed`.
                    let stage_transfer = self.state().running.front().and_then(|b| b.stage_transfer);
                    if let (Some(id), Some(transport)) = (stage_transfer, &self.inner.transport) {
                        transport.complete_dma(id);
                    }
                    let completed = self.state().running.pop_front();
                    if let Some(batch) = completed {
                        self.inner.plane.retire_issued(batch.batch_id);
                        drop(batch);
                    }
                    if let (Some(_), Some(transport)) = (stage_transfer, &self.inner.transport) {
                        transport.drain_deferred_syncs();
                    }
                }
                Err(error) => {
                    let msg = error.to_string();
                    let (faulted_ids, batch_ids): (Vec<TransferId>, Vec<BatchId>) = {
                        let mut state = self.state();
                        state.fault = Some(msg.clone());
                        let ids = state
                            .running
                            .iter()
                            .filter_map(|b| b.stage_transfer)
                            .collect();
                        let bids = state.running.iter().map(|b| b.batch_id).collect();
                        // Retain backing items of all in-flight batches in quarantine.
                        let drained: Vec<RunningBatch> = state.running.drain(..).collect();
                        for b in drained {
                            state.quarantined_items.push(b.items);
                        }
                        (ids, bids)
                    };
                    for bid in batch_ids {
                        self.inner.plane.quarantine_batch(bid, msg.clone());
                    }
                    if let Some(transport) = &self.inner.transport {
                        for id in faulted_ids {
                            transport.fault_transfer(id, msg.clone());
                        }
                        transport.drain_deferred_syncs();
                    }
                }
            }
            events.notify();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn cpu_copies() -> (seismic::Device, Copies) {
        let devices = seismic::Devices::discover(&[]).expect("discover devices");
        let device = devices
            .open_backend(seismic::BackendName::Cpu)
            .expect("open CPU device");
        let transfer = Transfer::new(&device).expect("CPU transfer");
        let events = Arc::new(Events::default());
        let copies = Copies::new(transfer, events);
        (device, copies)
    }

    /// Priority ordering (`PrefillFill > WriteBack > Stage { step, layer, p } > Promote`),
    /// `reprice`, and `cancel` enforced by `IoPlane` (never cancels an in-flight batch).
    #[test]
    fn priority_preemption_reprice_and_cancel_never_cancels_in_flight() {
        seismic::testing::setup_only();
        let (_dev, copies) = cpu_copies();
        let t_promote = copies.post(TransferPriority::Promote, Vec::new(), None, None);
        let t_stage_late = copies.post(
            TransferPriority::Stage { step: 5, layer: 2, p: 0.9 },
            Vec::new(),
            None,
            None,
        );
        let t_stage_l1_low = copies.post(
            TransferPriority::Stage { step: 4, layer: 1, p: 0.4 },
            Vec::new(),
            None,
            None,
        );
        let t_stage_l1_high = copies.post(
            TransferPriority::Stage { step: 4, layer: 1, p: 0.85 },
            Vec::new(),
            None,
            None,
        );
        let t_stage_l0 = copies.post(
            TransferPriority::Stage { step: 4, layer: 0, p: 0.3 },
            Vec::new(),
            None,
            None,
        );
        let t_wb = copies.post(TransferPriority::WriteBack, Vec::new(), None, None);
        let t_prefill = copies.post(TransferPriority::PrefillFill, Vec::new(), None, None);

        // PrefillFill (`Blocking` on H2D) preempts WriteBack, Stage, and Promote.
        let (b_prefill, p_prefill) = copies.claim_next_from_plane().expect("prefill claimed");
        assert_eq!(p_prefill.ticket, t_prefill);
        copies.inner.plane.confirm_issued(b_prefill, None);
        copies.inner.plane.retire_issued(b_prefill);

        // WriteBack (`Blocking` on D2H) preempts all Stage and Promote batches.
        let (b_wb, p_wb) = copies.claim_next_from_plane().expect("wb claimed");
        assert_eq!(p_wb.ticket, t_wb);
        assert!(!copies.cancel(t_wb), "cannot cancel a batch already claimed/submitting");
        copies.inner.plane.confirm_issued(b_wb, None);
        copies.inner.plane.retire_issued(b_wb);

        // Reprice `t_stage_late` to step 4, layer 0, p = 0.95 -> jumps ahead of `t_stage_l0`.
        assert!(copies.reprice(
            t_stage_late,
            TransferPriority::Stage { step: 4, layer: 0, p: 0.95 }
        ));
        let (b1, p1) = copies.claim_next_from_plane().expect("repriced stage");
        assert_eq!(p1.ticket, t_stage_late);
        copies.inner.plane.confirm_issued(b1, None);
        copies.inner.plane.retire_issued(b1);

        let (b2, p2) = copies.claim_next_from_plane().expect("stage l0");
        assert_eq!(p2.ticket, t_stage_l0);
        copies.inner.plane.confirm_issued(b2, None);
        copies.inner.plane.retire_issued(b2);

        // Cancel `t_stage_l1_low` while queued -> succeeds, removes from IoPlane.
        assert!(copies.cancel(t_stage_l1_low));
        assert!(!copies.cancel(t_stage_l1_low));

        // Remaining order: `t_stage_l1_high` (`Prefetch`) before `t_promote` (`Background`).
        let (b3, p3) = copies.claim_next_from_plane().expect("stage l1 high");
        assert_eq!(p3.ticket, t_stage_l1_high);
        copies.inner.plane.confirm_issued(b3, None);
        copies.inner.plane.retire_issued(b3);

        let (b4, p4) = copies.claim_next_from_plane().expect("promote");
        assert_eq!(p4.ticket, t_promote);
        copies.inner.plane.confirm_issued(b4, None);
        copies.inner.plane.retire_issued(b4);

        assert!(copies.claim_next_from_plane().is_none());
        let h2d = copies.plane().link_ledger(LINK_H2D).unwrap();
        let d2h = copies.plane().link_ledger(LINK_D2H).unwrap();
        assert_eq!(h2d.total_in_flight_bytes(), 0);
        assert_eq!(d2h.total_in_flight_bytes(), 0);
    }

    /// Stage 2: `Blocking` decode demand issued via `issue_decode_demand` shares
    /// the exact same `IoPlane` ledger as background `Copies`, updates exact
    /// in-flight/completed bytes, exposes lock-free shadow `quote()`, and
    /// drains cleanly on shutdown with zero hangs or leaked credits.
    #[test]
    fn blocking_decode_demand_shares_io_plane_and_drains_cleanly() {
        let (device, copies) = cpu_copies();
        let charges = seismic::Charges::tally(&device);
        let src = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[42, 0, 0, 0, 99, 0, 0, 0]).unwrap();
        let dst = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[0u8; 8]).unwrap();

        // Issue a Blocking decode demand transfer through `Copies::issue_decode_demand`.
        let items = [TransferItem::Run(TransferRun {
            source: &src,
            source_offset: 0,
            destination: &dst,
            destination_offset: 0,
            bytes: 8,
        })];
        let (_, sync, guard) = copies.issue_decode_demand(&items, 8).expect("demand DMA");
        assert!(sync.is_reached().unwrap());
        assert_eq!(
            copies.plane().link_ledger(LINK_H2D).unwrap().total_in_flight_bytes(),
            8,
            "IoPlane debits demand DMA bytes while flight guard is active"
        );
        let q = copies.shadow_quote_h2d(2 * 1024 * 1024);
        assert!(q.effective_bps > 0.0 && q.estimated_s > 0.0);

        // Retiring the flight guard returns in-flight bytes to exact 0.
        guard.retire();
        let ledger = copies.plane().link_ledger(LINK_H2D).unwrap();
        assert_eq!(ledger.total_in_flight_bytes(), 0);
        assert_eq!(ledger.completed_bytes_by_class[PriorityClass::Blocking.index()], 8);

        // Start issuer + watch threads, post background copy, and verify clean shutdown.
        let stopped = Arc::new(AtomicBool::new(false));
        let stop_flag = Arc::clone(&stopped);
        let threads = copies.start(move || stop_flag.load(Ordering::Acquire)).unwrap();
        let ticket = copies.post(
            TransferPriority::Promote,
            vec![BatchItem::Run(Run {
                source: src,
                source_offset: 0,
                destination: dst,
                destination_offset: 0,
                bytes: 8,
            })],
            None,
            None,
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !copies.completed(ticket).unwrap() {
            assert!(std::time::Instant::now() < deadline, "promote copy timed out");
            std::thread::sleep(Duration::from_millis(1));
        }
        stopped.store(true, Ordering::Release);
        copies.inner.events.notify();
        for t in threads {
            t.join().expect("thread exited cleanly");
        }
        assert_eq!(copies.plane().link_ledger(LINK_H2D).unwrap().total_in_flight_bytes(), 0);
        assert_eq!(copies.teardown_quarantined(), 0);
    }

    /// Stage 2: an active `Blocking` decode demand miss on `CopyEngineH2D`
    /// shares the `CoupledSet` with `Prefetch`/`Background`, a quarantined
    /// `DemandFlightGuard` increments `quarantined_ops` and never leaks
    /// in-flight bytes, and `teardown_quarantined()` drains cleanly.
    #[test]
    fn blocking_decode_quarantine_and_speculative_preemption_on_shared_plane() {
        let (device, copies) = cpu_copies();
        let charges = seismic::Charges::tally(&device);
        let src = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[7, 0, 0, 0, 8, 0, 0, 0]).unwrap();
        let dst = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[0u8; 8]).unwrap();

        // Queue a speculative Promote (Background) and a Blocking PrefillFill.
        let t_bg = copies.post(TransferPriority::Promote, Vec::new(), None, None);
        let t_blk = copies.post(TransferPriority::PrefillFill, Vec::new(), None, None);

        // Blocking strictly preempts Background on the shared IoPlane.
        let (b_blk, p_blk) = copies.claim_next_from_plane().expect("blocking preempts background");
        assert_eq!(p_blk.ticket, t_blk);
        copies.inner.plane.confirm_issued(b_blk, None);
        copies.inner.plane.retire_issued(b_blk);

        // Cancel the remaining queued background copy cleanly (zero bus debit).
        assert!(copies.cancel(t_bg));

        // Issue a Blocking decode demand and simulate ambiguous/partial failure -> Quarantined.
        let items = [TransferItem::Run(TransferRun {
            source: &src,
            source_offset: 0,
            destination: &dst,
            destination_offset: 0,
            bytes: 8,
        })];
        let (_, _, guard) = copies.issue_decode_demand(&items, 8).expect("demand DMA");
        guard.quarantine("simulated partial issue failure");

        let ledger = copies.plane().link_ledger(LINK_H2D).unwrap();
        assert_eq!(ledger.total_in_flight_bytes(), 0, "quarantined op releases wire credit");
        assert_eq!(ledger.quarantined_ops, 1);
        assert_eq!(copies.plane().quarantined_lease_count(), 1);
        assert_eq!(copies.teardown_quarantined(), 1);
        assert_eq!(copies.plane().quarantined_lease_count(), 0);
    }

    /// Stage 3 + regression guard (Causes A & B):
    /// - `TransferPriority::WriteBack` (`Streaming`) never trips cross-link
    ///   `Blocking` preemption on `LINK_STORAGE_NVME`.
    /// - When `BUS_HOST_DDR` is unsaturated (6 GB/s NVMe + 25 GB/s H2D < 60 GB/s),
    ///   a `Blocking` NVMe op does NOT stall `LINK_COPY_H2D` speculative promotes.
    /// - Once `BUS_HOST_DDR` is saturated (e.g. concurrent CPU-site reservation),
    ///   a `Blocking` NVMe op DOES preempt `LINK_COPY_H2D` speculative promotes,
    ///   and clearing the `Blocking` op fires the cross-link wakeup hook.
    #[test]
    fn cross_link_storage_blocking_preempts_h2d_speculative_on_shared_plane() {
        seismic::testing::setup_only();
        let devices = seismic::Devices::discover(&[]).expect("discover devices");
        let device = devices
            .open_backend(seismic::BackendName::Cpu)
            .expect("open CPU device");
        let transfer = Transfer::new(&device).expect("CPU transfer");
        let shared_plane = IoPlane::new_default();
        let events = Arc::new(Events::default());
        let copies = Copies::with_plane(
            Arc::clone(&shared_plane),
            transfer,
            Arc::clone(&events),
            None,
        );

        // 1. WriteBack (`Streaming` on `LINK_D2H`) must NEVER trip cross-link
        //    `Blocking` preemption on `LINK_STORAGE_NVME`, even on a saturated bus.
        let _ddr_sat_wb = shared_plane
            .reserve_bus_scoped(BUS_HOST_DDR, 35e9)
            .expect("saturate HostDdr");
        let t_wb = copies.post(TransferPriority::WriteBack, Vec::new(), None, None);
        assert!(
            !shared_plane.has_coupled_blocking_waiting(LINK_STORAGE_NVME),
            "WriteBack must not trip cross-link Blocking preemption on LINK_STORAGE_NVME"
        );
        assert!(copies.cancel(t_wb));
        drop(_ddr_sat_wb);

        // 2. Unsaturated HostDdr (6 + 25 = 31 GB/s < 60 GB/s) + full-duplex PCIe:
        //    Blocking NVMe must NOT stall H2D speculative Promote.
        let t_bg_unsat = copies.post(TransferPriority::Promote, Vec::new(), None, None);
        let storage_blocking_unsat = shared_plane
            .submit(TransferRequest {
                link: LINK_STORAGE_NVME,
                key: ScheduleKey::blocking(GlobalUrgency(0), 1, 0, 1),
                bytes: 2 * 1024 * 1024,
                content_key: None,
                placement: PlacementRequirement::FixedDestination,
                compatibility: BatchCompatibility::Isolated,
                src_lease: None,
                dst_lease: BufferLease::unmanaged(
                    DOMAIN_HOST_RAM,
                    NonNull::dangling(),
                    0,
                    0,
                    2 * 1024 * 1024,
                    4096,
                    0,
                ),
                direct: true,
            })
            .unwrap();
        let (bid_unsat, payload_unsat) = copies
            .claim_next_from_plane()
            .expect("unsaturated HostDdr must NOT cross-preempt H2D Promote");
        assert_eq!(payload_unsat.ticket, t_bg_unsat);
        shared_plane.confirm_issued(bid_unsat, None);
        shared_plane.retire_issued(bid_unsat);
        drop(storage_blocking_unsat);

        // 3. Saturated HostDdr (35 + 6 + 25 = 66 GB/s >= 60 GB/s):
        //    Blocking NVMe MUST preempt H2D speculative Promote, and clearing
        //    it wakes `Copies` so H2D Promote resumes cleanly.
        let _ddr_sat = shared_plane
            .reserve_bus_scoped(BUS_HOST_DDR, 35e9)
            .expect("saturate HostDdr");
        let t_bg = copies.post(TransferPriority::Promote, Vec::new(), None, None);
        let storage_blocking = shared_plane
            .submit(TransferRequest {
                link: LINK_STORAGE_NVME,
                key: ScheduleKey::blocking(GlobalUrgency(0), 1, 0, 2),
                bytes: 2 * 1024 * 1024,
                content_key: None,
                placement: PlacementRequirement::FixedDestination,
                compatibility: BatchCompatibility::Isolated,
                src_lease: None,
                dst_lease: BufferLease::unmanaged(
                    DOMAIN_HOST_RAM,
                    NonNull::dangling(),
                    0,
                    0,
                    2 * 1024 * 1024,
                    4096,
                    0,
                ),
                direct: true,
            })
            .unwrap();

        assert!(
            copies.claim_next_from_plane().is_none(),
            "H2D Background promote must yield while Blocking NVMe is queued on saturated HostDdr"
        );

        // Shadow quotes work lock-free for both storage and H2D links.
        assert!(copies.shadow_quote_storage(2 * 1024 * 1024).effective_bps > 0.0);
        assert!(copies.shadow_quote_h2d(2 * 1024 * 1024).effective_bps > 0.0);

        // Drop (cancel) the Blocking NVMe read -> H2D Promote is immediately admitted!
        drop(storage_blocking);
        let (bid, payload) = copies
            .claim_next_from_plane()
            .expect("H2D promote admitted once Blocking NVMe clears");
        assert_eq!(payload.ticket, t_bg);
        shared_plane.confirm_issued(bid, None);
        shared_plane.retire_issued(bid);
        drop(_ddr_sat);

        assert_eq!(shared_plane.bus_ledger(BUS_HOST_DDR).unwrap().in_flight_weighted_bytes, 0);
        assert_eq!(shared_plane.bus_ledger(BUS_PCIE_ROOT).unwrap().in_flight_weighted_bytes, 0);
    }

    /// Regression test for `expert-copies-issue` panic ("claimed op has payload
    /// in Copies"): hammers concurrent `post` + `reprice` + `cancel` + live
    /// `issue`/`watch` threads across TWO independent `Copies` instances sharing
    /// one `IoPlane`, proving neither publish-before-insert, cancel-before-detach,
    /// nor cross-instance claim can occur.
    #[test]
    fn concurrent_post_cancel_reprice_across_two_copies_on_shared_plane() {
        let devices = seismic::Devices::discover(&[]).expect("discover devices");
        let device = devices
            .open_backend(seismic::BackendName::Cpu)
            .expect("open CPU device");
        let charges = seismic::Charges::tally(&device);
        let src = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[1, 0, 0, 0, 2, 0, 0, 0]).unwrap();
        let dst = seismic::Tensor::from_host(&charges, seismic::Element::i32(), &[2], &[0u8; 8]).unwrap();

        let shared_plane = IoPlane::new_default();
        let copies_a = Copies::with_plane(
            Arc::clone(&shared_plane),
            Transfer::new(&device).expect("CPU transfer A"),
            Arc::new(Events::default()),
            None,
        );
        let copies_b = Copies::with_plane(
            Arc::clone(&shared_plane),
            Transfer::new(&device).expect("CPU transfer B"),
            Arc::new(Events::default()),
            None,
        );

        let stopped = Arc::new(AtomicBool::new(false));
        let stop_a = Arc::clone(&stopped);
        let stop_b = Arc::clone(&stopped);
        let threads_a = copies_a.start(move || stop_a.load(Ordering::Acquire)).unwrap();
        let threads_b = copies_b.start(move || stop_b.load(Ordering::Acquire)).unwrap();

        let mut tickets_a = Vec::new();
        let mut tickets_b = Vec::new();
        for i in 0..200u64 {
            let ta = copies_a.post(
                if i % 3 == 0 {
                    TransferPriority::Promote
                } else {
                    TransferPriority::Stage { step: i, layer: 0, p: 0.7 }
                },
                vec![BatchItem::Run(Run {
                    source: src.clone(),
                    source_offset: 0,
                    destination: dst.clone(),
                    destination_offset: 0,
                    bytes: 8,
                })],
                None,
                None,
            );
            let tb = copies_b.post(
                TransferPriority::WriteBack,
                vec![BatchItem::Run(Run {
                    source: src.clone(),
                    source_offset: 0,
                    destination: dst.clone(),
                    destination_offset: 0,
                    bytes: 8,
                })],
                None,
                None,
            );
            if i % 4 == 0 {
                copies_a.reprice(ta, TransferPriority::Stage { step: 0, layer: 0, p: 0.99 });
            }
            if i % 5 == 0 {
                let _ = copies_a.cancel(ta);
            } else {
                tickets_a.push(ta);
            }
            if i % 7 == 0 {
                let _ = copies_b.cancel(tb);
            } else {
                tickets_b.push(tb);
            }
        }

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        for &t in &tickets_a {
            while !copies_a.completed(t).unwrap() {
                assert!(std::time::Instant::now() < deadline, "copies_a ticket {t} timed out");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        for &t in &tickets_b {
            while !copies_b.completed(t).unwrap() {
                assert!(std::time::Instant::now() < deadline, "copies_b ticket {t} timed out");
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        stopped.store(true, Ordering::Release);
        copies_a.inner.events.notify();
        copies_b.inner.events.notify();
        for t in threads_a.into_iter().chain(threads_b) {
            t.join().expect("thread exited cleanly without panic");
        }
        assert_eq!(shared_plane.link_ledger(LINK_H2D).unwrap().total_in_flight_bytes(), 0);
        assert_eq!(shared_plane.link_ledger(LINK_D2H).unwrap().total_in_flight_bytes(), 0);
        assert_eq!(shared_plane.bus_ledger(BUS_HOST_DDR).unwrap().in_flight_weighted_bytes, 0);
        assert_eq!(shared_plane.bus_ledger(BUS_PCIE_ROOT).unwrap().in_flight_weighted_bytes, 0);
    }

    /// Proof cases #1, #2, and #4 end-to-end on `ExpertTransport` + `Copies`.
    #[test]
    fn delayed_fill_overlapping_readers_and_cancellation_slot_reuse_safety() {
        use crate::experts::transport::{DestSlot, ExpertKey, ExpertTransport, ScratchGenId, SourceRef};

        let transport = ExpertTransport::new_registry_only();

        let producer_transport = Arc::clone(&transport);
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(15));
            producer_transport.publish_prefill_sync(2, 10, 0, ScratchGenId(1), None);
        });
        let got = transport.take_prefill_sync(2, 10, 0).expect("delayed prefill sync");
        assert_eq!(got.generation, ScratchGenId(1));
        assert!(got.sync.is_none());
        producer.join().unwrap();

        let key_unbegun = ExpertKey { group: 0, layer: 1, expert: 3 };
        let slot_unbegun = DestSlot { group: 0, slot: 1 };
        let unbegun_id = transport.register_hint(key_unbegun, 11, slot_unbegun);
        assert!(!transport.can_reuse_slot(slot_unbegun));
        assert!(transport.cancel_unbegun_hint(unbegun_id));
        assert!(transport.can_reuse_slot(slot_unbegun));

        let key_live = ExpertKey { group: 0, layer: 1, expert: 7 };
        let src_live = SourceRef { tier_record: 22, generation: 2 };
        let slot_live = DestSlot { group: 0, slot: 2 };
        let live_id = transport.register_in_flight_hint(key_live, src_live, slot_live, None);
        assert!(!transport.cancel_unbegun_hint(live_id));
        let joined1 = transport.join_by_key(key_live, src_live).expect("reader 1");
        let joined2 = transport.join_by_key(key_live, src_live).expect("reader 2");
        assert!(!transport.can_reuse_slot(slot_live));

        transport.complete_dma(live_id);
        assert!(!transport.try_release_hint_slot(slot_live));
        assert!(!transport.can_reuse_slot(slot_live));

        joined1.lease.cancel();
        assert!(!transport.can_reuse_slot(slot_live));

        joined2.lease.release();
        assert!(transport.can_reuse_slot(slot_live));
        assert!(transport.try_release_hint_slot(slot_live));
    }
}
