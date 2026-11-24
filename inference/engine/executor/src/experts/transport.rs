//! Unified Copy-Engine-First `ExpertTransport`: the single owner of physical
//! Host-Tier <-> VRAM expert transfers across:
//!
//! 1. **Speculative decode lookahead (`LinkMover` staging into `MAP_STAGE`)**
//! 2. **On-demand decode DMA landing (`DecodeFront -> DMA -> DecodeBack`)**
//! 3. **Producer-enqueued prefill `HostTier -> Scratch` (+ seeded `Scratch -> Bank`) DMA**
//!
//! Placement (`Directory` / `Residency`) owns *where* experts live and when
//! slots are allocated or evicted; execution scheduling (`native_target.rs`)
//! owns graph submission order; `ExpertTransport` owns in-flight transfer
//! reservations, demand-joins-hint coalescing, slot lifetime pinning, fault
//! quarantine, and physical H2D/D2D copy execution.
//!
//! ## Progress & Lifetime Invariants
//!
//! - **Lock order:** `ExpertTransport` uses a short leaf mutex for metadata
//!   only. It NEVER acquires `Residency` and NEVER holds any lock across a
//!   CUDA driver call or wait. Zero `cuStreamWaitValue32`.
//! - **Demand joins in-flight hint:** When decode demand requests an expert
//!   whose `(group, layer, tier_record, source_generation)` is already in
//!   flight on the copy stream (or already landed in a pinned staging slot),
//!   demand attaches a [`ConsumerLease`] and reuses that transfer's
//!   `TransferSync` event and destination slot instead of issuing a duplicate
//!   PCIe copy.
//! - **Pinned lifetimes & safe cancellation:** Every active transfer pins
//!   `(source_generation, dest_slot)` through BOTH DMA completion AND release
//!   of the last attached consumer. Cancelling one consumer detaches only its
//!   own demand without recycling a pending DMA or freeing a slot another
//!   consumer still needs.
//! - **Partial enqueue fault quarantine:** If a hardware copy enqueue fails
//!   after reserving destination slots, every touched slot is quarantined so
//!   corrupt/partial bytes can never be joined or silently reused.

use super::Words;
use super::bank::Stores;
use crate::planning::{ExpertBankPlan, RecordTensors};
use seismic::{Device, Tensor, Transfer, TransferItem, TransferRun, TransferSync};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Identifies the logical expert payload being transferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ExpertKey {
    pub group: usize,
    pub layer: usize,
    pub expert: usize,
}

/// Identifies a specific host-tier source slot and its generation.
/// When a host-tier record is overwritten with a different expert, its
/// generation advances so stale transfers can never be joined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SourceRef {
    pub tier_record: usize,
    pub generation: u64,
}

/// Identifies a VRAM destination slot (`MAP_STAGE` ring or `DmaLanding`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DestSlot {
    pub group: usize,
    pub slot: usize,
}

/// Unique identifier for a tracked transfer in [`ExpertTransport`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TransferId(pub u64);

/// A demand consumer's RAII lease pinning a transfer's `(source_generation, dest_slot)`.
///
/// Calling [`Self::cancel`], [`Self::release`], or dropping the lease detaches
/// ONLY this consumer's demand. The destination slot remains pinned until DMA
/// completes AND every attached consumer has detached.
#[derive(Debug)]
pub(crate) struct ConsumerLease {
    transport: Arc<ExpertTransport>,
    transfer_id: TransferId,
    consumer_id: u64,
    active: bool,
}

impl ConsumerLease {
    /// Detach this consumer's demand due to cancellation without recycling
    /// pending DMA or disturbing other consumers on the same transfer.
    #[cfg(test)]
    pub(crate) fn cancel(mut self) {
        if self.active {
            self.active = false;
            self.transport.detach_consumer(self.transfer_id, self.consumer_id);
        }
    }

    /// Release this consumer's pin after it has finished reading the
    /// destination slot.
    pub(crate) fn release(mut self) {
        if self.active {
            self.active = false;
            self.transport.detach_consumer(self.transfer_id, self.consumer_id);
        }
    }

    /// Disarm this lease without detaching its consumer pin in `Registry`.
    /// Used when handing uncertain work to `ExpertTransport`'s retirement
    /// quarantine so the destination slot remains permanently pinned without
    /// creating an `Arc<ExpertTransport>` reference cycle.
    fn disarm(mut self) {
        self.active = false;
    }
}

impl Drop for ConsumerLease {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            self.transport.detach_consumer(self.transfer_id, self.consumer_id);
        }
    }
}

/// Result of joining an existing in-flight or landed transfer.
pub(crate) struct JoinedTransfer {
    pub dest: DestSlot,
    /// `Some(sync)` while the transfer is still in flight on the copy engine;
    /// `None` once DMA has already completed (`Landed`).
    pub sync: Option<TransferSync>,
    pub lease: ConsumerLease,
}

/// A unified VRAM slot freshly landed by `DecodeDma` (`Record::DemandLanding`)
/// eligible for O(1) zero-recopy promotion (`DemandLanding -> Held`) once its
/// `DecodeBack` completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DemandSlot {
    pub group: usize,
    pub slot: usize,
    pub staged: Option<(usize, usize)>,
}

/// RAII guard tracking an active `land_decode_misses` flight in
/// `ExpertTransport::demand_in_flight`. Increments on entry and decrements
/// unconditionally on release or drop so `demand_active()` can never leak.
pub(crate) struct DemandFlightGuard {
    transport: Arc<ExpertTransport>,
}

impl DemandFlightGuard {
    fn new(transport: Arc<ExpertTransport>) -> Self {
        transport.demand_in_flight.fetch_add(1, Ordering::AcqRel);
        Self { transport }
    }
}

impl Drop for DemandFlightGuard {
    fn drop(&mut self) {
        self.transport.demand_in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Explicit lifecycle state of one batch inside an [`IssuedDemandBatch`]
/// transaction.
///
/// Enforces that only proven-unissued or proven-completed batches (before
/// `DecodeBack` submission) can return slots to `Directory`, while any
/// `PartiallyIssued` (mid-batch failure / unknown hardware state) or
/// timed-out/errored batch is unconditionally quarantined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DemandBatchState {
    /// Own-write slots claimed from `Directory`; no hardware copy call has
    /// started yet. Safe to abort immediately back to `Directory` without
    /// waiting on the copy stream.
    Unissued,
    /// Own-write hardware copy/record call was attempted and failed (or was
    /// interrupted mid-issue). Accepted copies may exist beyond the last
    /// recorded timeline value; these slots MUST be quarantined and NEVER
    /// freed early.
    PartiallyIssued,
    /// Own-write batch accepted by the copy stream at timeline `value`.
    /// Ordered prefix on this producer: waiting for `value` (`Ok(true)`)
    /// proves this batch and all earlier `Submitted` batches have finished
    /// writing, provided `DecodeBack` has not been submitted.
    Submitted { value: u64 },
    /// Joined an existing in-flight or landed transfer owned by another
    /// producer (`LinkMover` or another demand flight). Holds only reader
    /// pins (`ConsumerLease`); never owns the destination write or slot
    /// lifecycle.
    JoinedExisting,
    /// Terminal state: all leases detached or handed to retirement, and safe
    /// slots returned.
    Retired,
}

/// Whether `DecodeBack` compute has been submitted to read an
/// [`IssuedDemandBatch`]'s landed/joined slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ComputeUse {
    /// Only the copy stream has touched own-write slots; `DecodeBack` has not
    /// been submitted yet.
    BeforeBack,
    /// `DecodeBack` has been submitted on the compute stream to read these
    /// slots. Copy completion alone is insufficient until compute completes.
    BackSubmitted,
}

enum DemandBatchPayload {
    /// Own H2D write batch: owns the `DemandLanding` slots, `TransferId`s,
    /// owner leases, and source/destination tensor backing.
    OwnWrite {
        slots: Vec<DemandSlot>,
        transfer_ids: Vec<TransferId>,
        owner_leases: Vec<ConsumerLease>,
        backing: Vec<Tensor>,
    },
    /// Joined existing transfer(s): holds only reader pins (`ConsumerLease`).
    Joined {
        reader_leases: Vec<ConsumerLease>,
    },
}

struct DemandBatch {
    state: DemandBatchState,
    payload: DemandBatchPayload,
}

/// Single-owner RAII transaction for one layer's on-demand decode DMA landing,
/// from first slot claim through `DecodeLanding` and final retirement.
///
/// - Tracks explicit per-batch states (`Unissued`, `PartiallyIssued`,
///   `Submitted`, `JoinedExisting`, `Retired`) and distinguishes own writes
///   from joined consumers.
/// - Retains source pins, destination backing, and reader pins until last use.
/// - Explicit `abort(...)` runs outside residency/queue locks and only returns
///   proven-safe slots to `Directory`.
/// - `Drop` hands any unretired/uncertain work to `ExpertTransport`'s
///   retirement quarantine without blocking or panicking.
/// - Happy path transfers ownership into [`DecodeLanding`] with zero extra
///   descriptor allocations and zero per-expert synchronization.
pub(crate) struct IssuedDemandBatch {
    transport: Arc<ExpertTransport>,
    layer: usize,
    sequence: u64,
    compute_use: ComputeUse,
    joined_syncs: Vec<TransferSync>,
    latest_dma_sync: Option<TransferSync>,
    batches: Vec<DemandBatch>,
    _flight: Option<DemandFlightGuard>,
}

impl IssuedDemandBatch {
    /// Start a new owning demand transaction for `(layer, sequence)`.
    pub(crate) fn new(transport: Arc<ExpertTransport>, layer: usize, sequence: u64) -> Self {
        let flight = DemandFlightGuard::new(Arc::clone(&transport));
        Self {
            transport,
            layer,
            sequence,
            compute_use: ComputeUse::BeforeBack,
            joined_syncs: Vec::new(),
            latest_dma_sync: None,
            batches: Vec::new(),
            _flight: Some(flight),
        }
    }

    /// Record a joined consumer lease (and optional cross-stream sync) for an
    /// existing transfer owned by another producer (`JoinedExisting`).
    pub(crate) fn record_joined(&mut self, lease: ConsumerLease, sync: Option<TransferSync>) {
        if let Some(sync) = sync {
            if !self.joined_syncs.iter().any(|seen| seen.same_batch(&sync)) {
                self.joined_syncs.push(sync);
            }
        }
        if let Some(DemandBatch {
            state: DemandBatchState::JoinedExisting,
            payload: DemandBatchPayload::Joined { reader_leases },
        }) = self.batches.iter_mut().find(|b| b.state == DemandBatchState::JoinedExisting)
        {
            reader_leases.push(lease);
        } else {
            self.batches.push(DemandBatch {
                state: DemandBatchState::JoinedExisting,
                payload: DemandBatchPayload::Joined { reader_leases: vec![lease] },
            });
        }
    }

    /// Claim an own-write batch in state `Unissued`, before any hardware copy
    /// call has started. Returns its batch index.
    pub(crate) fn claim_own_batch(
        &mut self,
        slots: Vec<DemandSlot>,
        transfer_ids: Vec<TransferId>,
        owner_leases: Vec<ConsumerLease>,
        backing: Vec<Tensor>,
    ) -> usize {
        let idx = self.batches.len();
        self.batches.push(DemandBatch {
            state: DemandBatchState::Unissued,
            payload: DemandBatchPayload::OwnWrite {
                slots,
                transfer_ids,
                owner_leases,
                backing,
            },
        });
        idx
    }

    /// Transition an `Unissued` own-write batch to `PartiallyIssued` right
    /// before attempting the hardware copy/record call.
    pub(crate) fn begin_issue(&mut self, idx: usize) {
        let batch = &mut self.batches[idx];
        assert_eq!(batch.state, DemandBatchState::Unissued, "begin_issue requires Unissued batch");
        batch.state = DemandBatchState::PartiallyIssued;
    }

    /// Revert a `PartiallyIssued` own-write batch back to `Unissued` when
    /// `copy_synced` returned a pre-issue refusal (`TransferError::refused_before_issue`),
    /// proving zero copies of this batch reached the device stream so `abort`
    /// unwinds it through pin-aware release without quarantining it.
    pub(crate) fn mark_refused_before_issue(&mut self, idx: usize) {
        let batch = &mut self.batches[idx];
        assert_eq!(
            batch.state,
            DemandBatchState::PartiallyIssued,
            "mark_refused_before_issue requires PartiallyIssued batch"
        );
        batch.state = DemandBatchState::Unissued;
    }

    /// Borrow the source/destination backing tensors of an own-write batch
    /// (`[src0, dst0, src1, dst1, ...]`) to construct `TransferItem` runs
    /// without cloning.
    fn own_backing(&self, idx: usize) -> &[Tensor] {
        match &self.batches[idx].payload {
            DemandBatchPayload::OwnWrite { backing, .. } => backing,
            DemandBatchPayload::Joined { .. } => unreachable!("own_backing called on Joined batch"),
        }
    }

    /// Transition a `PartiallyIssued` own-write batch to `Submitted { value }`
    /// after the hardware copy stream accepted all copies and recorded `sync`.
    pub(crate) fn mark_submitted(&mut self, idx: usize, value: u64, sync: Option<TransferSync>) {
        let batch = &mut self.batches[idx];
        assert_eq!(batch.state, DemandBatchState::PartiallyIssued, "mark_submitted requires PartiallyIssued batch");
        if let Some(sync) = sync {
            if let DemandBatchPayload::OwnWrite { transfer_ids, .. } = &batch.payload {
                for &id in transfer_ids {
                    self.transport.attach_sync(id, sync.clone());
                }
            }
            self.latest_dma_sync = Some(sync);
        }
        batch.state = DemandBatchState::Submitted { value };
    }

    /// Record that `DecodeBack` has been submitted on the compute stream to
    /// read this transaction's slots. Copy completion alone is no longer
    /// sufficient to retire or free storage.
    pub(crate) fn mark_back_submitted(&mut self) {
        self.compute_use = ComputeUse::BackSubmitted;
    }

    pub(crate) fn compute_use(&self) -> ComputeUse {
        self.compute_use
    }

    /// Highest timeline value among all `Submitted` own-write batches on this
    /// producer. Proves only an ordered prefix of `Submitted` batches; never
    /// proves a `PartiallyIssued` batch that errored mid-issue.
    pub(crate) fn highest_submitted(&self) -> Option<u64> {
        self.batches
            .iter()
            .filter_map(|b| match b.state {
                DemandBatchState::Submitted { value } => Some(value),
                _ => None,
            })
            .max()
    }

    #[cfg(test)]
    pub(crate) fn batch_states(&self) -> Vec<DemandBatchState> {
        self.batches.iter().map(|b| b.state).collect()
    }

    /// Explicitly abort this transaction outside all residency/queue locks.
    ///
    /// - If `DecodeBack` has already been submitted (`ComputeUse::BackSubmitted`),
    ///   copy completion alone is insufficient: all own writes and reader pins
    ///   are quarantined and zero slots are returned to `Directory`.
    /// - Otherwise (`ComputeUse::BeforeBack`):
    ///   - Waits once on [`Self::highest_submitted`] outside locks.
    ///   - `Unissued` batches return their slots to `abort_slots` immediately.
    ///   - `Submitted` batches return their slots ONLY if `wait_dma` returned
    ///     `Ok(true)`; on timeout (`Ok(false)`) or `Err(_)`, they are
    ///     quarantined with their backing/pins retained.
    ///   - `PartiallyIssued` batches are ALWAYS quarantined with their
    ///     backing/pins retained (never returned to `abort_slots`), even if
    ///     earlier `Submitted` batches completed.
    ///   - `JoinedExisting` batches release ONLY our own reader pins without
    ///     disturbing the writer or other readers.
    pub(crate) fn abort(
        &mut self,
        reason: &str,
        wait_dma: impl FnOnce(u64) -> Result<bool, String>,
        abort_slots: &mut impl FnMut(&[(usize, usize)]),
    ) {
        if self.batches.iter().all(|b| b.state == DemandBatchState::Retired) {
            self._flight.take();
            return;
        }
        if self.compute_use == ComputeUse::BackSubmitted {
            self.quarantine(reason);
            self.transport.drain_deferred_syncs();
            return;
        }

        let highest = self.highest_submitted();
        let prefix_proven = match highest {
            Some(value) => matches!(wait_dma(value), Ok(true)),
            None => true,
        };

        let mut slots_to_abort: Vec<(usize, usize)> = Vec::new();
        let mut any_quarantined = false;

        for batch in &mut self.batches {
            match batch.state {
                DemandBatchState::Retired => {}
                DemandBatchState::Unissued => {
                    if let DemandBatchPayload::OwnWrite {
                        slots,
                        transfer_ids,
                        owner_leases,
                        backing,
                    } = &mut batch.payload
                    {
                        self.transport.cancel_unissued_dma(transfer_ids, owner_leases);
                        slots_to_abort.extend(slots.drain(..).map(|s| (s.group, s.slot)));
                        backing.clear();
                    }
                    batch.state = DemandBatchState::Retired;
                }
                DemandBatchState::Submitted { .. } => {
                    if let DemandBatchPayload::OwnWrite {
                        slots,
                        transfer_ids,
                        owner_leases,
                        backing,
                    } = &mut batch.payload
                    {
                        if prefix_proven {
                            for id in transfer_ids.drain(..) {
                                self.transport.complete_dma(id);
                            }
                            for lease in owner_leases.drain(..) {
                                lease.release();
                            }
                            slots_to_abort.extend(slots.drain(..).map(|s| (s.group, s.slot)));
                            backing.clear();
                        } else {
                            let msg = format!("{reason}; prior H2D wait did not confirm completion");
                            self.transport.quarantine_uncertain_own(
                                transfer_ids,
                                owner_leases,
                                backing,
                                &msg,
                            );
                            slots.clear();
                            any_quarantined = true;
                        }
                    }
                    batch.state = DemandBatchState::Retired;
                }
                DemandBatchState::PartiallyIssued => {
                    if let DemandBatchPayload::OwnWrite {
                        slots,
                        transfer_ids,
                        owner_leases,
                        backing,
                    } = &mut batch.payload
                    {
                        self.transport.quarantine_uncertain_own(
                            transfer_ids,
                            owner_leases,
                            backing,
                            reason,
                        );
                        slots.clear();
                        any_quarantined = true;
                    }
                    batch.state = DemandBatchState::Retired;
                }
                DemandBatchState::JoinedExisting => {
                    if let DemandBatchPayload::Joined { reader_leases } = &mut batch.payload {
                        for lease in reader_leases.drain(..) {
                            lease.release();
                        }
                    }
                    batch.state = DemandBatchState::Retired;
                }
            }
        }

        if any_quarantined {
            let mut syncs = std::mem::take(&mut self.joined_syncs);
            if let Some(sync) = self.latest_dma_sync.take() {
                syncs.push(sync);
            }
            self.transport.defer_syncs(syncs);
        } else {
            self.joined_syncs.clear();
            let _ = self.latest_dma_sync.take();
        }
        self._flight.take();
        self.transport.drain_deferred_syncs();

        if !slots_to_abort.is_empty() {
            abort_slots(&slots_to_abort);
        }
    }

    /// Hand all unretired work to `ExpertTransport`'s retirement quarantine
    /// without blocking, waiting, or panicking. Safe to call from `Drop` or
    /// callback threads.
    pub(crate) fn quarantine(&mut self, reason: &str) {
        if self.batches.iter().all(|b| b.state == DemandBatchState::Retired) {
            self._flight.take();
            return;
        }
        for batch in &mut self.batches {
            if batch.state == DemandBatchState::Retired {
                continue;
            }
            match &mut batch.payload {
                DemandBatchPayload::OwnWrite {
                    slots,
                    transfer_ids,
                    owner_leases,
                    backing,
                } => {
                    self.transport.quarantine_uncertain_own(
                        transfer_ids,
                        owner_leases,
                        backing,
                        reason,
                    );
                    slots.clear();
                }
                DemandBatchPayload::Joined { reader_leases } => {
                    match self.compute_use {
                        ComputeUse::BeforeBack => {
                            for lease in reader_leases.drain(..) {
                                lease.release();
                            }
                        }
                        ComputeUse::BackSubmitted => {
                            // `DecodeBack` may still be reading these joined
                            // slots on the GPU: retain our reader pins in
                            // `Registry` without forming an `Arc` cycle.
                            for lease in reader_leases.drain(..) {
                                lease.disarm();
                            }
                        }
                    }
                }
            }
            batch.state = DemandBatchState::Retired;
        }
        let mut syncs = std::mem::take(&mut self.joined_syncs);
        if let Some(sync) = self.latest_dma_sync.take() {
            syncs.push(sync);
        }
        self.transport.defer_syncs(syncs);
        self._flight.take();
    }

    /// Happy-path retirement once `DecodeBack` is proven complete on the GPU:
    /// completes own transfers, releases all own and joined leases, drops
    /// backing, drains deferred syncs outside all residency locks, and returns
    /// `(layer, sequence, demand_slots)` for O(1) promotion or release in
    /// `Directory`.
    pub(crate) fn retire_completed(mut self) -> (usize, u64, Vec<DemandSlot>) {
        let mut demand_slots = Vec::new();
        for batch in &mut self.batches {
            if batch.state == DemandBatchState::Retired {
                continue;
            }
            match &mut batch.payload {
                DemandBatchPayload::OwnWrite {
                    slots,
                    transfer_ids,
                    owner_leases,
                    backing,
                } => {
                    for id in transfer_ids.drain(..) {
                        self.transport.complete_dma(id);
                    }
                    for lease in owner_leases.drain(..) {
                        lease.release();
                    }
                    demand_slots.append(slots);
                    backing.clear();
                }
                DemandBatchPayload::Joined { reader_leases } => {
                    for lease in reader_leases.drain(..) {
                        lease.release();
                    }
                }
            }
            batch.state = DemandBatchState::Retired;
        }
        self.joined_syncs.clear();
        let _ = self.latest_dma_sync.take();
        self._flight.take();
        self.transport.drain_deferred_syncs();
        (self.layer, self.sequence, demand_slots)
    }

    /// Transfer ownership into [`DecodeLanding`] on the happy path with zero
    /// extra descriptor allocations and zero per-expert synchronization.
    pub(crate) fn into_landing(mut self) -> DecodeLanding {
        let mut syncs = std::mem::take(&mut self.joined_syncs);
        if let Some(sync) = self.latest_dma_sync.take() {
            syncs.push(sync);
        }
        DecodeLanding {
            layer: self.layer,
            sequence: self.sequence,
            syncs,
            batch: self,
        }
    }
}

impl Drop for IssuedDemandBatch {
    fn drop(&mut self) {
        if self.batches.iter().any(|b| b.state != DemandBatchState::Retired) {
            self.quarantine("IssuedDemandBatch dropped without explicit retirement");
        }
    }
}

/// Outcome of [`ExpertTransport::land_decode_misses`]: all cross-stream sync
/// events `DecodeBack` must wait on (`cuStreamWaitEvent`), plus the single
/// owning [`IssuedDemandBatch`] transaction holding all own writes, joined
/// reader pins, backing tensors, and flight guard until `DecodeBack` retires.
pub(crate) struct DecodeLanding {
    pub layer: usize,
    pub sequence: u64,
    pub syncs: Vec<TransferSync>,
    batch: IssuedDemandBatch,
}

impl DecodeLanding {
    /// Record that `DecodeBack` has been submitted on the compute stream.
    pub(crate) fn mark_back_submitted(&mut self) {
        self.batch.mark_back_submitted();
    }

    pub(crate) fn compute_use(&self) -> ComputeUse {
        self.batch.compute_use()
    }

    /// Release all transport consumer leases, mark freshly issued demand
    /// transfers complete, drain deferred syncs outside all residency locks,
    /// and return `(layer, sequence, demand_slots)` for O(1) promotion or
    /// release in `Directory`.
    pub(crate) fn release_transport(self, _transport: &ExpertTransport) -> (usize, u64, Vec<DemandSlot>) {
        let Self { batch, .. } = self;
        batch.retire_completed()
    }

    /// Explicitly abort this landing before `DecodeBack` was submitted.
    pub(crate) fn abort(
        mut self,
        reason: &str,
        wait_dma: impl FnOnce(u64) -> Result<bool, String>,
        abort_slots: &mut impl FnMut(&[(usize, usize)]),
    ) {
        self.batch.joined_syncs.append(&mut self.syncs);
        self.batch.abort(reason, wait_dma, abort_slots);
    }

    /// Quarantine this landing without freeing uncertain slots (e.g. when
    /// `DecodeBack` compute wait failed during refusal).
    pub(crate) fn quarantine(mut self, reason: &str) {
        self.batch.joined_syncs.append(&mut self.syncs);
        self.batch.quarantine(reason);
    }

    #[cfg(test)]
    pub(crate) fn release(self, transport: &ExpertTransport) {
        let _ = self.release_transport(transport);
    }

    #[cfg(test)]
    fn own_transfer_count(&self) -> usize {
        self.batch
            .batches
            .iter()
            .map(|b| match &b.payload {
                DemandBatchPayload::OwnWrite { transfer_ids, .. } => transfer_ids.len(),
                DemandBatchPayload::Joined { .. } => 0,
            })
            .sum()
    }

    #[cfg(test)]
    fn own_slot_count(&self) -> usize {
        self.batch
            .batches
            .iter()
            .map(|b| match &b.payload {
                DemandBatchPayload::OwnWrite { slots, .. } => slots.len(),
                DemandBatchPayload::Joined { .. } => 0,
            })
            .sum()
    }
}

/// Host-tier prefill copy plan published by `Watcher` (`Scratch::fix_rounds`)
/// directly on the host — consumed by `StepSubmitter` right before `Pass` with
/// zero `cuEventSynchronize` drains.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PrefillPlan {
    /// `(tier_record, scratch_slot)` pairs to copy `HostTier -> Scratch`.
    pub listed: Vec<(i32, i32)>,
    /// All `(scratch_slot, seed_offset)` pairs from `SeedList` (non-empty only
    /// on a prompt group's last step). Filtered against the union of `listed`
    /// across the entire merge set in [`merge_set_seeded_tiered`] so experts
    /// filled from the host tier by an earlier chunk and seeded by the last
    /// chunk are always propagated `Scratch -> Bank` in stream order after H2D.
    pub seeds: Vec<(i32, i32)>,
}

/// Filter all `SeedList` pairs across a merge set's [`PrefillPlan`]s to those
/// whose `scratch_slot` is populated from the host tier by ANY chunk in the
/// merge set (not just the last chunk that owns `SeedList`).
pub(crate) fn merge_set_seeded_tiered(plans: &[PrefillPlan]) -> Vec<(i32, i32)> {
    let tiered_slots: HashSet<i32> = plans
        .iter()
        .flat_map(|plan| plan.listed.iter().map(|&(_, slot)| slot))
        .collect();
    if tiered_slots.is_empty() {
        return Vec::new();
    }
    plans
        .iter()
        .flat_map(|plan| plan.seeds.iter().copied())
        .filter(|(slot, _)| tiered_slots.contains(slot))
        .collect()
}

const SPLIT_DMA_WARMUP_CALLS: u32 = 8;
const SPLIT_DMA_COOLDOWN_STEPS: u32 = 16;
const SPLIT_DMA_STAGGER_SLOTS: u32 = 16;
const SPLIT_DMA_STAGGER_STRIDE: u32 = 1;

/// Row-width threshold separating narrow decode (`M = 1..3`) from wide
/// speculative verify bursts (`M >= 4`) in [`DecodeDmaGate`]. Separate lanes
/// ensure an `M = 1..2` warmup cooldown never strands an `M = 4..16` verify
/// burst on slower SM zero-copy PCIe.
const WIDE_VERIFY_ROWS: u64 = 4;

/// Physically derived minimum host-tier miss bytes per layer call for
/// splitting a decode layer into `DecodeFront -> flush_and_wait -> DMA ->
/// DecodeBack` vs running pipelined `RunStage::Whole`.
///
/// Derived purely from live machine measurements with zero hardcoded constants:
/// - `t_split_seconds = boundary_seconds + dma_latency_seconds` where
///   `boundary_seconds` is `LaunchPrices::step_round_trip_seconds()` (the
///   measured host-sync boundary cost) and `dma_latency_seconds` is
///   `TierCopyCosts::transfer.latency_seconds()` (the measured H2D batch
///   launch latency).
/// - `inv_b_zerocopy = link_weights_seconds / record_bytes` (measured by
///   `ExpertLines::gpu.link.weights`).
/// - `inv_b_split = dma_seconds_per_byte + resident_weights_seconds / record_bytes`
///   (measured by `TierCopyCosts::transfer` + `ExpertLines::gpu.resident.weights`).
///
/// `min_bytes = t_split_seconds / (inv_b_zerocopy - inv_b_split)`
///
/// Clamped safely to `u64::MAX` whenever `inv_b_split >= inv_b_zerocopy`
/// (DMA + resident read is not faster per byte than SM zero-copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitDmaCrossover {
    pub min_bytes: u64,
}

impl Default for SplitDmaCrossover {
    /// Conservative pre-measurement crossover (`1 B`): before link rates are
    /// measured, any non-zero host-tier miss stays on hardware DMA and only
    /// zero-miss layers gate onto pipelined `RunStage::Whole`.
    fn default() -> Self {
        Self { min_bytes: 1 }
    }
}

impl SplitDmaCrossover {
    /// Derive the miss-byte break-even crossover from measured per-byte times
    /// (`inv_b_split_s_per_byte` for async DMA + resident read,
    /// `inv_b_zerocopy_s_per_byte` for SM zero-copy) and measured fixed
    /// split-boundary overhead (`t_split_seconds`).
    pub(crate) fn derive(
        inv_b_split_s_per_byte: f64,
        inv_b_zerocopy_s_per_byte: f64,
        t_split_seconds: f64,
    ) -> Self {
        if !inv_b_split_s_per_byte.is_finite()
            || !inv_b_zerocopy_s_per_byte.is_finite()
            || inv_b_split_s_per_byte <= 0.0
            || inv_b_zerocopy_s_per_byte <= 0.0
            || inv_b_split_s_per_byte >= inv_b_zerocopy_s_per_byte
        {
            return Self { min_bytes: u64::MAX };
        }
        let delta_s_per_byte = inv_b_zerocopy_s_per_byte - inv_b_split_s_per_byte;
        if !delta_s_per_byte.is_finite() || delta_s_per_byte <= 0.0 {
            return Self { min_bytes: u64::MAX };
        }
        let overhead_s = if t_split_seconds.is_finite() && t_split_seconds > 0.0 {
            t_split_seconds
        } else {
            0.0
        };
        let bytes = overhead_s / delta_s_per_byte;
        let min_bytes = if !bytes.is_finite() || bytes >= u64::MAX as f64 {
            u64::MAX
        } else {
            (bytes.round() as u64).max(1)
        };
        Self { min_bytes }
    }

    /// Derive the crossover directly from measured [`crate::assessment::TierCopyCosts`],
    /// per-layer [`magnitude_cost_model::ExpertLines`], and optional
    /// [`magnitude_cost_model::LaunchPrices`].
    pub(crate) fn from_rates(
        copy: &crate::assessment::TierCopyCosts,
        lines: &magnitude_cost_model::ExpertLines,
        launch: Option<&magnitude_cost_model::LaunchPrices>,
    ) -> Self {
        let record_bytes = lines.host.record_bytes as f64;
        if record_bytes <= 0.0 {
            return Self::default();
        }
        let link_s = lines.gpu.link.weights.get() * 1e-9;
        let resident_s = lines.gpu.resident.weights.get() * 1e-9;
        let inv_b_zerocopy = link_s / record_bytes;
        let inv_b_split = copy.transfer.seconds_per_byte() + resident_s / record_bytes;
        let boundary_s = launch.map_or(0.0, magnitude_cost_model::LaunchPrices::step_round_trip_seconds);
        let t_split_s = copy.transfer.latency_seconds() + boundary_s;
        Self::derive(inv_b_split, inv_b_zerocopy, t_split_s)
    }
}

struct GateLane {
    /// Exponential moving average of un-staged host-tier miss bytes per call
    /// (alpha = 1/4: `ewma = (ewma * 3 + sample) / 4`).
    ewma_bytes: AtomicU64,
    /// Derived minimum miss-byte crossover for this lane (`SplitDmaCrossover`).
    min_bytes: AtomicU64,
    /// Number of observed split-decode calls on this lane.
    calls: AtomicU32,
    /// Remaining pipelined `RunStage::Whole` steps before probing split DMA
    /// again to detect working-set shifts.
    cooldown: AtomicU32,
    /// Whether this lane is currently gated onto pipelined `RunStage::Whole`.
    gated: AtomicBool,
    /// Per-layer initial cooldown phase offset so layers never exit cooldown
    /// on the exact same step and cause a synchronous probe storm.
    phase_offset: u32,
}

impl GateLane {
    fn new(phase_offset: u32, min_bytes: u64) -> Self {
        Self {
            ewma_bytes: AtomicU64::new(0),
            min_bytes: AtomicU64::new(min_bytes),
            calls: AtomicU32::new(0),
            cooldown: AtomicU32::new(0),
            gated: AtomicBool::new(false),
            phase_offset,
        }
    }

    fn set_min_bytes(&self, min_bytes: u64) {
        self.min_bytes.store(min_bytes, Ordering::Relaxed);
    }

    fn is_splitting(&self) -> bool {
        !self.gated.load(Ordering::Relaxed)
    }

    fn should_split(&self) -> bool {
        let calls = self.calls.load(Ordering::Relaxed);
        if calls < SPLIT_DMA_WARMUP_CALLS {
            return true;
        }
        let cd = self.cooldown.load(Ordering::Relaxed);
        if cd > 0 {
            self.cooldown.store(cd - 1, Ordering::Relaxed);
            return false;
        }
        true
    }

    fn observe(&self, miss_bytes: u64) {
        let min_bytes = self.min_bytes.load(Ordering::Relaxed);
        let prev_calls = self.calls.fetch_add(1, Ordering::Relaxed);
        let ewma = if prev_calls == 0 {
            miss_bytes
        } else {
            let old = self.ewma_bytes.load(Ordering::Relaxed);
            (old.saturating_mul(3).saturating_add(miss_bytes)) / 4
        };
        self.ewma_bytes.store(ewma, Ordering::Relaxed);
        if prev_calls + 1 >= SPLIT_DMA_WARMUP_CALLS && ewma < min_bytes && miss_bytes < min_bytes {
            let was_gated = self.gated.swap(true, Ordering::Relaxed);
            let cd = if was_gated {
                SPLIT_DMA_COOLDOWN_STEPS
            } else {
                SPLIT_DMA_COOLDOWN_STEPS + self.phase_offset
            };
            self.cooldown.store(cd, Ordering::Relaxed);
        } else {
            self.gated.store(false, Ordering::Relaxed);
            self.cooldown.store(0, Ordering::Relaxed);
        }
    }
}

/// Lock-free per-layer row-aware adaptive gate for split M7 decode DMA.
struct DecodeDmaGate {
    draft: bool,
    narrow: GateLane,
    wide: GateLane,
}

impl DecodeDmaGate {
    fn for_layer(layer: usize, draft: bool) -> Self {
        Self::for_layer_with_crossover(layer, draft, SplitDmaCrossover::default())
    }

    fn for_layer_with_crossover(layer: usize, draft: bool, crossover: SplitDmaCrossover) -> Self {
        let phase_offset = ((layer as u32) % SPLIT_DMA_STAGGER_SLOTS) * SPLIT_DMA_STAGGER_STRIDE;
        Self {
            draft,
            narrow: GateLane::new(phase_offset, crossover.min_bytes),
            wide: GateLane::new(phase_offset, crossover.min_bytes),
        }
    }

    fn set_crossover(&self, crossover: SplitDmaCrossover) {
        self.narrow.set_min_bytes(crossover.min_bytes);
        self.wide.set_min_bytes(crossover.min_bytes);
    }

    fn lane(&self, rows: u64) -> &GateLane {
        if rows >= WIDE_VERIFY_ROWS {
            &self.wide
        } else {
            &self.narrow
        }
    }

    fn is_splitting(&self) -> bool {
        if self.draft {
            return false;
        }
        let narrow_calls = self.narrow.calls.load(Ordering::Relaxed);
        let wide_calls = self.wide.calls.load(Ordering::Relaxed);
        match (narrow_calls > 0, wide_calls > 0) {
            (true, false) => self.narrow.is_splitting(),
            (false, true) => self.wide.is_splitting(),
            (false, false) => true,
            (true, true) => self.wide.is_splitting() || self.narrow.is_splitting(),
        }
    }

    fn should_split(&self, rows: u64) -> bool {
        if self.draft {
            return false;
        }
        self.lane(rows).should_split()
    }

    fn observe(&self, rows: u64, miss_bytes: u64) {
        if self.draft {
            return;
        }
        self.lane(rows).observe(miss_bytes);
    }
}

/// One ring slot of incremental Phase B landings for a layer, matching the
/// layer's `ExchangeTable` flight ring (`sequence % slots`).
#[derive(Clone, Default)]
struct FileReadySlot {
    sequence: u64,
    ready: Vec<(usize, i32)>,
}

/// On-demand decode DMA resources owned by [`ExpertTransport`].
pub(in crate::experts) struct DecodeDmaState {
    pub transfer: Transfer,
    pub host_stores: Arc<Stores>,
    pub bank_stores: Arc<Stores>,
    pub sources: Tensor,
    pub sources_words: Words,
    pub selected: usize,
    layer_gates: Vec<DecodeDmaGate>,
    /// Per-layer file exchange tables (`routed_route_streamed` -> `Watcher`),
    /// awaited host-side in Phase B of `land_decode_misses` concurrently with
    /// Phase A PCIe DMA of already-in-RAM misses.
    pub files: Vec<super::ExchangeTable>,
    /// Per-layer ring of newly landed Phase B `(choice, source)` pairs posted
    /// by `Answer::publish` as each absent expert finishes NVMe read + host
    /// conversion, drained incrementally by Phase B of `land_decode_misses`.
    file_ready: Vec<Mutex<Vec<FileReadySlot>>>,
    /// Shared watcher event channel woken when `Answer::publish` / `decode::fail`
    /// answers a file choice or slot and notified immediately when
    /// `land_decode_misses` sees a pending file request.
    pub events: Arc<super::events::Events>,
}

impl DecodeDmaState {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::experts) fn new(
        transfer: Transfer,
        host_stores: Arc<Stores>,
        bank_stores: Arc<Stores>,
        sources: Tensor,
        sources_words: Words,
        selected: usize,
        draft_layers: &[bool],
        files: Vec<super::ExchangeTable>,
        events: Arc<super::events::Events>,
    ) -> Self {
        let file_ready = files
            .iter()
            .map(|file| {
                let slots = usize::try_from(file.table().slots()).expect("file table slots fit usize");
                Mutex::new(vec![FileReadySlot::default(); slots])
            })
            .collect();
        Self {
            transfer,
            host_stores,
            bank_stores,
            sources,
            sources_words,
            selected,
            layer_gates: draft_layers
                .iter()
                .enumerate()
                .map(|(layer, &draft)| DecodeDmaGate::for_layer(layer, draft))
                .collect(),
            files,
            file_ready,
            events,
        }
    }

    fn drain_file_ready(&self, layer: usize, sequence: u64) -> Vec<(usize, i32)> {
        let Some(layer_slots) = self.file_ready.get(layer) else {
            return Vec::new();
        };
        let mut slots = layer_slots.lock().expect("decode file_ready");
        let slot_idx = (sequence as usize) % slots.len();
        let slot = &mut slots[slot_idx];
        if slot.sequence != sequence || slot.ready.is_empty() {
            return Vec::new();
        }
        std::mem::take(&mut slot.ready)
    }
}

#[derive(Clone)]
enum TransferState {
    /// Registered by `LinkMover` before `Copies::issue` has popped it.
    Queued,
    /// Enqueued on the copy engine with a cross-stream completion handle.
    InFlight { sync: Option<TransferSync> },
    /// Copy completed on the copy engine; destination bytes are valid.
    Landed,
    /// Enqueue or execution faulted; destination slot is quarantined.
    Faulted,
}

struct TransferEntry {
    key: ExpertKey,
    source: SourceRef,
    dest: DestSlot,
    hint_owner_active: bool,
    consumers: HashSet<u64>,
    state: TransferState,
}

impl TransferEntry {
    fn dma_pending(&self) -> bool {
        matches!(self.state, TransferState::Queued | TransferState::InFlight { .. })
    }

    fn is_joinable(&self) -> bool {
        matches!(
            self.state,
            TransferState::InFlight { sync: Some(_) } | TransferState::Landed
        )
    }

    fn can_retire(&self) -> bool {
        !self.hint_owner_active
            && self.consumers.is_empty()
            && matches!(self.state, TransferState::Landed)
    }
}

#[derive(Default)]
struct Registry {
    next_transfer: u64,
    next_consumer: u64,
    entries: HashMap<TransferId, TransferEntry>,
    /// Active transfer currently occupying `DestSlot`.
    by_dest: HashMap<DestSlot, TransferId>,
    /// Active joinable/tracked transfers indexed by `(group, tier_record)`.
    by_tier: HashMap<(usize, usize), Vec<TransferId>>,
    /// Per `(group, tier_record)` monotonic generation counter.
    generations: HashMap<(usize, usize), u64>,
    /// Destination slots quarantined after a partial enqueue or transfer fault.
    quarantined: HashSet<DestSlot>,
    /// Source/destination backing tensors retained for quarantined transfers
    /// whose hardware completion is unproven, kept alive until `ExpertTransport`
    /// drops at device teardown.
    quarantined_backing: Vec<Tensor>,
    /// Replaced/retired `TransferSync` (`Arc<seismic_cuda::Event>`) handles
    /// deferred to drop OUTSIDE both `self.registry` mutex and `Residency` /
    /// `NoBlockingAllocationGuard` (dropping a final `Event` calls
    /// `cuEventDestroy_v2`, which on CUDA driver 580 can block behind a
    /// GPU-spinning `routed_land` kernel and deadlock if `self.registry` or
    /// `Residency` is held).
    deferred_syncs: Vec<TransferSync>,
}

impl Registry {
    fn alloc_transfer_id(&mut self) -> TransferId {
        self.next_transfer += 1;
        TransferId(self.next_transfer)
    }

    fn alloc_consumer_id(&mut self) -> u64 {
        self.next_consumer += 1;
        self.next_consumer
    }

    fn source_generation(&self, group: usize, tier_record: usize) -> u64 {
        self.generations.get(&(group, tier_record)).copied().unwrap_or(0)
    }

    fn bump_source_generation(&mut self, group: usize, tier_record: usize) -> u64 {
        let entry = self.generations.entry((group, tier_record)).or_insert(0);
        *entry += 1;
        *entry
    }

    fn defer_state_sync(&mut self, state: TransferState) {
        if let TransferState::InFlight { sync: Some(sync) } = state {
            self.deferred_syncs.push(sync);
        }
    }

    fn set_entry_state(&mut self, id: TransferId, new_state: TransferState) -> bool {
        let Some(entry) = self.entries.get_mut(&id) else {
            return false;
        };
        let old = std::mem::replace(&mut entry.state, new_state);
        self.defer_state_sync(old);
        true
    }

    fn remove_entry(&mut self, id: TransferId) {
        if let Some(entry) = self.entries.remove(&id) {
            if self.by_dest.get(&entry.dest) == Some(&id) {
                self.by_dest.remove(&entry.dest);
            }
            if let Some(ids) = self.by_tier.get_mut(&(entry.key.group, entry.source.tier_record)) {
                ids.retain(|&existing| existing != id);
                if ids.is_empty() {
                    self.by_tier.remove(&(entry.key.group, entry.source.tier_record));
                }
            }
            self.defer_state_sync(entry.state);
        }
    }

    fn can_reuse_slot(&self, dest: DestSlot) -> bool {
        if self.quarantined.contains(&dest) {
            return false;
        }
        let Some(&id) = self.by_dest.get(&dest) else {
            return true;
        };
        let Some(entry) = self.entries.get(&id) else {
            return true;
        };
        !entry.dma_pending() && entry.consumers.is_empty() && !entry.hint_owner_active
    }
}

/// Unique monotonic identifier for one generation-scoped `ScratchRing` +
/// `WindowTable` + producer-pin reservation. Never reused across pass sets,
/// layers, or prompt groups, so late or duplicate completions from an older
/// generation can never release a newer generation's slots or `WindowTable`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ScratchGenId(pub u64);

/// Unique identifier for one registered GPU reader of a [`ScratchGenId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ReaderId(pub u64);

/// Identifies a prefill `Pass(generation, group, layer, set_start, window)`
/// whose GPU completion retires one reader on its generation-scoped
/// `ScratchRing` + `WindowTable` reservation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PassKey {
    pub generation: ScratchGenId,
    pub group: u64,
    pub layer: usize,
    pub set_start: usize,
    pub window: u64,
}

/// Atomically published readiness descriptor for `(layer, sequence, window)`:
/// the immutable generation token and optional transfer fence (`TransferSync`).
pub(crate) struct PrefillReady {
    pub generation: ScratchGenId,
    pub sync: Option<TransferSync>,
}

/// Lifecycle event for GPU readers of a generation-scoped prefill reservation,
/// delivered in FIFO order from submission threads -> `ExpertTransport` ->
/// `Scratch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PassReaderEvent {
    /// A GPU reader was registered for `key.generation` and is now in flight.
    Registered { key: PassKey, reader: ReaderId },
    /// The registered GPU reader retired cleanly on the device.
    Retired { key: PassKey, reader: ReaderId },
    /// Reader admission for `key.generation` is closed (no further readers will
    /// register; e.g. after submitting its pass(es) or on cancel-before-submit).
    Closed { key: PassKey },
    /// An accepted GPU submission failed completion registration or faulted on
    /// completion wait: quarantine the generation rather than freeing it.
    Quarantined { key: PassKey },
}

#[derive(Default)]
struct PassTracker {
    next_reader: u64,
    events: VecDeque<PassReaderEvent>,
    completions: VecDeque<(PassKey, ReaderId, seismic::NativeGraphCompletion)>,
}

#[derive(Default)]
struct PrefillRendezvous {
    /// Plans published by `Watcher` (`Scratch::fix_rounds`) keyed by
    /// `(layer, sequence, window)`.
    plans: HashMap<(usize, u64, u64), PrefillPlan>,
    /// Atomically published `(ScratchGenId, Option<TransferSync>)` descriptors
    /// keyed by `(layer, sequence, window)`.
    syncs: HashMap<(usize, u64, u64), PrefillReady>,
    /// Steps that failed before round 0 (`scratch::fail`).
    failed_steps: HashMap<u64, String>,
    /// Generations poisoned by a refused partial/final prefill DMA batch:
    /// `publish_prefill_sync` ignores them and `take_prefill_sync` rejects them.
    poisoned_generations: HashMap<ScratchGenId, String>,
    /// Fatal serving error (`Shared::end_serving`).
    ended: Option<String>,
}

/// Unified Copy-Engine-First `ExpertTransport`.
pub(crate) struct ExpertTransport {
    registry: Mutex<Registry>,
    prefill: Mutex<PrefillRendezvous>,
    prefill_ready: Condvar,
    /// In-flight prefill `Pass(L)` reader registrations, admission closes,
    /// quarantines, and GPU completions tracked so `Requests::serve` retires
    /// each generation only after its last registered GPU reader finishes.
    pass_tracker: Mutex<PassTracker>,
    decode: Option<DecodeDmaState>,
    /// Number of active `land_decode_misses` flights currently in flight
    /// between `DecodeFront` and `DecodeBack` release/completion.
    demand_in_flight: AtomicUsize,
    /// Diagnostic counters (lock-free).
    transfers_issued: AtomicU64,
    demand_joins: AtomicU64,
    prefill_dma_batches: AtomicU64,
    prefill_dma_skipped: AtomicU64,
}

impl std::fmt::Debug for ExpertTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertTransport")
            .field("transfers_issued", &self.transfers_issued.load(Ordering::Relaxed))
            .field("demand_joins", &self.demand_joins.load(Ordering::Relaxed))
            .field("prefill_dma_batches", &self.prefill_dma_batches.load(Ordering::Relaxed))
            .field("prefill_dma_skipped", &self.prefill_dma_skipped.load(Ordering::Relaxed))
            .finish()
    }
}

impl ExpertTransport {
    /// Create a transport without on-demand decode DMA (used on unified memory,
    /// whole banks, non-CUDA backends, and unit tests).
    pub(crate) fn new_registry_only() -> Arc<Self> {
        Arc::new(Self {
            registry: Mutex::default(),
            prefill: Mutex::default(),
            prefill_ready: Condvar::new(),
            pass_tracker: Mutex::default(),
            decode: None,
            demand_in_flight: AtomicUsize::new(0),
            transfers_issued: AtomicU64::new(0),
            demand_joins: AtomicU64::new(0),
            prefill_dma_batches: AtomicU64::new(0),
            prefill_dma_skipped: AtomicU64::new(0),
        })
    }

    /// Create a transport with optional M7 on-demand decode DMA state.
    pub(in crate::experts) fn new(decode: Option<DecodeDmaState>) -> Arc<Self> {
        Arc::new(Self {
            registry: Mutex::default(),
            prefill: Mutex::default(),
            prefill_ready: Condvar::new(),
            pass_tracker: Mutex::default(),
            decode,
            demand_in_flight: AtomicUsize::new(0),
            transfers_issued: AtomicU64::new(0),
            demand_joins: AtomicU64::new(0),
            prefill_dma_batches: AtomicU64::new(0),
            prefill_dma_skipped: AtomicU64::new(0),
        })
    }

    /// Update each streamed layer's [`SplitDmaCrossover`] from live measured
    /// link copy costs (`TierCopyCosts`), per-layer `ExpertLines`, and
    /// optional `LaunchPrices` (`step_round_trip_seconds`).
    pub(crate) fn update_decode_dma_crossover(
        &self,
        copy: &crate::assessment::TierCopyCosts,
        lines: &[magnitude_cost_model::ExpertLines],
        launch: Option<&magnitude_cost_model::LaunchPrices>,
    ) {
        let Some(dma) = &self.decode else {
            return;
        };
        let mut first = None;
        for (layer, gate) in dma.layer_gates.iter().enumerate() {
            let Some(layer_lines) = lines.get(layer) else {
                continue;
            };
            let crossover = SplitDmaCrossover::from_rates(copy, layer_lines, launch);
            gate.set_crossover(crossover);
            if layer == 0 {
                first = Some(crossover);
            }
        }
        if let Some(c) = first {
            tracing::info!(
                "derived SplitDmaCrossover (layer 0): min_bytes={} ({:.2} MiB)",
                c.min_bytes,
                c.min_bytes as f64 / (1024.0 * 1024.0),
            );
        }
    }

    /// Whether M7 on-demand decode DMA is active on this transport.
    pub(crate) fn has_decode_dma(&self) -> bool {
        self.decode.is_some()
    }

    /// Record newly landed Phase B choices for `(layer, sequence)` so
    /// `land_decode_misses` can stream their H2D copies immediately without
    /// waiting for straggler NVMe reads in the same layer.
    ///
    /// Returns `true` when decode DMA is active for this non-draft layer and
    /// the caller should wake `events` even before the final `FILE_ANSWER`.
    pub(crate) fn publish_decode_file_ready(
        &self,
        layer: usize,
        sequence: u64,
        choices: &[usize],
        source: i32,
    ) -> bool {
        let Some(dma) = &self.decode else {
            return false;
        };
        if dma.layer_gates.get(layer).is_some_and(|gate| gate.draft) {
            return false;
        }
        let Some(layer_slots) = dma.file_ready.get(layer) else {
            return false;
        };
        let mut slots = layer_slots.lock().expect("decode file_ready");
        let slot_idx = (sequence as usize) % slots.len();
        let slot = &mut slots[slot_idx];
        if slot.sequence < sequence {
            slot.sequence = sequence;
            slot.ready.clear();
        }
        if slot.sequence == sequence {
            slot.ready.extend(choices.iter().map(|&choice| (choice, source)));
        }
        true
    }

    /// Whether any on-demand decode DMA flight (`land_decode_misses`) is
    /// currently active (in flight or awaiting `DecodeBack` completion).
    pub(crate) fn demand_active(&self) -> bool {
        self.demand_in_flight.load(Ordering::Acquire) > 0
    }

    /// Enter a scoped demand flight (used by tests to exercise `demand_active`).
    #[cfg(test)]
    pub(crate) fn enter_demand_flight(self: &Arc<Self>) -> DemandFlightGuard {
        DemandFlightGuard::new(Arc::clone(self))
    }

    /// Whether on-demand decode DMA should split `layer` into
    /// `DecodeFront -> flush_and_wait -> DMA -> DecodeBack` on this step
    /// (returns `false` when the layer's adaptive miss-bytes EWMA is below
    /// its derived [`SplitDmaCrossover`], keeping low-miss layers on pipelined `Whole`).
    pub(crate) fn should_split_decode(&self, _group: usize, layer: usize, rows: u64) -> bool {
        let Some(dma) = &self.decode else {
            return false;
        };
        dma.layer_gates
            .get(layer)
            .is_none_or(|gate| gate.should_split(rows))
    }

    /// Whether streamed layer `layer` is currently splitting decode through M7
    /// async DMA (`DecodeFront -> DMA -> DecodeBack`) rather than gated onto
    /// pipelined `RunStage::Whole` (lock-free, non-mutating, for
    /// `Residency::publish_lines`).
    pub(crate) fn is_splitting_decode(&self, layer: usize) -> bool {
        let Some(dma) = &self.decode else {
            return false;
        };
        dma.layer_gates
            .get(layer)
            .is_some_and(DecodeDmaGate::is_splitting)
    }

    /// Contiguous `[SOURCE_ROWS, rows, K] i32` view of the pinned host-mapped
    /// `DecodeSources` buffer for `DecodeFront` and `DecodeBack`.
    pub(crate) fn decode_sources_view(&self, rows: u64) -> Result<Tensor, String> {
        let dma = self
            .decode
            .as_ref()
            .ok_or("decode_sources_view called without M7 decode DMA active")?;
        let source_rows = magnitude_kernels::constants::SOURCE_ROWS;
        let k = dma.selected as u64;
        let len = source_rows * rows * k;
        dma.sources
            .slice_leading(0, len)
            .and_then(|leading| leading.reshape(&[source_rows, rows, k]))
            .map_err(|error| format!("decode sources view failed: {error}"))
    }

    /// Total physical transfers registered/issued through this transport.
    #[cfg(test)]
    pub(crate) fn transfers_issued(&self) -> u64 {
        self.transfers_issued.load(Ordering::Relaxed)
    }

    /// Total decode demands that joined an existing in-flight or landed hint.
    #[cfg(test)]
    pub(crate) fn demand_joins(&self) -> u64 {
        self.demand_joins.load(Ordering::Relaxed)
    }

    /// Advance the generation for host-tier record `(group, tier_record)` when
    /// its resident expert is evicted or overwritten.
    pub(crate) fn bump_source_generation(&self, group: usize, tier_record: usize) -> u64 {
        self.registry
            .lock()
            .expect("transport registry")
            .bump_source_generation(group, tier_record)
    }

    /// Register a `LinkMover` opportunistic staging hint before posting to
    /// `Copies`. Pins `dest` against reuse until DMA completes and all
    /// consumers (including any demand joiners) release.
    pub(crate) fn register_hint(
        &self,
        key: ExpertKey,
        tier_record: usize,
        dest: DestSlot,
    ) -> TransferId {
        let mut reg = self.registry.lock().expect("transport registry");
        let generation = reg.source_generation(key.group, tier_record);
        let source = SourceRef { tier_record, generation };
        let id = reg.alloc_transfer_id();
        // Replace any stale retired entry on `dest` (an active pinned slot
        // must never be handed to `register_hint`).
        if let Some(&old_id) = reg.by_dest.get(&dest) {
            let can_drop = reg.entries.get(&old_id).is_none_or(TransferEntry::can_retire);
            assert!(
                can_drop,
                "invariant violated: register_hint reused pinned destination slot {dest:?}"
            );
            reg.remove_entry(old_id);
        }
        reg.entries.insert(
            id,
            TransferEntry {
                key,
                source,
                dest,
                hint_owner_active: true,
                consumers: HashSet::new(),
                state: TransferState::Queued,
            },
        );
        reg.by_dest.insert(dest, id);
        reg.by_tier
            .entry((key.group, tier_record))
            .or_default()
            .push(id);
        self.transfers_issued.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Register a hint with an explicit `SourceRef` and immediate `InFlight`
    /// state (used when the copy has already been enqueued with `sync`).
    #[cfg(test)]
    pub(crate) fn register_in_flight_hint(
        &self,
        key: ExpertKey,
        source: SourceRef,
        dest: DestSlot,
        sync: Option<TransferSync>,
    ) -> TransferId {
        let mut reg = self.registry.lock().expect("transport registry");
        assert!(
            reg.can_reuse_slot(dest),
            "invariant violated: destination slot {dest:?} is pinned or quarantined"
        );
        let id = reg.alloc_transfer_id();
        reg.generations.insert((key.group, source.tier_record), source.generation);
        reg.entries.insert(
            id,
            TransferEntry {
                key,
                source,
                dest,
                hint_owner_active: true,
                consumers: HashSet::new(),
                state: TransferState::InFlight { sync },
            },
        );
        reg.by_dest.insert(dest, id);
        reg.by_tier
            .entry((key.group, source.tier_record))
            .or_default()
            .push(id);
        self.transfers_issued.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Attach the hardware `TransferSync` event once `Copies::issue` submits
    /// the staging batch to the copy engine, making it joinable by demand.
    pub(crate) fn attach_sync(&self, id: TransferId, sync: TransferSync) {
        let mut reg = self.registry.lock().expect("transport registry");
        let can_attach = reg
            .entries
            .get(&id)
            .is_some_and(|entry| matches!(entry.state, TransferState::Queued | TransferState::InFlight { .. }));
        if can_attach {
            reg.set_entry_state(id, TransferState::InFlight { sync: Some(sync) });
        } else {
            reg.deferred_syncs.push(sync);
        }
    }

    /// Mark a transfer's hardware DMA complete (`InFlight -> Landed`).
    /// Retires the entry immediately if both hint ownership and all demand
    /// consumers have already detached. Never drops a `TransferSync` under
    /// `self.registry` lock (callers outside `Residency` call
    /// [`Self::drain_deferred_syncs`]).
    pub(crate) fn complete_dma(&self, id: TransferId) {
        let mut reg = self.registry.lock().expect("transport registry");
        let is_faulted = reg
            .entries
            .get(&id)
            .is_none_or(|entry| matches!(entry.state, TransferState::Faulted));
        if !is_faulted {
            reg.set_entry_state(id, TransferState::Landed);
        }
        let retire = reg.entries.get(&id).is_some_and(TransferEntry::can_retire);
        if retire {
            reg.remove_entry(id);
        }
    }

    /// Cancel an unbegun hint (still `Queued` in `Copies` with zero demand
    /// consumers). Returns `true` if removed.
    pub(crate) fn cancel_unbegun_hint(&self, id: TransferId) -> bool {
        let mut reg = self.registry.lock().expect("transport registry");
        let can_cancel = reg.entries.get(&id).is_some_and(|entry| {
            matches!(entry.state, TransferState::Queued) && entry.consumers.is_empty()
        });
        if can_cancel {
            reg.remove_entry(id);
        }
        can_cancel
    }

    /// Atomically cancel unissued own-write demand transfers under ONE registry
    /// lock: detaches and disarms the owner's leases and removes each entry
    /// from all join/destination indexes WITHOUT transitioning through
    /// `TransferState::Landed` (so `join_by_tier_record` can never observe an
    /// untouched destination as `Landed` with `sync = None`).
    pub(crate) fn cancel_unissued_dma(
        &self,
        transfer_ids: &mut Vec<TransferId>,
        owner_leases: &mut Vec<ConsumerLease>,
    ) {
        let mut reg = self.registry.lock().expect("transport registry");
        for lease in owner_leases.drain(..) {
            let (id, consumer_id) = (lease.transfer_id, lease.consumer_id);
            if let Some(entry) = reg.entries.get_mut(&id) {
                entry.consumers.remove(&consumer_id);
            }
            lease.disarm();
        }
        for id in transfer_ids.drain(..) {
            if let Some(entry) = reg.entries.get(&id) {
                assert!(
                    matches!(entry.state, TransferState::InFlight { sync: None }),
                    "cancel_unissued_dma called on non-unissued transfer {id:?}"
                );
                assert!(
                    entry.consumers.is_empty(),
                    "unissued transfer {id:?} (sync=None, non-joinable) must have no other consumers"
                );
                reg.remove_entry(id);
            }
        }
    }

    /// Mark a transfer faulted and quarantine its destination slot so neither
    /// joiners nor future allocations can observe partial/corrupt bytes.
    pub(crate) fn fault_transfer(&self, id: TransferId, _reason: String) {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let dest = if let Some(entry) = reg.entries.get_mut(&id) {
            entry.hint_owner_active = false;
            Some(entry.dest)
        } else {
            None
        };
        if let Some(dest) = dest {
            reg.set_entry_state(id, TransferState::Faulted);
            reg.quarantined.insert(dest);
        }
    }

    /// Quarantine an own-write batch whose hardware completion is unproven:
    /// faults every transfer, quarantines its destination slot, disarms owner
    /// leases (keeping consumer pins live in `Registry` without `Arc` cycles),
    /// and retains source/destination backing tensors until `ExpertTransport`
    /// teardown. Non-blocking and panic-free (safe to call from `Drop`).
    fn quarantine_uncertain_own(
        &self,
        transfer_ids: &mut Vec<TransferId>,
        owner_leases: &mut Vec<ConsumerLease>,
        backing: &mut Vec<Tensor>,
        _reason: &str,
    ) {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        for id in transfer_ids.drain(..) {
            let dest = if let Some(entry) = reg.entries.get_mut(&id) {
                entry.hint_owner_active = false;
                Some(entry.dest)
            } else {
                None
            };
            if let Some(dest) = dest {
                reg.set_entry_state(id, TransferState::Faulted);
                reg.quarantined.insert(dest);
            }
        }
        for lease in owner_leases.drain(..) {
            lease.disarm();
        }
        reg.quarantined_backing.append(backing);
    }

    /// Defer `TransferSync` handles to be dropped later outside locks and
    /// callback threads. Non-blocking and panic-free (safe to call from `Drop`).
    fn defer_syncs(&self, mut syncs: Vec<TransferSync>) {
        if syncs.is_empty() {
            return;
        }
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.deferred_syncs.append(&mut syncs);
    }

    /// Wait on the decode transfer stream for `value` (used when explicitly
    /// aborting a `BeforeBack` `DecodeLanding` outside all residency locks).
    pub(crate) fn wait_decode_dma(&self, value: u64, timeout: std::time::Duration) -> Result<bool, String> {
        let Some(dma) = &self.decode else {
            return Err("wait_decode_dma called without decode DMA state".into());
        };
        dma.transfer.wait(value, timeout).map_err(|e| e.to_string())
    }

    #[cfg(test)]
    pub(crate) fn quarantined_backing_len(&self) -> usize {
        self.registry
            .lock()
            .expect("transport registry")
            .quarantined_backing
            .len()
    }

    /// Drop retired `TransferSync` (`Arc<seismic_cuda::Event>`) handles OUTSIDE
    /// both `self.registry` and `Residency` / `NoBlockingAllocationGuard`.
    pub(crate) fn drain_deferred_syncs(&self) {
        seismic::memory::assert_blocking_allocation_allowed("ExpertTransport::drain_deferred_syncs");
        let syncs = {
            let mut reg = self.registry.lock().expect("transport registry");
            std::mem::take(&mut reg.deferred_syncs)
        };
        drop(syncs);
    }

    #[cfg(test)]
    pub(crate) fn deferred_syncs_len(&self) -> usize {
        self.registry
            .lock()
            .expect("transport registry")
            .deferred_syncs
            .len()
    }

    /// Quarantine a set of destination slots after a partial enqueue fault.
    #[cfg(test)]
    pub(crate) fn quarantine_slots(&self, slots: &[DestSlot]) {
        let mut reg = self.registry.lock().expect("transport registry");
        for &dest in slots {
            reg.quarantined.insert(dest);
        }
    }

    /// Whether `dest` is currently quarantined due to a transfer/enqueue fault.
    #[cfg(test)]
    pub(crate) fn is_quarantined(&self, dest: DestSlot) -> bool {
        self.registry
            .lock()
            .expect("transport registry")
            .quarantined
            .contains(&dest)
    }

    /// Clear quarantine on `dest` after device drain / recovery.
    #[cfg(test)]
    pub(crate) fn clear_quarantine(&self, dest: DestSlot) {
        let mut reg = self.registry.lock().expect("transport registry");
        reg.quarantined.remove(&dest);
        if let Some(&id) = reg.by_dest.get(&dest) {
            if reg
                .entries
                .get(&id)
                .is_some_and(|entry| matches!(entry.state, TransferState::Faulted) && entry.consumers.is_empty())
            {
                reg.remove_entry(id);
            }
        }
    }

    /// Called by `LinkMover` / `Directory::drain_staging_with` when a staging
    /// slot's target step has been served and the directory wants to return
    /// `dest` to `Record::StageIdle`.
    ///
    /// Detaches hint ownership, and returns `true` ONLY when DMA is complete,
    /// zero demand consumers remain pinned to `dest`, and `dest` is not
    /// quarantined.
    pub(crate) fn try_release_hint_slot(&self, dest: DestSlot) -> bool {
        let mut reg = self.registry.lock().expect("transport registry");
        if reg.quarantined.contains(&dest) {
            if let Some(&id) = reg.by_dest.get(&dest) {
                if let Some(entry) = reg.entries.get_mut(&id) {
                    entry.hint_owner_active = false;
                }
            }
            return false;
        }
        let Some(&id) = reg.by_dest.get(&dest) else {
            return true;
        };
        let Some(entry) = reg.entries.get_mut(&id) else {
            return true;
        };
        entry.hint_owner_active = false;
        if entry.can_retire() {
            reg.remove_entry(id);
            true
        } else {
            false
        }
    }

    /// Whether `dest` can be safely reused by a resize or slot allocator
    /// (zero pending DMA, zero consumers, not quarantined, hint owner gone).
    pub(crate) fn can_reuse_slot(&self, dest: DestSlot) -> bool {
        self.registry
            .lock()
            .expect("transport registry")
            .can_reuse_slot(dest)
    }

    /// Evaluate `f` with a batch slot-reuse predicate under a single
    /// `registry` mutex acquisition (avoids per-slot lock churn when scanning
    /// candidate slots in `Directory`).
    pub(crate) fn with_slot_reuse<R>(&self, f: impl FnOnce(&mut dyn FnMut(DestSlot) -> bool) -> R) -> R {
        let reg = self.registry.lock().expect("transport registry");
        let mut check = |dest: DestSlot| reg.can_reuse_slot(dest);
        f(&mut check)
    }

    /// Detach a single demand consumer without cancelling pending DMA or
    /// disturbing any other consumer on the same transfer.
    fn detach_consumer(&self, id: TransferId, consumer_id: u64) {
        let mut reg = self.registry.lock().expect("transport registry");
        let retire = if let Some(entry) = reg.entries.get_mut(&id) {
            entry.consumers.remove(&consumer_id);
            entry.can_retire()
        } else {
            false
        };
        if retire {
            reg.remove_entry(id);
        }
    }

    /// Try to join an active in-flight or landed transfer matching
    /// `(key, source)` (used by tests and callers holding exact `ExpertKey`).
    #[cfg(test)]
    pub(crate) fn join_by_key(
        self: &Arc<Self>,
        key: ExpertKey,
        source: SourceRef,
    ) -> Option<JoinedTransfer> {
        let mut reg = self.registry.lock().expect("transport registry");
        let id = reg
            .by_tier
            .get(&(key.group, source.tier_record))?
            .iter()
            .copied()
            .find(|id| {
                reg.entries.get(id).is_some_and(|entry| {
                    entry.key == key
                        && entry.source == source
                        && !reg.quarantined.contains(&entry.dest)
                        && matches!(entry.state, TransferState::InFlight { .. } | TransferState::Landed)
                })
            })?;
        let consumer_id = reg.alloc_consumer_id();
        let entry = reg.entries.get_mut(&id)?;
        entry.consumers.insert(consumer_id);
        let dest = entry.dest;
        let sync = match &entry.state {
            TransferState::InFlight { sync } => sync.clone(),
            TransferState::Landed => None,
            TransferState::Queued | TransferState::Faulted => unreachable!(),
        };
        self.demand_joins.fetch_add(1, Ordering::Relaxed);
        Some(JoinedTransfer {
            dest,
            sync,
            lease: ConsumerLease {
                transport: Arc::clone(self),
                transfer_id: id,
                consumer_id,
                active: true,
            },
        })
    }

    /// Try to join an active `InFlight { sync: Some(_) }` or `Landed` transfer
    /// for `(group, layer, tier_record)` at its current source generation.
    pub(crate) fn join_by_tier_record(
        self: &Arc<Self>,
        group: usize,
        layer: usize,
        tier_record: usize,
    ) -> Option<JoinedTransfer> {
        let mut reg = self.registry.lock().expect("transport registry");
        let generation = reg.source_generation(group, tier_record);
        let id = reg
            .by_tier
            .get(&(group, tier_record))?
            .iter()
            .copied()
            .find(|id| {
                reg.entries.get(id).is_some_and(|entry| {
                    entry.key.group == group
                        && entry.key.layer == layer
                        && entry.source == SourceRef { tier_record, generation }
                        && !reg.quarantined.contains(&entry.dest)
                        && entry.is_joinable()
                })
            })?;
        let consumer_id = reg.alloc_consumer_id();
        let entry = reg.entries.get_mut(&id)?;
        entry.consumers.insert(consumer_id);
        let dest = entry.dest;
        let sync = match &entry.state {
            TransferState::InFlight { sync } => sync.clone(),
            TransferState::Landed => None,
            TransferState::Queued | TransferState::Faulted => unreachable!(),
        };
        self.demand_joins.fetch_add(1, Ordering::Relaxed);
        Some(JoinedTransfer {
            dest,
            sync,
            lease: ConsumerLease {
                transport: Arc::clone(self),
                transfer_id: id,
                consumer_id,
                active: true,
            },
        })
    }

    /// Reserve an unpinned, unquarantined unified `DemandLanding` slot for a
    /// fresh demand copy under its true `(group, layer, expert)` identity,
    /// returning `(TransferId, ConsumerLease)`.
    pub(in crate::experts) fn begin_demand_transfer(
        self: &Arc<Self>,
        group: usize,
        layer: usize,
        expert: usize,
        tier_record: usize,
        dest: DestSlot,
    ) -> (TransferId, ConsumerLease) {
        let mut reg = self.registry.lock().expect("transport registry");
        let generation = reg.source_generation(group, tier_record);
        let id = reg.alloc_transfer_id();
        let consumer_id = reg.alloc_consumer_id();
        let mut consumers = HashSet::with_capacity(1);
        consumers.insert(consumer_id);
        reg.entries.insert(
            id,
            TransferEntry {
                key: ExpertKey { group, layer, expert },
                source: SourceRef { tier_record, generation },
                dest,
                hint_owner_active: false,
                consumers,
                state: TransferState::InFlight { sync: None },
            },
        );
        reg.by_dest.insert(dest, id);
        reg.by_tier
            .entry((group, tier_record))
            .or_default()
            .push(id);
        self.transfers_issued.fetch_add(1, Ordering::Relaxed);
        (
            id,
            ConsumerLease {
                transport: Arc::clone(self),
                transfer_id: id,
                consumer_id,
                active: true,
            },
        )
    }

    /// Inspect pinned `DecodeSources` written by `DecodeFront` for streamed
    /// layer `layer` over `rows` rows in two overlapped phases:
    ///
    /// - **Phase A (immediate, zero NVMe wait):** for every already-in-RAM
    ///   host-tier miss (`source <= -2`, `tier_record = -2 - source`), join any
    ///   in-flight or landed `LinkMover` (`Promote`/`Stage`) or demand transfer,
    ///   or claim a unified `Record::Free -> Record::DemandLanding` slot via
    ///   `claim_slot` and fire ONE batched `cuMemcpyHtoDAsync` (`copy_synced`)
    ///   immediately so PCIe Gen5 DMA runs concurrently with any pending NVMe
    ///   landing + CPU conversion.
    /// - **Phase B (incremental, overlapped with Phase A PCIe DMA):** when
    ///   `DecodeFront` posted a file request (`file.requested(sequence, 0)`),
    ///   wake `Requests` immediately, drain each newly-landed expert published
    ///   by `Answer::publish` via [`Self::publish_decode_file_ready`] and fire
    ///   its `cuMemcpyHtoDAsync` on the same FIFO DMA stream immediately while
    ///   slower NVMe reads in the same layer are still in flight (promoting any
    ///   CPU-pending `SOURCE_SITE == -2` to GPU DMA since `routed_site_request`
    ///   already ran in `DecodeFront`), then perform an authoritative final
    ///   pass once `file.answered(sequence, 0)` (or serving ends).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn land_decode_misses(
        self: &Arc<Self>,
        group: usize,
        layer: usize,
        sequence: u64,
        rows: u64,
        mut claim_batch: impl FnMut(usize, &[u32], usize) -> Vec<super::directory::DemandSlotClaim>,
        mut abort_slots: impl FnMut(&[(usize, usize)]),
        wake_site: impl FnOnce(),
        is_serving_ended: impl Fn() -> bool,
        record_dma: impl FnOnce(bool, u64, u64, u64),
    ) -> Result<Option<DecodeLanding>, String> {
        let dma = self
            .decode
            .as_ref()
            .ok_or("land_decode_misses called without M7 decode DMA active")?;
        let file = dma.files.get(layer);
        let file_pending = file.is_some_and(|f| f.requested(sequence, 0));
        if file_pending {
            // Wake `Requests` immediately so NVMe demand reads start with zero
            // poll-interval lag while we prepare Phase A DMA.
            dma.events.notify();
        }
        let count = (rows as usize) * dma.selected;
        let site_base = (magnitude_kernels::constants::SOURCE_SITE as usize) * count;
        let mut raw_sources = Vec::with_capacity(count);
        let mut unique_a: Vec<(usize, usize)> = Vec::new(); // (tier_record, hit_count)
        let mut any_sited = false;
        for idx in 0..count {
            let word = dma.sources_words.load(idx);
            raw_sources.push(word);
            if word <= -2 {
                let tier_record = (-2i64 - i64::from(word)) as usize;
                if let Some(entry) = unique_a.iter_mut().find(|(tr, _)| *tr == tier_record) {
                    entry.1 += 1;
                } else {
                    unique_a.push((tier_record, 1));
                }
            }
            if !any_sited && dma.sources_words.load(site_base + idx) >= 0 {
                any_sited = true;
            }
        }
        if any_sited {
            // Wake the CPU site's parked noticer immediately so `expert-compute`
            // overlaps with `cuMemcpyHtoDAsync` without waiting for
            // `NOTICER_PARK_INTERVAL` (100 us).
            wake_site();
        }
        if unique_a.is_empty() && !file_pending {
            if let Some(gate) = dma.layer_gates.get(layer) {
                gate.observe(rows, 0);
            }
            record_dma(true, 0, 0, 0);
            return Ok(None);
        }

        let mut tx = IssuedDemandBatch::new(Arc::clone(self), layer, sequence);

        let region_bytes = dma.bank_stores.region_bytes(group);
        let expert_bytes: u64 = region_bytes.iter().copied().sum();

        let mut resolved: Vec<(usize, usize)> = Vec::with_capacity(unique_a.len());
        let mut bytes_landed: u64 = 0;
        let mut overflow_experts: u64 = 0;

        // Shared helper: join existing transfers or claim unified `Record::Free`
        // slots in ONE batched `claim_batch` call (single `Residency` lock per
        // layer, optionally replenishing `Free` headroom in the same lock),
        // then issue one batched `copy_synced` on `dma.transfer` outside all
        // residency locks.
        let mut replenished = false;
        let mut issue_batch =
            |mut unique: Vec<(usize, usize)>,
             replenish_want: usize,
             resolved: &mut Vec<(usize, usize)>|
             -> Result<(), String> {
                if unique.is_empty() {
                    return Ok(());
                }
                // Prioritize highest row-reference count in case free slots are scarce.
                unique.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let mut backing: Vec<Tensor> =
                    Vec::with_capacity(unique.len() * region_bytes.len() * 2);
                let mut run_bytes: Vec<u64> =
                    Vec::with_capacity(unique.len() * region_bytes.len());
                // `(tier_record, slot, staged)`
                let mut fresh_landed: Vec<(usize, usize, Option<(usize, usize)>)> =
                    Vec::with_capacity(unique.len());
                let mut aborted: Vec<(usize, usize)> = Vec::new();
                let mut to_claim: Vec<u32> = Vec::with_capacity(unique.len());
                for (tier_record, _hits) in unique {
                    if resolved.iter().any(|&(tr, _)| tr == tier_record) {
                        continue;
                    }
                    if let Some(joined) = self.join_by_tier_record(group, layer, tier_record) {
                        resolved.push((tier_record, joined.dest.slot));
                        tx.record_joined(joined.lease, joined.sync);
                        continue;
                    }
                    to_claim.push(tier_record as u32);
                }
                if !to_claim.is_empty() {
                    let want = if replenished { 0 } else { replenish_want };
                    if want > 0 {
                        replenished = true;
                    }
                    let claims = claim_batch(group, &to_claim, want);
                    for (tier_record_u32, claim) in to_claim.into_iter().zip(claims) {
                        let tier_record = tier_record_u32 as usize;
                        match claim {
                            super::directory::DemandSlotClaim::AlreadyHeld { slot, .. } => {
                                resolved.push((tier_record, slot));
                            }
                            super::directory::DemandSlotClaim::Allocated { staged, slot } => {
                                let mut complete = true;
                                let before_backing = backing.len();
                                let before_runs = run_bytes.len();
                                for (region, &bytes) in region_bytes.iter().enumerate() {
                                    if bytes == 0 {
                                        continue;
                                    }
                                    let (Some(src), Some(dst)) = (
                                        dma.host_stores.view(group, region, tier_record),
                                        dma.bank_stores.view(group, region, slot),
                                    ) else {
                                        complete = false;
                                        break;
                                    };
                                    backing.push(src);
                                    backing.push(dst);
                                    run_bytes.push(bytes);
                                }
                                if complete {
                                    fresh_landed.push((tier_record, slot, staged));
                                } else {
                                    backing.truncate(before_backing);
                                    run_bytes.truncate(before_runs);
                                    aborted.push((group, slot));
                                }
                            }
                            super::directory::DemandSlotClaim::Overflow { .. } => {
                                overflow_experts += 1;
                            }
                        }
                    }
                }
                if !aborted.is_empty() {
                    abort_slots(&aborted);
                }
                if fresh_landed.is_empty() {
                    return Ok(());
                }
                let batch_bytes: u64 = run_bytes.iter().copied().sum();
                let mut batch_ids = Vec::with_capacity(fresh_landed.len());
                let mut batch_slots = Vec::with_capacity(fresh_landed.len());
                let mut batch_leases = Vec::with_capacity(fresh_landed.len());
                for &(tier_record, slot, staged) in &fresh_landed {
                    let (target_layer, target_expert) = staged.unwrap_or((layer, tier_record));
                    let (id, lease) = self.begin_demand_transfer(
                        group,
                        target_layer,
                        target_expert,
                        tier_record,
                        DestSlot { group, slot },
                    );
                    batch_ids.push(id);
                    batch_slots.push(DemandSlot { group, slot, staged });
                    batch_leases.push(lease);
                }
                let batch_idx = tx.claim_own_batch(batch_slots, batch_ids, batch_leases, backing);
                tx.begin_issue(batch_idx);
                let result = {
                    let tensors = tx.own_backing(batch_idx);
                    let items: Vec<TransferItem<'_>> = run_bytes
                        .iter()
                        .enumerate()
                        .map(|(i, &bytes)| {
                            TransferItem::Run(TransferRun {
                                source: &tensors[i * 2],
                                source_offset: 0,
                                destination: &tensors[i * 2 + 1],
                                destination_offset: 0,
                                bytes,
                            })
                        })
                        .collect();
                    dma.transfer.copy_synced(&items)
                };
                match result {
                    Ok((value, sync)) => {
                        // Wake `LinkMover` immediately so `Stage(L+1)` queues
                        // onto the transfer stream right after `DMA(L)` and
                        // overlaps with `DecodeBack(L) + DecodeFront(L+1)`.
                        dma.events.notify();
                        tx.mark_submitted(batch_idx, value, Some(sync));
                        for &(tier_record, slot, _) in &fresh_landed {
                            resolved.push((tier_record, slot));
                        }
                        bytes_landed += batch_bytes;
                        Ok(())
                    }
                    Err(error) => {
                        if error.refused_before_issue() {
                            tx.mark_refused_before_issue(batch_idx);
                        }
                        let msg = format!("M7 decode DMA copy failed: {error}");
                        tx.abort(
                            &msg,
                            |value| {
                                dma.transfer
                                    .wait(value, std::time::Duration::from_secs(5))
                                    .map_err(|e| e.to_string())
                            },
                            &mut abort_slots,
                        );
                        Err(msg)
                    }
                }
            };

        let target_free = dma.selected.saturating_mul(if rows >= WIDE_VERIFY_ROWS { 2 } else { 1 });

        // Phase A: fire PCIe DMA immediately for already-in-RAM misses before
        // waiting on any NVMe file answer. When no file answer is pending,
        // replenish `Free` headroom in the same single `Residency` lock.
        let total_unique_a = unique_a.len();
        issue_batch(unique_a, if file_pending { 0 } else { target_free }, &mut resolved)?;
        for (idx, &word) in raw_sources.iter().enumerate() {
            if word <= -2 {
                let tier_record = (-2i64 - i64::from(word)) as usize;
                if let Some(&(_, slot)) = resolved.iter().find(|&&(tr, _)| tr == tier_record) {
                    dma.sources_words.store(idx, slot as i32);
                }
            }
        }

        // Phase B: overlap host-side `FILE_ANSWER` wait with Phase A PCIe DMA
        // and stream newly-landed NVMe misses incrementally over PCIe as each
        // expert finishes NVMe read + host conversion, instead of stalling all
        // Phase B H2D copies behind the slowest straggler in the layer.
        let mut total_unique_b = 0usize;
        if file_pending {
            let file = file.expect("file_pending implies layer has a file table");
            let mut handled_choices = vec![false; count];
            let mut seen_b_tiers: Vec<usize> = Vec::new();
            {
                let mut apply_entries = |entries: &[(usize, i32)],
                                         replenish_want: usize,
                                         handled_choices: &mut [bool],
                                         resolved: &mut Vec<(usize, usize)>|
                 -> Result<(), String> {
                    let mut batch_choices: Vec<(usize, usize)> = Vec::new(); // (choice, tier_record)
                    let mut unique_batch: Vec<(usize, usize)> = Vec::new(); // (tier_record, hit_count)
                    for &(choice, source) in entries {
                        if choice >= count || handled_choices[choice] {
                            continue;
                        }
                        handled_choices[choice] = true;
                        if source == -1 {
                            dma.sources_words.store(choice, -1);
                            dma.sources_words.store(site_base + choice, -1);
                        } else if source >= 0 {
                            dma.sources_words.store(choice, source);
                            if dma.sources_words.load(site_base + choice) == -2 {
                                dma.sources_words.store(site_base + choice, -1);
                            }
                        } else {
                            let tier_record = (-2i64 - i64::from(source)) as usize;
                            if dma.sources_words.load(site_base + choice) == -2 {
                                // `routed_site_request` already ran in `DecodeFront`
                                // and skipped `-2`; route this newly-landed expert to
                                // GPU DMA in Phase B instead.
                                dma.sources_words.store(site_base + choice, -1);
                            }
                            // Store fallback `-2 - tier_record` first so overflow still
                            // resolves via SM zero-copy in `DecodeBack`.
                            dma.sources_words.store(choice, source);
                            batch_choices.push((choice, tier_record));
                            if !seen_b_tiers.contains(&tier_record) {
                                seen_b_tiers.push(tier_record);
                            }
                            if let Some(entry) =
                                unique_batch.iter_mut().find(|(tr, _)| *tr == tier_record)
                            {
                                entry.1 += 1;
                            } else {
                                unique_batch.push((tier_record, 1));
                            }
                        }
                    }
                    if !unique_batch.is_empty() {
                        issue_batch(unique_batch, replenish_want, resolved)?;
                        for &(choice, tier_record) in &batch_choices {
                            if let Some(&(_, slot)) =
                                resolved.iter().find(|&&(tr, _)| tr == tier_record)
                            {
                                dma.sources_words.store(choice, slot as i32);
                            }
                        }
                    }
                    Ok(())
                };

                loop {
                    let seen = dma.events.seen();
                    let finished = file.answered(sequence, 0) || is_serving_ended();
                    let ready = dma.drain_file_ready(layer, sequence);
                    if !ready.is_empty() {
                        apply_entries(
                            &ready,
                            if finished { target_free } else { 0 },
                            &mut handled_choices,
                            &mut resolved,
                        )?;
                    }
                    if finished {
                        break;
                    }
                    if ready.is_empty() {
                        dma.events.wait_at_most(seen, std::time::Duration::from_millis(5));
                    }
                }

                // Authoritative final pass once `FILE_ANSWER` lands (or serving ends):
                // - On `Answer::Answered`, pick up any absent choice not already
                //   dispatched incrementally (e.g. direct table answers).
                // - On `Answer::Failed` / serving ended, clear every absent choice's
                //   `SOURCE_RECORD` and `SOURCE_SITE` to `-1`.
                let answered = matches!(
                    file.answer_of(sequence, 0),
                    Some(super::exchange::Answer::Answered)
                );
                let file_count = usize::try_from(
                    file.load(sequence, 0, magnitude_kernels::constants::FILE_COUNT),
                )
                .unwrap_or(0)
                .min(count);
                if answered {
                    let mut remaining: Vec<(usize, i32)> = Vec::new();
                    for choice in 0..file_count {
                        if handled_choices[choice] {
                            continue;
                        }
                        let entry = magnitude_kernels::constants::FILE_ENTRY(choice as u64);
                        let expert =
                            file.load(sequence, 0, entry + magnitude_kernels::constants::FILE_CHOICE);
                        if expert < 0 {
                            continue;
                        }
                        let source =
                            file.load(sequence, 0, entry + magnitude_kernels::constants::FILE_RECORD);
                        remaining.push((choice, source));
                    }
                    if !remaining.is_empty() {
                        apply_entries(&remaining, target_free, &mut handled_choices, &mut resolved)?;
                    }
                } else {
                    for choice in 0..file_count {
                        let entry = magnitude_kernels::constants::FILE_ENTRY(choice as u64);
                        let expert =
                            file.load(sequence, 0, entry + magnitude_kernels::constants::FILE_CHOICE);
                        if expert < 0 {
                            continue;
                        }
                        dma.sources_words.store(choice, -1);
                        dma.sources_words.store(site_base + choice, -1);
                    }
                }
            }
            total_unique_b = seen_b_tiers.len();
        }

        if !replenished && (overflow_experts > 0 || !resolved.is_empty()) {
            let _ = claim_batch(group, &[], target_free.max(overflow_experts as usize).max(1));
        }

        if let Some(gate) = dma.layer_gates.get(layer) {
            gate.observe(rows, ((total_unique_a + total_unique_b) as u64).saturating_mul(expert_bytes));
        }

        if resolved.is_empty() {
            record_dma(true, 0, 0, overflow_experts);
            return Ok(None);
        }

        record_dma(false, resolved.len() as u64, bytes_landed, overflow_experts);

        Ok(Some(tx.into_landing()))
    }

    // -----------------------------------------------------------------------
    // Producer-enqueued Prefill `HostTier -> Scratch` (+ seeded `Scratch -> Bank`) DMA
    // -----------------------------------------------------------------------

    /// Called by `Watcher` (`Scratch::fix_rounds`) on the host as soon as
    /// round 0 fixes `(TierList, SeedList)` for `(layer, sequence, window)`.
    /// Pure in-memory insert + condvar notify (zero CUDA, zero blocking).
    pub(crate) fn publish_prefill_plan(
        &self,
        layer: usize,
        sequence: u64,
        window: u64,
        plan: PrefillPlan,
    ) {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        prefill.plans.insert((layer, sequence, window), plan);
        self.prefill_ready.notify_all();
    }

    /// Called by `Watcher` (`scratch::fail`) when a step fails before round 0,
    /// waking any submission thread waiting in [`Self::take_prefill_plan`].
    pub(crate) fn fail_prefill_step(&self, sequence: u64, reason: String) {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        prefill.failed_steps.entry(sequence).or_insert(reason);
        self.prefill_ready.notify_all();
    }

    /// Called by `Shared::end_serving` on fatal watcher/device failure so no
    /// prefill waiter can ever hang.
    pub(crate) fn end_serving(&self, reason: String) {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        if prefill.ended.is_none() {
            prefill.ended = Some(reason);
        }
        self.prefill_ready.notify_all();
    }

    /// Called by `Copies::issue` when any partial or final batch of
    /// `generation` is refused: poisons `generation`, fails `sequence` and any
    /// sibling sequences whose already-published syncs belonged to `generation`,
    /// purges those syncs, and wakes waiters.
    pub(crate) fn poison_prefill_generation(
        &self,
        generation: ScratchGenId,
        sequence: u64,
        reason: String,
    ) {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        prefill.poisoned_generations.entry(generation).or_insert_with(|| reason.clone());
        prefill.failed_steps.entry(sequence).or_insert_with(|| reason.clone());
        let mut purged_sequences = Vec::new();
        prefill.syncs.retain(|&(_, seq, _), ready| {
            if ready.generation == generation {
                purged_sequences.push(seq);
                false
            } else {
                true
            }
        });
        for seq in purged_sequences {
            prefill.failed_steps.entry(seq).or_insert_with(|| reason.clone());
        }
        self.prefill_ready.notify_all();
    }

    /// Called by `Copies::issue` (with `Some(sync)`) or `Scratch` (with `None`
    /// when a fill has 0 copies) once a prefill fill batch is submitted on
    /// `Copies`'s transfer stream. Atomically publishes the immutable
    /// `generation` token alongside its readiness dependency (`sync`), unless
    /// `generation` has already been poisoned by a refused partial batch (in
    /// which case `sequence` is failed and waiters are woken).
    pub(crate) fn publish_prefill_sync(
        &self,
        layer: usize,
        sequence: u64,
        window: u64,
        generation: ScratchGenId,
        sync: Option<TransferSync>,
    ) {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        if let Some(reason) = prefill.poisoned_generations.get(&generation).cloned() {
            prefill.failed_steps.entry(sequence).or_insert(reason);
            self.prefill_ready.notify_all();
            return;
        }
        prefill.syncs.insert((layer, sequence, window), PrefillReady { generation, sync });
        self.prefill_ready.notify_all();
    }

    /// Wait on the host condvar (zero `cuEventSynchronize`!) for `Copies` to
    /// publish the atomic [`PrefillReady`] descriptor for
    /// `(layer, sequence, window)`. Rejects poisoned generations.
    pub(crate) fn take_prefill_sync(
        &self,
        layer: usize,
        sequence: u64,
        window: u64,
    ) -> Result<PrefillReady, String> {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        loop {
            if let Some(ready) = prefill.syncs.remove(&(layer, sequence, window)) {
                if let Some(reason) = prefill.poisoned_generations.get(&ready.generation) {
                    return Err(reason.clone());
                }
                if ready.sync.is_some() {
                    self.prefill_dma_batches.fetch_add(1, Ordering::Relaxed);
                } else {
                    self.prefill_dma_skipped.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(ready);
            }
            if let Some(reason) = prefill.failed_steps.get(&sequence) {
                return Err(reason.clone());
            }
            if let Some(reason) = &prefill.ended {
                return Err(reason.clone());
            }
            prefill = self
                .prefill_ready
                .wait(prefill)
                .expect("transport prefill rendezvous");
        }
    }

    /// Register one GPU reader on `key.generation` without closing reader
    /// admission (supports multiple `Pass`es reading the same generation).
    #[allow(dead_code)]
    pub(crate) fn register_pass_reader(
        &self,
        key: PassKey,
        completion: seismic::NativeGraphCompletion,
    ) -> ReaderId {
        let mut tracker = self.pass_tracker.lock().expect("transport pass tracker");
        tracker.next_reader += 1;
        let reader = ReaderId(tracker.next_reader);
        tracker.events.push_back(PassReaderEvent::Registered { key, reader });
        tracker.completions.push_back((key, reader, completion));
        reader
    }

    /// Close reader admission for `key.generation` (e.g. zero readers on
    /// cancel-before-submit, or after the last reader of a multi-pass
    /// generation has been registered).
    pub(crate) fn close_pass_readers(&self, key: PassKey) {
        self.pass_tracker
            .lock()
            .expect("transport pass tracker")
            .events
            .push_back(PassReaderEvent::Closed { key });
    }

    /// Quarantine `key.generation` after an accepted GPU submission whose
    /// completion registration failed or whose wait faulted.
    pub(crate) fn quarantine_pass(&self, key: PassKey) {
        self.pass_tracker
            .lock()
            .expect("transport pass tracker")
            .events
            .push_back(PassReaderEvent::Quarantined { key });
    }

    /// Atomically register the sole GPU `Pass` reader for `key.generation` and
    /// close reader admission in FIFO order (BEFORE completion can retire).
    pub(crate) fn track_pass_completion(
        &self,
        key: PassKey,
        completion: seismic::NativeGraphCompletion,
    ) -> ReaderId {
        let mut tracker = self.pass_tracker.lock().expect("transport pass tracker");
        tracker.next_reader += 1;
        let reader = ReaderId(tracker.next_reader);
        tracker.events.push_back(PassReaderEvent::Registered { key, reader });
        tracker.events.push_back(PassReaderEvent::Closed { key });
        tracker.completions.push_back((key, reader, completion));
        reader
    }

    /// Non-blocking poll of tracked prefill `Pass` reader events and GPU
    /// completions. Drains queued FIFO lifecycle events and completed passes
    /// under a short mutex, then calls `completion.wait()` OUTSIDE all locks so
    /// CUDA event teardown never runs under a mutex or
    /// `NoBlockingAllocationGuard`. On `completion.wait()` failure, emits
    /// [`PassReaderEvent::Quarantined`] rather than retiring the generation.
    pub(crate) fn poll_completed_passes(&self) -> (Vec<PassReaderEvent>, Option<String>) {
        seismic::memory::assert_blocking_allocation_allowed("ExpertTransport::poll_completed_passes");
        let (mut events, done) = {
            let mut tracker = self.pass_tracker.lock().expect("transport pass tracker");
            let events: Vec<PassReaderEvent> = tracker.events.drain(..).collect();
            let mut ready = Vec::new();
            let mut remaining = VecDeque::with_capacity(tracker.completions.len());
            while let Some((key, reader, comp)) = tracker.completions.pop_front() {
                if comp.is_complete() {
                    ready.push((key, reader, comp));
                } else {
                    remaining.push_back((key, reader, comp));
                }
            }
            tracker.completions = remaining;
            (events, ready)
        };
        let mut first_error = None;
        for (key, reader, completion) in done {
            match completion.wait() {
                Ok(()) => events.push(PassReaderEvent::Retired { key, reader }),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error.to_string());
                    }
                    events.push(PassReaderEvent::Quarantined { key });
                }
            }
        }
        (events, first_error)
    }

    /// Wait on the host condvar (zero `cuEventSynchronize`!) for `Watcher` to
    /// publish the [`PrefillPlan`] for `(layer, sequence, window)`.
    pub(crate) fn take_prefill_plan(
        &self,
        layer: usize,
        sequence: u64,
        window: u64,
    ) -> Result<PrefillPlan, String> {
        let mut prefill = self.prefill.lock().expect("transport prefill rendezvous");
        loop {
            if let Some(plan) = prefill.plans.remove(&(layer, sequence, window)) {
                return Ok(plan);
            }
            if let Some(reason) = prefill.failed_steps.get(&sequence) {
                return Err(reason.clone());
            }
            if let Some(reason) = &prefill.ended {
                return Err(reason.clone());
            }
            prefill = self
                .prefill_ready
                .wait(prefill)
                .expect("transport prefill rendezvous");
        }
    }

    /// Coalesce all host-tier `HostTier -> Scratch` H2D copies (plus stream-
    /// ordered `Scratch -> Bank` D2D copies for any host-tier expert seeded on
    /// a prompt group's last step) across `plans` into ONE `NativeQueue::Fill`
    /// hardware DMA batch (`device.submit_fill_copies`).
    ///
    /// Returns `Ok(None)` with **zero device calls** when every plan in `plans`
    /// has an empty `listed`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::experts) fn submit_prefill_dma(
        &self,
        device: &Device,
        bank_plan: &ExpertBankPlan,
        scratch: &RecordTensors<Tensor>,
        host_stores: Option<&Arc<Stores>>,
        bank_stores: &Arc<Stores>,
        layer: usize,
        plans: &[PrefillPlan],
    ) -> Result<Option<seismic::NativeGraphCompletion>, String> {
        let total_listed: usize = plans.iter().map(|p| p.listed.len()).sum();
        if total_listed == 0 {
            self.prefill_dma_skipped.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }
        let host_stores = host_stores
            .ok_or_else(|| format!("layer {layer} published {total_listed} prefill tier copies without a host tier"))?;
        let group = bank_plan.layers()[layer].group;
        let layer_window = bank_plan.layer_window(layer);
        let seeded_from = bank_plan.seeded_from(group);
        let regions = bank_plan.regions(group);
        let seeded_tiered = merge_set_seeded_tiered(plans);

        let mut owned_runs: Vec<(Tensor, Tensor, u64)> =
            Vec::with_capacity((total_listed + seeded_tiered.len()) * regions.len());

        // Phase 1: HostTier -> Scratch H2D copies across the merge set.
        for plan in plans {
            for &(from, to) in &plan.listed {
                let tier_record = usize::try_from(from)
                    .map_err(|_| format!("invalid prefill tier record {from} at layer {layer}"))?;
                let slot = u64::try_from(to)
                    .map_err(|_| format!("invalid prefill scratch slot {to} at layer {layer}"))?;
                if slot >= layer_window {
                    return Err(format!(
                        "prefill scratch slot {slot} exceeds layer {layer} window {layer_window}"
                    ));
                }
                for (region, &(kind, _)) in regions.iter().enumerate() {
                    let src = host_stores
                        .view(group, region, tier_record)
                        .ok_or_else(|| format!("host tier record {tier_record} of group {group} is not held"))?;
                    let dst_stack = scratch
                        .get(kind)
                        .ok_or_else(|| format!("layer {layer} scratch lacks tensor {kind:?}"))?;
                    let dst = dst_stack
                        .slice_leading(slot, slot + 1)
                        .map_err(|error| format!("prefill scratch slice failed: {error}"))?;
                    let bytes = dst.byte_len();
                    if src.byte_len() < bytes {
                        return Err(format!(
                            "host tier record {tier_record} ({} bytes) smaller than scratch slot ({bytes} bytes)",
                            src.byte_len()
                        ));
                    }
                    owned_runs.push((src, dst, bytes));
                }
            }
        }

        // Phase 2: Stream-ordered Scratch -> Bank D2D copies for any host-tier
        // expert filled by ANY chunk in this merge set and seeded on the
        // prompt group's last step.
        for &(slot_i32, seed_offset_i32) in &seeded_tiered {
            let slot = u64::try_from(slot_i32)
                .map_err(|_| format!("invalid prefill seeded scratch slot {slot_i32}"))?;
            let seed_offset = u64::try_from(seed_offset_i32)
                .map_err(|_| format!("invalid prefill seed offset {seed_offset_i32}"))?;
            let bank_record = seeded_from
                .checked_add(seed_offset)
                .and_then(|record| usize::try_from(record).ok())
                .ok_or("prefill seeded bank record overflow")?;
            for (region, &(kind, _)) in regions.iter().enumerate() {
                let src_stack = scratch
                    .get(kind)
                    .ok_or_else(|| format!("layer {layer} scratch lacks tensor {kind:?}"))?;
                let src = src_stack
                    .slice_leading(slot, slot + 1)
                    .map_err(|error| format!("prefill seed scratch slice failed: {error}"))?;
                let dst = bank_stores
                    .view(group, region, bank_record)
                    .ok_or_else(|| format!("seeded bank record {bank_record} of group {group} is not held"))?;
                let bytes = src.byte_len();
                if dst.byte_len() < bytes {
                    return Err(format!(
                        "seeded bank record {bank_record} ({} bytes) smaller than scratch slot ({bytes} bytes)",
                        dst.byte_len()
                    ));
                }
                owned_runs.push((src, dst, bytes));
            }
        }

        if owned_runs.is_empty() {
            self.prefill_dma_skipped.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        let runs: Vec<TransferRun<'_>> = owned_runs
            .iter()
            .map(|(source, destination, bytes)| TransferRun {
                source,
                source_offset: 0,
                destination,
                destination_offset: 0,
                bytes: *bytes,
            })
            .collect();
        let completion = device
            .submit_fill_copies(&runs)
            .map_err(|error| format!("prefill host-tier DMA submit failed: {error}"))?;
        self.prefill_dma_batches.fetch_add(1, Ordering::Relaxed);
        Ok(Some(completion))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seismic::{Charges, Element};

    /// `astra`'s exact acceptance test:
    /// Demand joins an in-flight hint -> first consumer cancels -> resize tries
    /// to reuse destination -> second consumer still gets exact bytes (one
    /// transfer, zero early reuse, eventual release).
    #[test]
    fn demand_joins_in_flight_hint_cancel_blocks_early_reuse_until_last_consumer_releases() {
        let transport = ExpertTransport::new_registry_only();
        let key = ExpertKey { group: 0, layer: 3, expert: 17 };
        let source = SourceRef { tier_record: 42, generation: 7 };
        let dest = DestSlot { group: 0, slot: 5 };

        // Simulated VRAM slot storage verifying exact bytes survive until last consumer reads.
        let expected_bytes: Vec<u8> = (0..256u32).map(|i| (i * 13 + 7) as u8).collect();
        let mut vram_slot = expected_bytes.clone();

        // 1. LinkMover issues ONE in-flight hint transfer into `dest`.
        let hint_id = transport.register_in_flight_hint(key, source, dest, None);
        assert_eq!(transport.transfers_issued(), 1);
        assert!(!transport.can_reuse_slot(dest), "slot must be pinned while DMA is in flight");

        // 2. Consumer 1 (demand) joins the in-flight hint -> ZERO duplicate transfer!
        let joined_1 = transport
            .join_by_key(key, source)
            .expect("consumer 1 must join in-flight hint");
        assert_eq!(joined_1.lease.transfer_id, hint_id);
        assert_eq!(joined_1.dest, dest);
        assert_eq!(transport.transfers_issued(), 1, "joining must NOT issue a second transfer");
        assert_eq!(transport.demand_joins(), 1);

        // 3. Consumer 2 (demand) also joins the SAME in-flight hint.
        let joined_2 = transport
            .join_by_key(key, source)
            .expect("consumer 2 must join same in-flight hint");
        assert_eq!(joined_2.lease.transfer_id, hint_id);
        assert_eq!(joined_2.dest, dest);
        assert_eq!(transport.transfers_issued(), 1, "still exactly one physical transfer");
        assert_eq!(transport.demand_joins(), 2);

        // 4. First consumer cancels -> detaches ONLY its own demand, never recycles pending DMA.
        joined_1.lease.cancel();
        assert!(
            !transport.can_reuse_slot(dest),
            "resize must NOT reuse slot after consumer 1 cancels while DMA & consumer 2 remain"
        );
        assert!(
            !transport.try_release_hint_slot(dest),
            "hint owner release must NOT recycle slot while DMA & consumer 2 remain"
        );

        // 5. DMA completes, hint owner already detached -> slot STILL pinned by consumer 2!
        transport.complete_dma(hint_id);
        assert!(
            !transport.can_reuse_slot(dest),
            "resize must NOT reuse slot after DMA completes while consumer 2 is still reading"
        );
        assert!(
            !transport.try_release_hint_slot(dest),
            "try_release_hint_slot must still refuse reuse while consumer 2 is attached"
        );

        // 6. Consumer 2 reads exact bytes from `dest`, then releases its lease.
        assert_eq!(vram_slot, expected_bytes, "second consumer must observe exact untouched bytes");
        joined_2.lease.release();

        // 7. Eventual release: once DMA completed AND last consumer released, resize may reuse `dest`.
        assert!(
            transport.can_reuse_slot(dest),
            "slot must become reusable once DMA completed and last consumer released"
        );
        assert!(
            transport.try_release_hint_slot(dest),
            "try_release_hint_slot must succeed once unpinned"
        );
        // Simulate resize overwriting `dest` only after `can_reuse_slot` returned true.
        vram_slot.fill(0xAA);
        assert_eq!(transport.transfers_issued(), 1, "entire lifecycle used exactly one transfer");
    }

    /// Partial enqueue fault must quarantine every touched slot so neither
    /// joiners nor future allocations observe partial/corrupt bytes.
    #[test]
    fn partial_enqueue_fault_quarantines_touched_slots_until_explicit_clear() {
        let transport = ExpertTransport::new_registry_only();
        let key = ExpertKey { group: 0, layer: 1, expert: 9 };
        let source = SourceRef { tier_record: 12, generation: 1 };
        let slot_a = DestSlot { group: 0, slot: 2 };
        let slot_b = DestSlot { group: 0, slot: 3 };

        let id = transport.register_in_flight_hint(key, source, slot_a, None);
        transport.fault_transfer(id, "simulated CUDA enqueue error".to_owned());
        transport.quarantine_slots(&[slot_b]);

        assert!(transport.is_quarantined(slot_a));
        assert!(transport.is_quarantined(slot_b));
        assert!(!transport.can_reuse_slot(slot_a), "quarantined slot_a must not be reused");
        assert!(!transport.can_reuse_slot(slot_b), "quarantined slot_b must not be reused");
        assert!(
            transport.join_by_key(key, source).is_none(),
            "faulted/quarantined transfer must never be joined"
        );

        // Explicit recovery clears quarantine.
        transport.clear_quarantine(slot_a);
        transport.clear_quarantine(slot_b);
        assert!(transport.can_reuse_slot(slot_a));
        assert!(transport.can_reuse_slot(slot_b));
    }

    /// Prefill byte-identical HostTier -> Scratch DMA (including seeded
    /// Scratch -> Bank D2D propagation and empty-list zero-wait fast path).
    #[test]
    fn prefill_dma_copies_host_tier_to_scratch_and_seeds_bank_and_skips_empty() {
        let device = crate::test_support::open_backend(
            &seismic::Devices::discover(&[]).expect("discover devices"),
            seismic::BackendName::Cpu,
        )
        .expect("CPU device");
        let charges = Charges::tally(&device);
        let transport = ExpertTransport::new_registry_only();

        // Empty plan -> zero-wait, zero-device-call fast path.
        let empty = PrefillPlan::default();
        transport.publish_prefill_plan(0, 10, 0, empty);
        let taken = transport.take_prefill_plan(0, 10, 0).expect("empty plan");
        assert!(taken.listed.is_empty());
        assert!(taken.seeds.is_empty());

        // Non-empty HostTier -> Scratch H2D + Scratch -> Bank seeded D2D via
        // `Device::submit_fill_copies` on CPU backend.
        let src_bytes: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(5).wrapping_add(11)).collect();
        let host_tier = Tensor::shared(&charges, Element::i32(), &[2, 8]).expect("host_tier");
        host_tier.clone().write_from_host(&src_bytes).expect("write host_tier");

        let scratch = Tensor::shared(&charges, Element::i32(), &[2, 8]).expect("scratch");
        scratch.clone().write_from_host(&vec![0u8; 64]).expect("zero scratch");

        let bank = Tensor::shared(&charges, Element::i32(), &[2, 8]).expect("bank");
        bank.clone().write_from_host(&vec![0u8; 64]).expect("zero bank");

        let src_slot = host_tier.slice_leading(1, 2).expect("src slot 1");
        let dst_scratch = scratch.slice_leading(0, 1).expect("scratch slot 0");
        let dst_bank = bank.slice_leading(1, 2).expect("bank slot 1");

        // Submit H2D (host_tier[1] -> scratch[0]) + stream-ordered D2D (scratch[0] -> bank[1]).
        let runs = [
            TransferRun {
                source: &src_slot,
                source_offset: 0,
                destination: &dst_scratch,
                destination_offset: 0,
                bytes: 32,
            },
            TransferRun {
                source: &dst_scratch,
                source_offset: 0,
                destination: &dst_bank,
                destination_offset: 0,
                bytes: 32,
            },
        ];
        let completion = device.submit_fill_copies(&runs).expect("submit_fill_copies");
        completion.wait().expect("wait fill copies");

        let got_scratch = dst_scratch.read_to_host().expect("read scratch");
        let got_bank = dst_bank.read_to_host().expect("read bank");

        assert_eq!(got_scratch, &src_bytes[32..64], "HostTier -> Scratch bytes must match");
        assert_eq!(got_bank, &src_bytes[32..64], "Seeded Scratch -> Bank D2D bytes must match");
    }

    /// Regression test for cross-chunk seeded host-tier experts in a multi-chunk
    /// prefill merge set: when chunk 0 fills `scratch[slot]` from the host tier
    /// and chunk 1 (`sequence == last`, with empty `listed`) seeds `scratch[slot]`
    /// into `Bank`, `merge_set_seeded_tiered` MUST still emit the stream-ordered
    /// `Scratch -> Bank` D2D copy so `Bank` never retains stale bytes from the
    /// prior layer's scratch.
    #[test]
    fn cross_chunk_merge_set_seeds_host_tier_expert_filled_by_earlier_chunk() {
        let chunk_0 = PrefillPlan {
            listed: vec![(7, 3), (9, 4)],
            seeds: Vec::new(),
        };
        let chunk_1_last = PrefillPlan {
            listed: Vec::new(),
            // Slot 3 came from chunk 0's host-tier DMA; slot 5 was file-read.
            seeds: vec![(3, 0), (5, 1)],
        };
        let seeded_tiered = merge_set_seeded_tiered(&[chunk_0, chunk_1_last]);
        assert_eq!(
            seeded_tiered,
            vec![(3, 0)],
            "chunk 1's seed of chunk 0's host-tier slot must be propagated via D2D after H2D"
        );
    }

    /// Adaptive physics gate for split M7 decode DMA:
    /// - High-miss layers (e.g. GLM-5.3 at ~97 MB/layer) stay 100% split.
    /// - Low-miss layers (e.g. Qwen3.8 at ~0.7 MB/layer) enter pipelined
    ///   `RunStage::Whole` cooldown after warmup, avoiding per-layer
    ///   `flush_and_wait()` stalls, and re-engage immediately if a probe sees
    ///   high miss bytes.
    #[test]
    fn decode_dma_gate_skips_low_miss_layers_and_reengages_on_shift() {
        // High-miss regime (GLM-5.3: 64 MB/layer).
        let high = DecodeDmaGate::for_layer(0, false);
        for _ in 0..32 {
            assert!(high.should_split(1), "high-miss layer must always use split DMA");
            assert!(high.is_splitting(), "high-miss layer must report splitting");
            high.observe(1, 64 * 1024 * 1024);
        }

        // Draft layers (DSpark) NEVER split decode.
        let draft = DecodeDmaGate::for_layer(40, true);
        assert!(!draft.is_splitting(), "draft layer must never report splitting");
        assert!(!draft.should_split(1) && !draft.should_split(4));

        // Moderate-miss regime (1.4 MB/layer alternating with 0-miss calls):
        // must stay on split hardware DMA rather than falling back to SM zero-copy.
        let moderate = DecodeDmaGate::for_layer(0, false);
        for i in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(moderate.should_split(1), "warmup call {i} should measure split DMA");
            moderate.observe(1, if i % 3 == 0 { 1_400_000 } else { 0 });
        }
        assert!(moderate.should_split(1), "layers with misses must stay on split DMA");

        // Zero-miss resident regime: gates onto pipelined `Whole` after warmup.
        let low = DecodeDmaGate::for_layer(0, false);
        for i in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(low.should_split(1), "warmup call {i} should measure split DMA");
            low.observe(1, 0);
        }
        // Immediately after warmup, `SPLIT_DMA_COOLDOWN_STEPS` calls at M=1 must stay on pipelined `Whole`.
        for step in 0..SPLIT_DMA_COOLDOWN_STEPS {
            assert!(
                !low.should_split(1),
                "cooldown step {step} must skip split DMA to avoid per-layer flush_and_wait()"
            );
        }
        // Next step probes split DMA; if workload shifted to high miss bytes, it re-engages immediately.
        assert!(low.should_split(1), "probe step after cooldown must run split DMA");
        low.observe(1, 32 * 1024 * 1024);
        assert!(
            low.should_split(1),
            "high miss bytes on probe must immediately re-engage split DMA"
        );
    }

    /// Regression test for `69abfb` (RTX 5090 Qwen3.8): an `M = 1..2` warmup
    /// cooldown MUST NEVER strand an `M = 8` speculative verify burst on
    /// `RunStage::Whole` SM zero-copy PCIe.
    #[test]
    fn decode_dma_gate_m1_cooldown_never_strands_wide_verify_burst() {
        let gate = DecodeDmaGate::for_layer(3, false);
        for _ in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(gate.should_split(1));
            gate.observe(1, 0);
        }
        // Narrow M=1 lane is now in cooldown.
        assert!(!gate.should_split(1), "M=1 lane should be in cooldown");
        // Wide M=8 verify burst MUST still split through hardware DecodeDma!
        assert!(
            gate.should_split(8),
            "M=8 verify burst must NOT be stranded by M=1 warmup cooldown"
        );
        gate.observe(8, 8 * 1024 * 1024);
        assert!(gate.should_split(8), "high-miss M=8 lane must stay split");
    }

    /// Staggered probes across 48 layers: after warmup, layers enter cooldown
    /// with `layer % SPLIT_DMA_STAGGER_SLOTS` phase offsets so the peak
    /// concurrent probing cohort is bounded by `48 / SPLIT_DMA_STAGGER_SLOTS`
    /// (preventing a 48-layer synchronous probe storm) while every layer probes
    /// on a bounded periodic interval (`SPLIT_DMA_COOLDOWN_STEPS + 1`).
    #[test]
    fn decode_dma_gate_staggers_probes_across_48_layers() {
        const LAYERS: usize = 48;
        let gates: Vec<DecodeDmaGate> = (0..LAYERS).map(|l| DecodeDmaGate::for_layer(l, false)).collect();
        for _ in 0..SPLIT_DMA_WARMUP_CALLS {
            for gate in &gates {
                assert!(gate.should_split(1));
                gate.observe(1, 0);
            }
        }
        let expected_peak_cohort = LAYERS.div_ceil(SPLIT_DMA_STAGGER_SLOTS as usize);
        let max_initial_delay =
            (SPLIT_DMA_COOLDOWN_STEPS + (SPLIT_DMA_STAGGER_SLOTS - 1) * SPLIT_DMA_STAGGER_STRIDE) as usize;
        let expected_period = (SPLIT_DMA_COOLDOWN_STEPS + 1) as usize;
        let total_steps = max_initial_delay + expected_period * 2 + 1;
        let mut probe_steps: Vec<Vec<usize>> = vec![Vec::new(); LAYERS];
        let mut peak_cohort = 0usize;
        for step in 0..total_steps {
            let mut probing_this_step = 0usize;
            for (layer, gate) in gates.iter().enumerate() {
                if gate.should_split(1) {
                    probing_this_step += 1;
                    probe_steps[layer].push(step);
                    gate.observe(1, 0);
                }
            }
            peak_cohort = peak_cohort.max(probing_this_step);
        }
        assert_eq!(
            peak_cohort, expected_peak_cohort,
            "peak concurrent probing cohort across {LAYERS} layers should match {LAYERS}/{SPLIT_DMA_STAGGER_SLOTS}"
        );
        for (layer, steps) in probe_steps.iter().enumerate() {
            assert!(
                steps.len() >= 3,
                "layer {layer} probed only {} times over {total_steps} steps (expected >= 3)",
                steps.len()
            );
            assert!(
                steps[0] <= max_initial_delay,
                "layer {layer} first probe at step {} exceeded max initial delay {max_initial_delay}",
                steps[0]
            );
            for window in steps.windows(2) {
                assert_eq!(
                    window[1] - window[0],
                    expected_period,
                    "layer {layer} probe interval between steps {} and {} should equal {expected_period}",
                    window[0],
                    window[1]
                );
            }
        }
    }

    /// Derived [`SplitDmaCrossover`] physics (zero hardcoded constants):
    /// - RTX 5090 Gen5 (`48 GB/s` DMA, `24 GB/s` zero-copy, `5.1 us` DMA
    ///   latency + measured `step_round_trip`): naturally derives `~0.95 MiB`
    ///   at `15 us` sync (admitting DeepSeek `8.7 MiB`, blocking Qwen `0`
    ///   misses) and `~3.8 MiB` at `75 us` sync (gating Qwen `1-2` misses =
    ///   `1.35-2.7 MiB` onto pipelined `RunStage::Whole` while admitting
    ///   `8 MiB` bursts).
    /// - RTX 3090 Gen4 (`19.31 GB/s` DMA <= `22.86 GB/s` zero-copy) and any
    ///   `inv_b_split >= inv_b_zerocopy`: safely clamps to `u64::MAX` (never
    ///   splits).
    #[test]
    fn split_dma_crossover_derives_5090_narrow_and_wide_thresholds_and_clamps_slow_dma() {
        const MIB: u64 = 1024 * 1024;
        const KIB: u64 = 1024;

        // RTX 5090 PCIe 5.0 x16: 48 GB/s DMA + 1.5 TB/s resident read vs 24 GB/s SM zero-copy.
        let inv_zc = 1.0 / 24e9;
        let inv_split = 1.0 / 48e9 + 1.0 / 1.5e12;
        let fast_sync = SplitDmaCrossover::derive(inv_split, inv_zc, 5.1e-6 + 15.0e-6);
        assert!(
            (512 * KIB..=MIB).contains(&fast_sync.min_bytes),
            "5090 crossover at 20.1 us split overhead should lie in 0.5-1 MiB, got {} B",
            fast_sync.min_bytes
        );
        let heavy_sync = SplitDmaCrossover::derive(inv_split, inv_zc, 5.1e-6 + 75.0e-6);
        assert!(
            (3 * MIB..=5 * MIB).contains(&heavy_sync.min_bytes),
            "5090 crossover at 80.1 us split overhead should lie in 3-5 MiB, got {} B",
            heavy_sync.min_bytes
        );

        // (1) DeepSeek (8.7 MiB/layer) stays 100% on hardware DMA.
        let deepseek_m1 = DecodeDmaGate::for_layer_with_crossover(0, false, fast_sync);
        for _ in 0..8 {
            assert!(deepseek_m1.should_split(1));
            deepseek_m1.observe(1, 8_700_000);
        }
        assert!(deepseek_m1.should_split(1), "DeepSeek 8.7 MiB must stay on split DMA");

        // (2) Qwen (0 misses) gates onto pipelined RunStage::Whole.
        let qwen_m1 = DecodeDmaGate::for_layer_with_crossover(0, false, fast_sync);
        for _ in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(qwen_m1.should_split(1));
            qwen_m1.observe(1, 0);
        }
        assert!(!qwen_m1.should_split(1), "Qwen 0-miss must gate onto RunStage::Whole");

        // (3) Qwen M=12 1-2 misses (1.35-2.7 MiB) gates onto pipelined RunStage::Whole under 80.1 us boundary.
        let qwen_m12 = DecodeDmaGate::for_layer_with_crossover(0, false, heavy_sync);
        for i in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(qwen_m12.should_split(12));
            let miss_bytes = if i % 2 == 0 { 1_350_000 } else { 2_700_000 };
            qwen_m12.observe(12, miss_bytes);
        }
        assert!(
            !qwen_m12.should_split(12),
            "Qwen M=12 1-2 misses (1.35-2.7 MiB < {} B) must gate onto pipelined RunStage::Whole",
            heavy_sync.min_bytes
        );

        // (4) Heavy M=12 burst (8 MiB) stays on hardware DMA.
        let heavy_m12 = DecodeDmaGate::for_layer_with_crossover(0, false, heavy_sync);
        for _ in 0..8 {
            assert!(heavy_m12.should_split(12));
            heavy_m12.observe(12, 8 * MIB);
        }
        assert!(heavy_m12.should_split(12), "heavy M=12 8 MiB burst must stay on split DMA");

        // (5) RTX 3090 Gen4 (`19.31 GB/s` DMA <= `22.86 GB/s` zero-copy) and degenerate rates clamp to u64::MAX.
        for (inv_s, inv_z) in [
            (1.0 / 19.31e9, 1.0 / 22.86e9),
            (1.0 / 24e9, 1.0 / 24e9),
            (0.0, 1.0 / 24e9),
            (1.0 / 48e9, 0.0),
            (f64::NAN, 1.0 / 24e9),
        ] {
            let clamped = SplitDmaCrossover::derive(inv_s, inv_z, 20e-6);
            assert_eq!(clamped.min_bytes, u64::MAX);
        }

        // (6) `from_rates` derives cleanly from `TierCopyCosts` + `ExpertLines`.
        let transfer = magnitude_cost_model::ReadLine::new(5.1e-6, 1.0 / 48e9).expect("valid ReadLine");
        let copy = crate::assessment::TierCopyCosts {
            blocks: 2,
            whole_seconds: 1e-3,
            stall_share: 0.0,
            transfer,
            back: transfer,
            concurrent_host_byte_seconds: None,
        };
        let record_bytes = 3 * MIB;
        let link_ns = ((record_bytes as f64 / 24e9) * 1e9).round() as u32;
        let resident_ns = ((record_bytes as f64 / 1.5e12) * 1e9).round() as u32;
        let ns = |v: u32| magnitude_cost_model::Nanoseconds::try_from(v).unwrap();
        let expert_lines = magnitude_cost_model::ExpertLines {
            gpu: magnitude_cost_model::GpuLines {
                resident: magnitude_cost_model::PassLine { weights: ns(resident_ns), row: ns(500) },
                link: magnitude_cost_model::PassLine { weights: ns(link_ns), row: ns(500) },
                moves: magnitude_cost_model::Moves::NONE,
            },
            host: magnitude_cost_model::HostLines {
                cpu: None,
                memory: None,
                joint_memory: None,
                landing: None,
                record_bytes,
            },
        };
        let measured = SplitDmaCrossover::from_rates(&copy, &expert_lines, None);
        let expected = SplitDmaCrossover::derive(inv_split, inv_zc, 5.1e-6);
        assert!(
            measured.min_bytes.abs_diff(expected.min_bytes) <= 256,
            "from_rates {} should match derive {}",
            measured.min_bytes,
            expected.min_bytes
        );
    }

    /// Regression test for the driver-580 3-way deadlock (`3410c6`):
    /// `complete_dma`, `fault_transfer`, and `try_release_hint_slot` MUST
    /// defer dropping `TransferSync` (`Arc<seismic_cuda::Event>`) out of both
    /// `self.registry` and `Residency` / `NoBlockingAllocationGuard` so
    /// `cuEventDestroy_v2` can never execute while holding either lock.
    #[test]
    fn retired_transfer_syncs_are_deferred_outside_registry_and_residency_guard() {
        let device = crate::test_support::open_backend(
            &seismic::Devices::discover(&[]).expect("discover devices"),
            seismic::BackendName::Cpu,
        )
        .expect("CPU device");
        let charges = Charges::tally(&device);
        let transfer = Transfer::new(&device).expect("CPU transfer");
        let src = Tensor::shared(&charges, Element::i32(), &[4]).expect("src");
        let dst = Tensor::shared(&charges, Element::i32(), &[4]).expect("dst");
        let (_, sync) = transfer
            .copy_synced(&[TransferItem::Run(TransferRun {
                source: &src,
                source_offset: 0,
                destination: &dst,
                destination_offset: 0,
                bytes: 16,
            })])
            .expect("copy_synced");

        let transport = ExpertTransport::new_registry_only();
        let key = ExpertKey { group: 0, layer: 1, expert: 2 };
        let dest = DestSlot { group: 0, slot: 0 };

        {
            let _no_alloc = seismic::memory::enter_no_blocking_allocation();
            let id = transport.register_hint(key, 5, dest);
            transport.attach_sync(id, sync);
            assert_eq!(transport.deferred_syncs_len(), 0);
            transport.complete_dma(id);
            assert_eq!(
                transport.deferred_syncs_len(),
                1,
                "complete_dma must defer TransferSync drop instead of dropping under registry lock"
            );
            assert!(transport.try_release_hint_slot(dest));
            assert_eq!(
                transport.deferred_syncs_len(),
                1,
                "try_release_hint_slot under Residency must not drain deferred TransferSyncs"
            );
        }

        // Outside NoBlockingAllocationGuard and Registry lock, drain drops the syncs safely.
        transport.drain_deferred_syncs();
        assert_eq!(transport.deferred_syncs_len(), 0);
    }

    #[test]
    #[should_panic(expected = "drain_deferred_syncs")]
    fn drain_deferred_syncs_panics_under_no_blocking_allocation_guard() {
        let transport = ExpertTransport::new_registry_only();
        let _no_alloc = seismic::memory::enter_no_blocking_allocation();
        transport.drain_deferred_syncs();
    }

    /// Verify 2-phase overlapped `land_decode_misses`:
    /// 1. Phase A fires DMA for already-in-RAM Host-Tier misses (`source <= -2`)
    ///    and patches `DecodeSources` IMMEDIATELY — BEFORE `file.answer` arrives!
    /// 2. Phase B waits on `events` for `file.answer(seq, 0, false)`, resolves
    ///    both GPU-assigned (`SOURCE_SITE == -1`) and CPU-pending (`SOURCE_SITE == -2`,
    ///    promoted to GPU DMA because `routed_site_request` already ran in
    ///    `DecodeFront`) NVMe misses into free `DmaLanding` slots, and patches
    ///    `DecodeSources` cleanly.
    /// 3. On `Answer::Failed`, Phase B clears both `SOURCE_RECORD` and
    ///    `SOURCE_SITE` to `-1` without hanging.
    #[test]
    fn land_decode_misses_overlaps_phase_a_dma_with_phase_b_file_answer() {
        use crate::experts::events::Events;
        use crate::experts::exchange::{ExchangeTable, Pulse};
        use magnitude_kernels::constants;

        let device = crate::test_support::open_backend(
            &seismic::Devices::discover(&[]).expect("discover devices"),
            seismic::BackendName::Cpu,
        )
        .expect("CPU device");
        let regions = [seismic::SlabRegion { element: Element::bool(), row_shape: vec![64] }];
        let layout = seismic::SlabLayout::for_regions(1, 8, &regions).expect("slab layout");
        let granule = layout.slab_bytes.next_multiple_of(layout.slab_alignment);
        let scope = crate::memory::ledger::Ledgers::strict();
        let _owner = scope.changes().own();
        let ledger = scope.ledger(
            seismic::StoreIdentity::HostRam,
            Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        ledger.read(
            ledger.sampled(std::time::Instant::now()),
            crate::memory::supply::Reading::device(1 << 40, 1 << 40, crate::domain::fixtures::NO_LAG),
        );
        let changes = scope.changes().clone();
        let entries = ledger.pool_block(
            crate::memory::ledger::PoolClass::Records,
            std::num::NonZeroU64::new(granule).expect("granule"),
        );
        let pool_charges = entries.charges(&device, seismic::Contract::Shared);
        let pool = seismic::Pool::open(&pool_charges, seismic::Contract::Shared, granule, granule)
            .expect("pool");
        let make_stores = |count: u64| {
            let charges = Charges::tally(&device);
            let slab = seismic::SlabTensor::in_pool(&charges, &pool, 1, count, &regions).expect("slab");
            let stores = Arc::new(Stores::new(
                vec![(pool.clone(), seismic::BlockSize { bytes: granule, held: granule, absent: granule })],
                vec![0],
                vec![(slab, 0, vec![64])],
            ));
            changes.change(|_| {
                stores.claim_to(0, count).expect("claim_to");
            });
            stores
        };
        let host_stores = make_stores(8);
        let bank_stores = make_stores(8);
        let charges = Charges::tally(&device);
        let k = 4usize;
        let rows = 1u64;
        let count = (rows as usize) * k;
        let words_count = (constants::SOURCE_ROWS as u64) * (count as u64);
        let sources = Tensor::words(&charges, Element::i32(), &[words_count]).expect("sources");
        let sources_words = Words::new(sources.clone(), words_count, 0).expect("sources_words");

        let file_table_desc = magnitude_kernels::SlotTable::file(4, k as u64);
        let file_tensor = Tensor::words(&charges, Element::i32(), &[file_table_desc.words()]).expect("file");
        let file = ExchangeTable::new(file_tensor, file_table_desc, &Pulse::of(seismic::BackendName::Cpu))
            .expect("ExchangeTable");
        let events = Arc::new(Events::default());

        let transfer = Transfer::new(&device).expect("transfer");
        let dma_state = DecodeDmaState::new(
            transfer,
            host_stores,
            bank_stores,
            sources,
            sources_words.clone(),
            k,
            &[false],
            vec![file.clone()],
            events.clone(),
        );
        let transport = ExpertTransport::new(Some(dma_state));

        // Choice 0: already-in-RAM Host-Tier miss (tier_record = 1 -> -2 - 1 = -3)
        // Choice 1: NVMe miss, GPU-assigned (SOURCE_RECORD = -1, SOURCE_SITE = -1)
        // Choice 2: NVMe miss, CPU-pending (SOURCE_RECORD = -1, SOURCE_SITE = -2 -> promoted to GPU DMA)
        // Choice 3: resident VRAM hit (slot 0)
        let site_base = (constants::SOURCE_SITE as usize) * count;
        sources_words.store(0, -3);
        sources_words.store(1, -1);
        sources_words.store(2, -1);
        sources_words.store(3, 0);
        sources_words.store(site_base + 0, -1);
        sources_words.store(site_base + 1, -1);
        sources_words.store(site_base + 2, -2);
        sources_words.store(site_base + 3, -1);

        let seq = 1u64;
        file.store(seq, 0, constants::FILE_COUNT, k as i32);
        file.store(seq, 0, constants::FILE_ENTRY(0) + constants::FILE_CHOICE, -1);
        file.store(seq, 0, constants::FILE_ENTRY(1) + constants::FILE_CHOICE, 11);
        file.store(seq, 0, constants::FILE_ENTRY(2) + constants::FILE_CHOICE, 12);
        file.store(seq, 0, constants::FILE_ENTRY(3) + constants::FILE_CHOICE, -1);
        file.request(seq, 0);

        // Background "Watcher" thread waits until Phase A has ALREADY patched
        // choice 0 into a VRAM landing slot (proving Phase A DMA ran BEFORE
        // `file.answer`), then answers choices 1 & 2 with tier records 3 & 4.
        let bg_sources = sources_words.clone();
        let bg_file = file.clone();
        let bg_events = events.clone();
        let watcher = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while bg_sources.load(0) < 0 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Phase A DMA must patch choice 0 BEFORE file.answer is published!"
                );
                std::thread::yield_now();
            }
            bg_file.store(seq, 0, constants::FILE_ENTRY(1) + constants::FILE_RECORD, -2 - 3);
            bg_file.store(seq, 0, constants::FILE_ENTRY(2) + constants::FILE_RECORD, -2 - 4);
            bg_file.answer(seq, 0, false);
            bg_events.notify();
        });

        let mut free_pool: Vec<usize> = vec![2, 3, 4, 5];
        let mut recorded = (false, 0u64, 0u64, 0u64);
        let landing = transport
            .land_decode_misses(
                0,
                0,
                seq,
                rows,
                |_g, trs, _replenish| {
                    trs.iter()
                        .map(|&tr| {
                            if let Some(slot) = free_pool.pop() {
                                crate::experts::directory::DemandSlotClaim::Allocated {
                                    staged: Some((0, tr as usize)),
                                    slot,
                                }
                            } else {
                                crate::experts::directory::DemandSlotClaim::Overflow { staged: None }
                            }
                        })
                        .collect()
                },
                |_| {},
                || {},
                || false,
                |f, e, b, o| recorded = (f, e, b, o),
            )
            .expect("land_decode_misses")
            .expect("some landing");
        watcher.join().expect("watcher thread");

        // All 3 misses (Phase A choice 0 + Phase B choices 1 & 2) landed into
        // distinct unified slots in 2..6, and choice 2's SOURCE_SITE was
        // cleared from -2 to -1 so DecodeBack computes it on GPU.
        assert!(!recorded.0, "must not be fast-path skip");
        assert_eq!(recorded.1, 3, "expected 3 landed experts (1 Phase A + 2 Phase B)");
        assert_eq!(recorded.3, 0, "expected 0 overflow experts");
        let s0 = sources_words.load(0);
        let s1 = sources_words.load(1);
        let s2 = sources_words.load(2);
        assert!((2..6).contains(&s0), "choice 0 slot {s0} must be in 2..6");
        assert!((2..6).contains(&s1), "choice 1 slot {s1} must be in 2..6");
        assert!((2..6).contains(&s2), "choice 2 slot {s2} must be in 2..6");
        assert_ne!(s0, s1);
        assert_ne!(s0, s2);
        assert_ne!(s1, s2);
        assert_eq!(sources_words.load(3), 0, "VRAM hit untouched");
        assert_eq!(sources_words.load(site_base + 2), -1, "SOURCE_SITE == -2 promoted to GPU (-1)");

        landing.release(&transport);

        // Case 2: Failed file answer (`Answer::Failed`) clears SOURCE_RECORD
        // and SOURCE_SITE to -1 without hanging.
        let seq2 = 2u64;
        sources_words.store(0, -1);
        sources_words.store(site_base + 0, -2);
        for i in 1..k {
            sources_words.store(i, 0);
            sources_words.store(site_base + i, -1);
        }
        file.store(seq2, 0, constants::FILE_COUNT, k as i32);
        file.store(seq2, 0, constants::FILE_ENTRY(0) + constants::FILE_CHOICE, 20);
        for i in 1..k {
            file.store(seq2, 0, constants::FILE_ENTRY(i as u64) + constants::FILE_CHOICE, -1);
        }
        file.request(seq2, 0);
        file.answer(seq2, 0, true);
        let failed = transport
            .land_decode_misses(
                0,
                0,
                seq2,
                rows,
                |_, trs, _| {
                    vec![crate::experts::directory::DemandSlotClaim::Overflow { staged: None }; trs.len()]
                },
                |_| {},
                || {},
                || false,
                |_, _, _, _| {},
            )
            .expect("failed answer ok");
        assert!(failed.is_none());
        assert_eq!(sources_words.load(0), -1);
        assert_eq!(sources_words.load(site_base + 0), -1);
    }

    /// Prove incremental Phase B H2D dispatch:
    /// - Choices 0 & 1 land via `publish_decode_file_ready` + `events.notify()`
    ///   while straggler choice 2 is still in flight (`file.answer` NOT yet set).
    /// - The watcher thread blocks until choices 0 & 1 are ALREADY DMA-dispatched
    ///   and patched into VRAM slots in `DecodeSources` BEFORE releasing the
    ///   straggler and publishing `file.answer(seq, 0, false)`.
    /// - Final `DecodeLanding` preserves exact completion sync ordering (`syncs`),
    ///   tracks all 3 demand slots/leases, and replenishes `target_free` on the
    ///   final batch.
    #[test]
    fn land_decode_misses_dispatches_phase_b_incrementally_before_straggler() {
        use crate::experts::events::Events;
        use crate::experts::exchange::{ExchangeTable, Pulse};
        use magnitude_kernels::constants;

        let device = crate::test_support::open_backend(
            &seismic::Devices::discover(&[]).expect("discover devices"),
            seismic::BackendName::Cpu,
        )
        .expect("CPU device");
        let regions = [seismic::SlabRegion { element: Element::bool(), row_shape: vec![64] }];
        let layout = seismic::SlabLayout::for_regions(1, 8, &regions).expect("slab layout");
        let granule = layout.slab_bytes.next_multiple_of(layout.slab_alignment);
        let scope = crate::memory::ledger::Ledgers::strict();
        let _owner = scope.changes().own();
        let ledger = scope.ledger(
            seismic::StoreIdentity::HostRam,
            Arc::new(crate::memory::ledger::LoadShape { stores: Vec::new() }),
        );
        ledger.read(
            ledger.sampled(std::time::Instant::now()),
            crate::memory::supply::Reading::device(1 << 40, 1 << 40, crate::domain::fixtures::NO_LAG),
        );
        let changes = scope.changes().clone();
        let entries = ledger.pool_block(
            crate::memory::ledger::PoolClass::Records,
            std::num::NonZeroU64::new(granule).expect("granule"),
        );
        let pool_charges = entries.charges(&device, seismic::Contract::Shared);
        let pool = seismic::Pool::open(&pool_charges, seismic::Contract::Shared, granule, granule)
            .expect("pool");
        let make_stores = |count: u64| {
            let charges = Charges::tally(&device);
            let slab = seismic::SlabTensor::in_pool(&charges, &pool, 1, count, &regions).expect("slab");
            let stores = Arc::new(Stores::new(
                vec![(pool.clone(), seismic::BlockSize { bytes: granule, held: granule, absent: granule })],
                vec![0],
                vec![(slab, 0, vec![64])],
            ));
            changes.change(|_| {
                stores.claim_to(0, count).expect("claim_to");
            });
            stores
        };
        let host_stores = make_stores(8);
        let bank_stores = make_stores(8);
        let charges = Charges::tally(&device);
        let k = 4usize;
        let rows = 1u64;
        let count = (rows as usize) * k;
        let words_count = (constants::SOURCE_ROWS as u64) * (count as u64);
        let sources = Tensor::words(&charges, Element::i32(), &[words_count]).expect("sources");
        let sources_words = Words::new(sources.clone(), words_count, 0).expect("sources_words");

        let file_table_desc = magnitude_kernels::SlotTable::file(4, k as u64);
        let file_tensor = Tensor::words(&charges, Element::i32(), &[file_table_desc.words()]).expect("file");
        let file = ExchangeTable::new(file_tensor, file_table_desc, &Pulse::of(seismic::BackendName::Cpu))
            .expect("ExchangeTable");
        let events = Arc::new(Events::default());

        let transfer = Transfer::new(&device).expect("transfer");
        let dma_state = DecodeDmaState::new(
            transfer,
            host_stores,
            bank_stores,
            sources,
            sources_words.clone(),
            k,
            &[false],
            vec![file.clone()],
            events.clone(),
        );
        let transport = ExpertTransport::new(Some(dma_state));

        // Choices 0, 1, 2: cold NVMe misses (choice 2 CPU-pending -> promoted to GPU DMA).
        // Choice 3: resident VRAM hit (slot 0).
        let site_base = (constants::SOURCE_SITE as usize) * count;
        sources_words.store(0, -1);
        sources_words.store(1, -1);
        sources_words.store(2, -1);
        sources_words.store(3, 0);
        sources_words.store(site_base + 0, -1);
        sources_words.store(site_base + 1, -1);
        sources_words.store(site_base + 2, -2);
        sources_words.store(site_base + 3, -1);

        let seq = 1u64;
        file.store(seq, 0, constants::FILE_COUNT, k as i32);
        file.store(seq, 0, constants::FILE_ENTRY(0) + constants::FILE_CHOICE, 10);
        file.store(seq, 0, constants::FILE_ENTRY(1) + constants::FILE_CHOICE, 11);
        file.store(seq, 0, constants::FILE_ENTRY(2) + constants::FILE_CHOICE, 12);
        file.store(seq, 0, constants::FILE_ENTRY(3) + constants::FILE_CHOICE, -1);
        file.request(seq, 0);

        let bg_transport = Arc::clone(&transport);
        let bg_sources = sources_words.clone();
        let bg_file = file.clone();
        let bg_events = events.clone();
        let watcher = std::thread::spawn(move || {
            // Land choices 0 & 1 early (straggler choice 2 still reading on NVMe).
            bg_file.store(seq, 0, constants::FILE_ENTRY(0) + constants::FILE_RECORD, -2 - 1);
            assert!(bg_transport.publish_decode_file_ready(0, seq, &[0], -2 - 1));
            bg_file.store(seq, 0, constants::FILE_ENTRY(1) + constants::FILE_RECORD, -2 - 2);
            assert!(bg_transport.publish_decode_file_ready(0, seq, &[1], -2 - 2));
            bg_events.notify();

            // Wait until Phase B has ALREADY dispatched H2D and patched choices 0 & 1
            // BEFORE choice 2 finishes or `file.answer` is published!
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            while bg_sources.load(0) < 0 || bg_sources.load(1) < 0 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "Phase B must dispatch ready records (choices 0 & 1) BEFORE straggler completes!"
                );
                std::thread::yield_now();
            }
            assert!(!bg_file.answered(seq, 0), "file must still be unanswered while straggler lags");
            assert_eq!(bg_sources.load(2), -1, "straggler choice 2 must still be unresolved");

            // Now land straggler choice 2 and publish final `FILE_ANSWER`.
            bg_file.store(seq, 0, constants::FILE_ENTRY(2) + constants::FILE_RECORD, -2 - 3);
            assert!(bg_transport.publish_decode_file_ready(0, seq, &[2], -2 - 3));
            bg_file.answer(seq, 0, false);
            bg_events.notify();
        });

        let mut free_pool: Vec<usize> = vec![2, 3, 4, 5];
        let mut batches: Vec<(Vec<u32>, usize)> = Vec::new();
        let mut recorded = (false, 0u64, 0u64, 0u64);
        let landing = transport
            .land_decode_misses(
                0,
                0,
                seq,
                rows,
                |_g, trs, replenish| {
                    batches.push((trs.to_vec(), replenish));
                    trs.iter()
                        .map(|&tr| {
                            if let Some(slot) = free_pool.pop() {
                                crate::experts::directory::DemandSlotClaim::Allocated {
                                    staged: Some((0, tr as usize)),
                                    slot,
                                }
                            } else {
                                crate::experts::directory::DemandSlotClaim::Overflow { staged: None }
                            }
                        })
                        .collect()
                },
                |_| {},
                || {},
                || false,
                |f, e, b, o| recorded = (f, e, b, o),
            )
            .expect("land_decode_misses")
            .expect("some landing");
        watcher.join().expect("watcher thread");

        // At least 2 distinct Phase B claim batches: early subset before straggler
        // (replenish == 0) + final straggler batch (replenish == target_free > 0).
        assert!(
            batches.len() >= 2,
            "expected incremental Phase B batches before straggler, got {batches:?}"
        );
        assert_eq!(batches[0].1, 0, "intermediate Phase B batch must defer headroom replenish");
        assert!(
            batches.last().unwrap().1 >= k,
            "final Phase B batch must replenish target_free"
        );

        // Final synchronization and slot patching intact for all 3 misses.
        assert!(!recorded.0, "must not be fast-path skip");
        assert_eq!(recorded.1, 3, "expected 3 landed experts across incremental Phase B");
        assert_eq!(recorded.3, 0, "expected 0 overflow experts");
        assert_eq!(landing.own_transfer_count(), 3);
        assert_eq!(landing.own_slot_count(), 3);
        assert_eq!(landing.syncs.len(), 1, "single latest FIFO transfer_done sync orders all batches");
        let s0 = sources_words.load(0);
        let s1 = sources_words.load(1);
        let s2 = sources_words.load(2);
        assert!((2..6).contains(&s0), "choice 0 slot {s0} must be in 2..6");
        assert!((2..6).contains(&s1), "choice 1 slot {s1} must be in 2..6");
        assert!((2..6).contains(&s2), "choice 2 slot {s2} must be in 2..6");
        assert_ne!(s0, s1);
        assert_ne!(s0, s2);
        assert_ne!(s1, s2);
        assert_eq!(sources_words.load(3), 0, "VRAM hit untouched");
        assert_eq!(sources_words.load(site_base + 2), -1, "straggler SOURCE_SITE == -2 promoted to GPU (-1)");

        landing.release(&transport);
    }

    /// Partial issue before fence record: a batch in `PartiallyIssued` (no
    /// timeline value recorded) must be quarantined on abort and its slot MUST
    /// NEVER be returned to `Directory`'s free pool.
    #[test]
    fn issued_demand_batch_partial_issue_before_fence_record_quarantines_without_freeing() {
        let transport = ExpertTransport::new_registry_only();
        let group = 0usize;
        let slot_p = DestSlot { group, slot: 1 };
        let (id_p, lease_p) = transport.begin_demand_transfer(group, 0, 1, 1, slot_p);

        let mut tx = IssuedDemandBatch::new(Arc::clone(&transport), 0, 10);
        let idx = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_p.slot, staged: None }],
            vec![id_p],
            vec![lease_p],
            Vec::new(),
        );
        assert_eq!(tx.batch_states(), vec![DemandBatchState::Unissued]);
        tx.begin_issue(idx);
        assert_eq!(tx.batch_states(), vec![DemandBatchState::PartiallyIssued]);
        assert_eq!(tx.highest_submitted(), None);

        let mut waited = false;
        let mut aborted: Vec<(usize, usize)> = Vec::new();
        tx.abort(
            "mid-batch copy error before fence record",
            |_v| {
                waited = true;
                Ok(true)
            },
            &mut |slots| aborted.extend_from_slice(slots),
        );

        assert!(!waited, "no submitted batch to wait on");
        assert!(aborted.is_empty(), "partially issued slot must NEVER be returned to Directory");
        assert!(!transport.can_reuse_slot(slot_p), "partially issued slot must remain quarantined");
        assert!(transport.is_quarantined(slot_p));
        assert_eq!(tx.batch_states(), vec![DemandBatchState::Retired]);
        assert!(!transport.demand_active(), "flight guard released on abort");
    }

    /// Batch A succeeds (`Submitted { value: 42 }`), Batch B fails mid-issue
    /// (`PartiallyIssued`): `abort` waits on `42` (`Ok(true)`), retires Batch A
    /// cleanly back to `Directory`, and quarantines Batch B without freeing it.
    #[test]
    fn issued_demand_batch_a_succeeds_b_fails_retires_a_and_quarantines_b() {
        let transport = ExpertTransport::new_registry_only();
        let group = 0usize;
        let slot_a = DestSlot { group, slot: 3 };
        let slot_b = DestSlot { group, slot: 4 };
        let (id_a, lease_a) = transport.begin_demand_transfer(group, 0, 10, 10, slot_a);
        let (id_b, lease_b) = transport.begin_demand_transfer(group, 0, 11, 11, slot_b);

        let mut tx = IssuedDemandBatch::new(Arc::clone(&transport), 0, 20);
        let idx_a = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_a.slot, staged: None }],
            vec![id_a],
            vec![lease_a],
            Vec::new(),
        );
        tx.begin_issue(idx_a);
        tx.mark_submitted(idx_a, 42, None);

        let idx_b = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_b.slot, staged: None }],
            vec![id_b],
            vec![lease_b],
            Vec::new(),
        );
        tx.begin_issue(idx_b);
        assert_eq!(
            tx.batch_states(),
            vec![DemandBatchState::Submitted { value: 42 }, DemandBatchState::PartiallyIssued]
        );

        let mut waited_value = None;
        let mut aborted: Vec<(usize, usize)> = Vec::new();
        let order = std::cell::RefCell::new(Vec::<&'static str>::new());
        tx.abort(
            "synthetic phase B copy error",
            |v| {
                waited_value = Some(v);
                order.borrow_mut().push("wait");
                Ok(true)
            },
            &mut |slots| {
                aborted.extend_from_slice(slots);
                order.borrow_mut().push("abort");
            },
        );

        assert_eq!(waited_value, Some(42), "must wait for highest submitted timeline value");
        assert_eq!(*order.borrow(), vec!["wait", "abort"], "must wait before freeing A's slot");
        assert_eq!(aborted, vec![(group, slot_a.slot)], "only proven-complete A returns to Directory; uncertain B stays quarantined");
        assert!(transport.can_reuse_slot(slot_a), "Batch A slot must be reusable after wait");
        assert!(!transport.can_reuse_slot(slot_b), "PartiallyIssued Batch B slot must remain quarantined");
        assert!(transport.is_quarantined(slot_b));
        assert_eq!(tx.batch_states(), vec![DemandBatchState::Retired, DemandBatchState::Retired]);
    }

    /// Timeout on wait followed by late DMA completion: quarantined storage is
    /// never returned to `Directory`, never reused prematurely, and never
    /// double-freed on repeated abort/drop.
    #[test]
    fn issued_demand_batch_timeout_then_late_completion_never_double_frees_or_reuses() {
        let transport = ExpertTransport::new_registry_only();
        let group = 0usize;
        let slot_a = DestSlot { group, slot: 5 };
        let slot_b = DestSlot { group, slot: 6 };
        let (id_a, lease_a) = transport.begin_demand_transfer(group, 0, 20, 20, slot_a);
        let (id_b, lease_b) = transport.begin_demand_transfer(group, 0, 21, 21, slot_b);

        let mut tx = IssuedDemandBatch::new(Arc::clone(&transport), 0, 30);
        let idx_a = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_a.slot, staged: None }],
            vec![id_a],
            vec![lease_a],
            Vec::new(),
        );
        tx.begin_issue(idx_a);
        tx.mark_submitted(idx_a, 99, None);

        let idx_b = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_b.slot, staged: None }],
            vec![id_b],
            vec![lease_b],
            Vec::new(),
        );
        tx.begin_issue(idx_b);

        let mut aborted: Vec<(usize, usize)> = Vec::new();
        tx.abort(
            "synthetic phase B copy error",
            |_v| Ok(false), // timeout
            &mut |slots| aborted.extend_from_slice(slots),
        );

        assert!(aborted.is_empty(), "timeout must quarantine ALL uncertain slots");
        assert!(!transport.can_reuse_slot(slot_a));
        assert!(!transport.can_reuse_slot(slot_b));
        assert!(transport.is_quarantined(slot_a));
        assert!(transport.is_quarantined(slot_b));

        // Simulate late DMA completion arriving after timeout: quarantined
        // storage must remain quarantined and never become reusable.
        transport.complete_dma(id_a);
        assert!(
            !transport.can_reuse_slot(slot_a),
            "late completion must NEVER un-quarantine or reuse timed-out storage"
        );

        // Idempotent second abort / Drop must never double-free.
        tx.abort("second abort", |_| Ok(true), &mut |slots| aborted.extend_from_slice(slots));
        drop(tx);
        assert!(aborted.is_empty(), "repeated abort/drop must never free quarantined slots");
    }

    /// Shared transfer retains another reader until that reader retires, and a
    /// `BackSubmitted` reader dropped without completion retains its pin in
    /// quarantine.
    #[test]
    fn issued_demand_batch_shared_transfer_retains_other_reader_until_retired() {
        let transport = ExpertTransport::new_registry_only();
        let group = 0usize;
        let key = ExpertKey { group, layer: 0, expert: 7 };
        let src = SourceRef { tier_record: 7, generation: 1 };
        let slot = DestSlot { group, slot: 9 };
        let hint_id = transport.register_in_flight_hint(key, src, slot, None);

        let joined_1 = transport.join_by_key(key, src).expect("reader 1 joins");
        let joined_2 = transport.join_by_key(key, src).expect("reader 2 joins");

        let mut tx1 = IssuedDemandBatch::new(Arc::clone(&transport), 0, 40);
        tx1.record_joined(joined_1.lease, joined_1.sync);
        assert_eq!(tx1.batch_states(), vec![DemandBatchState::JoinedExisting]);

        let mut tx2 = IssuedDemandBatch::new(Arc::clone(&transport), 0, 41);
        tx2.record_joined(joined_2.lease, joined_2.sync);

        // Reader 1 aborts before DecodeBack: detaches ONLY reader 1.
        let mut aborted = Vec::new();
        tx1.abort("reader 1 aborted", |_| Ok(true), &mut |slots| aborted.extend_from_slice(slots));
        assert!(aborted.is_empty(), "joined reader never owns destination slot");

        // Hint DMA finishes, but reader 2 is still active -> slot must stay pinned.
        transport.complete_dma(hint_id);
        assert!(!transport.try_release_hint_slot(slot), "reader 2 still pins shared slot");
        assert!(!transport.can_reuse_slot(slot));

        // Reader 2 completes normally -> slot can now retire.
        let (_, _, own_slots) = tx2.retire_completed();
        assert!(own_slots.is_empty());
        assert!(transport.can_reuse_slot(slot), "slot becomes reusable once last reader retires");

        // Now verify BackSubmitted reader dropped unexpectedly retains its pin.
        let hint_id2 = transport.register_in_flight_hint(key, src, slot, None);
        let joined_3 = transport.join_by_key(key, src).expect("reader 3 joins");
        let mut tx3 = IssuedDemandBatch::new(Arc::clone(&transport), 0, 42);
        tx3.record_joined(joined_3.lease, joined_3.sync);
        tx3.mark_back_submitted();
        drop(tx3);
        transport.complete_dma(hint_id2);
        assert!(
            !transport.try_release_hint_slot(slot),
            "BackSubmitted reader dropped without completion must retain its pin"
        );
        assert!(!transport.can_reuse_slot(slot));
    }

    /// Device teardown / Drop path must NEVER panic or block locks (verified
    /// under `NoBlockingAllocationGuard`), and must hand uncertain work to
    /// `ExpertTransport`'s retirement quarantine while releasing the flight guard.
    #[test]
    fn issued_demand_batch_drop_under_no_blocking_guard_never_panics_or_blocks() {
        let transport = ExpertTransport::new_registry_only();
        let group = 0usize;
        let slot_u = DestSlot { group, slot: 10 };
        let slot_s = DestSlot { group, slot: 11 };
        let (id_u, lease_u) = transport.begin_demand_transfer(group, 0, 30, 30, slot_u);
        let (id_s, lease_s) = transport.begin_demand_transfer(group, 0, 31, 31, slot_s);

        let mut tx = IssuedDemandBatch::new(Arc::clone(&transport), 0, 50);
        let _idx_u = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_u.slot, staged: None }],
            vec![id_u],
            vec![lease_u],
            Vec::new(),
        );
        let idx_s = tx.claim_own_batch(
            vec![DemandSlot { group, slot: slot_s.slot, staged: None }],
            vec![id_s],
            vec![lease_s],
            Vec::new(),
        );
        tx.begin_issue(idx_s);
        tx.mark_submitted(idx_s, 77, None);
        tx.mark_back_submitted();
        assert!(transport.demand_active());

        {
            // Dropping under `NoBlockingAllocationGuard` proves Drop never
            // calls blocking waits or `drain_deferred_syncs`.
            let _no_block = seismic::memory::enter_no_blocking_allocation();
            drop(tx);
        }

        assert!(!transport.demand_active(), "Drop must release DemandFlightGuard");
        assert!(!transport.can_reuse_slot(slot_u));
        assert!(!transport.can_reuse_slot(slot_s));
        assert!(transport.is_quarantined(slot_u));
        assert!(transport.is_quarantined(slot_s));
        assert_eq!(transport.quarantined_backing_len(), 0);
    }

    /// Regression test (`mx5stm`): when an owner aborts a `Submitted` demand
    /// batch after H2D finishes (`prefix_proven == true`) while a joined
    /// borrower still holds its `ConsumerLease` on the destination slot, the
    /// pin-aware abort path MUST defer release (`defer_demand_release`) rather
    /// than freeing the slot back to `Record::Free`, and release it only once
    /// the borrower's lease drops.
    #[test]
    fn issued_demand_batch_owner_abort_while_borrower_still_reads_defers_slot_release() {
        use crate::experts::records::{BankRecords, Record};

        let transport = ExpertTransport::new_registry_only();
        let mut bank = BankRecords::new(&[4], &[(0, 8)], None);
        let group = 0usize;
        let layer = 0usize;
        let expert = 3usize;
        let tier_record = 12usize;

        // Owner claims a `Record::DemandLanding` slot in `BankRecords`.
        let slot = bank
            .claim_demand_landing(group, Some((layer, expert)), false, |s| {
                transport.can_reuse_slot(DestSlot { group, slot: s })
            })
            .expect("claim demand landing slot");
        assert!(matches!(bank.records(group)[slot], Record::DemandLanding { .. }));

        let dest = DestSlot { group, slot };
        let (id, owner_lease) = transport.begin_demand_transfer(group, layer, expert, tier_record, dest);
        let mut owner = IssuedDemandBatch::new(Arc::clone(&transport), layer, 60);
        let idx = owner.claim_own_batch(
            vec![DemandSlot { group, slot, staged: Some((layer, expert)) }],
            vec![id],
            vec![owner_lease],
            Vec::new(),
        );
        owner.begin_issue(idx);
        transport.complete_dma(id);
        owner.mark_submitted(idx, 99, None);

        // Borrower joins the owner's destination slot and holds its `ConsumerLease`.
        let borrower = transport
            .join_by_tier_record(group, layer, tier_record)
            .expect("borrower joins owner's destination slot");
        assert_eq!(borrower.dest, dest);

        // Owner aborts with H2D confirmed complete (`prefix_proven == true`),
        // routing aborted slots through the pin-aware release path (matching
        // `Watcher::abort_demand_slots` -> `Directory::release_demand_landing`).
        owner.abort("owner step aborted", |_| Ok(true), &mut |slots| {
            transport.with_slot_reuse(|can_reuse| {
                for &(g, s) in slots {
                    if can_reuse(DestSlot { group: g, slot: s }) {
                        bank.release_demand_landing(g, s);
                    } else {
                        bank.defer_demand_release(g, s);
                    }
                }
            });
        });

        // Slot MUST NOT be `Record::Free` while borrower still reads!
        assert!(
            matches!(bank.records(group)[slot], Record::DemandLanding { .. }),
            "aborted owner slot with active borrower must stay DemandLanding (deferred), got {:?}",
            bank.records(group)[slot]
        );
        assert!(!transport.can_reuse_slot(dest), "borrower lease still pins dest slot");
        assert!(
            !bank.drain_demand_landings(|g, s| transport.can_reuse_slot(DestSlot { group: g, slot: s })),
            "drain must not free slot while borrower still reads"
        );

        // Once the borrower finishes and releases its lease, drain frees the slot.
        borrower.lease.release();
        assert!(transport.can_reuse_slot(dest));
        assert!(bank.drain_demand_landings(|g, s| transport.can_reuse_slot(DestSlot { group: g, slot: s })));
        assert_eq!(bank.records(group)[slot], Record::Free);
    }

    /// Regression test for narrow-lane DMA stranding (`87f9af`): a layer must
    /// NOT gate on a 2-step zero-miss streak, and once gated after warmup must
    /// re-probe within `<= 32` steps.
    #[test]
    fn gate_lane_does_not_gate_on_two_zero_miss_streak_and_reprobes_within_32_steps() {
        let gate = DecodeDmaGate::for_layer(15, false); // maximum phase_offset = 15
        for _ in 0..2 {
            assert!(gate.should_split(1));
            gate.observe(1, 0);
        }
        assert!(
            gate.should_split(1),
            "2 consecutive zero-miss calls must NOT gate a miss-bearing layer"
        );
        assert!(gate.is_splitting());

        // Complete the remaining warmup calls with zero misses to trigger gating.
        for _ in 2..SPLIT_DMA_WARMUP_CALLS {
            assert!(gate.should_split(1));
            gate.observe(1, 0);
        }
        assert!(!gate.is_splitting(), "layer should gate after {SPLIT_DMA_WARMUP_CALLS} zero-miss calls");

        let mut skipped = 0u32;
        while !gate.should_split(1) {
            skipped += 1;
            assert!(skipped <= 32, "gated layer must re-probe within <= 32 steps, got {skipped}");
        }
        assert!(
            (SPLIT_DMA_COOLDOWN_STEPS..=31).contains(&skipped),
            "expected cooldown in {SPLIT_DMA_COOLDOWN_STEPS}..=31, got {skipped}"
        );
    }

    /// Acceptance check (`astra` `mb3c7r`): replaying two (or many) zero-miss
    /// narrow calls followed by a miss-heavy wide call in the same layer must
    /// leave the first wide decision eligible for split DMA regardless of how
    /// many narrow calls preceded it.
    #[test]
    fn gate_lane_narrow_zero_miss_streak_never_suppresses_first_wide_verify_call() {
        let gate = DecodeDmaGate::for_layer(0, false);
        // First 2 zero-miss narrow calls.
        for _ in 0..2 {
            assert!(gate.should_split(1));
            gate.observe(1, 0);
        }
        assert!(
            gate.should_split(WIDE_VERIFY_ROWS),
            "first wide call after 2 zero-miss narrow calls must remain eligible for split DMA"
        );

        // Even after gating the narrow lane completely (20 zero-miss narrow calls),
        // the wide lane's first decision is completely unaffected.
        for _ in 0..20 {
            if gate.should_split(1) {
                gate.observe(1, 0);
            }
        }
        assert!(!gate.should_split(1), "narrow lane is gated");
        assert!(
            gate.should_split(WIDE_VERIFY_ROWS),
            "wide lane must remain eligible for split DMA regardless of narrow-lane gating"
        );
        gate.observe(WIDE_VERIFY_ROWS, 9 * 1024 * 1024);
        assert!(gate.should_split(WIDE_VERIFY_ROWS));
    }
}
