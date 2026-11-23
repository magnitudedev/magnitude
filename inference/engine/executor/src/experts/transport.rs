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
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
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
    pub transfer_id: TransferId,
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

/// Outcome of [`ExpertTransport::land_decode_misses`]: all cross-stream sync
/// events `DecodeBack` must wait on (`cuStreamWaitEvent`), all consumer
/// leases pinning the landed/joined VRAM slots until `DecodeBack` finishes,
/// and the freshly landed `DemandSlot`s for O(1) zero-recopy promotion.
pub(crate) struct DecodeLanding {
    pub layer: usize,
    pub sequence: u64,
    pub syncs: Vec<TransferSync>,
    pub leases: Vec<ConsumerLease>,
    /// Demand transfers freshly issued on unified `DemandLanding` slots.
    demand_ids: Vec<TransferId>,
    /// Freshly landed unified VRAM slots to promote O(1) or release on completion.
    pub demand_slots: Vec<DemandSlot>,
}

impl DecodeLanding {
    /// Release all transport consumer leases, mark freshly issued demand
    /// transfers complete, drain deferred syncs outside all residency locks,
    /// and return `(layer, sequence, demand_slots)` for O(1) promotion or
    /// release in `Directory`.
    pub(crate) fn release_transport(self, transport: &ExpertTransport) -> (usize, u64, Vec<DemandSlot>) {
        for id in self.demand_ids {
            transport.complete_dma(id);
        }
        for lease in self.leases {
            lease.release();
        }
        transport.drain_deferred_syncs();
        (self.layer, self.sequence, self.demand_slots)
    }

    #[cfg(test)]
    pub(crate) fn release(self, transport: &ExpertTransport) {
        let _ = self.release_transport(transport);
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

/// Physics break-even threshold for splitting a decode layer into
/// `DecodeFront -> flush_and_wait -> DMA -> DecodeBack` vs running pipelined
/// `RunStage::Whole` (which already gets `LinkMover` `MAP_STAGE` hits and falls
/// back to SM zero-copy for occasional misses with zero per-layer host sync).
///
/// At ~11 GB/s SM zero-copy vs ~45 GB/s hardware DMA, 4 MiB saves ~0.28 ms of
/// PCIe wire time, breaking even with the split-graph + `flush_and_wait()`
/// pipeline cost. Layers well above this (e.g. GLM-5.3 at ~97 MB/layer) stay
/// 100% on M7 Decode DMA; layers far below (e.g. Qwen3.8 at ~0.7 MB/layer,
/// 67% zero-miss) stay on pipelined `RunStage::Whole` and probe periodically.
const SPLIT_DMA_MIN_BYTES: u64 = 4 * 1024 * 1024;
const SPLIT_DMA_WARMUP_CALLS: u32 = 8;
const SPLIT_DMA_COOLDOWN_STEPS: u32 = 256;
const SPLIT_DMA_STAGGER_SLOTS: u32 = 64;
const SPLIT_DMA_STAGGER_STRIDE: u32 = 4;

/// Row-width threshold separating narrow decode (`M = 1..3`) from wide
/// speculative verify bursts (`M >= 4`) in [`DecodeDmaGate`]. Separate lanes
/// ensure an `M = 1..2` warmup cooldown never strands an `M = 4..16` verify
/// burst on ~9.5 GB/s SM zero-copy PCIe.
const WIDE_VERIFY_ROWS: u64 = 4;

#[derive(Default)]
struct GateLane {
    /// Exponential moving average of un-staged host-tier miss bytes per call
    /// (alpha = 1/4: `ewma = (ewma * 3 + sample) / 4`).
    ewma_bytes: AtomicU64,
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
    fn new(phase_offset: u32) -> Self {
        Self { phase_offset, ..Self::default() }
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
        let prev_calls = self.calls.fetch_add(1, Ordering::Relaxed);
        let ewma = if prev_calls == 0 {
            miss_bytes
        } else {
            let old = self.ewma_bytes.load(Ordering::Relaxed);
            (old.saturating_mul(3).saturating_add(miss_bytes)) / 4
        };
        self.ewma_bytes.store(ewma, Ordering::Relaxed);
        if prev_calls + 1 >= SPLIT_DMA_WARMUP_CALLS && ewma < SPLIT_DMA_MIN_BYTES && miss_bytes < SPLIT_DMA_MIN_BYTES {
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
#[derive(Default)]
struct DecodeDmaGate {
    narrow: GateLane,
    wide: GateLane,
}

impl DecodeDmaGate {
    fn for_layer(layer: usize) -> Self {
        let phase_offset = ((layer as u32) % SPLIT_DMA_STAGGER_SLOTS) * SPLIT_DMA_STAGGER_STRIDE;
        Self {
            narrow: GateLane::new(phase_offset),
            wide: GateLane::new(phase_offset),
        }
    }

    fn lane(&self, rows: u64) -> &GateLane {
        if rows >= WIDE_VERIFY_ROWS {
            &self.wide
        } else {
            &self.narrow
        }
    }

    fn is_splitting(&self) -> bool {
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
        self.lane(rows).should_split()
    }

    fn observe(&self, rows: u64, miss_bytes: u64) {
        self.lane(rows).observe(miss_bytes);
    }
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
    /// Shared watcher event channel woken when `Answer::publish` / `decode::fail`
    /// answers a file slot and notified immediately when `land_decode_misses`
    /// sees a pending file request.
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
        num_layers: usize,
        files: Vec<super::ExchangeTable>,
        events: Arc<super::events::Events>,
    ) -> Self {
        Self {
            transfer,
            host_stores,
            bank_stores,
            sources,
            sources_words,
            selected,
            layer_gates: (0..num_layers).map(DecodeDmaGate::for_layer).collect(),
            files,
            events,
        }
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

#[derive(Default)]
struct PrefillRendezvous {
    /// Plans published by `Watcher` (`Scratch::fix_rounds`) keyed by
    /// `(layer, sequence, window)`.
    plans: HashMap<(usize, u64, u64), PrefillPlan>,
    /// Steps that failed before round 0 (`scratch::fail`).
    failed_steps: HashMap<u64, String>,
    /// Fatal serving error (`Shared::end_serving`).
    ended: Option<String>,
}

/// Unified Copy-Engine-First `ExpertTransport`.
pub(crate) struct ExpertTransport {
    registry: Mutex<Registry>,
    prefill: Mutex<PrefillRendezvous>,
    prefill_ready: Condvar,
    decode: Option<DecodeDmaState>,
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
            decode: None,
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
            decode,
            transfers_issued: AtomicU64::new(0),
            demand_joins: AtomicU64::new(0),
            prefill_dma_batches: AtomicU64::new(0),
            prefill_dma_skipped: AtomicU64::new(0),
        })
    }

    /// Whether on-demand decode DMA should split `layer` into
    /// `DecodeFront -> flush_and_wait -> DMA -> DecodeBack` on this step
    /// (returns `false` when the layer's adaptive miss-bytes EWMA is below
    /// `SPLIT_DMA_MIN_BYTES`, keeping low-miss layers on pipelined `Whole`).
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

    /// Mark a transfer faulted and quarantine its destination slot so neither
    /// joiners nor future allocations can observe partial/corrupt bytes.
    pub(crate) fn fault_transfer(&self, id: TransferId, _reason: String) {
        let mut reg = self.registry.lock().expect("transport registry");
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
            transfer_id: id,
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
            transfer_id: id,
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
    /// - **Phase B (overlapped with Phase A PCIe DMA):** when `DecodeFront`
    ///   posted a file request (`file.requested(sequence, 0)`), wake `Requests`
    ///   immediately and wait on `events` until `file.answered(sequence, 0)`
    ///   (or serving ends), resolve each absent choice from `FILE_RECORD`
    ///   (exact mirror of `streaming/tier_wait.hpp`, promoting any CPU-pending
    ///   `SOURCE_SITE == -2` to GPU DMA since `routed_site_request` already
    ///   ran in `DecodeFront`), and fire a second batched `cuMemcpyHtoDAsync`
    ///   on the same FIFO DMA stream for the newly-landed host-tier misses.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn land_decode_misses(
        self: &Arc<Self>,
        group: usize,
        layer: usize,
        sequence: u64,
        rows: u64,
        mut claim_slot: impl FnMut(usize, u32) -> super::directory::DemandSlotClaim,
        mut abort_slots: impl FnMut(&[(usize, usize)]),
        mut replenish_free: impl FnMut(usize, usize),
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

        let region_bytes = dma.bank_stores.region_bytes(group);
        let expert_bytes: u64 = region_bytes.iter().copied().sum();

        let mut resolved: Vec<(usize, usize)> = Vec::with_capacity(unique_a.len());
        let mut latest_joined_sync: Option<(TransferId, TransferSync)> = None;
        let mut latest_dma_sync: Option<TransferSync> = None;
        let mut leases: Vec<ConsumerLease> = Vec::new();
        let mut demand_ids: Vec<TransferId> = Vec::new();
        let mut demand_slots: Vec<DemandSlot> = Vec::new();
        let mut bytes_landed: u64 = 0;
        let mut overflow_experts: u64 = 0;

        // Shared helper: join existing transfers or claim unified `Record::Free`
        // slots via `claim_slot`, then issue one batched `copy_synced` on
        // `dma.transfer` outside all residency locks.
        let mut issue_batch =
            |mut unique: Vec<(usize, usize)>, resolved: &mut Vec<(usize, usize)>| -> Result<(), String> {
                if unique.is_empty() {
                    return Ok(());
                }
                // Prioritize highest row-reference count in case free slots are scarce.
                unique.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let mut pairs: Vec<(Tensor, Tensor, u64)> =
                    Vec::with_capacity(unique.len() * region_bytes.len());
                // `(tier_record, slot, staged)`
                let mut fresh_landed: Vec<(usize, usize, Option<(usize, usize)>)> =
                    Vec::with_capacity(unique.len());
                let mut aborted: Vec<(usize, usize)> = Vec::new();
                for (tier_record, _hits) in unique {
                    if resolved.iter().any(|&(tr, _)| tr == tier_record) {
                        continue;
                    }
                    if let Some(joined) = self.join_by_tier_record(group, layer, tier_record) {
                        resolved.push((tier_record, joined.dest.slot));
                        if let Some(sync) = joined.sync {
                            if latest_joined_sync
                                .as_ref()
                                .is_none_or(|(prev_id, _)| joined.transfer_id.0 > prev_id.0)
                            {
                                latest_joined_sync = Some((joined.transfer_id, sync));
                            }
                        }
                        leases.push(joined.lease);
                        continue;
                    }
                    match claim_slot(group, tier_record as u32) {
                        super::directory::DemandSlotClaim::AlreadyHeld { slot, .. } => {
                            resolved.push((tier_record, slot));
                        }
                        super::directory::DemandSlotClaim::Allocated { staged, slot } => {
                            let mut complete = true;
                            let before = pairs.len();
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
                                pairs.push((src, dst, bytes));
                            }
                            if complete {
                                fresh_landed.push((tier_record, slot, staged));
                            } else {
                                pairs.truncate(before);
                                aborted.push((group, slot));
                            }
                        }
                        super::directory::DemandSlotClaim::Overflow { .. } => {
                            overflow_experts += 1;
                        }
                    }
                }
                if !aborted.is_empty() {
                    abort_slots(&aborted);
                }
                if fresh_landed.is_empty() {
                    return Ok(());
                }
                let batch_bytes: u64 = pairs.iter().map(|(_, _, bytes)| *bytes).sum();
                let mut batch_ids = Vec::with_capacity(fresh_landed.len());
                let mut batch_slots = Vec::with_capacity(fresh_landed.len());
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
                    leases.push(lease);
                }
                let items: Vec<TransferItem<'_>> = pairs
                    .iter()
                    .map(|(source, destination, bytes)| {
                        TransferItem::Run(TransferRun {
                            source,
                            source_offset: 0,
                            destination,
                            destination_offset: 0,
                            bytes: *bytes,
                        })
                    })
                    .collect();
                match dma.transfer.copy_synced(&items) {
                    Ok((_, sync)) => {
                        for &id in &batch_ids {
                            self.attach_sync(id, sync.clone());
                        }
                        latest_dma_sync = Some(sync);
                        demand_ids.extend_from_slice(&batch_ids);
                        demand_slots.extend_from_slice(&batch_slots);
                        for &(tier_record, slot, _) in &fresh_landed {
                            resolved.push((tier_record, slot));
                        }
                        bytes_landed += batch_bytes;
                        Ok(())
                    }
                    Err(error) => {
                        let msg = format!("M7 decode DMA copy failed: {error}");
                        for &id in &batch_ids {
                            self.fault_transfer(id, msg.clone());
                        }
                        let faulted_slots: Vec<(usize, usize)> =
                            fresh_landed.iter().map(|&(_, slot, _)| (group, slot)).collect();
                        abort_slots(&faulted_slots);
                        Err(msg)
                    }
                }
            };

        // Phase A: fire PCIe DMA immediately for already-in-RAM misses before
        // waiting on any NVMe file answer.
        let total_unique_a = unique_a.len();
        issue_batch(unique_a, &mut resolved)?;
        for (idx, &word) in raw_sources.iter().enumerate() {
            if word <= -2 {
                let tier_record = (-2i64 - i64::from(word)) as usize;
                if let Some(&(_, slot)) = resolved.iter().find(|&&(tr, _)| tr == tier_record) {
                    dma.sources_words.store(idx, slot as i32);
                }
            }
        }

        // Phase B: overlap host-side `FILE_ANSWER` wait with Phase A PCIe DMA,
        // then DMA newly-landed NVMe misses on the same FIFO transfer stream.
        let mut total_unique_b = 0usize;
        if file_pending {
            let file = file.expect("file_pending implies layer has a file table");
            loop {
                let seen = dma.events.seen();
                if file.answered(sequence, 0) || is_serving_ended() {
                    break;
                }
                dma.events.wait_at_most(seen, std::time::Duration::from_millis(5));
            }
            let answered = matches!(
                file.answer_of(sequence, 0),
                Some(super::exchange::Answer::Answered)
            );
            let file_count = usize::try_from(
                file.load(sequence, 0, magnitude_kernels::constants::FILE_COUNT),
            )
            .unwrap_or(0)
            .min(count);
            let mut phase_b_choices: Vec<(usize, usize)> = Vec::new(); // (choice, tier_record)
            let mut unique_b: Vec<(usize, usize)> = Vec::new();
            for choice in 0..file_count {
                let entry = magnitude_kernels::constants::FILE_ENTRY(choice as u64);
                let expert = file.load(sequence, 0, entry + magnitude_kernels::constants::FILE_CHOICE);
                if expert < 0 {
                    continue;
                }
                let source = if answered {
                    file.load(sequence, 0, entry + magnitude_kernels::constants::FILE_RECORD)
                } else {
                    -1
                };
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
                    phase_b_choices.push((choice, tier_record));
                    if let Some(entry) = unique_b.iter_mut().find(|(tr, _)| *tr == tier_record) {
                        entry.1 += 1;
                    } else {
                        unique_b.push((tier_record, 1));
                    }
                }
            }
            total_unique_b = unique_b.len();
            if !unique_b.is_empty() {
                issue_batch(unique_b, &mut resolved)?;
                for &(choice, tier_record) in &phase_b_choices {
                    if let Some(&(_, slot)) = resolved.iter().find(|&&(tr, _)| tr == tier_record) {
                        dma.sources_words.store(choice, slot as i32);
                    }
                }
            }
        }

        if overflow_experts > 0 || !resolved.is_empty() {
            // Proactively maintain a small active floor of free demand slots
            // (K for narrow decode, 2*K for wide verify bursts) so cold-miss
            // bursts don't overflow to SM zero-copy while LinkMover::reserve == 0.
            let target_free = dma.selected.saturating_mul(if rows >= WIDE_VERIFY_ROWS { 2 } else { 1 });
            replenish_free(group, (overflow_experts as usize).max(target_free));
        }

        if let Some(gate) = dma.layer_gates.get(layer) {
            gate.observe(rows, ((total_unique_a + total_unique_b) as u64).saturating_mul(expert_bytes));
        }

        if resolved.is_empty() {
            record_dma(true, 0, 0, overflow_experts);
            return Ok(None);
        }

        record_dma(false, resolved.len() as u64, bytes_landed, overflow_experts);

        let mut syncs: Vec<TransferSync> = Vec::with_capacity(2);
        if let Some((_, sync)) = latest_joined_sync {
            syncs.push(sync);
        }
        if let Some(sync) = latest_dma_sync {
            syncs.push(sync);
        }

        Ok(Some(DecodeLanding {
            layer,
            sequence,
            syncs,
            leases,
            demand_ids,
            demand_slots,
        }))
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
        assert_eq!(joined_1.transfer_id, hint_id);
        assert_eq!(joined_1.dest, dest);
        assert_eq!(transport.transfers_issued(), 1, "joining must NOT issue a second transfer");
        assert_eq!(transport.demand_joins(), 1);

        // 3. Consumer 2 (demand) also joins the SAME in-flight hint.
        let joined_2 = transport
            .join_by_key(key, source)
            .expect("consumer 2 must join same in-flight hint");
        assert_eq!(joined_2.transfer_id, hint_id);
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
        let high = DecodeDmaGate::for_layer(0);
        for _ in 0..32 {
            assert!(high.should_split(1), "high-miss layer must always use split DMA");
            assert!(high.is_splitting(), "high-miss layer must report splitting");
            high.observe(1, 64 * 1024 * 1024);
        }

        // Low-miss regime (Qwen3.8: 0.7 MB/layer alternating with 0-miss calls).
        let low = DecodeDmaGate::for_layer(0);
        for i in 0..SPLIT_DMA_WARMUP_CALLS {
            assert!(low.should_split(1), "warmup call {i} should measure split DMA");
            low.observe(1, if i % 3 == 0 { 1_400_000 } else { 0 });
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
        let gate = DecodeDmaGate::for_layer(3);
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
    /// with distinct phase offsets so no two layers exit cooldown on the exact
    /// same step (preventing a 48-layer synchronous probe storm).
    #[test]
    fn decode_dma_gate_staggers_probes_across_48_layers() {
        let gates: Vec<DecodeDmaGate> = (0..48).map(DecodeDmaGate::for_layer).collect();
        for _ in 0..SPLIT_DMA_WARMUP_CALLS {
            for gate in &gates {
                assert!(gate.should_split(1));
                gate.observe(1, 0);
            }
        }
        let total_steps = (SPLIT_DMA_COOLDOWN_STEPS as usize) * 3;
        let mut total_probes = 0usize;
        for step in 0..total_steps {
            let mut probing_this_step = 0usize;
            for gate in &gates {
                if gate.should_split(1) {
                    probing_this_step += 1;
                    total_probes += 1;
                    gate.observe(1, 0);
                }
            }
            assert!(
                probing_this_step <= 1,
                "step {step} had {probing_this_step} layers probing simultaneously (expected <= 1)"
            );
        }
        assert!(
            total_probes >= 48 * 2,
            "expected every layer to probe at least twice over {total_steps} steps, got {total_probes}"
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
            1,
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
                |_g, tr| {
                    if let Some(slot) = free_pool.pop() {
                        crate::experts::directory::DemandSlotClaim::Allocated {
                            staged: Some((0, tr as usize)),
                            slot,
                        }
                    } else {
                        crate::experts::directory::DemandSlotClaim::Overflow { staged: None }
                    }
                },
                |_| {},
                |_, _| {},
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
                |_, _| crate::experts::directory::DemandSlotClaim::Overflow { staged: None },
                |_| {},
                |_, _| {},
                || {},
                || false,
                |_, _, _, _| {},
            )
            .expect("failed answer ok");
        assert!(failed.is_none());
        assert_eq!(sources_words.load(0), -1);
        assert_eq!(sources_words.load(site_base + 0), -1);
    }
}
