//! Exact per-link and per-bus ledger accounting, lock-free seqlock
//! [`FabricMetrics`] for ~15 ns [`IoPlane::quote`] reads, and [`BusReservation`]
//! for both host-scoped and GPU-fence-bound non-IO bus load.

use crate::fabric::{Fabric, LinkId, SharedBusId};
use crate::fence::{TimelineObserver, TimelinePoint};
use crate::schedule::PriorityClass;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Exact accounting ledger for one physical [`LinkId`], updated under its
/// [`crate::fabric::CoupledSetId`]'s admission lock on every state transition.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LinkLedger {
    pub queued_bytes_by_class: [u64; PriorityClass::COUNT],
    pub queued_ops_by_class: [u32; PriorityClass::COUNT],
    pub submitting_bytes: u64,
    pub submitting_ops: u32,
    pub in_flight_bytes_by_class: [u64; PriorityClass::COUNT],
    pub in_flight_ops_by_class: [u32; PriorityClass::COUNT],
    pub completed_bytes_by_class: [u64; PriorityClass::COUNT],
    pub completed_ops_by_class: [u64; PriorityClass::COUNT],
    pub direct_bytes: u64,
    pub bounced_bytes: u64,
    pub quarantined_ops: u64,
    pub busy_ns: u64,
    pub depth_integral_ns: u128,
    pub exposed_blocking_wait_ns: u64,
    pub hidden_overlap_ns: u64,
    pub contended_ns: u64,
}

impl LinkLedger {
    /// Total active bytes (both `Submitting` and `Issued`) across all classes.
    #[inline]
    pub fn total_in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes_by_class.iter().sum()
    }

    /// Total active ops (both `Submitting` and `Issued`) across all classes.
    #[inline]
    pub fn total_in_flight_ops(&self) -> u32 {
        self.in_flight_ops_by_class.iter().sum()
    }

    /// Total queued bytes across all classes.
    #[inline]
    pub fn total_queued_bytes(&self) -> u64 {
        self.queued_bytes_by_class.iter().sum()
    }

    /// Total queued ops across all classes.
    #[inline]
    pub fn total_queued_ops(&self) -> u32 {
        self.queued_ops_by_class.iter().sum()
    }

    /// Total completed bytes across all classes.
    #[inline]
    pub fn total_completed_bytes(&self) -> u64 {
        self.completed_bytes_by_class.iter().sum()
    }

    /// Speculative (`Prefetch + Background`) in-flight bytes.
    #[inline]
    pub fn speculative_in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes_by_class[PriorityClass::Prefetch.index()]
            + self.in_flight_bytes_by_class[PriorityClass::Background.index()]
    }

    /// Due (`Blocking + Streaming`) in-flight bytes.
    #[inline]
    pub fn due_in_flight_bytes(&self) -> u64 {
        self.in_flight_bytes_by_class[PriorityClass::Blocking.index()]
            + self.in_flight_bytes_by_class[PriorityClass::Streaming.index()]
    }

    /// Advance busy/depth/overlap/contention integrals from `last_at` to `now`.
    pub(crate) fn advance_integrals(&mut self, last_at: &mut Instant, now: Instant) {
        let dt_ns = now.saturating_duration_since(*last_at).as_nanos();
        if now > *last_at {
            *last_at = now;
        }
        if dt_ns == 0 {
            return;
        }
        let total_ops = self.total_in_flight_ops() as u128;
        if total_ops == 0 {
            return;
        }
        let dt_u64 = dt_ns.min(u64::MAX as u128) as u64;
        self.busy_ns = self.busy_ns.saturating_add(dt_u64);
        self.depth_integral_ns = self.depth_integral_ns.saturating_add(total_ops * dt_ns);

        let blocking_in_flight = self.in_flight_ops_by_class[PriorityClass::Blocking.index()] > 0;
        let blocking_queued = self.queued_ops_by_class[PriorityClass::Blocking.index()] > 0;
        let spec_in_flight = self.in_flight_ops_by_class[PriorityClass::Prefetch.index()] > 0
            || self.in_flight_ops_by_class[PriorityClass::Background.index()] > 0;

        if blocking_in_flight || blocking_queued {
            self.exposed_blocking_wait_ns = self.exposed_blocking_wait_ns.saturating_add(dt_u64);
            if spec_in_flight {
                self.contended_ns = self.contended_ns.saturating_add(dt_u64);
            }
        } else if spec_in_flight {
            self.hidden_overlap_ns = self.hidden_overlap_ns.saturating_add(dt_u64);
        }
    }
}

/// Exact accounting ledger for one [`SharedBusId`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BusLedger {
    /// Weighted sum of active IO bytes currently in flight across links coupled to this bus.
    pub in_flight_weighted_bytes: u64,
    /// Aggregate non-IO bandwidth currently reserved (Host-Scoped + unreached Fence-Bound).
    pub reserved_bps: f64,
    /// Number of active host-scoped + fence-bound reservations on this bus.
    pub active_reservations: u32,
}

/// Lock-free snapshot of one Link's live contention metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkMetrics {
    pub effective_bps: f64,
    pub in_flight_bytes: u64,
    pub queued_blocking_ops: u32,
    pub in_flight_ops: u32,
}

/// Lock-free snapshot of one SharedBus's live contention metrics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BusMetrics {
    /// Estimated utilization `U_bus` in `[0.0, 1.0+)` combining active IO + non-IO reservations.
    pub utilization: f64,
    /// Live non-IO reserved bandwidth (bytes/sec).
    pub reserved_bps: f64,
    /// Weighted active IO bytes on this bus.
    pub in_flight_weighted_bytes: u64,
}

/// Signed wall-time transfer quote returned by [`crate::plane::IoPlane::quote`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferQuote {
    pub link: LinkId,
    pub bytes: u64,
    /// Estimated completion time `λ + bytes / B_eff` (seconds).
    pub estimated_s: f64,
    /// Fixed per-submission latency `λ` (seconds).
    pub latency_s: f64,
    /// Live contention-adjusted effective bandwidth `B_eff` (bytes/sec).
    pub effective_bps: f64,
    /// Maximum coupled `SharedBus` utilization `U_bus` affecting this link.
    pub bus_utilization: f64,
    /// Relative measurement uncertainty `σ`.
    pub uncertainty_rel: f64,
}

/// One lock-free 3-word seqlock slot using pure `AtomicU64` (100% sound under
/// Rust's memory model, zero data races, zero allocation).
struct SeqlockSlot3 {
    seq: AtomicU64,
    w0: AtomicU64,
    w1: AtomicU64,
    w2: AtomicU64,
}

impl SeqlockSlot3 {
    fn new(w0: u64, w1: u64, w2: u64) -> Self {
        Self {
            seq: AtomicU64::new(0),
            w0: AtomicU64::new(w0),
            w1: AtomicU64::new(w1),
            w2: AtomicU64::new(w2),
        }
    }

    /// Publish `(w0, w1, w2)`. Caller MUST hold the owning `CoupledSet` lock so
    /// there is strictly one writer per slot at a time.
    #[inline]
    fn write(&self, w0: u64, w1: u64, w2: u64) {
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s.wrapping_add(1), Ordering::Relaxed);
        fence(Ordering::Release);
        self.w0.store(w0, Ordering::Relaxed);
        self.w1.store(w1, Ordering::Relaxed);
        self.w2.store(w2, Ordering::Relaxed);
        fence(Ordering::Release);
        self.seq.store(s.wrapping_add(2), Ordering::Release);
    }

    /// Lock-free consistent read (~10–15 ns).
    #[inline]
    fn read(&self) -> (u64, u64, u64) {
        loop {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let w0 = self.w0.load(Ordering::Relaxed);
            let w1 = self.w1.load(Ordering::Relaxed);
            let w2 = self.w2.load(Ordering::Relaxed);
            fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Relaxed);
            if s1 == s2 {
                return (w0, w1, w2);
            }
            std::hint::spin_loop();
        }
    }
}

/// Lock-free seqlock store for live [`LinkMetrics`] and [`BusMetrics`] across
/// the entire [`Fabric`]. Each `CoupledSet` updates only its own disjoint link
/// and bus slots (zero cross-set contention).
pub struct FabricMetricsStore {
    links: Box<[SeqlockSlot3]>,
    buses: Box<[SeqlockSlot3]>,
}

impl FabricMetricsStore {
    pub fn new(fabric: &Fabric) -> Self {
        let links = fabric
            .links()
            .iter()
            .map(|l| SeqlockSlot3::new(l.curve.peak_bps.to_bits(), 0, 0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let buses = fabric
            .buses()
            .iter()
            .map(|_| SeqlockSlot3::new(0.0f64.to_bits(), 0.0f64.to_bits(), 0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { links, buses }
    }

    #[inline]
    pub(crate) fn publish_link(&self, link_idx: usize, metrics: LinkMetrics) {
        let packed_ops = ((metrics.queued_blocking_ops as u64) << 32) | (metrics.in_flight_ops as u64);
        self.links[link_idx].write(
            metrics.effective_bps.to_bits(),
            metrics.in_flight_bytes,
            packed_ops,
        );
    }

    #[inline]
    pub(crate) fn publish_bus(&self, bus_idx: usize, metrics: BusMetrics) {
        self.buses[bus_idx].write(
            metrics.utilization.to_bits(),
            metrics.reserved_bps.to_bits(),
            metrics.in_flight_weighted_bytes,
        );
    }

    #[inline]
    pub fn read_link(&self, link_idx: usize) -> LinkMetrics {
        let (w0, w1, w2) = self.links[link_idx].read();
        LinkMetrics {
            effective_bps: f64::from_bits(w0),
            in_flight_bytes: w1,
            queued_blocking_ops: (w2 >> 32) as u32,
            in_flight_ops: (w2 & 0xFFFF_FFFF) as u32,
        }
    }

    #[inline]
    pub fn read_bus(&self, bus_idx: usize) -> BusMetrics {
        let (w0, w1, w2) = self.buses[bus_idx].read();
        BusMetrics {
            utilization: f64::from_bits(w0),
            reserved_bps: f64::from_bits(w1),
            in_flight_weighted_bytes: w2,
        }
    }

    /// Lock-free transfer quote for `bytes` on `link`, incorporating live
    /// coupled `SharedBus` utilization and non-IO reservations.
    pub fn quote(&self, fabric: &Fabric, link: LinkId, bytes: u64) -> Option<TransferQuote> {
        let link_idx = fabric.link_index(link)?;
        let spec = &fabric.links()[link_idx];
        let link_m = self.read_link(link_idx);

        let mut max_bus_util = 0.0f64;
        let mut bus_avail_bps = f64::INFINITY;
        let mut uncertainty_sq = spec.curve.uncertainty_rel * spec.curve.uncertainty_rel;

        for coupling in &spec.couplings {
            if let Some(bus_idx) = fabric.bus_index(coupling.bus) {
                let bus_spec = &fabric.buses()[bus_idx];
                let bus_m = self.read_bus(bus_idx);
                max_bus_util = max_bus_util.max(bus_m.utilization);
                let remaining = (bus_spec.peak_bps - bus_m.reserved_bps).max(bus_spec.peak_bps * 0.05);
                let per_weight = remaining / (coupling.weight as f64);
                bus_avail_bps = bus_avail_bps.min(per_weight);
                uncertainty_sq += bus_spec.uncertainty_rel * bus_spec.uncertainty_rel;
            }
        }

        let effective_bps = link_m
            .effective_bps
            .min(spec.curve.peak_bps)
            .min(bus_avail_bps)
            .max(1.0);
        let estimated_s = spec.curve.latency_s + (bytes as f64) / effective_bps;

        Some(TransferQuote {
            link,
            bytes,
            estimated_s,
            latency_s: spec.curve.latency_s,
            effective_bps,
            bus_utilization: max_bus_util,
            uncertainty_rel: uncertainty_sq.sqrt(),
        })
    }
}

/// Callback invoked when a host-scoped [`BusReservation`] drops.
pub(crate) type BusReleaseFn = Box<dyn FnOnce() + Send>;

/// Reservation of non-IO bandwidth on a [`SharedBusId`] (e.g., CPU-site expert
/// compute reading `HostDdr`, GPU SM zero-copy reading `PcieRoot` + `HostDdr`,
/// or dense GPU compute reading `UnifiedFabric`).
///
/// - **Host-Scoped (`IoPlane::reserve_bus_scoped`)**: debits on creation and
///   credits when this guard is dropped.
/// - **Fence-Bound (`IoPlane::reserve_bus_until_fence`)**: debits on creation
///   and credits ONLY when the [`TimelinePoint`] is observed reached by its
///   [`TimelineObserver`]; dropping the returned [`FenceReservationId`] or
///   returning from the host submit function does NOT credit it.
pub struct BusReservation {
    pub bus: SharedBusId,
    pub rate_bps: f64,
    on_drop: Option<BusReleaseFn>,
}

impl BusReservation {
    pub(crate) fn host_scoped(bus: SharedBusId, rate_bps: f64, on_drop: BusReleaseFn) -> Self {
        Self {
            bus,
            rate_bps,
            on_drop: Some(on_drop),
        }
    }
}

impl Drop for BusReservation {
    fn drop(&mut self) {
        if let Some(release) = self.on_drop.take() {
            release();
        }
    }
}

/// Identifier of an active GPU-fence-bound bus reservation owned by a
/// `CoupledSet` until its [`TimelinePoint`] is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FenceReservationId(pub u64);

/// Internal record of a fence-bound non-IO bus reservation.
pub(crate) struct FenceBusReservation {
    pub(crate) bus: SharedBusId,
    pub(crate) rate_bps: f64,
    pub(crate) point: TimelinePoint,
    pub(crate) observer: Arc<dyn TimelineObserver>,
}
