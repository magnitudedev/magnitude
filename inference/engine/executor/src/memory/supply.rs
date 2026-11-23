//! What a store can give the process and where others' working set lies
//! (design §4.2, §4.3).
//!
//! - **One reading per store** ([`Reading`]): the store's total, its supply
//!   (`free_beyond_target + clean_cache + cold_reclaimable`: bytes the OS
//!   gives without writing anyone's dirty data), on unified Metal its wired
//!   ceiling, on host RAM its paging signals, and the store's lag. Built
//!   from one host sample ([`seismic::HostSample`], which panics naming any
//!   figure it cannot read) or from one device grant. A figure that is not
//!   one (supply past the total) is an invariant violation naming it.
//! - **Supply is attributed** by the journal (`memory::journal`): what the
//!   store may let the process hold is what the sample shows of it beside
//!   the sample's supply, so the process's own release is never new supply
//!   and its own growth never taken twice.
//! - **The boundary** ([`Boundary`]): prevents breaking other apps (sustained
//!   active thrashing under scarcity and OS memory pressure) while allowing
//!   normal OS compression and lazy background page wakeups.
//!   - Others' harm only: the machine's pages in less the process's own,
//!     counted together in one snapshot ([`seismic::Harm`]), judged on the
//!     counters' own clock ([`seismic::Counted`]) and only over spans of at
//!     least the counters' lag, so a share of the process's own that a fold
//!     shows late (Linux memcg's flush) is never read as others'. Where the
//!     OS counts no share of the process's, harm cannot be judged: growth
//!     is bounded by supply and OS pressure there.
//!   - Normal OS compression is not contention: occasional or modest lazy
//!     decompressions/refaults, refaults while immediate free + clean cache
//!     exceeds the host scarcity cushion, and decompressions during exogenous
//!     memory releases (`free` or `room` jumping) set no cap and trigger no
//!     shrink.
//!   - Sustained thrashing under scarcity (`rate >= THRASH_RATE_BPS` across
//!     consecutive judged spans spanning at least `THRASH_CONFIRM_WINDOW`
//!     while immediate free + clean cache sits at or below the scarcity
//!     cushion and others are not freeing memory):
//!     - During an open growth step (`allocated > last.allocated` within
//!       `lag`), marks B at the holding before the step began (`step.from`).
//!     - Outside a growth step, lowers B by the sustained thrash excess, once
//!       per confirmed streak, tightening an existing cap in place without
//!       snapping down to lower held bytes.
//!   - Regrowth past B never clears B to `None` across an unverified cold
//!     overhang:
//!     - Exogenous free release (`held + free` rising above `cap.anchor` by
//!       `min_gain`, invariant under our own shrinks since `-Δheld + Δfree == 0`
//!       and immune to inactive `cold` page re-aging) raises B by that exact
//!       gain, clearing B to `None` only when no unverified cold overhang sits
//!       above the raised ceiling.
//!     - On judged OSes, sustained uninterrupted calm (`CALM_PROBE_BASE` with
//!       exponential backoff per consecutive wall strike) raises B by one
//!       bounded probe increment, never jumping straight to `None`.
//!   - Each OS pressure event (`seismic::PressureEvents`) while pressure is
//!     above `Normal` lowers B by one block (`unit`) per event; if pressure
//!     persists silently under immediate scarcity (`free <= scarcity_threshold`)
//!     without new edge events, `Critical` steps B down every `250 ms` on every
//!     OS, while standing `Warn` steps down every `1 s` only when the OS reports
//!     no harm counters (level-triggered OSes such as Windows
//!     `LowMemoryResourceNotification`). On macOS and judged Linux, standing
//!     `Warn` with `events == 0` never drains cache on its own. B never falls
//!     below committed memory.

use std::time::{Duration, Instant};

/// Sustained machine-minus-own paging rate above which compressed/swapped
/// working sets are actively thrashing rather than lazily waking up (64 MiB/s).
const THRASH_RATE_BPS: f64 = (64 * (1 << 20)) as f64;

/// Minimum elapsed time across consecutive thrashing spans (>= 2 spans) before
/// confirming contention so a single tab switch or process exit never locks `cap`.
const THRASH_CONFIRM_WINDOW: Duration = Duration::from_secs(1);

/// Increase in `free` or `room` across a span indicating an exogenous memory
/// release (another app quitting or freeing allocations) rather than scarcity contention.
const EXOGENOUS_RELIEF_TOLERANCE: u64 = 64 * (1 << 20);

/// Dwell intervals before a standing OS pressure level with no edge events
/// steps `cap` down by one block (`unit`).
const STANDING_WARN_DWELL: Duration = Duration::from_secs(1);
const STANDING_CRITICAL_DWELL: Duration = Duration::from_millis(250);

/// Base uninterrupted calm duration before probing a standing `Harm` cap upward
/// by one bounded increment; doubles on each consecutive probe strike up to 16x.
const CALM_PROBE_BASE: Duration = Duration::from_secs(30);
const CALM_PROBE_MAX_SHIFT: u32 = 4;
const CALM_PROBE_MIN_BYTES: u64 = 256 * (1 << 20);

/// Derived immediate-supply scarcity threshold in bytes:
///
/// `reclaim_band + unit + others_step`
///
/// - `reclaim_band`: the OS's own active reclaim band above the floor already
///   subtracted from `Supply::free_beyond_target` (Linux `/proc/zoneinfo`
///   `Σ(high - low)`, macOS `vm.vm_page_background_target - free_target`,
///   Windows `0` since `windows_floor` is already subtracted). Never
///   double-counts the floor already removed from `free_beyond_target`.
/// - `unit`: the smallest block we can give back to the OS (one pool granule /
///   expert slot).
/// - `others_step`: the competing-process allocation step measured by the
///   ledger (`guarantee`: others' attributed rise between judged readings).
///
/// When immediate non-swapping supply (`free_beyond_target + clean_cache`) is
/// at or below this threshold, either one more block of ours or one more step
/// of competing processes pushes the OS into its active reclaim band.
pub(crate) fn scarcity_threshold(reclaim_band: u64, unit: u64, others_step: u64) -> u64 {
    reclaim_band.saturating_add(unit).saturating_add(others_step)
}

/// Elastic VRAM cache reserve for discrete CUDA devices so background
/// expert-bank growth leaves enough free VRAM for an external process to
/// retain a CUDA primary context (`cuDevicePrimaryCtxRetain`, ~258 MiB on A10G
/// and ~300–450 MiB on RTX 4090/5090 or Seismic device open) and trigger
/// Magnitude's live shrink path instead of failing immediately with
/// `CUDA_ERROR_OUT_OF_MEMORY` (F-050).
///
/// Applied strictly to elastic bank cache sizing (`import_room`, `hold_room`,
/// `target`), never to `budget()` / `room()` / resident model admission.
pub(crate) fn cuda_context_cache_reserve(capacity: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if capacity < 4 * GIB {
        return (capacity / 8).min(384 * MIB);
    }
    (capacity / 32).clamp(384 * MIB, 512 * MIB)
}

/// One store's figures at one sample: the ledger's one OS input
/// ([`super::ledger::Ledger::read`]). Fields are private: `supply ≤ total`
/// holds by construction.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    total: u64,
    supply: seismic::Supply,
    /// The OS's active reclaim band above the floor already subtracted from
    /// `supply.free_beyond_target` (`0` on non-host device stores).
    reclaim_band: u64,
    /// Elastic-cache-only headroom reserved below `budget()` on discrete device
    /// stores (e.g. CUDA primary-context headroom for external processes); `0`
    /// on host stores and non-CUDA devices. Never subtracted from `budget()` or
    /// resident admission `room()`.
    cache_reserve: u64,
    /// A unified device's grant over host RAM (§4.3).
    grant: Option<HostGrant>,
    /// Host RAM's paging signals, where the store is host RAM.
    paging: Option<Paging>,
    /// The process's anonymous bytes the OS holds outside RAM, where the
    /// store is host RAM and the OS reports them
    /// ([`seismic::HostSample::own_reclaimed`]).
    own_reclaimed: Option<seismic::OwnReclaimed>,
    lag: seismic::Lag,
}

/// What a unified device grants the process of host RAM, where a backend
/// or OS limit bounds it beside host RAM's supply (§4.3).
#[derive(Clone, Copy, Debug)]
pub(crate) enum HostGrant {
    /// macOS Metal's kernel wired ceiling and current wire count, optionally
    /// paired with the live working-set headroom reported by the device
    /// (`recommendedMaxWorkingSetSize - currentAllocatedSize`) so uncharged
    /// Metal resources (pipelines, command buffers, driver heaps) never hide
    /// from the wired ceiling.
    Wired { ceiling: Ceiling, left: Option<u64> },
    /// A static byte ceiling on device-accessible host RAM (unified CUDA's
    /// total or usable memory).
    Limit(u64),
    /// A Vulkan heap budget on host RAM: `left` bytes still granted beyond
    /// current heap usage, bounded by `capacity`.
    Budget { capacity: u64, left: u64 },
}

/// The system's wired ceiling and what is wired now, the process's
/// included, at the instant `wire_count` was sampled.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ceiling {
    pub(crate) limit: u64,
    pub(crate) wired: u64,
    pub(crate) at: Instant,
}

impl Ceiling {
    /// Execution headroom reserved below Metal's wired working-set ceiling
    /// (`recommendedMaxWorkingSetSize` or `iogpu.wired_limit_mb`) for command
    /// buffers, residency-set commits, driver heaps, and transient kernel/IOGPU
    /// wiring. Subtracted from `Read::grant`, `Reading::capacity`, and
    /// `MemoryBudget::stable` / `live_status` so committed claims
    /// (`PromptScratch`, KV slabs, graph slots) cannot consume it and static
    /// assessment agrees with live load headroom.
    ///
    /// `max(256 MiB, limit / 20)` reserves ~573 MB on a 16 GB M1 (11.45 GB
    /// working set), ~1.34 GB on a 32 GB M2 Pro (26.80 GB working set), and
    /// ~1.93 GB on a 48 GB M4 Pro (38.65 GB working set).
    pub(crate) fn execution_reserve_for(limit: u64) -> u64 {
        const MIN_RESERVE: u64 = 256 << 20;
        (limit / 20).max(MIN_RESERVE).min(limit)
    }

    pub(crate) fn execution_reserve(&self) -> u64 {
        Self::execution_reserve_for(self.limit)
    }
}

/// Host RAM's counters at a sample, as of when they show the machine
/// (`counted`): cumulative pages brought in, the machine's and the
/// process's own, of `page_bytes` each, where the OS counts the process's
/// share; others' cold anonymous bytes ([`others_cold`]); and the OS's
/// pressure level, where it reports one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Paging {
    pub(crate) harm: Option<seismic::Harm>,
    pub(crate) page_bytes: u64,
    pub(crate) cold: u64,
    /// The process's own cold bytes, where the OS counts them: what the
    /// ledger leaves out of cold supply (`memory::ledger`).
    pub(crate) own_cold: Option<u64>,
    pub(crate) pressure: Option<seismic::PressureLevel>,
    pub(crate) counted: seismic::Counted,
}

impl Reading {
    /// Host RAM's reading from `sample`.
    ///
    /// # Panics
    /// Where the sample's supply exceeds its total: no reading.
    pub fn host(sample: &seismic::HostSample) -> Self {
        let supply = sample.supply.total();
        assert!(supply <= sample.total, "invariant violated: host RAM's supply reads {supply} of {} total", sample.total);
        Self {
            total: sample.total,
            supply: sample.supply,
            reclaim_band: sample.reclaim_band,
            cache_reserve: 0,
            grant: None,
            paging: Some(Paging {
                harm: sample.harm,
                page_bytes: sample.page_bytes,
                cold: others_cold(sample),
                own_cold: sample.cold.own,
                pressure: sample.pressure,
                counted: sample.counted,
            }),
            own_reclaimed: sample.own_reclaimed,
            lag: sample.lag,
        }
    }

    /// Unified Metal's reading from `sample`: host RAM's, its grant bounded
    /// by `working_set`, the device's `recommendedMaxWorkingSetSize`. On
    /// Apple silicon the bound is the wired ceiling: the wired limit set
    /// ([`seismic::WiredLimit::Set`]) within `working_set`, or the OS's
    /// default, which `working_set` reports
    /// ([`seismic::WiredLimit::OsDefault`]). Where the GPU wires no host RAM
    /// (the sample has no wired figures: an Intel Mac's integrated GPU, no
    /// IOGPU stack), `working_set` alone bounds it, a static ceiling within
    /// host RAM's supply. Its release lag is the host's, none: a pool
    /// block's buffer is made purgeable-Empty before it is released, which
    /// discards its pages in that call, so the free shows at once.
    pub fn unified(sample: &seismic::HostSample, working_set: u64) -> Self {
        Self::unified_with_left(sample, working_set, None)
    }

    /// [`Self::unified`] with the live device working-set headroom `left`
    /// (`recommendedMaxWorkingSetSize - currentAllocatedSize`), when sampled
    /// from an opened Metal device.
    pub fn unified_with_left(sample: &seismic::HostSample, working_set: u64, left: Option<u64>) -> Self {
        let grant = match sample.wired {
            Some(wired) => {
                let limit = match wired.limit {
                    seismic::WiredLimit::OsDefault => working_set,
                    seismic::WiredLimit::Set(bytes) => bytes.min(working_set),
                };
                HostGrant::Wired {
                    ceiling: Ceiling { limit, wired: wired.wire_count, at: wired.at },
                    left: left.map(|left| left.min(working_set)),
                }
            }
            None => match left {
                Some(left) => HostGrant::Budget { capacity: working_set, left: left.min(working_set) },
                None => HostGrant::Limit(working_set),
            },
        };
        Self { grant: Some(grant), ..Self::host(sample) }
    }

    /// Unified host RAM bounded by a static device ceiling `limit` (unified
    /// CUDA's total or usable memory on Linux or Windows).
    pub fn host_limit(sample: &seismic::HostSample, limit: u64) -> Self {
        Self { grant: Some(HostGrant::Limit(limit)), ..Self::host(sample) }
    }

    /// Unified host RAM bounded by a Vulkan heap budget (`left` left of
    /// `capacity` after current heap usage, reconciled in step with the
    /// driver's own memory).
    ///
    /// # Panics
    /// Where the grant leaves more than `capacity`.
    pub fn host_budget(sample: &seismic::HostSample, capacity: u64, left: u64) -> Self {
        assert!(left <= capacity, "invariant violated: a host-memory device grant leaves {left} of its {capacity} bytes");
        Self { grant: Some(HostGrant::Budget { capacity, left }), ..Self::host(sample) }
    }

    /// A device store's reading: `free` of `capacity` its grant leaves
    /// (CUDA's free, Vulkan's budget less usage), its changes showing after
    /// `lag` (the device's, held by the machine's assessment).
    ///
    /// # Panics
    /// Where the grant leaves more than the store's capacity.
    pub fn device(capacity: u64, free: u64, lag: seismic::Lag) -> Self {
        assert!(free <= capacity, "invariant violated: a device's grant leaves {free} of its {capacity} bytes");
        let supply = seismic::Supply { free_beyond_target: i64::try_from(free).expect("a store's bytes fit i64"), clean_cache: 0, cold_reclaimable: 0 };
        Self { total: capacity, supply, reclaim_band: 0, cache_reserve: 0, grant: None, paging: None, own_reclaimed: None, lag }
    }

    /// Attach an elastic-cache-only headroom reserve to a discrete device
    /// reading (never subtracted from `budget()` or resident admission).
    pub fn with_cache_reserve(mut self, reserve: u64) -> Self {
        self.cache_reserve = reserve.min(self.total);
        self
    }

    pub(crate) fn cache_reserve(&self) -> u64 {
        self.cache_reserve
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    /// The store's structural capacity: the most bytes it could grant the
    /// process beside nothing else, bounding a load's growth ceiling
    /// (`PlanningRoom::capacity`) independently of momentary free supply.
    /// On unified Metal (`HostGrant::Wired`), excludes
    /// [`Ceiling::execution_reserve`] so structural capacity agrees with live
    /// `Read::grant` and static assessment (`MemoryBudget::stable`).
    pub fn capacity(&self) -> u64 {
        match self.grant {
            Some(HostGrant::Wired { ceiling, .. }) => {
                self.total.min(ceiling.limit).saturating_sub(ceiling.execution_reserve())
            }
            Some(HostGrant::Limit(limit)) => self.total.min(limit),
            Some(HostGrant::Budget { capacity, .. }) => self.total.min(capacity),
            None => self.total,
        }
    }

    pub(crate) fn supply(&self) -> seismic::Supply {
        self.supply
    }

    pub(crate) fn reclaim_band(&self) -> u64 {
        self.reclaim_band
    }

    pub(crate) fn grant(&self) -> Option<HostGrant> {
        self.grant
    }

    pub(crate) fn ceiling(&self) -> Option<Ceiling> {
        match self.grant {
            Some(HostGrant::Wired { ceiling, .. }) => Some(ceiling),
            _ => None,
        }
    }

    pub(crate) fn paging(&self) -> Option<Paging> {
        self.paging
    }

    /// The process's anonymous bytes the OS holds outside RAM, compressed
    /// or swapped, where it reports them: none for a device store.
    pub(crate) fn own_reclaimed(&self) -> Option<seismic::OwnReclaimed> {
        self.own_reclaimed
    }

    /// The clean cache that is supply: all of it, unless the host gives no
    /// feedback of any kind (no harm and no pressure level: Linux without
    /// PSI, in a cgroup shared with other processes), where growth into
    /// cache others use would go unobserved, and the free pages above the
    /// target are the supply alone. Clean cache costs nothing to drop; what
    /// it risks is others' refaults, which harm or pressure would show.
    pub(crate) fn clean_cache_supply(&self) -> u64 {
        match self.paging {
            Some(Paging { harm: None, pressure: None, .. }) => 0,
            _ => self.supply.clean_cache,
        }
    }

    /// Immediate non-swapping supply: free pages beyond the OS target plus
    /// monitored clean file cache, never counting swap/compressor-backed
    /// `cold_reclaimable` pages. Signed because `free_beyond_target` may be
    /// negative when the OS has dipped below its target.
    pub(crate) fn immediate_supply(&self) -> i64 {
        let clean = i64::try_from(self.clean_cache_supply()).expect("a store's bytes fit i64");
        self.supply.free_beyond_target.saturating_add(clean)
    }

    /// OS/container reserve that unreclaimable pinned host allocations must
    /// leave untouched in immediate supply (`free_beyond_target + clean_cache`)
    /// beside `guarantee`: [`scarcity_threshold`] with `others_step = 0`
    /// (`reclaim_band + unit`) on host stores, and `0` on non-host device
    /// stores. [`Self::pinned_headroom`] and `Estimate::pinned_room` subtract
    /// `guarantee` (`others_step`) separately so total preserved immediate
    /// supply equals `scarcity_threshold(reclaim_band, unit, guarantee)`
    /// without double-counting `guarantee`.
    pub(crate) fn pinned_reserve(&self, unit: u64) -> u64 {
        match self.paging {
            Some(_) => scarcity_threshold(self.reclaim_band, unit, 0),
            None => 0,
        }
    }

    /// Live pinned-host headroom in bytes beyond the OS/container reserve
    /// ([`Self::pinned_reserve`]) and `guarantee` (others' rise):
    /// `max(0, immediate_supply - pinned_reserve(unit) - guarantee)`.
    ///
    /// Single owner for sizing and gating unreclaimable pinned host
    /// allocations (CUDA `cuMemHostAlloc`, linked host expert tiers across
    /// all backends) directly from a host reading.
    pub(crate) fn pinned_headroom(&self, guarantee: u64, unit: u64) -> u64 {
        let reserve = i128::from(self.pinned_reserve(unit)).saturating_add(i128::from(guarantee));
        let room = (i128::from(self.immediate_supply()) - reserve).max(0);
        u64::try_from(room).expect("a store's bytes fit u64")
    }

    pub fn lag(&self) -> seismic::Lag {
        self.lag
    }
}

/// Others' cold anonymous bytes in `sample`: the machine's less the
/// process's own, where the OS gives that (Linux, a cgroup v2 of the
/// process alone). The two are read apart (the cgroup's `memory.stat`,
/// flushed lazily, and the machine's `/proc/meminfo`), and on a host where
/// the process holds nearly every inactive page its own may read past the
/// machine's by the skew: others' then count none, the least they can be.
/// That gives no cold supply, and as regrowth evidence (below) it can
/// mislead by one step at most, as any wrong evidence can.
///
/// Where the OS gives no own share (macOS, Windows, a shared cgroup) the
/// figure is the machine's and includes the process's own cold pages: any
/// of its pageable bytes (CPU buffers, host blocks, a discrete device's
/// host tier, on unified Metal what lies outside the residency set). Such
/// cold bytes are never supply (`memory::ledger`'s `cold_supply`, which
/// counts none while the process holds pageable memory there), and serve
/// only as regrowth evidence past B, which needs a rise of a whole step's
/// bytes (11-11 §4.2): a rise of the process's own aging read as others'
/// admits at most one step, whose harm sets B again, one step's faults,
/// the bound the rule already pays for wrong evidence.
fn others_cold(sample: &seismic::HostSample) -> u64 {
    match sample.cold.own {
        Some(own) => sample.cold.machine.saturating_sub(own),
        None => sample.cold.machine,
    }
}

/// A store's boundary state, fed one reading at a time ([`Self::observe`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct Boundary {
    last: Option<Point>,
    /// Whether at least one non-growing stretch has spanned the counters' lag.
    judged: bool,
    quiet_since: Option<Instant>,
    step: Option<Step>,
    /// Consecutive scarce spans above `THRASH_RATE_BPS` without exogenous relief.
    thrash: Option<ThrashStreak>,
    cap: Option<Cap>,
    /// Sustained thrash excess past threshold at the latest reading, with no
    /// growth step open.
    harmed: Option<u64>,
    /// The OS's pressure level at the latest reading, where it is set.
    level: Option<seismic::PressureLevel>,
    /// When standing pressure began (or last stepped `cap` downward).
    pressure_at: Option<Instant>,
    /// The level when a pressure event or standing dwell step triggered at this reading.
    pressed: Option<seismic::PressureLevel>,
    /// The OS counts no share of the process's in the machine's harm:
    /// supply and OS pressure bound growth there.
    unjudged: bool,
    /// What the process has committed at the latest reading: B never falls
    /// below it, so a boundary sheds cache alone and never the floor the
    /// load serves with.
    committed: u64,
}

/// What one reading gives the boundary.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Observed {
    pub(crate) paging: Paging,
    /// The OS's active reclaim band above the floor already subtracted from
    /// `free_beyond_target`.
    pub(crate) reclaim_band: u64,
    /// Competing-process allocation step measured by the ledger (`guarantee`).
    pub(crate) others_step: u64,
    /// What the process had physically allocated as the counters show it (at
    /// `paging.counted.at`): what physical growth steps are judged against.
    pub(crate) allocated: u64,
    /// What the process holds in the ledger (`committed + cache + shed`):
    /// the unit that `cap` bounds across `import_room`, admission, and shortage.
    pub(crate) held: u64,
    /// What the process held and the supply beside it: what only others'
    /// changes move.
    pub(crate) room: u64,
    /// Immediate supply beside the process (free pages beyond target and clean
    /// file cache): bytes the OS can hand out without compressing anonymous
    /// working sets.
    pub(crate) free: u64,
    /// What one pressure event lowers B by: one block of the store's pool,
    /// the unit a shrink returns to the OS, of the cache there is.
    pub(crate) unit: u64,
    /// The pressure events the OS delivered (`seismic::PressureEvents`)
    /// that no reading of the store has taken yet
    /// (`memory::ledger::Ledger::watch_pressure`).
    pub(crate) events: u64,
    /// What the process has committed: the least B may be.
    pub(crate) committed: u64,
}

#[derive(Clone, Copy, Debug)]
struct Point {
    at: Instant,
    machine: u64,
    own: u64,
    allocated: u64,
    held: u64,
    free: u64,
    room: u64,
}

#[derive(Clone, Copy, Debug)]
struct ThrashStreak {
    since: Instant,
    spans: u32,
    excess_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct Step {
    /// What the process held as the step began.
    from: u64,
    /// The reading that last saw it grow.
    grown: Instant,
    /// Whether this growth step triggered `Harm` before closing.
    harmed: bool,
}

#[derive(Clone, Copy, Debug)]
struct Cap {
    /// The most the process may hold.
    bytes: u64,
    /// The step's bytes: minimum exogenous gain before raising `bytes`.
    step: u64,
    /// Immediate free supply and `held + free` anchor captured when this cap
    /// was created (tracking the trough on others' growth, invariant under our
    /// own shrinks since `-Δheld + Δfree == 0`, and excluding `cold` so
    /// inactive page re-aging never self-lifts B).
    free: u64,
    anchor: u64,
    /// When the current uninterrupted calm stretch under this cap began.
    calm_since: Option<Instant>,
    /// Consecutive times a probe past this wall re-triggered Harm; doubles the
    /// calm dwell before the next probe (up to `1 << CALM_PROBE_MAX_SHIFT`) and
    /// resets on a calm completed growth step or exogenous free release.
    strikes: u32,
}

impl Boundary {
    /// Fold in one reading.
    pub(crate) fn observe(&mut self, now: Observed) {
        let Observed { paging, reclaim_band, others_step, allocated, held, room, free, unit, events, committed } = now;
        self.committed = committed;
        let seismic::Counted { at, lag } = paging.counted;
        self.harmed = None;
        let threshold = scarcity_threshold(reclaim_band, unit, others_step);
        // Track the trough of immediate `free` and `held + free` while `cap`
        // stands: our own shrinks preserve `held + free` (`-Δheld + Δfree == 0`)
        // and other apps' pages cycling `active <-> inactive` never enter `free`,
        // so measuring scarcity against `cap.free + (held + free - cap.anchor)`
        // keeps scarcity invariant under our own shrinks while lifting as soon
        // as genuine exogenous free supply exceeds `threshold`.
        let effective_free = match self.cap.as_mut() {
            Some(cap) => {
                cap.free = cap.free.min(free);
                cap.anchor = cap.anchor.min(held.saturating_add(free));
                cap.free.saturating_add(held.saturating_add(free).saturating_sub(cap.anchor))
            }
            None => free,
        };
        let scarce_free = effective_free <= threshold;
        let prev_level = self.level;
        self.level = paging.pressure.filter(|level| *level != seismic::PressureLevel::Normal);
        // Immediate free + clean cache above `scarcity_threshold` means the host
        // can satisfy allocations without swapping/compressing anonymous pages
        // (e.g. Linux PSI firing on clean page-cache reclaim or compaction with
        // tens of GiB of clean cache free): never fire pressure or latch `cap`
        // while immediate supply is non-scarce.
        let mut pressure_trigger = self.level.filter(|_| events > 0 && scarce_free);
        match self.level {
            None => self.pressure_at = None,
            Some(level) => {
                let standing_dwell = match level {
                    seismic::PressureLevel::Critical if scarce_free => Some(STANDING_CRITICAL_DWELL),
                    seismic::PressureLevel::Warn if scarce_free && paging.harm.is_none() => Some(STANDING_WARN_DWELL),
                    _ => None,
                };
                match standing_dwell {
                    None => self.pressure_at = None,
                    Some(dwell) => {
                        if events > 0 || prev_level.is_none() || self.pressure_at.is_none() {
                            self.pressure_at = Some(at);
                        } else if unit > 0
                            && held > self.committed
                            && self.pressure_at.is_some_and(|since| at.saturating_duration_since(since) >= dwell)
                        {
                            pressure_trigger = Some(level);
                            self.pressure_at = Some(at);
                        }
                    }
                }
            }
        }
        self.pressed = pressure_trigger;
        if self.pressed.is_some() {
            // Each pressure event or standing dwell lowers B by one block
            // (or `events` blocks) even if an earlier async shrink is still
            // releasing toward the previous cap, never below committed memory.
            let step_units = events.max(1);
            let blocks = unit.saturating_mul(step_units);
            let bytes = held.saturating_sub(blocks).max(self.committed);
            let anchor = held.saturating_add(free);
            let cap = match self.cap {
                Some(cap) => {
                    let next = cap.bytes.saturating_sub(blocks).min(bytes).max(self.committed);
                    let lowered = cap.bytes.saturating_sub(next);
                    Cap {
                        bytes: next,
                        step: cap.step.saturating_add(lowered),
                        calm_since: None,
                        ..cap
                    }
                }
                None => Cap { bytes, step: blocks, free, anchor, calm_since: None, strikes: 0 },
            };
            self.cap = Some(cap);
        }
        let pressure_calm = self.level.is_none() || (self.pressed.is_none() && !scarce_free);
        let Some(harm) = paging.harm else {
            // No share of the process's is counted: nothing judges harm,
            // growth or a step; B, where pressure set it, lifts on others' release.
            self.unjudged = true;
            if pressure_calm {
                self.raise_on_exogenous_release(held, free, room, unit, threshold, at);
            } else if let Some(cap) = self.cap.as_mut() {
                cap.calm_since = None;
            }
            return;
        };
        self.unjudged = false;
        let point = Point { at, machine: harm.machine, own: harm.own, allocated, held, free, room };
        // Harm is judged only over spans of at least the counters' lag: a
        // share of the process's own a fold shows late is then shown.
        let Some(last) = self.last.filter(|last| last.at < at && at.duration_since(last.at) >= lag) else {
            // The growth since the judged reading, if any, opens its step at
            // the next one judged, from what was held there.
            self.last.get_or_insert(point);
            return;
        };
        self.last = Some(point);
        // λ_h: the counters' lag, over which a step's harm shows.
        let window = lag;
        let span = at.duration_since(last.at).as_secs_f64();
        let harmed = harm.machine.saturating_sub(last.machine).saturating_sub(harm.own.saturating_sub(last.own));
        let harmed_bytes = harmed.saturating_mul(paging.page_bytes);
        let rate_bps = harmed_bytes as f64 / span;
        let scarce = scarce_free;
        let no_exogenous_relief = free <= last.free.saturating_add(EXOGENOUS_RELIEF_TOLERANCE)
            && room <= last.room.saturating_add(EXOGENOUS_RELIEF_TOLERANCE);
        let excess_span = ((rate_bps - THRASH_RATE_BPS).max(0.0) * span).round() as u64;
        let confirmed_excess = if scarce && no_exogenous_relief && excess_span > 0 {
            let streak = self.thrash.get_or_insert(ThrashStreak {
                since: last.at,
                spans: 0,
                excess_bytes: 0,
            });
            streak.spans = streak.spans.saturating_add(1);
            streak.excess_bytes = streak.excess_bytes.saturating_add(excess_span);
            if streak.spans >= 2 && at.duration_since(streak.since) >= THRASH_CONFIRM_WINDOW {
                let confirmed = streak.excess_bytes;
                *streak = ThrashStreak { since: at, spans: 0, excess_bytes: 0 };
                Some(confirmed)
            } else {
                None
            }
        } else {
            self.thrash = None;
            None
        };
        if allocated > last.allocated {
            let (from, harmed) = self.step.map_or((last.held, false), |step| (step.from, step.harmed));
            self.step = Some(Step { from, grown: at, harmed });
            self.quiet_since = None;
        }
        let thrashing = confirmed_excess.is_some();
        match self.step {
            Some(step) if thrashing => {
                // Sustained thrashing within our growth step's window: return
                // B to the holding before the step began, never below committed.
                self.step = Some(Step { harmed: true, ..step });
                let floor_from = step.from.max(self.committed);
                let delta = held.saturating_sub(floor_from).max(unit).max(paging.page_bytes);
                self.settle(Cap {
                    bytes: floor_from,
                    step: delta,
                    free,
                    anchor: held.saturating_add(free),
                    calm_since: None,
                    strikes: 0,
                });
            }
            Some(step) if allocated <= last.allocated && last.at >= step.grown + window => {
                self.step = None;
                self.judged = true;
                if !step.harmed {
                    if let Some(cap) = self.cap.as_mut() {
                        cap.strikes = 0;
                    }
                }
            }
            Some(_) => {}
            None => {
                if let Some(excess) = confirmed_excess {
                    // Others' active working set is thrashing under scarcity:
                    // tighten B in place by the sustained excess without
                    // snapping down to lower `held`, falling below committed,
                    // or overwriting `anchor`.
                    self.harmed = Some(excess);
                    let anchor = held.saturating_add(free);
                    self.cap = Some(match self.cap {
                        Some(cap) => {
                            let next = cap.bytes.saturating_sub(excess).max(self.committed);
                            let lowered = cap.bytes.saturating_sub(next);
                            Cap {
                                bytes: next,
                                step: cap.step.saturating_add(lowered),
                                calm_since: None,
                                ..cap
                            }
                        }
                        None => Cap {
                            bytes: held.saturating_sub(excess).max(self.committed),
                            step: excess.max(unit).max(paging.page_bytes),
                            free,
                            anchor,
                            calm_since: None,
                            strikes: 0,
                        },
                    });
                }
                let since = *self.quiet_since.get_or_insert(last.at);
                if at.duration_since(since) >= window {
                    self.judged = true;
                }
            }
        }
        let calm = !thrashing && self.thrash.is_none() && self.step.is_none() && pressure_calm;
        if calm {
            if !self.raise_on_exogenous_release(held, free, room, unit, threshold, at) {
                self.probe_after_calm(held, room, unit, threshold, at);
            }
        } else if let Some(cap) = self.cap.as_mut() {
            cap.calm_since = None;
        }
    }

    /// Raise (or clear) a standing cap when genuine exogenous free supply
    /// (`held + free`, excluding `cold_reclaimable`) rises above `cap.anchor`
    /// by at least `min_gain`. Clears `cap` to `None` only when no unverified
    /// `cold` overhang remains above the raised ceiling (`raised >= room` or
    /// `room <= held + free + unit`), so an exogenous release never triggers an
    /// unbounded leap into a known `Harm` wall. Returns whether `cap` moved.
    fn raise_on_exogenous_release(&mut self, held: u64, free: u64, room: u64, unit: u64, threshold: u64, at: Instant) -> bool {
        let Some(cap) = self.cap else { return false };
        let max_lift_gain = threshold.max(unit);
        let min_gain = cap.step.min(max_lift_gain).max(unit);
        let held_plus_free = held.saturating_add(free);
        let gain = held_plus_free.saturating_sub(cap.anchor);
        let free_gain = free.saturating_sub(cap.free);
        if gain < min_gain || free_gain < min_gain {
            return false;
        }
        let raised = cap.bytes.saturating_add(gain);
        let no_cold_overhang = room <= held_plus_free.saturating_add(unit);
        if no_cold_overhang || raised >= room {
            self.cap = None;
        } else {
            self.cap = Some(Cap {
                bytes: raised,
                free,
                anchor: held_plus_free,
                calm_since: Some(at),
                strikes: 0,
                ..cap
            });
        }
        true
    }

    /// After sustained uninterrupted calm under a standing cap on a judged OS,
    /// raise `cap.bytes` by one bounded probe increment (never clearing to
    /// `None`) with exponential backoff across consecutive wall-hit strikes.
    fn probe_after_calm(&mut self, held: u64, room: u64, unit: u64, threshold: u64, at: Instant) {
        let Some(cap) = self.cap.as_mut() else { return };
        if unit == 0 {
            return;
        }
        let since = *cap.calm_since.get_or_insert(at);
        let dwell = CALM_PROBE_BASE.saturating_mul(1u32 << cap.strikes.min(CALM_PROBE_MAX_SHIFT));
        if at.saturating_duration_since(since) < dwell {
            return;
        }
        let probe = unit.max(CALM_PROBE_MIN_BYTES).min(threshold.max(unit));
        let next = cap.bytes.saturating_add(probe);
        if held.saturating_add(probe) > cap.bytes && room >= next {
            cap.bytes = next;
            cap.calm_since = Some(at);
        }
    }

    /// Hold no more than `cap` allows from now, a lower boundary standing.
    fn settle(&mut self, mut cap: Cap) {
        self.cap = Some(match self.cap {
            Some(standing) if standing.bytes <= cap.bytes => Cap {
                calm_since: None,
                strikes: standing.strikes.saturating_add(1),
                ..standing
            },
            Some(standing) => {
                cap.strikes = standing.strikes.saturating_add(1);
                cap
            }
            None => cap,
        });
    }

    /// Whether at least one quiet lag window has been observed.
    #[cfg(test)]
    pub(crate) fn judges(&self) -> bool {
        self.judged
    }

    /// The most the process may hold in the store, where B stands: never
    /// below what is committed at the latest reading, so B bounds the
    /// cache and the floor the load serves with is never shed to it.
    pub(crate) fn cap(&self) -> Option<u64> {
        self.cap.map(|cap| cap.bytes.max(self.committed))
    }

    /// Whether growth is admissible: at least one quiet lag window has settled
    /// and no growth step's window is open, so a step's effect is told from
    /// the next's; or the OS counts no share of the process's, where supply
    /// and OS pressure bound growth.
    pub(crate) fn admits_growth(&self) -> bool {
        self.unjudged || (self.judged && self.step.is_none())
    }

    /// End any active counter span before an unmonitored idle sleep, keeping
    /// the judged state and any standing boundary B: the next reading starts a
    /// fresh span at its wake instant instead of attributing paging while asleep
    /// to one active interval.
    pub(crate) fn settle_idle(&mut self) {
        self.last = None;
        self.quiet_since = None;
        self.step = None;
        self.thrash = None;
        self.harmed = None;
        self.pressed = None;
        if let Some(cap) = self.cap.as_mut() {
            cap.calm_since = None;
        }
    }

    /// Whether a growth step's window λ_h is open, so only a later reading
    /// shows whether the growth harmed others or closes.
    pub(crate) fn step_open(&self) -> bool {
        self.step.is_some()
    }

    /// Sustained thrashing excess at the latest reading with no step of the
    /// process's open, in bytes.
    pub(crate) fn harmed(&self) -> Option<u64> {
        self.harmed
    }

    /// The pressure level when an edge event or standing-pressure dwell step
    /// triggered at the latest reading.
    pub(crate) fn pressed(&self) -> Option<seismic::PressureLevel> {
        self.pressed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Unified Metal's wired ceiling: the OS's default is the device's
    /// working set, a limit set is bounded by it, and a host whose GPU
    /// wires nothing (an Intel Mac's integrated GPU) is bounded by the
    /// working set alone, never a panic.
    #[test]
    fn unified_metals_grant_is_its_working_set_under_any_wired_limit() {
        let working_set = 48 * GB;
        let wired = |limit| {
            let base = seismic::host_samples::m4_pro();
            seismic::HostSample {
                wired: Some(seismic::WiredFigures { limit, wire_count: 4 * GB, at: base.at }),
                ..base
            }
        };
        let limit = |sample: seismic::HostSample| Reading::unified(&sample, working_set).ceiling().map(|ceiling| ceiling.limit);
        assert_eq!(limit(wired(seismic::WiredLimit::OsDefault)), Some(working_set));
        assert_eq!(limit(wired(seismic::WiredLimit::Set(36 * GB))), Some(36 * GB));
        assert_eq!(limit(wired(seismic::WiredLimit::Set(60 * GB))), Some(working_set));
        let unwired = Reading::unified(&seismic::HostSample { wired: None, ..seismic::host_samples::m4_pro() }, working_set);
        assert!(matches!(unwired.grant(), Some(HostGrant::Limit(bytes)) if bytes == working_set));
    }

    const GB: u64 = 1 << 30;
    const MB: u64 = 1 << 20;

    /// The counters' lag, λ_h.
    const LAG: Duration = Duration::from_millis(100);

    /// Typical macOS 64 GiB XNU background reclaim band above `free_target`
    /// (`50_000 * 16 KiB - 256 MiB = 525.25 MiB`).
    const MACOS_64G_BAND: u64 = 50_000 * 16_384 - 256 * MB;

    fn obs(at: Instant, reclaim_band: u64, machine: u64, held: u64, free: u64, unit: u64, events: u64, pressure: seismic::PressureLevel) -> Observed {
        let paging = Paging {
            harm: Some(seismic::Harm { machine, own: 0 }),
            page_bytes: 16 * 1024,
            cold: 0,
            own_cold: None,
            pressure: Some(pressure),
            counted: seismic::Counted { at, lag: LAG },
        };
        Observed { paging, reclaim_band, others_step: 0, allocated: held, held, room: held + free, free, unit, events, committed: 20 * GB }
    }

    /// Normal OS compression/wakeup activity (e.g. a few MB/s of lazy
    /// decompressions, any burst when free+clean headroom is abundant, or an
    /// exogenous memory release where free rises) never locks `cap` or shrinks
    /// the model under `Normal` pressure.
    #[test]
    fn normal_background_decompressions_and_abundant_bursts_never_harm() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let band = MACOS_64G_BAND;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert!(boundary.judges());

        // Continuous 20 MB/s background decompressions even with free = 0 remain well below THRASH_RATE_BPS (64 MiB/s).
        let mut pages = 0u64;
        for tick in 2..=6 {
            pages += (20 * MB) / page;
            boundary.observe(obs(start + Duration::from_secs(tick), band, pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
            assert_eq!(boundary.cap(), None, "normal compression touch tick {tick} must not set B");
            assert_eq!(boundary.harmed(), None);
        }

        // Single huge tab-switch / app wakeup while free headroom is abundant (4 GB > threshold).
        pages += (512 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(7), band, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        pages += (512 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(8), band, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "paging under abundant free supply never sets B");

        // Multi-GB decompression burst when an app frees memory (free rises +4 GB).
        pages += (4 * GB) / page;
        boundary.observe(obs(start + Duration::from_secs(9), band, pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        pages += (4 * GB) / page;
        boundary.observe(obs(start + Duration::from_secs(10), band, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "exogenous memory release must not be read as contention");
    }

    /// A single short spike above THRASH_RATE_BPS clears cleanly when the next
    /// window calms down, without setting `cap`.
    #[test]
    fn transient_one_shot_spike_does_not_fire_thrashing() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let band = MACOS_64G_BAND;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));

        // Single 150 MB/s spike for 1s, then back to calm 4 MB/s.
        let spike = (150 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), band, spike, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None);
        boundary.observe(obs(start + Duration::from_secs(3), band, spike + (4 * MB) / page, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "one-shot wakeup never sets B");
    }

    /// Sustained active thrashing under scarcity (>= 2 spans over >= 1 s above
    /// 64 MiB/s) sets and tightens `cap` by sustained excess without snapping
    /// down to lower `held`, and our own shrink never self-lifts `cap` via
    /// freed bytes (`-Δheld + Δfree == 0`).
    #[test]
    fn sustained_thrashing_under_scarcity_caps_without_snapping_or_self_lifting() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let band = MACOS_64G_BAND;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));

        // Two consecutive 1s spans at 128 MB/s under scarcity (excess = 2 * 64 MB = 128 MB).
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), band, rate_pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "first span arms without firing yet");
        boundary.observe(obs(start + Duration::from_secs(3), band, 2 * rate_pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        let first_cap = boundary.cap().expect("second sustained thrash span sets B");
        assert_eq!(first_cap, 30 * GB - 128 * MB);

        // Model shrinks to `held = first_cap - 600 MB`, returning 600 MB + 128 MB to free.
        // Because `held + free` equals the captured anchor (`30 GB`), calm readings
        // must NOT self-lift B.
        let held_after_shrink = first_cap - 600 * MB;
        let free_from_shrink = 728 * MB;
        boundary.observe(obs(start + Duration::from_secs(4), band, 2 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), Some(first_cap), "own shrink must never self-lift B");

        // Renewed sustained thrashing at 128 MB/s tightens cap in place by excess alone, never snapping to `held`.
        boundary.observe(obs(start + Duration::from_secs(5), band, 3 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(6), band, 4 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        let second_cap = boundary.cap().expect("standing B tightens cleanly");
        assert!(second_cap > held_after_shrink + 400 * MB);
        assert_eq!(second_cap, first_cap - 128 * MB);

        // Calm exogenous app release lifts B once `held + free >= anchor + step` and `free >= cap.free + step`.
        boundary.observe(obs(start + Duration::from_secs(7), band, 4 * rate_pages, held_after_shrink, free_from_shrink + GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "exogenous free lifts B cleanly");
    }

    /// Standing OS pressure under scarcity yields one block per dwell interval
    /// without edge events:
    /// - `Warn` past 1 s only on level-triggered / unjudged OSes (`harm: None`,
    ///   e.g. Windows `LowMemoryResourceNotification`), never on judged OSes
    ///   (macOS/Linux) where standing `Warn` is normal desktop compressor state;
    /// - `Critical` past 250 ms on every OS while `free <= scarcity_threshold`;
    /// - neither fires when immediate `free` exceeds `scarcity_threshold`.
    #[test]
    fn standing_pressure_yields_past_dwell_even_without_edge_events() {
        let start = Instant::now();
        let band = MACOS_64G_BAND;
        let unit = 256 * MB;
        let unjudged = |at, held, free, pressure| {
            let mut o = obs(at, 0, 0, held, free, unit, 0, pressure);
            o.paging.harm = None;
            o
        };

        // Level-triggered / unjudged OS (Windows, reclaim_band = 0): Warn past 1s under scarcity yields.
        let mut boundary = Boundary::default();
        boundary.observe(unjudged(start, 30 * GB, 0, seismic::PressureLevel::Normal));
        boundary.observe(unjudged(start + Duration::from_millis(100), 30 * GB, 0, seismic::PressureLevel::Warn));
        boundary.observe(unjudged(start + Duration::from_millis(400), 30 * GB, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.pressed(), None);
        assert_eq!(boundary.cap(), None);

        boundary.observe(unjudged(start + Duration::from_millis(1100), 30 * GB, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.pressed(), Some(seismic::PressureLevel::Warn));
        assert_eq!(boundary.cap(), Some(30 * GB - unit));

        // Standing Critical steps faster (250ms dwell) under scarcity.
        boundary.observe(unjudged(start + Duration::from_millis(1350), 30 * GB, 0, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), Some(seismic::PressureLevel::Critical));
        assert_eq!(boundary.cap(), Some(30 * GB - 2 * unit));

        // Abundant free supply (+4 GB exogenous release above threshold)
        // suppresses standing dwell and lifts the standing cap even while Critical
        // has not yet cleared.
        boundary.observe(unjudged(start + Duration::from_millis(2500), 30 * GB, 4 * GB, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None);
        assert_eq!(boundary.cap(), None);

        // Judged OS (macOS / Linux): standing Warn with events == 0 NEVER drains,
        // even for 60s, while Critical under scarcity still yields at 250ms.
        let mut macos = Boundary::default();
        macos.observe(obs(start, band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        for sec in 1..=60 {
            macos.observe(obs(start + Duration::from_secs(sec), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Warn));
            assert_eq!(macos.pressed(), None, "standing Warn on macOS/Linux must never drain (sec {sec})");
            assert_eq!(macos.cap(), None);
        }
        // Escalation to Critical under scarcity starts dwell cleanly and yields after 250ms.
        macos.observe(obs(start + Duration::from_secs(61), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Critical));
        assert_eq!(macos.pressed(), None);
        macos.observe(obs(start + Duration::from_millis(61_250), band, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Critical));
        assert_eq!(macos.pressed(), Some(seismic::PressureLevel::Critical));
        assert_eq!(macos.cap(), Some(30 * GB - unit));
    }

    /// Standing macOS `Warn` with abundant free supply never triggers false
    /// `Harm` on lazy decompressions, and after a competing app grows (pushing
    /// `room` down and setting `cap`) then exits, `cap` lifts cleanly from the
    /// trough even if `Warn` remains standing.
    #[test]
    fn standing_warn_never_false_harms_and_lifts_cap_from_trough_on_recovery() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let band = MACOS_64G_BAND;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, band, 0, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert!(boundary.judges());

        // High-rate lazy decompressions (128 MiB/s) under standing Warn with 4 GB free (> threshold):
        // must NOT fire Harm because immediate free supply is abundant.
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), band, rate_pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        boundary.observe(obs(start + Duration::from_secs(3), band, 2 * rate_pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.harmed(), None, "standing Warn with abundant free must not fire Harm");
        assert_eq!(boundary.cap(), None);

        // Competing app grows +10 GB: triggers pressure events, pushes room down from 30 GB to 20 GB.
        boundary.observe(obs(start + Duration::from_secs(4), band, 2 * rate_pages, 30 * GB, 0, unit, 1, seismic::PressureLevel::Warn));
        assert_eq!(boundary.cap(), Some(30 * GB - unit));
        for tick in 5..=14 {
            let held = 30 * GB - (tick - 4) * GB;
            boundary.observe(obs(start + Duration::from_secs(tick), band, 2 * rate_pages, held, 0, unit, 1, seismic::PressureLevel::Warn));
        }
        assert!(boundary.cap().is_some());

        // Competing app exits, freeing 10 GB: room recovers from trough (20 GB) to 30 GB,
        // free jumps to 10 GB (> threshold), even while macOS stays at standing Warn.
        boundary.observe(obs(start + Duration::from_secs(15), band, 2 * rate_pages, 20 * GB, 10 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.cap(), None, "exogenous recovery from trough must lift cap even under standing Warn");

        // Abundant free supply suppresses even edge pressure events
        // (e.g. Linux PSI compaction stalls when tens of GiB are free).
        boundary.observe(obs(start + Duration::from_secs(16), band, 2 * rate_pages, 20 * GB, 36 * GB, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "pressure event with abundant free RAM must not fire");
        assert_eq!(boundary.cap(), None);
    }

    /// Linux GGUF page-cache scenario (`total = 64 GiB`, `free_beyond_target = 16 GiB`,
    /// `clean_cache = 24 GiB`, immediate reclaimable supply = `40 GiB`):
    /// `Critical` PSI blips and file-cache refaults must NOT fire `Pressure(Critical)`
    /// or `Harm` or ratchet down `Cap`. Also verifies `Cap::bytes` never ratchets
    /// below `committed` on repeated pressure events at floor.
    #[test]
    fn linux_gguf_clean_page_cache_suppresses_critical_psi_and_cap_stays_above_committed() {
        let start = Instant::now();
        let total = 64 * GB;
        let band = 64 * MB; // Linux 64 GiB zone (high - low)
        let unit = 256 * MB;
        let page = 16 * 1024;
        let sample = seismic::HostSample {
            total,
            supply: seismic::Supply {
                free_beyond_target: (16 * GB) as i64,
                clean_cache: 24 * GB,
                cold_reclaimable: 0,
            },
            reclaim_band: band,
            wired: None,
            pressure: Some(seismic::PressureLevel::Critical),
            ..seismic::host_samples::m4_pro()
        };
        let reading = Reading::host(&sample);
        assert_eq!(reading.clean_cache_supply(), 24 * GB);
        let immediate_supply = u64::try_from(reading.supply().free_beyond_target.max(0))
            .unwrap_or(0)
            .saturating_add(reading.clean_cache_supply());
        assert_eq!(immediate_supply, 40 * GB);

        let mut boundary = Boundary::default();
        boundary.observe(obs(start, band, 0, 22 * GB, immediate_supply, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, 22 * GB, immediate_supply, unit, 0, seismic::PressureLevel::Normal));
        assert!(boundary.judges());

        // Critical PSI edge event + high-rate refaults (128 MiB/s over 2s) while
        // free + clean_cache = 40 GiB: neither Pressure nor Harm fires.
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), band, rate_pages, 22 * GB, immediate_supply, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "Critical PSI with 40/64 GiB free+clean must not fire");
        assert_eq!(boundary.harmed(), None);
        assert_eq!(boundary.cap(), None);

        boundary.observe(obs(start + Duration::from_secs(3), band, 2 * rate_pages, 22 * GB, immediate_supply, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "repeated Critical PSI with 40/64 GiB free+clean must not fire");
        assert_eq!(boundary.harmed(), None, "Critical PSI with abundant free+clean must not mark host scarce for Harm");
        assert_eq!(boundary.cap(), None);

        // Under genuine scarcity at floor (held == committed == 20 GiB), repeated
        // pressure events never ratchet `cap.bytes` below committed.
        let mut floored = Boundary::default();
        for tick in 0..10 {
            floored.observe(obs(start + Duration::from_secs(tick), band, 0, 20 * GB, 0, unit, 1, seismic::PressureLevel::Critical));
            assert_eq!(floored.cap(), Some(20 * GB));
        }
        let mut lower_committed = obs(start + Duration::from_secs(10), band, 0, 20 * GB, 0, unit, 0, seismic::PressureLevel::Normal);
        lower_committed.committed = 16 * GB;
        floored.observe(lower_committed);
        assert_eq!(floored.cap(), Some(20 * GB), "cap.bytes must never have ratcheted below the committed floor");
    }

    /// Reproduces the 64 GiB M4 Max sawtooth oscillation (`PID 91361`):
    /// growing into `cold_reclaimable` triggers `Harm` at the working-set wall
    /// while other apps' pages wake up (`cold` and `room` dip); shrinking back
    /// returns our own bytes to `free`, and 2 s later those apps' pages age
    /// back to `cold` (`cold` and `room` jump +3.5 GiB). `Cap` must NOT clear
    /// to `None` on own shrink or `cold`/`room` re-aging; calm probing must
    /// step up in bounded increments with exponential backoff on repeated wall
    /// hits; and a genuine exogenous free release must raise `Cap` smoothly by
    /// the freed bytes without leaping into an unverified `cold` overhang.
    #[test]
    fn harm_wall_never_sawtooths_on_cold_reaging_or_own_shrink_and_backs_off_probes() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let unit = 68 * MB;
        let committed = 7 * GB;
        let page = 16 * 1024;
        let obs_with_cold = |at: Instant, machine: u64, held: u64, free: u64, cold: u64| Observed {
            paging: Paging {
                harm: Some(seismic::Harm { machine, own: 0 }),
                page_bytes: page,
                cold,
                own_cold: None,
                pressure: Some(seismic::PressureLevel::Normal),
                counted: seismic::Counted { at, lag: LAG },
            },
            reclaim_band: MACOS_64G_BAND,
            others_step: 0,
            allocated: held,
            held,
            room: held + free + cold,
            free,
            unit,
            events: 0,
            committed,
        };

        // Calm baseline at 35.7 GiB held with 3.5 GiB of inactive desktop pages (`cold`).
        let wall = 35 * GB + 700 * MB;
        boundary.observe(obs_with_cold(start, 0, wall, 2 * GB, 3500 * MB));
        boundary.observe(obs_with_cold(start + Duration::from_secs(1), 0, wall, 2 * GB, 3500 * MB));
        assert!(boundary.judges());

        // Bank grows +3.3 GiB into `cold_reclaimable`; apps wake their pages (`cold` dips to 500 MiB)
        // and thrash at 256 MiB/s across 2 spans -> Harm sets `cap = Some(wall)`.
        let grown = wall + 3300 * MB;
        let rate_pages = (256 * MB) / page;
        boundary.observe(obs_with_cold(start + Duration::from_secs(2), rate_pages, grown, 0, 500 * MB));
        boundary.observe(obs_with_cold(start + Duration::from_secs(3), 2 * rate_pages, grown, 0, 500 * MB));
        assert_eq!(boundary.cap(), Some(wall), "growth-step Harm must cap at pre-step holding");

        // Bank shrinks -3.3 GiB back to `wall`, returning 3.3 GiB to `free` (`held + free == anchor`).
        // 2 s later, other apps' pages age back to inactive (`cold` jumps +3.5 GiB, `room` jumps +3.5 GiB).
        for sec in 4..=25 {
            let cold = if sec <= 5 { 500 * MB } else { 4000 * MB };
            boundary.observe(obs_with_cold(start + Duration::from_secs(sec), 2 * rate_pages, wall, 3300 * MB, cold));
            assert_eq!(
                boundary.cap(),
                Some(wall),
                "own shrink and cold/room re-aging at sec {sec} must never clear Cap to None"
            );
        }

        // After CALM_PROBE_BASE (30 s) of uninterrupted calm (starting from step close at sec 4),
        // `cap` probes upward by one bounded increment (256 MiB), never clearing to `None`.
        boundary.observe(obs_with_cold(start + Duration::from_secs(35), 2 * rate_pages, wall, 3300 * MB, 4000 * MB));
        let probed = wall + CALM_PROBE_MIN_BYTES;
        assert_eq!(boundary.cap(), Some(probed), "sustained calm probes by one bounded increment");

        // Probe grows +256 MiB and hits the exact same Harm wall -> snaps back to `wall` with backoff strike 1.
        let mut pages = 2 * rate_pages;
        pages += rate_pages;
        boundary.observe(obs_with_cold(start + Duration::from_secs(36), pages, probed, 0, 500 * MB));
        pages += rate_pages;
        boundary.observe(obs_with_cold(start + Duration::from_secs(37), pages, probed, 0, 500 * MB));
        assert_eq!(boundary.cap(), Some(wall), "failed probe snaps back to wall");

        // Shrink back -256 MiB (returning 256 MiB to `free`, preserving `held + free == anchor`)
        // and wait 40 s (> 30 s base, < 60 s backed-off dwell): must NOT probe yet.
        let free_after_probe_shrink = CALM_PROBE_MIN_BYTES;
        boundary.observe(obs_with_cold(start + Duration::from_secs(38), pages, wall, free_after_probe_shrink, 4000 * MB));
        boundary.observe(obs_with_cold(start + Duration::from_secs(78), pages, wall, free_after_probe_shrink, 4000 * MB));
        assert_eq!(boundary.cap(), Some(wall), "backed-off dwell (60 s) holds steady at +40 s");

        // At +60 s of calm (sec 99), second bounded probe fires.
        boundary.observe(obs_with_cold(start + Duration::from_secs(99), pages, wall, free_after_probe_shrink, 4000 * MB));
        assert_eq!(boundary.cap(), Some(probed), "second probe fires once 60 s backoff dwell elapses");

        // Genuine exogenous app exit frees +2 GiB of real RAM while 4 GiB of `cold` still overhangs:
        // `cap` rises smoothly by +2 GiB (NOT `None`), preventing a +4 GiB leap into the cold wall.
        boundary.observe(obs_with_cold(start + Duration::from_secs(100), pages, wall, free_after_probe_shrink + 2 * GB, 4000 * MB));
        assert_eq!(
            boundary.cap(),
            Some(probed + 2 * GB),
            "exogenous free release under cold overhang raises cap by the exact freed bytes"
        );

        // Another +6 GiB exogenous exit while 4 GiB of `cold` still overhangs raises cap by +6 GiB
        // without leaping into the unverified cold overhang; once `cold` clears, an exogenous release
        // clears `cap` cleanly to `None`.
        boundary.observe(obs_with_cold(start + Duration::from_secs(101), pages, wall, free_after_probe_shrink + 8 * GB, 4000 * MB));
        assert_eq!(
            boundary.cap(),
            Some(probed + 8 * GB),
            "exogenous release under remaining cold overhang still bounds cap by exact freed bytes"
        );
        boundary.observe(obs_with_cold(start + Duration::from_secs(102), pages, wall, free_after_probe_shrink + 10 * GB, 0));
        assert_eq!(boundary.cap(), None, "exogenous release with no cold overhang clears cap to None");
    }

    /// Reproduces the `sparky` 122 GiB unified CUDA Linux PSI regression:
    /// with a 46 GiB model resident and 39 GiB of clean page cache (`< total/2`
    /// = 61 GiB, yet far above `scarcity_threshold` = `reclaim_band + unit`), a
    /// Linux PSI `Critical` edge event from clean-cache reclaim/compaction must
    /// NOT fire `Pressure(Critical)` or latch `cap`; and if a `cap` was latched
    /// under genuine scarcity, exogenous free recovery above
    /// `scarcity_threshold` must lift `cap` immediately even while PSI's 10 s
    /// `avg10` still reads `Critical`.
    #[test]
    fn non_scarce_clean_cache_below_half_total_suppresses_critical_psi_and_lifts_cap_on_recovery() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let total = 122 * GB;
        let band = 125 * MB; // Linux 122 GiB zone (high - low)
        let held = 46 * GB;
        let free = 39 * GB;
        let unit = 65 * MB;
        let page = 4096;
        assert!(free < total / 2, "below half of total RAM");
        assert!(free > scarcity_threshold(band, unit, 0), "far above derived scarcity threshold");

        boundary.observe(obs(start, band, 0, held, free, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), band, 0, held, free, unit, 0, seismic::PressureLevel::Normal));
        assert!(boundary.judges());

        // Transient Critical PSI edge event + high-rate file refaults with 39 GiB clean cache:
        // must neither fire Pressure(Critical) nor Harm nor latch cap.
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), band, rate_pages, held, free, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "Critical PSI with 39 GiB free+clean must not fire");
        assert_eq!(boundary.harmed(), None);
        assert_eq!(boundary.cap(), None, "cap must not latch when immediate supply is non-scarce");

        boundary.observe(obs(start + Duration::from_secs(3), band, 2 * rate_pages, held, free, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None);
        assert_eq!(boundary.harmed(), None);
        assert_eq!(boundary.cap(), None);

        // Under genuine scarcity (`free = 0 <= threshold`), Critical PSI fires and latches cap.
        boundary.observe(obs(start + Duration::from_secs(4), band, 2 * rate_pages, held, 0, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), Some(seismic::PressureLevel::Critical));
        assert_eq!(boundary.cap(), Some(held - unit));

        // Competing app exits, freeing 10 GiB (`free = 10 GiB > threshold`) while Linux PSI's
        // 10 s avg10 still reports standing Critical with 0 new events: cap lifts cleanly.
        boundary.observe(obs(start + Duration::from_secs(5), band, 2 * rate_pages, held - unit, 10 * GB, unit, 0, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None);
        assert_eq!(boundary.cap(), None, "exogenous recovery above scarcity threshold lifts cap even while PSI avg10 decays from Critical");
    }

    /// Derived scarcity threshold equals `reclaim_band + unit + others_step`:
    /// competing-process allocation bursts (`others_step`) expand the scarcity
    /// horizon without magic percentages, and `pinned_headroom` preserves the
    /// exact same total without double-counting `guarantee`.
    #[test]
    fn derived_scarcity_threshold_covers_reclaim_band_unit_and_others_step() {
        let band = 125 * MB;
        let unit = 384 * MB;
        let others_step = 512 * MB;
        assert_eq!(scarcity_threshold(band, unit, others_step), band + unit + others_step);

        // With free = 800 MiB: above `band + unit` (509 MiB) when idle, so a
        // Critical edge event does not fire; once competing processes rise by
        // 512 MiB (`others_step`), threshold = 1021 MiB >= 800 MiB and the
        // exact same free level is scarce.
        let start = Instant::now();
        let mut boundary = Boundary::default();
        boundary.observe(obs(start, band, 0, 30 * GB, 800 * MB, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "idle free above reclaim_band + unit is not scarce");

        let mut competing = obs(start + Duration::from_secs(1), band, 0, 30 * GB, 800 * MB, unit, 1, seismic::PressureLevel::Critical);
        competing.others_step = others_step;
        boundary.observe(competing);
        assert_eq!(boundary.pressed(), Some(seismic::PressureLevel::Critical), "competing allocation step makes free <= threshold scarce");

        // `pinned_reserve(unit) + guarantee == scarcity_threshold(band, unit, guarantee)`.
        let sample = seismic::HostSample {
            supply: seismic::Supply { free_beyond_target: (2 * GB) as i64, clean_cache: 0, cold_reclaimable: 0 },
            reclaim_band: band,
            ..seismic::host_samples::m4_pro()
        };
        let reading = Reading::host(&sample);
        assert_eq!(reading.pinned_reserve(unit), band + unit);
        assert_eq!(
            reading.pinned_headroom(others_step, unit),
            2 * GB - scarcity_threshold(band, unit, others_step),
            "pinned_headroom preserves reclaim_band + unit + guarantee without double-counting"
        );
    }
}
