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
//!   - Regrowth past B needs a real exogenous change with calm paging and
//!     `Normal` pressure: the store's `room` risen by `step` since B was set,
//!     others' cold pages risen by `step`, or free supply risen by
//!     `step.max(unit)` while `held + free` also exceeds `cap.anchor`
//!     (invariant under our own shrinks, since `-Δheld + Δfree == 0`).
//!   - Each OS pressure event (`seismic::PressureEvents`) while pressure is
//!     above `Normal` lowers B by one block (`unit`) per event; if pressure
//!     persists silently under immediate scarcity (`free <= scarcity_cushion`)
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

/// Immediate free + clean cache threshold at or below which the host is
/// considered scarce (relying on compression/swap), so sustained high-rate
/// paging can signal real contention even before OS pressure escalates.
fn scarcity_cushion(total: u64, unit: u64) -> u64 {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if total == 0 {
        return 0;
    }
    (total / 32).clamp(256 * MIB, 2 * GIB).max(unit).min(total / 4)
}

/// One store's figures at one sample: the ledger's one OS input
/// ([`super::ledger::Ledger::read`]). Fields are private: `supply ≤ total`
/// holds by construction.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    total: u64,
    supply: seismic::Supply,
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
    /// macOS Metal's kernel wired ceiling and current wire count.
    Wired(Ceiling),
    /// A static byte ceiling on device-accessible host RAM (unified CUDA's
    /// total or usable memory).
    Limit(u64),
    /// A Vulkan heap budget on host RAM: `left` bytes still granted beyond
    /// current heap usage, bounded by `capacity`.
    Budget { capacity: u64, left: u64 },
}

/// The system's wired ceiling and what is wired now, the process's
/// included.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ceiling {
    pub(crate) limit: u64,
    pub(crate) wired: u64,
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
        let grant = match sample.wired {
            Some(wired) => {
                let limit = match wired.limit {
                    seismic::WiredLimit::OsDefault => working_set,
                    seismic::WiredLimit::Set(bytes) => bytes.min(working_set),
                };
                HostGrant::Wired(Ceiling { limit, wired: wired.wire_count })
            }
            None => HostGrant::Limit(working_set),
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
        Self { total: capacity, supply, grant: None, paging: None, own_reclaimed: None, lag }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub(crate) fn supply(&self) -> seismic::Supply {
        self.supply
    }

    pub(crate) fn grant(&self) -> Option<HostGrant> {
        self.grant
    }

    pub(crate) fn ceiling(&self) -> Option<Ceiling> {
        match self.grant {
            Some(HostGrant::Wired(ceiling)) => Some(ceiling),
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
    /// The store's total bytes, used to scale the scarcity cushion.
    pub(crate) total: u64,
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
}

#[derive(Clone, Copy, Debug)]
struct Cap {
    /// The most the process may hold.
    bytes: u64,
    /// The step's bytes: the evidence a regrowth needs.
    step: u64,
    /// The room, others' cold pages, immediate free supply, and total
    /// `held + free` anchor captured when this cap was created (never mutated
    /// on own shrinks so `-Δheld + Δfree` cannot self-lift B).
    room: u64,
    cold: u64,
    free: u64,
    anchor: u64,
}

impl Boundary {
    /// Fold in one reading.
    pub(crate) fn observe(&mut self, now: Observed) {
        let Observed { paging, total, allocated, held, room, free, unit, events, committed } = now;
        self.committed = committed;
        let seismic::Counted { at, lag } = paging.counted;
        let cold = paging.cold;
        self.harmed = None;
        let scarce_free = free <= scarcity_cushion(total, unit);
        // When over half of total RAM is immediately free + clean cache
        // (e.g. Linux PSI firing on hugepage/DMA compaction with 77 GB free),
        // the host is not under memory scarcity.
        let abundant_free = total > 0 && free > total / 2;
        let prev_level = self.level;
        self.level = paging.pressure.filter(|level| *level != seismic::PressureLevel::Normal);
        let mut pressure_trigger = self.level.filter(|_| events > 0 && !abundant_free);
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
            // releasing toward the previous cap.
            let step_units = events.max(1);
            let blocks = unit.saturating_mul(step_units);
            let bytes = held.saturating_sub(blocks);
            let anchor = held.saturating_add(free);
            let cap = match self.cap {
                Some(cap) => Cap {
                    bytes: cap.bytes.saturating_sub(blocks).min(bytes),
                    step: cap.step.saturating_add(blocks),
                    ..cap
                },
                None => Cap { bytes, step: blocks, room, cold, free, anchor },
            };
            self.cap = Some(cap);
        }
        // Track the trough of exogenous room/cold/anchor while `cap` stands:
        // our own shrinks preserve `room` (`-Δin_use + Δsupply == 0`) and
        // `held + free` (`-Δheld + Δfree == 0`), so only others' growth lowers
        // these troughs. Recording the trough ensures that once a competing app
        // exits after pushing `room` down, recovery from the trough lifts `cap`.
        if let Some(cap) = self.cap.as_mut() {
            cap.room = cap.room.min(room);
            cap.cold = cap.cold.min(cold);
            cap.anchor = cap.anchor.min(held.saturating_add(free));
        }
        let pressure_calm = self.level.is_none()
            || (self.level == Some(seismic::PressureLevel::Warn)
                && self.pressed.is_none()
                && paging.harm.is_some()
                && !scarce_free);
        let max_lift_gain = scarcity_cushion(total, unit).max(unit);
        let lifted = |cap: &Cap| {
            let min_gain = cap.step.min(max_lift_gain).max(unit);
            room >= cap.room.saturating_add(min_gain)
                || cold >= cap.cold.saturating_add(min_gain)
                || (pressure_calm
                    && free >= cap.free.saturating_add(min_gain)
                    && held.saturating_add(free) >= cap.anchor.saturating_add(min_gain))
        };
        let Some(harm) = paging.harm else {
            // No share of the process's is counted: nothing judges harm,
            // growth or a step; B, where pressure set it, lifts on others' release.
            self.unjudged = true;
            if pressure_calm && self.cap.as_ref().is_some_and(lifted) {
                self.cap = None;
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
        let scarce = self.level == Some(seismic::PressureLevel::Critical) || scarce_free;
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
            let from = self.step.map_or(last.held, |step| step.from);
            self.step = Some(Step { from, grown: at });
            self.quiet_since = None;
        }
        let thrashing = confirmed_excess.is_some();
        match self.step {
            Some(step) if thrashing => {
                // Sustained thrashing within our growth step's window: return
                // B to the holding before the step began.
                let delta = held.saturating_sub(step.from).max(unit).max(paging.page_bytes);
                self.settle(Cap {
                    bytes: step.from,
                    step: delta,
                    room,
                    cold,
                    free,
                    anchor: held.saturating_add(free),
                });
            }
            Some(step) if allocated <= last.allocated && last.at >= step.grown + window => {
                self.step = None;
                self.judged = true;
            }
            Some(_) => {}
            None => {
                if let Some(excess) = confirmed_excess {
                    // Others' active working set is thrashing under scarcity:
                    // tighten B in place by the sustained excess without
                    // snapping down to lower `held` or overwriting `anchor`.
                    self.harmed = Some(excess);
                    let anchor = held.saturating_add(free);
                    self.cap = Some(match self.cap {
                        Some(cap) => Cap {
                            bytes: cap.bytes.saturating_sub(excess),
                            step: cap.step.saturating_add(excess),
                            ..cap
                        },
                        None => Cap {
                            bytes: held.saturating_sub(excess),
                            step: excess.max(unit).max(paging.page_bytes),
                            room,
                            cold,
                            free,
                            anchor,
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
        if calm && self.cap.as_ref().is_some_and(lifted) {
            self.cap = None;
        }
    }

    /// Hold no more than `cap` allows from now, a lower boundary standing.
    fn settle(&mut self, cap: Cap) {
        self.cap = Some(match self.cap {
            Some(standing) if standing.bytes <= cap.bytes => standing,
            _ => cap,
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
        let wired = |limit| seismic::HostSample {
            wired: Some(seismic::WiredFigures { limit, wire_count: 4 * GB }),
            ..seismic::host_samples::m4_pro()
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

    fn obs(at: Instant, total: u64, machine: u64, held: u64, free: u64, unit: u64, events: u64, pressure: seismic::PressureLevel) -> Observed {
        let paging = Paging {
            harm: Some(seismic::Harm { machine, own: 0 }),
            page_bytes: 16 * 1024,
            cold: 0,
            own_cold: None,
            pressure: Some(pressure),
            counted: seismic::Counted { at, lag: LAG },
        };
        Observed { paging, total, allocated: held, held, room: held + free, free, unit, events, committed: 20 * GB }
    }

    /// Normal OS compression/wakeup activity (e.g. a few MB/s of lazy
    /// decompressions, any burst when free+clean headroom is abundant, or an
    /// exogenous memory release where free rises) never locks `cap` or shrinks
    /// the model under `Normal` pressure.
    #[test]
    fn normal_background_decompressions_and_abundant_bursts_never_harm() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let total = 64 * GB;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert!(boundary.judges());

        // Continuous 20 MB/s background decompressions even with free = 0 remain well below THRASH_RATE_BPS (64 MiB/s).
        let mut pages = 0u64;
        for tick in 2..=6 {
            pages += (20 * MB) / page;
            boundary.observe(obs(start + Duration::from_secs(tick), total, pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
            assert_eq!(boundary.cap(), None, "normal compression touch tick {tick} must not set B");
            assert_eq!(boundary.harmed(), None);
        }

        // Single huge tab-switch / app wakeup while free headroom is abundant (4 GB > 2 GB cushion).
        pages += (512 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(7), total, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        pages += (512 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(8), total, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "paging under abundant free supply never sets B");

        // Multi-GB decompression burst when an app frees memory (free rises +4 GB).
        pages += (4 * GB) / page;
        boundary.observe(obs(start + Duration::from_secs(9), total, pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        pages += (4 * GB) / page;
        boundary.observe(obs(start + Duration::from_secs(10), total, pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "exogenous memory release must not be read as contention");
    }

    /// A single short spike above THRASH_RATE_BPS clears cleanly when the next
    /// window calms down, without setting `cap`.
    #[test]
    fn transient_one_shot_spike_does_not_fire_thrashing() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let total = 64 * GB;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));

        // Single 150 MB/s spike for 1s, then back to calm 4 MB/s.
        let spike = (150 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), total, spike, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None);
        boundary.observe(obs(start + Duration::from_secs(3), total, spike + (4 * MB) / page, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
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
        let total = 64 * GB;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(1), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));

        // Two consecutive 1s spans at 128 MB/s under scarcity (excess = 2 * 64 MB = 128 MB).
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), total, rate_pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "first span arms without firing yet");
        boundary.observe(obs(start + Duration::from_secs(3), total, 2 * rate_pages, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        let first_cap = boundary.cap().expect("second sustained thrash span sets B");
        assert_eq!(first_cap, 30 * GB - 128 * MB);

        // Model shrinks to `held = first_cap - 600 MB`, returning 600 MB + 128 MB to free.
        // Because `held + free` equals the captured anchor (`30 GB`), calm readings
        // must NOT self-lift B.
        let held_after_shrink = first_cap - 600 * MB;
        let free_from_shrink = 728 * MB;
        boundary.observe(obs(start + Duration::from_secs(4), total, 2 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), Some(first_cap), "own shrink must never self-lift B");

        // Renewed sustained thrashing at 128 MB/s tightens cap in place by excess alone, never snapping to `held`.
        boundary.observe(obs(start + Duration::from_secs(5), total, 3 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        boundary.observe(obs(start + Duration::from_secs(6), total, 4 * rate_pages, held_after_shrink, free_from_shrink, unit, 0, seismic::PressureLevel::Normal));
        let second_cap = boundary.cap().expect("standing B tightens cleanly");
        assert!(second_cap > held_after_shrink + 400 * MB);
        assert_eq!(second_cap, first_cap - 128 * MB);

        // Calm exogenous app release lifts B once `held + free >= anchor + step` and `free >= cap.free + step`.
        boundary.observe(obs(start + Duration::from_secs(7), total, 4 * rate_pages, held_after_shrink, free_from_shrink + GB, unit, 0, seismic::PressureLevel::Normal));
        assert_eq!(boundary.cap(), None, "exogenous free lifts B cleanly");
    }

    /// Standing OS pressure under scarcity yields one block per dwell interval
    /// without edge events:
    /// - `Warn` past 1 s only on level-triggered / unjudged OSes (`harm: None`,
    ///   e.g. Windows `LowMemoryResourceNotification`), never on judged OSes
    ///   (macOS/Linux) where standing `Warn` is normal desktop compressor state;
    /// - `Critical` past 250 ms on every OS while `free <= scarcity_cushion`;
    /// - neither fires when immediate `free` exceeds `scarcity_cushion`.
    #[test]
    fn standing_pressure_yields_past_dwell_even_without_edge_events() {
        let start = Instant::now();
        let total = 64 * GB;
        let unit = 256 * MB;
        let unjudged = |at, held, free, pressure| {
            let mut o = obs(at, total, 0, held, free, unit, 0, pressure);
            o.paging.harm = None;
            o
        };

        // Level-triggered / unjudged OS (Windows): Warn past 1s under scarcity yields.
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

        // Abundant free supply suppresses standing dwell even under Critical.
        boundary.observe(unjudged(start + Duration::from_millis(2500), 30 * GB, 4 * GB, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None);
        assert_eq!(boundary.cap(), Some(30 * GB - 2 * unit));

        // Judged OS (macOS / Linux): standing Warn with events == 0 NEVER drains,
        // even for 60s, while Critical under scarcity still yields at 250ms.
        let mut macos = Boundary::default();
        macos.observe(obs(start, total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Normal));
        for sec in 1..=60 {
            macos.observe(obs(start + Duration::from_secs(sec), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Warn));
            assert_eq!(macos.pressed(), None, "standing Warn on macOS/Linux must never drain (sec {sec})");
            assert_eq!(macos.cap(), None);
        }
        // Escalation to Critical under scarcity starts dwell cleanly and yields after 250ms.
        macos.observe(obs(start + Duration::from_secs(61), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Critical));
        assert_eq!(macos.pressed(), None);
        macos.observe(obs(start + Duration::from_millis(61_250), total, 0, 30 * GB, 0, unit, 0, seismic::PressureLevel::Critical));
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
        let total = 64 * GB;
        let unit = 256 * MB;
        let page = 16 * 1024;
        boundary.observe(obs(start, total, 0, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        boundary.observe(obs(start + Duration::from_secs(1), total, 0, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert!(boundary.judges());

        // High-rate lazy decompressions (128 MiB/s) under standing Warn with 4 GB free (> 2 GB cushion):
        // must NOT fire Harm because immediate free supply is abundant.
        let rate_pages = (128 * MB) / page;
        boundary.observe(obs(start + Duration::from_secs(2), total, rate_pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        boundary.observe(obs(start + Duration::from_secs(3), total, 2 * rate_pages, 30 * GB, 4 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.harmed(), None, "standing Warn with abundant free must not fire Harm");
        assert_eq!(boundary.cap(), None);

        // Competing app grows +10 GB: triggers pressure events, pushes room down from 30 GB to 20 GB.
        boundary.observe(obs(start + Duration::from_secs(4), total, 2 * rate_pages, 30 * GB, 0, unit, 1, seismic::PressureLevel::Warn));
        assert_eq!(boundary.cap(), Some(30 * GB - unit));
        for tick in 5..=14 {
            let held = 30 * GB - (tick - 4) * GB;
            boundary.observe(obs(start + Duration::from_secs(tick), total, 2 * rate_pages, held, 0, unit, 1, seismic::PressureLevel::Warn));
        }
        assert!(boundary.cap().is_some());

        // Competing app exits, freeing 10 GB: room recovers from trough (20 GB) to 30 GB,
        // free jumps to 10 GB (> cushion), even while macOS stays at standing Warn.
        boundary.observe(obs(start + Duration::from_secs(15), total, 2 * rate_pages, 20 * GB, 10 * GB, unit, 0, seismic::PressureLevel::Warn));
        assert_eq!(boundary.cap(), None, "exogenous recovery from trough must lift cap even under standing Warn");

        // Abundant free supply (> total/2) suppresses even edge pressure events
        // (e.g. Linux PSI compaction stalls when >50% of host RAM is free).
        boundary.observe(obs(start + Duration::from_secs(16), total, 2 * rate_pages, 20 * GB, 36 * GB, unit, 1, seismic::PressureLevel::Critical));
        assert_eq!(boundary.pressed(), None, "pressure event with >50% of total RAM free must not fire");
        assert_eq!(boundary.cap(), None);
    }
}
