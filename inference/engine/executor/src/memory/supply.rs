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
//! - **The boundary** ([`Boundary`]): others' working set observed as harm
//!   (pages they bring back), not bytes. Cold pages the OS compresses or
//!   swaps cost their owner nothing; a working set pushed out costs it
//!   everything, and only harm tells the two apart.
//!   - Others' harm only: the machine's pages in less the process's own,
//!     counted together in one snapshot ([`seismic::Harm`]), judged on the
//!     counters' own clock ([`seismic::Counted`]) and only over spans of at
//!     least the counters' lag, so a share of the process's own that a fold
//!     shows late (Linux memcg's flush) is never read as others'. Where the
//!     OS counts no share of the process's, harm and others' cold pages
//!     cannot be judged: growth is bounded by the supply alone there.
//!   - The baseline is the largest harm rate over a stretch of readings in
//!     which the process did not grow, once the stretch spans the
//!     counters' lag; learned again in every such stretch, never while the
//!     process grows, so its growth never moves the figure it is judged by.
//!   - A growth opens a step, or joins the one open; the step's window
//!     λ_h is the counters' lag, so a step's harm shows within it. It
//!     closes at the first reading whose whole span lies past λ_h after the
//!     latest growth.
//!   - Harm is judged against the baseline's spread across readings: the
//!     pages past what its rate counts over each span are summed, never
//!     below none, until they pass one page, the counters' unit, and then
//!     every whole page of them is excess. So no reading's span forgives a
//!     page of its own and the verdict does not move with the reading
//!     cadence; pages not judged yet join no baseline.
//!   - Harm above the baseline within a step marks the boundary B: the
//!     process may hold no more than it held as the step began.
//!   - Harm above the baseline with no step open is others' working set
//!     growing: B falls by the harm's excess bytes, once; the rise joins
//!     the stretch's baseline, so the same level does not yield again.
//!   - Regrowth past B needs a real change in others, with harm at the
//!     baseline: the store's room (what the sample shows the process holds
//!     and the supply beside it) risen by the step's bytes since B was set,
//!     or others' cold pages risen by them. The process's own release moves
//!     neither (§4.2.1), so its own settle never admits its regrowth.
//!   - Others' cold pages are supply only once a baseline judges harm:
//!     before it, growth into them could not be told from harm.
//!   - Each pressure event the OS delivers (`seismic::PressureEvents`: the
//!     macOS memorypressure source, a Linux PSI trigger, Windows' low-memory
//!     notification) while its level is set is a real shortage whatever
//!     caused it. It names no bytes, so B falls by one block of the store's
//!     pool (the unit a shrink returns to the OS) per new event: the drain
//!     is the OS's edges, never the reading cadence or a window. A level
//!     that stands with no new event drains nothing more, and a reading of
//!     a host that delivers no events (Linux without PSI triggers) takes no
//!     pressure step at all: harm alone moves B there. B lifts only once
//!     the level is normal and B's regrowth rule holds for the blocks
//!     yielded.

use std::time::Instant;

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
    /// The largest rate of others' harm over the latest stretch without
    /// growth that spanned λ_h, in pages a second.
    baseline: Option<f64>,
    /// Others' pages past the baseline's spread not yet judged excess
    /// ([`judge`]).
    unexplained: f64,
    quiet: Option<Quiet>,
    step: Option<Step>,
    cap: Option<Cap>,
    /// Others' harm past the baseline at the latest reading, with no step
    /// of the process's open: bytes others brought back.
    harmed: Option<u64>,
    /// The OS's pressure level at the latest reading, where it is set.
    level: Option<seismic::PressureLevel>,
    /// The level, where a new pressure event came with it at the latest
    /// reading.
    pressed: Option<seismic::PressureLevel>,
    /// The OS counts no share of the process's in the machine's harm: harm
    /// and others' cold pages cannot be judged.
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
    /// What the process had physically allocated as the counters show it (at
    /// `paging.counted.at`): what physical growth steps are judged against.
    pub(crate) allocated: u64,
    /// What the process holds in the ledger (`committed + cache`): the unit
    /// that `cap` bounds across `import_room` and admission.
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
}

#[derive(Clone, Copy, Debug)]
struct Quiet {
    since: Instant,
    spread: f64,
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
    /// The room, others' cold pages, and immediate free supply when it was set.
    room: u64,
    cold: u64,
    free: u64,
}

/// Others' pages brought back past what the baseline's spread counts,
/// summed across readings since harm was last judged excess, never below
/// none: `unexplained` before a reading of `harmed` pages over `span`
/// seconds, judged against a baseline of `spread` pages a second. Excess
/// is every whole page once they pass one, the counters' unit, so no
/// reading's span forgives a page of its own and the verdict does not
/// move with the reading cadence. What the reading leaves unexplained,
/// and its excess pages.
fn judge(unexplained: f64, harmed: u64, spread: f64, span: f64) -> (f64, u64) {
    let unexplained = (unexplained + harmed as f64 - spread * span).max(0.0);
    match unexplained > 1.0 {
        true => (unexplained.fract(), unexplained.trunc() as u64),
        false => (unexplained, 0),
    }
}

impl Boundary {
    /// Fold in one reading.
    pub(crate) fn observe(&mut self, now: Observed) {
        let Observed { paging, allocated, held, room, free, unit, events, committed } = now;
        self.committed = committed;
        let seismic::Counted { at, lag } = paging.counted;
        let cold = paging.cold;
        self.harmed = None;
        self.level = paging.pressure.filter(|level| *level != seismic::PressureLevel::Normal);
        self.pressed = self.level.filter(|_| events > 0);
        if self.pressed.is_some() {
            // Each new pressure event names no bytes: B holds the process to
            // one block less than it holds per event, the blocks' sum the
            // evidence a regrowth needs.
            let blocks = unit.saturating_mul(events);
            let bytes = held.saturating_sub(blocks);
            let cap = match self.cap {
                Some(cap) => Cap { bytes: cap.bytes.min(bytes), step: cap.step.saturating_add(blocks), ..cap },
                None => Cap { bytes, step: blocks, room, cold, free },
            };
            self.cap = Some(cap);
        }
        let level = self.level;
        let lifted = |cap: &Cap| {
            room >= cap.room.saturating_add(cap.step)
                || cold >= cap.cold.saturating_add(cap.step)
                || (level.is_none()
                    && free >= cap.free.saturating_add(cap.step.max(unit))
                    && held.saturating_add(free) > cap.bytes.saturating_add(cap.step))
        };
        let Some(harm) = paging.harm else {
            // No share of the process's is counted: nothing judges harm,
            // growth or a step; B, where pressure set it, lifts on others' release.
            self.unjudged = true;
            if self.level.is_none() && self.cap.as_ref().is_some_and(lifted) {
                self.cap = None;
            }
            return;
        };
        self.unjudged = false;
        let point = Point { at, machine: harm.machine, own: harm.own, allocated, held };
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
        // Others' pages past what the baseline's spread counts, judged
        // across readings; the pages this reading leaves unexplained are
        // not judged yet, so they join no baseline.
        let (excess, carried) = match self.baseline {
            Some(spread) => {
                let before = self.unexplained;
                let (unexplained, excess) = judge(before, harmed, spread, span);
                self.unexplained = unexplained;
                (Some(excess), (unexplained - before).max(0.0))
            }
            None => (None, 0.0),
        };
        let rate = (harmed as f64 - carried) / span;
        let risen = excess.is_some_and(|excess| excess > 0);
        if allocated > last.allocated {
            let from = self.step.map_or(last.held, |step| step.from);
            self.step = Some(Step { from, grown: at });
            self.quiet = None;
        }
        match self.step {
            Some(step) if risen => {
                // Harm within the window of the process's growth: the
                // boundary is the holding before it.
                self.settle(Cap { bytes: step.from, step: held.saturating_sub(step.from), room, cold, free });
            }
            Some(step) if allocated <= last.allocated && last.at >= step.grown + window => self.step = None,
            Some(_) => {}
            None => {
                if let Some(excess) = excess.filter(|excess| *excess > 0) {
                    let excess = excess.saturating_mul(paging.page_bytes);
                    // Others' working set grew of itself: what the process
                    // may hold falls by the harm's excess, once. Tighten a
                    // standing cap in place instead of snapping it down to
                    // `held` or overwriting its regrowth anchor.
                    self.harmed = Some(excess);
                    self.cap = Some(match self.cap {
                        Some(cap) => Cap {
                            bytes: cap.bytes.saturating_sub(excess),
                            step: cap.step.saturating_add(excess),
                            ..cap
                        },
                        None => Cap {
                            bytes: held.saturating_sub(excess),
                            step: excess,
                            room,
                            cold,
                            free,
                        },
                    });
                }
                let quiet = self.quiet.get_or_insert(Quiet { since: last.at, spread: 0.0 });
                quiet.spread = quiet.spread.max(rate);
                if at.duration_since(quiet.since) >= window {
                    self.baseline = Some(quiet.spread);
                }
            }
        }
        let calm = excess == Some(0) && self.step.is_none() && self.level.is_none();
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

    /// Whether a baseline judges harm: others' cold pages are supply from
    /// then.
    pub(crate) fn judges(&self) -> bool {
        self.baseline.is_some()
    }

    /// The most the process may hold in the store, where B stands: never
    /// below what is committed at the latest reading, so B bounds the
    /// cache and the floor the load serves with is never shed to it.
    pub(crate) fn cap(&self) -> Option<u64> {
        self.cap.map(|cap| cap.bytes.max(self.committed))
    }

    /// Whether growth is admissible: a baseline judges harm, and no
    /// growth step's window is open, so a step's harm is told from the
    /// next's; or the OS counts no share of the process's, where nothing
    /// judges harm and the supply alone bounds growth.
    pub(crate) fn admits_growth(&self) -> bool {
        self.unjudged || (self.baseline.is_some() && self.step.is_none())
    }

    /// End any active counter span before an unmonitored idle sleep, keeping
    /// the host's judged baseline and any standing boundary B: the next
    /// reading starts a fresh span at its wake instant instead of attributing
    /// paging while asleep to one active interval.
    pub(crate) fn settle_idle(&mut self) {
        self.last = None;
        self.unexplained = 0.0;
        self.quiet = None;
        self.step = None;
        self.harmed = None;
        self.pressed = None;
    }

    /// Whether a growth step's window λ_h is open, so only a later reading
    /// shows whether the growth harmed others or closes.
    pub(crate) fn step_open(&self) -> bool {
        self.step.is_some()
    }

    /// Others' harm past the baseline at the latest reading with no step
    /// of the process's open, in bytes.
    pub(crate) fn harmed(&self) -> Option<u64> {
        self.harmed
    }

    /// The pressure level, where a new pressure event came with it at the
    /// latest reading.
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

    /// One reading of a process holding 1 GB, never growing, its counters
    /// showing `machine` pages brought back at `at`, none of them its own.
    fn reading(at: Instant, machine: u64) -> Observed {
        let paging = Paging {
            harm: Some(seismic::Harm { machine, own: 0 }),
            page_bytes: 4096,
            cold: 0,
            own_cold: Some(0),
            pressure: Some(seismic::PressureLevel::Normal),
            counted: seismic::Counted { at, lag: LAG },
        };
        Observed { paging, allocated: GB, held: GB, room: 2 * GB, free: GB, unit: 0, events: 0, committed: 0 }
    }

    /// Whether harm sets B within one second of readings `cadence` apart,
    /// after a baseline of 0.01 pages a second (one page over a quiet
    /// 100 s), where others bring a page back every 200 ms (`harming`) or
    /// none.
    fn judged_at(cadence: Duration, harming: bool) -> bool {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        boundary.observe(reading(start, 0));
        let quiet = start + Duration::from_secs(100);
        boundary.observe(reading(quiet, 1));
        assert!(boundary.judges(), "the quiet stretch's baseline judges harm");
        let readings = (Duration::from_secs(1).as_millis() / cadence.as_millis()) as u32;
        for reading_at in (1..=readings).map(|n| quiet + cadence * n) {
            let brought = match harming {
                true => (reading_at - quiet).as_millis() as u64 / 200,
                false => 0,
            };
            boundary.observe(reading(reading_at, 1 + brought));
        }
        boundary.cap().is_some()
    }

    /// Harm is judged across readings, so the reading cadence never moves
    /// the verdict: five pages brought back over a second against a
    /// baseline of 0.01 pages a second set B whether the second is read
    /// every 100 ms (a page every other reading, which a page forgiven per
    /// reading never judged) or once; no harm sets it at neither.
    #[test]
    fn the_harm_verdict_is_the_same_at_any_cadence() {
        for harming in [true, false] {
            let verdicts = [Duration::from_millis(100), Duration::from_secs(1)].map(|cadence| judged_at(cadence, harming));
            assert_eq!(verdicts, [harming; 2], "harming: {harming}");
        }
    }

    /// Repeated small harm ticks while `held` sits 600 MB below a standing
    /// `cap.bytes` subtract only `excess` from `cap.bytes` and preserve the
    /// regrowth anchor instead of snapping `cap` down to `held` (no one-way
    /// cap ratchet).
    #[test]
    fn repeated_small_harm_never_snaps_cap_down_to_held() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let unit = 256 * MB;
        let obs = |at: Instant, machine: u64, held: u64, free: u64| Observed {
            paging: Paging {
                harm: Some(seismic::Harm { machine, own: 0 }),
                page_bytes: 16 * 1024,
                cold: 0,
                own_cold: None,
                pressure: Some(seismic::PressureLevel::Normal),
                counted: seismic::Counted { at, lag: LAG },
            },
            allocated: held,
            held,
            room: held + free,
            free,
            unit,
            events: 0,
            committed: 20 * GB,
        };
        // Learn a zero-harm baseline at 30 GB held with 0 free (scarce).
        boundary.observe(obs(start, 0, 30 * GB, 0));
        boundary.observe(obs(start + Duration::from_secs(1), 0, 30 * GB, 0));
        assert!(boundary.judges());

        // First 3 MB harm tick sets cap at 30 GB - 3 MB.
        let three_mb_pages = (3 * MB) / (16 * 1024);
        boundary.observe(obs(start + Duration::from_secs(2), three_mb_pages, 30 * GB, 0));
        let first_cap = boundary.cap().expect("first harm tick sets B");
        assert_eq!(first_cap, 30 * GB - 3 * MB);

        // Now `held` sits 600 MB below `cap.bytes` (e.g. bank headroom/reserve),
        // and another 6 MB harm tick arrives (double rate so it exceeds the
        // updated quiet baseline). Cap must fall by only 3 MB excess (the
        // delta above the 3 MB/s baseline), NEVER snap down to `held`.
        let held_below = first_cap - 600 * MB;
        boundary.observe(obs(start + Duration::from_secs(3), three_mb_pages + 2 * three_mb_pages, held_below, 0));
        let second_cap = boundary.cap().expect("cap remains standing");
        assert!(
            second_cap > held_below + 500 * MB,
            "cap ({second_cap}) must tighten by excess alone, never snap down to held ({held_below})"
        );
        assert_eq!(second_cap, first_cap - 3 * MB);
    }

    /// Steady background OS decompressions raise the quiet baseline after the
    /// first excess tick, so subsequent ticks at the same rate emit no further
    /// Harm and never ratchet `cap` downward.
    #[test]
    fn steady_background_decompression_raises_baseline_without_ratcheting_cap() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let unit = 256 * MB;
        let obs = |at: Instant, machine: u64| Observed {
            paging: Paging {
                harm: Some(seismic::Harm { machine, own: 0 }),
                page_bytes: 16 * 1024,
                cold: 0,
                own_cold: None,
                pressure: Some(seismic::PressureLevel::Normal),
                counted: seismic::Counted { at, lag: LAG },
            },
            allocated: 30 * GB,
            held: 30 * GB,
            room: 36 * GB,
            free: 6 * GB,
            unit,
            events: 0,
            committed: 20 * GB,
        };
        boundary.observe(obs(start, 0));
        boundary.observe(obs(start + Duration::from_secs(10), 0));
        assert!(boundary.judges());

        // First 4 MB tick sets cap to 30 GB - 4 MB and raises baseline to 4 MB / 10s.
        let four_mb_pages = (4 * MB) / (16 * 1024);
        let mut machine = four_mb_pages;
        boundary.observe(obs(start + Duration::from_secs(20), machine));
        assert_eq!(boundary.cap(), Some(30 * GB - 4 * MB));

        // Subsequent 4 MB ticks at the same cadence are within the learned
        // baseline: zero additional Harm and cap never ratchets further.
        for tick in 3..=6 {
            machine += four_mb_pages;
            boundary.observe(obs(start + Duration::from_secs(tick * 10), machine));
            assert_eq!(boundary.cap(), Some(30 * GB - 4 * MB), "steady background decompression must not ratchet B on tick {tick}");
            assert_eq!(boundary.harmed(), None, "steady background decompression must not emit Harm on tick {tick}");
        }
    }

    /// A standing `cap` lifts on a calm reading once `PressureLevel::Normal`
    /// and abundant free supply exceeds `cap.bytes + cap.step`.
    #[test]
    fn cap_lifts_on_calm_when_abundant_free_supply_exceeds_cap_plus_step() {
        let mut boundary = Boundary::default();
        let start = Instant::now();
        let unit = 256 * MB;
        let obs = |at: Instant, machine: u64, free: u64| Observed {
            paging: Paging {
                harm: Some(seismic::Harm { machine, own: 0 }),
                page_bytes: 16 * 1024,
                cold: 0,
                own_cold: None,
                pressure: Some(seismic::PressureLevel::Normal),
                counted: seismic::Counted { at, lag: LAG },
            },
            allocated: 30 * GB,
            held: 30 * GB,
            room: 30 * GB + free,
            free,
            unit,
            events: 0,
            committed: 20 * GB,
        };
        boundary.observe(obs(start, 0, 0));
        boundary.observe(obs(start + Duration::from_secs(1), 0, 0));
        // Scarce harm sets cap.
        let pages = (4 * MB) / (16 * 1024);
        boundary.observe(obs(start + Duration::from_secs(2), pages, 0));
        assert_eq!(boundary.cap(), Some(30 * GB - 4 * MB));

        // Calm reading with 2 GB free (> unit and allocated + free > cap + step) lifts cap.
        boundary.observe(obs(start + Duration::from_secs(3), pages, 2 * GB));
        assert_eq!(boundary.cap(), None, "abundant calm free supply lifts standing B");
    }
}
