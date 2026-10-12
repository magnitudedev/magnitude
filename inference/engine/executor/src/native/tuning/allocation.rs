//! One allocator spends a first-load tuning budget on the units' optional
//! work by value.
//!
//! The budget governs what is started, never what is running. Every unit
//! enters holding a legal published choice, and its choice is legal and
//! published after every piece of its work. What a unit's work adds to its
//! record besides its choice, its measurements, is published once, when
//! the allocator stops: a unit whose record changed since it was published
//! owes that publication ([`Unit::owed`]), priced like any work, and a
//! piece is started only when its whole price and everything owed once it
//! has run fit what remains of the budget ([`seismic::Quote::fits`]). The
//! allocator stops when no unit has such work left and settles what is
//! owed ([`Unit::settle`]) inside the budget. The budget is used in full,
//! to the smallest piece that still fits: a unit runs out of work only
//! when its domain is exhausted.
//! A piece runs to its end or to its cutoff: the instant past which what
//! is owed after it would no longer fit the budget ([`Unit::perform`]). A
//! forecast that was wrong leaves less for what follows; time past the
//! budget is reported with the work that ran into it, its forecast and
//! what it took, and fails nothing.
//!
//! # The ending is work
//!
//! What ends the optional work is owed like any piece of it, and every
//! start holds it back ([`Ending`]). Nothing of it starts unadmitted:
//!
//! - A unit that will be given no more work has its final choice, and what
//!   it owes is settled then, not at the stop: a publication that has run
//!   needs no forecast, and only the units still working at the stop are
//!   held back for by one.
//! - The allocator's own time is of classes it observes as it runs: a pass
//!   that decides what starts, a unit narrowing its piece, a unit giving
//!   up its inputs. A piece is started only if, after it, the pass that
//!   finds nothing more to start and the inputs then given up also fit,
//!   each at its class's price. A class the allocator has not yet run is
//!   held at what bounds it from what the load has observed ([`Own`]): a
//!   pass at the longest operation completed, a narrowing at a pass, and a
//!   unit giving up its inputs at what building them took
//!   ([`Unit::built`]). Inputs are given up after everything owed is
//!   settled, so a release that outruns its price delays no publication.
//! - Narrowing ([`Unit::narrow`]) is work of the allocator's: a unit's
//!   piece is narrowed only while the narrowing and everything owed still
//!   fit, the piece worth the most first. Units keep their inputs while
//!   pieces are narrowed to what is left.
//!
//! Each unit names its next piece of work with its price and the share of
//! its default service the work is expected to recover ([`seismic::Work`]).
//! The allocator starts, of the pieces that fit:
//!
//! 1. a conversion ([`Stage::Promote`]) before anything else: a measured
//!    lead not yet served is budget already spent with nothing to show;
//! 2. then discovery ([`Stage::Discovery`]), the work whose classes this
//!    load has not observed and so cannot price: a unit never started
//!    first, then the largest share first (preparation order while no
//!    share is known). Discovery is ahead of every challenger because
//!    nothing can be valued without it: a unit's challenger has no price
//!    and no gain before its defaults are measured, and a unit's share is
//!    its defaults' time against every unit's, so until every unit is
//!    discovered the shares that rank challengers overstate the units
//!    measured so far and rank the others at nothing;
//! 3. then the first representatives of structurally different forms
//!    ([`Stage::Form`]), the piece worth the most per second first: no
//!    refinement inside one form, of any unit, says whether another form
//!    is faster, so every unit's forms that fit are measured before any
//!    unit refines one. (A unit whose forms are compiling is in flight;
//!    the device meanwhile takes other work rather than wait.)
//! 4. then the challenger ([`Stage::Measure`]) or the conversion of a
//!    lead not yet resolved ([`Stage::Confirm`]) worth the most per second
//!    of its price: the unit's share of served time, times the gain the
//!    unit expects of it.
//!
//! A unit composed of other units' choices reads them as they stand when
//! its own work starts; from then they are sealed and receive no more work.
//!
//! # Compiles overlap the device
//!
//! The budget is wall time, and a piece's compile ([`Work::forming`]) needs
//! no device: it runs on a worker ([`Workers`]) while the device measures
//! other work, and its piece measures when its programs are held. The piece
//! is still the unit of admission. It is admitted whole, on its whole
//! price, when its compile starts, and from then it is in flight: its
//! measurement runs when its compile has ended, before anything new is
//! admitted.
//!
//! A compile cannot be interrupted, and one that outruns its price must
//! not take the time held back for the ending, nor run past the budget.
//! A piece that compiles is started only if its compile ends within the
//! budget at the most a compile has taken as far as the load has seen:
//! the longest that has ended, and one that nothing served
//! ([`Unit::cold_compile`]). Where a system cache serves most compiles, the
//! compiles a load observes are its hits, and a miss's cost is seen only
//! in the compile that nothing served. The ending begins at the
//! cutoff of the pieces in flight, whatever their compiles are doing: the
//! allocator waits for a compile no later than that, and from it nothing
//! more is admitted or measured. A piece whose measurement has not run by
//! then is abandoned ([`Abandoned`]), its unit's choice and record as they
//! were. A compile selects nothing, and every unit holds a legal published
//! choice, so what the units owe is settled while the abandoned compiles
//! still run; they are waited for only once it is, and what they formed is
//! dropped.
//!
//! Three resources are contended, and each is counted as itself.
//!
//! *The device* measures one piece at a time, and takes the admitted
//! pieces in the order they are ready for it: a piece in flight when its
//! own compile is forecast to end ([`Flight::forming_ends`]), a piece that
//! compiles nothing now. The forecast of when the admitted work ends is
//! that queue run forward ([`device_ends`]): the device, free now, takes
//! each piece when it is ready and the pieces ready before it are done. A
//! new piece is admitted when the whole queue with it in its place, and
//! what the units then owe, end within the budget: the one admission rule
//! ([`seismic::Quote::fits`]), which with nothing in flight is that rule
//! unchanged. A compile that is forecast to end long after a new piece
//! would have measured is behind it in the queue and holds nothing of it
//! back; only its own device time must still fit after.
//!
//! *Worker threads* run the compiles, and a compile needs as many as it
//! forms at once ([`Workers::needs`]). There are as many as the host runs
//! in parallel ([`Workers::free`]), and a compile starts only when the
//! threads it needs are free: a compile started on threads it shares runs
//! longer than it was priced. Nothing else limits how many compiles run at
//! once.
//!
//! *Memory*: a piece in flight holds its footprint from the moment its
//! compile starts (below), so compiles overlap only as far as their
//! pieces' bytes fit under the ceiling together.
//!
//! A compile is started so that its piece is ready when the device is done
//! with the pieces ready before it, and no sooner: at the forecast end of
//! that work less the compile's own forecast, or now if that is past.
//! Started later, the device would wait for it. Started earlier, it would
//! commit the device to a piece that the evidence of the pieces ahead of
//! it might outrank, and gain nothing: its piece would wait its turn. With
//! compiles of one second and measurements of a fifth, five compiles are
//! in flight at once, each started a fifth of a second after the last.
//! Until a compile is due the allocator waits, for that moment, for a
//! compile to end or for the ending, whichever is first
//! ([`Workers::wait`]).
//!
//! Time past the budget is at most what work in flight when the budget
//! ended took: a piece performed across the budget's end because its
//! runtime could not stop it sooner, or an abandoned compile that ran past
//! the settling. Nothing is admitted after it.
//!
//! # Bytes are admitted like seconds
//!
//! The load's tuning has a ceiling on what it holds ([`Unit::free`] is what
//! is left under it). A piece is priced in bytes as it is in seconds
//! ([`seismic::Footprint`]): what it adds, at its most, to what is held when
//! it starts. It is started only if that fits what is free, less the bytes
//! of the pieces in flight; a piece that does not fit is not started, and
//! nothing raises the ceiling for it.
//!
//! Where the device has memory of its own, host RAM is a second domain with
//! its own ceiling ([`Unit::host_free`]) and its own price
//! ([`Unit::host_footprint`]). The two do not substitute: a piece starts
//! only if it fits both, and one that the host refuses is made room for as
//! one the device refuses is, by what the load returns and what other units
//! give up.
//!
//! A unit's built inputs are owned by its work. They stay between two of
//! its pieces while both hold: the unit still offers work that fits what
//! remains of the budget, and no piece chosen ahead of it needs the bytes.
//! Keeping them costs nothing while they fit, and building them again
//! costs seconds of the budget, so nothing is given up without need. A
//! unit with no work left that can start gives its inputs up at once: one
//! whose work no longer fits the budget, and one refused for its bytes for
//! the load, whose piece with its own inputs needs more than the optional
//! work may ever hold ([`Unit::capacity`]). A unit refused for bytes that
//! others still hold keeps its inputs while other work may free them, and
//! gives them up when the allocator stops. When
//! the chosen piece does not fit, the units not in flight give theirs up
//! one at a time, the one whose own next work is worth least first, until
//! it fits; a unit that gave its inputs up offers its next piece with
//! their rebuilding priced in, so it competes at what it now costs.
//!
//! The programs a unit formed are held the same way ([`Unit::programs`]):
//! they stay for as long as they fit, so nothing is compiled twice, and
//! are given up by need alone, never by what they were formed for. When
//! the chosen piece does not fit, a unit gives up its programs before its
//! inputs ([`Unit::give_up`]), the unit whose next work is worth least
//! first, and its next piece is priced with forming what it runs again.
//! Everything a unit still holds leaves with it when the optional work
//! ends.

use seismic::{Quote, Stage, Work};
use std::time::Duration;

/// What is free of host RAM where it is a memory domain of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostRoom {
    /// The host is not a domain apart, or has no ceiling.
    Unbounded,
    /// The bytes left under the host's ceiling.
    Free(u64),
    /// The load's tuning holds more on the host than its ceiling: nothing
    /// starts, whatever it is priced, until what is held for nothing has
    /// been returned or given up. Most of what a compile takes it frees
    /// again, and freed memory counts until it is returned
    /// ([`Unit::reclaim`]), so this is also what returns it in time.
    Exceeded,
}

impl HostRoom {
    /// Whether a piece taken to add `bytes` may start beside pieces in
    /// flight taken to add `flying`.
    fn admits(self, bytes: u64, flying: u64) -> bool {
        match self {
            Self::Unbounded => true,
            Self::Free(free) => bytes <= free.saturating_sub(flying),
            Self::Exceeded => false,
        }
    }
}

/// Time since the tuning began.
pub(crate) trait Clock {
    fn now(&self) -> Duration;
}

/// One tuning unit's optional work. The unit holds a legal published choice
/// before, between and after every piece of it.
pub(crate) trait Unit {
    type Context;
    type Failure;
    /// The compile of a piece of work, to run off the device, and what it
    /// formed.
    type Forming;
    type Formed;
    /// Recompute the shared served-work objective from current evidence.
    /// Refresh performs no device work.
    fn refresh(_units: &mut [Self], _context: &mut Self::Context)
    where
        Self: Sized,
    {
    }
    /// The units whose current choices this unit's work reads.
    fn prerequisites(&self) -> Vec<usize> {
        Vec::new()
    }
    /// The unit's share of expected step time at its defaults.
    fn share(&self) -> f64;
    /// The unit's next piece of work, priced in full; none when it has
    /// none left.
    fn next(&self) -> Option<Work>;
    /// The compile of the unit's next piece of work ([`Work::forming`]);
    /// none when the piece compiles nothing ahead of its measurement.
    fn forming(&self) -> Option<Self::Forming>;
    /// Hold what the next piece's compile formed, which took `took`.
    fn formed(
        &mut self,
        formed: Self::Formed,
        took: Duration,
        context: &mut Self::Context,
    ) -> Result<(), Self::Failure>;
    /// Run the unit's next piece of work to its end, or until `before`, the
    /// piece's cutoff, whichever is first: what the piece has completed by
    /// then is kept, and nothing incomplete becomes evidence. Returns the
    /// longest single operation the unit has completed.
    fn perform(
        &mut self,
        context: &mut Self::Context,
        before: Duration,
    ) -> Result<Duration, Self::Failure>;
    /// The bytes of the inputs the unit holds, built for its work; zero
    /// when it holds none.
    fn holding(&self) -> u64 {
        0
    }
    /// What building the inputs the unit holds took, priced: what giving
    /// them up is held to before the load has observed inputs given up.
    fn built(&self) -> Quote {
        Quote::default()
    }
    /// Give up the inputs the unit holds. Its next piece builds them again
    /// and is priced with that, in seconds and in bytes.
    fn release(&mut self, _context: &mut Self::Context) {}
    /// The bytes of the formed programs the unit holds for work to come;
    /// zero when it holds none.
    fn programs(&self) -> u64 {
        0
    }
    /// Give up the formed programs the unit holds. Its next piece forms
    /// what it runs again and is priced with that.
    fn give_up(&mut self, _context: &mut Self::Context) {}
    /// The most a single piece of the unit's work has added to the bytes
    /// held: what an unpriced footprint is held to.
    fn largest_footprint(&self) -> u64 {
        0
    }
    /// The bytes the load's tuning may still take under its ceiling; none
    /// when no ceiling is known, and nothing is then bounded.
    fn free(_context: &Self::Context) -> Option<u64>
    where
        Self: Sized,
    {
        None
    }
    /// Offer the unit's next piece with one challenger fewer; false when it
    /// has no fewer to offer. Asked when the piece's seconds do not fit
    /// what remains of the budget and nothing else can start: the last
    /// pieces of a load are sized to what is left, so no second of the
    /// budget stays unused while a smaller piece would fit it.
    fn narrow(&mut self, _context: &mut Self::Context) -> bool {
        false
    }
    /// The allocator stopped with the unit's next piece refused for its
    /// bytes ([`Refusal::Bytes`]): nothing in flight, what is held for
    /// nothing returned, and nothing another unit could give up for it.
    /// Told before what the unit owes is settled, so the record it then
    /// publishes states what stopped its search.
    fn refused_for_bytes(&mut self, _context: &mut Self::Context) {}
    /// Give back what the load holds for nothing: memory it has freed that
    /// still counts against its ceiling. Asked when a chosen piece's bytes
    /// do not fit, before any unit gives its inputs up for it.
    fn reclaim(_context: &mut Self::Context)
    where
        Self: Sized,
    {
    }
    /// What the unit owes when the allocator stops, priced: publishing its
    /// record, when its optional work changed the record since it was last
    /// published. `worked`: what it owes once one more piece of its work
    /// has run.
    fn owed(&self, _worked: bool) -> Quote {
        Quote::default()
    }
    /// Settle what the unit owes ([`Unit::owed`]): the allocator stopped.
    fn settle(&mut self, _context: &mut Self::Context) -> Result<(), Self::Failure> {
        Ok(())
    }
    /// The bytes free under the ceiling when the units' optional work holds
    /// nothing: the most any piece can ever be given. None without a
    /// ceiling.
    fn capacity(_context: &Self::Context) -> Option<u64>
    where
        Self: Sized,
    {
        None
    }
    /// The host bytes the load's tuning may still take, where host RAM is
    /// a memory domain apart from the one [`Unit::free`] counts (a device
    /// whose driver holds its allocations); none where it is not, or
    /// without a ceiling. The two domains do not substitute: a piece is
    /// started only if it fits both.
    fn host_free(_context: &Self::Context) -> HostRoom
    where
        Self: Sized,
    {
        HostRoom::Unbounded
    }
    /// What a piece of the unit's work is taken to add to that host domain:
    /// the most one has added. None before a piece of the unit has run:
    /// its price is then not known, and a piece without a price runs with
    /// no other such piece in flight.
    fn host_footprint(&self, _context: &Self::Context) -> Option<u64> {
        Some(0)
    }
    /// What a compile took that nothing served, once the load has run one
    /// ([`Workers`] compiles may be served by a cache the load cannot see):
    /// with the longest compile the load has run, what a compile is held
    /// to end within ([`allocate`]); nothing before it has run.
    fn cold_compile(_context: &Self::Context) -> Duration
    where
        Self: Sized,
    {
        Duration::ZERO
    }
    /// The host bytes one thread of a compile holds while it runs, as this
    /// load observed its compiles; nothing where the host is not a domain
    /// apart.
    fn compile_host(_context: &Self::Context) -> u64
    where
        Self: Sized,
    {
        0
    }
}

/// Where units' compiles run while the device does other work.
pub(crate) trait Workers<U: Unit> {
    /// The threads `forming` runs on at once.
    fn needs(&self, forming: &U::Forming) -> usize;
    /// The threads no compile in flight runs on.
    fn free(&self) -> usize;
    /// The host bytes one thread of a compile holds while it runs, as the
    /// compiles run here so far showed; nothing before one has.
    fn compile_host(&self) -> u64 {
        0
    }
    /// Start `unit`'s compile. It runs to its end.
    fn start(&mut self, unit: usize, forming: U::Forming);
    /// The compiles that have ended, without waiting: each unit, what its
    /// compile formed and what it took.
    fn ended(&mut self) -> Vec<(usize, U::Formed, Duration)>;
    /// Wait for the next compile to end, no longer than `within` when one
    /// is given: none when none ended in that time.
    fn wait(&mut self, within: Option<Duration>) -> Option<(usize, U::Formed, Duration)>;
}

/// A piece of work the allocator started, and what it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Admitted {
    pub unit: usize,
    pub stage: Stage,
    /// Its price when it was started.
    pub quote: Quote,
    /// When it was started.
    pub began: Duration,
    /// What its compile took on a worker, when it ran ahead.
    pub formed: Option<Duration>,
    /// When its work on the device began and what that took.
    pub performed: Duration,
    pub actual: Duration,
}

/// A piece of work admitted whose compile runs, or has ended and awaits
/// the device.
struct Flight {
    unit: usize,
    work: Work,
    began: Duration,
    /// When its compile is forecast to end.
    forming_ends: Duration,
    /// The forecast of its work on the device.
    device: Duration,
    /// What its compile took, once it ended.
    formed: Option<Duration>,
    /// The host bytes its compile is taken to hold while it runs
    /// ([`Unit::compile_host`], on each of its threads).
    compile_host: u64,
    /// It was admitted with bytes of a price not yet observed: until it
    /// has run, no other such piece starts.
    unpriced: bool,
}

/// A piece of work admitted whose measurement never ran: the allocator's
/// ending began before it could.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Abandoned {
    pub unit: usize,
    pub stage: Stage,
    /// Its price when it was started.
    pub quote: Quote,
    /// When it was started.
    pub began: Duration,
    /// What its compile took on a worker.
    pub formed: Duration,
}

/// What a piece of work that was not started did not fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// Its seconds did not fit what remained of the budget.
    Seconds,
    /// Its bytes did not fit under the ceiling, with every other unit's
    /// inputs given up.
    Bytes,
}

/// A piece of work not started because its price did not fit what remained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Refused {
    pub unit: usize,
    pub stage: Stage,
    pub quote: Quote,
    pub footprint: seismic::Footprint,
    pub refusal: Refusal,
}

/// What the allocator did with the budget: for the trace, whatever the
/// load's outcome.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Report {
    pub budget: Duration,
    /// Every piece of work started, in order.
    pub admitted: Vec<Admitted>,
    /// The work still offered when the allocator stopped: none of it fit.
    pub refused: Vec<Refused>,
    /// The pieces in flight when the ending began, never measured.
    pub abandoned: Vec<Abandoned>,
    /// What settling the units' owed publications took, those settled
    /// when their units ran out of work with those settled at the stop.
    pub settling: Duration,
    /// When the allocator stopped, everything owed settled.
    pub ended: Duration,
    /// The allocator's own ending.
    pub ending: Ending,
}

/// The allocator's own work from the end of the last piece it performed to
/// its stop, beside the publications it settles: what it took, and what
/// was held back for it when that piece was started.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Ending {
    /// When the last piece performed ended; when the allocator began, with
    /// none.
    pub began: Duration,
    /// What the start of that piece held back for the allocator's own
    /// ending: a pass, and each unit's inputs given up.
    pub held: Duration,
    /// The passes since, and what deciding in them took.
    pub passes: usize,
    pub deciding: Duration,
    /// The pieces narrowed since, and what narrowing them took.
    pub narrowed: usize,
    pub narrowing: Duration,
    /// The units that gave their inputs up since, and what that took.
    pub released: usize,
    pub releasing: Duration,
    /// Waiting, once everything owed was settled, for the compiles of
    /// abandoned pieces to end.
    pub draining: Duration,
}

/// The allocator's own operations by class, each priced by what it has
/// taken in this load ([`seismic::Observed`]).
#[derive(Default)]
struct Own {
    /// A pass: deciding what starts, apart from what the pass then runs.
    deciding: seismic::Observed,
    /// One unit narrowing its piece and pricing the narrower one.
    narrowing: seismic::Observed,
    /// One unit giving up its inputs.
    releasing: seismic::Observed,
}

impl Own {
    /// A class's price, once it was observed.
    fn observed(class: &seismic::Observed) -> Option<Duration> {
        class.price().map(Duration::from_secs_f64)
    }

    /// What a pass is held to: its class's price, and before the first
    /// pass has ended the longest operation the load has completed, as any
    /// operation not yet observed ([`seismic::Quote::fits`]).
    fn pass(&self, longest: Duration) -> Duration {
        Self::observed(&self.deciding).unwrap_or(longest)
    }

    /// What one unit's narrowing is held to: its class's price, and before
    /// a piece was narrowed a pass's. Pricing the narrower piece is what a
    /// pass does for the unit whose piece just ran, and a pass does that
    /// and more.
    fn narrow(&self, longest: Duration) -> Duration {
        Self::observed(&self.narrowing).unwrap_or_else(|| self.pass(longest))
    }

    /// What `unit` giving up its inputs is held to: the class's price, and
    /// before any unit gave its inputs up what building them took
    /// ([`Unit::built`]): it frees what that allocated and writes nothing,
    /// so building bounds it.
    fn release<U: Unit>(&self, unit: &U, longest: Duration) -> Duration {
        Self::observed(&self.releasing).unwrap_or_else(|| forecast(&unit.built(), longest))
    }
}

/// `unit` gives up its inputs: an operation of the allocator's own, timed
/// in its class. Returns what it took.
fn give_up_inputs<U: Unit>(
    units: &mut [U],
    unit: usize,
    context: &mut U::Context,
    clock: &impl Clock,
    own: &mut Own,
    ending: &mut Ending,
) -> Duration {
    let began = clock.now();
    units[unit].release(context);
    let took = clock.now().saturating_sub(began);
    own.releasing.observe(took.as_secs_f64());
    ending.released += 1;
    ending.releasing += took;
    took
}

impl Report {
    /// Budget left when the allocator stopped.
    pub fn unused(&self) -> Duration {
        self.budget.saturating_sub(self.ended)
    }

    /// Time past the budget when the allocator stopped: at most what the
    /// work in flight when the budget ended took.
    pub fn overrun(&self) -> Duration {
        self.ended.saturating_sub(self.budget)
    }
}

/// What work is taken to last: its price, with each unpriced operation at
/// the longest operation completed, as in its admission.
fn forecast(quote: &Quote, longest: Duration) -> Duration {
    u32::try_from(quote.unpriced)
        .ok()
        .and_then(|unpriced| longest.checked_mul(unpriced))
        .map_or(Duration::MAX, |unpriced| quote.priced.saturating_add(unpriced))
}

/// When the device, free at `now`, is done with `jobs`, each ready for it
/// at its first time and taking its second: it takes them in the order
/// they are ready, the first given among equals, each when it is ready and
/// the ones before it are done.
fn device_ends(now: Duration, mut jobs: Vec<(Duration, Duration)>) -> Duration {
    jobs.sort_by_key(|(ready, _)| *ready);
    jobs.iter().fold(now, |free, (ready, device)| {
        free.max(*ready).saturating_add(*device)
    })
}

/// Spend `total` on the units' work, starting only work whose price fits
/// what remains. `longest` is the longest single operation the load
/// completed before the allocator began. The report is complete whether or
/// not a unit's work fails.
pub(crate) fn allocate<U: Unit>(
    units: &mut [U],
    context: &mut U::Context,
    clock: &impl Clock,
    workers: &mut impl Workers<U>,
    total: Duration,
    mut longest: Duration,
) -> (Report, Result<(), U::Failure>) {
    let mut report = Report {
        budget: total,
        ..Report::default()
    };
    let mut started = vec![false; units.len()];
    let mut sealed = vec![false; units.len()];
    // Units whose next piece does not fit the bytes free with every other
    // unit's inputs given up: nothing changes that until a piece runs.
    let mut starved = vec![false; units.len()];
    // Units whose next piece's compile needs more threads than are free:
    // nothing changes that until a compile ends.
    let mut crowded = vec![false; units.len()];
    // Units whose next piece holds bytes of a price not yet observed while
    // another such piece is in flight: nothing changes that until a piece
    // runs or a compile ends.
    let mut alone = vec![false; units.len()];
    // In the order admitted.
    let mut flights = Vec::<Flight>::new();
    // The longest compile that has ended.
    let mut compiled = Duration::ZERO;
    // The allocator's own operations, priced as they are observed.
    let mut own = Own::default();
    // Units settled since their last piece of work.
    let mut published = vec![false; units.len()];
    // When the pass before this one began, and what of the time since was
    // not its deciding: work performed, waits, and operations of the
    // allocator's other classes.
    let mut opened = None::<Duration>;
    let mut apart = Duration::ZERO;
    report.ending.began = clock.now();
    let outcome = loop {
        let opening = clock.now();
        if let Some(opened) = opened {
            let deciding = opening.saturating_sub(opened).saturating_sub(apart);
            own.deciding.observe(deciding.as_secs_f64());
            report.ending.passes += 1;
            report.ending.deciding += deciding;
        }
        opened = Some(opening);
        apart = Duration::ZERO;
        let ended = workers.ended();
        if !ended.is_empty() {
            crowded.fill(false);
            alone.fill(false);
        }
        let landed = ended.into_iter().try_for_each(|ended| {
            compiled = compiled.max(ended.2);
            land(units, context, &mut flights, ended)
        });
        if let Err(failure) = landed {
            break Err(failure);
        }
        U::refresh(units, context);
        let flying = |unit: usize| flights.iter().any(|flight| flight.unit == unit);
        let offers = units
            .iter()
            .enumerate()
            .filter(|(unit, _)| !sealed[*unit] && !flying(*unit))
            .filter_map(|(unit, offered)| offered.next().map(|work| (unit, work)))
            .collect::<Vec<_>>();
        // A unit that will be given no more work, sealed or with none left,
        // has its final choice: what it owes is settled now. Held to the
        // stop it would be held back for by forecast, and its publication
        // would be no evidence of what one takes.
        let mut failed = None;
        for unit in 0..units.len() {
            if published[unit]
                || flying(unit)
                || offers.iter().any(|(offering, _)| *offering == unit)
            {
                continue;
            }
            published[unit] = true;
            if units[unit].owed(false) == Quote::default() {
                continue;
            }
            let settling = clock.now();
            let outcome = units[unit].settle(context);
            let took = clock.now().saturating_sub(settling);
            report.settling += took;
            apart += took;
            if let Err(failure) = outcome {
                failed = Some(failure);
                break;
            }
        }
        if let Some(failure) = failed {
            break Err(failure);
        }
        let now = clock.now();
        let remaining = total.saturating_sub(now);
        // The allocator's own ending once `working` has run one more
        // piece: the pass that finds nothing more to start, and every unit
        // that then holds inputs giving them up, `working` among them.
        // A unit that holds none now is held to the class's price alone:
        // what its piece builds is in the piece's own price.
        let releases = units
            .iter()
            .filter(|unit| unit.holding() > 0)
            .fold(Duration::ZERO, |sum, unit| {
                sum.saturating_add(own.release(unit, longest))
            });
        let ending = (0..units.len())
            .map(|working| {
                let joining = if units[working].holding() == 0 {
                    Own::observed(&own.releasing).unwrap_or_default()
                } else {
                    Duration::ZERO
                };
                own.pass(longest)
                    .saturating_add(releases)
                    .saturating_add(joining)
            })
            .collect::<Vec<_>>();
        // What is owed when the allocator stops, once `working` has run
        // one more piece: its own ending, and what the units owe, where
        // the units with work in flight will have worked too.
        let owed = (0..units.len())
            .map(|working| {
                units
                    .iter()
                    .enumerate()
                    .fold(ending[working], |sum, (unit, offered)| {
                        sum.saturating_add(forecast(
                            &offered.owed(unit == working || flying(unit)),
                            longest,
                        ))
                    })
            })
            .collect::<Vec<_>>();
        // The ending begins at the earliest cutoff of the pieces in flight
        // ([`Unit::perform`]): from then what they would measure and what
        // is owed after it no longer fit, whatever their compiles do.
        let concluding = flights
            .iter()
            .map(|flight| total.saturating_sub(owed[flight.unit]))
            .min();
        // A piece that needs more bytes than the load's optional work may
        // ever hold, its own unit's inputs with what it adds to them, is
        // refused for the load: nothing another unit gives up changes it.
        let capacity = U::capacity(context);
        let barred = (0..units.len())
            .map(|unit| {
                offers.iter().any(|(offering, work)| {
                    *offering == unit
                        && capacity.is_some_and(|capacity| {
                            units[unit].holding().saturating_add(
                                work.footprint.forecast(units[unit].largest_footprint()),
                            ) > capacity
                        })
                })
            })
            .collect::<Vec<_>>();
        // A unit with no work that can still start holds no inputs while
        // other work can: what remains only shrinks, so work that does not
        // fit it alone never will, and work refused for the load for its
        // bytes never starts. When no unit's work can start and none is in
        // flight, the allocator is at its stop: the pieces offered are
        // narrowed to what is left with their inputs as they are, and what
        // is held leaves when everything owed is settled.
        let startable = (0..units.len())
            .map(|unit| {
                !barred[unit]
                    && offers.iter().any(|(offering, work)| {
                        *offering == unit
                            && work
                                .quote
                                .fits(remaining.saturating_sub(owed[unit]), longest)
                    })
            })
            .collect::<Vec<_>>();
        if !flights.is_empty() || startable.iter().any(|startable| *startable) {
            for unit in 0..units.len() {
                if !flying(unit) && !startable[unit] && units[unit].holding() > 0 {
                    apart += give_up_inputs(
                        units,
                        unit,
                        context,
                        clock,
                        &mut own,
                        &mut report.ending,
                    );
                }
            }
        }
        // An operation whose bytes a unit has not observed is held to the
        // most a piece of that unit has added: what another unit builds
        // says nothing of it.
        //
        // This is where compiles are bounded by memory: a piece's bytes
        // count from when its compile starts, and its footprint is priced
        // from what its unit's pieces were observed to add to everything
        // the load holds, the process's host bytes with the device's
        // charges, so what a compile holds and leaves held is in it.
        let flying_bytes = flights.iter().fold(0u64, |sum, flight| {
            sum.saturating_add(
                flight
                    .work
                    .footprint
                    .forecast(units[flight.unit].largest_footprint()),
            )
        });
        // A dependant's discovery reads nothing of its prerequisites. Its
        // other work reads their choices, once their defaults are measured,
        // no lead of theirs awaits conversion and no work of theirs is in
        // flight.
        let settled = |unit: usize| {
            units[unit].prerequisites().iter().all(|prerequisite| {
                sealed[*prerequisite]
                    || !(flying(*prerequisite)
                        || offers.iter().any(|(offering, work)| {
                            offering == prerequisite
                                && matches!(work.stage, Stage::Discovery | Stage::Promote)
                        }))
            })
        };
        // The device's queue: each piece in flight, when it is ready for
        // the device (now, once its compile has ended) and the device time
        // admitted with it.
        let queue = flights
            .iter()
            .map(|flight| {
                let ready = match flight.formed {
                    Some(_) => now,
                    None => flight.forming_ends.max(now),
                };
                (ready, flight.device)
            })
            .collect::<Vec<_>>();
        let ahead = |work: &Work| work.forming != Quote::default();
        // When a new piece would be ready for the device were it started
        // now, and the device time it takes.
        let job = |work: &Work| {
            let whole = forecast(&work.quote, longest);
            if ahead(work) {
                let compile = forecast(&work.forming, longest);
                (now.saturating_add(compile), whole.saturating_sub(compile))
            } else {
                (now, whole)
            }
        };
        // A compile cannot be interrupted, and abandoned at the ending it is
        // waited for. So beside its forecast it is held to end within the
        // budget at the most a compile can take as far as the load has
        // seen: the longest that has ended, and one that nothing served.
        let cold = compiled.max(U::cold_compile(context));
        // The one admission rule over the whole queue: everything admitted,
        // the new piece in its place among it, and what is owed after end
        // within the budget; and a piece's compile, at the most a compile
        // has taken, ends within it.
        let fits = |unit: usize, work: &Work| {
            let mut jobs = queue.clone();
            jobs.push(job(work));
            device_ends(now, jobs).saturating_add(owed[unit]) <= total
                && (!ahead(work)
                    || now.saturating_add(forecast(&work.forming, longest).max(cold)) <= total)
        };
        let (fitting, refused): (Vec<_>, Vec<_>) = offers
            .iter()
            .copied()
            .filter(|(unit, work)| work.stage == Stage::Discovery || settled(*unit))
            .partition(|(unit, work)| !starved[*unit] && !barred[*unit] && fits(*unit, work));
        // At the ending nothing more is admitted or measured: what is in
        // flight is abandoned, and every offer still standing is refused.
        if concluding.is_some_and(|concluding| now >= concluding) {
            let standing = fitting.into_iter().chain(refused).collect();
            report.refused = refuse(units, context, standing, |unit| starved[unit] || barred[unit]);
            break Ok(());
        }
        // When a piece's compile is due to start: so that the piece is
        // ready when the device is done with the pieces ready before it.
        let due = |work: &Work| {
            let (ready, _) = job(work);
            let before = queue
                .iter()
                .copied()
                .filter(|(queued, _)| *queued <= ready)
                .collect();
            device_ends(now, before).saturating_sub(ready.saturating_sub(now))
        };
        // A compile not yet due waits, as does one whose threads are taken:
        // the device meanwhile takes other work.
        let earliest = fitting
            .iter()
            .filter(|(unit, work)| ahead(work) && !crowded[*unit])
            .map(|(_, work)| due(work))
            .filter(|due| *due > now)
            .min();
        let fitting = fitting
            .into_iter()
            .filter(|(unit, work)| !ahead(work) || (!crowded[*unit] && due(work) <= now))
            .filter(|(unit, _)| !alone[*unit])
            .collect::<Vec<_>>();
        let value = |(unit, work): &(usize, Work)| {
            units[*unit].share() * work.gain
                / forecast(&work.quote, longest)
                    .as_secs_f64()
                    .max(f64::MIN_POSITIVE)
        };
        let most_valuable = |stages: &[Stage]| {
            fitting
                .iter()
                .filter(|(_, work)| stages.contains(&work.stage))
                // The first of equals: preparation order.
                .fold(None, |best: Option<&(usize, Work)>, offer| match best {
                    Some(best) if value(best) >= value(offer) => Some(best),
                    _ => Some(offer),
                })
        };
        let discovery = fitting
            .iter()
            .filter(|(_, work)| work.stage == Stage::Discovery)
            .fold(None, |best: Option<&(usize, Work)>, offer| match best {
                Some(best)
                    if (!started[best.0], units[best.0].share())
                        >= (!started[offer.0], units[offer.0].share()) =>
                {
                    Some(best)
                }
                _ => Some(offer),
            });
        let choice = most_valuable(&[Stage::Promote])
            .or(discovery)
            .or_else(|| most_valuable(&[Stage::Form]))
            .or_else(|| most_valuable(&[Stage::Measure, Stage::Confirm]))
            .copied();
        // The chosen piece starts only if its bytes fit what is free. When
        // they do not, the unit whose own next work is worth least gives
        // its inputs up; with none left to give, the piece is not started.
        // Bytes of a price not yet observed: in the device's domain a
        // footprint with an unpriced part whose unit has run no piece, in
        // the host's a unit that has run no piece, or a compile before any
        // compile's bytes were seen. Nothing says what such a piece adds,
        // so no room can be divided by it: it runs with no other such
        // piece in flight, on the room as measured, and what it added is
        // then its kind's price.
        let unpriced = choice.is_some_and(|(unit, work)| {
            let device = U::free(context).is_some()
                && work.footprint.unpriced > 0
                && units[unit].largest_footprint() == 0;
            let host = U::host_free(context) != HostRoom::Unbounded
                && (units[unit].host_footprint(context).is_none()
                    || (ahead(&work)
                        && U::compile_host(context).max(workers.compile_host()) == 0));
            device || host
        });
        if let Some((unit, _)) = choice {
            if unpriced && flights.iter().any(|flight| flight.unpriced) {
                alone[unit] = true;
                continue;
            }
        }
        if let Some((unit, work)) = choice {
            let largest = units[unit].largest_footprint();
            let free = |context: &U::Context| {
                U::free(context).map(|free| free.saturating_sub(flying_bytes))
            };
            // Where host RAM is a domain of its own, the piece's host
            // bytes fit what is free of it beside the pieces in flight:
            // what is held now, as measured, with the whole price of every
            // piece in flight, which is no less than what each has yet to
            // add.
            let host = units[unit].host_footprint(context).unwrap_or(0);
            let flying_host = flights.iter().fold(0u64, |sum, flight| {
                // A compile that has ended holds nothing more.
                sum.saturating_add(units[flight.unit].host_footprint(context).unwrap_or(0))
                    .saturating_add(if flight.formed.is_none() {
                        flight.compile_host
                    } else {
                        0
                    })
            });
            let host_fits = |context: &U::Context| U::host_free(context).admits(host, flying_host);
            // Before any unit gives up inputs it would build again, the
            // load gives back what it holds for nothing.
            if !work.footprint.fits(free(context), largest) || !host_fits(context) {
                U::reclaim(context);
            }
            let free = free(context);
            if !work.footprint.fits(free, largest) || !host_fits(context) {
                let worth = |holder: usize| {
                    offers
                        .iter()
                        .find(|(offering, _)| *offering == holder)
                        .map_or(f64::NEG_INFINITY, value)
                };
                // What a unit not in flight holds for work to come: the
                // programs it formed and its inputs.
                let holders = (0..units.len())
                    .filter(|holder| {
                        *holder != unit
                            && !flying(*holder)
                            && (units[*holder].programs() > 0 || units[*holder].holding() > 0)
                    })
                    .collect::<Vec<_>>();
                // Nothing is given up for a piece that would not fit even
                // with everything given up.
                let given_up = holders.iter().fold(0u64, |sum, holder| {
                    sum.saturating_add(units[*holder].programs())
                        .saturating_add(units[*holder].holding())
                });
                let victim = holders
                    .into_iter()
                    .filter(|_| {
                        work.footprint
                            .fits(free.map(|free| free.saturating_add(given_up)), largest)
                    })
                    .min_by(|left, right| worth(*left).total_cmp(&worth(*right)));
                // The unit whose own next work is worth least gives up
                // first: its programs, and when it holds none, its inputs.
                match victim {
                    Some(victim) if units[victim].programs() > 0 => units[victim].give_up(context),
                    Some(victim) => {
                        apart += give_up_inputs(
                            units,
                            victim,
                            context,
                            clock,
                            &mut own,
                            &mut report.ending,
                        );
                    }
                    None => starved[unit] = true,
                }
                continue;
            }
        }
        // A piece started seals what its unit reads.
        let mut start = |units: &[U], unit: usize, work: &Work| {
            if work.stage != Stage::Discovery {
                for prerequisite in units[unit].prerequisites() {
                    sealed[prerequisite] = true;
                }
            }
            started[unit] = true;
        };
        // The chosen piece's compile starts; the device then takes the
        // admitted work that is ready for it before anything new.
        let forming = choice
            .filter(|(_, work)| ahead(work))
            .and_then(|(unit, work)| Some((unit, work, units[unit].forming()?)));
        let ready = flights.iter().position(|flight| flight.formed.is_some());
        // A compile whose threads are taken is not started on them: the
        // unit waits for a compile to end, and the next choice is made
        // without it.
        if let Some((unit, _, forming)) = &forming {
            if workers.needs(forming) > workers.free() {
                crowded[*unit] = true;
                continue;
            }
        }
        // A compile holds host memory of its own on every thread it runs
        // on, while it runs. Where host RAM is a domain of its own, a
        // compile starts only if that fits beside the compiles in flight:
        // this, not the threads the host has, is then what bounds how many
        // run at once. One that does not fit waits for a compile to end;
        // with none in flight, the load first returns what it holds for
        // nothing, and the piece is refused if that is not enough.
        // A thread's compile is priced at the most one was seen to hold:
        // by the load before its optional work, or by a burst of it.
        let compile_host = forming.as_ref().map_or(0, |(_, _, forming)| {
            U::compile_host(context)
                .max(workers.compile_host())
                .saturating_mul(workers.needs(forming) as u64)
        });
        if let Some((unit, _, _)) = &forming {
            let piece = units[*unit]
                .host_footprint(context)
                .unwrap_or(0)
                .saturating_add(compile_host);
            let flying = flights.iter().fold(0u64, |sum, flight| {
                sum.saturating_add(units[flight.unit].host_footprint(context).unwrap_or(0))
                    .saturating_add(if flight.formed.is_none() {
                        flight.compile_host
                    } else {
                        0
                    })
            });
            if !U::host_free(context).admits(piece, flying) {
                if !flights.is_empty() {
                    crowded[*unit] = true;
                    continue;
                }
                U::reclaim(context);
                if !U::host_free(context).admits(piece, flying) {
                    starved[*unit] = true;
                    continue;
                }
            }
        }
        let (unit, work, began, formed) = match (forming, ready, choice) {
            (Some((unit, work, forming)), _, _) => {
                start(units, unit, &work);
                workers.start(unit, forming);
                let whole = forecast(&work.quote, longest);
                let compile = forecast(&work.forming, longest);
                flights.push(Flight {
                    unit,
                    work,
                    began: now,
                    forming_ends: now.saturating_add(compile),
                    device: whole.saturating_sub(compile),
                    formed: None,
                    compile_host,
                    unpriced,
                });
                continue;
            }
            (None, Some(ready), _) => {
                let flight = flights.remove(ready);
                (flight.unit, flight.work, flight.began, flight.formed)
            }
            (None, None, Some((unit, work))) => {
                start(units, unit, &work);
                (unit, work, now, None)
            }
            (None, None, None) if flights.is_empty() => {
                // Nothing starts as offered. A piece refused for its
                // seconds is offered again with a challenger fewer, until
                // it fits what is left or has one challenger: the piece
                // worth the most first, and each only while its narrowing
                // and everything owed after it fit what is left.
                let mut narrowing = refused
                    .iter()
                    .filter(|(unit, _)| !starved[*unit] && !barred[*unit])
                    .map(|offer| (offer.0, value(offer)))
                    .collect::<Vec<_>>();
                narrowing.sort_by(|left, right| right.1.total_cmp(&left.1));
                let mut narrowed = false;
                for (unit, _) in narrowing {
                    let began = clock.now();
                    if began
                        .saturating_add(own.narrow(longest))
                        .saturating_add(owed[unit])
                        > total
                    {
                        continue;
                    }
                    if !units[unit].narrow(context) {
                        continue;
                    }
                    // The narrower piece is priced here, in its class.
                    let _ = units[unit].next();
                    let took = clock.now().saturating_sub(began);
                    own.narrowing.observe(took.as_secs_f64());
                    report.ending.narrowed += 1;
                    report.ending.narrowing += took;
                    apart += took;
                    narrowed = true;
                }
                if narrowed {
                    continue;
                }
                report.refused = refuse(units, context, refused, |unit| {
                    starved[unit] || barred[unit]
                });
                break Ok(());
            }
            // Work is in flight and nothing can start now: wait for a
            // compile to end, for the first compile that waits to be due,
            // or for the ending.
            (None, None, None) => {
                let waiting = clock.now();
                let until = earliest.into_iter().chain(concluding).min();
                let ended = workers.wait(until.map(|until| until.saturating_sub(now)));
                apart += clock.now().saturating_sub(waiting);
                let Some(ended) = ended else {
                    continue;
                };
                crowded.fill(false);
                alone.fill(false);
                compiled = compiled.max(ended.2);
                match land(units, context, &mut flights, ended) {
                    Ok(()) => continue,
                    Err(failure) => break Err(failure),
                }
            }
        };
        let held = ending[unit];
        let performing = clock.now();
        // Preserve the original cutoff and terminal obligations even when
        // admitted work waits behind an underestimated piece.
        let performed = units[unit].perform(context, total.saturating_sub(owed[unit]));
        starved.fill(false);
        alone.fill(false);
        published[unit] = false;
        let ended = clock.now();
        apart += ended.saturating_sub(performing);
        report.admitted.push(Admitted {
            unit,
            stage: work.stage,
            quote: work.quote,
            began,
            formed,
            performed: now,
            actual: ended.saturating_sub(now),
        });
        // The allocator's ending is what follows the last piece performed.
        report.ending = Ending {
            began: ended,
            held,
            ..Ending::default()
        };
        match performed {
            Ok(observed) => longest = longest.max(observed),
            Err(failure) => break Err(failure),
        }
    };
    if let Some(opened) = opened {
        report.ending.passes += 1;
        report.ending.deciding += clock.now().saturating_sub(opened).saturating_sub(apart);
    }
    // Every unit's record is published as it stands, whatever became of
    // the work, while the compiles of the pieces abandoned in flight run.
    let settling = clock.now();
    let settled = units
        .iter_mut()
        .try_for_each(|unit| unit.settle(context));
    report.settling += clock.now().saturating_sub(settling);
    // Everything owed is settled: what the units still hold of their
    // inputs leaves with the optional work.
    for unit in 0..units.len() {
        if units[unit].holding() > 0 {
            give_up_inputs(units, unit, context, clock, &mut own, &mut report.ending);
        }
    }
    // A compile runs to its end: those still running are waited for, and
    // what they formed is dropped.
    let draining = clock.now();
    while flights.iter().any(|flight| flight.formed.is_none()) {
        let (unit, _, took) = workers
            .wait(None)
            .expect("a compile in flight ends");
        flights
            .iter_mut()
            .find(|flight| flight.unit == unit)
            .expect("a compile that ended was started")
            .formed = Some(took);
    }
    report.ending.draining = clock.now().saturating_sub(draining);
    report.abandoned = flights
        .into_iter()
        .map(|flight| Abandoned {
            unit: flight.unit,
            stage: flight.work.stage,
            quote: flight.work.quote,
            began: flight.began,
            formed: flight.formed.expect("every compile in flight has ended"),
        })
        .collect();
    report.ended = clock.now();
    (report, outcome.and(settled))
}

/// Nothing more starts: every offer still `standing` is refused for good,
/// for its bytes where `bytes` says so and otherwise for its seconds. A
/// unit refused for its bytes is told before what it owes is settled: its
/// record states what stopped it.
fn refuse<U: Unit>(
    units: &mut [U],
    context: &mut U::Context,
    standing: Vec<(usize, Work)>,
    bytes: impl Fn(usize) -> bool,
) -> Vec<Refused> {
    let refused = standing
        .into_iter()
        .map(|(unit, work)| Refused {
            unit,
            stage: work.stage,
            quote: work.quote,
            footprint: work.footprint,
            refusal: if bytes(unit) {
                Refusal::Bytes
            } else {
                Refusal::Seconds
            },
        })
        .collect::<Vec<_>>();
    for refused in &refused {
        if refused.refusal == Refusal::Bytes {
            units[refused.unit].refused_for_bytes(context);
        }
    }
    refused
}

/// `unit`'s compile ended: the unit holds what it formed, and its piece
/// awaits the device.
fn land<U: Unit>(
    units: &mut [U],
    context: &mut U::Context,
    flights: &mut [Flight],
    (unit, formed, took): (usize, U::Formed, Duration),
) -> Result<(), U::Failure> {
    // The compile ended whether or not its unit can hold what it formed.
    flights
        .iter_mut()
        .find(|flight| flight.unit == unit)
        .expect("a compile that ended was started")
        .formed = Some(took);
    units[unit].formed(formed, took, context)
}

/// The load's worker threads ([`seismic::Workers`]): each compile runs on
/// them to its end, with the threads it forms on at once counted against
/// the set's width. No compile has a thread of its own: the set's threads
/// are reused from the first compile of the load to the last.
pub(crate) struct Threads<U: Unit> {
    workers: std::sync::Arc<seismic::Workers>,
    run: fn(U::Forming) -> U::Formed,
    needs: fn(&U::Forming) -> usize,
    /// The threads of each compile in flight, by its unit.
    taken: Vec<(usize, usize)>,
    sender: std::sync::mpsc::Sender<(usize, std::thread::Result<U::Formed>, Duration)>,
    receiver: std::sync::mpsc::Receiver<(usize, std::thread::Result<U::Formed>, Duration)>,
    /// The burst of compiles in flight, from the first started with none
    /// running to the last to end.
    burst: Option<Burst>,
    /// The most a burst raised the peak of the process's footprint, for
    /// each thread it compiled on at once ([`Workers::compile_host`]), and
    /// no less than what the load had seen a thread's compile hold before
    /// these workers began.
    compile_host: u64,
    /// Each burst that showed a thread's compile to hold more than it was
    /// priced at: the price it was admitted on and what it held.
    mispriced: Vec<(u64, u64)>,
}

/// A burst of compiles: the process's footprint when it began, the peak of
/// the footprint until then ([`seismic::ProcessMemory::footprint_peak`]),
/// and the most threads the burst compiled on at once.
struct Burst {
    footprint: u64,
    peak: u64,
    threads: usize,
}

impl<U: Unit> Threads<U> {
    /// Workers that `run` each compile, which forms on `needs` threads at
    /// once, on the threads of `workers`. `compile_host`: what the load
    /// has seen one thread's compile hold so far, nothing if it has seen
    /// none.
    pub fn new(
        workers: std::sync::Arc<seismic::Workers>,
        run: fn(U::Forming) -> U::Formed,
        needs: fn(&U::Forming) -> usize,
        compile_host: u64,
    ) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();
        Self {
            workers,
            run,
            needs,
            taken: Vec::new(),
            sender,
            receiver,
            burst: None,
            compile_host,
            mispriced: Vec::new(),
        }
    }

    /// The bursts that held more for each thread than a compile was priced
    /// at when they ran: that price, and what a thread held.
    pub fn mispriced(&self) -> &[(u64, u64)] {
        &self.mispriced
    }

    /// A compile ended: its threads are free. A compile that panicked
    /// panics the load here, where its end is read.
    fn joined(
        &mut self,
        (unit, formed, took): (usize, std::thread::Result<U::Formed>, Duration),
    ) -> (usize, U::Formed, Duration) {
        self.taken.retain(|(compiling, _)| *compiling != unit);
        // The burst is over when its last compile ends. If the peak of the
        // process's footprint rose in it, the peak was reached while its
        // compiles ran, and what it rose by over the burst's start is what
        // that many threads of compiles held at their most (with whatever
        // else the load took meanwhile, which only prices a compile
        // higher). A burst below the peak already kept says nothing: there
        // is one mark for the life of the process. A burst that held more
        // than its compiles were priced at raises the price to what it
        // held, and is kept as a mispricing.
        if self.taken.is_empty() {
            if let Some(burst) = self.burst.take() {
                let peak = seismic::ProcessMemory::footprint_peak();
                if let Some(peak) = peak.filter(|peak| *peak > burst.peak && burst.threads > 0) {
                    let held = peak.saturating_sub(burst.footprint) / burst.threads as u64;
                    if held > self.compile_host {
                        if self.compile_host > 0 {
                            self.mispriced.push((self.compile_host, held));
                        }
                        self.compile_host = held;
                    }
                }
            }
        }
        match formed {
            Ok(formed) => (unit, formed, took),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}

impl<U: Unit> Workers<U> for Threads<U>
where
    U::Forming: Send + 'static,
    U::Formed: Send + 'static,
{
    fn needs(&self, forming: &U::Forming) -> usize {
        (self.needs)(forming)
    }

    fn free(&self) -> usize {
        self.workers
            .width()
            .saturating_sub(self.taken.iter().map(|(_, threads)| threads).sum())
    }

    fn compile_host(&self) -> u64 {
        self.compile_host
    }

    fn start(&mut self, unit: usize, forming: U::Forming) {
        if self.taken.is_empty() {
            self.burst = seismic::ProcessMemory::footprint_now()
                .zip(seismic::ProcessMemory::footprint_peak())
                .map(|(footprint, peak)| Burst {
                    footprint,
                    peak,
                    threads: 0,
                });
        }
        self.taken.push((unit, (self.needs)(&forming)));
        if let Some(burst) = &mut self.burst {
            burst.threads = burst
                .threads
                .max(self.taken.iter().map(|(_, threads)| threads).sum());
        }
        let (run, sender) = (self.run, self.sender.clone());
        self.workers.spawn(move || {
            let began = std::time::Instant::now();
            let formed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || run(forming)));
            // The allocator reads every compile it started before it ends.
            let _ = sender.send((unit, formed, began.elapsed()));
        });
    }

    fn ended(&mut self) -> Vec<(usize, U::Formed, Duration)> {
        let ended = self.receiver.try_iter().collect::<Vec<_>>();
        ended.into_iter().map(|ended| self.joined(ended)).collect()
    }

    fn wait(&mut self, within: Option<Duration>) -> Option<(usize, U::Formed, Duration)> {
        use std::sync::mpsc::RecvTimeoutError;
        let ended = match within {
            Some(within) => match self.receiver.recv_timeout(within) {
                Ok(ended) => ended,
                Err(RecvTimeoutError::Timeout) => return None,
                Err(RecvTimeoutError::Disconnected) => {
                    unreachable!("the allocator holds a sender of its own")
                }
            },
            None => self
                .receiver
                .recv()
                .expect("the allocator holds a sender of its own"),
        };
        Some(self.joined(ended))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    #[derive(Clone, Default)]
    struct Fake(Rc<Cell<Duration>>);

    impl Fake {
        fn pass(&self, time: Duration) {
            self.0.set(self.0.get() + time);
        }
    }

    impl Clock for Fake {
        fn now(&self) -> Duration {
            self.0.get()
        }
    }

    const SECOND: Duration = Duration::from_secs(1);

    type Log = RefCell<Vec<(usize, Stage, Duration)>>;

    /// A unit that discovers `points` points, then measures `challengers`,
    /// each the cost it will measure relative to the defaults, converting
    /// each lead before it measures on. Every piece takes `slow` times its
    /// quote. The unit serves `served` at every instant: its defaults at
    /// cost 1 until a challenger is promoted.
    struct Synthetic {
        name: usize,
        clock: Fake,
        share: f64,
        points: usize,
        discovery: Duration,
        measurement: Duration,
        /// What each challenger's compile takes, ahead of its measurement,
        /// and how many times that its compile takes besides `slow`.
        compile: Duration,
        compile_slow: f64,
        /// The next challenger's compile has run.
        compiled: bool,
        promotion: Duration,
        /// Discovery is of classes never observed: it has no price.
        unpriced_discovery: bool,
        slow: f64,
        interruptible: bool,
        stopped: bool,
        challengers: Vec<f64>,
        /// A lead whose coverage check fails is not promoted.
        covers: bool,
        prerequisites: Vec<usize>,
        discovered: usize,
        /// Every challenger measured, in order: its record.
        measured: Vec<f64>,
        lead: Option<f64>,
        served: f64,
        /// The longest piece completed.
        longest: Duration,
        /// The bytes the unit's inputs hold while they are resident, and
        /// whether they are; every piece of the unit needs them.
        inputs: u64,
        resident: bool,
        /// How often the unit's inputs were built.
        built: usize,
        /// After discovery, the unit's pieces hold bytes of a class it has
        /// not observed: one unpriced operation each.
        unobserved: bool,
        /// How many of the unit's first challengers are other forms'
        /// representatives; the rest refine a form.
        forms: usize,
        /// Bytes the unit's discovery leaves held that are not inputs:
        /// giving its inputs up does not give them back.
        kept: u64,
        /// What publishing the unit's record takes, whether a piece changed
        /// the record since it was published, and how often it was.
        publication: Duration,
        changed: bool,
        published: usize,
        /// What publishing a promoted choice's serving artifact takes, and
        /// whether the served choice was promoted since its artifact was
        /// published: owed by a unit that promoted, and after a piece only
        /// if that piece is a conversion.
        artifact: Duration,
        unformed: bool,
        /// How many challengers a piece measures after its one compile, and
        /// how many the next piece is offered with ([`Unit::narrow`]).
        batch: usize,
        size: usize,
        /// Bytes each challenger's compile leaves in the process for good:
        /// nothing the load does gives them back.
        residue: u64,
        /// Bytes each challenger's compile holds and frees, which count
        /// until the load returns its freed memory ([`Unit::reclaim`]).
        garbage: u64,
        /// Bytes each challenger's formed program holds from its piece on,
        /// until the unit gives its programs up ([`Unit::give_up`]); what
        /// the unit's programs hold now, and how often it gave them up.
        program: u64,
        programs_held: u64,
        given_up: usize,
        /// Host bytes the unit's inputs hold beside their bytes on the
        /// device, where host RAM is a domain of its own.
        host: u64,
    }

    /// The bytes the synthetic units hold, the most they held, the ceiling
    /// on them (none unless a test sets one), and how many of the bytes
    /// held are freed and not yet returned.
    #[derive(Clone, Copy, Default)]
    struct Memory {
        held: u64,
        peak: u64,
        ceiling: Option<u64>,
        freed: u64,
        /// Where host RAM is a domain of its own: the host bytes the units'
        /// inputs hold, the most they held, and the ceiling on them.
        host: u64,
        host_peak: u64,
        host_ceiling: Option<u64>,
    }

    thread_local! {
        /// What a compile that nothing served took ([`Unit::cold_compile`]):
        /// nothing unless a test sets it.
        static COLD: Cell<Duration> = const { Cell::new(Duration::ZERO) };
        static MEMORY: Cell<Memory> = const {
            Cell::new(Memory {
                held: 0,
                peak: 0,
                ceiling: None,
                freed: 0,
                host: 0,
                host_peak: 0,
                host_ceiling: None,
            })
        };
    }

    impl Synthetic {
        /// The compile the next challenger still needs.
        fn uncompiled(&self) -> Duration {
            if self.compiled {
                Duration::ZERO
            } else {
                self.compile
            }
        }

        fn new(name: usize, clock: &Fake, share: f64, challengers: &[f64]) -> Self {
            Self {
                compile: Duration::ZERO,
                compile_slow: 1.,
                compiled: false,
                name,
                clock: clock.clone(),
                share,
                points: 1,
                discovery: SECOND,
                measurement: SECOND,
                promotion: SECOND * 2,
                unpriced_discovery: false,
                slow: 1.,
                interruptible: false,
                stopped: false,
                challengers: challengers.to_vec(),
                covers: true,
                prerequisites: Vec::new(),
                discovered: 0,
                measured: Vec::new(),
                lead: None,
                served: 1.,
                longest: Duration::ZERO,
                inputs: 0,
                resident: false,
                built: 0,
                unobserved: false,
                forms: 0,
                kept: 0,
                publication: Duration::ZERO,
                changed: false,
                published: 0,
                artifact: Duration::ZERO,
                unformed: false,
                batch: 1,
                size: 1,
                residue: 0,
                garbage: 0,
                program: 0,
                programs_held: 0,
                given_up: 0,
                host: 0,
            }
        }

        /// The challengers the next piece measures.
        fn challenging(&self) -> usize {
            self.size.min(self.challengers.len() - self.measured.len())
        }

        /// What measuring the next challenger adds to the bytes held: the
        /// unit's inputs when they are not resident, and what its compile
        /// holds and leaves held.
        fn measuring(&self) -> seismic::Footprint {
            let mut footprint = self.footprint();
            footprint.priced +=
                self.residue + self.garbage + self.program * self.challenging() as u64;
            footprint
        }

        /// What the next piece adds to the bytes held: the unit's inputs,
        /// when they are not resident.
        fn footprint(&self) -> seismic::Footprint {
            seismic::Footprint {
                priced: if self.resident { 0 } else { self.inputs },
                unpriced: usize::from(self.unobserved && self.discovered >= self.points),
            }
        }
    }

    fn priced(price: Duration) -> Quote {
        Quote {
            priced: price,
            unpriced: 0,
        }
    }

    impl Unit for Synthetic {
        type Context = Log;
        type Failure = String;
        type Forming = Duration;
        type Formed = ();

        fn prerequisites(&self) -> Vec<usize> {
            self.prerequisites.clone()
        }

        fn share(&self) -> f64 {
            self.share
        }

        fn next(&self) -> Option<Work> {
            if self.stopped {
                return None;
            }
            if self.discovered < self.points {
                return Some(Work {
                    stage: Stage::Discovery,
                    quote: if self.unpriced_discovery {
                        Quote {
                            priced: Duration::ZERO,
                            unpriced: 1,
                        }
                    } else {
                        priced(self.discovery)
                    },
                    forming: Quote::default(),
                    gain: 0.,
                    footprint: self.footprint(),
                });
            }
            if let Some(lead) = self.lead {
                return Some(Work {
                    stage: Stage::Promote,
                    quote: priced(self.promotion),
                    forming: Quote::default(),
                    gain: self.served - lead,
                    footprint: self.footprint(),
                });
            }
            (self.measured.len() < self.challengers.len()).then(|| Work {
                stage: if self.measured.len() < self.forms {
                    Stage::Form
                } else {
                    Stage::Measure
                },
                quote: priced(
                    self.measurement * self.challenging() as u32 + self.uncompiled(),
                ),
                forming: priced(self.uncompiled()),
                gain: self.served / (self.measured.len() + 2) as f64,
                footprint: self.measuring(),
            })
        }

        fn narrow(&mut self, _: &mut Log) -> bool {
            let challenging = self.challenging();
            if self.lead.is_some() || challenging <= 1 {
                return false;
            }
            self.size = challenging - 1;
            true
        }

        fn reclaim(_: &mut Log) {
            MEMORY.with(|memory| {
                let mut now = memory.get();
                now.held -= std::mem::take(&mut now.freed);
                memory.set(now);
            });
        }

        fn holding(&self) -> u64 {
            if self.resident { self.inputs } else { 0 }
        }

        fn release(&mut self, _: &mut Log) {
            if std::mem::take(&mut self.resident) {
                MEMORY.with(|memory| {
                    let mut now = memory.get();
                    now.held -= self.inputs;
                    now.host -= self.host;
                    memory.set(now);
                });
            }
        }

        fn host_free(_: &Log) -> HostRoom {
            let memory = MEMORY.with(Cell::get);
            match memory.host_ceiling {
                None => HostRoom::Unbounded,
                Some(ceiling) => ceiling
                    .checked_sub(memory.host)
                    .map_or(HostRoom::Exceeded, HostRoom::Free),
            }
        }

        fn host_footprint(&self, _: &Log) -> Option<u64> {
            Some(if self.resident { 0 } else { self.host })
        }

        fn programs(&self) -> u64 {
            self.programs_held
        }

        fn give_up(&mut self, _: &mut Log) {
            let given_up = std::mem::take(&mut self.programs_held);
            self.given_up += 1;
            MEMORY.with(|memory| {
                let mut now = memory.get();
                now.held -= given_up;
                memory.set(now);
            });
        }

        fn largest_footprint(&self) -> u64 {
            if self.built > 0 { self.inputs } else { 0 }
        }

        fn free(_: &Log) -> Option<u64> {
            let memory = MEMORY.with(Cell::get);
            memory
                .ceiling
                .map(|ceiling| ceiling.saturating_sub(memory.held))
        }

        fn capacity(_: &Log) -> Option<u64> {
            MEMORY.with(Cell::get).ceiling
        }

        fn cold_compile(_: &Log) -> Duration {
            COLD.with(Cell::get)
        }

        fn owed(&self, worked: bool) -> Quote {
            let mut owed = Duration::ZERO;
            // Only a conversion changes the served choice.
            if self.unformed || (worked && self.lead.is_some()) {
                owed += self.artifact;
            }
            if self.changed || worked {
                owed += self.publication;
            }
            priced(owed)
        }

        fn settle(&mut self, _: &mut Log) -> Result<(), String> {
            if std::mem::take(&mut self.unformed) {
                self.clock.pass(self.artifact);
            }
            if std::mem::take(&mut self.changed) {
                self.clock.pass(self.publication);
                self.published += 1;
            }
            Ok(())
        }

        fn forming(&self) -> Option<Duration> {
            let work = self.next()?;
            (work.forming != Quote::default())
                .then(|| self.uncompiled().mul_f64(self.slow * self.compile_slow))
        }

        fn formed(&mut self, (): (), took: Duration, _: &mut Log) -> Result<(), String> {
            self.compiled = true;
            self.longest = self.longest.max(took);
            Ok(())
        }

        fn perform(&mut self, log: &mut Log, before: Duration) -> Result<Duration, String> {
            let work = self.next().expect("a unit performs the work it named");
            log.borrow_mut()
                .push((self.name, work.stage, self.clock.now()));
            // Every piece holds the unit's inputs, built when they are not
            // resident.
            if !std::mem::replace(&mut self.resident, true) {
                self.built += 1;
                MEMORY.with(|memory| {
                    let mut now = memory.get();
                    now.held += self.inputs;
                    now.peak = now.peak.max(now.held);
                    now.host += self.host;
                    now.host_peak = now.host_peak.max(now.host);
                    memory.set(now);
                });
            }
            let quoted = match work.stage {
                Stage::Discovery => self.discovery,
                // A compile that did not run ahead runs with its piece.
                Stage::Form | Stage::Measure => {
                    self.measurement * self.challenging() as u32 + self.uncompiled()
                }
                Stage::Promote | Stage::Confirm => self.promotion,
            };
            let actual = quoted.mul_f64(self.slow);
            let remaining = before.saturating_sub(self.clock.now());
            if self.interruptible && actual > remaining {
                self.clock.pass(remaining);
                self.stopped = true;
                self.changed = true;
                return Ok(self.longest);
            }
            // Work begun runs to its end, whatever the clock reads.
            self.clock.pass(actual);
            self.longest = self.longest.max(actual);
            self.changed = true;
            match work.stage {
                Stage::Discovery => {
                    self.discovered += 1;
                    MEMORY.with(|memory| {
                        let mut now = memory.get();
                        now.held += self.kept;
                        now.peak = now.peak.max(now.held);
                        memory.set(now);
                    });
                }
                Stage::Form | Stage::Measure => {
                    let programs = self.program * self.challenging() as u64;
                    self.programs_held += programs;
                    MEMORY.with(|memory| {
                        let mut now = memory.get();
                        now.held += programs + self.residue + self.garbage;
                        now.freed += self.garbage;
                        now.peak = now.peak.max(now.held);
                        memory.set(now);
                    });
                    for _ in 0..self.challenging() {
                        let cost = self.challengers[self.measured.len()];
                        self.measured.push(cost);
                        if cost < self.served {
                            self.lead = Some(cost);
                        }
                    }
                    self.compiled = false;
                    self.size = self.batch;
                }
                Stage::Promote | Stage::Confirm => {
                    let lead = self.lead.take().expect("a promotion converts a lead");
                    if self.covers {
                        self.served = lead;
                        self.unformed = true;
                    }
                }
            }
            Ok(self.longest)
        }
    }

    /// Workers on the fake clock: a compile ends when the clock has passed
    /// what it takes, whatever the device did meanwhile. Each compile runs
    /// on one of `threads` threads.
    struct Overlapping {
        clock: Fake,
        threads: usize,
        /// Each compile in flight: its unit, when it ends, what it takes.
        compiling: Vec<(usize, Duration, Duration)>,
        /// The most compiles in flight at once.
        most: usize,
    }

    impl Overlapping {
        fn new(clock: &Fake, threads: usize) -> Self {
            Self {
                clock: clock.clone(),
                threads,
                compiling: Vec::new(),
                most: 0,
            }
        }
    }

    impl Workers<Synthetic> for Overlapping {
        fn needs(&self, _: &Duration) -> usize {
            1
        }

        fn free(&self) -> usize {
            self.threads - self.compiling.len()
        }

        fn start(&mut self, unit: usize, takes: Duration) {
            self.compiling
                .push((unit, self.clock.now() + takes, takes));
            self.most = self.most.max(self.compiling.len());
        }

        fn ended(&mut self) -> Vec<(usize, (), Duration)> {
            let now = self.clock.now();
            let (ended, compiling) = std::mem::take(&mut self.compiling)
                .into_iter()
                .partition(|(_, ends, _)| *ends <= now);
            self.compiling = compiling;
            ended
                .into_iter()
                .map(|(unit, _, took)| (unit, (), took))
                .collect()
        }

        fn wait(&mut self, within: Option<Duration>) -> Option<(usize, (), Duration)> {
            let first = (0..self.compiling.len())
                .min_by_key(|index| self.compiling[*index].1)
                .expect("a compile is in flight");
            let left = self.compiling[first].1.saturating_sub(self.clock.now());
            if let Some(within) = within.filter(|within| *within < left) {
                self.clock.pass(within);
                return None;
            }
            let (unit, _, took) = self.compiling.remove(first);
            self.clock.pass(left);
            Some((unit, (), took))
        }
    }

    /// No workers: a compile runs where it is started, the device waiting.
    struct Serial(Fake, Vec<(usize, Duration)>);

    impl Workers<Synthetic> for Serial {
        fn needs(&self, _: &Duration) -> usize {
            1
        }

        fn free(&self) -> usize {
            1
        }

        fn start(&mut self, unit: usize, takes: Duration) {
            self.0.pass(takes);
            self.1.push((unit, takes));
        }

        fn ended(&mut self) -> Vec<(usize, (), Duration)> {
            self.1.drain(..).map(|(unit, took)| (unit, (), took)).collect()
        }

        fn wait(&mut self, _: Option<Duration>) -> Option<(usize, (), Duration)> {
            unreachable!("a serial compile has ended when it is started")
        }
    }

    fn run(units: &mut [Synthetic], clock: &Fake, total: u64) -> (Vec<(usize, Stage, Duration)>, Report) {
        let mut log = Log::default();
        let (report, outcome) = allocate(
            units,
            &mut log,
            clock,
            &mut Overlapping::new(clock, usize::MAX),
            SECOND * total as u32,
            Duration::ZERO,
        );
        outcome.expect("no synthetic work fails");
        (log.into_inner(), report)
    }

    /// Units whose every challenger compiles for 3 s and measures for 1 s.
    fn compiling(clock: &Fake, units: usize, challengers: usize, slow: f64) -> Vec<Synthetic> {
        (0..units)
            .map(|name| Synthetic {
                compile: SECOND * 3,
                slow,
                ..Synthetic::new(name, clock, 1. / units as f64, &vec![1.; challengers])
            })
            .collect()
    }

    /// Compiles run while the device measures: the same work ends sooner
    /// than with the device waiting on each compile, every compile and
    /// every measurement takes what it was quoted, and the device goes
    /// from one measurement to the next.
    #[test]
    fn compiles_overlap_the_device_and_the_same_work_ends_sooner() {
        let clock = Fake::default();
        let mut serial = compiling(&clock, 4, 3, 1.);
        let (report, outcome) = allocate(
            &mut serial,
            &mut Log::default(),
            &clock,
            &mut Serial(clock.clone(), Vec::new()),
            SECOND * 600,
            Duration::ZERO,
        );
        outcome.unwrap();
        // 4 discoveries, then 12 challengers at 3 s + 1 s.
        assert_eq!(report.ended, SECOND * 52);
        let clock = Fake::default();
        let mut units = compiling(&clock, 4, 3, 1.);
        let (log, report) = run(&mut units, &clock, 600);
        assert!(units.iter().all(|unit| unit.measured.len() == 3));
        // 4 discoveries, one wait of 3 s for the first compiles, then the
        // device measures without a gap: every later compile ended behind
        // the measurements admitted before it.
        assert_eq!(report.ended, SECOND * 19);
        let measured = log
            .iter()
            .filter(|(_, stage, _)| *stage == Stage::Measure)
            .map(|(_, _, began)| *began)
            .collect::<Vec<_>>();
        assert_eq!(measured, (7..19).map(|second| SECOND * second).collect::<Vec<_>>());
        for admitted in &report.admitted {
            if admitted.stage == Stage::Measure {
                assert_eq!(admitted.formed, Some(SECOND * 3));
                assert_eq!(admitted.actual, SECOND);
                assert_eq!(admitted.quote.forecast(), Some(SECOND * 4));
                // Admitted on its whole price: it ended no sooner than that.
                assert!(admitted.performed + admitted.actual >= admitted.began + SECOND * 4);
            }
        }
    }

    /// Forecasts stay honest under overlap: with accurate prices, work in
    /// flight is counted against what remains, so nothing is admitted that
    /// would end past the budget, and what does not fit behind the work
    /// in flight is not started.
    #[test]
    fn work_in_flight_is_counted_against_what_remains() {
        let clock = Fake::default();
        let mut units = compiling(&clock, 4, 3, 1.);
        let (_, report) = run(&mut units, &clock, 12);
        // The four first challengers' compiles start a second apart from
        // 4 s, each so that its piece is ready as the device ends the one
        // before, and their measurements end at 11 s. The first unit's
        // second challenger compiles from 8 s, while the device measures
        // the others, and measures from 11 s: it ends with the budget, and
        // nothing more does.
        assert_eq!(report.ended, SECOND * 12);
        assert_eq!((report.unused(), report.overrun()), (Duration::ZERO, Duration::ZERO));
        let compiles = report
            .admitted
            .iter()
            .filter(|admitted| admitted.stage == Stage::Measure)
            .map(|admitted| admitted.began)
            .collect::<Vec<_>>();
        assert_eq!(compiles, [4, 5, 6, 7, 8].map(|second| SECOND * second));
        assert_eq!(
            units.iter().map(|unit| unit.measured.len()).collect::<Vec<_>>(),
            [2, 1, 1, 1]
        );
        assert_eq!(report.refused.len(), 4);
        // One second less and the fourth does not fit behind the other
        // three: it is not started.
        let clock = Fake::default();
        let mut units = compiling(&clock, 4, 3, 1.);
        let (_, report) = run(&mut units, &clock, 10);
        assert_eq!(report.ended, SECOND * 10);
        assert_eq!(report.overrun(), Duration::ZERO);
        assert_eq!(
            units.iter().map(|unit| unit.measured.len()).sum::<usize>(),
            3
        );
    }

    /// Run `units` on workers of `threads` threads: the log, the report, and
    /// the most compiles that were in flight at once.
    fn threaded(
        units: &mut [Synthetic],
        clock: &Fake,
        total: Duration,
        threads: usize,
    ) -> (Vec<(usize, Stage, Duration)>, Report, usize) {
        let mut log = Log::default();
        let mut workers = Overlapping::new(clock, threads);
        let (report, outcome) =
            allocate(units, &mut log, clock, &mut workers, total, Duration::ZERO);
        outcome.expect("no synthetic work fails");
        (log.into_inner(), report, workers.most)
    }

    /// One unit's challenger compiles for 10 s beside three units whose
    /// challengers compile for 1 s and measure for a fifth of one. The long
    /// compile holds back nothing but its own device time: the short
    /// compiles overlap it and each other, each started as the device nears
    /// the end of the pieces before it, their pieces are measured while it
    /// runs, and a budget that ends with the long piece's measurement
    /// admits every one of them. Counted together with the long compile,
    /// as they were, the short pieces were forecast to end behind it: none
    /// of them fit this budget.
    #[test]
    fn a_short_compile_is_not_forecast_behind_an_unrelated_long_one() {
        let clock = Fake::default();
        let short = |name: usize| Synthetic {
            compile: SECOND,
            measurement: SECOND / 5,
            ..Synthetic::new(name, &clock, 0.1, &[1.; 3])
        };
        let mut units = vec![
            Synthetic {
                compile: SECOND * 10,
                ..Synthetic::new(0, &clock, 0.7, &[1.])
            },
            short(1),
            short(2),
            short(3),
        ];
        // Four discoveries; the long piece, started then, measures from
        // 14 s to 15 s.
        let (log, report, most) = threaded(&mut units, &clock, SECOND * 15, usize::MAX);
        assert_eq!(
            units.iter().map(|unit| unit.measured.len()).collect::<Vec<_>>(),
            [1, 3, 3, 3]
        );
        assert_eq!((report.ended, report.overrun()), (SECOND * 15, Duration::ZERO));
        let long = report
            .admitted
            .iter()
            .find(|admitted| admitted.unit == 0 && admitted.stage == Stage::Measure)
            .unwrap();
        assert_eq!((long.began, long.performed), (SECOND * 4, SECOND * 14));
        // Every short piece ran, start to end, inside the long compile.
        for admitted in &report.admitted {
            if admitted.unit != 0 && admitted.stage == Stage::Measure {
                assert!(admitted.began >= long.began, "{admitted:?}");
                assert!(admitted.performed + admitted.actual <= long.performed, "{admitted:?}");
                assert_eq!(admitted.formed, Some(SECOND));
            }
        }
        // The long compile and the three units' short ones at once.
        assert_eq!(most, 4);
        // The device went from one short piece to the next: after the
        // first, none waited for a compile that could have started sooner.
        let measured = log
            .iter()
            .filter(|(unit, stage, _)| *unit != 0 && *stage == Stage::Measure)
            .map(|(_, _, began)| *began)
            .collect::<Vec<_>>();
        assert_eq!(measured[0], SECOND * 5);
        assert_eq!(measured[..3], [50, 52, 54].map(|tenths| SECOND * tenths / 10));
    }

    /// Compiles overlap as far as the worker threads there are, and no
    /// further: each compile runs on a thread of its own.
    #[test]
    fn compiles_overlap_up_to_the_worker_threads() {
        let most = |threads: usize| {
            let clock = Fake::default();
            let mut units = compiling(&clock, 6, 2, 1.);
            let (_, report, most) = threaded(&mut units, &clock, SECOND * 600, threads);
            assert!(units.iter().all(|unit| unit.measured.len() == 2));
            (most, report.ended)
        };
        // Compiles of 3 s and measurements of 1 s keep three compiles in
        // flight when nothing limits them.
        let (unlimited, sooner) = most(usize::MAX);
        assert_eq!(unlimited, 3);
        let (two, later) = most(2);
        assert_eq!(two, 2);
        let (one, latest) = most(1);
        assert_eq!(one, 1);
        assert!(sooner < later && later < latest, "{sooner:?} {later:?} {latest:?}");
    }

    /// Compiles overlap as far as their pieces' bytes fit under the ceiling
    /// together: a piece in flight holds its footprint from the moment its
    /// compile starts.
    #[test]
    fn compiles_overlap_up_to_the_bytes_their_pieces_hold() {
        let most = |ceiling: Option<u64>| {
            let clock = Fake::default();
            let mut units = compiling(&clock, 6, 2, 1.);
            for unit in &mut units {
                unit.inputs = 100;
            }
            MEMORY.with(|memory| {
                memory.set(Memory {
                    ceiling,
                    ..Memory::default()
                })
            });
            let (_, _, most) = threaded(&mut units, &clock, SECOND * 600, usize::MAX);
            let memory = MEMORY.with(Cell::take);
            assert!(units.iter().all(|unit| unit.measured.len() == 2));
            (most, memory.peak)
        };
        assert_eq!(most(None).0, 3);
        // Two units' inputs fit under the ceiling: two pieces fly at once.
        let (two, peak) = most(Some(250));
        assert_eq!(two, 2);
        assert!(peak <= 250, "{peak}");
    }

    /// Every compile and measurement takes ten times its quote: nothing is
    /// admitted once the budget has ended, and the time past the budget is
    /// at most what the work in flight when it ended took: pieces performed
    /// across it, and compiles abandoned at the ending.
    #[test]
    fn overlapping_work_overruns_by_at_most_the_pieces_in_flight() {
        let clock = Fake::default();
        let mut units = compiling(&clock, 4, 3, 10.);
        let budget = SECOND * 100;
        let (log, report) = run(&mut units, &clock, 100);
        assert_eq!(log.len(), report.admitted.len());
        assert!(report.admitted.iter().all(|admitted| admitted.began < budget));
        assert!(report.abandoned.iter().all(|abandoned| abandoned.began < budget));
        let performed = report
            .admitted
            .iter()
            .filter(|admitted| admitted.performed + admitted.actual > budget)
            .map(|admitted| admitted.formed.unwrap_or_default() + admitted.actual)
            .sum::<Duration>();
        let abandoned = report
            .abandoned
            .iter()
            .map(|abandoned| abandoned.formed)
            .sum::<Duration>();
        assert!(report.overrun() <= performed + abandoned, "{report:?}");
        // Each piece measured was admitted, and no abandoned piece was
        // measured.
        assert_eq!(
            report
                .admitted
                .iter()
                .filter(|admitted| admitted.stage == Stage::Measure)
                .count(),
            units.iter().map(|unit| unit.measured.len()).sum::<usize>()
        );
    }

    /// A compile priced at 1 s takes 5.5 s. Its piece was admitted at 14 s,
    /// forecast to end at 16 s with 2 s of publication after, but its
    /// compile ends at 19.5 s. Waiting for it and then measuring would take
    /// the time held back for the publication and end at 22.5 s. Instead
    /// the ending begins at the piece's cutoff, 18 s: the piece is
    /// abandoned unmeasured, and the publication runs while its compile
    /// still does. The budget ends with the publication, the compile
    /// having ended inside it.
    #[test]
    fn an_underpriced_compile_does_not_take_the_time_held_for_the_ending() {
        let clock = Fake::default();
        let mut units = [Synthetic {
            compile: SECOND,
            compile_slow: 5.5,
            publication: SECOND * 2,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (log, report) = run(&mut units, &clock, 20);
        // Discovery, then two pieces of 5.5 s compiles and 1 s measurements.
        assert_eq!(log.len(), 3);
        assert_eq!(units[0].measured.len(), 2);
        assert_eq!(
            report.abandoned,
            [Abandoned {
                unit: 0,
                stage: Stage::Measure,
                quote: priced(SECOND * 2),
                began: SECOND * 14,
                formed: SECOND * 11 / 2,
            }]
        );
        assert_eq!(units[0].published, 1);
        assert_eq!(report.settling, SECOND * 2);
        assert_eq!(report.ended, SECOND * 20);
        assert_eq!(report.overrun(), Duration::ZERO);
        // The unit was in flight: it offered nothing to refuse.
        assert!(report.refused.is_empty());
    }

    /// A compile priced at 1 s takes 12 s. Once it has ended, no compile
    /// starts that would run past the budget were it to take as long: the
    /// next piece, forecast to fit at 14 s, is not started, where started
    /// it would have compiled to 26 s.
    #[test]
    fn a_compile_starts_only_if_the_longest_seen_ends_within_the_budget() {
        let clock = Fake::default();
        let mut units = [Synthetic {
            compile: SECOND,
            compile_slow: 12.,
            publication: SECOND * 2,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (log, report) = run(&mut units, &clock, 20);
        // Discovery to 1 s, then a piece whose compile runs from 1 s to
        // 13 s and measures to 14 s; its publication is settled then.
        assert_eq!(log.len(), 2);
        assert!(report.abandoned.is_empty());
        assert_eq!(report.ended, SECOND * 16);
        assert_eq!(report.overrun(), Duration::ZERO);
        assert_eq!(report.refused.len(), 1);
    }

    /// Compiles a system cache serves take a tenth of what one it does not
    /// serve takes, and every compile the load runs is served: the load
    /// sees only quick compiles. A compile that nothing served
    /// ([`Unit::cold_compile`]) took 5 s, and no compile starts that would
    /// run past the budget were it to take that long.
    #[test]
    fn a_compile_is_held_to_one_no_cache_served() {
        COLD.with(|cold| cold.set(SECOND * 5));
        let clock = Fake::default();
        let mut units = [Synthetic {
            compile: SECOND / 2,
            measurement: SECOND / 2,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (_, report) = run(&mut units, &clock, 20);
        COLD.with(|cold| cold.set(Duration::ZERO));
        assert!(
            report
                .admitted
                .iter()
                .filter(|admitted| admitted.formed.is_some())
                .all(|admitted| admitted.began + SECOND * 5 <= SECOND * 20),
            "{report:?}"
        );
        assert!(report.abandoned.is_empty());
        assert_eq!(report.overrun(), Duration::ZERO);
        // Without it, the same load compiles until its last second.
        let clock = Fake::default();
        let mut units = [Synthetic {
            compile: SECOND / 2,
            measurement: SECOND / 2,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (_, report) = run(&mut units, &clock, 20);
        assert!(
            report
                .admitted
                .iter()
                .any(|admitted| admitted.formed.is_some() && admitted.began > SECOND * 15)
        );
    }

    fn stages(log: &[(usize, Stage, Duration)]) -> Vec<(usize, Stage)> {
        log.iter().map(|(unit, stage, _)| (*unit, *stage)).collect()
    }

    /// Accurate quotes: the allocator stops at or before the budget with no
    /// overrun, every piece it started took what it was quoted, and it
    /// stops early, the remaining budget unused, when nothing that is left
    /// fits.
    #[test]
    fn accurate_quotes_end_within_the_budget_and_stop_early_when_nothing_fits() {
        let clock = Fake::default();
        let mut units = [
            Synthetic::new(0, &clock, 0.6, &[1.; 100]),
            Synthetic::new(1, &clock, 0.4, &[1.; 100]),
        ];
        // No challenger leads: only discovery and measurement run.
        let (_, report) = run(&mut units, &clock, 20);
        assert_eq!(report.ended, SECOND * 20);
        assert_eq!((report.unused(), report.overrun()), (Duration::ZERO, Duration::ZERO));
        assert!(
            report
                .admitted
                .iter()
                .all(|admitted| admitted.quote.forecast() == Some(admitted.actual))
        );
        // Pieces that do not fit what is left are never started: 2.5 s of
        // work does not fit the last 2 s, and the budget is left unused.
        let clock = Fake::default();
        let mut units = [Synthetic {
            measurement: SECOND * 5 / 2,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (log, report) = run(&mut units, &clock, 8);
        // 1 s of discovery, two measurements, and 2 s left.
        assert_eq!(report.ended, SECOND * 6);
        assert_eq!(report.unused(), SECOND * 2);
        assert_eq!(report.overrun(), Duration::ZERO);
        assert_eq!(log.len(), 3);
        assert_eq!(
            report.refused,
            [Refused {
                unit: 0,
                stage: Stage::Measure,
                quote: priced(SECOND * 5 / 2),
                footprint: seismic::Footprint::default(),
                refusal: Refusal::Seconds,
            }]
        );
        // With budget for everything, the allocator stops when the work
        // ends, not when the budget does.
        let clock = Fake::default();
        let mut units = [Synthetic::new(0, &clock, 1., &[1., 1.])];
        let (_, report) = run(&mut units, &clock, 600);
        assert_eq!(report.ended, SECOND * 3);
        assert!(report.refused.is_empty());
        assert_eq!(units[0].measured.len(), 2);
    }

    /// Every piece takes ten times its quote: every unit still serves a
    /// legal choice, nothing fails and nothing begun is abandoned, and the
    /// time past the budget is at most what the last piece started took
    /// beyond its forecast, reported with that forecast and what it took.
    #[test]
    fn work_ten_times_its_quote_overruns_by_at_most_the_piece_in_flight() {
        let clock = Fake::default();
        let mut units = [0.5, 0.3, 0.2]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                slow: 10.,
                ..Synthetic::new(name, &clock, share, &[0.9, 0.8, 0.7, 0.6])
            })
            .collect::<Vec<_>>();
        let (log, report) = run(&mut units, &clock, 55);
        // Each piece started ran to its end and changed its unit's state.
        assert_eq!(log.len(), report.admitted.len());
        let last = report.admitted.last().unwrap();
        assert!(last.began < SECOND * 55, "nothing starts past the budget");
        assert!(last.began + last.quote.priced <= SECOND * 55);
        assert_eq!(report.ended, last.began + last.actual);
        assert!(report.overrun() > Duration::ZERO);
        assert!(report.overrun() <= last.actual - last.quote.priced);
        assert_eq!(last.actual, last.quote.forecast().unwrap() * 10);
        for unit in &units {
            // A legal choice at every instant: the defaults, or a
            // challenger measured faster and promoted.
            assert!(unit.served == 1. || unit.measured.contains(&unit.served));
            assert!(unit.served <= 1.);
        }
    }

    #[test]
    fn interrupted_work_keeps_the_original_cutoff_and_publication_reserve() {
        let clock = Fake::default();
        let mut units = [Synthetic {
            slow: 10.,
            interruptible: true,
            publication: SECOND,
            ..Synthetic::new(0, &clock, 1., &[1.; 100])
        }];
        let (_, report) = run(&mut units, &clock, 55);
        assert!(
            units[0].stopped,
            "the underestimated piece reached its cutoff"
        );
        assert_eq!(report.overrun(), Duration::ZERO);
        assert_eq!(
            report.ended,
            SECOND * 55,
            "publication stays inside the allowance"
        );
        assert_eq!(
            units[0].served, 1.,
            "the incomplete challenger cannot replace the choice"
        );
        assert!(units[0].published > 0);
        assert!(units[0].measured.len() < 100);
    }

    /// A measured lead is converted before any new challenger is measured,
    /// whatever the other units' shares, and the challenger that measured
    /// faster and passed its conversion is the unit's served choice.
    #[test]
    fn a_lead_is_converted_before_new_measurement_and_the_promoted_challenger_serves() {
        let clock = Fake::default();
        let mut units = [
            Synthetic::new(0, &clock, 0.1, &[0.5, 1., 1.]),
            Synthetic::new(1, &clock, 0.9, &[1.; 20]),
        ];
        let (log, report) = run(&mut units, &clock, 30);
        let log = stages(&log);
        let lead = log
            .iter()
            .position(|step| *step == (0, Stage::Measure))
            .unwrap();
        assert_eq!(
            log[lead + 1],
            (0, Stage::Promote),
            "the conversion follows its measurement at once: {log:?}"
        );
        assert_eq!(units[0].served, 0.5, "the promoted challenger is served");
        assert_eq!(units[1].served, 1.);
        assert_eq!(report.overrun(), Duration::ZERO);
        // A lead that fails its coverage is not served, and the next
        // challenger is measured against the choice that stands.
        let clock = Fake::default();
        let mut units = [Synthetic {
            covers: false,
            ..Synthetic::new(0, &clock, 1., &[0.5, 0.4])
        }];
        let (log, _) = run(&mut units, &clock, 30);
        assert_eq!(
            stages(&log)[1..],
            [
                (0, Stage::Measure),
                (0, Stage::Promote),
                (0, Stage::Measure),
                (0, Stage::Promote)
            ]
        );
        assert_eq!(units[0].served, 1.);
        assert_eq!(units[0].measured, [0.5, 0.4]);
    }

    /// A conversion whose whole price does not fit what remains is never
    /// started: its challenger stays recorded as a measurement and does not
    /// serve, and the unit measures nothing more it could not convert.
    #[test]
    fn a_conversion_that_does_not_fit_is_not_started_and_its_measurement_stays_recorded() {
        let clock = Fake::default();
        let mut units = [Synthetic {
            promotion: SECOND * 5,
            ..Synthetic::new(0, &clock, 1., &[0.5, 0.4, 0.3])
        }];
        // Discovery and one measurement leave 4 s: less than the 5 s
        // conversion.
        let (log, report) = run(&mut units, &clock, 6);
        assert_eq!(
            stages(&log),
            [(0, Stage::Discovery), (0, Stage::Measure)]
        );
        assert_eq!(report.ended, SECOND * 2);
        assert_eq!(
            report.refused,
            [Refused {
                unit: 0,
                stage: Stage::Promote,
                quote: priced(SECOND * 5),
                footprint: seismic::Footprint::default(),
                refusal: Refusal::Seconds,
            }]
        );
        assert_eq!(units[0].measured, [0.5], "the measurement is recorded");
        assert_eq!(units[0].lead, Some(0.5));
        assert_eq!(units[0].served, 1., "an unconverted lead does not serve");
        assert_eq!(report.overrun(), Duration::ZERO);
    }

    /// Discovery comes before any challenger: every unit's first piece
    /// before any unit's second, a unit never started first.
    #[test]
    fn every_unit_is_discovered_before_any_challenger_is_measured() {
        let clock = Fake::default();
        // Preparation order is not share order.
        let mut units = [0.1, 0.5, 0.15, 0.25]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                points: 2,
                ..Synthetic::new(name, &clock, share, &[1.; 50])
            })
            .collect::<Vec<_>>();
        let (log, _) = run(&mut units, &clock, 60);
        let log = stages(&log);
        // Each unit's first point before any unit's second, the largest
        // share first both times.
        assert_eq!(
            log[..8],
            [1, 3, 2, 0, 1, 3, 2, 0].map(|unit| (unit, Stage::Discovery))
        );
        assert!(log[8..].iter().all(|(_, stage)| *stage == Stage::Measure));
    }

    /// Challengers go where a second of their price recovers most: by
    /// share, and less to a unit the more it has measured, so the time
    /// follows the shares instead of one unit taking it all.
    #[test]
    fn measurement_follows_share_and_expected_gain_per_second() {
        let clock = Fake::default();
        let mut units = [0.5, 0.3, 0.2]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic::new(name, &clock, share, &[1.; 1000]))
            .collect::<Vec<_>>();
        let (log, _) = run(&mut units, &clock, 103);
        // The largest share measures first.
        assert_eq!(stages(&log)[3], (0, Stage::Measure));
        let measured = units.iter().map(|unit| unit.measured.len()).collect::<Vec<_>>();
        assert_eq!(measured.iter().sum::<usize>(), 100);
        for (unit, share) in [0.5, 0.3, 0.2].into_iter().enumerate() {
            assert!(
                (measured[unit] as f64 - 100. * share).abs() <= 2.,
                "{measured:?}"
            );
        }
        // Equal shares: the unit whose challengers cost four times as much
        // measures a quarter as many, for the same time.
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                measurement: SECOND * 4,
                ..Synthetic::new(0, &clock, 0.5, &[1.; 1000])
            },
            Synthetic::new(1, &clock, 0.5, &[1.; 1000]),
        ];
        run(&mut units, &clock, 202);
        let (slow, fast) = (units[0].measured.len() as f64, units[1].measured.len() as f64);
        assert!((fast / slow - 4.).abs() < 0.5, "{slow} {fast}");
    }

    /// A unit composed of another's choice reads it as it stands when its
    /// own work starts: the prerequisite is sealed from then on and
    /// receives no more work, and nothing is paid to seal it.
    #[test]
    fn a_dependants_work_seals_its_prerequisites_at_no_cost() {
        let clock = Fake::default();
        let mut units = [
            Synthetic::new(0, &clock, 0.5, &[0.5, 1., 1., 1.]),
            Synthetic {
                prerequisites: vec![0],
                ..Synthetic::new(1, &clock, 0.5, &[1.])
            },
        ];
        let (log, report) = run(&mut units, &clock, 60);
        let log = stages(&log);
        let dependant = log
            .iter()
            .position(|step| *step == (1, Stage::Measure))
            .unwrap();
        // The prerequisite's lead was converted before it was read.
        assert!(log[..dependant].contains(&(0, Stage::Promote)));
        assert!(
            log[dependant..].iter().all(|(unit, _)| *unit != 0),
            "a sealed prerequisite receives no more work: {log:?}"
        );
        assert_eq!(units[0].served, 0.5);
        // Sealing took no time: the clock is the sum of the pieces.
        assert_eq!(
            report.ended,
            report.admitted.iter().map(|admitted| admitted.actual).sum()
        );
    }

    /// Work of a class never observed has no price. It is started while
    /// the longest operation completed fits what remains, and not after.
    #[test]
    fn unpriced_work_is_not_started_late() {
        let unit = |clock: &Fake, name, challengers: &[f64]| Synthetic {
            unpriced_discovery: name == 1,
            discovery: if name == 1 { SECOND } else { SECOND * 3 },
            ..Synthetic::new(name, clock, 0.5, challengers)
        };
        // With the whole budget left the unpriced discovery runs first
        // among the never-started units' pieces it shares the order with.
        let clock = Fake::default();
        let mut units = [unit(&clock, 0, &[]), unit(&clock, 1, &[])];
        let (log, _) = run(&mut units, &clock, 10);
        assert_eq!(
            stages(&log),
            [(0, Stage::Discovery), (1, Stage::Discovery)]
        );
        // The first unit's 3 s discovery is the longest operation the load
        // has completed; with 2 s left the unpriced discovery is late.
        let clock = Fake::default();
        let mut units = [unit(&clock, 0, &[]), unit(&clock, 1, &[])];
        let (log, report) = run(&mut units, &clock, 5);
        assert_eq!(stages(&log), [(0, Stage::Discovery)]);
        assert_eq!(report.ended, SECOND * 3);
        assert_eq!(report.refused.len(), 1);
        assert_eq!(report.refused[0].quote.forecast(), None);
        assert_eq!(units[1].served, 1., "the unit keeps its published defaults");
    }

    /// Run `units` under `ceiling` bytes and return what they held at most.
    fn bounded(units: &mut [Synthetic], clock: &Fake, total: u64, ceiling: u64) -> (Memory, Report) {
        MEMORY.with(|memory| {
            memory.set(Memory {
                ceiling: Some(ceiling),
                ..Memory::default()
            })
        });
        let (_, report) = run(units, clock, total);
        (MEMORY.with(Cell::take), report)
    }

    /// Many units whose inputs together are several times the ceiling: what
    /// they hold never exceeds it at any instant, every unit still gets its
    /// work, and inputs are given up only when a chosen piece needs their
    /// bytes. This fails if any piece is started that does not fit.
    #[test]
    fn what_the_units_hold_never_exceeds_the_ceiling() {
        let clock = Fake::default();
        let mut units = (0..12)
            .map(|name| Synthetic {
                inputs: 100 + 10 * name as u64,
                ..Synthetic::new(name, &clock, 1. / 12., &[1.; 6])
            })
            .collect::<Vec<_>>();
        let whole = units.iter().map(|unit| unit.inputs).sum::<u64>();
        let ceiling = 400;
        assert!(whole > 4 * ceiling);
        let (memory, report) = bounded(&mut units, &clock, 600, ceiling);
        assert!(memory.peak <= ceiling, "{} bytes held", memory.peak);
        assert!(memory.peak > ceiling / 2, "the bytes free were used");
        // Every unit was discovered and measured to its end.
        assert!(units.iter().all(|unit| unit.measured.len() == 6));
        assert!(report.refused.is_empty());
        // Nothing is held when no unit has work left.
        assert_eq!(memory.held, 0);
        assert!(units.iter().all(|unit| !unit.resident));
        // With bytes for everything, nothing is built twice: inputs are
        // kept between a unit's pieces while they fit.
        let clock = Fake::default();
        let mut units = (0..12)
            .map(|name| Synthetic {
                inputs: 100,
                ..Synthetic::new(name, &clock, 1. / 12., &[1.; 6])
            })
            .collect::<Vec<_>>();
        let (memory, _) = bounded(&mut units, &clock, 600, 1200);
        assert_eq!(memory.peak, 1200);
        assert!(units.iter().all(|unit| unit.built == 1));
    }

    /// The whole budget is used, whatever it is. A toolchain whose compile
    /// call costs 3 s whatever it forms, with a second of measurement a
    /// challenger: pieces of eight challengers share one call, and the
    /// last piece of a budget is offered with as many challengers as what
    /// is left fits, so the time unused is less than the smallest piece
    /// (one call and one challenger). Nothing is refused for its seconds
    /// while a smaller piece would fit. A domain smaller than the budget
    /// measures is measured whole, and the load then ends early with
    /// nothing refused.
    #[test]
    fn the_budget_is_used_to_the_smallest_piece_whatever_a_compile_call_costs() {
        let (call, challenger, batch) = (SECOND * 3, SECOND, 8);
        let unit = |clock: &Fake, challengers: usize| Synthetic {
            compile: call,
            measurement: challenger,
            batch,
            size: batch,
            ..Synthetic::new(0, clock, 1., &vec![1.; challengers])
        };
        let mut measured = Vec::new();
        for budget in [5u64, 20, 60, 120] {
            let clock = Fake::default();
            let mut units = [unit(&clock, 1000)];
            let (log, report) = run(&mut units, &clock, budget);
            let unused = report.unused();
            assert!(unused < call + challenger, "{budget} s: {unused:?} unused");
            assert_eq!(report.overrun(), Duration::ZERO);
            // What is refused when nothing more starts is a piece of one
            // challenger, and it does not fit.
            assert_eq!(report.refused.len(), 1);
            assert_eq!(units[0].challenging(), 1);
            assert!(report.refused[0].quote.priced > unused);
            // As many configurations a compile call as a piece holds, but
            // for the last piece: a call for each challenger would measure
            // one for every four seconds.
            let calls = log
                .iter()
                .filter(|(_, stage, _)| *stage == Stage::Measure)
                .count();
            let formed = units[0].measured.len();
            assert!(formed > batch * (calls - 1), "{formed} in {calls} calls");
            if budget >= 20 {
                let each_alone = (budget - 1) / (call + challenger).as_secs();
                assert!(formed as u64 >= 2 * each_alone, "{formed} against {each_alone}");
            }
            measured.push(formed);
        }
        assert!(measured.windows(2).all(|pair| pair[0] < pair[1]), "{measured:?}");
        // A domain the budget exhausts: everything is measured, nothing is
        // refused, and the time left is the domain's doing.
        let clock = Fake::default();
        let mut units = [unit(&clock, 5)];
        let (_, report) = run(&mut units, &clock, 120);
        assert_eq!(units[0].measured.len(), 5);
        assert!(report.refused.is_empty());
        assert!(report.unused() > SECOND * 100);
        // Units that promoted beside one still measuring: each owes its
        // choice's serving artifact, a compile call, when the work stops,
        // and nothing else is held back for it. The measuring unit's pieces
        // promote nothing and add no artifact to what is owed, so the
        // budget is still used to the smallest piece: what is unused is
        // less than one call and one challenger, and what was owed is
        // settled inside the budget.
        for budget in [20u64, 60, 120] {
            let clock = Fake::default();
            let promoting = |name: usize| Synthetic {
                artifact: call,
                promotion: challenger,
                ..Synthetic::new(name, &clock, 0.1, &[0.5])
            };
            let mut units = [
                Synthetic {
                    artifact: call,
                    ..unit(&clock, 1000)
                },
                promoting(1),
                promoting(2),
            ];
            let (_, report) = run(&mut units, &clock, budget);
            assert!(units[1..].iter().all(|unit| unit.served == 0.5));
            assert!(units.iter().all(|unit| !unit.unformed));
            let unused = report.unused();
            assert!(unused < call + challenger, "{budget} s: {unused:?} unused");
            assert_eq!(report.overrun(), Duration::ZERO);
            assert_eq!(report.settling, call * 2);
        }
    }

    /// Formed programs are held by need, not by role. While everything
    /// fits, nothing a unit formed is given up, whatever it forms after.
    /// When a chosen piece does not fit, another unit gives up its programs
    /// before any unit gives up its inputs, and only as often as the bytes
    /// are needed: every challenger is still measured, no inputs are built
    /// twice, and the ceiling holds at every instant.
    #[test]
    fn formed_programs_stay_held_while_they_fit_and_go_before_inputs() {
        let (inputs, program, challengers) = (100, 30, 5);
        let pair = |clock: &Fake| {
            [0, 1].map(|name| Synthetic {
                inputs,
                program,
                ..Synthetic::new(name, clock, 0.5, &vec![1.; challengers])
            })
        };
        let everything = 2 * (inputs + program * challengers as u64);
        // Room for everything: nothing is given up.
        let clock = Fake::default();
        let mut units = pair(&clock);
        let (memory, report) = bounded(&mut units, &clock, 600, everything);
        assert!(report.refused.is_empty());
        assert!(units.iter().all(|unit| unit.given_up == 0));
        assert!(units.iter().all(|unit| unit.measured.len() == challengers));
        assert_eq!(
            memory.held,
            units.iter().map(|unit| unit.programs_held).sum::<u64>()
        );
        // Room for less: programs go, by need, and inputs stay.
        let ceiling = 400;
        assert!(ceiling < everything);
        let clock = Fake::default();
        let mut units = pair(&clock);
        let (memory, report) = bounded(&mut units, &clock, 600, ceiling);
        assert!(memory.peak <= ceiling, "{} bytes held", memory.peak);
        assert!(report.refused.is_empty());
        assert!(units.iter().all(|unit| unit.measured.len() == challengers));
        assert!(units.iter().any(|unit| unit.given_up > 0));
        assert!(units.iter().all(|unit| unit.built == 1));
        // No more was given up than the bytes that did not fit: one
        // unit's programs, at most twice.
        assert!(units.iter().map(|unit| unit.given_up).sum::<usize>() <= 2);
    }

    /// Compiles hold bytes too. What a challenger's compile frees counts
    /// until the load returns it, which it does before any unit gives its
    /// inputs up; what a compile leaves for good is given back by nothing,
    /// so the unit stops forming challengers before the next would cross
    /// the ceiling, with budget left, and is refused for its bytes. The
    /// ceiling is never exceeded, what compiles left for good included.
    #[test]
    fn compiles_are_bounded_by_the_bytes_they_hold_and_leave_held() {
        let (inputs, residue, garbage, ceiling) = (100, 30, 50, 400);
        let challengers = 20;
        let unit = |clock: &Fake, residue, garbage| Synthetic {
            inputs,
            residue,
            garbage,
            ..Synthetic::new(0, clock, 1., &vec![1.; challengers])
        };
        let clock = Fake::default();
        let mut units = [unit(&clock, residue, garbage)];
        let (memory, report) = bounded(&mut units, &clock, 600, ceiling);
        assert!(memory.peak <= ceiling, "{} bytes held", memory.peak);
        // The inputs and one compile's freed bytes fit beside what the
        // compiles left for good until the tenth: 100 + 50 + 30 n <= 400.
        let formed = units[0].measured.len() as u64;
        assert_eq!(formed, (ceiling - inputs - garbage) / residue);
        assert!(formed < challengers as u64);
        assert_eq!(
            report
                .refused
                .iter()
                .map(|refused| (refused.unit, refused.refusal))
                .collect::<Vec<_>>(),
            [(0, Refusal::Bytes)]
        );
        assert!(clock.now() < SECOND * 600, "the budget was not what ended it");
        // What the compiles left for good is still held when the unit has
        // given everything else up.
        assert_eq!(memory.held - memory.freed, formed * residue);
        // Compiles that leave nothing for good are bounded by nothing: what
        // they free is returned whenever a piece needs the bytes.
        let clock = Fake::default();
        let mut units = [unit(&clock, 0, garbage)];
        let (memory, report) = bounded(&mut units, &clock, 600, ceiling);
        assert!(memory.peak <= ceiling, "{} bytes held", memory.peak);
        assert_eq!(units[0].measured.len(), challengers);
        assert!(report.refused.is_empty());
    }

    /// A piece whose bytes do not fit under the ceiling, whatever every
    /// other unit gives up, is not started: its unit keeps its published
    /// defaults, holds nothing, and the others' work goes on.
    #[test]
    fn a_piece_whose_bytes_do_not_fit_is_not_started() {
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                inputs: 500,
                ..Synthetic::new(0, &clock, 0.9, &[0.5])
            },
            Synthetic {
                inputs: 100,
                ..Synthetic::new(1, &clock, 0.1, &[0.5])
            },
        ];
        let (memory, report) = bounded(&mut units, &clock, 60, 300);
        assert!(memory.peak <= 300);
        assert_eq!(units[0].discovered, 0, "the piece that does not fit never ran");
        assert_eq!(units[0].built, 0);
        assert_eq!(units[0].served, 1.);
        assert_eq!(units[1].served, 0.5, "the unit that fits was tuned");
        assert_eq!(
            report.refused,
            [Refused {
                unit: 0,
                stage: Stage::Discovery,
                quote: priced(SECOND),
                footprint: seismic::Footprint {
                    priced: 500,
                    unpriced: 0,
                },
                refusal: Refusal::Bytes,
            }]
        );
    }

    /// Host RAM and a device with memory of its own are two domains, and
    /// room in one admits nothing in the other: with a device that could
    /// hold every unit's inputs many times over, what the units hold on the
    /// host still never passes the host's ceiling. Two units whose host
    /// bytes do not fit together are tuned one after the other, and a unit
    /// whose host bytes alone pass the ceiling is never started.
    #[test]
    fn host_bytes_are_admitted_against_the_hosts_own_ceiling() {
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                inputs: 100,
                host: 200,
                ..Synthetic::new(0, &clock, 0.5, &[0.5])
            },
            Synthetic {
                inputs: 100,
                host: 200,
                ..Synthetic::new(1, &clock, 0.3, &[0.5])
            },
            Synthetic {
                inputs: 100,
                host: 500,
                ..Synthetic::new(2, &clock, 0.2, &[0.5])
            },
        ];
        MEMORY.with(|memory| {
            memory.set(Memory {
                ceiling: Some(1_000_000),
                host_ceiling: Some(300),
                ..Memory::default()
            })
        });
        let (_, report) = run(&mut units, &clock, 600);
        let memory = MEMORY.with(Cell::take);
        assert!(memory.peak <= 300, "the device was never near its ceiling");
        assert!(memory.host_peak <= 300, "{} host bytes held", memory.host_peak);
        assert_eq!(memory.host_peak, 200, "one unit's inputs at a time");
        assert_eq!((units[0].served, units[1].served), (0.5, 0.5));
        assert_eq!(units[2].built, 0, "the unit the host cannot hold never ran");
        assert_eq!(
            report
                .refused
                .iter()
                .map(|refused| (refused.unit, refused.refusal))
                .collect::<Vec<_>>(),
            [(2, Refusal::Bytes)]
        );
    }

    /// Under the host's ceiling a piece starts when its bytes fit beside
    /// those in flight; past the ceiling nothing starts, a piece priced at
    /// nothing included, so what is held for nothing is returned first.
    #[test]
    fn nothing_starts_while_the_host_holds_more_than_its_ceiling() {
        assert!(HostRoom::Unbounded.admits(u64::MAX, u64::MAX));
        assert!(HostRoom::Free(300).admits(200, 100));
        assert!(!HostRoom::Free(300).admits(200, 101));
        assert!(HostRoom::Free(0).admits(0, 0));
        assert!(!HostRoom::Exceeded.admits(0, 0));
    }

    /// A piece whose bytes have no observed price is admitted on nothing
    /// but the room as it is, so two such pieces are never in flight
    /// together: units whose compiles would overlap compile one at a time
    /// while their pieces' bytes stay unpriced, and all their work still
    /// runs.
    #[test]
    fn a_piece_without_a_price_runs_with_no_other_such_piece_in_flight() {
        let clock = Fake::default();
        let overlap = |unobserved: bool| {
            let mut units = compiling(&clock, 3, 2, 1.);
            for unit in &mut units {
                unit.unobserved = unobserved;
            }
            MEMORY.with(|memory| {
                memory.set(Memory {
                    ceiling: Some(1_000_000),
                    ..Memory::default()
                })
            });
            let mut workers = Overlapping::new(&clock, usize::MAX);
            let (_, outcome) = allocate(
                &mut units,
                &mut Log::default(),
                &clock,
                &mut workers,
                clock.now() + SECOND * 600,
                Duration::ZERO,
            );
            MEMORY.with(Cell::take);
            outcome.expect("no synthetic work fails");
            assert!(units.iter().all(|unit| unit.measured.len() == 2));
            workers.most
        };
        assert!(overlap(false) > 1, "priced pieces' compiles overlap");
        assert_eq!(overlap(true), 1);
    }

    /// Every unit's other forms are measured before any unit refines a
    /// form, the more valuable unit's first: a low-share unit's form
    /// representative comes before the refinement of a unit worth nine
    /// times as much, and no refinement of the lesser unit precedes a form
    /// of the greater.
    #[test]
    fn forms_are_measured_by_value_before_any_refinement() {
        let clock = Fake::default();
        let mut units = [0.1, 0.6, 0.3]
            .into_iter()
            .enumerate()
            .map(|(name, share)| Synthetic {
                forms: 2,
                ..Synthetic::new(name, &clock, share, &[1.; 6])
            })
            .collect::<Vec<_>>();
        let (log, _) = run(&mut units, &clock, 600);
        let log = stages(&log);
        let measured = log
            .iter()
            .filter(|(_, stage)| matches!(stage, Stage::Form | Stage::Measure))
            .copied()
            .collect::<Vec<_>>();
        // Six form representatives, the largest share's first, then
        // refinement.
        assert_eq!(
            measured[..6],
            [1, 1, 2, 2, 0, 0].map(|unit| (unit, Stage::Form)),
            "{measured:?}"
        );
        assert!(measured[6..].iter().all(|(_, stage)| *stage == Stage::Measure));
        for (position, (unit, stage)) in measured.iter().enumerate() {
            if *stage == Stage::Measure {
                assert!(
                    measured[position..].iter().all(|(later, stage)| {
                        *stage != Stage::Form || units[*later].share <= units[*unit].share
                    }),
                    "a refinement of unit {unit} preceded a form of a unit worth more"
                );
            }
        }
    }

    /// A mix of units as a load has them: four whose launches are formed
    /// separately, one short compile a piece, and holding nearly all of
    /// step time; and one of a few percent whose programs are formed whole,
    /// a compile eight times its measurement for every piece (on CUDA two
    /// such units once spent most of a budget compiling on the device's
    /// thread while most units stopped at their defaults). Its compiles run
    /// ahead on a worker like any other, so they take none of the device's
    /// time; every unit's other forms are measured, the units worth more
    /// first, before any unit refines one; and the device is given to the
    /// lesser unit for its measurements alone.
    #[test]
    fn a_low_share_units_long_compiles_do_not_crowd_out_other_units_forms() {
        let clock = Fake::default();
        let whole = Synthetic {
            compile: SECOND * 8,
            forms: 2,
            ..Synthetic::new(0, &clock, 0.03, &[1.; 8])
        };
        let mut units = std::iter::once(whole)
            .chain((1..=4).map(|name| Synthetic {
                compile: SECOND,
                forms: 2,
                ..Synthetic::new(name, &clock, 0.2425, &[1.; 4])
            }))
            .collect::<Vec<_>>();
        let budget = 40;
        let (log, report) = run(&mut units, &clock, budget);
        assert_eq!(report.overrun(), Duration::ZERO);
        // Every unit's other forms were measured, and every unit worth
        // more refined its forms too: none stopped at its defaults.
        assert!(units.iter().all(|unit| unit.measured.len() >= unit.forms));
        assert!(units[1..].iter().all(|unit| unit.measured.len() == 4));
        // No refinement of any unit was admitted before every unit's forms
        // were: a form whose long compile is in flight is admitted, and the
        // device takes other work meanwhile rather than wait for it.
        let began = |stage: Stage| {
            report
                .admitted
                .iter()
                .filter(move |admitted| admitted.stage == stage)
                .map(|admitted| admitted.began)
        };
        assert!(began(Stage::Form).max() <= began(Stage::Measure).min());
        // Of the forms, the units worth more were measured first.
        let pieces = stages(&log)
            .into_iter()
            .filter(|(_, stage)| matches!(stage, Stage::Form | Stage::Measure))
            .collect::<Vec<_>>();
        let first_of_whole = pieces
            .iter()
            .position(|(unit, _)| *unit == 0)
            .expect("the lesser unit's form was measured");
        assert!(
            (1..=4).all(|unit| pieces[..first_of_whole]
                .iter()
                .any(|(earlier, _)| *earlier == unit)),
            "{pieces:?}"
        );
        // Every piece of the lesser unit compiled ahead for the eight
        // seconds it was quoted and held the device for its one second of
        // measurement: its compiles cost the device nothing.
        let lesser = report
            .admitted
            .iter()
            .filter(|admitted| admitted.unit == 0 && admitted.stage != Stage::Discovery)
            .collect::<Vec<_>>();
        assert!(lesser.len() >= 2);
        for admitted in &lesser {
            assert_eq!(admitted.formed, Some(SECOND * 8));
            assert_eq!(admitted.actual, SECOND);
        }
        // What the device did is every piece's own time, compiles
        // excluded: the 5 discoveries, 16 measurements of the units worth
        // more, and the lesser unit's.
        let device = report
            .admitted
            .iter()
            .fold(Duration::ZERO, |sum, admitted| sum + admitted.actual);
        assert_eq!(device, SECOND * (5 + 16 + lesser.len() as u32));
    }

    /// What one unit builds says nothing of another unit's work: a unit
    /// whose first challengers' bytes are not yet observed starts them
    /// beside a unit that made a large build, where holding each of its
    /// unobserved operations to that build would refuse it for the load.
    #[test]
    fn an_unobserved_piece_is_not_held_to_another_units_largest_build() {
        let clock = Fake::default();
        let mut units = [
            // A large build: more than is free beside it.
            Synthetic {
                inputs: 600,
                unobserved: true,
                ..Synthetic::new(0, &clock, 0.1, &[1.; 3])
            },
            Synthetic {
                inputs: 50,
                unobserved: true,
                ..Synthetic::new(1, &clock, 0.9, &[0.5, 1., 1.])
            },
        ];
        let (memory, report) = bounded(&mut units, &clock, 60, 1000);
        assert!(memory.peak <= 1000);
        assert_eq!(units[1].measured.len(), 3, "{:?}", report.refused);
        assert_eq!(units[1].served, 0.5);
        // Nothing was given up for it: both units' inputs were built once.
        assert_eq!((units[0].built, units[1].built), (1, 1));
        // The large unit's own unobserved piece is held to its own build,
        // which does not fit: it is refused for its bytes, and says so.
        assert!(units[0].measured.is_empty());
        assert_eq!(
            report
                .refused
                .iter()
                .map(|refused| (refused.unit, refused.refusal))
                .collect::<Vec<_>>(),
            [(0, Refusal::Bytes)]
        );
    }

    /// Each unit whose work changed its record owes one publication when
    /// the allocator stops. The publications are priced and work is
    /// admitted only so that they fit: the allocator stops with them
    /// settled inside the budget, where pieces admitted on their own price
    /// alone would leave them to run past it.
    #[test]
    fn publications_owed_at_the_stop_fit_the_budget_by_planning() {
        let clock = Fake::default();
        let unit = |name: usize, clock: &Fake, challengers: usize| Synthetic {
            publication: SECOND * 2,
            ..Synthetic::new(name, clock, 0.5, &vec![1.; challengers])
        };
        let mut units = [unit(0, &clock, 100), unit(1, &clock, 100), unit(2, &clock, 0)];
        // The third unit has no points to discover and no work at all.
        units[2].points = 0;
        let (log, report) = run(&mut units, &clock, 20);
        // Two discoveries, each leaving 2 s owed; then pieces of 1 s while
        // 4 s stay for the two publications: 16 s of work, 4 s of settling.
        assert_eq!(log.len(), 16);
        assert_eq!(report.settling, SECOND * 4);
        assert_eq!(report.ended, SECOND * 20);
        assert_eq!((report.unused(), report.overrun()), (Duration::ZERO, Duration::ZERO));
        // Each unit that worked was published once; the one that did not
        // owed nothing.
        assert_eq!(units.each_ref().map(|unit| unit.published), [1, 1, 0]);
        assert!(units.iter().all(|unit| !unit.changed));
        // A budget that holds one unit's discovery and its publication but
        // not the other's too starts one unit only.
        let clock = Fake::default();
        let mut units = [unit(0, &clock, 0), unit(1, &clock, 0)];
        let (log, report) = run(&mut units, &clock, 5);
        assert_eq!(stages(&log), [(0, Stage::Discovery)]);
        assert_eq!((report.ended, report.overrun()), (SECOND * 3, Duration::ZERO));
        assert_eq!(report.refused.len(), 1);
    }

    /// A unit whose next piece, with its own inputs, needs more than the
    /// ceiling is refused for the load: it gives its inputs up at once, so
    /// the next unit's inputs are never resident beside them.
    #[test]
    fn a_unit_refused_for_bytes_for_the_load_holds_nothing() {
        let clock = Fake::default();
        let mut units = [
            // Its unobserved challenger is held to its own build: 600 on
            // top of its 600 of inputs never fits 1000.
            Synthetic {
                inputs: 600,
                unobserved: true,
                ..Synthetic::new(0, &clock, 0.5, &[1.; 3])
            },
            Synthetic {
                inputs: 50,
                ..Synthetic::new(1, &clock, 0.4, &[1.; 3])
            },
            Synthetic {
                inputs: 300,
                ..Synthetic::new(2, &clock, 0.1, &[1.; 3])
            },
        ];
        let (memory, report) = bounded(&mut units, &clock, 60, 1000);
        assert!(units[0].measured.is_empty());
        assert_eq!(units[0].built, 1);
        assert_eq!(
            report
                .refused
                .iter()
                .map(|refused| (refused.unit, refused.refusal))
                .collect::<Vec<_>>(),
            [(0, Refusal::Bytes)]
        );
        // Its refusal is known when its discovery ends: no other unit's
        // inputs were ever held beside its 600.
        assert_eq!(memory.peak, 600);
        assert_eq!(memory.held, 0);
        assert!(units.iter().skip(1).all(|unit| unit.measured.len() == 3));
    }

    /// A unit refused for bytes that another unit's kept work holds keeps
    /// its inputs while that could change, and holds none when the
    /// allocator stops.
    #[test]
    fn a_unit_refused_for_bytes_holds_nothing_when_the_allocator_stops() {
        let clock = Fake::default();
        let mut units = [
            Synthetic {
                kept: 300,
                ..Synthetic::new(0, &clock, 0.5, &[])
            },
            // 400 of inputs and 400 more for its unobserved challenger fit
            // the ceiling, but not beside the other unit's 300.
            Synthetic {
                inputs: 400,
                unobserved: true,
                ..Synthetic::new(1, &clock, 0.5, &[1.; 3])
            },
        ];
        let (memory, report) = bounded(&mut units, &clock, 60, 1000);
        assert!(units[1].measured.is_empty());
        assert_eq!(
            report
                .refused
                .iter()
                .map(|refused| (refused.unit, refused.refusal))
                .collect::<Vec<_>>(),
            [(1, Refusal::Bytes)]
        );
        assert!(!units[1].resident);
        assert_eq!(memory.held, 300, "only what is not inputs stays held");
    }

    /// A unit whose remaining work no longer fits the budget gives its
    /// inputs up at once, not when the load's tuning ends.
    #[test]
    fn a_unit_with_no_work_that_can_start_holds_no_inputs() {
        let clock = Fake::default();
        let mut units = [
            // Its challengers take longer than what is left after its
            // discovery.
            Synthetic {
                inputs: 100,
                measurement: SECOND * 30,
                ..Synthetic::new(0, &clock, 0.5, &[1.; 4])
            },
            Synthetic {
                inputs: 100,
                ..Synthetic::new(1, &clock, 0.5, &[1.; 4])
            },
        ];
        let (memory, _) = bounded(&mut units, &clock, 20, 1000);
        assert!(units[0].measured.is_empty());
        assert_eq!(units[0].built, 1);
        // The two were never resident together past the first's discovery:
        // it gave its inputs up before the second unit's work held more.
        assert_eq!(memory.peak, 100);
        assert_eq!(units[1].measured.len(), 4);
    }
}
