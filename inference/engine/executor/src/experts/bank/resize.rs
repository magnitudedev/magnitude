use super::{ExpertBank, host_views, landed_records};
use crate::experts::source::Records;
use crate::planning::{ExpertBankPlan, Placer};
use crate::experts::watcher::{Candidates, MoveBudget, ShrinkPlan};
use crate::domain::{BankHolding, BankResized, BankStep, DomainError, ResizeTimes, Trip};
use crate::CompletionWake;
use std::time::{Duration, Instant};
use crate::native::BankMoves;
use magnitude_kernels::routed_move;
use seismic::{NativeTensorBatch, NativeTensorBatchCompletion, PreparedRecommit, RecommitRelease, Resizer, Tensor, TensorError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use crate::experts::watcher::Join;

/// What the bank commits: the slots its directory serves from, between the
/// floor and every expert, and its record tensors' resize. A record
/// tensor's commitment is never below its directory's records.
pub(super) struct Commitment {
    /// Written by whoever holds the resizers: a shrink on the submitting
    /// thread, a growth's slots on the resize thread as each joins.
    pub(super) slots: Arc<AtomicU64>,
    pub(super) resize: Resize,
    /// The last shrink's moves, submitted and not yet observed: the
    /// flights queued behind them read what they wrote.
    pub(super) moves: Option<NativeTensorBatchCompletion>,
}

/// A host tier's host slots committed (`ExpertBankPlan::host_records`),
/// written as [`Commitment::slots`] is, the most it holds (every expert,
/// `ExpertBankPlan::host_ceiling`), and its record tensors' resize.
pub(super) struct HostTier {
    pub(super) slots: Arc<AtomicU64>,
    pub(super) ceiling: std::num::NonZeroU64,
    pub(super) resize: HostResize,
    /// Its host slots' last finished resize
    /// ([`crate::domain::ElasticBank::record_resize`]).
    pub(super) last_resize: Option<crate::domain::BankResize>,
}

/// The resize of a host tier's record tensors, group by group in routed
/// tensor order.
pub(super) enum HostResize {
    /// No resize runs: each record tensor's resizer.
    Idle(Vec<Resizer>),
    /// A growth the domain began ([`ExpertBank::grow_host`]), which its
    /// advance reports.
    Growing(Growing),
    /// The tier backed from its floor to the plan's host slots as the bank
    /// opens ([`ExpertBank::back_host`]): its load-time size, which no
    /// domain resize began, so its end reports none. While it runs no
    /// resize of the tier begins, and a shrink stops it at its next slot.
    Backing(Growing),
    /// A resize step panicked or broke an invariant, taking the resizers
    /// with it: no resize of the tier runs again, and each asked fails
    /// with why.
    Broken(String),
}

/// A growth of a holding's record tensors from `from` slots (the bank's,
/// or a host tier's host slots), the records charged `charged` before it,
/// decided at `decided`: on the resize thread, slot by slot toward its
/// target, each backed by every tensor, installed and joined to the watcher
/// at once ([`grow_by_slots`]), so its first slots serve while later ones
/// are still backed, and nothing waits for a gap between flights. `stop`
/// ends it at its next slot.
pub(super) struct Growing {
    from: u64,
    charged: u64,
    decided: Instant,
    stop: Arc<AtomicBool>,
    done: Done<Grown>,
}

/// How a growth ended on the resize thread: each resizer back, when its
/// first and its last slot joined (none where none did), what cut it short
/// of its target, if anything did, and where its time went.
pub(super) struct Grown {
    pub(super) resizers: Vec<Resizer>,
    pub(super) joined: Option<(Instant, Instant)>,
    pub(super) cut: Cut,
    pub(super) spent: Spent,
}

/// A growth's time on the resize thread by phase, over the slots it
/// prepared: every tensor's prepare (its storage backed), and of it what
/// Seismic's backing spent by phase (`seismic::BackingTimes`, process-wide,
/// so beside it any other backing then), every install (with the wait for
/// the bank's charge lock), every release, and the slot's join posted.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Spent {
    slots: u64,
    prepare: Duration,
    backing: seismic::BackingTimes,
    install: Duration,
    release: Duration,
    join: Duration,
}

impl std::fmt::Display for Spent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mean = |phase: Duration| phase.as_secs_f64() * 1e3 / self.slots.max(1) as f64;
        let backing = self.backing;
        write!(
            formatter,
            "a slot's mean over {}: prepare {:.1} ms (fault {:.1}, collapse {:.1}, pin {:.1}, zero {:.1}), install {:.3} ms, release {:.3} ms, join {:.3} ms",
            self.slots,
            mean(self.prepare),
            mean(backing.fault),
            mean(backing.collapse),
            mean(backing.pin),
            mean(backing.zero),
            mean(self.install),
            mean(self.release),
            mean(self.join),
        )
    }
}

/// What ended a growth short of its target.
pub(super) enum Cut {
    /// Nothing: it reached its target, or was stopped (`Growing::stop`).
    None,
    /// A tensor's storage refused its next slot, which joined nothing (a
    /// tensor that installed it backs more than the directory uses, which
    /// it never reads).
    Refused(TensorError),
    /// Its next slot's join could not be formed: a broken invariant.
    Unjoined(String),
}

/// What a growth's slots join as each is backed (`watcher::Join`): the
/// bank's records, which the device places into, or records the host lands
/// into, addressed by their views.
#[derive(Clone, Copy)]
enum Joining {
    Bank,
    Landed,
}

/// A holding's records of group `group` at `slots` slots: the bank's
/// (`ExpertBankPlan::records_at`) or a host tier's
/// (`ExpertBankPlan::host_records`).
type RecordsAt = fn(&ExpertBankPlan, usize, u64) -> u64;

/// Grow `resizers` from `from` to `to` slots, one slot at a time unless
/// `stop` is set: each tensor prepares the slot's `rows`, then installs
/// them (a growth installs beside submission,
/// `seismic::PreparedRecommit::install`), then each install's release
/// returns once the device work submitted before it completed, so no work
/// that could not reach the slot's rows (on Metal, committed before its
/// chunks joined the queue's residency set) can be handed them; only then
/// `join` forms the slot's join, `post` hands it to the watcher
/// (`Shared::join`, taken at its next pass), and `slots` holds it. A
/// refusal ends it at the slots joined before. The installs, where the
/// tensors' charges rise, hold `rises` ([`ExpertBank::hold_rises`]). Each
/// slot begins by `begin`, held until it is joined, none where it gives
/// none (the load's traffic is held, `watcher::Shared::backing_slot`).
pub(super) fn grow_by_slots<S>(
    mut resizers: Vec<Resizer>,
    (from, to): (u64, u64),
    rows: impl Fn(u64) -> Vec<u64>,
    join: impl Fn(u64) -> Result<Join, String>,
    post: impl Fn(Join),
    (slots, rises): (&AtomicU64, &Mutex<()>),
    stop: &AtomicBool,
    begin: impl Fn() -> Option<S>,
) -> Grown {
    let mut joined: Option<(Instant, Instant)> = None;
    let mut spent = Spent::default();
    for at in from + 1..=to {
        // Each slot is backed only as the load's traffic may run
        // (`watcher::Shared::backing_slot`), until it is joined.
        let Some(_slot) = begin().filter(|_| !stop.load(Ordering::Acquire)) else {
            break;
        };
        spent.slots += 1;
        let (preparing, backed) = (Instant::now(), seismic::backing_times());
        let prepared = prepare(resizers, rows(at));
        spent.prepare += preparing.elapsed();
        spent.backing += seismic::backing_times().since(backed);
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err((resizers, refusal)) => return Grown { resizers, joined, cut: Cut::Refused(refusal), spent },
        };
        let mut refusal = None;
        // Every tensor installs, then the releases wait together.
        let installing = Instant::now();
        let installed = {
            let _rises = rises.lock().expect("the bank's charge rises");
            prepared
                .into_iter()
                .map(|prepared| match prepared.install() {
                    Ok(release) => Installed::Release(release),
                    Err(refused) => {
                        refusal.get_or_insert(refused.error);
                        Installed::Kept(refused.resizer)
                    }
                })
                .collect::<Vec<_>>()
        };
        let releasing = Instant::now();
        spent.install += releasing - installing;
        resizers = installed.into_iter().map(Installed::release).collect();
        spent.release += releasing.elapsed();
        if let Some(refusal) = refusal {
            return Grown { resizers, joined, cut: Cut::Refused(refusal), spent };
        }
        let joining = Instant::now();
        match join(at) {
            Ok(join) => post(join),
            Err(reason) => return Grown { resizers, joined, cut: Cut::Unjoined(reason), spent },
        }
        spent.join += joining.elapsed();
        slots.store(at, Ordering::Release);
        let now = Instant::now();
        joined = Some((joined.map_or(now, |(first, _)| first), now));
    }
    Grown { resizers, joined, cut: Cut::None, spent }
}

impl Growing {
    /// Its end, once the resize thread finished it: the resizers back,
    /// what it came to for `holding` at `slots` slots now, its records
    /// charged `charged` now, and where its time went; or why the holding is broken (its step
    /// panicked, a slot backed could not join, or its charge is unread).
    fn ended(
        &self,
        holding: BankHolding,
        slots: u64,
        charged: impl FnOnce() -> Result<u64, DomainError>,
    ) -> Option<Result<(Vec<Resizer>, BankResized, Spent), String>> {
        let (outcome, released) = self.done.lock().expect("bank growth outcome").take()?;
        let Grown { resizers, joined, cut, spent } = match outcome {
            Ok(grown) => grown,
            Err(panicked) => return Some(Err(panicked)),
        };
        let refused = match cut {
            Cut::None => None,
            Cut::Refused(refusal) => Some(refusal),
            Cut::Unjoined(reason) => return Some(Err(reason)),
        };
        // No slot joined: refused at its first, or stopped before it.
        let Some((first, last)) = joined else {
            return Some(Ok(match refused {
                Some(refusal) => (resizers, BankResized::Refused { holding, at: slots, refusal }, spent),
                None => (resizers, BankResized::Stopped { holding, at: slots }, spent),
            }));
        };
        let grown = match charged().and_then(|after| {
            after.checked_sub(self.charged).ok_or_else(|| DomainError::invariant("a growth lowered its records' charge"))
        }) {
            Ok(grown) => grown,
            Err(error) => return Some(Err(error.to_string())),
        };
        let zero = Duration::ZERO;
        let begun = Begun { decided: self.decided, queued: zero, planned: zero, trips: Default::default(), moves: zero };
        let times = begun.times(first, last, released);
        Some(Ok((resizers, BankResized::Grown { holding, from: self.from, to: slots, grown, refused, times }, spent)))
    }
}

/// A resize step's outcome once it ends off the submitting thread, or why it
/// panicked (which leaves the holding without the resizers it took), and
/// when.
pub(super) type Done<T> = Arc<Mutex<Option<(Result<T, String>, Instant)>>>;

/// The resize of every record tensor, group by group in routed tensor
/// order, one at a time.
pub(super) enum Resize {
    /// No resize runs: each record tensor's resizer.
    Idle(Vec<Resizer>),
    /// A shrink from `from` slots to the commitment's, its storage prepared
    /// off the submitting thread, the records charged `charged` before it,
    /// which installs at the next boundary.
    Preparing {
        from: u64,
        charged: u64,
        begun: Begun,
        done: Done<Result<Vec<PreparedRecommit>, (Vec<Resizer>, TensorError)>>,
    },
    /// A shrink installed (`refusal` where a tensor refused its install,
    /// keeping its storage), its surplus released off the submitting thread
    /// once the device work before its install completes.
    Releasing {
        from: u64,
        charged: u64,
        refusal: Option<TensorError>,
        begun: Begun,
        prepared: Instant,
        installed: Instant,
        done: Done<Vec<Resizer>>,
    },
    Growing(Growing),
    /// A resize step panicked or broke an invariant, taking the resizers
    /// with it: no resize of the records runs again, and each asked fails
    /// with why.
    Broken(String),
}

/// How a resize began on the submitting thread (`ResizeTimes`).
#[derive(Clone, Copy)]
pub(super) struct Begun {
    decided: Instant,
    queued: Duration,
    planned: Duration,
    trips: [Trip; 2],
    moves: Duration,
}

impl Begun {
    fn times(self, prepared: Instant, installed: Instant, released: Instant) -> ResizeTimes {
        let Self { decided, queued, planned, trips, moves } = self;
        ResizeTimes { decided, queued, planned, trips, moves, prepared, installed, released }
    }
}

/// A record tensor's resize, installed or refused at its install.
enum Installed {
    Release(RecommitRelease),
    Kept(Resizer),
}

impl Installed {
    fn release(self) -> Resizer {
        match self {
            Self::Release(release) => release.release(),
            Self::Kept(resizer) => resizer,
        }
    }
}

/// The thread a bank's resizes do their storage work on, started with the
/// bank and joined when it drops. A resize only hands it work: the kernel
/// creates a thread from memory it may wait for, and memory is short exactly
/// when a shrink is due. A 72-slot shrink of Qwen3.6-35B-A3B under 8 GB of
/// pressure held the submitting thread 483.9 ms, all but its 3.5 ms plan
/// and moves in the spawn of its prepare's thread (the prepare of a shrink
/// backs, zeroes and forms nothing; it was done 483.4 ms after the
/// decision), M4 Pro, 2026-10-27.
pub(super) struct ResizeThread {
    work: Option<std::sync::mpsc::Sender<Work>>,
    /// Its one thread, busy beside decode while a resize or the host tier's
    /// backing ([`HostResize::Backing`]) runs ([`RESIZE_THREADS`]).
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A step of a resize, run on the resize thread, which catches its panic.
type Work = Box<dyn FnOnce() + Send>;

impl ResizeThread {
    /// Its thread, placed on every CPU the process may run on (`seismic::pin`,
    /// which also keeps it out of the serving executor's placement,
    /// `HostCompute::serving`): a host tier's backing first touches its
    /// pages on the threads it starts, which inherit that placement, so the
    /// tier is spread over every memory node. On the runtime's CPUs alone
    /// (nodes 2 and 3 of an EPYC 7313) the tier lay on half the memory channels
    /// and decode measured 45% slower than unplaced (one alternated
    /// pair); spread, 7.6%; on the workers' own nodes, 21.9 ms
    /// against 11.9.
    pub(super) fn start() -> Result<Self, String> {
        let every = seismic::Processors::discover().map_err(|error| format!("the host's processors: {error}"))?.allowed().iter().copied().collect::<Vec<_>>();
        let (work, received) = std::sync::mpsc::channel::<Work>();
        let (placed, placement) = std::sync::mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("expert-bank-resize".into())
            .spawn(move || {
                let pinned = seismic::pin(&every);
                let serves = pinned.is_ok();
                placed.send(pinned).expect("the resize thread's start waits for its placement");
                if serves {
                    received.iter().for_each(|work| work());
                }
            })
            .map_err(|error| format!("the expert bank's resize thread: {error}"))?;
        placement.recv().map_err(|error| format!("the expert bank's resize thread: {error}"))?.map_err(|error| format!("the expert bank's resize thread's placement: {error}"))?;
        Ok(Self {
            work: Some(work),
            thread: Some(thread),
        })
    }

    /// Run `work` on the resize thread: its outcome lands in the returned
    /// slot, or why it panicked (a panic is caught, so the thread, and every
    /// later resize of the bank, outlives it), then `wake` completes. It
    /// cannot fail: the thread takes work until the bank drops its sender,
    /// so the work, and every resizer it owns, always reaches it.
    pub(super) fn run<T: Send + 'static>(&self, wake: CompletionWake, work: impl FnOnce() -> T + Send + 'static) -> Done<T> {
        let done = Arc::new(Mutex::new(None));
        let slot = done.clone();
        self.work
            .as_ref()
            .expect("the resize thread takes work until the bank drops")
            .send(Box::new(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .map_err(|payload| format!("the expert bank's resize panicked: {}", super::watcher::panic_message(payload.as_ref())));
                *slot.lock().expect("bank resize outcome") = Some((outcome, Instant::now()));
                wake.complete();
            }))
            .expect("the resize thread catches every panic, so it takes work until the bank drops");
        done
    }
}

impl Drop for ResizeThread {
    fn drop(&mut self) {
        drop(self.work.take());
        if let Some(thread) = self.thread.take() {
            thread.join().expect("the expert bank's resize thread panicked");
        }
    }
}

/// Prepare each resizer's tensor at its `rows`, in order: on a refusal
/// every prepared one is abandoned and each resizer returned, with why.
fn prepare(resizers: Vec<Resizer>, rows: Vec<u64>) -> Result<Vec<PreparedRecommit>, (Vec<Resizer>, TensorError)> {
    let mut prepared = Vec::with_capacity(resizers.len());
    let mut each = resizers.into_iter().zip(rows);
    while let Some((resizer, rows)) = each.next() {
        match resizer.prepare(rows) {
            Ok(next) => prepared.push(next),
            Err(refusal) => {
                let back = prepared
                    .into_iter()
                    .map(PreparedRecommit::abandon)
                    .chain([refusal.resizer])
                    .chain(each.map(|(resizer, _)| resizer))
                    .collect();
                return Err((back, refusal.error));
            }
        }
    }
    Ok(prepared)
}

impl ExpertBank {
    /// A host tier's record tensors, group by group; none without a tier.
    fn host_tensors(&self) -> impl Iterator<Item = (usize, &Tensor)> {
        self.layout
            .host
            .iter()
            .flatten()
            .enumerate()
            .flat_map(|(group, tier)| tier.iter().map(move |(_, tensor)| (group, tensor)))
    }

    /// The host tier's record tensors' charge, as Seismic measures it.
    pub(super) fn host_charged(&self) -> Result<u64, DomainError> {
        self.host_tensors().try_fold(0u64, |bytes, (_, tensor)| {
            tensor
                .observe_storage()
                .charged_bytes()
                .map(|charged| bytes + charged)
                .ok_or_else(|| DomainError::invariant("a held allocation is charged"))
        })
    }

    /// Each group's records the host lands into at `slots` (a host tier's
    /// host slots, or the one store's slots) as the host addresses them
    /// ([`host_views`]).
    fn host_views(&self, slots: u64) -> Result<Vec<Vec<Records>>, String> {
        let tensors = self.landed.as_ref().ok_or("host records of a bank the host lands into none of")?;
        host_views(&self.plan, tensors, |group| landed_records(&self.plan, group, slots))
    }

    /// The host tier's record tensors' groups, in tensor order.
    fn host_groups(&self) -> Vec<usize> {
        self.host_tensors().map(|(group, _)| group).collect()
    }

    /// The host tier's record tensors' rows at `slots` host slots, in
    /// tensor order.
    fn host_rows(&self, slots: u64) -> Vec<u64> {
        self.host_groups().into_iter().map(|group| self.plan.host_records(group, slots)).collect()
    }

    /// Each group's host tier records at `slots` host slots.
    fn host_records(&self, slots: u64) -> Result<Vec<u32>, DomainError> {
        (0..self.plan.groups.len())
            .map(|group| {
                u32::try_from(self.plan.host_records(group, slots)).map_err(|_| DomainError::invariant("host records exceed u32"))
            })
            .collect()
    }

    /// The host tier's resizers, taken while no resize of it runs (none
    /// while one does): the one token a resize of it holds until it returns
    /// them; why it is broken, if it is.
    fn idle_host(tier: &mut HostTier) -> Result<Option<Vec<Resizer>>, DomainError> {
        match std::mem::replace(&mut tier.resize, HostResize::Idle(Vec::new())) {
            HostResize::Idle(resizers) => Ok(Some(resizers)),
            HostResize::Broken(reason) => {
                tier.resize = HostResize::Broken(reason.clone());
                Err(DomainError::invariant(reason))
            }
            running => {
                tier.resize = running;
                Ok(None)
            }
        }
    }

    /// Begin growing the host tier to `to` host slots on the resize thread,
    /// each host slot joining the tier as soon as every record tensor backs
    /// it ([`Self::growing`]); while a resize of it runs, none begins.
    pub(super) fn grow_host(&self, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        Ok(match self.host_growth(to, wake, HostResize::Growing)? {
            true => BankStep::Begun,
            false => BankStep::Running,
        })
    }

    /// Back the host tier from the floor it opens at to the plan's host
    /// slots, its load-time size (`ExpertBankPlan::host_slots`), on the
    /// resize thread: each host slot joins the tier, and its fill, as soon
    /// as every record tensor backs it, so the load waits for none of it
    /// (backing tens of gigabytes of pinned host storage before the bank
    /// opened held a 6.89 GB load of the 3090 18-26 s, its import 25-33 s
    /// against 5-10 s at the floor, 2026-11-01). It is the open's, no
    /// resize: its end reports none ([`HostResize::Backing`]).
    pub(super) fn back_host(&self) -> Result<(), DomainError> {
        self.host_growth(self.plan.host_slots, CompletionWake::new(|| {}), HostResize::Backing).map(|_| ())
    }

    /// Begin growing the host tier to `to` host slots, as `resize` names
    /// the growth; false while a resize of it runs, which begins none.
    fn host_growth(&self, to: u64, wake: CompletionWake, resize: fn(Growing) -> HostResize) -> Result<bool, DomainError> {
        let host = self.host.as_ref().ok_or_else(|| DomainError::invariant("a host tier growth without one"))?;
        let mut tier = host.lock().expect("host tier");
        let charged = self.host_charged()?;
        let Some(resizers) = Self::idle_host(&mut tier)? else {
            return Ok(false);
        };
        let records: RecordsAt = ExpertBankPlan::host_records;
        let tensors = (self.host_groups(), records);
        match self.growing(resizers, (&tier.slots, to), tensors, Joining::Landed, charged, wake) {
            Ok(growing) => tier.resize = resize(growing),
            Err((resizers, error)) => {
                tier.resize = HostResize::Idle(resizers);
                return Err(error);
            }
        }
        Ok(true)
    }

    /// Begin growing a holding's record tensors (`resizers`, each of group
    /// `groups[i]`, at `records`) from its `slots` now to `to` on the resize
    /// thread ([`grow_by_slots`]), its records charged `charged` now, each
    /// slot joining the watcher as `joining` says as soon as every tensor
    /// backs it; `wake` completed once it ends. Refused, the resizers
    /// come back.
    fn growing(
        &self,
        resizers: Vec<Resizer>,
        (slots, to): (&Arc<AtomicU64>, u64),
        (groups, records): (Vec<usize>, RecordsAt),
        joining: Joining,
        charged: u64,
        wake: CompletionWake,
    ) -> Result<Growing, (Vec<Resizer>, DomainError)> {
        let landed = match (joining, &self.landed) {
            (Joining::Bank, _) => None,
            (Joining::Landed, Some(landed)) => Some(landed.clone()),
            (Joining::Landed, None) => {
                return Err((resizers, DomainError::invariant("a landed growth of a bank the host lands into none of")));
            }
        };
        let (decided, from) = (Instant::now(), slots.load(Ordering::Acquire));
        let plan = Arc::new(self.plan.clone());
        let (shared, slots, stop) = (self.watcher.shared().clone(), slots.clone(), Arc::new(AtomicBool::new(false)));
        // The records the host lands into grow from this decision until the
        // thread's last slot joined: the tier fills until then.
        let lands = landed.is_some();
        if lands {
            shared.join(Join::Growing);
        }
        let (stopping, rises) = (stop.clone(), self.rises.clone());
        let join = {
            let plan = plan.clone();
            move |at: u64| -> Result<Join, String> {
                let counts = (0..plan.groups.len()).map(|group| records(&plan, group, at));
                match &landed {
                    None => Ok(Join::Bank {
                        records: counts
                            .map(|count| usize::try_from(count).map_err(|_| "bank records exceed the host"))
                            .collect::<Result<_, _>>()?,
                    }),
                    Some(tensors) => Ok(Join::Landed {
                        records: counts
                            .map(|count| u32::try_from(count).map_err(|_| "landed records exceed u32"))
                            .collect::<Result<_, _>>()?,
                        views: host_views(&plan, tensors, |group| records(&plan, group, at))?,
                    }),
                }
            }
        };
        let done = self.resize_thread.run(wake, move || {
            let rows = |at| groups.iter().map(|&group| records(&plan, group, at)).collect();
            // The growth ends however its slots end, a panic among them too
            // (re-raised to the thread's catch): no slot joins after.
            let grown = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grow_by_slots(resizers, (from, to), rows, join, |join| shared.join(join), (&slots, &rises), &stopping, || shared.backing_slot(&stopping))
            }));
            if lands {
                shared.join(Join::Grown);
            }
            grown.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        });
        Ok(Growing { from, charged, decided, stop, done })
    }

    /// End a running growth at its next slot ([`Growing::stop`]): a shrink
    /// waits for it no longer than that slot's backing. The host tier's
    /// growth or backing ends only where `host`: host memory needs it, or
    /// a measurement's claim beside the load, whose hold would leave its
    /// slots waiting with the resize thread, and every shrink behind them.
    pub(super) fn stop_growth(&self, host: bool) {
        if let Resize::Growing(growing) = &self.commitment().resize {
            growing.stop.store(true, Ordering::Release);
        }
        if let Some(host) = self.host.as_ref().filter(|_| host) {
            if let HostResize::Growing(growing) | HostResize::Backing(growing) = &host.lock().expect("host tier").resize {
                growing.stop.store(true, Ordering::Release);
            }
        }
        // A growth waiting between its slots on a hold looks again.
        self.watcher.shared().events().notify();
    }

    /// Finish a host tier's growth once its resize thread ended it: its
    /// outcome (grown to the host slots its slots reached, or refused
    /// before any joined), none of its backing at open, which no domain
    /// resize began.
    pub(super) fn advance_host(&self) -> Result<Option<BankResized>, DomainError> {
        let Some(host) = self.host.as_ref() else {
            return Ok(None);
        };
        let mut tier = host.lock().expect("host tier");
        let (HostResize::Growing(growing) | HostResize::Backing(growing)) = &tier.resize else {
            return Ok(None);
        };
        let slots = tier.slots.load(Ordering::Acquire);
        let Some(ended) = growing.ended(BankHolding::Host, slots, || self.host_charged()) else {
            return Ok(None);
        };
        let reported = matches!(tier.resize, HostResize::Growing(_));
        match ended {
            Ok((resizers, resized, spent)) => {
                let kind = if reported { "growth" } else { "backing" };
                let (planned, ceiling) = (self.plan.host_slots, tier.ceiling);
                tracing::info!("the host tier's {kind} ended at {slots} of {planned} planned host slots ({ceiling} every expert), {spent}: {resized:?}");
                tier.resize = HostResize::Idle(resizers);
                Ok(reported.then_some(resized))
            }
            Err(reason) => {
                tier.resize = HostResize::Broken(reason.clone());
                Err(DomainError::invariant(reason))
            }
        }
    }

    /// Whether a host tier's growth runs, until its advance finishes it.
    pub(super) fn host_growing(&self) -> bool {
        self.host.as_ref().is_some_and(|host| matches!(host.lock().expect("host tier").resize, HostResize::Growing(_)))
    }

    /// Whether the host tier's backing to its planned slots runs
    /// ([`Self::back_host`]), until its advance finishes it.
    pub(super) fn host_backing(&self) -> bool {
        self.host.as_ref().is_some_and(|host| matches!(host.lock().expect("host tier").resize, HostResize::Backing(_)))
    }

    /// Shrink the host tier to `to` host slots now, between flights: the
    /// watcher keeps each group's most valuable records below its new
    /// records (their bytes copied there), then each tensor releases the
    /// rest. Its size before and after, and Seismic's charge of it before
    /// and after, or the refusal of a tensor's storage (which keeps more
    /// backed than the directory uses, which it never reads); none while a
    /// growth of it runs.
    pub(super) fn shrink_host(&self, to: u64) -> Result<Option<LandedShrink>, DomainError> {
        let host = self.host.as_ref().ok_or_else(|| DomainError::invariant("a host tier resize without one"))?;
        let mut tier = host.lock().expect("host tier");
        // Everything that can fail before the resizers are taken; an ask
        // the watcher fails returns them, the storage as it was.
        let (before, records) = (self.host_charged()?, self.host_records(to)?);
        // The records committed after the shrink, which the watcher's tier
        // tensors address from then on: its moves copy within the views it
        // held, which it drops before it answers, so nothing pins what the
        // recommit then releases.
        let views = self.host_views(to).map_err(DomainError::invariant)?;
        let Some(resizers) = Self::idle_host(&mut tier)? else {
            return Ok(None);
        };
        let from = tier.slots.load(Ordering::Acquire);
        if let Err(reason) = self.watcher.shared().shrink_host(records, views) {
            tier.resize = HostResize::Idle(resizers);
            return Err(DomainError::invariant(reason));
        }
        tier.slots.store(to, Ordering::Release);
        let (resizers, refusal) = recommit(resizers, self.host_rows(to));
        tier.resize = HostResize::Idle(resizers);
        Ok(Some(LandedShrink { from, to, before, after: self.host_charged()?, refusal }))
    }

    /// Shrink the one store (where the host places) to `to` slots now,
    /// between flights: the watcher keeps each group's most valuable
    /// records below its new records (their bytes copied by the host within
    /// the views it holds), then addresses the records at `to` by their
    /// views, and each record tensor releases the rest. The device moves
    /// nothing. Its slots before and after, Seismic's charge of the records
    /// before and after, or the refusal of a tensor's storage (which keeps
    /// what it holds, more backed than the directory uses); none while its
    /// records resize.
    pub(super) fn shrink_store(&self, to: u64) -> Result<Option<LandedShrink>, DomainError> {
        let mut commitment = self.commitment();
        // Everything that can fail before the resizers are taken; an ask
        // the watcher fails returns them, the storage as it was.
        let before = self.charged()?;
        let records = (0..self.plan.groups.len())
            .map(|group| {
                u32::try_from(self.plan.records_at(group, to)).map_err(|_| DomainError::invariant("bank records exceed u32"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let views = self.host_views(to).map_err(DomainError::invariant)?;
        let Some(resizers) = Self::idle(&mut commitment)? else {
            return Ok(None);
        };
        let from = commitment.slots.load(Ordering::Acquire);
        if let Err(reason) = self.watcher.shared().shrink_host(records, views) {
            commitment.resize = Resize::Idle(resizers);
            return Err(DomainError::invariant(reason));
        }
        commitment.slots.store(to, Ordering::Release);
        let (resizers, refusal) = recommit(resizers, self.rows_at(to));
        commitment.resize = Resize::Idle(resizers);
        drop(commitment);
        Ok(Some(LandedShrink { from, to, before, after: self.charged()?, refusal }))
    }

    /// Each group's records at `slots` slots.
    fn committed_records(&self, slots: u64) -> Result<Vec<usize>, DomainError> {
        (0..self.plan.groups.len())
            .map(|group| {
                usize::try_from(self.plan.records_at(group, slots))
                    .map_err(|_| DomainError::invariant("bank records exceed the host"))
            })
            .collect()
    }

    /// The rows each record tensor commits at `slots` slots, in resizer
    /// order.
    fn rows_at(&self, slots: u64) -> Vec<u64> {
        self.record_groups().into_iter().map(|group| self.plan.records_at(group, slots)).collect()
    }

    /// Each record tensor's group, in resizer order.
    fn record_groups(&self) -> Vec<usize> {
        (0..self.plan.groups.len())
            .flat_map(|group| std::iter::repeat_n(group, self.plan.groups[group].record.form().tensors().len()))
            .collect()
    }

    /// Every record tensor's charge, as Seismic measures it: the one reading
    /// a resize is credited by.
    fn charged(&self) -> Result<u64, DomainError> {
        self.layout
            .groups
            .iter()
            .flat_map(|group| group.records.iter().map(|(_, tensor)| tensor))
            .try_fold(0u64, |bytes, tensor| {
                tensor
                    .observe_storage()
                    .charged_bytes()
                    .map(|charged| bytes + charged)
                    .ok_or_else(|| DomainError::invariant("a held allocation is charged"))
            })
    }

    /// The resizers, taken while no resize runs (none while one does); why
    /// the records are broken, if they are.
    fn idle(commitment: &mut Commitment) -> Result<Option<Vec<Resizer>>, DomainError> {
        match std::mem::replace(&mut commitment.resize, Resize::Idle(Vec::new())) {
            Resize::Idle(resizers) => Ok(Some(resizers)),
            Resize::Broken(reason) => {
                commitment.resize = Resize::Broken(reason.clone());
                Err(DomainError::invariant(reason))
            }
            running => {
                commitment.resize = running;
                Ok(None)
            }
        }
    }

    /// Begin a shrink from `from` to `to` slots, the records charged
    /// `charged` now: each record tensor's storage at `to` prepared off this
    /// thread.
    fn prepare_off_thread(&self, resizers: Vec<Resizer>, (from, to): (u64, u64), charged: u64, begun: Begun, wake: CompletionWake) -> Resize {
        let rows = self.rows_at(to);
        let done = self.resize_thread.run(wake, move || prepare(resizers, rows));
        Resize::Preparing { from, charged, begun, done }
    }

    /// Submit `plan`'s moves to the device in a submission of their own: each
    /// group's records, by its move list. The device runs
    /// every later flight after them, in queue order, so no wait is needed
    /// before the plan commits; a submission refused leaves it uncommitted.
    fn carry_moves(
        &self,
        plan: &ShrinkPlan,
        kernels: &[BankMoves],
    ) -> Result<Option<NativeTensorBatchCompletion>, DomainError> {
        let failed = |error: &dyn std::fmt::Display| DomainError::invariant(format!("expert bank moves: {error}"));
        let mut batch = None;
        let groups = self.layout.groups.iter().zip(plan.moves()).zip(&self.move_lists).zip(kernels);
        for (((group, moves), list), kernels) in groups {
            if moves.is_empty() {
                continue;
            }
            let pairs = moves
                .iter()
                .map(|&(from, to)| Ok((i32::try_from(from)?, i32::try_from(to)?)))
                .collect::<Result<Vec<_>, std::num::TryFromIntError>>()
                .map_err(|error| failed(&error))?;
            // The records every one of the group's record tensors holds.
            let records = group.records.iter().map(|(_, tensor)| tensor.extents()[0]).min().expect("a group has record tensors");
            list.write_list(&pairs, crate::experts::ListBounds { from: records, to: records });
            let batch = batch.get_or_insert_with(|| NativeTensorBatch::new(&group.moves.device()));
            for ((_, tensor), (_, kernel)) in group.records.iter().zip(kernels.records.iter()) {
                let mut tensor = tensor.clone();
                batch
                    .push(kernel, routed_move::Args { records: &mut tensor, list: &group.moves }, routed_move::OutputArgs {})
                    .map_err(|error| failed(&error))?;
            }
        }
        batch.map(|batch| batch.submit().map_err(|error| failed(&error))).transpose()
    }

    /// [`ElasticBank::carried`].
    pub(super) fn carried(&self) -> Result<(), DomainError> {
        Self::observe_moves(&mut self.commitment())
    }

    /// Observe the last shrink's moves: their device failure, if they failed.
    fn observe_moves(commitment: &mut Commitment) -> Result<(), DomainError> {
        let failed = |error: seismic::CallError| {
            DomainError::Device(crate::DeviceError::Execution(format!("expert bank moves: {error}")))
        };
        commitment.moves.take().map_or(Ok(()), |moves| moves.wait().map_err(failed))
    }

    /// The device bytes a growth to `to` slots backs, what it claims before
    /// it: each record tensor's charge at `to` slots past its charge at the
    /// slots now (`ExpertBankPlan::resident_charge`), what its reservation backs anew
    /// (the record tables formed over it are held already); no fewer at
    /// more slots.
    pub fn growth_bytes(&self, to: u64) -> Result<u64, String> {
        Ok(self.plan.resident_charge(to)? - self.plan.resident_charge(self.slots())?)
    }

    /// [`ElasticBank::shrink`] from the nearest slots that release storage,
    /// `shallowest`, toward `deepest`, those covering the deficit
    /// (`ServedBank::releasing`), in a gap between flights, decided at
    /// `decided`, its moves carried by `kernels`. The plan's
    /// evictions are published first; nothing is mapped to a record a move
    /// writes until the moves are queued ahead of every later step and the
    /// plan is committed. A plan dropped on an error before then leaves
    /// every kept expert where its bytes are; once queued, the moves' device
    /// failure fails every flight after them ([`ElasticBank::carried`]), so
    /// nothing read from a record they failed to write is taken. With a
    /// `budget`, the device seconds one resize's moves may take, the shrink
    /// takes the most slots toward the deficit whose planned moves fit it, at least
    /// `shallowest`, and a later boundary takes the rest; without one, the whole
    /// deficit.
    pub(super) fn shrink(
        &self,
        kernels: &[BankMoves],
        (shallowest, deepest): (u64, u64),
        budget: Option<MoveBudget>,
        decided: Instant,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        let mut commitment = self.commitment();
        let charged = self.charged()?;
        let Some(resizers) = Self::idle(&mut commitment)? else {
            return Ok(BankStep::Running);
        };
        let from = commitment.slots.load(Ordering::Acquire);
        if from <= self.plan.floor() {
            commitment.resize = Resize::Idle(resizers);
            return Ok(BankStep::AtLimit);
        }
        let shared = self.watcher.shared();
        let planned = (|| {
            Self::observe_moves(&mut commitment)?;
            let queued = decided.elapsed();
            let count = |slots: u64| usize::try_from(slots).map_err(|_| DomainError::invariant("bank slots exceed the host"));
            // Every commitment from the nearest that releases storage to the
            // deficit's: the budget takes the deepest whose moves fit,
            // forming the records only of those it prices. A group's records
            // grow with the slots, so the shallowest candidate's fitting the
            // host fits them all.
            self.committed_records(shallowest)?;
            let plan = Arc::new(self.plan.clone());
            let candidates = Candidates {
                shallowest: count(shallowest)?,
                deepest: count(deepest)?,
                records: Box::new(move |slots| {
                    (0..plan.groups.len()).map(|group| plan.records_at(group, slots as u64) as usize).collect()
                }),
            };
            let plan = shared.plan_shrink(candidates, budget).map_err(DomainError::invariant)?;
            let to = plan.slots() as u64;
            let moving = Instant::now();
            commitment.moves = self.carry_moves(&plan, kernels)?;
            let moves = moving.elapsed();
            let trips = [plan.trip(), shared.commit_shrink(plan).map_err(DomainError::invariant)?];
            let planned = decided.elapsed() - queued - moves;
            Ok((to, Begun { decided, queued, planned, trips, moves }))
        })();
        let (to, begun) = match planned {
            Ok(planned) => planned,
            Err(error) => {
                commitment.resize = Resize::Idle(resizers);
                return Err(error);
            }
        };
        commitment.slots.store(to, Ordering::Release);
        commitment.resize = self.prepare_off_thread(resizers, (from, to), charged, begun, wake);
        Ok(BankStep::Begun)
    }

    /// [`ElasticBank::grow`] to `to` slots, its growth target, on the resize
    /// thread, each slot joining the bank as soon as every record tensor
    /// backs it ([`Self::growing`]): the device's records where it places,
    /// the one store's landed records where the host does.
    pub(super) fn grow(&self, to: u64, wake: CompletionWake) -> Result<BankStep, DomainError> {
        let mut commitment = self.commitment();
        let charged = self.charged()?;
        let Some(resizers) = Self::idle(&mut commitment)? else {
            return Ok(BankStep::Running);
        };
        let joining = match self.plan.placer() {
            Placer::Tier => Joining::Bank,
            Placer::Bank => Joining::Landed,
        };
        let records: RecordsAt = ExpertBankPlan::records_at;
        let tensors = (self.record_groups(), records);
        match self.growing(resizers, (&commitment.slots, to), tensors, joining, charged, wake) {
            Ok(growing) => commitment.resize = Resize::Growing(growing),
            Err((resizers, error)) => {
                commitment.resize = Resize::Idle(resizers);
                return Err(error);
            }
        }
        Ok(BankStep::Begun)
    }

    /// [`ElasticBank::advance`]: a shrink's prepared storage installs at
    /// this boundary and its release is taken back at a later one; a growth
    /// finishes once its resize thread ended it.
    pub(super) fn advance(&self, wake: CompletionWake) -> Result<Option<BankResized>, DomainError> {
        fn take<T>(done: &Done<T>) -> Option<(Result<T, String>, Instant)> {
            done.lock().expect("bank resize outcome").take()
        }
        let mut commitment = self.commitment();
        let slots = commitment.slots.load(Ordering::Acquire);
        // A step that panicked or broke an invariant took the resizers: the
        // records are broken, and the error says why.
        let broken = |commitment: &mut Commitment, reason: String| {
            commitment.resize = Resize::Broken(reason.clone());
            Err(DomainError::invariant(reason))
        };
        let (resize, finished) = match std::mem::replace(&mut commitment.resize, Resize::Idle(Vec::new())) {
            Resize::Preparing { from, charged, begun, done } => match take(&done) {
                None => (Resize::Preparing { from, charged, begun, done }, None),
                Some((Err(panicked), _)) => return broken(&mut commitment, panicked),
                Some((Ok(Err((resizers, refusal))), _)) => {
                    let refused = BankResized::Refused { holding: BankHolding::Experts, at: slots, refusal };
                    (Resize::Idle(resizers), Some(refused))
                }
                Some((Ok(Ok(prepared)), at)) => (self.install(prepared, (from, charged), (begun, at), wake), None),
            },
            Resize::Releasing { from, charged, refusal, begun, prepared, installed, done } => match take(&done) {
                None => {
                    let releasing = Resize::Releasing { from, charged, refusal, begun, prepared, installed, done };
                    (releasing, None)
                }
                Some((Err(panicked), _)) => return broken(&mut commitment, panicked),
                Some((Ok(resizers), released)) => {
                    commitment.resize = Resize::Idle(resizers);
                    let holding = BankHolding::Experts;
                    return Ok(Some(match refusal {
                        Some(refusal) => BankResized::Refused { holding, at: slots, refusal },
                        None => BankResized::Shrunk {
                            holding,
                            from,
                            to: slots,
                            released: charged
                                .checked_sub(self.charged()?)
                                .ok_or_else(|| DomainError::invariant("an expert bank shrink raised its charge"))?,
                            times: begun.times(prepared, installed, released),
                        },
                    }));
                }
            },
            Resize::Growing(growing) => match growing.ended(BankHolding::Experts, slots, || self.charged()) {
                None => (Resize::Growing(growing), None),
                Some(Ok((resizers, grown, _))) => (Resize::Idle(resizers), Some(grown)),
                Some(Err(reason)) => return broken(&mut commitment, reason),
            },
            idle => (idle, None),
        };
        commitment.resize = resize;
        Ok(finished)
    }

    /// Install a shrink's `prepared` storage at a boundary between flights
    /// and release its surplus off this thread, once the device work before
    /// the install completes.
    fn install(
        &self,
        recommits: Vec<PreparedRecommit>,
        (from, charged): (u64, u64),
        (begun, prepared): (Begun, Instant),
        wake: CompletionWake,
    ) -> Resize {
        let installed = Instant::now();
        let mut refusal = None;
        let tensors = recommits
            .into_iter()
            .map(|prepared| match prepared.install() {
                Ok(release) => Installed::Release(release),
                Err(refused) => {
                    refusal.get_or_insert(refused.error);
                    Installed::Kept(refused.resizer)
                }
            })
            .collect::<Vec<_>>();
        let done = self.resize_thread.run(wake, move || tensors.into_iter().map(Installed::release).collect());
        Resize::Releasing { from, charged, refusal, begun, prepared, installed, done }
    }

    /// Whether a resize runs, until its advance finishes it (broken records
    /// run none: a resize asked of them fails with why).
    pub(super) fn resizing(&self) -> bool {
        !matches!(self.commitment().resize, Resize::Idle(_) | Resize::Broken(_))
    }
}

/// A shrink of the records the host lands into, done: the slots it took them
/// from and to, Seismic's charge of them before and after, and the refusal
/// of a tensor's storage, which keeps what it holds.
pub(super) struct LandedShrink {
    pub(super) from: u64,
    pub(super) to: u64,
    pub(super) before: u64,
    pub(super) after: u64,
    pub(super) refusal: Option<TensorError>,
}

/// Commit each of `resizers` at its `rows` now: every resizer back, and the
/// first refusal of a tensor's storage, which keeps what it holds.
fn recommit(resizers: Vec<Resizer>, rows: Vec<u64>) -> (Vec<Resizer>, Option<TensorError>) {
    let mut refusal = None;
    let kept = resizers
        .into_iter()
        .zip(rows)
        .map(|(resizer, rows)| {
            resizer.recommit(rows).unwrap_or_else(|refused| {
                refusal.get_or_insert(refused.error);
                refused.resizer
            })
        })
        .collect();
    (kept, refusal)
}
