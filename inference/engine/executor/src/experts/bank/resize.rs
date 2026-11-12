use super::{ExpertBank, host_views};
use crate::planning::{BankTier, ExpertBankPlan, Placer};
use crate::experts::watcher::Candidates;
use crate::domain::{BankHolding, BankResized, BankStep, DomainError, MemoryMoment, ResizeTimes, RunningBegun, RunningPressure, Trip};
use crate::CompletionWake;
use std::time::{Duration, Instant};
use crate::native::BankMoves;
use magnitude_kernels::routed_move;
use seismic::{NativeTensorBatch, NativeTensorBatchCompletion, PreparedRecommit, RecommitRelease, Resizer, Tensor, TensorError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use crate::experts::watcher::Join;
use crate::storage_thread::{Done, StorageThread};

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
    /// A cut begun beside a running chain the domain has not yet taken
    /// ([`ExpertBank::take_running_begun`]).
    pub(super) running: Option<RunningBegun>,
}

/// A host tier's host slots committed (`ExpertBankPlan::host_records`),
/// written as [`Commitment::slots`] is, the most it holds (every expert,
/// `ExpertBankPlan::host_ceiling`), and its record tensors' resize.
pub(super) struct HostTier {
    pub(super) slots: Arc<AtomicU64>,
    pub(super) ceiling: u64,
    pub(super) resize: HostResize,
    /// The thread the tier's storage work runs on: host RAM is a store of
    /// its own, so its backing and growth never queue a resize of the
    /// device's records behind them, nor wait on one.
    pub(super) thread: StorageThread,
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
/// `resident` records the slot's memory, which exists from its backing (a
/// reservation backs a granule by touching or wiring every page), and
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
    resident: impl Fn(u64),
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
        // The slot's storage is resident (a reservation's granules are
        // backed by touching or wiring every page, `seismic_reservation`):
        // its memory exists from here, and its growth's claim says so.
        resident(at);
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
                Some(refusal) => (resizers, BankResized::GrowthRefused { holding, at: slots, refusal }, spent),
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

/// The resize of every record tensor, group by group in routed tensor
/// order, one at a time.
pub(super) enum Resize {
    /// No resize runs: each record tensor's resizer.
    Idle(Vec<Resizer>),
    /// A cut of the one store asked of the watcher beside a running chain
    /// (`watcher::Shared::cut_running`), not yet answered.
    Cutting(Cutting),
    /// A shrink from `from` slots to the commitment's, its storage prepared
    /// off the submitting thread, the records charged `charged` before it,
    /// which installs where `install` says.
    Preparing {
        from: u64,
        charged: u64,
        begun: Begun,
        install: Install,
        done: Done<Result<Vec<PreparedRecommit>, (Vec<Resizer>, TensorError)>>,
    },
    /// A shrink installed (`refusal` where a tensor refused its install,
    /// keeping its storage, which breaks the records once the others'
    /// releases end), its surplus released off the submitting thread once the
    /// device work before its install completes.
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

/// A cut of the one store asked beside the running chain whose last step
/// is `last` ([`ExpertBank::release_running`]): from `from` slots, the
/// records charged `charged` then, decided at `decided` with the process
/// holding `moment`, its asking having held the submitting thread `held`;
/// the resizers held until it is answered.
pub(super) struct Cutting {
    from: u64,
    charged: u64,
    last: u64,
    decided: Instant,
    held: Duration,
    moment: MemoryMoment,
    resizers: Vec<Resizer>,
}

/// Where a shrink's prepared storage installs: at a boundary between
/// flights; or, a cut beside the running chain whose last step is `last`,
/// also at a submission boundary of that chain once every pass of block
/// `after` is submitted (at once where none: its steps had decided no hit
/// at the cut), and at any boundary of a later chain. Every run that may
/// read a record past the cut is then submitted before the install, whose
/// release waits for it (`PreparedRecommit::install`), and every run
/// submitted after resolves only records below it.
#[derive(Clone, Copy)]
pub(super) enum Install {
    Between,
    Running { last: u64, after: Option<usize> },
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
        ResizeTimes { decided, queued, planned, trips, moves, prepared, installed, released, unpinned: Duration::ZERO }
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

    /// The host tier's record tensors' groups, in tensor order.
    fn host_groups(&self) -> Vec<usize> {
        self.host_tensors().map(|(group, _)| group).collect()
    }

    /// The host tier's record tensors' rows at `slots` host slots, in
    /// tensor order.
    fn host_rows(&self, slots: u64) -> Vec<u64> {
        self.host_groups().into_iter().map(|group| self.counts.host_records(group, slots)).collect()
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
            true => BankStep::Begun(BankTier::Host),
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
        match self.growing(resizers, (&tier.slots, to), tensors, Joining::Landed, (charged, crate::planning::BankCounts::host_charge), &tier.thread, wake) {
            Ok(growing) => tier.resize = resize(growing),
            Err((resizers, error)) => {
                tier.resize = HostResize::Idle(resizers);
                return Err(error);
            }
        }
        Ok(true)
    }

    /// Begin growing a holding's record tensors (`resizers`, each of group
    /// `groups[i]`, at `records`) from its `slots` now to `to` on its store's
    /// resize `thread` ([`grow_by_slots`]), its records charged `charged` now, each
    /// slot recorded allocated in the bank's cache at its `charge` as soon
    /// as every tensor backs it (`BankCache::backed`), then joining the
    /// watcher as `joining` says; `wake` completed once it ends. Refused,
    /// the resizers come back.
    fn growing(
        &self,
        resizers: Vec<Resizer>,
        (slots, to): (&Arc<AtomicU64>, u64),
        (groups, records): (Vec<usize>, RecordsAt),
        joining: Joining,
        (charged, charge): (u64, fn(&crate::planning::BankCounts, u64) -> u64),
        thread: &StorageThread,
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
        let backed = {
            let (cache, counts) = (self.cache.clone(), self.counts.clone());
            move |at: u64| cache.backed(charge(&counts, at))
        };
        let join = {
            let plan = plan.clone();
            move |at: u64| -> Result<Join, String> {
                let counts = (0..plan.groups().len()).map(|group| records(&plan, group, at));
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
        let done = thread.run(wake, move || {
            let rows = |at| groups.iter().map(|&group| records(&plan, group, at)).collect();
            // The growth ends however its slots end, a panic among them too
            // (re-raised to the thread's catch): no slot joins after.
            let grown = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                grow_by_slots(resizers, (from, to), rows, backed, join, |join| shared.join(join), (&slots, &rises), &stopping, || shared.backing_slot(&stopping))
            }));
            if lands {
                shared.join(Join::Grown);
            }
            grown.unwrap_or_else(|payload| std::panic::resume_unwind(payload))
        });
        Ok(Growing { from, charged, decided, stop, done })
    }

    /// End a running growth of the holdings in `tier` storage at its next
    /// slot ([`Growing::stop`]): a shrink of that store waits for it no
    /// longer than that slot's backing. The host tier's backing is its
    /// growth.
    pub(super) fn stop_growth(&self, tier: BankTier) {
        match tier {
            BankTier::Bank => {
                if let Resize::Growing(growing) = &self.commitment().resize {
                    growing.stop.store(true, Ordering::Release);
                }
            }
            BankTier::Host => {
                if let Some(host) = self.host.as_ref() {
                    if let HostResize::Growing(growing) | HostResize::Backing(growing) = &host.lock().expect("host tier").resize {
                        growing.stop.store(true, Ordering::Release);
                    }
                }
            }
        }
        // A growth waiting between its slots on a hold looks again.
        self.watcher.shared().events().notify();
    }

    /// Finish a host tier's growth, or its backing at open, once its resize
    /// thread ended it: its outcome (grown to the host slots its slots
    /// reached, or refused before any joined), and whether a domain resize
    /// began it (the backing's none did, so its end is reported to none).
    pub(super) fn advance_host(&self) -> Result<Option<(BankResized, bool)>, DomainError> {
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
                Ok(Some((resized, reported)))

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

    /// The candidate cuts of a shrink of the records the host lands into,
    /// from `shallowest` slots down to `deepest` ([`Candidates`]): each
    /// group's records at a candidate's slots ([`landed_records`]), which fit
    /// u32 as the shallowest's do.
    fn landed_cuts(&self, (shallowest, deepest): (u64, u64)) -> Result<Candidates, DomainError> {
        assert!(deepest <= shallowest, "a shrink's deepest cut {deepest} lies below its shallowest {shallowest}");
        let slots = |slots: u64| usize::try_from(slots).map_err(|_| DomainError::invariant("bank slots exceed the host"));
        for group in 0..self.plan.groups().len() {
            u32::try_from(self.counts.landed_records(group, shallowest))
                .map_err(|_| DomainError::invariant("the records the host lands into exceed u32"))?;
        }
        let (counts, groups) = (self.counts.clone(), self.plan.groups().len());
        Ok(Candidates {
            shallowest: slots(shallowest)?,
            deepest: slots(deepest)?,
            records: Box::new(move |slots| (0..groups).map(|group| counts.landed_records(group, slots as u64) as usize).collect()),
        })
    }

    /// Shrink the host tier now, between flights, to the deepest of
    /// `shallowest..=deepest` host slots that keeps every record in flight
    /// (`Shared::shrink_tier`): the watcher keeps each group's most valuable
    /// records below the cut (their bytes copied there), then each tensor
    /// releases the rest. Its size before and after, and Seismic's charge of
    /// it before and after (a tensor's storage refusing breaks it,
    /// [`shrink_refused`]); not while a growth of it runs, nor where every
    /// cut falls below a record in flight.
    pub(super) fn shrink_host(&self, cuts: (u64, u64)) -> Result<Shrinking<LandedShrink>, DomainError> {
        let host = self.host.as_ref().ok_or_else(|| DomainError::invariant("a host tier resize without one"))?;
        let mut tier = host.lock().expect("host tier");
        // Everything that can fail before the resizers are taken; an ask
        // the watcher fails returns them, the storage as it was.
        let (before, candidates) = (self.host_charged()?, self.landed_cuts(cuts)?);
        let Some(resizers) = Self::idle_host(&mut tier)? else {
            return Ok(Shrinking::Running);
        };
        let from = tier.slots.load(Ordering::Acquire);
        let cut = match self.watcher.shared().shrink_tier(candidates) {
            Ok(Some(cut)) => cut,
            Ok(None) => {
                tier.resize = HostResize::Idle(resizers);
                return Ok(Shrinking::Deferred);
            }
            Err(reason) => {
                tier.resize = HostResize::Idle(resizers);
                return Err(DomainError::invariant(reason));
            }
        };
        let to = cut.slots as u64;
        tier.slots.store(to, Ordering::Release);
        let unpinning = seismic::backing_times();
        let recommitted = recommit(resizers, self.host_rows(to));
        let unpinned = seismic::backing_times().since(unpinning).unpin;
        match recommitted {
            Ok(resizers) => tier.resize = HostResize::Idle(resizers),
            Err(refusal) => {
                let reason = shrink_refused(refusal);
                tier.resize = HostResize::Broken(reason.clone());
                return Err(DomainError::invariant(reason));
            }
        }
        Ok(Shrinking::Begun(LandedShrink { unpinned, from, to, before, after: self.host_charged()? }))
    }

    /// Shrink the one store (where the host places), decided at `decided`
    /// and queued `queued` behind the flights before it: now, between
    /// flights, the watcher cuts it at the deepest of `shallowest..=deepest`
    /// slots that keeps every record in flight, keeps each group's most
    /// valuable records below the cut and addresses those alone, so no step
    /// reads the rest, and the device moves the kept records' bytes there by
    /// `kernels` ([`Self::carry_moves`]) ahead of every later step, in queue
    /// order, so this thread only queues them (`Shared::shrink_store`); each
    /// record tensor's storage then releases the rest off this thread once
    /// the device work before its install completes, the moves among it,
    /// and a later boundary finishes the shrink ([`Self::advance`]). Not
    /// while its records resize, nor where every cut falls below a record
    /// in flight.
    pub(super) fn shrink_store(&self, kernels: &[BankMoves], cuts: (u64, u64), decided: Instant, queued: Duration, wake: CompletionWake) -> Result<Shrinking<()>, DomainError> {
        let mut commitment = self.commitment();
        // Everything that can fail before the resizers are taken; an ask
        // the watcher fails returns them, the storage as it was.
        let (before, candidates) = (self.charged()?, self.landed_cuts(cuts)?);
        let Some(resizers) = Self::idle(&mut commitment)? else {
            return Ok(Shrinking::Running);
        };
        let from = commitment.slots.load(Ordering::Acquire);
        let answered = Self::observe_moves(&mut commitment)
            .and_then(|()| self.watcher.shared().shrink_store(candidates).map_err(DomainError::invariant));
        let (cut, kept) = match answered {
            Ok(Some(answered)) => answered,
            Ok(None) => {
                commitment.resize = Resize::Idle(resizers);
                return Ok(Shrinking::Deferred);
            }
            Err(error) => {
                commitment.resize = Resize::Idle(resizers);
                return Err(error);
            }
        };
        let to = cut.slots as u64;
        let planned = decided.elapsed() - queued;
        let moving = Instant::now();
        let moves = kept.into_iter().map(|moves| moves.into_iter().map(|(from, to)| (from as usize, to as usize)).collect());
        // The watcher has mapped the kept experts to their new records: a
        // refused submission leaves them reading records no move wrote, a
        // bank no step may read.
        commitment.moves = match self.carry_moves(moves, kernels) {
            Ok(moves) => moves,
            Err(error) => {
                commitment.resize = Resize::Broken(format!("a shrink's moves were refused after the watcher committed it: {error}"));
                return Err(error);
            }
        };
        let moves = moving.elapsed();
        commitment.slots.store(to, Ordering::Release);
        let begun = Begun { decided, queued, planned, trips: [cut.trip, Trip::default()], moves };
        commitment.resize = self.prepare_off_thread(resizers, (from, to), before, begun, Install::Between, wake);
        Ok(Shrinking::Begun(()))
    }

    /// Each group's records at `slots` slots.
    fn committed_records(&self, slots: u64) -> Result<Vec<usize>, DomainError> {
        (0..self.plan.groups().len())
            .map(|group| {
                usize::try_from(self.counts.records_at(group, slots))
                    .map_err(|_| DomainError::invariant("bank records exceed the host"))
            })
            .collect()
    }

    /// The rows each record tensor commits at `slots` slots, in resizer
    /// order.
    fn rows_at(&self, slots: u64) -> Vec<u64> {
        self.record_groups().into_iter().map(|group| self.counts.records_at(group, slots)).collect()
    }

    /// Each record tensor's group, in resizer order.
    fn record_groups(&self) -> Vec<usize> {
        (0..self.plan.groups().len())
            .flat_map(|group| std::iter::repeat_n(group, self.plan.groups()[group].record.form().tensors().len()))
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
    /// thread, installed where `install` says.
    fn prepare_off_thread(&self, resizers: Vec<Resizer>, (from, to): (u64, u64), charged: u64, begun: Begun, install: Install, wake: CompletionWake) -> Resize {
        let rows = self.rows_at(to);
        let done = self.resize_thread.run(wake, move || prepare(resizers, rows));
        Resize::Preparing { from, charged, begun, install, done }
    }

    /// Submit a shrink's `moves`, each group's (from, to), to the device in a
    /// submission of their own: each group's records, by its move list. The
    /// device runs every later flight after them, in queue order, so no wait
    /// is needed before what they move is mapped. A submission refused
    /// leaves a shrink the watcher has not committed uncommitted; one it has
    /// committed, broken.
    fn carry_moves(
        &self,
        moves: impl Iterator<Item = Vec<(usize, usize)>>,
        kernels: &[BankMoves],
    ) -> Result<Option<NativeTensorBatchCompletion>, DomainError> {
        let failed = |error: &dyn std::fmt::Display| DomainError::invariant(format!("expert bank moves: {error}"));
        let mut batch = None;
        let groups = self.layout.groups.iter().zip(moves).zip(&self.move_lists).zip(kernels);
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
            list.write_list(&pairs, crate::experts::ListBounds { from: records, to: records, lent: 0 });
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

    /// [`ElasticBank::shrink`] to `deepest` slots, the settled size past
    /// the nearest that releases storage (`ServedBank::releasing`), in one
    /// resize, in a gap between flights, decided at
    /// `decided`, its moves carried by `kernels`. The plan's
    /// evictions are published first; nothing is mapped to a record a move
    /// writes until the moves are queued ahead of every later step and the
    /// plan is committed. A plan dropped on an error before then leaves
    /// every kept expert where its bytes are; once queued, the moves' device
    /// failure fails every flight after them ([`ElasticBank::carried`]), so
    /// nothing read from a record they failed to write is taken.
    pub(super) fn shrink(
        &self,
        kernels: &[BankMoves],
        deepest: u64,
        decided: Instant,
        wake: CompletionWake,
    ) -> Result<BankStep, DomainError> {
        let mut commitment = self.commitment();
        let charged = self.charged()?;
        let Some(resizers) = Self::idle(&mut commitment)? else {
            return Ok(BankStep::Running);
        };
        let from = commitment.slots.load(Ordering::Acquire);
        if from <= self.counts.floor() {
            commitment.resize = Resize::Idle(resizers);
            return Ok(BankStep::AtLimit);
        }
        let shared = self.watcher.shared();
        let planned = (|| {
            Self::observe_moves(&mut commitment)?;
            let queued = decided.elapsed();
            let slots = usize::try_from(deepest).map_err(|_| DomainError::invariant("bank slots exceed the host"))?;
            let plan = shared.plan_shrink(slots, self.committed_records(deepest)?).map_err(DomainError::invariant)?;
            let to = plan.slots() as u64;
            let moving = Instant::now();
            commitment.moves = self.carry_moves(plan.moves(), kernels)?;
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
        commitment.resize = self.prepare_off_thread(resizers, (from, to), charged, begun, Install::Between, wake);
        Ok(BankStep::Begun(BankTier::Bank))
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
        match self.growing(resizers, (&commitment.slots, to), tensors, joining, (charged, crate::planning::BankCounts::resident_charge), &self.resize_thread, wake) {
            Ok(growing) => commitment.resize = Resize::Growing(growing),
            Err((resizers, error)) => {
                commitment.resize = Resize::Idle(resizers);
                return Err(error);
            }
        }
        Ok(BankStep::Begun(BankTier::Bank))
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
            // Between flights every run of a chain a cut was asked beside is
            // submitted: its answer is read, and its storage installs.
            Resize::Cutting(cutting) => (self.cut_answered(cutting, &mut commitment, wake), None),
            Resize::Preparing { from, charged, begun, install, done } => match take(&done) {
                None => (Resize::Preparing { from, charged, begun, install, done }, None),
                Some((Err(panicked), _)) => return broken(&mut commitment, panicked),
                Some((Ok(Err((_, refusal))), _)) => return broken(&mut commitment, shrink_refused(refusal)),
                Some((Ok(Ok(prepared)), at)) => (self.install(prepared, (from, charged), (begun, at), wake), None),
            },
            Resize::Releasing { from, charged, refusal, begun, prepared, installed, done } => match take(&done) {
                None => {
                    let releasing = Resize::Releasing { from, charged, refusal, begun, prepared, installed, done };
                    (releasing, None)
                }
                Some((Err(panicked), _)) => return broken(&mut commitment, panicked),
                Some((Ok(resizers), released)) => {
                    if let Some(refusal) = refusal {
                        return broken(&mut commitment, shrink_refused(refusal));
                    }
                    commitment.resize = Resize::Idle(resizers);
                    return Ok(Some(BankResized::Shrunk {
                        holding: BankHolding::Experts,
                        from,
                        to: slots,
                        released: charged
                            .checked_sub(self.charged()?)
                            .ok_or_else(|| DomainError::invariant("an expert bank shrink raised its charge"))?,
                        times: begun.times(prepared, installed, released),
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

    /// Install a shrink's `prepared` storage at a boundary its
    /// [`Install`] allows and release its surplus off this thread, once the
    /// device work before the install completes.
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
        let done = self.release_thread.run(wake, move || tensors.into_iter().map(Installed::release).collect());
        Resize::Releasing { from, charged, refusal, begun, prepared, installed, done }
    }

    /// Whether a resize runs, until its advance finishes it (broken records
    /// run none: a resize asked of them fails with why).
    pub(super) fn resizing(&self) -> bool {
        !matches!(self.commitment().resize, Resize::Idle(_) | Resize::Broken(_))
    }

    /// At a submission boundary of the scratch-served chain whose steps run
    /// from `first` through `last`, every pass of its blocks through
    /// `submitted` submitted (none where no block's is), the one store's
    /// release beside it, a step at a time:
    /// - no resize running, where `pressure` finds the bank's storage short
    ///   (it reads memory at most every reading interval), ask the watcher
    ///   to cut the store beside the chain, from the nearest slots that
    ///   release storage toward those releasing what it is short of
    ///   (`watcher::Shared::cut_running`), the resizers held;
    /// - that cut answered, its storage prepared off this thread
    ///   ([`Self::cut_answered`]);
    /// - prepared, installed once its [`Install`] allows, its surplus
    ///   released off this thread (`Resize::Releasing`): the next advance
    ///   reports it as any shrink.
    ///
    /// The block whose passes' submission an install awaits, where one
    /// does. Only where the host places (`Placer::Bank`): a host tier's
    /// shrink copies its kept records first, and a device bank's moves
    /// them, between flights alone.
    pub(crate) fn release_running(&self, (first, last): (u64, u64), submitted: Option<usize>, pressure: &dyn Fn() -> Option<RunningPressure>) -> Option<usize> {
        if self.plan.placer() != Placer::Bank {
            return None;
        }
        let started = Instant::now();
        // The release reports to the advance that takes it: no wake.
        let wake = || CompletionWake::new(|| {});
        // Only this thread leaves the resizers idle.
        let short = {
            let commitment = self.commitment();
            matches!(commitment.resize, Resize::Idle(_)) && commitment.slots.load(Ordering::Acquire) > self.counts.floor()
        };
        let short = short.then(pressure).flatten().filter(|pressure| pressure.bytes > 0);
        let mut commitment = self.commitment();
        let (resize, awaits) = match std::mem::replace(&mut commitment.resize, Resize::Idle(Vec::new())) {
            Resize::Idle(resizers) => match short {
                Some(short) => {
                    let from = commitment.slots.load(Ordering::Acquire);
                    (self.cut_beside(resizers, from, (first, last), short, started), None)
                }
                None => (Resize::Idle(resizers), None),
            },
            Resize::Cutting(cutting) => (self.cut_answered(cutting, &mut commitment, wake()), None),
            Resize::Preparing { from, charged, begun, install, done } => {
                // A shrink decided between flights installs between flights
                // alone, as it always has.
                let (due, after) = match install {
                    Install::Between => (false, None),
                    Install::Running { last: of, after } if of == last => {
                        match after.filter(|&block| submitted.is_none_or(|through| through < block)) {
                            Some(block) => (false, Some(block)),
                            None => (true, None),
                        }
                    }
                    Install::Running { .. } => (true, None),
                };
                let prepared = match due {
                    false => None,
                    true => done.lock().expect("bank resize outcome").take(),
                };
                match prepared {
                    None => (Resize::Preparing { from, charged, begun, install, done }, after),
                    Some((Err(panicked), _)) => (Resize::Broken(panicked), None),
                    Some((Ok(Err((_, refusal))), _)) => (Resize::Broken(shrink_refused(refusal)), None),
                    Some((Ok(Ok(prepared)), at)) => (self.install(prepared, (from, charged), (begun, at), wake()), None),
                }
            }
            other => (other, None),
        };
        commitment.resize = resize;
        awaits
    }

    /// Ask the watcher to cut the one store beside the running chain whose
    /// steps run from `first` through `last`, from `from` slots, its storage
    /// `short` as read at `started`: from the nearest slots that release
    /// storage toward those releasing what it is short of, the resizers
    /// held until it answers. Resizers idle as they were where no slots
    /// above the floor release storage, and broken records where the cut
    /// cannot be formed (the charge unread, the records past u32).
    fn cut_beside(&self, resizers: Vec<Resizer>, from: u64, (first, last): (u64, u64), short: RunningPressure, started: Instant) -> Resize {
        let RunningPressure { bytes, held: moment } = short;
        let floor = self.counts.floor();
        let decided = Instant::now();
        let charge = |at| Ok::<_, std::convert::Infallible>(self.counts.resident_charge(at));
        let releasing = |bytes| crate::planning::releasing(from, floor, bytes, charge).unwrap_or_else(|never| match never {});
        let (Some(shallowest), Some(deepest)) = (releasing(0), releasing(bytes)) else {
            return Resize::Idle(resizers);
        };
        let asked = self.charged().and_then(|charged| Ok((charged, self.landed_cuts((shallowest, deepest))?)));
        let (charged, candidates) = match asked {
            Ok(asked) => asked,
            Err(error) => return Resize::Broken(format!("a cut beside a running chain: {error}")),
        };
        self.watcher.shared().cut_running((first, last), candidates);
        tracing::info!(from, shallowest, deepest, short = bytes, "the expert bank is cut beside a running chain");
        Resize::Cutting(Cutting { from, charged, last, decided, held: started.elapsed(), moment, resizers })
    }

    /// A cut asked beside a running chain, its answer read now: where it
    /// cut, the commitment's slots follow at once (no step from then on
    /// resolves a record past them), the domain is owed its beginning
    /// ([`Self::take_running_begun`]), and each record tensor's storage at
    /// the cut is prepared off this thread, installed where its
    /// [`Install`] allows; where it cut nothing, the resizers idle as they
    /// were; where serving ended, the records broken with why (the next
    /// advance fails with it); unanswered, it waits.
    fn cut_answered(&self, cutting: Cutting, commitment: &mut Commitment, wake: CompletionWake) -> Resize {
        let Some(answer) = self.watcher.shared().running_cut() else {
            return Resize::Cutting(cutting);
        };
        let Cutting { from, charged, last, decided, held, moment, resizers } = cutting;
        let cut = match answer {
            Ok(Some(cut)) => cut,
            Ok(None) => return Resize::Idle(resizers),
            Err(reason) => return Resize::Broken(reason),
        };
        let to = cut.slots as u64;
        commitment.slots.store(to, Ordering::Release);
        assert!(commitment.running.replace(RunningBegun { decided, held, moment }).is_none(), "a running cut's beginning is taken before the next begins");
        let after = cut.decided.map(|layer| self.plan.layers()[layer].sublayer().block as usize);
        let zero = Duration::ZERO;
        let begun = Begun { decided, queued: zero, planned: cut.trip.picked + cut.trip.served, trips: [cut.trip, Trip::default()], moves: zero };
        self.prepare_off_thread(resizers, (from, to), charged, begun, Install::Running { last, after }, wake)
    }

    /// [`ElasticBank::running_begun`](crate::domain::ElasticBank::running_begun).
    pub(super) fn take_running_begun(&self) -> Option<RunningBegun> {
        self.commitment().running.take()
    }
}

/// What a shrink of the records the host lands into came to: begun (or, of
/// a host tier, done), with what it reports; not begun while a resize of
/// them runs, which a later boundary finishes; or not begun where every cut
/// falls below a record in flight, which the next boundary decides again.
pub(super) enum Shrinking<T> {
    Begun(T),
    Running,
    Deferred,
}

/// A shrink of the records the host lands into, done: the slots it took them
/// from and to, and Seismic's charge of them before and after.
pub(super) struct LandedShrink {
    /// Its release's unregistration of host pages.
    pub(super) unpinned: Duration,
    pub(super) from: u64,
    pub(super) to: u64,
    pub(super) before: u64,
    pub(super) after: u64,
}

/// Commit each of `resizers` at its `rows` now, a shrink's: every resizer
/// back, or the first refusal of a tensor's storage ([`shrink_refused`]).
fn recommit(resizers: Vec<Resizer>, rows: Vec<u64>) -> Result<Vec<Resizer>, TensorError> {
    resizers
        .into_iter()
        .zip(rows)
        .map(|(resizer, rows)| resizer.recommit(rows).map_err(|refused| refused.error))
        .collect()
}

/// Why records whose storage refused a shrink are broken. A shrink needs no
/// memory, and the views the watcher addresses records by are handed over
/// before any shrink releases them, so nothing pins what it releases: a
/// refusal is a broken invariant or a failed device, and the records keep
/// storage the directory no longer uses.
fn shrink_refused(refusal: TensorError) -> String {
    format!("the expert bank's storage refused a shrink: {refusal}")
}
