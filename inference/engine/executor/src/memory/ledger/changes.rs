//! The scope's own allocation changes ([`Changes`], [`Changing`], [`Owner`],
//! [`OpenChange`]) and the instants readings are sampled at ([`Sampled`],
//! [`Sample`]).
//!
//! The one change API, and its rules:
//!
//! - **What a change spans.** One OS effect of the process's memory (a
//!   native allocation or free, a driver call that takes or returns memory)
//!   and every record of its bytes in the ledgers, together: it begins
//!   before the effect and ends after the last record. No effect is made
//!   outside the change that records it, and no record outside one: a
//!   record outside any change panics (`Locked`), in the process's scope.
//!   Seismic keeps this by construction: every record of an entry takes the
//!   `&seismic::memory::Change` it is made in, which only an entry begins
//!   ([`seismic::memory::Entry::change`]), so no record is written outside
//!   one; and a native object and its charge are one owner whose allocation
//!   and free run inside the change (`seismic::memory::Backed`, or an
//!   object's `seismic::memory::DropScope` where it cannot be one). The
//!   ledger's `Locked` panic remains for an embedder's own records.
//! - **Who may begin one.** The owner's thread, as an owner's change
//!   ([`Changes::change`], [`OpenChange`]: a load, a domain); any thread, as
//!   a step ([`Changes::back`]: a bank's growth, a release); and an entry's
//!   own change ([`seismic::memory::Entry::change`]: a Seismic allocation or
//!   free on any thread, the pool-return thread's included), which begins at
//!   once and waits for no sample. A change begun where its thread has one
//!   under way joins it: one logical change, one number.
//! - **Who may wait inside one.** A change holds readings off for its
//!   length: a sample ([`Sample::try_now`]) stands only where none was under
//!   way, and a settled one ([`Sample::settled`]) waits for them to end. So
//!   what waits inside a change (a backend waiting for the device to finish
//!   with a block's storage before it frees it) holds readings off that
//!   long, and nothing inside a change waits for a sample: a step off the
//!   owner's thread waits for a settled sample before it begins, never
//!   inside.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[cfg(any(test, feature = "fixtures"))] use super::store::Ledgers;
#[cfg(test)] use super::view::Ledger;

/// A scope's own allocation changes across its stores ([`Ledgers`]): above
/// [`IN_FLIGHT_BITS`], how many were begun (a journaled change's number, and the
/// place a sample takes among them, [`Sampled`]); below, how many are under
/// way ([`Changing`]). A sample stands only where none was under way as it
/// began and none began while it read ([`Sampled::attempt`]): a change
/// numbered below its place was made, OS effect and record alike, before
/// every figure of the sample, and one at or above it after every figure.
/// The product's one scope is the process's; a test's own scope counts only
/// its own changes.
pub struct Changes {
    /// A test's scope, no OS behind it ([`Ledgers::new`]): a record outside
    /// a change is a change of its own, until the test holds it to the
    /// process's rule ([`Self::strict`]).
    #[cfg(any(test, feature = "fixtures"))]
    pub(super) alone: std::sync::atomic::AtomicBool,
    pub(super) count: AtomicU64,
    /// The threads that own the scope's memory now, one per load or domain
    /// ([`Owner`]): only they begin an owner's change.
    pub(super) owners: Mutex<Vec<std::thread::ThreadId>>,
    /// Where a change's end and a settled sample waiting for one meet
    /// ([`Sample::settled`]).
    pub(super) meet: Mutex<Meet>,
    /// Signalled as a change ends and as a waiting sample lets go.
    pub(super) moved: Condvar,
    /// The owner's clock, which samples and changes are stamped by: the
    /// process's is `Instant::now`, a scripted test's its own.
    pub(super) clock: Clock,
}

/// How many changes ended, how many settled samples wait to stand (a step
/// off the owner's thread begins none while one waits, [`Changes::back`],
/// so a sample waits for the steps under way and the entry changes beside
/// it, [`Sampled::settled`]), and how many
/// changes under way are the owner's and whose thread is not itself waiting
/// in a settled sample (a step never waits beside one: the owner may be
/// waiting on the step; but an owner waiting for a sample is waiting on no
/// step, so its change stops counting from the wait's start to its end,
/// [`Sampled::settled`]).
#[derive(Default)]
pub(super) struct Meet {
    ended: u64,
    wanted: usize,
    owner: usize,
}

/// A scope's clock.
pub(super) type Clock = Box<dyn Fn() -> Instant + Send + Sync>;

impl Default for Changes {
    fn default() -> Self {
        Self {
            #[cfg(any(test, feature = "fixtures"))]
            alone: std::sync::atomic::AtomicBool::default(),
            count: AtomicU64::default(),
            owners: Mutex::default(),
            meet: Mutex::default(),
            moved: Condvar::default(),
            clock: Box::new(Instant::now),
        }
    }
}

/// The bits of [`Changes`] that count changes under way.
const IN_FLIGHT_BITS: u32 = 20;

/// One change begun, in [`Changes`].
const ONE_CHANGE: u64 = 1 << IN_FLIGHT_BITS;

impl Changes {
    /// Now, on the scope's clock.
    pub(super) fn now(&self) -> Instant {
        (self.clock)()
    }

    fn meet(&self) -> MutexGuard<'_, Meet> {
        self.meet.lock().expect("the changes' meet is never left by a panic")
    }

    /// Whether this is a test's scope in which a record outside a change is
    /// a change of its own ([`Ledgers::new`]); never the process's.
    pub(super) fn alone(&self) -> bool {
        #[cfg(any(test, feature = "fixtures"))]
        return self.alone.load(Ordering::Relaxed);
        #[cfg(not(any(test, feature = "fixtures")))]
        return false;
    }

    /// A change of this scope begun on this thread, or the one this thread
    /// has under way already, which it joins: one logical change, one
    /// number, however its parts nest.
    pub(super) fn begin(self: &Arc<Self>, owner: bool) -> Changing {
        if let Some(number) = self.open_here() {
            return Changing { changes: self.clone(), number, owner, joined: true };
        }
        if owner {
            let here = std::thread::current().id();
            let owned = self.owners.lock().expect("the changes' owners are never left by a panic").contains(&here);
            assert!(owned || self.alone(), "invariant violated: an owner's change begun off its owner's thread");
            self.meet().owner += 1;
        }
        let number = self.count.fetch_add(ONE_CHANGE + 1, Ordering::SeqCst) >> IN_FLIGHT_BITS;
        OPEN.with_borrow_mut(|open| open.push((self.key(), number, owner)));
        Changing { changes: self.clone(), number, owner, joined: false }
    }

    /// This scope's identity among the threads' open changes.
    fn key(&self) -> usize {
        std::ptr::from_ref(self) as usize
    }

    /// The number of the change of this scope this thread has under way.
    pub(super) fn open_here(&self) -> Option<u64> {
        OPEN.with_borrow(|open| open.iter().find(|(key, ..)| *key == self.key()).map(|(_, number, _)| *number))
    }

    /// Whether the change of this scope this thread has under way is an
    /// owner's ([`Self::change`], [`OpenChange`]).
    fn owner_open_here(&self) -> bool {
        OPEN.with_borrow(|open| open.iter().any(|(key, _, owner)| *key == self.key() && *owner))
    }

    /// From now, a record outside a change panics in this test's scope, as
    /// in the process's.
    #[cfg(test)]
    pub(crate) fn strict(&self) {
        self.alone.store(false, Ordering::Relaxed);
    }

    /// `change`, an OS change of the scope's memory on the owner's thread
    /// and its records in the ledgers, run as one: no sample beside it
    /// stands. The owner takes no reading while it runs.
    pub fn change<R>(self: &Arc<Self>, change: impl FnOnce(&Changing) -> R) -> R {
        change(&self.begin(true))
    }

    /// The calling thread as an owner of the scope's memory until the
    /// returned [`Owner`] drops: a load's or a domain's thread.
    pub fn own(self: &Arc<Self>) -> Owner {
        let thread = std::thread::current().id();
        self.owners.lock().expect("the changes' owners are never left by a panic").push(thread);
        Owner { changes: self.clone(), thread }
    }

    /// One step off the owner's thread (a slot of a bank's growth, a
    /// release on the release thread) as one change: no sample beside it
    /// stands. It begins once no settled sample waits to stand
    /// ([`Sample::settled`]), so such a sample waits for the steps
    /// under way, however long a growth of many steps runs; beside a change
    /// of the owner's, which may wait on it, it waits for none. A step its
    /// thread runs within a change under way joins it.
    pub fn back<R>(self: &Arc<Self>, step: impl FnOnce(&Changing) -> R) -> R {
        if self.open_here().is_none() {
            let meet = self.meet();
            let waiting = |meet: &mut Meet| meet.wanted > 0 && meet.owner == 0;
            drop(self.moved.wait_while(meet, waiting).expect("the changes' meet is never left by a panic"));
        }
        step(&self.begin(false))
    }
}

/// A thread's ownership of a scope's memory ([`Changes::own`]), held by a
/// load and a domain for their lives: an owner's change begun on any other
/// thread panics.
pub struct Owner {
    changes: Arc<Changes>,
    thread: std::thread::ThreadId,
}

impl Drop for Owner {
    fn drop(&mut self) {
        let mut owners = self.changes.owners.lock().expect("the changes' owners are never left by a panic");
        let at = owners.iter().position(|thread| *thread == self.thread).expect("an owner is listed until it drops");
        owners.swap_remove(at);
    }
}

/// A change held open until its holder drops, its thread owning the scope
/// meanwhile: a load while it runs, a domain while its fields drop. Every
/// free and record its thread makes meanwhile is part of it, so none is
/// made apart from its change.
#[derive(Default)]
pub struct OpenChange {
    _open: Option<(Owner, Changing)>,
}

impl OpenChange {
    /// A change of `changes`' scope on the calling thread, its owner's, open
    /// from now.
    pub fn begin(changes: &Arc<Changes>) -> Self {
        let owner = changes.own();
        Self { _open: Some((owner, changes.begin(true))) }
    }
}

thread_local! {
    /// The change each scope has under way on this thread
    /// ([`Changes::begin`]): every record this thread makes is part of it.
    static OPEN: std::cell::RefCell<Vec<(usize, u64, bool)>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A change of the scope's memory under way ([`Changes::change`],
/// [`Changes::back`], [`seismic::memory::Entry::change`]), from before its
/// OS effect until the ledger records it: no sample beside it stands. Only
/// the ledger makes one, and every record of an allocation change, a
/// holder's drop and a charge Seismic makes through an entry included, is
/// part of the one its thread has under way: one outside any panics
/// (`Locked`), so a record never lies outside the change of the effect it
/// records.
pub struct Changing {
    changes: Arc<Changes>,
    /// Its place among the scope's changes.
    pub(super) number: u64,
    /// An owner's change ([`Changes::change`]).
    owner: bool,
    /// Part of a change its thread had under way, which ends it.
    joined: bool,
}

impl Drop for Changing {
    fn drop(&mut self) {
        if self.joined {
            return;
        }
        OPEN.with_borrow_mut(|open| {
            let at = open.iter().rposition(|(key, ..)| *key == self.changes.key()).expect("a change under way is open on its thread");
            open.remove(at);
        });
        self.changes.count.fetch_sub(1, Ordering::SeqCst);
        let mut meet = self.changes.meet();
        meet.ended += 1;
        meet.owner -= usize::from(self.owner);
        drop(meet);
        self.changes.moved.notify_all();
    }
}

/// The instant a sample was taken at: its time on the owner's clock, and
/// its place among the process's own allocation changes ([`Changes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampled {
    at: Instant,
    after: u64,
}

impl Sampled {
    /// `read`'s figures and the instant they were read at, where no change
    /// of the process's was under way as they began beside the `mine` its
    /// own thread has (the figures are then not read), and none began while
    /// they were read; none otherwise.
    pub(super) fn attempt<T>(changes: &Changes, now: &impl Fn() -> Instant, read: &impl Fn() -> T, mine: u64) -> Option<(Self, T)> {
        let before = changes.count.load(Ordering::SeqCst);
        if before % ONE_CHANGE != mine {
            return None;
        }
        let at = Self { at: now(), after: before >> IN_FLIGHT_BITS };
        let figures = read();
        (changes.count.load(Ordering::SeqCst) == before).then_some((at, figures))
    }

    /// `reconcile`, which settles the driver's own memory in the ledger and
    /// returns the grant its one query read (Vulkan's
    /// `VK_EXT_memory_budget`), then [`Self::attempt`]: the reconcile is a
    /// change of its own that ends before the sample is stamped, so the
    /// sample's place counts it shown, and the grant read in its query is of
    /// the usage it charged (design §4.1, observed visibility).
    pub(super) fn reconciled<G, T>(
        changes: &Changes,
        now: &impl Fn() -> Instant,
        reconcile: impl FnOnce() -> G,
        read: &impl Fn() -> T,
    ) -> Option<(Self, G, T)> {
        let mine = u64::from(changes.open_here().is_some());
        let grant = reconcile();
        Self::attempt(changes, now, read, mine).map(|(at, figures)| (at, grant, figures))
    }

    /// [`Self::reconciled`] where it stands, waiting for the changes beside
    /// it to end where it does not: each failed try waits on the scope's
    /// signal that a change ended, and steps off the owner's thread begin
    /// none while it waits ([`Changes::back`]), so it waits for the steps
    /// under way and the entry changes begun meanwhile. The reconcile runs again with each try, so its
    /// grant is always of the instant the sample stands at.
    ///
    /// The bound, one rule for every thread that takes one: a settled
    /// sample waits for the steps under way when it began, and for the entry
    /// changes ([`Entry::change`](seismic::memory::Entry::change): a Seismic
    /// allocation or free, one native call, on any thread) that begin while
    /// it waits. An entry change cannot be held back as a step is: it never
    /// waits, since Seismic may begin one holding a lock this sample's
    /// reconcile takes, and a wait there would deadlock with it. So the
    /// wait is bounded by the steps under way plus the entry changes that
    /// overlap it, each one native call, and by no growth's length (a
    /// growth is steps, which are held back). A
    /// thread that waits here waits on no step, so the owner's change it
    /// has open (a load's, which takes its reading inside it) stops
    /// counting in [`Meet::owner`] while it waits, and steps begun
    /// meanwhile are held back ([`Changes::back`]) exactly as beside a
    /// sample with no change open. The change itself stays open (its
    /// number still counts in the sample's own place, `mine`); it counts
    /// again as the wait ends.
    fn settled<G, T>(changes: &Changes, now: &impl Fn() -> Instant, reconcile: impl Fn() -> G, read: &impl Fn() -> T) -> (Self, G, T) {
        /// The wait for a sample, let go however it ends.
        struct Wanted<'a>(&'a Changes, usize);
        impl Drop for Wanted<'_> {
            fn drop(&mut self) {
                let mut meet = self.0.meet();
                meet.wanted -= 1;
                meet.owner += self.1;
                drop(meet);
                self.0.moved.notify_all();
            }
        }
        let owned = usize::from(changes.owner_open_here());
        let mut meet = changes.meet();
        meet.wanted += 1;
        meet.owner -= owned;
        drop(meet);
        let _wanted = Wanted(changes, owned);
        let mine = u64::from(changes.open_here().is_some());
        loop {
            let grant = reconcile();
            let ended = changes.meet().ended;
            if let Some((at, figures)) = Self::attempt(changes, now, read, mine) {
                return (at, grant, figures);
            }
            let meet = changes.meet();
            drop(changes.moved.wait_while(meet, |meet| meet.ended == ended).expect("the changes' meet is never left by a panic"));
        }
    }

    /// Whether this sample was begun no later than `other`, in time and
    /// among the changes alike.
    pub(super) fn precedes(&self, other: &Self) -> bool {
        self.at <= other.at && self.after <= other.after
    }

    /// Its instant on the owner's clock.
    pub(crate) fn at(&self) -> Instant {
        self.at
    }

    /// Its place among the scope's changes: every change numbered below it
    /// was made before its figures.
    pub(crate) fn after(&self) -> u64 {
        self.after
    }
}

/// One sample of the machine at one instant: the device's status, which
/// carries host RAM's figures ([`seismic::HostSample`]) beside the device's
/// grants, and the grant the device's driver read in the query its own
/// memory was settled from, where it reads one there (Vulkan), read with no
/// change of the process's beside it, so the ledgers pair it with what the
/// process had allocated then ([`DeviceLedgers::read`]).
pub struct Sample {
    status: seismic::Status,
    /// The grant of the reconcile's query ([`Sampled::reconciled`]).
    pub(super) grant: Option<seismic::memory::Grant>,
    at: Sampled,
}

impl Sample {
    /// A sample taken now on the owner's thread at an event (a flight
    /// boundary, an own change, an OS pressure event), which never waits:
    /// `device`'s driver memory settled first and its grant kept
    /// ([`seismic::Device::reconcile_driver_memory`]), then `status` read
    /// at one stamped instant. None where a change of another thread's (a
    /// bank's resize) is beside it ([`Sampled::attempt`]). A change the
    /// owner has under way (a load) does not stop it: the owner reads
    /// between its own allocations, each recorded as it is made.
    pub fn try_now(changes: &Changes, device: &seismic::Device, now: impl Fn() -> Instant, status: impl Fn() -> seismic::Status) -> Option<Self> {
        Self::try_with(changes, || device.reconcile_driver_memory(), now, status)
    }

    pub(crate) fn try_with(
        changes: &Changes,
        reconcile: impl Fn() -> Option<seismic::memory::Grant>,
        now: impl Fn() -> Instant,
        status: impl Fn() -> seismic::Status,
    ) -> Option<Self> {
        let (at, grant, status) = Sampled::reconciled(changes, &now, &reconcile, &status)?;
        Some(Self { status, grant, at })
    }

    /// A sample taken now on the calling thread with no change of the
    /// process's beside it, as [`Self::try_now`] takes one, waiting where
    /// one is beside it until it ends ([`Sampled::settled`]): a load's, which
    /// must price against a sample that stands.
    pub fn settled(changes: &Changes, device: &seismic::Device, now: impl Fn() -> Instant, status: impl Fn() -> seismic::Status) -> Self {
        let (at, grant, status) = Sampled::settled(changes, &now, || device.reconcile_driver_memory(), &status);
        Self { status, grant, at }
    }

    /// A test's sample of host RAM as `host` records it, at its instant,
    /// taken as a load's is ([`Self::settled`]) beside `changes`: no
    /// device, so no grant read and no driver memory settled.
    #[cfg(any(test, feature = "fixtures"))]
    pub fn scripted(changes: &Changes, host: seismic::HostSample) -> Self {
        let at = host.at;
        let status = seismic::Status {
            sample: seismic::Sample { taken: std::time::SystemTime::now(), host, volumes: Vec::new(), executors: Vec::new(), grants: Vec::new() },
            runtime: seismic::RuntimeStatus::default(),
        };
        let (at, grant, status) = Sampled::settled(changes, &|| at, || None, &|| status.clone());
        Self { status, grant, at }
    }

    pub fn status(&self) -> &seismic::Status {
        &self.status
    }

    /// Host RAM at the sample's instant.
    pub fn host(&self) -> &seismic::HostSample {
        &self.status.sample.host
    }

    pub fn at(&self) -> Sampled {
        self.at
    }
}

/// A scripted test's scope: its own stores, its own clock.
#[cfg(any(test, feature = "fixtures"))]
impl Ledgers {
    pub(crate) fn scripted(clock: impl Fn() -> Instant + Send + Sync + 'static) -> Self {
        Self {
            stores: Mutex::default(),
            changes: Arc::new(Changes { alone: std::sync::atomic::AtomicBool::new(true), clock: Box::new(clock), ..Changes::default() }),
            footprints: Arc::new(super::store::Unobserved),
        }
    }
}

#[cfg(test)]
impl Ledger {
    /// A test's sample of this store taken now, at `at` on its clock, with
    /// no change beside it. Every other sample is [`Sample::try_now`]'s.
    #[cfg(test)]
    pub(crate) fn sampled(&self, at: Instant) -> Sampled {
        Sampled { at, after: self.store.changes.count.load(Ordering::SeqCst) >> IN_FLIGHT_BITS }
    }
}
