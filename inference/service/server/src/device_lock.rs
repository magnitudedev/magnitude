//! The exclusions on the device between residency and the jobs that search or measure kernels
//! on it (device topology spec §2.15). Loads share the device for their lifetime. A job (the
//! measurement job, or a catalog installation's optimization) runs only with the device to
//! itself: a search beside a resident model would take the memory that model's verdict promised
//! its requests, and would time kernels against its serving. A job never queues ahead of a load
//! as a waiting writer would: it tries to take the device and waits for a release to try again,
//! so a job deferred by a resident model runs once the device is free. A share wanted while a job
//! holds the device preempts the job: its owner kills it at once, so a load waits only for that
//! kill's process to be shown gone.
//!
//! The measurement job alone may also run beside a resident model while that model is idle (no
//! request holds it), in the model's idle time: residency publishes the idle model
//! ([`DeviceLock::resident_idle`]), whose worker the job asks for room first
//! ([`ResidentRoom`]). Such a job holds the device as a job does against every other share and
//! job, and is preempted as one is, and also by any change of the idle model: a request leasing
//! it, or its release. The model's idleness and the job's start are one decision, so no job
//! starts once a request holds the model, and a request is served only once the job it
//! preempted is gone ([`DeviceLock::admit_request`]): no point is timed beside a request.

use std::sync::Arc;
use std::time::Instant;

use futures_util::future::BoxFuture;
use magnitude_engine::error::RequestError;
use magnitude_engine::worker::protocol::RoomAnswer;
use magnitude_service_contracts::models::ModelInstanceId;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, watch};

use crate::dying_process::{self, DyingProcess};

/// What a measurement beside an idle resident model asks of that model's worker.
pub trait ResidentRoom: Send + Sync {
    /// Make `bytes` of device memory free beside the model, shrinking what it holds toward its
    /// floor: answered once made, with why it cannot be, or released by [`Self::release`] first;
    /// failing once the model is gone. The model grows nothing until [`Self::release`].
    fn claim(&self, bytes: u64) -> BoxFuture<'static, Result<RoomAnswer, RequestError>>;
    /// The measurement ended: the model may grow into the room again.
    fn release(&self);
    /// The measurement stored new records: the model prices its experts again from them.
    fn reprice(&self);
}

/// The resident model while no request holds it: the load it is, and what a measurement beside
/// it asks of its worker.
#[derive(Clone)]
pub struct IdleResident {
    pub instance: ModelInstanceId,
    pub room: Arc<dyn ResidentRoom>,
}

/// The device exclusion, shared by residency and the jobs. `released` counts the holds that
/// ended, shares and jobs alike, so a job waiting for the device tries again after each.
#[derive(Clone)]
pub struct DeviceLock {
    lock: Arc<RwLock<()>>,
    released: Arc<watch::Sender<u64>>,
    /// How many shares have waited on a job: each one preempts the job holding the device then.
    wanted: Arc<watch::Sender<u64>>,
    /// The shares held, the idle resident model, and whether a measurement runs beside it.
    sharing: Arc<watch::Sender<Sharing>>,
}

/// The shares held, the resident model while no request holds it, and the job holding the
/// device, to itself or beside that model: while one does, no share is taken.
#[derive(Clone, Default)]
struct Sharing {
    shares: usize,
    idle: Option<IdleResident>,
    /// Counts the idle model's changes: each preempts a measurement beside it.
    idleness: u64,
    job: Option<HeldJob>,
    /// Host memory is short ([`DeviceLock::hold_off_jobs`]): no job starts, and a running one
    /// gives way before any model does.
    held_off: bool,
}

/// A job holding the device, as far as host memory pressure reads it.
#[derive(Clone, Debug)]
enum HeldJob {
    /// Running, or told to end by its preemption and not yet killed: its owner kills it at once.
    Live,
    /// Its process has been killed: it gives way as the system reports its death
    /// ([`DyingProcess`]).
    Dying(DyingProcess),
    /// The system refused its kill or its reap ([`DeviceJob::unretired`]): it keeps the device
    /// until a retry shows it gone, and gives no memory back meanwhile.
    Unretired,
}

impl Default for DeviceLock {
    fn default() -> Self {
        Self {
            lock: Arc::default(),
            released: Arc::new(watch::Sender::new(0)),
            wanted: Arc::new(watch::Sender::new(0)),
            sharing: Arc::new(watch::Sender::new(Sharing::default())),
        }
    }
}

/// Residency's hold on the device: a load's, held by every worker it starts until that worker
/// is proven retired.
pub struct DeviceShare {
    /// Held until the share ends; `None` only inside `drop`.
    guard: Option<OwnedRwLockReadGuard<()>>,
    released: Arc<watch::Sender<u64>>,
    sharing: Arc<watch::Sender<Sharing>>,
}

impl Drop for DeviceShare {
    /// Release the share, then wake a waiting job: its next try sees the device released.
    fn drop(&mut self) {
        self.sharing.send_modify(|sharing| sharing.shares -= 1);
        drop(self.guard.take());
        self.released
            .send_modify(|released| *released = released.wrapping_add(1));
    }
}

/// A job's hold on the device: nothing of residency runs on it, but an idle resident model
/// beside the measurement job.
pub struct DeviceJob {
    hold: JobHold,
    sharing: Arc<watch::Sender<Sharing>>,
    released: Arc<watch::Sender<u64>>,
    preemption: Preemption,
}

enum JobHold {
    /// The device to itself; `None` only inside `drop`.
    Exclusive(Option<OwnedRwLockWriteGuard<()>>),
    /// Beside the idle resident model's share.
    Beside { resident: Arc<dyn ResidentRoom> },
}

impl Drop for DeviceJob {
    /// Release the device, then wake another waiting job, and a request waiting on the job. Beside
    /// an idle resident model, the room its worker made is the model's again
    /// ([`ResidentRoom::release`]) before a request waiting on the job is admitted.
    fn drop(&mut self) {
        match &mut self.hold {
            JobHold::Exclusive(guard) => {
                self.sharing.send_modify(|sharing| sharing.job = None);
                drop(guard.take());
            }
            JobHold::Beside { resident } => {
                resident.release();
                self.sharing.send_modify(|sharing| sharing.job = None);
            }
        }
        self.released
            .send_modify(|released| *released = released.wrapping_add(1));
    }
}

impl DeviceJob {
    /// Return once a share is wanted while the job holds the device, or, beside an idle resident
    /// model, once that model is no longer idle: the job is preempted.
    pub async fn preempted(&mut self) {
        self.preemption.preempted().await;
    }

    /// The job's preemption apart from its hold, for a hold its worker keeps until retired.
    pub fn preemption(&self) -> Preemption {
        self.preemption.clone()
    }

    /// The idle resident model the job runs beside; none when it has the device to itself.
    pub fn beside(&self) -> Option<&Arc<dyn ResidentRoom>> {
        match &self.hold {
            JobHold::Exclusive(_) => None,
            JobHold::Beside { resident } => Some(resident),
        }
    }

    /// The system refused the job's process its kill or its reap: it keeps the device until a
    /// retry shows it gone, and a host memory emergency spares the resident model for it no
    /// longer ([`DeviceLock::job_giving_way`]).
    pub fn unretired(&self) {
        self.sharing.send_if_modified(|sharing| {
            let marked = matches!(sharing.job, Some(HeldJob::Live | HeldJob::Dying(_)));
            if marked {
                sharing.job = Some(HeldJob::Unretired);
            }
            marked
        });
    }

    /// The job's process `pid` is being killed: host memory pressure judges it by its death
    /// from here on ([`DeviceLock::job_giving_way`]). One the system refused stays so: a retry
    /// is not a death it shows. Nothing waits on this, so no one is woken.
    pub fn dying(&self, pid: u32) {
        self.sharing.send_if_modified(|sharing| {
            if matches!(sharing.job, Some(HeldJob::Live)) {
                sharing.job = Some(HeldJob::Dying(DyingProcess::new(pid)));
            }
            false
        });
    }
}

/// When the shares wanted since a job took the device preempt it, and, beside an idle resident
/// model, any change of the idle model since.
#[derive(Clone)]
pub struct Preemption {
    wanted: watch::Receiver<u64>,
    /// Beside an idle model: the device's sharing, and its idleness last seen, first when the
    /// job took the device.
    idle: Option<(watch::Receiver<Sharing>, u64)>,
}

impl Preemption {
    /// Return once a share is wanted, or the idle model the job runs beside changed, since the
    /// job took the device or since this last returned: each preemption is seen once, as a
    /// change of a watch is.
    pub async fn preempted(&mut self) {
        match &mut self.idle {
            None => changed(&mut self.wanted).await,
            Some((sharing, idleness)) => {
                let seen = *idleness;
                tokio::select! {
                    () = changed(&mut self.wanted) => {}
                    // The sender lives as long as the lock the hold is of.
                    changed = sharing.wait_for(|sharing| sharing.idleness != seen) => {
                        if let Ok(sharing) = changed {
                            *idleness = sharing.idleness;
                        }
                    }
                }
            }
        }
    }
}

/// Return once `receiver` changes. Its sender lives as long as the lock the hold is of, so a
/// change is the only end.
async fn changed<T>(receiver: &mut watch::Receiver<T>) {
    if receiver.changed().await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// The device's changes a measurement owner waits for: a hold ended, or a resident model became
/// idle that has not been (a load ended, and its model is resident). A request's lease and
/// its end are none: the load they lease is the same. What changed in between is never missed:
/// each is seen at the next wait.
pub struct DeviceChanges {
    released: watch::Receiver<u64>,
    sharing: watch::Receiver<Sharing>,
    /// The load last seen idle.
    idle: Option<ModelInstanceId>,
}

impl DeviceChanges {
    /// Return at the next change since the last; never once the lock is gone.
    pub async fn changed(&mut self) {
        loop {
            tokio::select! {
                () = changed(&mut self.released) => return,
                () = changed(&mut self.sharing) => {
                    let idle = self.sharing.borrow().idle.as_ref().map(|idle| idle.instance.clone());
                    if idle.is_some() && idle != self.idle {
                        self.idle = idle;
                        return;
                    }
                }
            }
        }
    }
}

/// What an inference worker holds of the device until it is proven retired: its load's share,
/// or its optimization's job.
pub enum DeviceHold {
    Share(Arc<DeviceShare>),
    Job(DeviceJob),
}

impl DeviceHold {
    /// The system refused the worker's process its kill or its reap: a job's hold says so
    /// ([`DeviceJob::unretired`]); a load's share is no job's.
    pub fn unretired(&self) {
        if let Self::Job(job) = self {
            job.unretired();
        }
    }

    /// The worker's process `pid` is being killed: a job's hold judges it by its death
    /// ([`DeviceJob::dying`]); a load's share is no job's.
    pub fn dying(&self, pid: u32) {
        if let Self::Job(job) = self {
            job.dying(pid);
        }
    }
}

impl DeviceLock {
    /// Share the device, without waiting: `None` while a job holds it.
    pub fn try_share(&self) -> Option<DeviceShare> {
        let guard = Arc::clone(&self.lock).try_read_owned().ok()?;
        self.share_of(guard)
    }

    /// Share the device, preempting a job that holds it: the job's owner kills it and releases
    /// the device once its process is shown gone, so this waits for that.
    pub async fn share(&self) -> DeviceShare {
        if let Some(share) = self.try_share() {
            return share;
        }
        self.wanted.send_modify(|wanted| *wanted += 1);
        self.share_after_measurement().await
    }

    /// Share the device once no measurement job holds it, without preempting one: for work that
    /// would rather wait for a job, bounded by its hang deadline, than end it.
    pub async fn share_after_measurement(&self) -> DeviceShare {
        let mut sharing = self.sharing.subscribe();
        loop {
            let guard = Arc::clone(&self.lock).read_owned().await;
            if let Some(share) = self.share_of(guard) {
                return share;
            }
            // The sender lives as long as this lock.
            let _ = sharing.wait_for(|sharing| sharing.job.is_none()).await;
        }
    }

    /// A share of `guard`, unless a measurement runs beside the idle resident model (`guard`
    /// excludes a job with the device to itself): decided with the count of shares, which that
    /// measurement's start reads.
    fn share_of(&self, guard: OwnedRwLockReadGuard<()>) -> Option<DeviceShare> {
        let mut shared = false;
        self.sharing.send_if_modified(|sharing| {
            shared = sharing.job.is_none();
            if shared {
                sharing.shares += 1;
            }
            shared
        });
        shared.then(|| DeviceShare {
            guard: Some(guard),
            released: Arc::clone(&self.released),
            sharing: Arc::clone(&self.sharing),
        })
    }

    /// The resident model is idle (no request holds it), and a measurement may run beside it, or
    /// it no longer is (`None`): every change preempts a measurement beside it, and from a
    /// `None` on none starts. Residency's alone to publish, before it hands out a lease.
    pub fn resident_idle(&self, resident: Option<IdleResident>) {
        self.sharing.send_modify(|sharing| {
            sharing.idle = resident;
            sharing.idleness = sharing.idleness.wrapping_add(1);
        });
    }

    /// Return once no measurement runs beside the resident model: a request holding a lease on
    /// it, which ended its idleness and so preempted such a measurement, is served only then,
    /// once the measurement's process is shown gone, which its owner waits for.
    pub async fn admit_request(&self) {
        // The sender lives as long as this lock.
        let _ = self.sharing.subscribe().wait_for(|sharing| sharing.job.is_none()).await;
    }

    /// Hold jobs off while host memory is short (`held`), or no longer: the memory supervisor's
    /// alone to say. Held off, no job starts, and one running is preempted at once: a job gives
    /// way before any model does ([`Self::job_giving_way`]). Whether this call preempted a
    /// job: only the hold's start preempts, so a job still ending from it is not preempted
    /// again.
    pub fn hold_off_jobs(&self, held: bool) -> bool {
        let mut preempted = false;
        self.sharing.send_if_modified(|sharing| {
            let changed = sharing.held_off != held;
            preempted = changed && held && sharing.job.is_some();
            sharing.held_off = held;
            changed
        });
        if preempted {
            self.wanted.send_modify(|wanted| *wanted += 1);
        }
        preempted
    }

    /// Whether a job holds the device that is still giving way at `now`, the host at `headroom`:
    /// running or about to be killed, or killed and not shown hung ([`DyingProcess`]), its kill or
    /// reap not refused by the system ([`DeviceJob::unretired`]). While one does, host memory
    /// pressure waits for what it frees before taking a model.
    pub fn job_giving_way(&self, now: Instant, headroom: u64) -> bool {
        // The look is taken outside the watch's lock, which loads and jobs take.
        let pid = match &self.sharing.borrow().job {
            Some(HeldJob::Live) => return true,
            None | Some(HeldJob::Unretired) => return false,
            Some(HeldJob::Dying(dying)) => dying.pid,
        };
        let look = dying_process::look(pid);
        let mut giving_way = true;
        self.sharing.send_if_modified(|sharing| {
            // A job gone or refused since the look is judged at the next sample, on memory read
            // after it.
            if let Some(HeldJob::Dying(dying)) = &mut sharing.job {
                giving_way = dying.giving_way(look, now, headroom);
            }
            false
        });
        giving_way
    }

    /// The device's changes from now on ([`DeviceChanges`]).
    pub fn changes(&self) -> DeviceChanges {
        let sharing = self.sharing.subscribe();
        let idle = sharing.borrow().idle.as_ref().map(|idle| idle.instance.clone());
        DeviceChanges {
            released: self.released.subscribe(),
            sharing,
            idle,
        }
    }

    /// Take the device for a job, without waiting: `None` while residency shares it or another
    /// job holds it. Only shares wanted from here on preempt the job.
    pub fn try_job(&self) -> Option<DeviceJob> {
        // Held off, the write guard is not even tried, so no share is refused for it.
        if self.sharing.borrow().held_off {
            return None;
        }
        let preemption = Preemption {
            wanted: self.wanted.subscribe(),
            idle: None,
        };
        let guard = Arc::clone(&self.lock).try_write_owned().ok()?;
        // A measurement beside a resident model whose share has ended holds the device until it
        // ends; no share starts while the write guard is held, so none starts beside it.
        let mut taken = false;
        self.sharing.send_if_modified(|sharing| {
            taken = sharing.job.is_none() && !sharing.held_off;
            if taken {
                sharing.job = Some(HeldJob::Live);
            }
            taken
        });
        taken.then(|| DeviceJob {
            hold: JobHold::Exclusive(Some(guard)),
            sharing: Arc::clone(&self.sharing),
            released: Arc::clone(&self.released),
            preemption,
        })
    }

    /// Take the device for the measurement job beside the idle resident model, without waiting:
    /// `None` unless that model is idle, its share is the one held, and nothing else runs beside
    /// it, all read in one decision with the model's idleness.
    fn try_beside(&self) -> Option<DeviceJob> {
        let wanted = self.wanted.subscribe();
        let mut taken = None;
        self.sharing.send_if_modified(|sharing| {
            if let (Some(idle), 1, None, false) = (&sharing.idle, sharing.shares, &sharing.job, sharing.held_off) {
                taken = Some((Arc::clone(&idle.room), sharing.idleness));
                sharing.job = Some(HeldJob::Live);
            }
            taken.is_some()
        });
        let (resident, idleness) = taken?;
        Some(DeviceJob {
            hold: JobHold::Beside { resident },
            sharing: Arc::clone(&self.sharing),
            released: Arc::clone(&self.released),
            preemption: Preemption {
                wanted,
                idle: Some((self.sharing.subscribe(), idleness)),
            },
        })
    }

    /// Take the device for the measurement job once nothing holds it, or once the only share
    /// held is an idle resident model's, beside which it runs ([`DeviceJob::beside`]), trying
    /// again after each release and each change of the device's sharing.
    pub async fn measurement(&self) -> DeviceJob {
        let mut released = self.released.subscribe();
        let mut sharing = self.sharing.subscribe();
        loop {
            if let Some(job) = self.try_job().or_else(|| self.try_beside()) {
                return job;
            }
            tokio::select! {
                () = changed(&mut released) => {}
                () = changed(&mut sharing) => {}
            }
        }
    }

    /// Take the device for a job once nothing holds it, trying again after each release: a job
    /// defers this way while a model is resident or loading, or another job holds the device.
    pub async fn job(&self) -> DeviceJob {
        let mut released = self.released.subscribe();
        let mut sharing = self.sharing.subscribe();
        loop {
            if let Some(job) = self.try_job() {
                return job;
            }
            // Every hold's end changes `released` after its release, and jobs held off and let go
            // again change `sharing`; one since the subscriptions is not missed.
            tokio::select! {
                () = changed(&mut released) => {}
                () = changed(&mut sharing) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job excludes shares and shares exclude a job; a share ending lets a waiting job take
    /// the device.
    #[tokio::test]
    async fn shares_and_a_job_exclude_each_other() {
        let lock = DeviceLock::default();
        let share = lock.try_share().unwrap();
        assert!(lock.try_job().is_none());
        let waiting = tokio::spawn({
            let lock = lock.clone();
            async move { lock.job().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        drop(share);
        let job = waiting.await.unwrap();
        assert!(lock.try_share().is_none());
        drop(job);
        assert!(lock.try_share().is_some());
    }

    /// A share wanted while a job holds the device preempts it, and is granted once the job
    /// releases the device; a share taken while no job holds it preempts nothing.
    #[tokio::test]
    async fn a_wanted_share_preempts_the_job() {
        let lock = DeviceLock::default();
        drop(lock.share().await);
        let mut job = lock.try_job().unwrap();
        let waiting = tokio::spawn({
            let lock = lock.clone();
            async move { lock.share().await }
        });
        job.preempted().await;
        drop(job);
        let share = waiting.await.unwrap();
        assert!(lock.try_job().is_none());
        drop(share);
    }

    /// Another job holding the device is waited out, and the preemption a worker's owner keeps
    /// apart from its hold fires when a share is wanted.
    #[tokio::test]
    async fn a_job_waits_out_another_job_and_is_preempted_by_a_load() {
        let lock = DeviceLock::default();
        let measuring = lock.try_job().unwrap();
        let waiting = tokio::spawn({
            let lock = lock.clone();
            async move { lock.job().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        drop(measuring);
        let job = waiting.await.unwrap();
        let mut preemption = job.preemption();
        let hold = DeviceHold::Job(job);
        let load = tokio::spawn({
            let lock = lock.clone();
            async move { lock.share().await }
        });
        preemption.preempted().await;
        drop(hold);
        drop(load.await.unwrap());
    }

    /// `room`, idle as the load `instance`.
    fn idle(instance: &str, room: Arc<dyn ResidentRoom>) -> Option<IdleResident> {
        Some(IdleResident { instance: ModelInstanceId(instance.to_owned()), room })
    }

    /// A resident model that answers every claim at once.
    struct Room;

    impl ResidentRoom for Room {
        fn claim(&self, _: u64) -> BoxFuture<'static, Result<RoomAnswer, RequestError>> {
            Box::pin(async { Ok(RoomAnswer::Made) })
        }
        fn release(&self) {}
        fn reprice(&self) {}
    }

    /// A request ending the model's idleness preempts the measurement beside it and is admitted
    /// only once that measurement's hold ends; once a request holds the model, no measurement
    /// starts beside it.
    #[tokio::test]
    async fn a_request_is_admitted_only_once_the_measurement_beside_it_ends() {
        let lock = DeviceLock::default();
        let _resident = lock.try_share().unwrap();
        lock.resident_idle(idle("load", Arc::new(Room)));
        let mut job = lock.measurement().await;
        assert!(job.beside().is_some());
        // A lease: the model is no longer idle.
        lock.resident_idle(None);
        let admitted = tokio::spawn({
            let lock = lock.clone();
            async move { lock.admit_request().await }
        });
        job.preempted().await;
        tokio::task::yield_now().await;
        assert!(!admitted.is_finished(), "admitted while the measurement holds the device");
        drop(job);
        admitted.await.unwrap();
        assert!(lock.try_beside().is_none(), "no measurement starts beside a leased model");
    }

    /// A preemption is seen once: once the idle model's change preempted the measurement beside
    /// it, waiting again waits for a further change, so a retirement retried on the preemption
    /// sleeps between its tries.
    #[tokio::test]
    async fn a_measurement_beside_the_idle_model_is_preempted_once_per_change() {
        use futures_util::FutureExt;
        let lock = DeviceLock::default();
        let _resident = lock.try_share().unwrap();
        lock.resident_idle(idle("load", Arc::new(Room)));
        let mut job = lock.measurement().await;
        assert!(job.preempted().now_or_never().is_none(), "preempted before any change");
        lock.resident_idle(None);
        assert!(job.preempted().now_or_never().is_some(), "the lease preempts it");
        assert!(job.preempted().now_or_never().is_none(), "the same lease preempts it again");
        lock.resident_idle(idle("load", Arc::new(Room)));
        assert!(job.preempted().now_or_never().is_some(), "a further change preempts it");
    }

    /// A load that becomes idle is a change once: a request's lease and its end are none, however
    /// its worker's room is allocated, and the next load is one whatever the last left behind.
    #[tokio::test]
    async fn each_load_that_becomes_idle_is_one_change() {
        use std::time::Duration;
        let lock = DeviceLock::default();
        let mut changes = lock.changes();
        let room: Arc<dyn ResidentRoom> = Arc::new(Room);
        lock.resident_idle(idle("first", Arc::clone(&room)));
        changes.changed().await;
        lock.resident_idle(None);
        lock.resident_idle(idle("first", Arc::new(Room)));
        let waited = tokio::time::timeout(Duration::from_millis(20), changes.changed()).await;
        assert!(waited.is_err(), "a request's lease and its end are no change");
        lock.resident_idle(None);
        // The same room, as an allocator may hand a later load the last one's address.
        lock.resident_idle(idle("second", room));
        changes.changed().await;
    }

    /// A resident model that counts its room's releases.
    #[derive(Default)]
    struct Releases(std::sync::atomic::AtomicUsize);

    impl ResidentRoom for Releases {
        fn claim(&self, _: u64) -> BoxFuture<'static, Result<RoomAnswer, RequestError>> {
            Box::pin(async { Ok(RoomAnswer::Made) })
        }
        fn release(&self) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        fn reprice(&self) {}
    }

    /// The measurement's room is the model's again before a request waiting on the measurement
    /// is admitted: the hold's end releases it, once.
    #[tokio::test]
    async fn the_measurement_room_is_released_before_a_request_is_admitted() {
        let lock = DeviceLock::default();
        let _resident = lock.try_share().unwrap();
        let room = Arc::new(Releases::default());
        lock.resident_idle(idle("load", room.clone()));
        let job = lock.measurement().await;
        lock.resident_idle(None);
        let admitted = tokio::spawn({
            let (lock, room) = (lock.clone(), room.clone());
            async move {
                lock.admit_request().await;
                room.0.load(std::sync::atomic::Ordering::SeqCst)
            }
        });
        tokio::task::yield_now().await;
        assert_eq!(room.0.load(std::sync::atomic::Ordering::SeqCst), 0, "released while held");
        drop(job);
        assert_eq!(admitted.await.unwrap(), 1, "admitted before its room was released");
    }
}
