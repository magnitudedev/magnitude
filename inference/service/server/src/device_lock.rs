//! The exclusions on the device between residency and the jobs that search or measure kernels
//! on it (device topology spec §2.15). Loads share the device for their lifetime. A job (the
//! measurement job, or a catalog installation's optimization) runs only with the device to
//! itself: a search beside a resident model would take the memory that model's verdict promised
//! its requests, and would time kernels against its serving. A job never queues ahead of a load
//! as a waiting writer would: it tries to take the device and waits for a release to try again,
//! and an optimization is skipped while residency holds the device. A share wanted while a job
//! holds the device preempts the job: its owner kills it at once, so a load waits only for that
//! bounded kill.

use std::sync::Arc;

use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, watch};

/// The device exclusion, shared by residency and the jobs. `released` counts the holds that
/// ended, shares and jobs alike, so a job waiting for the device tries again after each.
#[derive(Clone)]
pub struct DeviceLock {
    lock: Arc<RwLock<()>>,
    released: Arc<watch::Sender<u64>>,
    /// How many shares have waited on a job: each one preempts the job holding the device then.
    wanted: Arc<watch::Sender<u64>>,
}

impl Default for DeviceLock {
    fn default() -> Self {
        Self {
            lock: Arc::default(),
            released: Arc::new(watch::Sender::new(0)),
            wanted: Arc::new(watch::Sender::new(0)),
        }
    }
}

/// Residency's hold on the device: a load's, held by every worker it starts until that worker
/// is proven retired.
pub struct DeviceShare {
    /// Held until the share ends; `None` only inside `drop`.
    guard: Option<OwnedRwLockReadGuard<()>>,
    released: Arc<watch::Sender<u64>>,
}

impl Drop for DeviceShare {
    /// Release the share, then wake a waiting job: its next try sees the device released.
    fn drop(&mut self) {
        drop(self.guard.take());
        self.released
            .send_modify(|released| *released = released.wrapping_add(1));
    }
}

/// A job's hold on the device: nothing of residency runs on it.
pub struct DeviceJob {
    guard: Option<OwnedRwLockWriteGuard<()>>,
    released: Arc<watch::Sender<u64>>,
    preemption: Preemption,
}

impl Drop for DeviceJob {
    /// Release the device, then wake another waiting job.
    fn drop(&mut self) {
        drop(self.guard.take());
        self.released
            .send_modify(|released| *released = released.wrapping_add(1));
    }
}

impl DeviceJob {
    /// Return once a share is wanted while the job holds the device: the job is preempted.
    pub async fn preempted(&mut self) {
        self.preemption.preempted().await;
    }

    /// The job's preemption apart from its hold, for a hold its worker keeps until retired.
    pub fn preemption(&self) -> Preemption {
        self.preemption.clone()
    }
}

/// When the shares wanted since a job took the device preempt it.
#[derive(Clone)]
pub struct Preemption(watch::Receiver<u64>);

impl Preemption {
    /// Return once a share is wanted after the job took the device.
    pub async fn preempted(&mut self) {
        // The sender lives as long as the lock the hold is of, so a change is the only end.
        if self.0.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

/// What an inference worker holds of the device until it is proven retired: its load's share,
/// or its optimization's job.
pub enum DeviceHold {
    Share(Arc<DeviceShare>),
    Job(DeviceJob),
}

impl DeviceLock {
    /// Share the device, without waiting: `None` while a job holds it.
    pub fn try_share(&self) -> Option<DeviceShare> {
        let guard = Arc::clone(&self.lock).try_read_owned().ok()?;
        Some(self.share_of(guard))
    }

    /// Share the device, preempting a job that holds it: the job's owner kills it and releases
    /// the device after its bounded exit wait, so this waits no longer than that.
    pub async fn share(&self) -> DeviceShare {
        if let Some(share) = self.try_share() {
            return share;
        }
        self.wanted.send_modify(|wanted| *wanted += 1);
        let guard = Arc::clone(&self.lock).read_owned().await;
        self.share_of(guard)
    }

    /// Share the device once no measurement job holds it, without preempting one: for work that
    /// would rather wait for a job, bounded by its hang deadline, than end it.
    pub async fn share_after_measurement(&self) -> DeviceShare {
        let guard = Arc::clone(&self.lock).read_owned().await;
        self.share_of(guard)
    }

    fn share_of(&self, guard: OwnedRwLockReadGuard<()>) -> DeviceShare {
        DeviceShare {
            guard: Some(guard),
            released: Arc::clone(&self.released),
        }
    }

    /// Take the device for a job, without waiting: `None` while residency shares it or another
    /// job holds it. Only shares wanted from here on preempt the job.
    pub fn try_job(&self) -> Option<DeviceJob> {
        let preemption = Preemption(self.wanted.subscribe());
        Arc::clone(&self.lock)
            .try_write_owned()
            .ok()
            .map(|guard| DeviceJob {
                guard: Some(guard),
                released: Arc::clone(&self.released),
                preemption,
            })
    }

    /// Take the device for a job once nothing holds it, trying again after each release: the
    /// measurement job defers this way while a model is resident or loading.
    pub async fn job(&self) -> DeviceJob {
        let mut released = self.released.subscribe();
        loop {
            if let Some(job) = self.try_job() {
                return job;
            }
            // Every hold's end changes this after its release; one that ended since the
            // subscription is not missed.
            let _ = released.changed().await;
        }
    }

    /// Take the device for a job unless residency holds it: `None` while a model is resident
    /// or loading. Another job holding the device is waited out.
    pub async fn job_unless_resident(&self) -> Option<DeviceJob> {
        let mut released = self.released.subscribe();
        loop {
            if let Some(job) = self.try_job() {
                return Some(job);
            }
            // The take failed with no job holding the device: residency shares it.
            if Arc::clone(&self.lock).try_read_owned().is_ok() {
                return None;
            }
            let _ = released.changed().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Longer than any take that does not wait.
    const TRY: Duration = Duration::from_secs(5);

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

    /// An optimization never runs beside a resident or loading model: while residency shares
    /// the device it is skipped, not queued behind the share.
    #[tokio::test]
    async fn a_job_unless_resident_is_refused_while_residency_shares_the_device() {
        let lock = DeviceLock::default();
        let share = lock.try_share().unwrap();
        let refused = tokio::time::timeout(TRY, lock.job_unless_resident())
            .await
            .expect("the job never waits behind a resident model");
        assert!(refused.is_none());
        drop(share);
        assert!(lock.job_unless_resident().await.is_some());
    }

    /// Another job holding the device is waited out, and the preemption a worker's owner keeps
    /// apart from its hold fires when a share is wanted.
    #[tokio::test]
    async fn a_job_unless_resident_waits_out_another_job_and_is_preempted_by_a_load() {
        let lock = DeviceLock::default();
        let measuring = lock.try_job().unwrap();
        let waiting = tokio::spawn({
            let lock = lock.clone();
            async move { lock.job_unless_resident().await }
        });
        tokio::task::yield_now().await;
        assert!(!waiting.is_finished());
        drop(measuring);
        let job = waiting.await.unwrap().expect("no model holds the device");
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
}
