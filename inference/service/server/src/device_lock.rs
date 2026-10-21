//! The one exclusion between residency and the measurement job (device topology spec §2.15). A
//! model load and catalog optimization share the device for their lifetime; the measurement
//! job measures only with the device to itself, and only ever tries to take it, so it never
//! queues ahead of a load as a waiting writer would. A share wanted while a job holds the device
//! preempts the job: its owner kills it at once, so a load waits only for that bounded kill.

use std::sync::Arc;

use tokio::sync::{Notify, OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock, watch};

/// The device exclusion, shared by residency and the measurement job.
#[derive(Clone)]
pub struct DeviceLock {
    lock: Arc<RwLock<()>>,
    /// Woken when a share ends, so the job can try again once residency is idle.
    released: Arc<Notify>,
    /// How many shares have waited on a measurement job: each one preempts the job holding the
    /// device then.
    wanted: Arc<watch::Sender<u64>>,
}

impl Default for DeviceLock {
    fn default() -> Self {
        Self {
            lock: Arc::default(),
            released: Arc::default(),
            wanted: Arc::new(watch::Sender::new(0)),
        }
    }
}

/// Residency's hold on the device: a load's or an optimization's, held by every worker it
/// starts until that worker is proven retired.
pub struct DeviceShare {
    /// Held until the share ends; `None` only inside `drop`.
    guard: Option<OwnedRwLockReadGuard<()>>,
    released: Arc<Notify>,
}

impl Drop for DeviceShare {
    /// Release the share, then wake the job: a woken job's try sees the device released.
    fn drop(&mut self) {
        drop(self.guard.take());
        self.released.notify_one();
    }
}

/// The measurement job's hold on the device: nothing of residency runs on it.
pub struct DeviceMeasurement {
    _guard: OwnedRwLockWriteGuard<()>,
    /// The shares wanted since the job took the device.
    wanted: watch::Receiver<u64>,
}

impl DeviceMeasurement {
    /// Return once a share is wanted while the job holds the device: the job is preempted.
    pub async fn preempted(&mut self) {
        // The sender lives as long as the lock this hold is of, so a change is the only end.
        if self.wanted.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

impl DeviceLock {
    /// Share the device, without waiting: `None` while a measurement job holds it.
    pub fn try_share(&self) -> Option<DeviceShare> {
        let guard = Arc::clone(&self.lock).try_read_owned().ok()?;
        Some(self.share_of(guard))
    }

    /// Share the device, preempting a measurement job that holds it: the job's owner kills it
    /// and releases the device after its bounded exit wait, so this waits no longer than that.
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

    /// Take the device for a measurement job, without waiting: `None` while residency shares
    /// it. Only shares wanted from here on preempt the job.
    pub fn try_measure(&self) -> Option<DeviceMeasurement> {
        let wanted = self.wanted.subscribe();
        Arc::clone(&self.lock)
            .try_write_owned()
            .ok()
            .map(|guard| DeviceMeasurement {
                _guard: guard,
                wanted,
            })
    }

    /// Return when a share ends, after its release: residency may be idle. A share that ends
    /// before this is awaited is not missed, as its wake is kept for the next wait.
    pub async fn released(&self) {
        self.released.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A measurement excludes shares and shares exclude a measurement; a share ending wakes a
    /// job waiting to try again.
    #[tokio::test]
    async fn shares_and_a_measurement_exclude_each_other() {
        let lock = DeviceLock::default();
        let share = lock.try_share().unwrap();
        assert!(lock.try_measure().is_none());
        drop(share);
        lock.released().await;
        let measurement = lock.try_measure().unwrap();
        assert!(lock.try_share().is_none());
        drop(measurement);
        assert!(lock.try_share().is_some());
    }

    /// A share wanted while a job holds the device preempts it, and is granted once the job
    /// releases the device; a share taken while no job holds it preempts nothing.
    #[tokio::test]
    async fn a_wanted_share_preempts_the_job() {
        let lock = DeviceLock::default();
        drop(lock.share().await);
        let mut measurement = lock.try_measure().unwrap();
        let waiting = tokio::spawn({
            let lock = lock.clone();
            async move { lock.share().await }
        });
        measurement.preempted().await;
        drop(measurement);
        let share = waiting.await.unwrap();
        assert!(lock.try_measure().is_none());
        drop(share);
    }
}
