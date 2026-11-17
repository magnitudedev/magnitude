//! The CPUs the service's processes may keep busy, and the share of them each process the
//! service starts may run its auxiliary threads on. An engine process's auxiliary pools (memory
//! fault-in, native-target workers, repack, formation) size themselves by its share
//! (`seismic::threads::share`), which the process is handed at its spawn ([`ThreadShare`]): the
//! serving worker the whole quota, a background job (the measurement job, a catalog
//! optimization) all but [`SERVING_RESERVE`] of it. The reserve does not reach a device's own
//! pools (the CPU device's, the expert-compute pool): their size is part of the device's
//! identity, which a job measures and serving then opens, so in the measurement job the reserve
//! reaches only its auxiliary pools. The service sets its own process's share to background work
//! at startup, which sizes its catalog assessment, inventory scans and the engine pools it opens.

use std::num::NonZeroUsize;
use std::sync::LazyLock;

/// The CPUs background work's auxiliary pools leave the serving worker: its dispatch thread and
/// the service's runtime, which receive a request and preempt the job.
pub const SERVING_RESERVE: usize = 2;

/// What a process the service starts is for, which decides its share of the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Duty {
    /// The worker serving a load: requests wait on it.
    Serving,
    /// Work nothing waits on, which yields to serving: the measurement job, a catalog
    /// optimization, the service's catalog assessment and inventory scans.
    Background,
}

/// The threads one process may run at once, handed to an engine worker at its spawn
/// ([`crate::worker_process::WorkerLauncher::command`]) and set as its share at its startup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadShare(NonZeroUsize);

impl ThreadShare {
    pub fn get(self) -> NonZeroUsize {
        self.0
    }
}

/// The machine as the service sees it: the logical CPUs its processes may keep busy at once, its
/// quota (`seismic::threads::quota`), read once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MachineBudget {
    quota: NonZeroUsize,
}

static HOST: LazyLock<MachineBudget> = LazyLock::new(|| MachineBudget { quota: seismic::threads::quota() });

impl MachineBudget {
    /// This machine's.
    pub fn host() -> &'static Self {
        &HOST
    }

    /// The share of a process for `duty`: the whole quota for serving; for background work all of
    /// it but [`SERVING_RESERVE`], and one CPU on a machine of no more.
    pub fn share(&self, duty: Duty) -> ThreadShare {
        ThreadShare(match duty {
            Duty::Serving => self.quota,
            Duty::Background => {
                NonZeroUsize::new(self.quota.get().saturating_sub(SERVING_RESERVE)).unwrap_or(NonZeroUsize::MIN)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No share exceeds the quota, and a background job's leaves the serving worker its reserve
    /// wherever the quota is more than that.
    #[test]
    fn shares_never_exceed_the_quota_and_background_leaves_the_reserve() {
        for quota in 1..=256 {
            let budget = MachineBudget { quota: NonZeroUsize::new(quota).unwrap() };
            let (serving, background) = (budget.share(Duty::Serving).get().get(), budget.share(Duty::Background).get().get());
            assert_eq!(serving, quota);
            assert!(background <= quota);
            match quota > SERVING_RESERVE {
                true => assert_eq!(background, quota - SERVING_RESERVE),
                false => assert_eq!(background, 1),
            }
        }
    }

    /// The host's serving worker gets the process's whole quota of logical CPUs: what its pools
    /// sized themselves by before they read a share.
    #[test]
    fn the_host_serves_on_its_whole_quota() {
        assert_eq!(MachineBudget::host().share(Duty::Serving).get(), seismic::threads::quota());
    }
}
